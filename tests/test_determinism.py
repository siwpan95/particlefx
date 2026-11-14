"""确定性（可复现）模拟测试 —— v1.2 种子化随机源。

验证：同种子可复现、不同种子相异、未播种不保证一致（接口可用）、
每发射器独立确定性子流、发射器级 RNG 回滚、系统级 save/restore API。
"""

import random

import pytest

from particlefx import ParticleSystem, Emitter, ForceField, NoiseField
from particlefx.core.particle import Particle


def _snapshot(ps):
    """把活动粒子快照为可比较的元组序列（保留 9 位精度）。"""
    return [
        (round(p.x, 9), round(p.y, 9), round(p.vx, 9), round(p.vy, 9),
         round(p.life, 9), tuple(p.color))
        for p in ps.particles
    ]


def _build(seed, emitters=1):
    ps = ParticleSystem(max_particles=4000, seed=seed)
    for k in range(emitters):
        ps.add_emitter(Emitter(
            x=400 + k * 50, y=300, rate=200.0, speed=120.0,
            spread=0.8, life=2.0, size=2.0,
        ))
    ps.add_force_field(ForceField(field_type='gravity', strength=60.0))
    return ps


def test_seed_reproducible():
    """相同系统 seed + 相同推进序列 => 逐位一致。"""
    def run():
        ps = _build(12345)
        for _ in range(40):
            ps.update(1 / 60)
        return _snapshot(ps)

    a, b = run(), run()
    assert a == b
    assert len(a) > 0


def test_different_seeds_differ():
    """不同系统 seed => 模拟结果不同（至少发射位置/速度序列不同）。"""
    def run(seed):
        ps = _build(seed)
        for _ in range(40):
            ps.update(1 / 60)
        return _snapshot(ps)

    assert run(1) != run(2)


def test_unseeded_basic_functional():
    """未播种时退化为全局 random（旧行为）：接口可用、能正常发射。"""
    ps = ParticleSystem(max_particles=500)
    ps.add_emitter(Emitter(x=0, y=0, rate=100.0, speed=50.0, spread=1.0, life=1.0))
    ps.update(1 / 60)
    assert ps.get_particle_count() > 0
    # 未播种：save_rng_state 应为 None（无系统级种子化状态）
    assert ps.save_rng_state() is None


def test_per_emitter_independent_streams():
    """同种子系统内，各发射器由添加顺序派发独立子流，互不干扰且跨次一致。"""
    def run():
        ps = _build(999, emitters=2)
        for _ in range(10):
            ps.update(1 / 60)
        # 前半为发射器 0、后半为发射器 1 的粒子
        return [(round(p.x, 6), round(p.y, 6)) for p in ps.particles]

    a, b = run(), run()
    assert a == b
    mid = len(a) // 2
    # 两个发射器位置应当显著不同（不同子流），而非偶然重合
    assert any(
        abs(a[i][0] - a[j][0]) > 1e-3 or abs(a[i][1] - a[j][1]) > 1e-3
        for i in range(0, mid) for j in range(mid, len(a))
    )


def test_emitter_rng_save_restore_roundtrip():
    """发射器级 RNG 可序列化 / 回滚：恢复起始状态后随机序列重现。"""
    em = Emitter(x=0, y=0, rate=10.0, speed=10.0, seed=42)
    state = em._rng.getstate()  # 起始状态
    vals1 = [em.rng.uniform(0, 1) for _ in range(5)]
    _ = [em.rng.uniform(0, 1) for _ in range(3)]  # 推进若干步
    em._rng.setstate(state)  # 回滚到起始状态
    vals2 = [em.rng.uniform(0, 1) for _ in range(5)]
    assert vals1 == vals2


def test_emitter_rng_rollback_emission():
    """发射器级 RNG 回滚：恢复状态后同一次 emit 产出逐位一致的粒子。"""
    em = Emitter(x=400, y=300, rate=100.0, speed=50.0, spread=1.0, life=1.0, seed=7)
    state = em._rng.getstate()
    p1 = em.emit(0.1)
    em._rng.setstate(state)
    p2 = em.emit(0.1)
    assert [(p.x, p.y, p.vx, p.vy, p.rotation, p.mass) for p in p1] == \
           [(p.x, p.y, p.vx, p.vy, p.rotation, p.mass) for p in p2]


def test_system_rng_state_api():
    """系统级 save/restore 仅在播种时有效，且 round-trip 不报错。"""
    ps = ParticleSystem(seed=1)
    assert ps.save_rng_state() is not None
    ps.restore_rng_state(ps.save_rng_state())
    ps2 = ParticleSystem()  # 未播种
    assert ps2.save_rng_state() is None


def test_reseed_reproducible_from_new_seed():
    """reseed 到新种子（在尚未推进前）等价于直接用新种子构造；且不同于旧种子。"""
    def build(seed):
        ps = ParticleSystem(max_particles=2000, seed=seed)
        ps.add_emitter(Emitter(x=400, y=300, rate=150.0, speed=100.0, spread=1.0, life=2.0))
        ps.add_force_field(ForceField(field_type='gravity', strength=40.0))
        return ps

    fresh = build(555)
    for _ in range(30):
        fresh.update(1 / 60)
    fresh_snap = _snapshot(fresh)

    # 用临时种子构造，尚未推进即 reseed 到 555 -> 等价于直接用 555 构造
    ps = build(0)
    ps.reseed(555)
    for _ in range(30):
        ps.update(1 / 60)
    assert _snapshot(ps) == fresh_snap

    # reseed 到其他种子 -> 结果不同
    ps2 = build(0)
    ps2.reseed(999)
    for _ in range(30):
        ps2.update(1 / 60)
    assert _snapshot(ps2) != fresh_snap


# ============================================================
#  v1.2：力场 / 噪声场内部噪声种子化
# ============================================================

def test_force_field_seed_derived_deterministically():
    """系统播种时，未自带种子的力场由添加序号派发确定性子流（跨系统一致）。"""
    def build(seed):
        ps = ParticleSystem(max_particles=10, seed=seed)
        ff0 = ForceField(field_type='noise', noise_strength=3.0, noise_scale=40.0)
        ff1 = ForceField(field_type='turbulence', turb_strength=2.0, turb_scale=35.0)
        ps.add_force_field(ff0)
        ps.add_force_field(ff1)
        return ps, ff0, ff1

    ps_a, ffa0, ffa1 = build(42)
    ps_b, ffb0, ffb1 = build(42)
    # 同系统种子 -> 相同派生种子 + 相同噪声时间偏移
    assert ffa0.seed == ffb0.seed
    assert ffa1.seed == ffb1.seed
    assert ffa0._noise_time_offset == ffb0._noise_time_offset
    assert ffa0._system_derived and ffa1._system_derived
    # 不同系统种子 -> 派生种子不同
    ps_c, ffc0, _ = build(43)
    assert ffc0.seed != ffa0.seed


def test_force_field_explicit_seed_preserved_under_system_seed():
    """用户显式带种子的力场不被系统重派（保留用户意图）。"""
    ps = ParticleSystem(max_particles=10, seed=42)
    ff = ForceField(field_type='noise', noise_strength=3.0, seed=12345)
    ps.add_force_field(ff)
    assert ff.seed == 12345
    assert ff._system_derived is False


def test_noise_field_seed_derived_unless_explicit():
    """系统播种时，未显式指定 seed 的噪声场被派生；显式 seed 保留。"""
    ps = ParticleSystem(max_particles=10, seed=42)
    nf = NoiseField(strength=1.5, scale=60.0)  # 未显式 seed
    ps.add_noise_field(nf)
    assert nf.seed is not None
    assert nf._system_derived is True

    ps2 = ParticleSystem(max_particles=10, seed=42)
    nf2 = NoiseField(strength=1.5, scale=60.0, seed=777)  # 显式 seed
    ps2.add_noise_field(nf2)
    assert nf2.seed == 777
    assert nf2._system_derived is False


def test_determinism_with_noise_force_fields():
    """含噪声场 / 湍流场 / 独立噪声场的系统，在相同种子下逐位可复现。"""
    def run():
        ps = ParticleSystem(max_particles=2000, seed=999)
        ps.add_emitter(Emitter(x=400, y=300, rate=80.0, speed=120.0,
                               spread=0.8, life=2.0, size=2.0))
        ps.add_force_field(ForceField(field_type='noise', noise_strength=3.0,
                                      noise_scale=40.0))
        ps.add_force_field(ForceField(field_type='turbulence', turb_strength=2.0,
                                      turb_scale=35.0))
        ps.add_noise_field(NoiseField(strength=1.5, scale=60.0))
        for _ in range(30):
            ps.update(1 / 60)
        return _snapshot(ps)

    assert run() == run()
    assert len(run()) > 0


def test_reseed_rederives_force_fields():
    """reseed 一并重派由系统派生的力场 / 噪声场种子。"""
    ps = ParticleSystem(max_particles=10, seed=1)
    ff = ForceField(field_type='noise', noise_strength=3.0)
    nf = NoiseField(strength=1.5)
    ps.add_force_field(ff)
    ps.add_noise_field(nf)
    old_ff = ff.seed
    old_nf = nf.seed
    ps.reseed(2)
    assert ff.seed != old_ff
    assert nf.seed != old_nf
    # 重新派生后再次 equal（确定性）
    ps2 = ParticleSystem(max_particles=10, seed=2)
    ff2 = ForceField(field_type='noise', noise_strength=3.0)
    nf2 = NoiseField(strength=1.5)
    ps2.add_force_field(ff2)
    ps2.add_noise_field(nf2)
    assert ff2.seed == ff.seed
    assert nf2.seed == nf.seed

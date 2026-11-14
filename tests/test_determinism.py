"""确定性（可复现）模拟测试 —— v1.2 种子化随机源。

验证：同种子可复现、不同种子相异、未播种不保证一致（接口可用）、
每发射器独立确定性子流、发射器级 RNG 回滚、系统级 save/restore API。
"""

import random

import pytest

from particlefx import ParticleSystem, Emitter, ForceField
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

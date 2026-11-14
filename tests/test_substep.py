"""固定步长子步进测试 —— v1.2 数值稳定性。

验证：子步进精度逼近细积分、默认关闭时与旧行为逐位一致、
spiral-of-death 上限保护、子步累计时间等于完整 dt。
"""

import math

import pytest

from particlefx import ParticleSystem, Emitter, ForceField
from particlefx.core.particle import Particle


def _seed_particles(ps, n=3):
    """手动注入固定初值的粒子（rate=0 发射器不产生新粒子，隔离积分路径）。"""
    for i in range(n):
        ps.particles.append(Particle(
            x=400.0 + i * 10, y=300.0, vx=0.0, vy=0.0,
            life=1.0, max_life=10.0, size=2.0, color=(1, 2, 3),
        ))
    return ps


def _make(fixed_dt=None, max_substeps=100):
    ps = ParticleSystem(max_particles=200, fixed_dt=fixed_dt, max_substeps=max_substeps)
    ps.add_emitter(Emitter(x=400, y=300, rate=0.0, speed=0.0, life=10.0, size=2.0))
    ps.add_force_field(ForceField(field_type='gravity', strength=200.0))
    return _seed_particles(ps, n=3)


def _state(ps):
    return [(p.x, p.y, p.vx, p.vy) for p in ps.particles]


def test_substep_matches_fine_integration():
    """把大 dt 拆成固定子步，应逼近以同子步长逐帧推进的细积分（纯重力半隐式欧拉）。"""
    dt, steps = 1 / 60, 20

    # 细积分参考：20 帧、每帧 dt
    fine = _make()
    for _ in range(steps):
        fine.update(dt)
    fine_state = _state(fine)

    # 子步进：一帧推进 dt*steps，fixed_dt=dt -> 拆成 steps 步
    sub = _make(fixed_dt=dt, max_substeps=steps + 5)
    sub.update(dt * steps)
    sub_state = _state(sub)

    # 粗积分：一帧直接推进 dt*steps（旧行为，无子步进）
    coarse = _make()
    coarse.update(dt * steps)
    coarse_state = _state(coarse)

    def max_pos_err(a, b):
        return max(math.hypot(ax - bx, ay - by)
                   for (ax, ay, _, _), (bx, by, _, _) in zip(a, b))

    err_sub = max_pos_err(sub_state, fine_state)
    err_coarse = max_pos_err(coarse_state, fine_state)

    # 子步进与细积分逐位一致（同子步长、同子步数）
    assert err_sub < 1e-9
    # 子步进相比粗积分显著改善精度（粗积分在大 dt 下偏差更大）
    assert err_sub < err_coarse


def test_substep_disabled_by_default():
    """fixed_dt=None（默认）时，update 与旧版单步积分逐位一致。"""
    dt = 0.1
    default = _make()
    default.update(dt)
    enabled = _make(fixed_dt=1.0)  # fixed_dt > dt => 仍只走单步
    enabled.update(dt)
    assert _state(default) == _state(enabled)
    assert default._last_substep_count == 1
    assert enabled._last_substep_count == 1


def test_substep_cap_avoids_spiral_of_death():
    """子步数受 max_substeps 上限保护，避免卡顿后无限追赶（spiral of death）。"""
    ps = _make(fixed_dt=0.01, max_substeps=4)
    ps.update(10.0)  # 期望 ceil(10/0.01)=1000 步，但被 cap 到 4
    assert ps._last_substep_count == 4


def test_substep_time_accumulates_full_dt():
    """子步进累计推进的仿真时间应等于完整 dt。"""
    ps = _make(fixed_dt=0.01, max_substeps=100)
    ps.update(0.05)  # ceil(0.05/0.01)=5 步
    assert ps._last_substep_count == 5
    assert abs(ps.time - 0.05) < 1e-9

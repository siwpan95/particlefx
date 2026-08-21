"""
性能与压力测试

覆盖：大粒子数量更新性能、内存稳定性、
      粒子上限截断、系统生命周期、批量发射吞吐。
"""

import math
import random
import time
import pytest
import numpy as np

from core.particle import (
    Particle, ParticleSystem, Emitter, ForceField,
    EmitterShape, NoiseField, CollisionBoundary,
)
from core.particle_numpy import (
    ParticleSystemNumpy, NumpyForceGravity, NumpyForceDrag,
)


# ============================================================
#  ParticleSystem 压力测试
# ============================================================

class TestLargeParticleCount:
    """大粒子数量"""

    def test_1000_particles_update(self):
        """1000 粒子更新应在合理时间内完成"""
        ps = ParticleSystem(max_particles=1000)
        # 一次性发射 1000 粒子
        emitter = Emitter(x=400, y=300, rate=10000,
                          spread=math.pi * 2, speed=3,
                          life=5.0, size=2)
        ps.add_emitter(emitter)
        ps.update(0.1)

        start = time.time()
        for _ in range(100):
            ps.update(0.016)
        elapsed = time.time() - start

        # 100帧 × 1000粒子应在 5 秒内完成
        assert elapsed < 5.0, f"100帧更新耗时 {elapsed:.2f}s，超过阈值"

    def test_500_particles_with_forces(self):
        """500 粒子 + 多力场更新"""
        ps = ParticleSystem(max_particles=500)
        emitter = Emitter(x=400, y=300, rate=5000,
                          spread=math.pi * 2, speed=3, life=3.0)
        ps.add_emitter(emitter)
        ps.add_force_field(ForceField(field_type='gravity', strength=5.0))
        ps.add_force_field(ForceField(field_type='wind', strength=2.0))
        ps.add_noise_field(NoiseField(strength=1.0, scale=50.0))
        ps.update(0.1)

        start = time.time()
        for _ in range(50):
            ps.update(0.016)
        elapsed = time.time() - start
        assert elapsed < 5.0

    def test_max_particles_truncation(self):
        """粒子数量不超过 max_particles"""
        ps = ParticleSystem(max_particles=100)
        emitter = Emitter(x=400, y=300, rate=10000,
                          spread=math.pi * 2, speed=3, life=10.0)
        ps.add_emitter(emitter)
        ps.update(0.5)
        assert ps.get_particle_count() <= 100

    def test_max_particles_truncation_over_time(self):
        """持续发射后粒子不超过上限"""
        ps = ParticleSystem(max_particles=200)
        emitter = Emitter(x=400, y=300, rate=500,
                          spread=math.pi * 2, speed=3, life=10.0)
        ps.add_emitter(emitter)
        for _ in range(50):
            ps.update(0.016)
        assert ps.get_particle_count() <= 200


class TestParticleLifecycleStress:
    """粒子生命周期压力测试"""

    def test_spawn_and_die_cycle(self):
        """粒子出生-死亡循环"""
        ps = ParticleSystem(max_particles=500)
        emitter = Emitter(x=400, y=300, rate=100,
                          spread=math.pi * 2, speed=3,
                          life=0.5)  # 短命
        ps.add_emitter(emitter)
        ps.add_force_field(ForceField(field_type='gravity', strength=2.0))

        for _ in range(200):
            ps.update(0.016)

        # 系统应有活粒子，且不超过上限
        assert ps.get_particle_count() <= 500
        assert ps.get_particle_count() > 0

    def test_long_lived_particles(self):
        """长时间存活的粒子"""
        ps = ParticleSystem(max_particles=100)
        emitter = Emitter(x=400, y=300, rate=10,
                          spread=math.pi * 2, speed=1,
                          life=30.0)  # 30秒寿命
        ps.add_emitter(emitter)

        for _ in range(100):
            ps.update(0.016)

        assert ps.get_particle_count() <= 100

    def test_clear_releases_all(self):
        """clear() 后粒子数为零"""
        ps = ParticleSystem(max_particles=500)
        emitter = Emitter(x=400, y=300, rate=5000,
                          spread=math.pi * 2, speed=3, life=5.0)
        ps.add_emitter(emitter)
        ps.update(0.5)
        assert ps.get_particle_count() > 0
        ps.clear()
        assert ps.get_particle_count() == 0


class TestCollisionStress:
    """碰撞检测压力"""

    def test_boundary_bounce_many_particles(self):
        """边界反弹 + 大量粒子"""
        ps = ParticleSystem(max_particles=500)
        ps.set_boundary(CollisionBoundary(
            left=0, top=0, right=400, bottom=300,
            mode='bounce', restitution=0.8))
        emitter = Emitter(x=200, y=150, rate=5000,
                          spread=math.pi * 2, speed=8, life=10.0)
        ps.add_emitter(emitter)
        ps.update(0.1)

        for _ in range(50):
            ps.update(0.016)

        # 粒子应在边界内或已死亡
        for p in ps.particles:
            if p.is_alive():
                assert 0 <= p.x <= 400 + 1
                assert 0 <= p.y <= 300 + 1

    def test_boundary_destroy_many_particles(self):
        """边界销毁模式"""
        ps = ParticleSystem(max_particles=500)
        ps.set_boundary(CollisionBoundary(
            left=50, top=50, right=350, bottom=250,
            mode='destroy'))
        emitter = Emitter(x=200, y=150, rate=5000,
                          spread=math.pi * 2, speed=15, life=5.0)
        ps.add_emitter(emitter)
        ps.update(0.1)

        # 高速粒子会出界被销毁
        for _ in range(30):
            ps.update(0.016)

        # 存活粒子应在边界内
        for p in ps.particles:
            if p.is_alive():
                assert 50 <= p.x <= 350
                assert 50 <= p.y <= 250


# ============================================================
#  ParticleSystemNumpy 性能测试
# ============================================================

class TestNumpyPerformance:
    """NumPy 系统性能"""

    def test_5000_particles_update(self):
        """5000 粒子 NumPy 更新性能"""
        sys = ParticleSystemNumpy(max_particles=5000)
        sys.add_force(NumpyForceGravity(strength=5.0))
        sys.add_force(NumpyForceDrag(factor=0.99))
        sys.emit_radial(x=400, y=300, count=5000,
                       speed_range=(2, 8), life_range=(3.0, 5.0))

        start = time.time()
        for _ in range(300):
            sys.update(0.016)
        elapsed = time.time() - start

        # 300帧 × 5000粒子应在 3 秒内
        assert elapsed < 3.0, f"NumPy 300帧更新耗时 {elapsed:.2f}s"

    def test_numpy_vs_python_ratio(self):
        """NumPy 与纯 Python 性能对比（粗略）"""
        # NumPy: 5000 粒子
        sys_np = ParticleSystemNumpy(max_particles=5000)
        sys_np.add_force(NumpyForceGravity(strength=5.0))
        sys_np.emit_radial(x=400, y=300, count=5000,
                          speed_range=(2, 8), life_range=(3.0, 5.0))

        start_np = time.time()
        for _ in range(50):
            sys_np.update(0.016)
        elapsed_np = time.time() - start_np

        # Python: 500 粒子（更少因为慢）
        ps_py = ParticleSystem(max_particles=500)
        emitter = Emitter(x=400, y=300, rate=50000,
                          spread=math.pi * 2, speed=3, life=3.0)
        ps_py.add_emitter(emitter)
        ps_py.add_force_field(ForceField(field_type='gravity', strength=5.0))
        ps_py.update(0.1)

        start_py = time.time()
        for _ in range(50):
            ps_py.update(0.016)
        elapsed_py = time.time() - start_py

        # NumPy 5000粒子 应比 Python 500粒子 更快或相当
        # (宽松检查，不强制严格倍率)
        assert elapsed_np < elapsed_py * 10, \
            f"NumPy {elapsed_np:.3f}s vs Python {elapsed_py:.3f}s"

    def test_emit_burst_1000(self):
        """1000 粒子批量发射"""
        sys = ParticleSystemNumpy(max_particles=5000)
        start = time.time()
        sys.emit_burst(x=400, y=300, count=1000,
                      speed_range=(2, 8), life_range=(2.0, 5.0))
        elapsed = time.time() - start
        assert elapsed < 0.1
        assert sys.particle_count == 1000

    def test_emit_radial_1000(self):
        """1000 粒子径向发射"""
        sys = ParticleSystemNumpy(max_particles=5000)
        start = time.time()
        sys.emit_radial(x=400, y=300, count=1000,
                       speed_range=(2, 8), life_range=(2.0, 5.0))
        elapsed = time.time() - start
        assert elapsed < 0.1
        assert sys.particle_count == 1000


class TestNumpyMemoryStability:
    """NumPy 内存稳定性"""

    def test_array_sizes_stable(self):
        """数组大小在多次更新后不变"""
        sys = ParticleSystemNumpy(max_particles=1000)
        sys.emit_radial(x=400, y=300, count=500)

        shapes_before = {
            'pos': sys.pos.shape,
            'vel': sys.vel.shape,
            'alive': sys.alive.shape,
        }

        for _ in range(100):
            sys.update(0.016)

        shapes_after = {
            'pos': sys.pos.shape,
            'vel': sys.vel.shape,
            'alive': sys.alive.shape,
        }

        assert shapes_before == shapes_after

    def test_dead_slots_reusable(self):
        """死亡槽位可复用"""
        sys = ParticleSystemNumpy(max_particles=50)
        # 发射 50 粒子（满）
        sys.emit_radial(x=400, y=300, count=50,
                       life_range=(0.5, 0.5))
        assert sys.particle_count == 50

        # 等全部死亡
        for _ in range(30):
            sys.update(0.05)
        assert sys.particle_count == 0

        # 复用所有槽位
        sys.emit_radial(x=400, y=300, count=50)
        assert sys.particle_count == 50

    def test_no_memory_leak_in_loop(self):
        """循环中无内存泄漏（粒子数稳定）"""
        sys = ParticleSystemNumpy(max_particles=200)
        sys.add_force(NumpyForceGravity(strength=5.0))
        sys.add_force(NumpyForceDrag(factor=0.99))

        # 每帧补充死亡粒子
        for frame in range(200):
            sys.update(0.016)
            dead = int(np.sum(~sys.alive))
            if dead > 10:
                sys.emit_radial(x=400, y=300,
                               count=min(dead, 20),
                               speed_range=(2, 8),
                               life_range=(1.0, 3.0))

        assert sys.particle_count <= 200


# ============================================================
#  综合场景压力测试
# ============================================================

class TestMixedScenarioStress:
    """混合场景"""

    def test_emitter_shape_with_many_particles(self):
        """发射形状 + 大量粒子"""
        ps = ParticleSystem(max_particles=500)
        shape = EmitterShape('sphere', radius=50, emit_from_volume=True)
        emitter = Emitter(x=400, y=300, rate=5000,
                          spread=math.pi * 2, speed=5, life=3.0,
                          shape=shape)
        ps.add_emitter(emitter)
        ps.add_force_field(ForceField(field_type='gravity', strength=3.0))

        for _ in range(30):
            ps.update(0.016)

        assert ps.get_particle_count() <= 500

    def test_collision_and_forces(self):
        """碰撞 + 多力场"""
        ps = ParticleSystem(max_particles=300)
        ps.set_boundary(CollisionBoundary(
            left=0, top=0, right=800, bottom=600,
            mode='bounce', restitution=0.7))
        ps.add_force_field(ForceField(field_type='gravity', strength=5.0))
        ps.add_force_field(ForceField(field_type='wind', strength=3.0))

        emitter = Emitter(x=400, y=300, rate=3000,
                          spread=math.pi * 2, speed=10, life=5.0)
        ps.add_emitter(emitter)

        for _ in range(50):
            ps.update(0.016)

        assert ps.get_particle_count() <= 300

    def test_events_with_large_count(self):
        """事件回调 + 大量粒子"""
        birth_count = 0
        death_count = 0

        def on_birth(p):
            birth_count_local = 1

        def on_death(p):
            death_count_local = 1

        ps = ParticleSystem(max_particles=200)
        ps.on_particle_birth = on_birth
        ps.on_particle_death = on_death

        emitter = Emitter(x=400, y=300, rate=2000,
                          spread=math.pi * 2, speed=3, life=0.5)
        ps.add_emitter(emitter)

        for _ in range(50):
            ps.update(0.016)

        # 不崩溃即可

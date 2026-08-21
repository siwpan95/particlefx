"""
Numba 加速后端测试（v5.0 增强 ②）

numba 未安装时（如 Python 3.14）测试回退路径与向量化参考实现的等价性；
numba 可用时额外测试 JIT 路径与父类路径逐位一致。
"""
import math
import random

import numpy as np
import pytest

from core.particle_numpy import (
    ParticleSystemNumpy, NumpyForceGravity, NumpyForceWind, NumpyForceDrag,
)
from core.particle_numba import (
    HAS_NUMBA, NumbaParticleSystem, update_kernel_numpy, _update_kernel,
)


class TestKernelMath:
    def test_update_kernel_matches_numpy_reference(self):
        """逐元素内核与 NumPy 参考实现的数学等价性"""
        rng = np.random.default_rng(123)
        n = 200
        def make_arrays():
            pos = rng.uniform(0, 800, (n, 2)).astype(np.float32)
            vel = rng.uniform(-50, 50, (n, 2)).astype(np.float32)
            life = rng.uniform(0.1, 1.0, n).astype(np.float32)
            max_life = rng.uniform(0.5, 3.0, n).astype(np.float32)
            alpha = life.copy()
            alive = np.ones(n, dtype=bool)
            mass = rng.uniform(0.5, 2.0, n).astype(np.float32)
            return pos, vel, life, max_life, alpha, alive, mass

        dt, g, w, d = 0.016, 3.0, 1.5, 0.99
        a = make_arrays()
        b = [x.copy() for x in a]                 # 相同输入
        _update_kernel(*a, dt, g, w, d)           # 逐元素（参考）
        update_kernel_numpy(*b, dt, g, w, d)      # 向量化（参考）
        for x, y in zip(a, b):
            assert np.allclose(x, y, atol=1e-4), "内核与向量化参考实现不一致"

    def test_kernel_equivalence_with_base_system(self):
        """NumbaParticleSystem（回退路径）与 ParticleSystemNumpy+力场 完全一致"""
        def run_numba():
            sys_np = NumbaParticleSystem(max_particles=1000, gravity=3.0,
                                         wind=1.0, drag=0.99)
            rng = np.random.default_rng(7)
            sys_np.pos[:] = rng.uniform(0, 800, (1000, 2)).astype(np.float32)
            sys_np.vel[:] = rng.uniform(-20, 20, (1000, 2)).astype(np.float32)
            sys_np.life[:] = rng.uniform(0.1, 1.0, 1000).astype(np.float32)
            sys_np.max_life[:] = rng.uniform(0.5, 3.0, 1000).astype(np.float32)
            sys_np.alpha[:] = sys_np.life.copy()
            sys_np.mass[:] = rng.uniform(0.5, 2.0, 1000).astype(np.float32)
            sys_np.alive[:] = True
            for _ in range(60):
                sys_np.update(1 / 60)
            return (sys_np.pos.copy(), sys_np.vel.copy(),
                    sys_np.life.copy(), sys_np.alive.copy())

        def run_base():
            sys_np = ParticleSystemNumpy(max_particles=1000)
            sys_np.add_force(NumpyForceGravity(strength=3.0))
            sys_np.add_force(NumpyForceWind(strength=1.0))
            sys_np.add_force(NumpyForceDrag(factor=0.99))
            rng = np.random.default_rng(7)
            sys_np.pos[:] = rng.uniform(0, 800, (1000, 2)).astype(np.float32)
            sys_np.vel[:] = rng.uniform(-20, 20, (1000, 2)).astype(np.float32)
            sys_np.life[:] = rng.uniform(0.1, 1.0, 1000).astype(np.float32)
            sys_np.max_life[:] = rng.uniform(0.5, 3.0, 1000).astype(np.float32)
            sys_np.alpha[:] = sys_np.life.copy()
            sys_np.mass[:] = rng.uniform(0.5, 2.0, 1000).astype(np.float32)
            sys_np.alive[:] = True
            for _ in range(60):
                sys_np.update(1 / 60)
            return (sys_np.pos.copy(), sys_np.vel.copy(),
                    sys_np.life.copy(), sys_np.alive.copy())

        a = run_numba()
        b = run_base()
        for x, y in zip(a, b):
            assert np.allclose(x, y, atol=1e-5)


class TestNumbaParticleSystem:
    def test_fallback_without_numba(self):
        sys_np = NumbaParticleSystem(max_particles=500, gravity=2.0)
        assert isinstance(sys_np, ParticleSystemNumpy)
        sys_np.emit_burst(x=100, y=100, count=100, life_range=(1.0, 2.0))
        for _ in range(10):
            sys_np.update(0.016)
        assert 0 <= sys_np.particle_count <= 100
        assert sys_np.time > 0

    def test_force_fallback_applies_constants_and_custom(self):
        """添加自定义力场后回退到向量化路径，且常量力仍生效"""
        sys_np = NumbaParticleSystem(max_particles=200, gravity=2.0, wind=1.0)
        assert sys_np.jit_enabled is (HAS_NUMBA and not sys_np._forces)
        sys_np.add_force(NumpyForceGravity(strength=9.0))
        assert sys_np.jit_enabled is False
        sys_np.emit_burst(x=0, y=0, count=50, speed_range=(0.0, 0.0),
                          life_range=(2.0, 2.0))
        vy_before = sys_np.vel[sys_np.alive, 1].copy()
        vx_before = sys_np.vel[sys_np.alive, 0].copy()
        sys_np.update(0.016)  # 不应崩溃
        # 常量重力(2)+自定义重力(9)=11，均作用在 vy 上；风(1)作用在 vx 上
        vy_after = sys_np.vel[sys_np.alive, 1]
        vx_after = sys_np.vel[sys_np.alive, 0]
        assert np.allclose(vy_after - vy_before, 11.0 * 0.016, atol=1e-3)
        assert np.allclose(vx_after - vx_before, 1.0 * 0.016, atol=1e-3)
        assert sys_np.particle_count == 50

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba 未安装")
    def test_jit_path_identical_to_fallback(self):
        """numba 可用时：JIT 路径与父类向量化路径逐位一致"""
        sys_jit = NumbaParticleSystem(max_particles=500, gravity=3.0, wind=1.0, drag=0.99)
        sys_base = ParticleSystemNumpy(max_particles=500)
        sys_base.add_force(NumpyForceGravity(strength=3.0))
        sys_base.add_force(NumpyForceWind(strength=1.0))
        sys_base.add_force(NumpyForceDrag(factor=0.99))
        rng = np.random.default_rng(3)
        # 两个系统使用完全相同的初始数据（一次性抽样后复制）
        pos = rng.uniform(0, 800, (500, 2)).astype(np.float32)
        vel = rng.uniform(-30, 30, (500, 2)).astype(np.float32)
        life = rng.uniform(0.2, 1.0, 500).astype(np.float32)
        max_life = rng.uniform(1.0, 4.0, 500).astype(np.float32)
        for s in (sys_jit, sys_base):
            s.pos[:] = pos
            s.vel[:] = vel
            s.life[:] = life
            s.max_life[:] = max_life
            s.alive[:] = True
        assert sys_jit.jit_enabled is True
        for _ in range(30):
            sys_jit.update(1 / 60)
            sys_base.update(1 / 60)
        # 仅允许 float32 舍入级差异（约 1e-4）
        assert np.allclose(sys_jit.pos, sys_base.pos, atol=1e-2)
        assert np.allclose(sys_jit.vel, sys_base.vel, atol=1e-2)
        assert np.array_equal(sys_jit.alive, sys_base.alive)

    def test_render_smoke(self):
        sys_np = NumbaParticleSystem(max_particles=100, gravity=3.0)
        sys_np.emit_burst(x=200, y=200, count=60, life_range=(1.0, 2.0))
        sys_np.update(0.016)
        img = sys_np.render_to_image(width=400, height=400)
        assert img.size == (400, 400)

    def test_benchmark_smoke(self):
        sys_np = NumbaParticleSystem(max_particles=1000, gravity=3.0)
        sys_np.emit_burst(x=400, y=300, count=500, life_range=(1.0, 3.0))
        ms = sys_np.benchmark_update(frames=10, dt=1 / 60) * 1000
        assert ms >= 0

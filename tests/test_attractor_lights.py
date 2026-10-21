"""
粒子吸引子与粒子光源测试（v5.0 增强 ⑤）
"""
import math
import random

import pytest

from particlefx.core.particle import (
    Particle, ParticleSystem, Emitter, ForceField, LightSource,
    ParticleAttractor, create_attractor,
)


def _mk(x, y, vx=0.0, vy=0.0, life=1.0, mass=1.0):
    return Particle(x=x, y=y, vx=vx, vy=vy, life=life, max_life=2.0,
                    size=2.0, color=(255, 255, 255), mass=mass)


# ================================================================
#  ParticleAttractor
# ================================================================

class TestParticleAttractor:
    def test_attracts_toward_center(self):
        a = ParticleAttractor(x=100, y=100, strength=5.0, radius=300.0)
        p = _mk(200, 100)  # 在吸引子右侧
        fx, fy = a.get_force(p)
        assert fx < 0 and abs(fy) < 1e-9  # 指向左（中心方向）

    def test_radius_cutoff(self):
        a = ParticleAttractor(x=0, y=0, strength=5.0, radius=100.0)
        assert a.get_force(_mk(50, 0)) != (0.0, 0.0)
        assert a.get_force(_mk(101, 0)) == (0.0, 0.0)

    def test_repulsion_polarity(self):
        a = ParticleAttractor(x=0, y=0, strength=5.0, radius=200.0, polarity=-1.0)
        fx, _ = a.get_force(_mk(50, 0))
        assert fx > 0  # 排斥：力背离中心

    def test_linear_falloff_weaker_near_edge(self):
        a = ParticleAttractor(x=0, y=0, strength=10.0, radius=200.0, falloff='linear')
        f_near = math.hypot(*a.get_force(_mk(20, 0)))
        f_far = math.hypot(*a.get_force(_mk(180, 0)))
        assert f_near > f_far  # 线性衰减：近处更强
        assert f_far < 1e-6 or f_far < f_near * 0.3

    def test_coordinate_mode_call(self):
        a = ParticleAttractor(x=0, y=0, strength=5.0, radius=300.0)
        fx, fy = a.get_force(50.0, 0.0)  # 坐标模式
        assert fx < 0

    def test_follow_callback(self):
        target = [300.0, 200.0]
        a = ParticleAttractor(x=0, y=0, strength=5.0, radius=500.0,
                              follow=lambda: (target[0], target[1]))
        a.update(0.016)
        assert (a.x, a.y) == (300.0, 200.0)

    def test_factory(self):
        a = create_attractor(x=10, y=20, strength=3.0, radius=150.0, polarity=-1)
        assert isinstance(a, ParticleAttractor)
        assert a.radius == 150.0
        assert a.polarity == -1.0

    def test_works_inside_particle_system(self):
        ps = ParticleSystem(max_particles=300)
        a = ParticleAttractor(x=200, y=200, strength=20.0, radius=500.0)
        ps.add_force_field(a)
        e = Emitter(x=400, y=200, rate=200, spread=0.0, speed=0.0, life=3.0)
        ps.add_emitter(e)
        ps.update(0.1)
        # 粒子应被吸引向左移动
        assert all(p.vx < 0 for p in ps.particles)


# ================================================================
#  粒子间作用力（pairwise force）
# ================================================================

class TestPairwiseForce:
    def test_attraction_pulls_particles_together(self):
        ps = ParticleSystem(max_particles=50)
        ps.enable_pairwise_force(strength=30.0, radius=100.0)
        ps.particles.append(_mk(100, 100))
        ps.particles.append(_mk(120, 100))
        for _ in range(30):
            ps.update(0.016)
        d = math.hypot(ps.particles[0].x - ps.particles[1].x,
                       ps.particles[0].y - ps.particles[1].y)
        assert d < 20.0  # 初始距离 20，吸引后应显著缩小

    def test_repulsion_pushes_apart(self):
        ps = ParticleSystem(max_particles=50)
        ps.enable_pairwise_force(strength=50.0, radius=100.0, repulsion=True)
        ps.particles.append(_mk(100, 100))
        ps.particles.append(_mk(110, 100))
        for _ in range(30):
            ps.update(0.016)
        d = math.hypot(ps.particles[0].x - ps.particles[1].x,
                       ps.particles[0].y - ps.particles[1].y)
        assert d > 10.0  # 排斥后距离应大于初始距离

    def test_radius_limit_no_force_beyond(self):
        ps = ParticleSystem(max_particles=50)
        ps.enable_pairwise_force(strength=50.0, radius=50.0)
        p1 = _mk(0, 0)
        p2 = _mk(200, 0)  # 超出半径
        ps.particles.extend([p1, p2])
        ps.update(0.016)
        assert p1.vx == 0.0 and p1.vy == 0.0

    def test_grid_path_matches_linear_path(self):
        """空间哈希加速路径与线性扫描路径结果一致（同种子）"""
        def run(use_grid):
            random.seed(21)
            ps = ParticleSystem(max_particles=120, use_spatial_hash=use_grid,
                                spatial_hash_cell_size=30.0)
            ps.enable_pairwise_force(strength=15.0, radius=80.0)
            for i in range(60):
                ps.particles.append(_mk(
                    random.uniform(0, 300), random.uniform(0, 300),
                    life=1.0))
            for _ in range(20):
                ps.update(0.016)
            return [(round(p.x, 4), round(p.y, 4)) for p in ps.particles]
        assert run(False) == run(True)

    def test_no_effect_when_disabled(self):
        ps = ParticleSystem(max_particles=50)
        p1 = _mk(0, 0)
        p2 = _mk(10, 0)
        ps.particles.extend([p1, p2])
        ps.update(0.016)
        assert p1.vx == 0.0 and p2.vx == 0.0  # 默认零开销


# ================================================================
#  粒子光源（particle lights）
# ================================================================

class TestParticleLights:
    def test_enable_creates_light_pool(self):
        ps = ParticleSystem(max_particles=100)
        ps.enable_particle_lights(max_lights=4)
        assert len(ps.particle_lights) == 4
        assert ps.lighting_enabled is True

    def test_brightest_emissive_become_lights(self):
        ps = ParticleSystem(max_particles=100)
        ps.enable_particle_lights(max_lights=2, min_emissive_strength=0.5)
        dim = _mk(0, 0)
        bright = _mk(50, 50)
        bright.emissive = (255, 200, 100)
        bright.emissive_strength = 2.0
        dim.emissive = (10, 10, 10)
        dim.emissive_strength = 0.1  # 低于阈值
        ps.particles.extend([dim, bright])
        ps.update(0.016)
        active = [l for l in ps.particle_lights if l.intensity > 0]
        assert len(active) == 1
        assert (active[0].x, active[0].y) == (50.0, 50.0)

    def test_max_lights_cap(self):
        ps = ParticleSystem(max_particles=100)
        ps.enable_particle_lights(max_lights=3)
        for i in range(8):
            p = _mk(i * 10, 0)
            p.emissive = (255, 255, 255)
            p.emissive_strength = 1.0 + i * 0.1
            ps.particles.append(p)
        ps.update(0.016)
        active = [l for l in ps.particle_lights if l.intensity > 0]
        assert len(active) == 3

    def test_render_with_particle_lights(self):
        ps = ParticleSystem(max_particles=100)
        ps.enable_particle_lights(max_lights=2)
        p = _mk(10, 10)
        p.emissive = (255, 255, 255)
        p.emissive_strength = 2.0
        ps.particles.append(p)
        ps.update(0.016)

        class SpyRenderer:
            def __init__(self):
                self.rendered = None
            def render(self, particles):
                self.rendered = particles
        r = SpyRenderer()
        ps.render(r)
        # 光照管线应被使用（渲染的是光照后的副本）
        assert r.rendered is not None
        assert len(r.rendered) == 1
        assert r.rendered[0] is not p  # 是光照后的副本，非原对象

    def test_disabled_zero_overhead(self):
        ps = ParticleSystem(max_particles=100)
        assert ps._particle_lights_enabled is False
        assert ps.particle_lights == []

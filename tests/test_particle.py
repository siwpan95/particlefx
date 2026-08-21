"""
Particle 核心模块单元测试

覆盖：Particle 生命周期、Emitter 发射、ForceField 力场计算、
      EmitterShape 形状采样、ParticleSystem 更新管线、
      碰撞检测、曲线插值辅助函数。
"""

import math
import random
import pytest
from core.particle import (
    Particle,
    ParticleSystem,
    Emitter,
    ForceField,
    EmitterShape,
    NoiseField,
    CollisionBox,
    CollisionBoundary,
    LightSource,
    BlinnPhongParams,
    Renderer,
    apply_lighting_to_particle,
    lerp_color,
    evaluate_curve,
    evaluate_color_curve,
    evaluate_curve_vel,
    gradient_to_curve,
    size_curve_fadeout,
    size_curve_pulse,
    velocity_curve_accelerate,
    velocity_curve_decelerate,
    create_point_light,
    create_directional_light,
    create_ambient_light,
    create_vortex,
    create_turbulence,
    create_magnet,
)


# ============================================================
#  曲线/插值辅助函数
# ============================================================

class TestLerpColor:
    def test_endpoints(self):
        assert lerp_color((0, 0, 0), (255, 255, 255), 0.0) == (0, 0, 0)
        assert lerp_color((0, 0, 0), (255, 255, 255), 1.0) == (255, 255, 255)

    def test_midpoint(self):
        r, g, b = lerp_color((0, 0, 0), (100, 200, 50), 0.5)
        assert r == 50
        assert g == 100
        assert b == 25

    def test_clamp(self):
        assert lerp_color((10, 20, 30), (40, 50, 60), -0.5) == (10, 20, 30)
        assert lerp_color((10, 20, 30), (40, 50, 60), 1.5) == (40, 50, 60)


class TestEvaluateCurve:
    def test_empty(self):
        assert evaluate_curve([], 0.5) == 1.0

    def test_single_point(self):
        assert evaluate_curve([(0.0, 5.0)], 0.3) == 5.0

    def test_two_points(self):
        curve = [(0.0, 0.0), (1.0, 10.0)]
        assert evaluate_curve(curve, 0.0) == 0.0
        assert evaluate_curve(curve, 1.0) == 10.0
        assert evaluate_curve(curve, 0.5) == 5.0

    def test_clamp(self):
        curve = [(0.2, 1.0), (0.8, 2.0)]
        assert evaluate_curve(curve, 0.0) == 1.0
        assert evaluate_curve(curve, 1.0) == 2.0


class TestEvaluateColorCurve:
    def test_empty(self):
        assert evaluate_color_curve([], 0.5) == (255, 255, 255)

    def test_single_point(self):
        assert evaluate_color_curve([(0.0, (10, 20, 30))], 0.5) == (10, 20, 30)

    def test_two_points(self):
        curve = [(0.0, (0, 0, 0)), (1.0, (100, 200, 50))]
        result = evaluate_color_curve(curve, 0.5)
        assert result == (50, 100, 25)


class TestEvaluateCurveVel:
    def test_empty(self):
        assert evaluate_curve_vel([], 0.5) == (1.0, 1.0)

    def test_two_points(self):
        curve = [(0.0, (1.0, 1.0)), (1.0, (2.0, 3.0))]
        assert evaluate_curve_vel(curve, 0.5) == (1.5, 2.0)


class TestGradientToCurve:
    def test_empty(self):
        assert gradient_to_curve([]) == []

    def test_single_color(self):
        result = gradient_to_curve([(255, 0, 0)])
        assert result == [(0.0, (255, 0, 0))]

    def test_two_colors(self):
        colors = [(0, 0, 0), (255, 255, 255)]
        curve = gradient_to_curve(colors)
        assert curve[0] == (0.0, (0, 0, 0))
        assert curve[-1] == (1.0, (255, 255, 255))

    def test_three_colors_has_midpoints(self):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        curve = gradient_to_curve(colors)
        assert len(curve) > 3  # 包含中间插值点


class TestSizeCurves:
    def test_fadeout(self):
        curve = size_curve_fadeout(10.0)
        assert curve[0] == (0.0, 10.0)
        assert curve[-1] == (1.0, 0.0)

    def test_pulse(self):
        curve = size_curve_pulse(10.0, peak_at=0.3, peak_mult=1.5)
        assert curve[0] == (0.0, 10.0)
        assert curve[1] == (0.3, 15.0)
        assert curve[-1] == (1.0, 2.0)


class TestVelocityCurves:
    def test_accelerate(self):
        curve = velocity_curve_accelerate(2.0)
        assert curve[0] == (0.0, (1.0, 1.0))
        assert curve[-1] == (1.0, (2.0, 2.0))

    def test_decelerate(self):
        curve = velocity_curve_decelerate(0.3)
        assert curve[0] == (0.0, (1.0, 1.0))
        assert curve[-1] == (1.0, (0.3, 0.3))


# ============================================================
#  Particle
# ============================================================

class TestParticle:
    def _make_particle(self, **kwargs):
        defaults = dict(
            x=100, y=200, vx=10, vy=-5,
            life=1.0, max_life=2.0,
            size=5, color=(255, 0, 0),
        )
        defaults.update(kwargs)
        return Particle(**defaults)

    def test_alive(self):
        p = self._make_particle(life=0.5)
        assert p.is_alive() is True

    def test_dead(self):
        p = self._make_particle(life=0.0)
        assert p.is_alive() is False

    def test_update_decreases_life(self):
        p = self._make_particle(life=1.0, max_life=2.0)
        p.update(0.5)
        assert p.life < 1.0

    def test_update_moves_position(self):
        p = self._make_particle(x=0, y=0, vx=10, vy=20)
        p.update(1.0)
        assert p.x == 10
        assert p.y == 20

    def test_update_rotation(self):
        p = self._make_particle(angular_velocity=5.0, rotation=0.0)
        p.update(0.5)
        assert p.rotation == 2.5

    def test_alpha_fades_with_life(self):
        p = self._make_particle(life=0.5)
        p.update(0.0)  # 不推进时间，只检查 alpha
        assert abs(p.alpha - 0.5) < 0.01

    def test_enable_trail(self):
        p = self._make_particle()
        p.enable_trail(length=5)
        assert p.trail_enabled is True
        assert p.trail_max_length == 5
        assert len(p.trail_history) == 1

    def test_trail_records_history(self):
        p = self._make_particle(x=0, y=0)
        p.enable_trail(length=3)
        p.update(1.0)  # x=10, y=20
        p.update(1.0)  # x=20, y=40
        assert len(p.trail_history) <= 3

    def test_color_over_lifetime(self):
        p = self._make_particle(
            color_over_lifetime=[(0.0, (255, 0, 0)), (1.0, (0, 0, 255))],
            life=1.0, max_life=1.0,
        )
        p.update(0.5)  # 生命消耗一半
        # 颜色应该在红色和蓝色之间
        r, g, b = p.color
        assert r > 0
        assert b > 0

    def test_size_over_lifetime(self):
        p = self._make_particle(
            size_over_lifetime=[(0.0, 10.0), (1.0, 0.0)],
            life=1.0, max_life=1.0,
        )
        p.update(0.5)
        assert p.size < 10.0

    def test_texture_animation(self):
        p = self._make_particle(
            texture_frames=['frame0', 'frame1', 'frame2'],
            texture_fps=10.0,
        )
        p.update(0.15)  # 1.5 帧间隔
        assert p.texture_current_frame >= 1

    def test_get_current_texture(self):
        frames = ['a', 'b', 'c']
        p = self._make_particle(texture_frames=frames)
        assert p.get_current_texture() == 'a'
        p.texture_current_frame = 2
        assert p.get_current_texture() == 'c'

    def test_get_trail_points_includes_current(self):
        p = self._make_particle(x=50, y=60)
        p.enable_trail(length=3)
        p.update(1.0)
        points = p.get_trail_points()
        assert (50, 60) in points or (60, 60) in points  # 含当前位置


# ============================================================
#  EmitterShape
# ============================================================

class TestEmitterShape:
    def test_point_shape(self):
        shape = EmitterShape(shape_type='point')
        x, y, vx, vy = shape.sample_position(100, 200, 50, 0.1)
        assert x == 100
        assert y == 200
        assert abs(vx - 50) < 5  # spread 很小时接近 50

    def test_sphere_shape_within_radius(self):
        random.seed(42)
        shape = EmitterShape(shape_type='sphere', radius=30)
        for _ in range(100):
            x, y, _, _ = shape.sample_position(400, 300, 100, 0.5)
            dist = math.sqrt((x - 400)**2 + (y - 300)**2)
            assert dist <= 30 + 0.1

    def test_sphere_shell(self):
        random.seed(42)
        shape = EmitterShape(shape_type='sphere', radius=30, emit_from_shell=True)
        for _ in range(50):
            x, y, _, _ = shape.sample_position(400, 300, 100, 0.5)
            dist = math.sqrt((x - 400)**2 + (y - 300)**2)
            assert abs(dist - 30) < 2.0  # 壳上

    def test_circle_shape_within_radius(self):
        random.seed(42)
        shape = EmitterShape(shape_type='circle', radius=50)
        for _ in range(100):
            x, y, _, _ = shape.sample_position(0, 0, 100, 0.5)
            dist = math.sqrt(x**2 + y**2)
            assert dist <= 50 + 0.1

    def test_rectangle_shape_within_bounds(self):
        random.seed(42)
        shape = EmitterShape(shape_type='rectangle', width=100, rect_height=60)
        for _ in range(100):
            x, y, _, _ = shape.sample_position(200, 200, 100, 0.5)
            assert 150 <= x <= 250
            assert 170 <= y <= 230

    def test_edge_shape(self):
        random.seed(42)
        shape = EmitterShape(shape_type='edge', edge_length=200, edge_angle=0.0)
        for _ in range(100):
            x, y, _, _ = shape.sample_position(400, 300, 100, 0.5)
            assert 300 <= x <= 500
            assert abs(y - 300) < 1.0  # 水平边

    def test_cone_shape(self):
        random.seed(42)
        shape = EmitterShape(shape_type='cone', radius=40, height=60)
        x, y, vx, vy = shape.sample_position(100, 100, 50, 0.3)
        # 锥底在 y=100+60=160
        assert abs(y - 160) < 1.0

    def test_unknown_shape_falls_back_to_point(self):
        shape = EmitterShape(shape_type='unknown')
        x, y, _, _ = shape.sample_position(50, 50, 30, 0.1)
        assert x == 50
        assert y == 50


# ============================================================
#  ForceField
# ============================================================

class TestForceField:
    def test_gravity(self):
        ff = ForceField(field_type='gravity', strength=9.8)
        fx, fy = ff.get_force(100, 200)
        assert fx == 0.0
        assert fy == 9.8

    def test_wind(self):
        ff = ForceField(field_type='wind', strength=5.0)
        fx, fy = ff.get_force(0, 0)
        assert fx == 5.0
        assert fy == 0.0

    def test_attraction(self):
        ff = ForceField(field_type='attraction', x=100, y=100, strength=10.0)
        fx, fy = ff.get_force(110, 100)
        assert fx < 0  # 向吸引点方向
        assert abs(fy) < 0.1

    def test_attraction_at_center(self):
        ff = ForceField(field_type='attraction', x=50, y=50, strength=10.0)
        fx, fy = ff.get_force(50, 50)
        assert fx == 0.0
        assert fy == 0.0

    def test_noise_returns_nonzero(self):
        ff = ForceField(field_type='noise', noise_strength=5.0)
        fx, fy = ff.get_force(100, 200, time=0.5)
        # 噪声场不一定非零，但大概率非零
        assert isinstance(fx, float)
        assert isinstance(fy, float)

    def test_vortex(self):
        ff = ForceField(field_type='vortex', x=0, y=0, strength=10.0)
        fx, fy = ff.get_force(100, 0)
        # 在右侧的粒子，漩涡切向力应该朝上或朝下
        assert abs(fx) < 1.0  # 切向力主要在 y 方向
        assert abs(fy) > 0.1

    def test_magnet_attraction(self):
        ff = ForceField(field_type='magnet', x=0, y=0, strength=10.0,
                        magnet_polarity=1.0, magnet_range=200)
        fx, fy = ff.get_force(50, 0)
        assert fx < 0  # 吸引方向

    def test_magnet_repulsion(self):
        ff = ForceField(field_type='magnet', x=0, y=0, strength=10.0,
                        magnet_polarity=-1.0, magnet_range=200)
        fx, fy = ff.get_force(50, 0)
        assert fx > 0  # 排斥方向

    def test_magnet_out_of_range(self):
        ff = ForceField(field_type='magnet', x=0, y=0, strength=10.0,
                        magnet_range=100)
        fx, fy = ff.get_force(200, 0)
        assert fx == 0.0
        assert fy == 0.0

    def test_turbulence(self):
        ff = ForceField(field_type='turbulence', turb_strength=3.0)
        fx, fy = ff.get_force(100, 100, time=0.5)
        assert isinstance(fx, float)
        assert isinstance(fy, float)

    def test_unknown_type(self):
        ff = ForceField(field_type='unknown')
        fx, fy = ff.get_force(0, 0)
        assert fx == 0.0
        assert fy == 0.0

    def test_set_position(self):
        ff = ForceField(field_type='attraction', x=0, y=0)
        ff.set_position(100, 200)
        assert ff.x == 100
        assert ff.y == 200


class TestFactoryFunctions:
    def test_create_point_light(self):
        light = create_point_light(100, 200)
        assert light.light_type == 'point'
        assert light.x == 100
        assert light.y == 200

    def test_create_directional_light(self):
        light = create_directional_light(math.pi / 2)
        assert light.light_type == 'directional'
        assert abs(light.direction - math.pi / 2) < 0.01

    def test_create_ambient_light(self):
        light = create_ambient_light()
        assert light.light_type == 'ambient'

    def test_create_vortex(self):
        ff = create_vortex(100, 200, strength=5.0)
        assert ff.field_type == 'vortex'
        assert ff.x == 100
        assert ff.y == 200

    def test_create_turbulence(self):
        ff = create_turbulence(strength=3.0)
        assert ff.field_type == 'turbulence'

    def test_create_magnet(self):
        ff = create_magnet(50, 50, polarity=1.0)
        assert ff.field_type == 'magnet'
        assert ff.x == 50
        assert ff.y == 50


# ============================================================
#  NoiseField
# ============================================================

class TestNoiseField:
    def test_update_increases_time(self):
        nf = NoiseField(seed=42)
        nf.update(0.5)
        assert nf._time == 0.5

    def test_get_force_returns_floats(self):
        nf = NoiseField(strength=2.0, seed=42)
        p = Particle(x=100, y=200, vx=0, vy=0, life=1.0, max_life=1.0,
                     size=5, color=(255, 255, 255))
        fx, fy = nf.get_force(p)
        assert isinstance(fx, float)
        assert isinstance(fy, float)


# ============================================================
#  CollisionBox / CollisionBoundary
# ============================================================

class TestCollisionBox:
    def test_contains_inside(self):
        box = CollisionBox(x=100, y=100, width=50, height=50)
        assert box.contains_point(120, 120) is True

    def test_contains_outside(self):
        box = CollisionBox(x=100, y=100, width=50, height=50)
        assert box.contains_point(200, 200) is False

    def test_contains_boundary(self):
        box = CollisionBox(x=100, y=100, width=50, height=50)
        assert box.contains_point(100, 100) is True
        assert box.contains_point(150, 150) is True


# ============================================================
#  LightSource / BlinnPhong
# ============================================================

class TestLightSource:
    def test_ambient(self):
        light = LightSource(light_type='ambient', intensity=0.5)
        color, intensity = light.get_illumination(100, 100)
        assert intensity == 0.5

    def test_point_light_in_range(self):
        light = LightSource(light_type='point', x=100, y=100, range_=200)
        color, intensity = light.get_illumination(110, 100)
        assert intensity > 0

    def test_point_light_out_of_range(self):
        light = LightSource(light_type='point', x=0, y=0, range_=50)
        color, intensity = light.get_illumination(200, 200)
        assert intensity == 0.0


class TestApplyLighting:
    def test_basic_lighting(self):
        p = Particle(x=100, y=100, vx=0, vy=0, life=1.0, max_life=1.0,
                     size=5, color=(255, 0, 0))
        light = LightSource(light_type='ambient', intensity=0.5)
        result = apply_lighting_to_particle(p, [light])
        assert len(result) == 4
        r, g, b, a = result
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255

    def test_emissive_particle(self):
        p = Particle(x=100, y=100, vx=0, vy=0, life=1.0, max_life=1.0,
                     size=5, color=(50, 50, 50),
                     emissive=(255, 200, 100), emissive_strength=2.0)
        result = apply_lighting_to_particle(p, [])
        r, g, b, a = result
        assert r > 50  # emissive 增加了亮度


# ============================================================
#  Emitter
# ============================================================

class TestEmitter:
    def test_emit_zero_dt(self):
        em = Emitter(x=100, y=100, rate=100)
        particles = em.emit(0.0)
        assert len(particles) == 0

    def test_emit_proportional_to_rate(self):
        em = Emitter(x=100, y=100, rate=100)
        particles = em.emit(0.1)  # 100 * 0.1 = 10
        assert len(particles) == 10

    def test_emit_particle_attributes(self):
        em = Emitter(x=100, y=200, rate=10, speed=50, life=2.0, size=5,
                     color=(255, 0, 0))
        particles = em.emit(0.1)
        assert len(particles) == 1
        p = particles[0]
        assert p.max_life == 2.0
        assert p.size == 5
        assert p.color == (255, 0, 0)

    def test_emit_with_shape(self):
        shape = EmitterShape(shape_type='rectangle', width=100, rect_height=50)
        em = Emitter(x=200, y=200, rate=100, shape=shape)
        particles = em.emit(0.1)
        for p in particles:
            assert 150 <= p.x <= 250
            assert 175 <= p.y <= 225

    def test_emit_with_audio(self):
        em = Emitter(x=100, y=100, rate=100)
        audio = {'rms': 2.0}
        particles = em.emit(0.1, audio_data=audio)
        # rate * (1 + 2*2) = 100 * 5 = 500, 500 * 0.1 = 50
        assert len(particles) == 50

    def test_set_position(self):
        em = Emitter(x=0, y=0, rate=10)
        em.set_position(300, 400)
        assert em.x == 300
        assert em.y == 400

    def test_trail_enabled(self):
        em = Emitter(x=0, y=0, rate=10, trail_enabled=True, trail_length=5)
        particles = em.emit(0.1)
        for p in particles:
            assert p.trail_enabled is True
            assert p.trail_max_length == 5


# ============================================================
#  ParticleSystem
# ============================================================

class TestParticleSystem:
    def test_init(self):
        ps = ParticleSystem(max_particles=500)
        assert ps.max_particles == 500
        assert len(ps.particles) == 0
        assert len(ps.emitters) == 0

    def test_add_emitter(self):
        ps = ParticleSystem()
        em = Emitter(x=100, y=100, rate=10)
        ps.add_emitter(em)
        assert len(ps.emitters) == 1

    def test_add_force_field(self):
        ps = ParticleSystem()
        ff = ForceField(field_type='gravity', strength=5.0)
        ps.add_force_field(ff)
        assert len(ps.force_fields) == 1

    def test_add_collision_box(self):
        ps = ParticleSystem()
        box = CollisionBox(x=0, y=0, width=100, height=100)
        ps.add_collision_box(box)
        assert len(ps.collision_boxes) == 1

    def test_set_boundary(self):
        ps = ParticleSystem()
        boundary = CollisionBoundary(mode='bounce')
        ps.set_boundary(boundary)
        assert ps.boundary is not None

    def test_add_light(self):
        ps = ParticleSystem()
        light = LightSource(light_type='ambient')
        ps.add_light(light)
        assert len(ps.lights) == 1
        assert ps.lighting_enabled is True

    def test_update_emits_particles(self):
        ps = ParticleSystem(max_particles=100)
        em = Emitter(x=100, y=100, rate=100)
        ps.add_emitter(em)
        ps.update(0.1)
        assert len(ps.particles) > 0

    def test_update_applies_force_fields(self):
        ps = ParticleSystem(max_particles=100)
        em = Emitter(x=0, y=0, rate=10, speed=0)
        ps.add_emitter(em)
        ff = ForceField(field_type='gravity', strength=100)
        ps.add_force_field(ff)
        ps.update(0.1)  # 发射 1 个粒子
        p = ps.particles[0]
        old_vy = p.vy
        ps.update(0.1)  # 力场应该改变 vy
        # 粒子受重力影响 vy 应该增大
        assert any(p.vy > old_vy for p in ps.particles)

    def test_max_particles_limit(self):
        ps = ParticleSystem(max_particles=5)
        em = Emitter(x=0, y=0, rate=1000, life=10.0)
        ps.add_emitter(em)
        ps.update(0.1)  # 尝试发射 100 个
        assert len(ps.particles) <= 5

    def test_boundary_bounce(self):
        ps = ParticleSystem(max_particles=10)
        boundary = CollisionBoundary(left=0, top=0, right=100, bottom=100,
                                     mode='bounce', restitution=1.0)
        ps.set_boundary(boundary)
        em = Emitter(x=50, y=50, rate=10, speed=0, life=10.0)
        ps.add_emitter(em)
        ps.update(0.1)
        # 手动把粒子移到边界外
        for p in ps.particles:
            p.x = -10
            p.vx = -5
        ps.update(0.01)
        for p in ps.particles:
            assert p.x >= -0.1  # 应被拉回边界内

    def test_boundary_destroy(self):
        ps = ParticleSystem(max_particles=10)
        boundary = CollisionBoundary(left=0, top=0, right=100, bottom=100,
                                     mode='destroy')
        ps.set_boundary(boundary)
        em = Emitter(x=50, y=50, rate=10, speed=0, life=10.0)
        ps.add_emitter(em)
        ps.update(0.1)
        for p in ps.particles:
            p.x = 200  # 移出边界
        ps.update(0.01)
        assert len(ps.particles) == 0

    def test_boundary_wrap(self):
        ps = ParticleSystem(max_particles=10)
        boundary = CollisionBoundary(left=0, top=0, right=100, bottom=100,
                                     mode='wrap')
        ps.set_boundary(boundary)
        em = Emitter(x=50, y=50, rate=10, speed=0, life=10.0)
        ps.add_emitter(em)
        ps.update(0.1)
        for p in ps.particles:
            p.x = 150  # 超出右边界
        ps.update(0.01)
        for p in ps.particles:
            assert p.x <= 100  # 应被环绕到左边界

    def test_collision_box(self):
        ps = ParticleSystem(max_particles=10)
        box = CollisionBox(x=40, y=40, width=20, height=20, restitution=1.0)
        ps.add_collision_box(box)
        em = Emitter(x=50, y=50, rate=10, speed=0, life=10.0)
        ps.add_emitter(em)
        ps.update(0.1)
        # 粒子在碰撞盒内，碰撞后应被弹出到边缘外
        for p in ps.particles:
            assert not box.contains_point(p.x, p.y)

    def test_event_on_birth(self):
        ps = ParticleSystem(max_particles=100)
        em = Emitter(x=50, y=50, rate=100)
        ps.add_emitter(em)
        births = []
        ps.on_particle_birth = lambda p: births.append(p)
        ps.update(0.1)
        assert len(births) > 0

    def test_event_on_death(self):
        ps = ParticleSystem(max_particles=100)
        em = Emitter(x=50, y=50, rate=10, life=0.1)
        ps.add_emitter(em)
        deaths = []
        ps.on_particle_death = lambda p: deaths.append(p)
        ps.update(0.1)   # 发射
        ps.update(0.2)   # 粒子死亡
        assert len(deaths) > 0

    def test_clear(self):
        ps = ParticleSystem(max_particles=100)
        em = Emitter(x=50, y=50, rate=100)
        ps.add_emitter(em)
        ps.update(0.1)
        assert len(ps.particles) > 0
        ps.clear()
        assert len(ps.particles) == 0

    def test_get_particle_count(self):
        ps = ParticleSystem(max_particles=100)
        em = Emitter(x=50, y=50, rate=100)
        ps.add_emitter(em)
        ps.update(0.1)
        assert ps.get_particle_count() > 0

    def test_time_property(self):
        ps = ParticleSystem()
        assert ps.time == 0.0
        ps.update(0.5)
        assert abs(ps.time - 0.5) < 0.01

    def test_render_calls_renderer(self):
        class MockRenderer(Renderer):
            def __init__(self):
                self.called = False
                self.particle_count = 0
            def render(self, particles):
                self.called = True
                self.particle_count = len(particles)

        ps = ParticleSystem(max_particles=100)
        em = Emitter(x=50, y=50, rate=100)
        ps.add_emitter(em)
        ps.update(0.1)

        renderer = MockRenderer()
        ps.render(renderer)
        assert renderer.called is True
        assert renderer.particle_count > 0


# ============================================================
#  Renderer 接口
# ============================================================

class TestRenderer:
    def test_base_renderer_raises(self):
        r = Renderer()
        with pytest.raises(NotImplementedError):
            r.render([])

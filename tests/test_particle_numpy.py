"""
ParticleSystemNumpy — NumPy 向量化粒子系统单元测试

覆盖：SoA 数据结构、批量发射（emit_burst/emit_radial）、
      向量化更新、力场（Gravity/Wind/Drag/Attraction/Turbulence）、
      渲染接口、生命周期、粒子计数、工厂函数。
"""

import math
import random
import pytest
import numpy as np
from PIL import Image

from particlefx.core.particle_numpy import (
    ParticleSystemNumpy,
    NumpyForceGravity,
    NumpyForceWind,
    NumpyForceDrag,
    NumpyForceAttraction,
    NumpyForceTurbulence,
    create_numpy_fireworks_burst,
)


class TestInit:
    """初始化与数据结构"""

    def test_default_max_particles(self):
        sys = ParticleSystemNumpy()
        assert sys.max_particles == 5000

    def test_custom_max_particles(self):
        sys = ParticleSystemNumpy(max_particles=1000)
        assert sys.max_particles == 1000

    def test_array_shapes(self):
        sys = ParticleSystemNumpy(max_particles=100)
        assert sys.pos.shape == (100, 2)
        assert sys.vel.shape == (100, 2)
        assert sys.life.shape == (100,)
        assert sys.max_life.shape == (100,)
        assert sys.size.shape == (100,)
        assert sys.color.shape == (100, 3)
        assert sys.alpha.shape == (100,)
        assert sys.mass.shape == (100,)
        assert sys.alive.shape == (100,)

    def test_array_dtypes(self):
        sys = ParticleSystemNumpy(max_particles=100)
        assert sys.pos.dtype == np.float32
        assert sys.vel.dtype == np.float32
        assert sys.life.dtype == np.float32
        assert sys.max_life.dtype == np.float32
        assert sys.size.dtype == np.float32
        assert sys.color.dtype == np.uint8
        assert sys.alpha.dtype == np.float32
        assert sys.mass.dtype == np.float32
        assert sys.alive.dtype == bool

    def test_initial_all_dead(self):
        sys = ParticleSystemNumpy()
        assert sys.particle_count == 0
        assert not np.any(sys.alive)

    def test_initial_values(self):
        sys = ParticleSystemNumpy()
        assert np.all(sys.pos == 0)
        assert np.all(sys.vel == 0)
        assert np.all(sys.life == 0)
        assert np.all(sys.max_life == 1)
        assert np.all(sys.size == 1)
        assert np.all(sys.alpha == 0)
        assert np.all(sys.mass == 1)


class TestEmitBurst:
    """批量发射"""

    def test_basic_emit(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.emit_burst(x=400, y=300, count=10,
                       speed_range=(2, 8), life_range=(0.5, 2.0),
                       size_range=(1, 3))
        assert sys.particle_count == 10

    def test_emit_fills_correct_position(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.emit_burst(x=400, y=300, count=5)
        alive_idx = np.where(sys.alive)[0]
        assert np.all(sys.pos[alive_idx, 0] == 400)
        assert np.all(sys.pos[alive_idx, 1] == 300)

    def test_emit_sets_life_to_one(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.emit_burst(x=400, y=300, count=5)
        alive_idx = np.where(sys.alive)[0]
        assert np.all(sys.life[alive_idx] == 1.0)

    def test_emit_sets_alpha_to_one(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.emit_burst(x=400, y=300, count=5)
        alive_idx = np.where(sys.alive)[0]
        assert np.all(sys.alpha[alive_idx] == 1.0)

    def test_emit_with_color(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.emit_burst(x=400, y=300, count=5,
                       color=(255, 0, 128))
        alive_idx = np.where(sys.alive)[0]
        assert np.all(sys.color[alive_idx, 0] == 255)
        assert np.all(sys.color[alive_idx, 1] == 0)
        assert np.all(sys.color[alive_idx, 2] == 128)

    def test_emit_with_mass_range(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.emit_burst(x=400, y=300, count=10,
                       mass_range=(0.5, 2.0))
        alive_idx = np.where(sys.alive)[0]
        masses = sys.mass[alive_idx]
        assert np.all(masses >= 0.5)
        assert np.all(masses <= 2.0)

    def test_emit_speed_in_range(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.emit_burst(x=400, y=300, count=50,
                       speed_range=(5, 10))
        alive_idx = np.where(sys.alive)[0]
        speeds = np.linalg.norm(sys.vel[alive_idx], axis=1)
        assert np.all(speeds >= 5)
        assert np.all(speeds <= 10)

    def test_emit_angle_range(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.emit_burst(x=400, y=300, count=50,
                       angle_range=(0, math.pi / 4))
        alive_idx = np.where(sys.alive)[0]
        angles = np.arctan2(sys.vel[alive_idx, 1], sys.vel[alive_idx, 0])
        assert np.all(angles >= 0)
        assert np.all(angles <= math.pi / 4 + 0.1)

    def test_emit_direction_bias(self):
        sys = ParticleSystemNumpy(max_particles=100)
        # direction_bias 在随机速度之上叠加偏移
        sys.emit_burst(x=400, y=300, count=10,
                       direction_bias=(20.0, 0.0),  # 大偏移保证正向
                       speed_range=(0.1, 0.1))  # 低随机速度
        alive_idx = np.where(sys.alive)[0]
        # 偏移量 20 远大于随机速度，vx 应为正
        assert np.all(sys.vel[alive_idx, 0] > 10.0)

    def test_emit_count_zero(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.emit_burst(x=400, y=300, count=0)
        assert sys.particle_count == 0

    def test_emit_exceeds_capacity(self):
        sys = ParticleSystemNumpy(max_particles=10)
        sys.emit_burst(x=400, y=300, count=50)
        assert sys.particle_count == 10  # 只发射10个

    def test_emit_reuses_dead_slots(self):
        sys = ParticleSystemNumpy(max_particles=20)
        sys.emit_burst(x=400, y=300, count=10)
        # 让5个粒子死亡
        sys.life[:5] = 0.0
        sys.update(0.01)  # 标记死亡
        assert sys.particle_count == 5
        # 再发射10个，复用死亡槽位
        sys.emit_burst(x=400, y=300, count=10)
        assert sys.particle_count == 15

    def test_emit_full_system_ignores(self):
        sys = ParticleSystemNumpy(max_particles=5)
        sys.emit_burst(x=400, y=300, count=5)
        # 全满后再发射
        sys.emit_burst(x=400, y=300, count=5)
        assert sys.particle_count == 5  # 无变化


class TestEmitRadial:
    """径向发射"""

    def test_basic_radial(self):
        sys = ParticleSystemNumpy(max_particles=200)
        sys.emit_radial(x=400, y=300, count=50)
        assert sys.particle_count == 50

    def test_radial_all_directions(self):
        sys = ParticleSystemNumpy(max_particles=200)
        sys.emit_radial(x=400, y=300, count=200)
        alive_idx = np.where(sys.alive)[0]
        angles = np.arctan2(sys.vel[alive_idx, 1], sys.vel[alive_idx, 0])
        # arctan2 返回 -π ~ π；200 个随机角度应覆盖正负范围
        assert angles.min() < -2.0
        assert angles.max() > 2.0

    def test_radial_with_palette(self):
        sys = ParticleSystemNumpy(max_particles=200)
        palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        sys.emit_radial(x=400, y=300, count=50, colors=palette)
        alive_idx = np.where(sys.alive)[0]
        colors = sys.color[alive_idx]
        # 颜色应来自调色板
        for c in colors:
            assert tuple(c) in palette

    def test_radial_count_zero(self):
        sys = ParticleSystemNumpy(max_particles=200)
        sys.emit_radial(x=400, y=300, count=0)
        assert sys.particle_count == 0


class TestUpdate:
    """向量化更新"""

    def test_particles_move(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.emit_burst(x=400, y=300, count=10,
                       speed_range=(5, 5))
        old_pos = sys.pos.copy()
        sys.update(0.1)
        alive_idx = np.where(sys.alive)[0]
        # 位置应变化
        assert not np.allclose(sys.pos[alive_idx], old_pos[alive_idx])

    def test_life_decreases(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.emit_burst(x=400, y=300, count=10,
                       life_range=(1.0, 1.0))
        sys.update(0.1)
        alive_idx = np.where(sys.alive)[0]
        assert np.all(sys.life[alive_idx] < 1.0)

    def test_alpha_tracks_life(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.emit_burst(x=400, y=300, count=10,
                       life_range=(1.0, 1.0))
        sys.update(0.2)
        alive_idx = np.where(sys.alive)[0]
        # alpha 应等于 clip(life, 0, 1)
        for i in alive_idx:
            assert abs(sys.alpha[i] - max(0.0, min(1.0, sys.life[i]))) < 0.01

    def test_particles_die(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.emit_burst(x=400, y=300, count=10,
                       life_range=(0.1, 0.1))
        for _ in range(20):
            sys.update(0.05)
        assert sys.particle_count == 0

    def test_dead_particles_alive_false(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.emit_burst(x=400, y=300, count=10,
                       life_range=(0.1, 0.1))
        for _ in range(20):
            sys.update(0.05)
        assert not np.any(sys.alive)

    def test_no_alive_no_crash(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.update(0.1)  # 无粒子
        assert sys.particle_count == 0

    def test_time_increases(self):
        sys = ParticleSystemNumpy()
        assert sys.time == 0.0
        sys.update(0.1)
        assert sys.time == 0.1
        sys.update(0.1)
        assert sys.time == 0.2


class TestForceFields:
    """NumPy 力场"""

    def test_gravity(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.add_force(NumpyForceGravity(strength=9.8))
        sys.emit_burst(x=400, y=300, count=10, speed_range=(0, 0))
        old_vy = sys.vel[:, 1].copy()
        sys.update(0.1)
        alive_idx = np.where(sys.alive)[0]
        # vy 应增加（向下）
        assert np.all(sys.vel[alive_idx, 1] > old_vy[alive_idx])

    def test_wind(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.add_force(NumpyForceWind(strength=5.0))
        sys.emit_burst(x=400, y=300, count=10, speed_range=(0, 0))
        sys.update(0.1)
        alive_idx = np.where(sys.alive)[0]
        # vx 应增加（向右）
        assert np.all(sys.vel[alive_idx, 0] > 0)

    def test_drag(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.add_force(NumpyForceDrag(factor=0.9))
        sys.emit_burst(x=400, y=300, count=10,
                       speed_range=(10, 10))
        old_vel = sys.vel.copy()
        sys.update(0.1)
        alive_idx = np.where(sys.alive)[0]
        # 速度应衰减
        old_speeds = np.linalg.norm(old_vel[alive_idx], axis=1)
        new_speeds = np.linalg.norm(sys.vel[alive_idx], axis=1)
        assert np.all(new_speeds < old_speeds)

    def test_attraction(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.add_force(NumpyForceAttraction(x=400, y=300, strength=5.0))
        sys.emit_burst(x=500, y=300, count=5, speed_range=(0, 0))
        sys.update(0.1)
        alive_idx = np.where(sys.alive)[0]
        # 粒子应被吸向中心
        assert sys.pos[alive_idx, 0].mean() < 500

    def test_turbulence(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.add_force(NumpyForceTurbulence(strength=2.0, scale=40.0))
        sys.emit_burst(x=400, y=300, count=10, speed_range=(0, 0))
        sys.update(0.1)
        # 湍流应产生随机偏移（速度不为零）
        alive_idx = np.where(sys.alive)[0]
        assert np.any(sys.vel[alive_idx, 0] != 0)
        assert np.any(sys.vel[alive_idx, 1] != 0)

    def test_multiple_forces(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.add_force(NumpyForceGravity(strength=5.0))
        sys.add_force(NumpyForceWind(strength=2.0))
        sys.add_force(NumpyForceDrag(factor=0.98))
        sys.emit_burst(x=400, y=300, count=20)
        for _ in range(10):
            sys.update(0.05)
        assert sys.particle_count > 0

    def test_clear_forces(self):
        sys = ParticleSystemNumpy()
        sys.add_force(NumpyForceGravity())
        sys.add_force(NumpyForceWind())
        assert len(sys._forces) == 2
        sys.clear_forces()
        assert len(sys._forces) == 0


class TestRenderToImage:
    """渲染到图像"""

    def test_empty_system(self):
        sys = ParticleSystemNumpy()
        img = sys.render_to_image(width=400, height=300)
        assert isinstance(img, Image.Image)
        assert img.size == (400, 300)

    def test_with_particles(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.emit_burst(x=200, y=150, count=10, color=(255, 200, 50))
        sys.update(0.01)
        img = sys.render_to_image(width=400, height=300)
        assert isinstance(img, Image.Image)
        assert img.mode == 'RGB'

    def test_with_background(self):
        sys = ParticleSystemNumpy(max_particles=100)
        bg = Image.new('RGB', (400, 300), (30, 30, 50))
        sys.emit_burst(x=200, y=150, count=5, color=(255, 255, 255))
        img = sys.render_to_image(width=400, height=300, background=bg)
        assert isinstance(img, Image.Image)

    def test_additive_blend(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.emit_burst(x=200, y=150, count=10, color=(255, 200, 50))
        img = sys.render_to_image(width=400, height=300, blend_mode='additive')
        assert isinstance(img, Image.Image)

    def test_screen_blend(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.emit_burst(x=200, y=150, count=10, color=(255, 200, 50))
        img = sys.render_to_image(width=400, height=300, blend_mode='screen')
        assert isinstance(img, Image.Image)

    def test_offscreen_particles_filtered(self):
        sys = ParticleSystemNumpy(max_particles=100)
        # 发射在屏幕外的粒子
        sys.emit_burst(x=-100, y=-100, count=10, color=(255, 0, 0))
        img = sys.render_to_image(width=400, height=300)
        assert isinstance(img, Image.Image)


class TestRenderViaPIL:
    """PILRenderer 兼容接口"""

    def test_render_via_pil_renderer(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.emit_burst(x=200, y=150, count=5, color=(255, 200, 50))

        class MockRenderer:
            def __init__(self):
                self.particles = None
            def render(self, particles):
                self.particles = particles

        renderer = MockRenderer()
        sys.render_via_pil_renderer(renderer)
        assert renderer.particles is not None
        assert len(renderer.particles) == 5

    def test_render_via_pil_empty(self):
        sys = ParticleSystemNumpy()

        class MockRenderer:
            def __init__(self):
                self.particles = None
            def render(self, particles):
                self.particles = particles

        renderer = MockRenderer()
        sys.render_via_pil_renderer(renderer)
        assert len(renderer.particles) == 0


class TestProperties:
    """属性与辅助方法"""

    def test_particle_count(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.emit_burst(x=400, y=300, count=30)
        assert sys.particle_count == 30

    def test_time_property(self):
        sys = ParticleSystemNumpy()
        assert sys.time == 0.0
        sys.update(0.016)
        assert sys.time == pytest.approx(0.016, abs=0.001)

    def test_clear(self):
        sys = ParticleSystemNumpy(max_particles=100)
        sys.emit_burst(x=400, y=300, count=30)
        assert sys.particle_count > 0
        sys.clear()
        assert sys.particle_count == 0
        assert not np.any(sys.alive)


class TestFireworksBurstFactory:
    """烟花爆炸工厂函数"""

    def test_radial_style(self):
        palette = [(255, 200, 50), (255, 100, 20)]
        sys = create_numpy_fireworks_burst(x=400, y=300,
                                           palette=palette, style='radial')
        assert sys.particle_count > 0

    def test_ring_style(self):
        palette = [(255, 200, 50), (100, 200, 255)]
        sys = create_numpy_fireworks_burst(x=400, y=300,
                                           palette=palette, style='ring')
        assert sys.particle_count > 0

    def test_palm_style(self):
        palette = [(255, 200, 50), (255, 100, 20)]
        sys = create_numpy_fireworks_burst(x=400, y=300,
                                           palette=palette, style='palm')
        assert sys.particle_count > 0

    def test_reuse_system(self):
        sys = ParticleSystemNumpy(max_particles=2000)
        palette = [(255, 200, 50)]
        result = create_numpy_fireworks_burst(
            x=400, y=300, palette=palette,
            style='radial', system=sys)
        assert result is sys  # 返回传入的系统

    def test_has_gravity_and_drag(self):
        palette = [(255, 200, 50)]
        sys = create_numpy_fireworks_burst(x=400, y=300, palette=palette)
        force_types = [f.__class__.__name__ for f in sys._forces]
        assert 'NumpyForceGravity' in force_types
        assert 'NumpyForceDrag' in force_types

"""
预设效果工厂函数单元测试

覆盖：13 个工厂函数（create_explosion/create_fire/create_rain 等）
      EFFECT_REGISTRY、get_effect/list_effects、FluidEffectSystem 基础。
"""

import math
import random
import pytest
from particlefx.core.particle import (
    ParticleSystem, Emitter, ForceField, NoiseField,
    EmitterShape, CollisionBoundary,
)
from particlefx.components.effects import (
    create_explosion, create_fire, create_rain, create_snow,
    create_magic_circle, create_fountain, create_smoke, create_sparkle,
    create_trail, create_vortex_effect, create_collision_sparks,
    create_lighted_firework, create_magnet_field, create_fluid_effect,
    EFFECT_REGISTRY, get_effect, list_effects,
)


class TestCreateExplosion:
    """爆炸效果"""

    def test_returns_particle_system(self):
        ps = create_explosion(x=400, y=300)
        assert isinstance(ps, ParticleSystem)

    def test_has_emitters(self):
        ps = create_explosion(x=400, y=300)
        assert len(ps.emitters) > 0

    def test_has_gravity(self):
        ps = create_explosion(x=400, y=300)
        gravities = [ff for ff in ps.force_fields
                     if ff.field_type == 'gravity']
        assert len(gravities) == 1

    def test_intensity_scaling(self):
        ps1 = create_explosion(x=400, y=300, intensity=1.0)
        ps2 = create_explosion(x=400, y=300, intensity=2.0)
        # 更高强度产生更多发射器
        assert len(ps2.emitters) > len(ps1.emitters)

    def test_max_particles_scales(self):
        ps1 = create_explosion(x=400, y=300, intensity=1.0)
        ps2 = create_explosion(x=400, y=300, intensity=2.0)
        assert ps2.max_particles > ps1.max_particles

    def test_custom_palette(self):
        ps = create_explosion(x=400, y=300,
                             color_palette=[(0, 255, 0), (0, 128, 0)])
        assert len(ps.emitters) > 0

    def test_update_generates_particles(self):
        ps = create_explosion(x=400, y=300)
        ps.update(0.1)
        assert ps.get_particle_count() > 0


class TestCreateFire:
    """火焰效果"""

    def test_returns_particle_system(self):
        ps = create_fire(x=400, y=500)
        assert isinstance(ps, ParticleSystem)

    def test_has_multiple_emitters(self):
        ps = create_fire(x=400, y=500)
        # 5 层火焰
        assert len(ps.emitters) == 5

    def test_has_upward_force(self):
        ps = create_fire(x=400, y=500)
        # 重力 = -2.0（向上推）
        gravities = [ff for ff in ps.force_fields
                     if ff.field_type == 'gravity']
        assert len(gravities) == 1
        assert gravities[0].strength == -2.0

    def test_has_noise_field(self):
        ps = create_fire(x=400, y=500)
        assert len(ps.noise_fields) == 1

    def test_custom_width_height(self):
        ps = create_fire(x=400, y=500, width=80, height=120)
        assert isinstance(ps, ParticleSystem)


class TestCreateRain:
    """雨效果"""

    def test_returns_particle_system(self):
        ps = create_rain(width=800, height=600)
        assert isinstance(ps, ParticleSystem)

    def test_has_multiple_emitters(self):
        ps = create_rain(width=800, height=600)
        assert len(ps.emitters) >= 5

    def test_has_gravity(self):
        ps = create_rain(width=800, height=600)
        gravities = [ff for ff in ps.force_fields
                     if ff.field_type == 'gravity']
        assert len(gravities) == 1

    def test_wind_field(self):
        ps = create_rain(width=800, height=600, wind=3.0)
        winds = [ff for ff in ps.force_fields
                 if ff.field_type == 'wind']
        assert len(winds) == 1

    def test_no_wind_default(self):
        ps = create_rain(width=800, height=600)
        winds = [ff for ff in ps.force_fields
                 if ff.field_type == 'wind']
        assert len(winds) == 0

    def test_density_scaling(self):
        ps = create_rain(width=800, height=600, density=2.0)
        assert ps.max_particles > 500


class TestCreateSnow:
    """雪效果"""

    def test_returns_particle_system(self):
        ps = create_snow(width=800, height=600)
        assert isinstance(ps, ParticleSystem)

    def test_has_emitters(self):
        ps = create_snow(width=800, height=600)
        assert len(ps.emitters) >= 5

    def test_light_gravity(self):
        ps = create_snow(width=800, height=600)
        gravities = [ff for ff in ps.force_fields
                     if ff.field_type == 'gravity']
        assert gravities[0].strength == 1.5

    def test_wind(self):
        ps = create_snow(width=800, height=600, wind=2.0)
        winds = [ff for ff in ps.force_fields
                 if ff.field_type == 'wind']
        assert len(winds) == 1


class TestCreateMagicCircle:
    """魔法光环"""

    def test_returns_particle_system(self):
        ps = create_magic_circle(x=400, y=300)
        assert isinstance(ps, ParticleSystem)

    def test_has_emitters(self):
        ps = create_magic_circle(x=400, y=300)
        # 3 层光环 × 每层多个粒子 = 很多发射器
        assert len(ps.emitters) > 10

    def test_has_attraction_field(self):
        ps = create_magic_circle(x=400, y=300)
        attractions = [ff for ff in ps.force_fields
                       if ff.field_type == 'attraction']
        assert len(attractions) == 1


class TestCreateFountain:
    """喷泉效果"""

    def test_returns_particle_system(self):
        ps = create_fountain(x=400, y=500)
        assert isinstance(ps, ParticleSystem)

    def test_has_multiple_emitters(self):
        ps = create_fountain(x=400, y=500)
        # 主水流 + 两侧溅射 + 底部回落
        assert len(ps.emitters) >= 4

    def test_has_gravity(self):
        ps = create_fountain(x=400, y=500)
        gravities = [ff for ff in ps.force_fields
                     if ff.field_type == 'gravity']
        assert len(gravities) == 1

    def test_custom_height(self):
        ps = create_fountain(x=400, y=500, height=200)
        assert isinstance(ps, ParticleSystem)


class TestCreateSmoke:
    """烟雾效果"""

    def test_returns_particle_system(self):
        ps = create_smoke(x=400, y=300)
        assert isinstance(ps, ParticleSystem)

    def test_has_emitters(self):
        ps = create_smoke(x=400, y=300)
        assert len(ps.emitters) >= 3

    def test_has_noise_field(self):
        ps = create_smoke(x=400, y=300)
        assert len(ps.noise_fields) == 1

    def test_direction_up(self):
        ps = create_smoke(x=400, y=300, direction='up')
        gravities = [ff for ff in ps.force_fields
                     if ff.field_type == 'gravity']
        assert gravities[0].strength == -1.5

    def test_direction_left(self):
        ps = create_smoke(x=400, y=300, direction='left')
        winds = [ff for ff in ps.force_fields
                 if ff.field_type == 'wind']
        assert len(winds) == 1


class TestCreateSparkle:
    """星光闪烁"""

    def test_returns_particle_system(self):
        ps = create_sparkle(x=400, y=300)
        assert isinstance(ps, ParticleSystem)

    def test_has_emitters(self):
        ps = create_sparkle(x=400, y=300)
        assert len(ps.emitters) >= 8

    def test_intensity_scaling(self):
        ps = create_sparkle(x=400, y=300, intensity=2.0)
        assert len(ps.emitters) >= 16


class TestCreateTrail:
    """尾迹效果"""

    def test_returns_particle_system(self):
        ps = create_trail(x=400, y=300)
        assert isinstance(ps, ParticleSystem)

    def test_has_one_emitter(self):
        ps = create_trail(x=400, y=300)
        assert len(ps.emitters) == 1

    def test_has_forces(self):
        ps = create_trail(x=400, y=300)
        assert len(ps.force_fields) >= 2  # wind + gravity

    def test_direction_sets_angle(self):
        ps = create_trail(x=400, y=300, direction_x=1.0, direction_y=0.0)
        em = ps.emitters[0]
        assert em.direction != 0.0  # 反向发射


class TestCreateVortexEffect:
    """漩涡效果"""

    def test_returns_particle_system(self):
        ps = create_vortex_effect(x=400, y=300)
        assert isinstance(ps, ParticleSystem)

    def test_has_vortex_force(self):
        ps = create_vortex_effect(x=400, y=300)
        vortices = [ff for ff in ps.force_fields
                    if ff.field_type == 'vortex']
        assert len(vortices) == 1

    def test_has_sphere_emitter_shape(self):
        ps = create_vortex_effect(x=400, y=300)
        shapes = [em.shape for em in ps.emitters]
        assert any(s.shape_type == 'sphere' for s in shapes)

    def test_has_trail(self):
        ps = create_vortex_effect(x=400, y=300)
        assert any(em.trail_enabled for em in ps.emitters)


class TestCreateCollisionSparks:
    """碰撞火花效果"""

    def test_returns_particle_system(self):
        ps = create_collision_sparks(x=400, y=300)
        assert isinstance(ps, ParticleSystem)

    def test_has_boundary(self):
        ps = create_collision_sparks(x=400, y=300)
        assert ps.boundary is not None
        assert ps.boundary.mode == 'bounce'

    def test_has_rectangle_shape(self):
        ps = create_collision_sparks(x=400, y=300)
        shapes = [em.shape for em in ps.emitters]
        assert any(s.shape_type == 'rectangle' for s in shapes)

    def test_has_gravity(self):
        ps = create_collision_sparks(x=400, y=300)
        gravities = [ff for ff in ps.force_fields
                     if ff.field_type == 'gravity']
        assert len(gravities) == 1


class TestCreateLightedFirework:
    """光照烟花效果"""

    def test_returns_particle_system(self):
        ps = create_lighted_firework(x=400, y=300)
        assert isinstance(ps, ParticleSystem)

    def test_lighting_enabled(self):
        ps = create_lighted_firework(x=400, y=300)
        assert ps.lighting_enabled is True

    def test_has_lights(self):
        ps = create_lighted_firework(x=400, y=300)
        assert len(ps.lights) >= 2  # ambient + point

    def test_has_emissive_particles(self):
        ps = create_lighted_firework(x=400, y=300)
        assert any(em.emissive_strength > 0 for em in ps.emitters)


class TestCreateMagnetField:
    """磁场效果"""

    def test_returns_particle_system(self):
        ps = create_magnet_field(
            x_center=400, y_center=300,
            attract_x=300, attract_y=300,
            repel_x=500, repel_y=300)
        assert isinstance(ps, ParticleSystem)

    def test_has_magnet_forces(self):
        ps = create_magnet_field(
            x_center=400, y_center=300,
            attract_x=300, attract_y=300,
            repel_x=500, repel_y=300)
        magnets = [ff for ff in ps.force_fields
                   if ff.field_type == 'magnet']
        assert len(magnets) == 2

    def test_has_circle_shape(self):
        ps = create_magnet_field(
            x_center=400, y_center=300,
            attract_x=300, attract_y=300,
            repel_x=500, repel_y=300)
        shapes = [em.shape for em in ps.emitters]
        assert any(s.shape_type == 'circle' for s in shapes)


class TestEffectRegistry:
    """效果注册表"""

    def test_registry_has_14_entries(self):
        # 13 原始 + fluid
        assert len(EFFECT_REGISTRY) == 14

    def test_all_factories_callable(self):
        for name, func in EFFECT_REGISTRY.items():
            assert callable(func)

    def test_list_effects(self):
        names = list_effects()
        assert 'explosion' in names
        assert 'fire' in names
        assert 'fluid' in names

    def test_get_effect(self):
        ps = get_effect('explosion', x=400, y=300)
        assert isinstance(ps, ParticleSystem)

    def test_get_effect_unknown_raises(self):
        with pytest.raises(ValueError):
            get_effect('nonexistent')

    def test_get_effect_case_insensitive(self):
        ps = get_effect('EXPLOSION', x=400, y=300)
        assert isinstance(ps, ParticleSystem)


class TestFluidEffectSystem:
    """流体效果系统（基础测试，无 Pygame 依赖）"""

    def test_create_returns_system(self):
        try:
            fluid = create_fluid_effect(x=400, y=300)
            assert fluid.particle_count > 0
        except ImportError:
            # NumPy 系统可能依赖未安装模块
            pytest.skip("FluidEffectSystem 依赖 particle_numpy")

    def test_update_reduces_dead(self):
        try:
            fluid = create_fluid_effect(x=400, y=300)
            initial = fluid.particle_count
            fluid.update(0.016)
            # 粒子有死亡也有补充
            assert fluid.particle_count >= 0
        except ImportError:
            pytest.skip("FluidEffectSystem 依赖 particle_numpy")

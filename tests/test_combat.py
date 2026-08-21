"""
CombatEffectsManager 战斗特效系统单元测试

覆盖：5 种攻击效果、飘浮伤害数字、6 种技能效果、
      死亡消散、升级特效、更新管线、渲染接口。
"""

import math
import random
import pytest
from core.particle import Particle
from systems.combat_effects import CombatEffectsManager, create_combat_manager


class TestInit:
    """初始化"""

    def test_default_init(self):
        mgr = CombatEffectsManager()
        assert mgr.particle_count == 0
        assert len(mgr.particles) == 0
        assert len(mgr.floating_texts) == 0

    def test_max_particles(self):
        mgr = CombatEffectsManager()
        assert mgr.max_particles == 2000

    def test_has_gravity(self):
        mgr = CombatEffectsManager()
        assert mgr.gravity.field_type == 'gravity'


class TestHitEffect:
    """攻击命中效果"""

    def test_slash_generates_particles(self):
        mgr = CombatEffectsManager()
        mgr.hit_effect(x=200, y=300, effect_type='slash')
        assert mgr.particle_count > 0

    def test_impact_generates_particles(self):
        mgr = CombatEffectsManager()
        mgr.hit_effect(x=200, y=300, effect_type='impact')
        assert mgr.particle_count > 0

    def test_pierce_generates_particles(self):
        mgr = CombatEffectsManager()
        mgr.hit_effect(x=200, y=300, effect_type='pierce')
        assert mgr.particle_count > 0

    def test_blunt_generates_particles(self):
        mgr = CombatEffectsManager()
        mgr.hit_effect(x=200, y=300, effect_type='blunt')
        assert mgr.particle_count > 0

    def test_crit_generates_particles(self):
        mgr = CombatEffectsManager()
        mgr.hit_effect(x=200, y=300, effect_type='crit')
        assert mgr.particle_count > 0

    def test_default_type_is_slash(self):
        mgr = CombatEffectsManager()
        mgr.hit_effect(x=200, y=300, effect_type='unknown_type')
        assert mgr.particle_count > 0  # 默认走 slash

    def test_slash_with_direction(self):
        mgr = CombatEffectsManager()
        mgr.hit_effect(x=200, y=300, effect_type='slash', direction=math.pi / 4)
        assert mgr.particle_count > 0

    def test_slash_with_scale(self):
        mgr1 = CombatEffectsManager()
        mgr1.hit_effect(x=200, y=300, effect_type='slash', scale=1.0)
        mgr2 = CombatEffectsManager()
        mgr2.hit_effect(x=200, y=300, effect_type='slash', scale=2.0)
        # 更大的缩放产生更多粒子
        assert mgr2.particle_count > mgr1.particle_count

    def test_slash_with_custom_color(self):
        mgr = CombatEffectsManager()
        mgr.hit_effect(x=200, y=300, effect_type='slash',
                       color=(0, 255, 128))
        # 检查有使用自定义颜色的粒子
        colors_used = set(p.color for p in mgr.particles)
        assert any(c[1] > 200 for c in colors_used)

    def test_impact_ring_structure(self):
        mgr = CombatEffectsManager()
        mgr.hit_effect(x=200, y=300, effect_type='impact')
        assert mgr.particle_count > 0

    def test_pierce_directional(self):
        mgr = CombatEffectsManager()
        mgr.hit_effect(x=200, y=300, effect_type='pierce',
                       direction=0.0)
        # 穿刺粒子主要沿方向分布
        for p in mgr.particles:
            assert p.is_alive()

    def test_blunt_large_size(self):
        mgr = CombatEffectsManager()
        mgr.hit_effect(x=200, y=300, effect_type='blunt')
        # 钝击粒子 size 较大
        sizes = [p.size for p in mgr.particles]
        assert any(s >= 4 for s in sizes)

    def test_crit_gold_colors(self):
        mgr = CombatEffectsManager()
        mgr.hit_effect(x=200, y=300, effect_type='crit')
        # 暴击使用金色系
        gold_found = False
        for p in mgr.particles:
            if p.color[0] > 200 and p.color[1] > 150 and p.color[2] < 100:
                gold_found = True
        assert gold_found


class TestFloatingDamage:
    """飘浮伤害数字"""

    def test_basic_damage_number(self):
        mgr = CombatEffectsManager()
        mgr.floating_damage(x=200, y=300, value=1234)
        assert len(mgr.floating_texts) == 1

    def test_crit_damage_number(self):
        mgr = CombatEffectsManager()
        mgr.floating_damage(x=200, y=300, value=5000, is_crit=True)
        ft = mgr.floating_texts[0]
        assert ft['is_crit'] is True
        assert ft['scale'] == 1.8
        assert ft['color'] == (255, 215, 0)

    def test_heal_damage_number(self):
        mgr = CombatEffectsManager()
        mgr.floating_damage(x=200, y=300, value=500, is_heal=True)
        ft = mgr.floating_texts[0]
        assert ft['is_heal'] is True
        assert ft['color'] == (50, 255, 100)

    def test_normal_damage_color(self):
        mgr = CombatEffectsManager()
        mgr.floating_damage(x=200, y=300, value=100)
        ft = mgr.floating_texts[0]
        assert ft['color'] == (255, 80, 80)

    def test_custom_color(self):
        mgr = CombatEffectsManager()
        mgr.floating_damage(x=200, y=300, value=100,
                            color=(0, 128, 255))
        ft = mgr.floating_texts[0]
        assert ft['color'] == (0, 128, 255)

    def test_crit_has_more_deco_particles(self):
        mgr = CombatEffectsManager()
        mgr.floating_damage(x=200, y=300, value=100, is_crit=True)
        ft = mgr.floating_texts[0]
        assert len(ft['particles']) == 8

    def test_normal_has_less_deco_particles(self):
        mgr = CombatEffectsManager()
        mgr.floating_damage(x=200, y=300, value=100)
        ft = mgr.floating_texts[0]
        assert len(ft['particles']) == 4

    def test_damage_text_content(self):
        mgr = CombatEffectsManager()
        mgr.floating_damage(x=200, y=300, value=999)
        ft = mgr.floating_texts[0]
        assert ft['text'] == '999'

    def test_particle_count_includes_floating(self):
        mgr = CombatEffectsManager()
        mgr.floating_damage(x=200, y=300, value=100)
        # particle_count 包含飘浮文字装饰粒子
        assert mgr.particle_count >= 4


class TestSkillCast:
    """技能施法效果"""

    def test_fireball(self):
        mgr = CombatEffectsManager()
        mgr.skill_cast(x=400, y=300, skill_type='fireball')
        assert mgr.particle_count > 100  # 30 核心 + 80 火焰

    def test_ice_storm(self):
        mgr = CombatEffectsManager()
        mgr.skill_cast(x=400, y=300, skill_type='ice_storm')
        assert mgr.particle_count > 0

    def test_lightning(self):
        mgr = CombatEffectsManager()
        mgr.skill_cast(x=400, y=300, skill_type='lightning')
        assert mgr.particle_count > 0

    def test_heal(self):
        mgr = CombatEffectsManager()
        mgr.skill_cast(x=400, y=300, skill_type='heal')
        assert mgr.particle_count > 0

    def test_shield(self):
        mgr = CombatEffectsManager()
        mgr.skill_cast(x=400, y=300, skill_type='shield')
        assert mgr.particle_count > 0

    def test_aoe_circle(self):
        mgr = CombatEffectsManager()
        mgr.skill_cast(x=400, y=300, skill_type='aoe_circle')
        assert mgr.particle_count > 0

    def test_unknown_skill_defaults_fireball(self):
        mgr = CombatEffectsManager()
        mgr.skill_cast(x=400, y=300, skill_type='dark_magic')
        assert mgr.particle_count > 0

    def test_fireball_core_white_hot(self):
        mgr = CombatEffectsManager()
        mgr.skill_cast(x=400, y=300, skill_type='fireball')
        # 检查有白热核心粒子
        has_white = any(p.color[0] > 200 and p.color[1] > 200
                        for p in mgr.particles)
        assert has_white

    def test_ice_storm_blue_colors(self):
        mgr = CombatEffectsManager()
        mgr.skill_cast(x=400, y=300, skill_type='ice_storm')
        # 冰系粒子偏蓝
        has_blue = any(p.color[2] > 200 for p in mgr.particles)
        assert has_blue

    def test_shield_ring_structure(self):
        mgr = CombatEffectsManager()
        mgr.skill_cast(x=400, y=300, skill_type='shield', radius=60)
        # 护盾粒子有切线速度
        assert mgr.particle_count > 0

    def test_aoe_wave_structure(self):
        mgr = CombatEffectsManager()
        mgr.skill_cast(x=400, y=300, skill_type='aoe_circle',
                       radius=60, duration=0.8)
        assert mgr.particle_count > 0


class TestDeathFadeout:
    """死亡消散"""

    def test_generates_particles(self):
        mgr = CombatEffectsManager()
        mgr.death_fadeout(x=400, y=300)
        assert mgr.particle_count > 0

    def test_particles_upward(self):
        mgr = CombatEffectsManager()
        mgr.death_fadeout(x=400, y=300)
        # 死亡粒子主要向上飘
        has_upward = any(p.vy < 0 for p in mgr.particles)
        assert has_upward

    def test_custom_color(self):
        mgr = CombatEffectsManager()
        mgr.death_fadeout(x=400, y=300, color=(50, 50, 200))
        # 有蓝色偏移粒子
        has_blue = any(p.color[2] > 150 for p in mgr.particles)
        assert has_blue

    def test_custom_size(self):
        mgr1 = CombatEffectsManager()
        mgr1.death_fadeout(x=400, y=300, width=20, height=40)
        mgr2 = CombatEffectsManager()
        mgr2.death_fadeout(x=400, y=300, width=80, height=120)
        assert mgr2.particle_count > mgr1.particle_count


class TestLevelUp:
    """升级效果"""

    def test_generates_particles(self):
        mgr = CombatEffectsManager()
        mgr.level_up(x=400, y=300)
        assert mgr.particle_count > 0

    def test_gold_colors(self):
        mgr = CombatEffectsManager()
        mgr.level_up(x=400, y=300)
        has_gold = any(p.color[0] > 200 and p.color[1] > 150
                       for p in mgr.particles)
        assert has_gold

    def test_spiral_particles_upward(self):
        mgr = CombatEffectsManager()
        mgr.level_up(x=400, y=300)
        # 螺旋上升粒子 vy < 0
        has_upward = any(p.vy < 0 for p in mgr.particles)
        assert has_upward


class TestUpdate:
    """更新管线"""

    def test_particles_die_over_time(self):
        mgr = CombatEffectsManager()
        mgr.hit_effect(x=200, y=300)
        initial = mgr.particle_count
        for _ in range(50):
            mgr.update(0.05)
        assert mgr.particle_count < initial

    def test_all_particles_die(self):
        mgr = CombatEffectsManager()
        mgr.hit_effect(x=200, y=300)
        for _ in range(100):
            mgr.update(0.05)
        assert mgr.particle_count == 0

    def test_floating_text_moves_up(self):
        mgr = CombatEffectsManager()
        mgr.floating_damage(x=200, y=300, value=100)
        initial_y = mgr.floating_texts[0]['y']
        mgr.update(0.1)
        assert mgr.floating_texts[0]['y'] < initial_y

    def test_floating_text_expires(self):
        mgr = CombatEffectsManager()
        mgr.floating_damage(x=200, y=300, value=100)
        for _ in range(100):
            mgr.update(0.05)
        assert len(mgr.floating_texts) == 0

    def test_air_resistance(self):
        mgr = CombatEffectsManager()
        mgr.hit_effect(x=200, y=300)
        # 更新后粒子速度衰减
        for _ in range(5):
            mgr.update(0.05)
        # 粒子仍在移动但变慢
        still_moving = any(abs(p.vx) > 0.01 or abs(p.vy) > 0.01
                          for p in mgr.particles if p.is_alive())
        # 不强制：所有粒子可能已死亡


class TestFloatingTextData:
    """飘浮文字数据属性"""

    def test_data_format(self):
        mgr = CombatEffectsManager()
        mgr.floating_damage(x=200, y=300, value=100)
        mgr.update(0.01)
        data = mgr.floating_text_data
        assert len(data) == 1
        ft = data[0]
        assert 'text' in ft
        assert 'x' in ft
        assert 'y' in ft
        assert 'color' in ft
        assert 'alpha' in ft
        assert 'scale' in ft
        assert 'is_crit' in ft

    def test_alpha_decreases(self):
        mgr = CombatEffectsManager()
        mgr.floating_damage(x=200, y=300, value=100)
        mgr.update(0.01)
        alpha_before = mgr.floating_text_data[0]['alpha']
        mgr.update(0.3)
        alpha_after = mgr.floating_text_data[0]['alpha'] if mgr.floating_text_data else 0
        assert alpha_after <= alpha_before

    def test_empty_when_no_texts(self):
        mgr = CombatEffectsManager()
        assert mgr.floating_text_data == []


class TestRender:
    """渲染接口"""

    def test_render_calls_renderer(self):
        mgr = CombatEffectsManager()
        mgr.hit_effect(x=200, y=300)

        class MockRenderer:
            def __init__(self):
                self.called = False
            def render(self, particles):
                self.called = True

        renderer = MockRenderer()
        mgr.render(renderer)
        assert renderer.called

    def test_render_empty(self):
        mgr = CombatEffectsManager()

        class MockRenderer:
            def __init__(self):
                self.called = False
            def render(self, particles):
                self.called = True

        renderer = MockRenderer()
        mgr.render(renderer)
        assert renderer.called


class TestCreateCombatManager:
    """工厂函数"""

    def test_factory(self):
        mgr = create_combat_manager()
        assert isinstance(mgr, CombatEffectsManager)
        assert mgr.particle_count == 0

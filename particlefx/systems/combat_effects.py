"""
战斗特效系统

为游戏战斗场景设计的粒子特效集合，支持：
- 攻击打击效果（斩击、冲击、穿透）
- 受伤反馈（数字飘浮、血花）
- 技能释放（魔法阵、剑气、范围AOE）
- Buff/Debuff 图标光环
- 死亡消散效果

使用示例：
    from particlefx.systems.combat_effects import CombatEffectsManager

    mgr = CombatEffectsManager()
    # 攻击命中
    mgr.hit_effect(x=200, y=300, damage=1234, effect_type='slash')
    # 更新
    while running:
        mgr.update(dt=0.016)
        mgr.render(renderer)
"""

import random
import math
import time

from ..core.particle import Particle, ForceField


class CombatEffectsManager:
    """
    战斗特效管理器

    管理所有战斗场景的粒子效果，提供简单的触发接口。
    所有效果通过方法调用触发，内部自动管理生命周期。
    """

    def __init__(self):
        self.particles: list = []
        self.floating_texts: list = []  # 飘浮文字数据
        self.gravity = ForceField(field_type='gravity', strength=2.0)

        # 效果池配置
        self.max_particles = 2000

    @property
    def particle_count(self) -> int:
        return len(self.particles) + sum(len(ft.get('particles', [])) for ft in self.floating_texts)

    # ================================================================
    #  攻击效果
    # ================================================================

    def hit_effect(self, x: float, y: float,
                   effect_type: str = 'slash',
                   direction: float = 0.0,
                   scale: float = 1.0,
                   color: tuple = None):
        """
        触发攻击命中特效

        Args:
            x, y: 命中位置
            effect_type: 效果类型 ('slash', 'impact', 'pierce', 'blunt', 'crit')
            direction: 攻击方向角度（弧度，0=向右）
            scale: 缩放倍数
            color: 主色调
        """
        if color is None:
            color = (255, 255, 255)

        effect_type = effect_type.lower()

        if effect_type == 'slash':
            self._effect_slash(x, y, direction, scale, color)
        elif effect_type == 'impact':
            self._effect_impact(x, y, scale, color)
        elif effect_type == 'pierce':
            self._effect_pierce(x, y, direction, scale, color)
        elif effect_type == 'blunt':
            self._effect_blunt(x, y, scale, color)
        elif effect_type == 'crit':
            self._effect_crit(x, y, scale)
        else:
            # 默认用 slash
            self._effect_slash(x, y, direction, scale, color)

    def _effect_slash(self, x, y, direction, scale, color):
        """斩击效果：弧形刀光粒子"""
        arc_length = math.pi * 0.8 * scale
        num_particles = int(40 * scale)

        colors = [color,
                  (min(255, color[0] + 50), min(255, color[1] + 50), min(255, color[2] + 30)),
                  (255, 255, 240)]

        for i in range(num_particles):
            t = i / num_particles
            angle = direction - arc_length / 2 + t * arc_length
            speed = random.uniform(3, 9) * scale * (1.1 - t * 0.3)  # 前端更快

            p = Particle(
                x=x + random.uniform(-5, 5),
                y=y + random.uniform(-5, 5),
                vx=math.cos(angle) * speed + random.uniform(-1, 1),
                vy=math.sin(angle) * speed + random.uniform(-1, 1),
                life=random.uniform(0.15, 0.35), max_life=0.35,
                size=random.uniform(2, 5) * scale * (1 - t * 0.4),
                color=random.choice(colors),
                alpha=1.0,
            )
            self.particles.append(p)

    def _effect_impact(self, x, y, scale, color):
        """冲击效果：环形扩散的撞击波"""
        num_rings = 2
        for ring in range(num_rings):
            count = int((30 + ring * 20) * scale)
            base_speed = (4 + ring * 4) * scale
            for _ in range(count):
                angle = random.uniform(0, 2 * math.pi)
                speed = base_speed + random.uniform(-1, 1)
                p = Particle(
                    x=x, y=y,
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    life=random.uniform(0.2, 0.45), max_life=0.45,
                    size=random.uniform(2, 4) * scale,
                    color=color if ring == 0 else (
                        min(255, color[0] + 40),
                        min(255, color[1] + 40),
                        min(255, color[2] + 40),
                    ),
                )
                self.particles.append(p)

    def _effect_pierce(self, x, y, direction, scale, color):
        """穿刺效果：沿攻击方向的细长粒子束"""
        length = 60 * scale
        num_particles = int(30 * scale)

        for i in range(num_particles):
            offset = random.gauss(0, 6 * scale)
            perp_x = -math.sin(direction) * offset
            perp_y = math.cos(direction) * offset

            dist_along = random.uniform(-length * 0.3, length * 0.7)
            px = x + math.cos(direction) * dist_along + perp_x
            py = y + math.sin(direction) * dist_along + perp_y

            p = Particle(
                x=px, y=py,
                vx=math.cos(direction) * random.uniform(0.5, 3),
                vy=math.sin(direction) * random.uniform(0.5, 3),
                life=random.uniform(0.12, 0.28), max_life=0.28,
                size=random.uniform(1, 3) * scale,
                color=color,
            )
            self.particles.append(p)

    def _effect_blunt(self, x, y, scale, color):
        """钝击效果：短促、粗大、快速消失的粒子"""
        count = int(25 * scale)
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5) * scale
            p = Particle(
                x=x + random.uniform(-8, 8), y=y + random.uniform(-8, 8),
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                life=random.uniform(0.08, 0.18), max_life=0.18,
                size=random.uniform(4, 10) * scale,
                color=(
                    min(255, color[0] + random.randint(-20, 20)),
                    min(255, color[1] + random.randint(-20, 20)),
                    min(255, color[2] + random.randint(-20, 20)),
                ),
            )
            self.particles.append(p)

    def _effect_crit(self, x, y, scale):
        """暴击效果：金色放射状 + 星芒闪烁"""
        # 金色放射
        gold_colors = [(255, 215, 0), (255, 240, 100), (255, 255, 200)]
        num_arms = 12
        for arm in range(num_arms):
            base_angle = (2 * math.pi * arm / num_arms)
            for _ in range(8):
                spread = random.gauss(0, 0.05)
                angle = base_angle + spread
                speed = random.uniform(5, 14) * scale
                p = Particle(
                    x=x, y=y,
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    life=random.uniform(0.3, 0.7), max_life=0.7,
                    size=random.uniform(2, 5) * scale,
                    color=random.choice(gold_colors),
                )
                self.particles.append(p)

        # 中心闪光
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            p = Particle(
                x=x, y=y,
                vx=math.cos(angle) * random.uniform(0.3, 2),
                vy=math.sin(angle) * random.uniform(0.3, 2),
                life=random.uniform(0.1, 0.25), max_life=0.25,
                size=random.uniform(4, 9) * scale,
                color=(255, 255, 250),
            )
            self.particles.append(p)

    # ================================================================
    #  飘浮伤害数字
    # ================================================================

    def floating_damage(self, x: float, y: float,
                        value: int,
                        is_crit: bool = False,
                        is_heal: bool = False,
                        color: tuple = None):
        """
        显示飘浮伤害/治疗数字

        Args:
            x, y: 起始位置
            value: 数值
            is_crit: 是否暴击（更大字号、不同颜色）
            is_heal: 是否治疗量（绿色系）
            color: 自定义颜色
        """
        if color is None:
            if is_heal:
                color = (50, 255, 100)
            elif is_crit:
                color = (255, 215, 0)
            else:
                color = (255, 80, 80)

        text_data = {
            'text': str(value),
            'x': x,
            'y': y,
            'vy': -60 if is_crit else -40,  # 向上速度
            'life': 1.2 if is_crit else 0.9,
            'max_life': 1.2 if is_crit else 0.9,
            'color': color,
            'is_crit': is_crit,
            'is_heal': is_heal,
            'scale': 1.8 if is_crit else 1.0,
            'particles': [],  # 数字周围的装饰粒子
        }

        # 添加装饰粒子
        deco_color = color
        deco_count = 8 if is_crit else 4
        for _ in range(deco_count):
            p = Particle(
                x=x + random.uniform(-15, 15),
                y=y + random.uniform(-10, 10),
                vx=random.uniform(-15, 15),
                vy=random.uniform(-25, -10),
                life=random.uniform(0.3, 0.6), max_life=0.6,
                size=random.uniform(1, 2.5),
                color=deco_color,
            )
            text_data['particles'].append(p)

        self.floating_texts.append(text_data)

    # ================================================================
    #  技能效果
    # ================================================================

    def skill_cast(self, x: float, y: float,
                   skill_type: str = 'fireball',
                   radius: float = 60,
                   duration: float = 0.8):
        """
        技能施法效果

        Args:
            x, y: 施法中心
            skill_type: 技能类型 ('fireball', 'ice_storm', 'lightning',
                        'heal', 'shield', 'aoe_circle')
            radius: 效果半径
            duration: 持续时间
        """
        skill_type = skill_type.lower()

        if skill_type == 'fireball':
            self._skill_fireball(x, y, radius)
        elif skill_type == 'ice_storm':
            self._skill_ice_storm(x, y, radius)
        elif skill_type == 'lightning':
            self._skill_lightning(x, y, radius)
        elif skill_type == 'heal':
            self._skill_heal(x, y, radius)
        elif skill_type == 'shield':
            self._skill_shield(x, y, radius)
        elif skill_type == 'aoe_circle':
            self._skill_aoe_circle(x, y, radius, duration)
        else:
            self._skill_fireball(x, y, radius)

    def _skill_fireball(self, x, y, radius):
        """火球术：爆炸+火焰残留"""
        # 核心白热
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            p = Particle(
                x=x, y=y,
                vx=math.cos(angle) * random.uniform(1, 4),
                vy=math.sin(angle) * random.uniform(1, 4),
                life=random.uniform(0.1, 0.25), max_life=0.25,
                size=random.uniform(4, 9),
                color=(255, 255, 220),
            )
            self.particles.append(p)

        # 火焰外扩
        fire_colors = [(255, 200, 50), (255, 120, 20), (255, 60, 0)]
        for _ in range(80):
            angle = random.uniform(0, 2 * math.pi)
            dist_factor = random.uniform(0.3, 1.0)
            p = Particle(
                x=x, y=y,
                vx=math.cos(angle) * random.uniform(3, 10) * dist_factor,
                vy=math.sin(angle) * random.uniform(3, 10) * dist_factor,
                life=random.uniform(0.3, 0.8), max_life=0.8,
                size=random.uniform(2, 5),
                color=random.choice(fire_colors),
            )
            self.particles.append(p)

    def _skill_ice_storm(self, x, y, radius):
        """冰风暴：冰晶下落+地面扩散"""
        ice_colors = [(180, 220, 255), (150, 200, 255), (220, 240, 255), (100, 180, 255)]

        # 从上方落下
        for _ in range(60):
            px = x + random.uniform(-radius, radius)
            p = Particle(
                x=px, y=y - radius - random.uniform(0, 50),
                vx=random.uniform(-2, 2),
                vy=random.uniform(3, 8),
                life=random.uniform(0.5, 1.2), max_life=1.2,
                size=random.uniform(2, 5),
                color=random.choice(ice_colors),
            )
            self.particles.append(p)

        # 地面扩散冰晶
        for _ in range(40):
            angle = random.uniform(0, 2 * math.pi)
            p = Particle(
                x=x, y=y + random.uniform(-5, 5),
                vx=math.cos(angle) * random.uniform(1, 4),
                vy=math.sin(angle) * random.uniform(0, 2),
                life=random.uniform(0.4, 1.0), max_life=1.0,
                size=random.uniform(2, 4),
                color=random.choice(ice_colors[:2]),
            )
            self.particles.append(p)

    def _skill_lightning(self, x, y, radius):
        """闪电：随机折线粒子束"""
        lightning_colors = [(200, 200, 255), (150, 170, 255), (255, 255, 255)]

        num_bolts = random.choice([3, 4, 5])
        for _ in range(num_bolts):
            # 起点
            cx, cy = x + random.uniform(-radius * 0.3, radius * 0.3), \
                     y - radius - random.uniform(0, 30)

            # 分段生成闪电路径
            segments = random.randint(5, 10)
            for seg in range(segments):
                nx = cx + random.uniform(-radius * 0.25, radius * 0.25)
                ny = cy + radius / segments + random.uniform(-5, 15)

                # 在线段上放置粒子
                steps = 3
                for s in range(steps):
                    t = s / steps
                    p = Particle(
                        x=cx + (nx - cx) * t,
                        y=cy + (ny - cy) * t,
                        vx=random.uniform(-1, 1),
                        vy=random.uniform(-0.5, 1),
                        life=random.uniform(0.06, 0.15), max_life=0.15,
                        size=random.uniform(1, 3),
                        color=random.choice(lightning_colors),
                    )
                    self.particles.append(p)

                cx, cy = nx, ny

    def _skill_heal(self, x, y, radius):
        """治疗：上升的光点 + 绿色圆环"""
        heal_colors = [(100, 255, 150), (150, 255, 180), (200, 255, 210), (255, 255, 230)]

        # 上升光点
        for _ in range(40):
            px = x + random.uniform(-radius * 0.5, radius * 0.5)
            py = y + random.uniform(-radius * 0.3, radius * 0.3)
            p = Particle(
                x=px, y=py,
                vx=random.uniform(-1, 1),
                vy=random.uniform(-4, -1.5),
                life=random.uniform(0.6, 1.5), max_life=1.5,
                size=random.uniform(2, 5),
                color=random.choice(heal_colors),
            )
            self.particles.append(p)

        # 扩散圆环
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            p = Particle(
                x=x, y=y,
                vx=math.cos(angle) * random.uniform(1, 4),
                vy=math.sin(angle) * random.uniform(1, 4),
                life=random.uniform(0.3, 0.7), max_life=0.7,
                size=random.uniform(1.5, 3.5),
                color=heal_colors[0],
            )
            self.particles.append(p)

    def _skill_shield(self, x, y, radius):
        """护盾：旋转的光环粒子"""
        shield_colors = [(80, 140, 255), (120, 170, 255), (160, 200, 255)]

        num_rings = 2
        for ring in range(num_rings):
            r = radius * (0.85 + ring * 0.2)
            count = int(24 + ring * 12)
            for i in range(count):
                angle = 2 * math.pi * i / count
                tangent_angle = angle + math.pi / 2
                px = x + r * math.cos(angle)
                py = y + r * math.sin(angle)

                p = Particle(
                    x=px, y=py,
                    vx=math.cos(tangent_angle) * random.uniform(1, 3),
                    vy=math.sin(tangent_angle) * random.uniform(1, 3),
                    life=random.uniform(0.8, 2.0), max_life=2.0,
                    size=random.uniform(2, 4),
                    color=shield_colors[ring % len(shield_colors)],
                )
                self.particles.append(p)

        # 吸引场保持形状
        self.particles[-count * num_rings:]  # 不做特殊处理，靠 update 自然运动

    def _skill_aoe_circle(self, x, y, radius, duration):
        """AOE 圈：从中心扩散的圆环波纹"""
        aoe_colors = [(255, 100, 100), (255, 150, 100), (255, 200, 100)]

        num_waves = 3
        for wave in range(num_waves):
            delay = wave * (duration / num_waves)
            count = int(36 + wave * 12)
            base_speed = radius / duration * (0.8 + wave * 0.2)

            for i in range(count):
                angle = 2 * math.pi * i / count
                p = Particle(
                    x=x, y=y,
                    vx=math.cos(angle) * base_speed * random.uniform(0.9, 1.1),
                    vy=math.sin(angle) * base_speed * random.uniform(0.9, 1.1),
                    life=duration - delay, max_life=duration - delay,
                    size=random.uniform(2, 4),
                    color=aoe_colors[min(wave, len(aoe_colors) - 1)],
                )
                self.particles.append(p)

    # ================================================================
    #  其他特效
    # ================================================================

    def death_fadeout(self, x: float, y: float,
                      width: float = 40,
                      height: float = 60,
                      color: tuple = (100, 80, 60)):
        """
        死亡消散效果：角色位置粒子化并向上飘散消失

        Args:
            x, y: 角色中心位置
            width, height: 角色大致尺寸
            color: 主色调
        """
        count = int(width * height / 15)
        for _ in range(count):
            px = x + random.uniform(-width / 2, width / 2)
            py = y + random.uniform(-height / 2, height / 2)
            p = Particle(
                x=px, y=py,
                vx=random.uniform(-1, 1),
                vy=random.uniform(-3, -0.5),  # 主要向上
                life=random.uniform(0.8, 2.0), max_life=2.0,
                size=random.uniform(2, 5),
                color=(
                    color[0] + random.randint(-20, 20),
                    color[1] + random.randint(-20, 20),
                    color[2] + random.randint(-20, 20),
                ),
                alpha=0.8,
            )
            self.particles.append(p)

    def level_up(self, x: float, y: float):
        """
        升级效果：金色螺旋上升 + 光柱
        """
        gold = [(255, 215, 0), (255, 240, 100), (255, 255, 200)]

        # 螺旋上升粒子
        for i in range(60):
            angle = i * 0.4 + random.uniform(-0.2, 0.2)
            r = i * 0.8
            px = x + r * math.cos(angle)
            py = y - i * 1.5 + random.uniform(-5, 5)
            p = Particle(
                x=px, y=py,
                vx=math.cos(angle + math.pi / 2) * 1.5,
                vy=-random.uniform(2, 5),
                life=random.uniform(0.5, 1.5), max_life=1.5,
                size=random.uniform(2, 5),
                color=random.choice(gold),
            )
            self.particles.append(p)

        # 底部光柱
        for _ in range(20):
            px = x + random.uniform(-15, 15)
            p = Particle(
                x=px, y=y + 20,
                vx=random.uniform(-0.5, 0.5),
                vy=-random.uniform(8, 15),
                life=random.uniform(0.3, 0.7), max_life=0.7,
                size=random.uniform(3, 7),
                color=(255, 255, 240),
            )
            self.particles.append(p)

    # ================================================================
    #  更新 & 渲染
    # ================================================================

    def update(self, dt: float):
        """
        更新所有战斗特效

        Args:
            dt: 时间步长
        """
        # 更新主粒子列表
        alive = []
        for p in self.particles:
            if not p.is_alive():
                continue
            fx, fy = self.gravity.get_force(p)
            p.vx += fx * dt * 0.3
            p.vy += fy * dt * 0.3
            p.vx *= 0.96  # 空气阻力
            p.vy *= 0.96
            p.update(dt)
            if p.is_alive():
                alive.append(p)
        self.particles = alive

        # 更新飘浮文字
        active_texts = []
        for ft in self.floating_texts:
            ft['y'] += ft['vy'] * dt
            ft['life'] -= dt

            # 更新装饰粒子
            ft_alive = []
            for p in ft.get('particles', []):
                if p.is_alive():
                    fx, fy = self.gravity.get_force(p)
                    p.vx += fx * dt
                    p.vy += fy * dt
                    p.update(dt)
                    if p.is_alive():
                        ft_alive.append(p)
            ft['particles'] = ft_alive

            if ft['life'] > 0:
                active_texts.append(ft)
        self.floating_texts = active_texts

    def render(self, renderer):
        """
        渲染所有特效

        Args:
            renderer: 渲染器实例
        """
        if hasattr(renderer, 'render'):
            renderer.render(self.particles)

    @property
    def floating_text_data(self) -> list:
        """
        返回当前活跃的飘浮文字数据列表，
        用于外部渲染引擎自行绘制文字。

        每项包含：text, x, y, color, alpha, scale, is_crit
        """
        result = []
        for ft in self.floating_texts:
            alpha = max(0, ft['life'] / ft['max_life'])
            result.append({
                'text': ft['text'],
                'x': ft['x'],
                'y': ft['y'],
                'color': ft['color'],
                'alpha': alpha,
                'scale': ft['scale'],
                'is_crit': ft['is_crit'],
            })
        return result


def create_combat_manager() -> CombatEffectsManager:
    """创建并返回一个 CombatEffectsManager 实例"""
    return CombatEffectsManager()

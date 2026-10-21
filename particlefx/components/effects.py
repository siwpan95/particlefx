"""
粒子预设效果库 (v3.0 — 全功能版)

提供开箱即用的粒子效果，每个效果是一个工厂函数，
返回配置好的 ParticleSystem + Emitter，直接调用 update() 和 render() 即可。

=== v3.0 新增特性 ===
- 发射形状（Shape Module）：sphere/cone/circle/rectangle/edge
- 拖尾（Trails）：位置历史 + 渲染连线
- 碰撞（Collision）：AABB 检测 + 反弹
- 力场扩展：vortex(漩涡)/turbulence(湍流)/magnet(磁场)
- 事件系统：on_birth/death/collision/tick 完整回调链
- 光照模块：Blinn-Phong 着色（Ambient+Diffuse+Specular+Fresnel）
- LOD 系统：距离相机越远粒子越少
- 音频联动：音频驱动发射率
- 网格体粒子：绑定 Surface/Image
- 纹理动画：精灵表帧动画

使用示例：
    from .effects import create_explosion, create_vortex_effect

    boom = create_explosion(x=400, y=300)
    tornado = create_vortex_effect(x=400, y=300)
    while running:
        boom.update(dt=0.016)
        tornado.update(dt=0.016)
        boom.render(renderer)
        tornado.render(renderer)
"""

import random
import math

from ..core.particle import (
    Particle, ParticleSystem, Emitter, ForceField,
    FIRE_COLOR_CURVE, SMOKE_COLOR_CURVE, MAGIC_COLOR_CURVE,
    EXPLOSION_SIZE_CURVE, SPARK_SIZE_CURVE,
    gradient_to_curve, size_curve_fadeout, size_curve_pulse,
    velocity_curve_accelerate, velocity_curve_decelerate,
    NoiseField,
    # v3.0 新增导入
    EmitterShape, CollisionBox, CollisionBoundary,
    LightSource, BlinnPhongParams,
    create_point_light, create_directional_light, create_ambient_light,
    create_vortex, create_turbulence, create_magnet,
)


# ============================================================
#  辅助函数
# ============================================================

def _random_color(palette: list) -> tuple:
    """从调色板中随机选一个颜色"""
    return random.choice(palette)


# ============================================================
#  预设效果
# ============================================================

def create_explosion(x: float, y: float,
                     intensity: float = 1.0,
                     color_palette: list = None) -> ParticleSystem:
    """
    爆炸效果（增强版）

    从中心点向四周喷射大量粒子，带闪光和残骸下落。
    使用 Color over Lifetime 和 Size over Lifetime 实现真实爆炸效果。

    Args:
        x, y: 爆炸中心坐标
        intensity: 强度倍数（1.0 为标准，2.0 为大型爆炸）
        color_palette: 颜色列表，默认为火焰色调

    Returns:
        配置好的 ParticleSystem
    """
    if color_palette is None:
        color_palette = [
            (255, 200, 50),   # 亮黄
            (255, 120, 20),   # 橙色
            (255, 60, 10),    # 红橙
            (255, 255, 200),  # 闪光白
            (100, 100, 100),  # 烟灰
        ]

    system = ParticleSystem(max_particles=int(300 * intensity))

    # 主爆：高速向外喷射（增强：颜色+大小生命曲线）
    for _ in range(int(5 * intensity)):
        angle_offset = random.uniform(0, 2 * math.pi)
        emitter = Emitter(
            x=x, y=y,
            rate=80 * intensity,
            spread=math.pi * 2,
            speed=random.uniform(8, 20) * intensity,
            life=random.uniform(0.2, 0.6),
            size=random.uniform(2, 6) * intensity,
            color=_random_color(color_palette),
            color_over_lifetime=FIRE_COLOR_CURVE,
            size_over_lifetime=EXPLOSION_SIZE_CURVE,
        )
        system.add_emitter(emitter)

    # 残骸：慢速大粒子下落（增强：烟雾颜色曲线）
    for _ in range(int(3 * intensity)):
        emitter = Emitter(
            x=x, y=y,
            rate=30 * intensity,
            spread=math.pi * 0.8,
            speed=random.uniform(1, 5),
            life=random.uniform(0.5, 1.5),
            size=random.uniform(3, 8),
            color=_random_color([(80, 60, 40), (120, 90, 60), (60, 50, 40)]),
            color_over_lifetime=SMOKE_COLOR_CURVE,
        )
        system.add_emitter(emitter)

    # 重力场：让残骸自然下落
    system.add_force_field(ForceField(field_type='gravity', strength=5.0))

    return system


def create_fire(x: float, y: float,
                width: float = 40,
                height: float = 80,
                intensity: float = 1.0) -> ParticleSystem:
    """
    火焰效果（增强版）

    从底部向上升腾的火焰粒子，带摇曳感。
    使用 Color over Lifetime + NoiseField 实现真实火焰效果。

    Args:
        x, y: 火焰底部中心位置
        width: 火焰宽度
        height: 火焰高度
        intensity: 强度倍数

    Returns:
        配置好的 ParticleSystem
    """
    system = ParticleSystem(max_particles=int(400 * intensity))

    fire_colors = [
        (255, 255, 150),  # 内核白黄
        (255, 200, 50),   # 亮黄
        (255, 120, 20),   # 橙色
        (255, 60, 0),     # 红橙
        (180, 20, 0),     # 暗红（外焰）
    ]

    # 多层火焰发射器（从底到顶）
    num_layers = 5
    for i in range(num_layers):
        layer_y = y - (i / num_layers) * height * 0.6
        layer_spread = math.pi * (0.3 + i * 0.15)
        layer_speed = (3 + i * 2) * intensity
        layer_rate = (40 - i * 5) * intensity
        layer_size = (4 - i * 0.5) * intensity

        emitter = Emitter(
            x=x + random.uniform(-width / 4, width / 4),
            y=layer_y,
            rate=layer_rate,
            spread=layer_spread,
            speed=layer_speed,
            life=random.uniform(0.3, 0.9 - i * 0.08),
            size=layer_size,
            color=fire_colors[min(i, len(fire_colors) - 1)],
            color_over_lifetime=FIRE_COLOR_CURVE,
            size_over_lifetime=size_curve_pulse(layer_size, peak_at=0.2, peak_mult=1.3),
        )
        system.add_emitter(emitter)

    # 轻微向上推的力场（模拟热气流）
    system.add_force_field(ForceField(field_type='gravity', strength=-2.0))

    # 噪声场：让火焰有摇曳感（核心增强！）
    system.add_noise_field(NoiseField(
        strength=0.8,
        scale=30.0,
        octaves=3,
        seed=random.randint(0, 100000)
    ))

    return system


def create_rain(width: int, height: int,
                density: float = 1.0,
                wind: float = 0.0,
                color: tuple = (150, 180, 255)) -> ParticleSystem:
    """
    雨效果（增强版）

    屏幕顶部落下雨滴粒子。雨滴带旋转，更真实。

    Args:
        width: 场景宽度
        height: 场景高度
        density: 密度倍数
        wind: 风力强度（负=左，正=右）
        color: 雨滴颜色

    Returns:
        配置好的 ParticleSystem
    """
    system = ParticleSystem(max_particles=int(500 * density))

    # 在整个宽度上均匀分布多个发射点
    num_sources = max(5, int(width / 80))
    for i in range(num_sources):
        emitter = Emitter(
            x=(i + 0.5) * width / num_sources + random.uniform(-20, 20),
            y=0,
            rate=30 * density,
            spread=0.1,
            speed=random.uniform(15, 25),
            life=height / 18,
            size=random.uniform(1, 2),
            color=color,
            rotation_range=(0.0, 2 * math.pi),
            angular_velocity_range=(-2.0, 2.0),
        )
        system.add_emitter(emitter)

    system.add_force_field(ForceField(field_type='gravity', strength=20.0))
    if wind != 0:
        system.add_force_field(ForceField(field_type='wind', strength=wind))

    return system


def create_snow(width: int, height: int,
                density: float = 0.5,
                wind: float = 0.0) -> ParticleSystem:
    """
    雪效果（增强版）

    缓缓飘落的雪花，大小不一，有左右飘动感和旋转。

    Args:
        width: 场景宽度
        height: 场景高度
        density: 密度倍数
        wind: 基础风力

    Returns:
        配置好的 ParticleSystem
    """
    system = ParticleSystem(max_particles=int(300 * density))

    snow_colors = [(255, 255, 255), (230, 240, 255), (200, 220, 255)]

    num_sources = max(5, int(width / 100))
    for i in range(num_sources):
        emitter = Emitter(
            x=(i + 0.5) * width / num_sources + random.uniform(-30, 30),
            y=0,
            rate=8 * density,
            spread=math.pi * 0.3,
            speed=random.uniform(0.8, 2.5),
            life=height / 2.5,
            size=random.uniform(1.5, 4.5),
            color=_random_color(snow_colors),
            rotation_range=(0.0, 2 * math.pi),
            angular_velocity_range=(-0.8, 0.8),
        )
        system.add_emitter(emitter)

    # 轻微重力
    system.add_force_field(ForceField(field_type='gravity', strength=1.5))
    # 左右摇摆的风力模拟
    if abs(wind) > 0.01:
        system.add_force_field(ForceField(field_type='wind', strength=wind))

    return system


def create_magic_circle(x: float, y: float,
                        radius: float = 80,
                        rotation_speed: float = 1.0,
                        color: tuple = (100, 150, 255)) -> ParticleSystem:
    """
    魔法光环效果（增强版）

    粒子沿圆形轨道旋转，形成魔法阵/光环视觉效果。
    使用 Magic Color Curve 实现颜色渐变。

    Args:
        x, y: 光环圆心坐标
        radius: 光环半径
        rotation_speed: 旋转速度
        color: 主颜色

    Returns:
        配置好的 ParticleSystem
    """
    system = ParticleSystem(max_particles=500)

    inner_color = (200, 220, 255)
    outer_color = color

    # 多层光环
    for ring in range(3):
        r = radius * (0.6 + ring * 0.25)
        particles_per_ring = int(12 + ring * 8)

        for _ in range(particles_per_ring):
            angle = random.uniform(0, 2 * math.pi)
            px = x + r * math.cos(angle)
            py = y + r * math.sin(angle)

            # 切线速度（形成旋转效果）
            tangent_angle = angle + math.pi / 2
            tv = rotation_speed * (2 + ring)

            emitter = Emitter(
                x=px,
                y=py,
                rate=3,
                spread=math.pi * 0.3,
                speed=tv,
                life=random.uniform(0.5, 1.5),
                size=random.uniform(1.5, 3.5),
                color=_lerp_color(inner_color, outer_color, ring / 2),
                color_over_lifetime=MAGIC_COLOR_CURVE,
                rotation_range=(0.0, 2 * math.pi),
                angular_velocity_range=(-0.5, 0.5),
            )
            system.add_emitter(emitter)

    # 中心吸引场（保持粒子不散开太远）
    system.add_force_field(ForceField(
        field_type='attraction',
        x=x,
        y=y,
        strength=0.8,
    ))

    return system


def create_fountain(x: float, y: float,
                    height: float = 120,
                    spread: float = 0.4) -> ParticleSystem:
    """
    喷泉效果（增强版）

    从底部喷出后呈抛物线下落的水花效果。
    使用水色曲线实现更真实的效果。

    Args:
        x, y: 喷口位置
        height: 最大喷射高度
        spread: 水花扩散角度（弧度）

    Returns:
        配置好的 ParticleSystem
    """
    system = ParticleSystem(max_particles=600)

    water_colors = [
        (150, 200, 255),  # 浅蓝
        (100, 170, 255),  # 中蓝
        (200, 230, 255),  # 高光白蓝
    ]

    # 主水流
    main = Emitter(
        x=x,
        y=y,
        rate=60,
        spread=spread,
        speed=height / 10,
        life=2.0,
        size=2.5,
        color=water_colors[0],
        color_over_lifetime=gradient_to_curve(water_colors),
    )
    system.add_emitter(main)

    # 两侧溅射水花
    for side in [-1, 1]:
        splash = Emitter(
            x=x + side * 10,
            y=y - height * 0.35,
            rate=25,
            spread=math.pi * (0.4 + random.uniform(0, 0.3)),
            speed=random.uniform(2, 6),
            life=random.uniform(0.5, 1.2),
            size=random.uniform(1, 3),
            color=_random_color(water_colors),
        )
        system.add_emitter(splash)

    # 底部回落水花
    landing = Emitter(
        x=x,
        y=y - 5,
        rate=20,
        spread=math.pi * 0.9,
        speed=random.uniform(0.5, 2),
        life=random.uniform(0.3, 0.7),
        size=random.uniform(1, 2),
        color=water_colors[2],
    )
    system.add_emitter(landing)

    # 重力
    system.add_force_field(ForceField(field_type='gravity', strength=8.0))

    return system


def create_smoke(x: float, y: float,
                 direction: str = 'up',
                 density: float = 1.0) -> ParticleSystem:
    """
    烟雾效果（增强版）

    缓慢扩散、逐渐消散的烟雾粒子。
    使用 Smoke Color Curve + NoiseField 实现真实烟雾效果。

    Args:
        x, y: 烟源位置
        direction: 烟雾方向 ('up', 'left', 'right', 'down')
        density: 密度倍数

    Returns:
        配置好的 ParticleSystem
    """
    system = ParticleSystem(max_particles=int(250 * density))

    smoke_colors = [
        (80, 80, 85),    # 深灰
        (110, 110, 115),  # 中灰
        (140, 140, 145),  # 浅灰
        (160, 155, 150),  # 微褐色
    ]

    direction_map = {
        'up': (-math.pi / 2, 0.0, -1.5),
        'down': (math.pi / 2, 0.0, 1.5),
        'left': (math.pi, -0.8, 0.0),
        'right': (0.0, 0.8, 0.0),
    }
    base_angle, wind_x, gravity_y = direction_map.get(direction, direction_map['up'])

    # 烟源发射（增强：颜色曲线 + 旋转）
    for _ in range(3):
        emitter = Emitter(
            x=x + random.uniform(-5, 5),
            y=y + random.uniform(-5, 5),
            rate=15 * density,
            spread=math.pi * 0.4,
            speed=random.uniform(0.5, 2),
            life=random.uniform(1.5, 3.5),
            size=random.uniform(4, 12),
            color=_random_color(smoke_colors),
            color_over_lifetime=SMOKE_COLOR_CURVE,
            size_over_lifetime=size_curve_fadeout(random.uniform(4, 12)),
            rotation_range=(0.0, 2 * math.pi),
            angular_velocity_range=(-1.0, 1.0),
        )
        system.add_emitter(emitter)

    # 噪声场：让烟雾飘散更自然（核心增强！）
    system.add_noise_field(NoiseField(
        strength=0.6,
        scale=60.0,
        octaves=2,
    ))

    # 轻微力场
    system.add_force_field(ForceField(field_type='gravity', strength=gravity_y))
    if abs(wind_x) > 0.01:
        system.add_force_field(ForceField(field_type='wind', strength=wind_x))

    return system


def create_sparkle(x: float, y: float,
                   area_size: float = 60,
                   intensity: float = 1.0) -> ParticleSystem:
    """
    星光闪烁效果（增强版）

    区域内随机闪现的小星星/光点，适合 UI 强调、拾取物品等场景。
    使用 Spark Size Curve 实现大小脉冲效果。

    Args:
        x, y: 效果区域中心
        area_size: 效果区域大小
        intensity: 强度倍数

    Returns:
        配置好的 ParticleSystem
    """
    system = ParticleSystem(max_particles=int(100 * intensity))

    sparkle_colors = [
        (255, 255, 220),  # 暖白
        (255, 240, 180),  # 金色
        (220, 230, 255),  # 冷白
        (255, 200, 220),  # 粉白
    ]

    # 在区域内分布多个微小发射器
    num_points = int(8 * intensity)
    for _ in range(num_points):
        px = x + random.uniform(-area_size / 2, area_size / 2)
        py = y + random.uniform(-area_size / 2, area_size / 2)

        emitter = Emitter(
            x=px,
            y=py,
            rate=random.uniform(2, 8),
            spread=math.pi * 2,
            speed=random.uniform(0.2, 1.0),
            life=random.uniform(0.2, 0.6),
            size=random.uniform(1, 3),
            color=_random_color(sparkle_colors),
            size_over_lifetime=SPARK_SIZE_CURVE,
        )
        system.add_emitter(emitter)

    return system


def create_trail(x: float, y: float,
                 direction_x: float = 1.0,
                 direction_y: float = 0.0,
                 color: tuple = (255, 200, 100),
                 length: float = 1.0) -> ParticleSystem:
    """
    尾迹效果（增强版）

    移动物体留下的拖尾粒子，适合子弹、角色移动等场景。
    使用 Velocity over Lifetime 实现速度变化效果。

    Args:
        x, y: 初始尾迹起点
        direction_x, direction_y: 运动方向（用于设置初始速度方向）
        color: 尾迹颜色
        length: 尾迹长度倍数

    Returns:
        配置好的 ParticleSystem（含一个 Emitter，可用 set_position 追踪移动目标）
    """
    system = ParticleSystem(max_particles=int(200 * length))

    # 计算运动方向的角度
    base_angle = math.atan2(direction_y, direction_x)
    # 反向发射（向后喷射尾迹粒子）
    emit_angle = base_angle + math.pi

    emitter = Emitter(
        x=x,
        y=y,
        rate=40 * length,
        direction=emit_angle,
        spread=math.pi * 0.4,
        speed=random.uniform(0.5, 2),
        life=random.uniform(0.2, 0.6) * length,
        size=random.uniform(1, 4),
        color=color,
        velocity_over_lifetime=velocity_curve_decelerate(0.3),
    )
    system.add_emitter(emitter)

    # 轻微阻力让粒子快速减速消失
    system.add_force_field(ForceField(field_type='wind',
                                      strength=-direction_x * 0.5))
    system.add_force_field(ForceField(field_type='gravity',
                                      strength=-direction_y * 0.5))

    return system


# ============================================================
#  v3.0 新增效果（展示新特性）
# ============================================================

def create_vortex_effect(x: float, y: float,
                         radius: float = 80,
                         strength: float = 4.0,
                         particle_count: int = 200) -> ParticleSystem:
    """
    漩涡/龙卷风效果（展示 Vortex 力场 + Sphere 发射形状 + 拖尾）
    
    粒子从中心被漩涡力场卷起，形成螺旋上升的效果。
    带拖尾让轨迹可见。
    """
    system = ParticleSystem(max_particles=particle_count)
    
    # 从球形区域发射
    emitter = Emitter(
        x=x, y=y,
        rate=60,
        spread=math.pi * 2,
        speed=random.uniform(1, 3),
        life=random.uniform(1.5, 3.5),
        size=random.uniform(2, 5),
        color=_random_color([(150, 180, 220), (180, 200, 230), (200, 220, 240)]),
        shape=EmitterShape('sphere', radius=radius, emit_from_volume=True),
        trail_enabled=True,
        trail_length=12,
        rotation_range=(0, math.pi * 2),
        angular_velocity_range=(-3, 3),
        size_over_lifetime=size_curve_fadeout(4),
    )
    system.add_emitter(emitter)
    
    # 漩涡力场
    system.add_force_field(create_vortex(x=x, y=y - radius * 0.3,
                                         strength=strength, clockwise=True))
    
    # 轻微向上推 + 湍流扰动
    system.add_force_field(ForceField(field_type='gravity', strength=-1.5))
    system.add_force_field(create_turbulence(strength=1.5, scale=50))
    
    return system


def create_collision_sparks(x: float, y: float,
                            width: float = 200,
                            height: float = 150,
                            intensity: float = 1.0) -> ParticleSystem:
    """
    碰撞火花效果（展示 Collision 边界反弹 + 矩形发射形状）
    
    火花在边界内弹跳，模拟金属碰撞的火花四溅。
    """
    system = ParticleSystem(max_particles=int(300 * intensity))
    
    # 设置边界反弹
    system.set_boundary(CollisionBoundary(
        left=x - width/2, top=y - height/2,
        right=x + width/2, bottom=y + height/2,
        mode='bounce', restitution=0.7, friction=0.15,
    ))
    
    # 从矩形区域内发射火花
    for _ in range(int(4 * intensity)):
        emitter = Emitter(
            x=x, y=y,
            rate=40 * intensity,
            spread=math.pi * 2,
            speed=random.uniform(8, 18) * intensity,
            life=random.uniform(0.3, 1.2),
            size=random.uniform(1, 3),
            color=_random_color([(255, 220, 100), (255, 180, 50), (255, 140, 30)]),
            shape=EmitterShape('rectangle', width=width * 0.6, rect_height=height * 0.6),
            size_over_lifetime=SPARK_SIZE_CURVE,
            velocity_over_lifetime=velocity_curve_decelerate(0.2),
            trail_enabled=True,
            trail_length=5,
        )
        system.add_emitter(emitter)
    
    # 重力
    system.add_force_field(ForceField(field_type='gravity', strength=3.0))
    
    return system


def create_lighted_firework(x: float, y: float,
                            intensity: float = 1.0) -> ParticleSystem:
    """
    光照烟花效果（展示 Blinn-Phong 光照 + 自发光粒子）
    
    烟花粒子带有自发光属性，在光照系统下呈现明亮的发光效果。
    需要渲染器支持光照模式（system.render(renderer) 自动处理）。
    """
    system = ParticleSystem(max_particles=int(500 * intensity))
    
    # 配置光照
    system.add_light(create_ambient_light(color=(30, 25, 45), intensity=0.2))
    system.add_light(create_point_light(x=x, y=y - 100,
                                        color=(255, 240, 200), intensity=1.2, range_=300))
    system.blinn_phong = BlinnPhongParams(
        ambient_strength=0.2, diffuse_strength=0.6,
        specular_strength=0.8, shininess=48.0,
        fresnel_power=4.0, fresnel_strength=0.35,
    )
    system.lighting_enabled = True
    
    # 核心闪光 — 强自发光
    core = Emitter(
        x=x, y=y,
        rate=30 * intensity,
        spread=math.pi * 2,
        speed=random.uniform(0.5, 3) * intensity,
        life=random.uniform(0.15, 0.4),
        size=random.uniform(4, 10) * intensity,
        color=(255, 255, 230),
        shape=EmitterShape('sphere', radius=5, emit_from_shell=True),
        emissive=(255, 255, 200), emissive_strength=2.0,
        size_over_lifetime=[(0.0, 1.0), (0.3, 2.5), (1.0, 0.0)],
    )
    system.add_emitter(core)
    
    # 外圈火花 — 弱自发光
    for _ in range(int(6 * intensity)):
        angle_offset = random.uniform(0, 2 * math.pi)
        emitter = Emitter(
            x=x, y=y,
            rate=50 * intensity,
            spread=math.pi * 2,
            speed=random.uniform(6, 16) * intensity,
            life=random.uniform(0.3, 0.8),
            size=random.uniform(1, 4) * intensity,
            color=_random_color([(255, 200, 50), (255, 120, 20), (255, 80, 10)]),
            emissive=(255, 150, 30), emissive_strength=0.8,
            color_over_lifetime=FIRE_COLOR_CURVE,
            size_over_lifetime=EXPLOSION_SIZE_CURVE,
        )
        system.add_emitter(emitter)
    
    return system


def create_magnet_field(x_center: float, y_center: float,
                        attract_x: float, attract_y: float,
                        repel_x: float, repel_y: float,
                        num_particles: int = 300) -> ParticleSystem:
    """
    磁场效果（展示 Magnet 吸引/排斥力场）
    
    粒子同时受吸引和排斥磁场影响，形成动态流动。
    """
    system = ParticleSystem(max_particles=num_particles)
    
    # 圆形发射源
    emitter = Emitter(
        x=x_center, y=y_center,
        rate=80, spread=math.pi * 2,
        speed=random.uniform(0.5, 2),
        life=random.uniform(2, 5),
        size=random.uniform(2, 4),
        color=(100, 180, 255),
        shape=EmitterShape('circle', radius=30, emit_from_shell=True),
        trail_enabled=True, trail_length=8,
    )
    system.add_emitter(emitter)
    
    # 吸引磁极
    system.add_force_field(create_magnet(attract_x, attract_y,
                                         polarity=1.0, strength=4.0, range_=250))
    # 排斥磁极
    system.add_force_field(create_magnet(repel_x, repel_y,
                                         polarity=-1.0, strength=3.0, range_=200))
    
    # 轻微阻尼
    system.add_force_field(ForceField(field_type='wind', strength=-0.1))
    
    return system


# ============================================================

EFFECT_REGISTRY = {
    'explosion': create_explosion,
    'fire': create_fire,
    'rain': create_rain,
    'snow': create_snow,
    'magic_circle': create_magic_circle,
    'fountain': create_fountain,
    'smoke': create_smoke,
    'sparkle': create_sparkle,
    'trail': create_trail,
    # v3.0 新增
    'vortex': create_vortex_effect,
    'collision_sparks': create_collision_sparks,
    'lighted_firework': create_lighted_firework,
    'magnet_field': create_magnet_field,
}


def get_effect(name: str, **kwargs) -> ParticleSystem:
    """
    按名称获取效果实例

    Args:
        name: 效果名称（见 EFFECT_REGISTRY 的键）
        **kwargs: 传递给对应工厂函数的参数

    Returns:
        ParticleSystem 实例

    Raises:
        ValueError: 效果名称不存在
    """
    name = name.lower().strip()
    if name not in EFFECT_REGISTRY:
        available = ', '.join(EFFECT_REGISTRY.keys())
        raise ValueError(f"未知效果 '{name}'，可选: {available}")
    return EFFECT_REGISTRY[name](**kwargs)


def list_effects() -> list:
    """列出所有可用的效果名称"""
    return list(EFFECT_REGISTRY.keys())


# ============================================================
#  内部辅助（避免循环引用）
# ============================================================

def _lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    """颜色插值（内部使用）"""
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


# ============================================================
#  流体视觉效果（需要 FluidRenderer）
# ============================================================

def create_fluid_effect(x: float = 400, y: float = 300,
                        radius: float = 100,
                        fluid_color: tuple = (50, 180, 255),
                        num_particles: int = 800,
                        gravity: float = 3.0,
                        turbulence: float = 2.0,
                        ) -> 'FluidEffectSystem':
    """
    流体视觉效果（展示 FluidRenderer 渲染器）
    
    粒子用 FluidRenderer 渲染，呈现连续液面效果（Metaball融合）。
    需要配合 FluidRenderer 使用，不能直接用普通渲染器。
    
    Args:
        x, y: 流体中心位置
        radius: 流体影响半径
        fluid_color: 流体颜色 (R, G, B)
        num_particles: 粒子数量（越多越连续，性能开销越大）
        gravity: 重力强度
        turbulence: 湍流强度
    
    Returns:
        FluidEffectSystem 实例
    """
    return FluidEffectSystem(
        x=x, y=y,
        radius=radius,
        fluid_color=fluid_color,
        num_particles=num_particles,
        gravity=gravity,
        turbulence=turbulence,
    )


class FluidEffectSystem:
    """
    流体效果系统（封装 ParticleSystemNumpy + FluidRenderer 接口）。
    
    提供简单的 update() + render_to_surface() 接口，
    隐藏底层 NumPy 系统和渲染器细节。
    """
    
    def __init__(self,
                 x: float = 400,
                 y: float = 300,
                 radius: float = 100,
                 fluid_color: tuple = (50, 180, 255),
                 num_particles: int = 800,
                 gravity: float = 3.0,
                 turbulence: float = 2.0,
                 ):
        from ..core.particle_numpy import (
            ParticleSystemNumpy, NumpyForceGravity,
            NumpyForceTurbulence, NumpyForceDrag
        )
        import numpy as np
        
        self.x = x
        self.y = y
        self.radius = radius
        self.fluid_color = fluid_color
        
        # 创建 NumPy 粒子系统
        self.sys_np = ParticleSystemNumpy(max_particles=num_particles)
        
        # 力场
        self.sys_np.add_force(NumpyForceGravity(strength=gravity))
        self.sys_np.add_force(NumpyForceTurbulence(strength=turbulence, scale=40.0))
        self.sys_np.add_force(NumpyForceDrag(factor=0.98))
        
        # 初始发射
        self.sys_np.emit_radial(
            x=x, y=y,
            count=num_particles,
            speed_range=(0.5, 3.0),
            life_range=(2.0, 5.0),
            size_range=(8, 20),
            colors=[fluid_color],
        )
        
        # 持续发射定时器
        self._emit_timer = 0.0
        self._emit_interval = 0.1
    
    def update(self, dt: float = 0.016):
        """更新粒子系统"""
        import random
        import numpy as np
        
        self.sys_np.update(dt)
        
        # 持续补充死亡粒子
        self._emit_timer += dt
        if self._emit_timer >= self._emit_interval:
            self._emit_timer = 0.0
            dead_count = int(np.sum(~self.sys_np.alive))
            if dead_count > 0:
                self.sys_np.emit_radial(
                    x=self.x + random.uniform(-self.radius * 0.3, self.radius * 0.3),
                    y=self.y + random.uniform(-self.radius * 0.3, self.radius * 0.3),
                    count=min(dead_count, 50),
                    speed_range=(0.5, 2.0),
                    life_range=(2.0, 4.0),
                    size_range=(8, 18),
                    colors=[self.fluid_color],
                )
    
    def render_to_surface(self,
                          fluid_renderer,
                          pygame_screen=None,
                          ):
        """渲染到 Pygame Surface"""
        import pygame
        img_array = fluid_renderer.render_from_numpy_system(self.sys_np)
        
        if pygame_screen is not None:
            pygame.surfarray.blit_array(pygame_screen, img_array.swapaxes(0, 1))
            return pygame_screen
        
        surface = pygame.Surface((fluid_renderer.width, fluid_renderer.height))
        pygame.surfarray.blit_array(surface, img_array.swapaxes(0, 1))
        return surface
    
    def render_to_image(self, fluid_renderer, background=None):
        """渲染到 numpy 数组（无 Pygame 依赖）"""
        return fluid_renderer.render_from_numpy_system(self.sys_np, background)
    
    @property
    def particle_count(self):
        return self.sys_np.particle_count
    
    def clear(self):
        self.sys_np.clear()


# 更新注册表
EFFECT_REGISTRY['fluid'] = create_fluid_effect

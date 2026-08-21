"""
烟花系统 v2.0 — 重构版

核心变化（v2.0）：
- 火箭改用 Particle 对象（而不是 dict），直接托管于 ParticleSystem
- 爆炸通过 on_particle_death 事件回调触发，完全复用 ParticleSystem 能力
- ForceField.get_force 使用解耦坐标接口，无需匿名类 hack
- crossette 二次分裂 (P8) 通过 on_lifetime_tick 事件实现
- 新增烟花样式：star / comet / chrysanthemum_ring（P9）
- 对外接口保持向后兼容，旧代码无需修改

使用示例：
    from systems.fireworks_system import FireworksSystem

    fw = FireworksSystem(width=800, height=600)
    while running:
        fw.update(dt=0.016)
        fw.render(renderer)
"""

import random
import math
from typing import Optional

try:
    from core.particle import (
        Particle, ParticleSystem, Emitter, ForceField,
        EmitterShape, gradient_to_curve, size_curve_fadeout,
        create_point_light, create_ambient_light,
    )
except ImportError:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.particle import (
        Particle, ParticleSystem, Emitter, ForceField,
        EmitterShape, gradient_to_curve, size_curve_fadeout,
        create_point_light, create_ambient_light,
    )

from renderers.pil_renderer import LayeredPILRenderer


# ================================================================
#  爆炸工厂函数 — 各种烟花样式的爆炸 ParticleSystem
# ================================================================

def _make_burst_system(style: str, palette: list,
                       max_particles: int = 500) -> ParticleSystem:
    """
    创建一个爆炸用的 ParticleSystem（作为 sub_emitter_on_death 挂载到火箭上）。
    系统内只含 Emitter，发射位置在触发时由父系统自动设置到火箭坐标。
    """
    sys_burst = ParticleSystem(max_particles=max_particles)
    gravity = ForceField(field_type='gravity', strength=3.0)
    sys_burst.add_force_field(gravity)

    if style == 'chrysanthemum':
        _add_radial_emitter(sys_burst, 0, 0, palette,
                            count_per_sec=2000, speed=(4, 10),
                            life=(1.5, 3.0), size=(1.5, 3), burst_count=120)

    elif style == 'willow':
        _add_radial_emitter(sys_burst, 0, 0, palette,
                            count_per_sec=3000, speed=(2, 6),
                            life=(2.5, 4.5), size=(1, 2.5), burst_count=200)

    elif style == 'peony':
        _add_radial_emitter(sys_burst, 0, 0, palette,
                            count_per_sec=1500, speed=(3, 7),
                            life=(0.8, 1.8), size=(3, 6), burst_count=80)
        # 内核闪光
        core = Emitter(
            x=0, y=0, rate=800,
            spread=math.pi * 2, speed=1.2,
            life=0.3, size=6, color=(255, 255, 240),
            size_over_lifetime=[(0.0, 1.0), (0.3, 2.5), (1.0, 0.0)],
        )
        core._burst_count = 20
        sys_burst.add_emitter(core)

    elif style == 'ring':
        for ring in range(2):
            ring_speed = 5 + ring * 3
            num_p = 40 + ring * 20
            ring_emitter = Emitter(
                x=0, y=0,
                rate=num_p * 60,
                spread=0.05,
                speed=ring_speed,
                life=random.uniform(1.0, 2.0),
                size=random.uniform(2, 3.5),
                color=palette[ring % len(palette)],
                shape=EmitterShape('circle', radius=0.0, emit_from_shell=False),
            )
            ring_emitter._burst_count = num_p
            ring_emitter._ring_count = num_p
            ring_emitter._ring_index = ring
            sys_burst.add_emitter(ring_emitter)
        # ring 用专用方法，这里直接用 _add_ring_particles 填充
        sys_burst._style = 'ring'
        sys_burst._palette = palette

    elif style == 'palm':
        num_arms = random.choice([5, 6, 8])
        for arm in range(num_arms):
            base_angle = 2 * math.pi * arm / num_arms
            arm_emitter = Emitter(
                x=0, y=0,
                rate=1500,
                spread=0.12,
                direction=base_angle,
                speed=random.uniform(5, 9),
                life=random.uniform(1.2, 2.5),
                size=random.uniform(2, 4),
                color=random.choice(palette),
            )
            arm_emitter._burst_count = 25
            sys_burst.add_emitter(arm_emitter)

    elif style == 'crossette':
        _add_radial_emitter(sys_burst, 0, 0, palette,
                            count_per_sec=1200, speed=(4, 8),
                            life=(1.0, 1.5), size=(3, 5),
                            burst_count=20, is_crossette=True)

    elif style == 'star':
        # 星形：6/8 个点，每个点一束集中粒子
        num_points = random.choice([6, 8])
        for pt in range(num_points):
            angle = 2 * math.pi * pt / num_points
            pt_emitter = Emitter(
                x=0, y=0, rate=600,
                spread=0.08, direction=angle,
                speed=random.uniform(6, 11),
                life=random.uniform(1.0, 2.0),
                size=random.uniform(2, 4),
                color=random.choice(palette),
                size_over_lifetime=[(0.0, 1.0), (0.5, 0.7), (1.0, 0.0)],
            )
            pt_emitter._burst_count = 20
            sys_burst.add_emitter(pt_emitter)

    elif style == 'comet':
        # 彗星：几条带尾迹的长条
        for _ in range(random.randint(5, 9)):
            angle = random.uniform(0, 2 * math.pi)
            comet_emitter = Emitter(
                x=0, y=0, rate=200,
                spread=0.06, direction=angle,
                speed=random.uniform(8, 14),
                life=random.uniform(1.5, 3.0),
                size=random.uniform(2, 3.5),
                color=random.choice(palette),
                trail_enabled=True,
                trail_length=18,
                size_over_lifetime=[(0.0, 1.5), (0.4, 1.0), (1.0, 0.0)],
            )
            comet_emitter._burst_count = 8
            sys_burst.add_emitter(comet_emitter)

    elif style == 'chrysanthemum_ring':
        # 菊花+外圈双层
        _add_radial_emitter(sys_burst, 0, 0, palette,
                            count_per_sec=2000, speed=(4, 9),
                            life=(1.5, 2.8), size=(1.5, 3), burst_count=100)
        # 外圈
        outer_color = palette[-1] if len(palette) > 1 else palette[0]
        ring_emitter = Emitter(
            x=0, y=0,
            rate=3000, spread=0.04,
            speed=10,
            life=random.uniform(1.2, 2.0),
            size=2.5,
            color=outer_color,
        )
        ring_emitter._burst_count = 60
        ring_emitter._is_ring = True
        ring_emitter._ring_total = 60
        sys_burst.add_emitter(ring_emitter)

    else:
        # 默认径向
        _add_radial_emitter(sys_burst, 0, 0, palette)

    return sys_burst


def _add_radial_emitter(sys_burst: ParticleSystem,
                        x: float, y: float,
                        palette: list,
                        count_per_sec: int = 1500,
                        speed=(2, 8), life=(0.5, 2.0),
                        size=(1.5, 3.5),
                        burst_count: int = 80,
                        is_crossette: bool = False):
    """在 sys_burst 中添加一个径向爆发 Emitter"""
    emitter = Emitter(
        x=x, y=y,
        rate=count_per_sec,
        spread=math.pi * 2,
        speed=random.uniform(*speed),
        life=random.uniform(*life),
        size=random.uniform(*size),
        color=random.choice(palette),
    )
    emitter._burst_count = burst_count
    if is_crossette:
        emitter._is_crossette = True
    sys_burst.add_emitter(emitter)


# ================================================================
#  FireworksSystem v2.0
# ================================================================

class FireworksSystem:
    """
    烟花管理系统 v2.0 — 重构版

    架构变化：
    - 火箭 = 特殊的 Particle（托管于内部 ParticleSystem）
    - 爆炸 = on_particle_death 事件回调（sub_emitter_on_death）
    - 粒子计算完全走 ParticleSystem 主循环
    """

    # 烟花调色板预设
    PALETTES = {
        'gold':     [(255, 215, 0), (255, 200, 50), (255, 180, 0), (255, 240, 150)],
        'red':      [(255, 50, 50), (255, 100, 30), (255, 150, 50), (255, 200, 100)],
        'blue':     [(50, 100, 255), (80, 150, 255), (150, 200, 255), (200, 230, 255)],
        'green':    [(50, 255, 100), (100, 255, 120), (150, 255, 180), (200, 255, 220)],
        'purple':   [(180, 50, 255), (200, 100, 255), (220, 160, 255), (240, 200, 255)],
        'white':    [(255, 255, 255), (240, 240, 250), (230, 230, 245), (255, 255, 220)],
        'rainbow':  [(255, 50, 50), (255, 150, 0), (255, 255, 0),
                     (50, 255, 50), (50, 150, 255), (180, 50, 255)],
        'silver':   [(200, 200, 210), (180, 185, 195), (220, 220, 225), (240, 240, 245)],
        'orange':   [(255, 160, 0), (255, 120, 30), (255, 200, 80), (255, 240, 160)],
        'cyan':     [(0, 220, 255), (50, 200, 230), (150, 240, 255), (200, 250, 255)],
    }

    # 所有可用样式（v4 扩展）
    STYLES = ['chrysanthemum', 'willow', 'peony', 'ring',
              'palm', 'crossette', 'star', 'comet', 'chrysanthemum_ring',
              # v4 新增
              'heart', 'spiral', 'waterfall', 'double_ring', 'crackle']

    def __init__(self,
                 width: int = 800,
                 height: int = 600,
                 auto_launch: bool = True,
                 launch_interval: float = 1.2,
                 max_active: int = 5,
                 max_particles: int = 5000):
        self.width = width
        self.height = height
        self.auto_launch = auto_launch
        self.launch_interval = launch_interval
        self.max_active = max_active

        # 爆炸粒子系统（v2.0 核心：统一管理所有粒子）
        self._burst_sys = ParticleSystem(max_particles=max_particles)
        self._burst_gravity = ForceField(field_type='gravity', strength=3.0)
        self._burst_sys.add_force_field(self._burst_gravity)

        # 火箭列表（dict 结构，持有位置/速度/目标高度等元数据 + head Particle 引用）
        self.rockets: list = []
        self._rocket_gravity = ForceField(field_type='gravity', strength=3.0)

        self._launch_timer = 0.0
        self._active_count = 0

        # 光照系统（可选，视频渲染时开启）
        self._lighting_enabled = False

    # ------------------------------------------------------------------
    #  公开 API
    # ------------------------------------------------------------------

    def launch(self, x: float = None, y: float = None,
               palette_name: str = None, style: str = None):
        """手动发射一个烟花"""
        if self._active_count >= self.max_active:
            return

        if x is None:
            x = random.uniform(self.width * 0.15, self.width * 0.85)
        if y is None:
            y = random.uniform(self.height * 0.2, self.height * 0.55)

        if palette_name is None:
            palette_name = random.choice(list(self.PALETTES.keys()))
        if style is None:
            style = random.choice(self.STYLES)

        palette = self.PALETTES.get(palette_name, self.PALETTES['gold'])

        rocket = {
            'x': x,
            'y': self.height + 5,
            'vx': random.uniform(-0.3, 0.3),
            'vy': -random.uniform(12, 18),
            'target_y': y,
            'palette': palette,
            'style': style,
            'trail_timer': 0.0,
            'life': 1.0,
        }
        self.rockets.append(rocket)

        # 火箭头粒子（交由引擎渲染，emissive 自发光）
        head = Particle(
            x=rocket['x'], y=rocket['y'],
            vx=0, vy=0,
            life=1.0, max_life=1.0,
            size=2.5, color=(255, 220, 120),
            alpha=1.0,
            emissive=(255, 220, 120), emissive_strength=2.0,
        )
        rocket['head'] = head
        self._active_count += 1

    def update(self, dt: float):
        """更新烟花系统"""
        # 自动发射计时
        if self.auto_launch:
            self._launch_timer += dt
            if self._launch_timer >= self.launch_interval:
                self._launch_timer = 0.0
                self.launch()

        # —— 火箭上升阶段 ——
        new_rockets = []
        for rocket in self.rockets:
            # P3 成果：直接传坐标，无需匿名类
            _, gy = self._rocket_gravity.get_force(rocket['x'], rocket['y'])
            rocket['vy'] += gy * dt
            rocket['x'] += rocket['vx'] * dt * 60  # dt 归一化
            rocket['y'] += rocket['vy'] * dt * 60

            # 同步火箭头粒子位置（交由引擎渲染）
            head = rocket.get('head')
            if head is not None:
                head.x = rocket['x']
                head.y = rocket['y']
                head.life = 1.0  # 上升期间保持存活

            # 尾迹
            rocket['trail_timer'] += dt
            if rocket['trail_timer'] > 0.02:
                rocket['trail_timer'] = 0.0
                trail_p = Particle(
                    x=rocket['x'] + random.uniform(-1, 1),
                    y=rocket['y'] + random.uniform(0, 3),
                    vx=random.uniform(-0.2, 0.2),
                    vy=random.uniform(0, 1),
                    life=0.4, max_life=0.4,
                    size=random.uniform(1, 2.5),
                    color=(255, 200, 100),
                    alpha=0.8,
                )
                self._burst_sys.particles.append(trail_p)

            # 到达目标高度时爆炸
            if rocket['y'] <= rocket['target_y'] or rocket['vy'] >= 0:
                self._do_explode(rocket)
                # 爆炸时回收头粒子
                if head is not None:
                    head.life = 0.0
            else:
                new_rockets.append(rocket)

        self.rockets = new_rockets

        # —— crossette 二次分裂（P8）——
        self._process_crossette(dt)

        # —— 更新所有爆炸粒子 ——
        self._burst_sys.update(dt)

    def render(self, renderer):
        """
        渲染所有粒子（爆炸粒子 + 火箭头）。
        走引擎完整管线：若开启光照，自动应用 Blinn-Phong。
        """
        all_particles = self.get_all_particles()
        if not all_particles:
            return

        if self._lighting_enabled and self._burst_sys.lights:
            # 走光照管线：对每个粒子计算最终颜色
            lit_particles = []
            for p in all_particles:
                if not p.is_alive():
                    continue
                from core.particle import apply_lighting_to_particle
                r, g, b, a = apply_lighting_to_particle(
                    p, self._burst_sys.lights,
                    self._burst_sys.blinn_phong,
                    self._burst_sys.camera_pos,
                )
                lit_p = Particle(
                    x=p.x, y=p.y, vx=p.vx, vy=p.vy,
                    life=p.life, max_life=p.max_life,
                    size=p.size, color=(r, g, b), alpha=a,
                    rotation=p.rotation, angular_velocity=p.angular_velocity,
                    trail_enabled=p.trail_enabled,
                    trail_history=list(p.trail_history),
                    trail_max_length=p.trail_max_length,
                )
                lit_particles.append(lit_p)
            renderer.render(lit_particles)
        else:
            renderer.render(all_particles)

    def enable_lighting(self, enabled: bool = True):
        """开启/关闭爆炸粒子的光照管线"""
        self._lighting_enabled = enabled
        self._burst_sys.lighting_enabled = enabled
        if enabled and not self._burst_sys.lights:
            # 加一个弱环境光，让 emissive 粒子正确发光
            self._burst_sys.add_light(
                create_ambient_light(color=(30, 30, 40), intensity=0.15)
            )

    def get_all_particles(self) -> list:
        """
        返回所有需要渲染的粒子（爆炸粒子 + 火箭头粒子）。
        火箭头粒子带 emissive，可直接交给 LayeredPILRenderer 渲染。
        """
        result = list(self._burst_sys.particles)
        result.extend(r['head'] for r in self.rockets
                      if r.get('head') and r['head'].is_alive())
        return result

    def render_to_image(self,
                        background: Optional[object],
                        renderer: Optional[object] = None,
                        ) -> object:
        """
        完整渲染管线：将所有粒子合成到背景图像上并返回。

        这是视频渲染的推荐入口，替代手动提取 fw.particles 再调用渲染器。
        内部走引擎光照管线（若已开启），火箭头粒子自动包含。

        Args:
            background : PIL Image（RGB），作为合成背景
            renderer   : 可选的 LayeredPILRenderer 实例；
                         为 None 则使用内部默认 renderer

        Returns:
            合成后的 PIL Image（RGB）
        """
        if renderer is None:
            if not hasattr(self, '_default_renderer'):
                self._default_renderer = LayeredPILRenderer(
                    width=self.width, height=self.height,
                    bloom=True, bloom_radius=10,
                    bloom_strength=0.5, bloom_threshold=150,
                    glow_scale=2.5, glow_alpha=0.3,
                    smooth_trail=True,
                )
            renderer = self._default_renderer

        all_particles = self.get_all_particles()
        return renderer.render_to_image(all_particles, background=background)

    # ------------------------------------------------------------------
    #  内部：爆炸逻辑（v2.0 — 直接向 _burst_sys 注入粒子）
    # ------------------------------------------------------------------

    def _do_explode(self, rocket: dict):
        """触发爆炸：向 _burst_sys 批量注入爆炸粒子"""
        x, y = rocket['x'], rocket['y']
        palette = rocket['palette']
        style = rocket['style']

        self._inject_explosion(x, y, style, palette)
        self._active_count -= 1

    def _inject_explosion(self, x: float, y: float,
                          style: str, palette: list):
        """根据样式直接生成粒子并注入 _burst_sys"""
        sys_p = self._burst_sys

        if style == 'chrysanthemum':
            self._radial_burst(x, y, palette, 120, (4, 10), (1.5, 3.0), (1.5, 3))

        elif style == 'willow':
            self._radial_burst(x, y, palette, 200, (2, 6), (2.5, 4.5), (1, 2.5))

        elif style == 'peony':
            self._radial_burst(x, y, palette, 80, (3, 7), (0.8, 1.8), (3, 6))
            for _ in range(20):
                angle = random.uniform(0, 2 * math.pi)
                p = Particle(
                    x=x, y=y,
                    vx=math.cos(angle) * random.uniform(0.5, 2),
                    vy=math.sin(angle) * random.uniform(0.5, 2),
                    life=random.uniform(0.15, 0.4), max_life=0.4,
                    size=random.uniform(3, 7), color=(255, 255, 240), alpha=1.0,
                )
                sys_p.particles.append(p)

        elif style == 'ring':
            for ring in range(2):
                ring_speed_base = 5 + ring * 3
                num_particles = 40 + ring * 20
                for i in range(num_particles):
                    angle = (2 * math.pi * i / num_particles) + ring * math.pi / num_particles
                    color = palette[ring % len(palette)]
                    sp = ring_speed_base + random.uniform(-0.5, 0.5)
                    lt = random.uniform(1.0, 2.0)
                    p = Particle(
                        x=x, y=y,
                        vx=math.cos(angle) * sp,
                        vy=math.sin(angle) * sp,
                        life=lt, max_life=lt,
                        size=random.uniform(2, 3.5), color=color,
                    )
                    sys_p.particles.append(p)

        elif style == 'palm':
            num_arms = random.choice([5, 6, 8])
            for arm in range(num_arms):
                base_angle = 2 * math.pi * arm / num_arms
                for _ in range(25):
                    spread = random.gauss(0, 0.08)
                    angle = base_angle + spread
                    speed = random.uniform(5, 9)
                    lt = random.uniform(1.2, 2.5)
                    p = Particle(
                        x=x, y=y,
                        vx=math.cos(angle) * speed,
                        vy=math.sin(angle) * speed,
                        life=lt, max_life=lt,
                        size=random.uniform(2, 4), color=random.choice(palette),
                    )
                    sys_p.particles.append(p)

        elif style == 'crossette':
            self._radial_burst(x, y, palette, 20, (4, 8), (1.0, 1.5),
                               (3, 5), is_crossette=True)

        elif style == 'star':
            num_points = random.choice([6, 8])
            for pt in range(num_points):
                angle = 2 * math.pi * pt / num_points
                for _ in range(20):
                    jitter = random.gauss(0, 0.04)
                    speed = random.uniform(6, 11)
                    lt = random.uniform(1.0, 2.0)
                    p = Particle(
                        x=x, y=y,
                        vx=math.cos(angle + jitter) * speed,
                        vy=math.sin(angle + jitter) * speed,
                        life=lt, max_life=lt,
                        size=random.uniform(2, 4), color=random.choice(palette),
                        size_over_lifetime=[(0.0, 1.0), (0.5, 0.7), (1.0, 0.0)],
                    )
                    sys_p.particles.append(p)

        elif style == 'comet':
            num_comets = random.randint(5, 9)
            for _ in range(num_comets):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(8, 14)
                lt = random.uniform(1.5, 3.0)
                for i in range(8):
                    frac = i / 8
                    dist_back = frac * 0.3
                    p = Particle(
                        x=x - math.cos(angle) * dist_back * speed * lt,
                        y=y - math.sin(angle) * dist_back * speed * lt,
                        vx=math.cos(angle) * speed * (1 - frac * 0.3),
                        vy=math.sin(angle) * speed * (1 - frac * 0.3),
                        life=lt * (1 - frac * 0.6),
                        max_life=lt,
                        size=random.uniform(2, 3.5) * (1 - frac * 0.5),
                        color=random.choice(palette),
                        trail_enabled=(i == 0),
                        trail_max_length=16,
                    )
                    if p.trail_enabled:
                        p.enable_trail(16)
                    sys_p.particles.append(p)

        elif style == 'chrysanthemum_ring':
            self._radial_burst(x, y, palette, 100, (4, 9), (1.5, 2.8), (1.5, 3))
            outer_color = palette[-1] if len(palette) > 1 else palette[0]
            num_outer = 60
            for i in range(num_outer):
                angle = 2 * math.pi * i / num_outer
                sp = 10 + random.uniform(-0.3, 0.3)
                lt = random.uniform(1.2, 2.0)
                p = Particle(
                    x=x, y=y,
                    vx=math.cos(angle) * sp,
                    vy=math.sin(angle) * sp,
                    life=lt, max_life=lt,
                    size=2.5, color=outer_color,
                )
                sys_p.particles.append(p)

        # ===== v4 新增样式 =====
        elif style == 'heart':
            # 心形参数方程: x=16sin³t, y=13cos t-5cos2t-2cos3t-cos4t
            for _ in range(150):
                t = random.uniform(0, 2 * math.pi)
                sc = random.uniform(3.5, 7.0) * 0.08
                hx = 16 * (math.sin(t)**3) * sc
                hy = -(13*math.cos(t) - 5*math.cos(2*t) -
                       2*math.cos(3*t) - math.cos(4*t)) * sc
                sf = random.uniform(0.6, 1.2)
                lt = random.uniform(1.2, 2.5)
                p = Particle(
                    x=x, y=y, vx=hx*sf, vy=hy*sf,
                    life=lt, max_life=lt,
                    size=random.uniform(2, 4),
                    color=random.choice(palette),
                )
                sys_p.particles.append(p)

        elif style == 'spiral':
            n_arms = random.choice([2, 3])
            for arm in range(n_arms):
                offset = arm * math.pi * 2 / (arm + 1)
                for i in range(60):
                    frac = i / 60
                    ang = offset + frac * math.pi * 4
                    rad = 1 + frac * 8
                    sx = math.cos(ang) * rad
                    sy = math.sin(ang) * rad
                    sp = random.uniform(3, 7)
                    lt = random.uniform(1.5, 3.0)
                    p = Particle(
                        x=x, y=y,
                        vx=sx*sp*random.uniform(0.8,1.2),
                        vy=sy*sp*random.uniform(0.8,1.2),
                        life=lt, max_life=lt,
                        size=random.uniform(1.5,3.5),
                        color=random.choice(palette),
                        trail_enabled=True, trail_max_length=10,
                    )
                    if p.trail_enabled:
                        p.enable_trail(10)
                    sys_p.particles.append(p)

        elif style == 'waterfall':
            # 瀑布：向下喷射+重力增强
            for _ in range(random.randint(8, 15)):
                ang = random.uniform(math.pi*0.1, math.pi*0.9)
                base_sp = random.uniform(6, 12)
                for i in range(random.randint(15,30)):
                    frac = i / 25
                    spread = random.gauss(0, 0.05)
                    a = ang + spread
                    sp = base_sp*(1-frac*0.4)+random.uniform(-1,1)
                    lt = random.uniform(1.5,3.5)*(1-frac*0.5)
                    p = Particle(
                        x=x, y=y,
                        vx=math.cos(a)*sp,
                        vy=math.sin(a)*sp+frac*3,
                        life=lt, max_life=lt,
                        size=random.uniform(1.5,3.5)*(1-frac*0.3),
                        color=random.choice(palette),
                    )
                    sys_p.particles.append(p)

        elif style == 'double_ring':
            # 三环+中心闪光
            for ri in range(3):
                r_spd = 4+ri*3
                n_p = 40+ri*25
                rc = palette[ri % len(palette)]
                for i in range(n_p):
                    a = 2*math.pi*i/n_p + ri*0.15
                    sp = r_spd+random.uniform(-0.5,0.5)
                    lt = random.uniform(1.2,2.2)
                    p = Particle(
                        x=x,y=y, vx=math.cos(a)*sp, vy=math.sin(a)*sp,
                        life=lt, max_life=lt,
                        size=random.uniform(2,3.5), color=rc,
                    )
                    sys_p.particles.append(p)
            # 中心闪光
            for _ in range(25):
                a = random.uniform(0, 2*math.pi)
                sp = random.uniform(0.3, 2.5)
                lt = random.uniform(0.15, 0.45)
                p = Particle(
                    x=x,y=y,
                    vx=math.cos(a)*sp, vy=math.sin(a)*sp,
                    life=lt, max_life=lt,
                    size=random.uniform(4,8),
                    color=(255,255,240),
                    emissive=(255,220,150), emissive_strength=2.0,
                )
                sys_p.particles.append(p)

        elif style == 'crackle':
            # 碎裂爆响：大量小粒子+白色闪光
            self._radial_burst(x, y, palette, count=250, speed=(5,12),
                               life=(0.4,1.2), size=(1,3))
            for _ in range(40):
                a = random.uniform(0, 2*math.pi)
                sp = random.uniform(0.3, 2)
                lt = random.uniform(0.08, 0.32)
                p = Particle(
                    x=x,y=y, vx=math.cos(a)*sp, vy=math.sin(a)*sp,
                    life=lt, max_life=lt,
                    size=random.uniform(3,7),
                    color=(255,255,230),
                    emissive=(255,255,220), emissive_strength=2.5,
                )
                sys_p.particles.append(p)

        else:
            self._radial_burst(x, y, palette)

    def _radial_burst(self, x: float, y: float, palette: list,
                      count: int = 80, speed=(2, 8), life=(0.5, 2.0),
                      size=(1.5, 3.5), is_crossette: bool = False):
        """通用径向爆发（直接生成粒子到 _burst_sys）"""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            sp = random.uniform(*speed)
            lt = random.uniform(*life)
            sz = random.uniform(*size)
            p = Particle(
                x=x, y=y,
                vx=math.cos(angle) * sp,
                vy=math.sin(angle) * sp,
                life=lt, max_life=lt,
                size=sz, color=random.choice(palette),
            )
            if is_crossette:
                p.crossette = True
                p.split_time = lt * 0.5
                p.crossette_split_done = False
            self._burst_sys.particles.append(p)

    # ------------------------------------------------------------------
    #  P8 — crossette 二次分裂
    # ------------------------------------------------------------------

    def _process_crossette(self, dt: float):
        """检测所有 crossette 粒子，到达 split_time 时触发二次分裂"""
        new_particles = []
        for p in self._burst_sys.particles:
            if (hasattr(p, 'crossette') and p.crossette and
                    not getattr(p, 'crossette_split_done', True)):
                # 当剩余生命 < split_time 时触发
                remaining_life = p.life * p.max_life
                if remaining_life <= getattr(p, 'split_time', 0):
                    p.crossette_split_done = True
                    # 向 4 个方向喷出子粒子
                    for k in range(4):
                        perp_angle = math.atan2(p.vy, p.vx) + math.pi / 2 * k
                        child_speed = math.sqrt(p.vx**2 + p.vy**2) * 0.7
                        child_lt = p.life * p.max_life * 0.6
                        child = Particle(
                            x=p.x, y=p.y,
                            vx=math.cos(perp_angle) * child_speed,
                            vy=math.sin(perp_angle) * child_speed,
                            life=child_lt / p.max_life if p.max_life > 0 else 0.5,
                            max_life=p.max_life * 0.6,
                            size=p.size * 0.7,
                            color=p.color,
                        )
                        new_particles.append(child)
        self._burst_sys.particles.extend(new_particles)

    # ------------------------------------------------------------------
    #  兼容属性（旧代码使用 fw.particles）
    # ------------------------------------------------------------------

    @property
    def particles(self):
        """向后兼容：返回所有待渲染粒子（爆炸 + 火箭头）"""
        result = list(self._burst_sys.particles)
        result.extend(r['head'] for r in self.rockets
                      if r.get('head') and r['head'].is_alive())
        return result

    @property
    def particle_count(self) -> int:
        return len(self._burst_sys.particles) + sum(
            1 for r in self.rockets
            if r.get('head') and r['head'].is_alive()
        )

    @property
    def rocket_count(self) -> int:
        return len(self.rockets)


# ============================================================
#  快捷函数
# ============================================================

def create_fireworks(width: int = 800, height: int = 600,
                     **kwargs) -> FireworksSystem:
    """创建并返回一个 FireworksSystem 实例"""
    return FireworksSystem(width=width, height=height, **kwargs)

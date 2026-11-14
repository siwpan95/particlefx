"""
交互式粒子光场系统（LightFieldSystem）v1.0

把 ``G:/assets/particle-light-field.html`` 那套零依赖 Canvas 交互模型，
移植为 particlefx 的**原生 Python 系统**，二者共享同一套交互语义：

- 确定性随机数（mulberry32 + 按粒子序号派生 seed）-> 相同 seed 下画面完全可复现
- 环境流场：让画面持续流动且不依赖随机（可复现）
- 鼠标交互：移动 = 吸引，按住 = 排斥，点击 = 冲击波
- 速度 -> 冷暖色相连续渐变（Aurora / Ember / Toxic 三套配色，可平滑过渡）
- 阻尼、限速（防数值爆炸）、软边界回弹

渲染交给库内渲染器：
- 实时：``PygameRenderer(blend_mode='additive', bloom_enabled=True)`` 得到发光+拖尾观感
- 离线/导出：``LayeredPILRenderer``（save_image 内部使用，需 Pillow）

与 HTML 的差异（刻意为之）：
- 物理为纯 Python 逐粒子积分，复用 ``Particle`` 数据结构，可直接被任意渲染器消费；
- 拖尾由渲染器的 additive + bloom / smooth_trail 体现，而非 Canvas 的半透明覆盖；
- 冲击波环为可选项，由调用方（Demo）在渲染后叠加绘制（headless 下不参与计算）。

无 pygame / Pillow 依赖也可做无头仿真与单元测试（见 selfcheck / tests）。
"""

import math
from typing import List, Optional, Tuple

from ..core.particle import Particle, ParticleSystem


# ================================================================
#  确定性随机数（与 HTML 版 mulberry32 / hashSeed 逐位对齐）
#  统一从 particlefx.core.random 导入，避免与核心库 / 其他系统重复实现。
# ================================================================

from ..core.random import mulberry32, hash_seed


# ================================================================
#  颜色辅助
# ================================================================

def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
    """HSL -> RGB（h:0~360, s/l:0~1），返回 0~255 整数三元组。"""
    h = h % 360.0
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60.0) % 2 - 1))
    m = l - c / 2
    if h < 60:
        r, g, b = c, x, 0.0
    elif h < 120:
        r, g, b = x, c, 0.0
    elif h < 180:
        r, g, b = 0.0, c, x
    elif h < 240:
        r, g, b = 0.0, x, c
    elif h < 300:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


# ================================================================
#  光场配色（对齐 HTML：bg + cold 色相 + warm 色相）
#  粒子速度低 -> cold 色相，速度高 -> warm 色相，二者间连续渐变。
# ================================================================

LIGHT_FIELD_PALETTES = [
    {"name": "Aurora", "bg": (6, 8, 20), "cold": 190.0, "warm": 335.0},  # 青 -> 品红
    {"name": "Ember",  "bg": (14, 7, 6), "cold": 225.0, "warm": 28.0},   # 蓝 -> 橙
    {"name": "Toxic",  "bg": (4, 12, 9), "cold": 165.0, "warm": 0.0},    # 绿 -> 红
]


# ================================================================
#  LightFieldSystem
# ================================================================

class LightFieldSystem:
    """
    交互式粒子光场系统。

    物理参数（类属性，可按需覆盖）：
        DAMP / ATTRACTION / REPULSE / INFLUENCE / SOFTEN /
        COLOR_SPEED / MAX_SPEED / FADE_ALPHA

    交互 API：
        set_pointer(x, y, active=True, pressing=False)
        trigger_shockwave(x=None, y=None)
        cycle_palette()
        reset(seed=None)            # 用 seed 重新确定性播种
        set_count(n)                # 增减粒子数（保持现有粒子的 seed 语义）
        update(dt)                  # 推进一帧
        render(renderer)            # 交给任意渲染器绘制
        save_image(path)            # 离线导出 PNG（需 Pillow）

    可复现性：相同 seed + 相同 dt 序列 + 相同指针序列 => 逐粒子位置完全一致。
    """

    # ---- 物理 / 渲染常量（对齐 HTML）----
    DAMP = 0.985          # 阻尼（每帧基准）
    ATTRACTION = 0.55     # 鼠标引力强度（移动）
    REPULSE = 1.10        # 鼠标斥力强度（按住）
    INFLUENCE = 240.0     # 鼠标影响半径 (px)
    SOFTEN = 8.0          # 软化项，避免近距奇点
    COLOR_SPEED = 5.5     # 映射到暖色所需速度阈值
    MAX_SPEED = 14.0      # 速度上限，防止数值爆炸
    FADE_ALPHA = 0.16     # 拖尾衰减（Demo 用半透明覆盖实现）

    def __init__(self,
                 width: int = 800,
                 height: int = 600,
                 count: int = 1500,
                 seed: int = 0,
                 palette_index: int = 0,
                 max_particles: int = 3000):
        self.width = int(width)
        self.height = int(height)
        self.count = max(0, min(int(count), int(max_particles)))
        self.max_particles = int(max_particles)
        self.seed = int(seed)

        # 内嵌一个 ParticleSystem 作为粒子容器，复用库的渲染/计数约定
        # （不调用其 update，物理由本系统自行积分，保持对 HTML 模型的精确复刻）
        self._sys = ParticleSystem(max_particles=self.max_particles)

        # 指针交互状态
        self.pointer_x = 0.0
        self.pointer_y = 0.0
        self.pointer_active = False
        self.pointer_pressing = False

        # 冲击波环列表（仅元数据；渲染由调用方叠加）
        self.shockwaves: List[dict] = []

        # 配色状态
        self.pal_index = int(palette_index) % len(LIGHT_FIELD_PALETTES)
        p0 = LIGHT_FIELD_PALETTES[self.pal_index]
        self.cur_cold = p0["cold"]
        self.cur_warm = p0["warm"]
        self.transitioning = False

        self._time = 0.0
        self.paused = False

        # 播种粒子
        self._seed_particles(self.count)

    # ------------------------------------------------------------------
    #  粒子播种（确定性）
    # ------------------------------------------------------------------

    def _make_particle(self, i: int) -> Particle:
        """用 (seed, i) 派生第 i 颗粒子的确定性初始属性。"""
        rng = mulberry32(hash_seed(self.seed, i))
        x = rng() * self.width
        y = rng() * self.height
        vx = (rng() - 0.5) * 1.2
        vy = (rng() - 0.5) * 1.2
        phase = rng() * math.pi * 2.0
        size = 6.0 + rng() * 12.0
        return Particle(
            x=x, y=y, vx=vx, vy=vy,
            life=1.0, max_life=1.0,
            size=size, color=(180, 200, 255), alpha=1.0,
            # 把 phase 暂存进 rotation 字段（Particle 无 phase 字段，复用可序列化字段）
            rotation=phase,
        )

    def _seed_particles(self, n: int):
        """重建恰好 n 颗确定性粒子。"""
        self._sys.particles.clear()
        self.count = int(n)
        for i in range(self.count):
            self._sys.particles.append(self._make_particle(i))

    # ------------------------------------------------------------------
    #  公开 API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None):
        """用 seed 重新确定性播种全部粒子（画面可复现）。"""
        if seed is not None:
            self.seed = int(seed)
        self._seed_particles(self.count)

    def set_count(self, n: int):
        """增减粒子数：增加时按当前 seed 续播，减少时截断尾部。"""
        n = max(0, min(int(n), self.max_particles))
        cur = len(self._sys.particles)
        if n > cur:
            for i in range(cur, n):
                self._sys.particles.append(self._make_particle(i))
        elif n < cur:
            del self._sys.particles[n:]
        self.count = n

    def set_pointer(self, x: float, y: float,
                    active: bool = True, pressing: bool = False):
        """设置指针位置与状态：移动=吸引，按住=排斥。"""
        self.pointer_x = float(x)
        self.pointer_y = float(y)
        self.pointer_active = bool(active)
        self.pointer_pressing = bool(pressing)

    def trigger_shockwave(self, x: Optional[float] = None, y: Optional[float] = None):
        """在指定位置（默认当前指针）触发冲击波：给附近粒子向外冲量 + 记录可见环。"""
        cx = self.pointer_x if x is None else float(x)
        cy = self.pointer_y if y is None else float(y)
        max_r = self.INFLUENCE * 1.6
        for p in self._sys.particles:
            dx = p.x - cx
            dy = p.y - cy
            d = math.hypot(dx, dy) or 1.0
            f = 9.0 * (1.0 / (1.0 + d * 0.012))
            p.vx += (dx / d) * f
            p.vy += (dy / d) * f
        self.shockwaves.append({"x": cx, "y": cy, "r": 0.0,
                                "life": 1.0, "max": max_r})

    def cycle_palette(self):
        """切换到下一套配色，并平滑过渡。"""
        self.pal_index = (self.pal_index + 1) % len(LIGHT_FIELD_PALETTES)
        self.transitioning = True

    def current_palette_name(self) -> str:
        return LIGHT_FIELD_PALETTES[self.pal_index]["name"]

    def current_bg(self) -> Tuple[int, int, int]:
        """返回当前配色的背景色（Demo 用于清屏 / 拖尾底色）。"""
        return LIGHT_FIELD_PALETTES[self.pal_index]["bg"]

    # ------------------------------------------------------------------
    #  主更新循环
    # ------------------------------------------------------------------

    def update(self, dt: float):
        """推进一帧物理（dt 单位为毫秒级帧时长；内部按 16.667ms 归一）。"""
        if self.paused:
            return

        df = min(dt / 16.667, 2.2)          # 帧率归一系数
        damp = math.pow(self.DAMP, df)
        t = self._time * 0.001              # 秒

        # 配色平滑过渡：仅在过渡期重算色相，稳态不浪费 CPU
        if self.transitioning:
            tg = LIGHT_FIELD_PALETTES[self.pal_index]
            self.cur_cold += (tg["cold"] - self.cur_cold) * 0.06
            self.cur_warm += (tg["warm"] - self.cur_warm) * 0.06
            if (abs(tg["cold"] - self.cur_cold) < 0.4 and
                    abs(tg["warm"] - self.cur_warm) < 0.4):
                self.cur_cold = tg["cold"]
                self.cur_warm = tg["warm"]
                self.transitioning = False

        infl2 = self.INFLUENCE * self.INFLUENCE
        sign = -1.0 if self.pointer_pressing else 1.0
        strength = self.REPULSE if self.pointer_pressing else self.ATTRACTION

        for p in self._sys.particles:
            # (a) 确定性环境流场：让画面持续流动（可复现）
            ax = math.cos(p.y * 0.012 + t * 0.35 + p.rotation) * 0.018
            ay = math.sin(p.x * 0.012 + t * 0.35 + p.rotation) * 0.018

            # (b) 鼠标力场（引力/斥力，随距离衰减）
            if self.pointer_active:
                dx = self.pointer_x - p.x
                dy = self.pointer_y - p.y
                d2 = dx * dx + dy * dy
                if d2 < infl2:
                    d = math.sqrt(d2) or 1.0
                    fall = 1.0 - d / self.INFLUENCE
                    f = strength * fall / (d + self.SOFTEN)
                    ax += (dx / d) * f * sign
                    ay += (dy / d) * f * sign

            # (c) 速度积分 + 阻尼（惯性）
            p.vx = (p.vx + ax * df) * damp
            p.vy = (p.vy + ay * df) * damp

            # (d) 限速，防爆炸
            sp = math.hypot(p.vx, p.vy)
            if sp > self.MAX_SPEED:
                s = self.MAX_SPEED / sp
                p.vx *= s
                p.vy *= s
                sp = self.MAX_SPEED

            # (e) 位移
            p.x += p.vx * df
            p.y += p.vy * df

            # (f) 边界软反弹（保持粒子在屏内）
            if p.x < 0.0:
                p.x = 0.0
                p.vx *= -0.7
            elif p.x > self.width:
                p.x = self.width
                p.vx *= -0.7
            if p.y < 0.0:
                p.y = 0.0
                p.vy *= -0.7
            elif p.y > self.height:
                p.y = self.height
                p.vy *= -0.7

            # (g) 按速度映射到冷暖色相，并设透明度（越快越亮）
            tt = min(sp / self.COLOR_SPEED, 1.0)
            hue = self.cur_cold + (self.cur_warm - self.cur_cold) * tt
            p.color = hsl_to_rgb(hue, 1.0, 0.70)
            p.alpha = min(1.0, 0.55 + tt * 0.35)

        # 冲击波环推进与回收
        for i in range(len(self.shockwaves) - 1, -1, -1):
            sw = self.shockwaves[i]
            sw["r"] += 9.0 * df
            sw["life"] -= 0.02 * df
            if sw["life"] <= 0.0 or sw["r"] > sw["max"]:
                self.shockwaves.pop(i)

        self._time += dt

    # ------------------------------------------------------------------
    #  渲染与导出
    # ------------------------------------------------------------------

    @property
    def particles(self) -> List[Particle]:
        return self._sys.particles

    def get_particle_count(self) -> int:
        return len(self._sys.particles)

    def render(self, renderer):
        """交给任意 renderer 绘制（update 已完成逐粒子着色）。"""
        renderer.render(self._sys.particles)

    def save_image(self, path: str,
                   background: Optional["object"] = None,
                   bloom: bool = True) -> "object":
        """
        离线导出当前帧为 PNG（需 Pillow，即 ``pip install particlefx[pil]``）。

        Args:
            path      : 输出文件路径（.png）
            background: 可选 PIL Image 作为背景（RGB）；缺省用当前配色底色
            bloom     : 是否追加 Bloom 后处理
        Returns:
            合成后的 PIL Image（RGB）
        """
        from PIL import Image
        from ..renderers.pil_renderer import LayeredPILRenderer

        bg = background
        if bg is None:
            bg = Image.new("RGB", (self.width, self.height), self.current_bg())
        renderer = LayeredPILRenderer(
            width=self.width, height=self.height,
            bloom=bloom, bloom_radius=12,
            bloom_strength=0.6, bloom_threshold=120,
            glow_scale=2.5, glow_alpha=0.35, smooth_trail=False,
        )
        img = renderer.render_to_image(self._sys.particles, background=bg)
        img.save(path)
        return img


def create_light_field(width: int = 800, height: int = 600,
                       count: int = 1500, seed: int = 0,
                       palette_index: int = 0, **kwargs) -> LightFieldSystem:
    """创建并返回一个 LightFieldSystem 实例。"""
    return LightFieldSystem(width=width, height=height, count=count,
                            seed=seed, palette_index=palette_index, **kwargs)

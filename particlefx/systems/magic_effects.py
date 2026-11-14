"""
魔法效果系统（v1.2 新增）

为奇幻 / 魔法主题场景设计的粒子特效集合，支持：
- 奥术爆发（arcane_burst）：径向爆发的奥术粒子与余烬
- 符文环（rune_ring）：绕中心旋转的符文光环（持续一段时间）
- 治愈微光（healing）：向上飘浮的治愈光点
- 虚空触须（void_tendril）：受漩涡力场卷曲的虚空粒子
- 魔法飞弹（magic_missile）：从起点飞向目标并拖尾，命中后小爆

设计要点（与 v1.2 可复现内核一致）：
- 所有随机性都走 ``self._rng``（``random.Random`` 实例，由 ``seed`` 构造）；
  ``seed=None`` 时回退全局 ``random``（旧风格），保证向后兼容。
- 因此「相同 seed + 相同施法序列 + 相同 dt」可逐位复现，便于确定性测试与回放。

使用示例：
    from particlefx.systems.magic_effects import MagicEffectsManager

    magic = MagicEffectsManager(width=800, height=600, seed=42)
    magic.cast('arcane_burst', x=400, y=300)
    magic.cast('rune_ring', x=400, y=300)
    while running:
        magic.update(dt=0.016)
        magic.render(renderer)
"""

import random
import math

from ..core.particle import Particle


class MagicEffectsManager:
    """魔法特效管理器。

    通过 :meth:`cast` 触发效果，内部自动管理粒子生命周期；
    :meth:`update` 推进所有活动效果，:meth:`render` 交由渲染器绘制。
    """

    # 魔法调色板
    ARCANE = (150, 90, 255)        # 奥术紫
    ARCANE_BRIGHT = (210, 170, 255)
    VOID = (90, 40, 160)           # 虚空紫黑
    HEAL = (120, 255, 170)         # 治愈绿
    HEAL_BRIGHT = (220, 255, 230)
    EMBER = (255, 180, 90)         # 余烬橙

    # 效果元信息（供 EffectsFactory / UI 列举）
    MAGIC_PRESETS = {
        'arcane_burst': {
            'label': '奥术爆发',
            'description': '径向爆发的奥术粒子与余烬',
            'duration': 1.2,
        },
        'rune_ring': {
            'label': '符文环',
            'description': '绕中心旋转的符文光环',
            'duration': 2.4,
        },
        'healing': {
            'label': '治愈微光',
            'description': '向上飘浮的治愈光点',
            'duration': 1.6,
        },
        'void_tendril': {
            'label': '虚空触须',
            'description': '受漩涡力场卷曲的虚空粒子',
            'duration': 1.8,
        },
        'magic_missile': {
            'label': '魔法飞弹',
            'description': '飞向目标并拖尾，命中后小爆',
            'duration': 1.5,
        },
    }

    def __init__(self, width: int = 800, height: int = 600, seed: int = None):
        """
        Args:
            width, height: 场景尺寸（用于默认坐标 / 边界）。
            seed: 可选随机种子；给定 -> 确定性复现；None -> 全局 random（旧风格）。
        """
        self.width = width
        self.height = height

        self.particles: list = []
        # v1.2：可复现随机源；seed=None 回退全局 random。
        self._rng = random.Random(seed) if seed is not None else random

        # 活动效果队列（每个是带 age / duration 的状态字典）
        self._effects: list = []

    @property
    def particle_count(self) -> int:
        return len(self.particles)

    # ============================================================
    #  施法接口
    # ============================================================

    def cast(self, name: str, x: float, y: float, **kwargs):
        """触发一个魔法效果。

        Args:
            name: 效果名（见 ``MAGIC_PRESETS`` 的键）。
            x, y: 施法中心坐标。
            **kwargs: 效果专属参数（如 magic_missile 的 tx/ty / speed）。
        """
        name = name.lower().strip()
        preset = self.MAGIC_PRESETS.get(name)
        if preset is None:
            available = ', '.join(self.MAGIC_PRESETS.keys())
            raise ValueError(f"未知魔法效果 '{name}'，可选: {available}")

        fx = {
            'type': name,
            'x': x, 'y': y,
            'age': 0.0,
            'duration': kwargs.get('duration', preset['duration']),
            'data': kwargs,
            'missile_p': None,
        }
        self._spawn(fx)
        self._effects.append(fx)
        return fx

    # ============================================================
    #  生成
    # ============================================================

    def _spawn(self, fx: dict):
        t = fx['type']
        if t == 'arcane_burst':
            self._spawn_arcane_burst(fx)
        elif t == 'rune_ring':
            self._spawn_rune_ring(fx)
        elif t == 'healing':
            self._spawn_healing(fx)
        elif t == 'void_tendril':
            self._spawn_void_tendril(fx)
        elif t == 'magic_missile':
            self._spawn_magic_missile(fx)

    def _spawn_arcane_burst(self, fx: dict):
        x, y = fx['x'], fx['y']
        for _ in range(60):
            angle = self._rng.uniform(0, 2 * math.pi)
            speed = self._rng.uniform(2.0, 9.0)
            roll = self._rng.random()
            color = self.ARCANE if roll < 0.6 else (
                self.ARCANE_BRIGHT if roll < 0.9 else self.EMBER)
            p = Particle(
                x=x + self._rng.uniform(-4, 4),
                y=y + self._rng.uniform(-4, 4),
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                life=self._rng.uniform(0.4, 1.0), max_life=1.0,
                size=self._rng.uniform(2.0, 5.0),
                color=color, alpha=1.0,
                mass=self._rng.uniform(0.6, 1.4),
            )
            self.particles.append(p)
        # 少量余烬（更慢、更长寿命）
        for _ in range(20):
            angle = self._rng.uniform(0, 2 * math.pi)
            speed = self._rng.uniform(0.5, 3.0)
            p = Particle(
                x=x, y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed - 1.0,
                life=self._rng.uniform(0.8, 1.4), max_life=1.4,
                size=self._rng.uniform(1.0, 2.5),
                color=self.EMBER, alpha=1.0, mass=1.0,
            )
            self.particles.append(p)

    def _spawn_rune_ring(self, fx: dict):
        x, y = fx['x'], fx['y']
        count = 48
        radius = fx['data'].get('radius', 70.0)
        for i in range(count):
            angle = (i / count) * 2 * math.pi
            px = x + math.cos(angle) * radius
            py = y + math.sin(angle) * radius
            tx_ = -math.sin(angle)
            ty_ = math.cos(angle)
            tangential = self._rng.uniform(20.0, 40.0)
            p = Particle(
                x=px, y=py,
                vx=tx_ * tangential,
                vy=ty_ * tangential,
                life=fx['duration'], max_life=fx['duration'],
                size=self._rng.uniform(2.0, 4.0),
                color=self._rng.choice([self.ARCANE, self.ARCANE_BRIGHT]),
                alpha=1.0, mass=1.0,
            )
            self.particles.append(p)

    def _spawn_healing(self, fx: dict):
        x, y = fx['x'], fx['y']
        spread = fx['data'].get('spread', 40.0)
        for _ in range(50):
            p = Particle(
                x=x + self._rng.uniform(-spread, spread),
                y=y + self._rng.uniform(-spread * 0.3, spread * 0.3),
                vx=self._rng.uniform(-0.6, 0.6),
                vy=-self._rng.uniform(1.0, 3.5),   # 向上飘
                life=self._rng.uniform(0.8, 1.6), max_life=1.6,
                size=self._rng.uniform(2.0, 5.0),
                color=self._rng.choice([self.HEAL, self.HEAL_BRIGHT]),
                alpha=1.0, mass=0.8,
            )
            self.particles.append(p)

    def _spawn_void_tendril(self, fx: dict):
        x, y = fx['x'], fx['y']
        for _ in range(70):
            angle = self._rng.uniform(0, 2 * math.pi)
            r = self._rng.uniform(10.0, 50.0)
            px = x + math.cos(angle) * r
            py = y + math.sin(angle) * r
            p = Particle(
                x=px, y=py,
                vx=math.cos(angle) * self._rng.uniform(1.0, 3.0),
                vy=math.sin(angle) * self._rng.uniform(1.0, 3.0),
                life=self._rng.uniform(0.6, 1.8), max_life=1.8,
                size=self._rng.uniform(1.5, 4.0),
                color=self._rng.choice([self.VOID, self.ARCANE]),
                alpha=1.0, mass=self._rng.uniform(0.5, 1.2),
            )
            # 记录漩涡中心，update 中绕其卷曲
            p._swirl_cx = x
            p._swirl_cy = y
            self.particles.append(p)

    def _spawn_magic_missile(self, fx: dict):
        x, y = fx['x'], fx['y']
        tx = fx['data'].get('tx', x)
        ty = fx['data'].get('ty', y - 200.0)
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy) or 1.0
        speed = fx['data'].get('speed', 12.0)
        p = Particle(
            x=x, y=y,
            vx=dx / dist * speed,
            vy=dy / dist * speed,
            life=fx['duration'], max_life=fx['duration'],
            size=4.0,
            color=self.ARCANE_BRIGHT, alpha=1.0, mass=1.0,
            trail_enabled=True, trail_max_length=12,
        )
        fx['target'] = (tx, ty)
        fx['missile_p'] = p
        self.particles.append(p)

    # ============================================================
    #  更新
    # ============================================================

    def update(self, dt: float):
        """推进所有活动效果与粒子。"""
        # 1) 持续类效果在生命周期内补充粒子（符文环 / 治愈 / 虚空触须）
        for fx in self._effects:
            fx['age'] += dt
            t = fx['type']
            if t in ('rune_ring', 'healing', 'void_tendril'):
                if fx['age'] < fx['duration'] * 0.85:
                    self._emit_continuous(fx, dt)

        # 2) 推进粒子：漩涡卷曲 + 积分 + 寿命；清理死亡粒子
        alive = []
        for p in self.particles:
            if not p.is_alive():
                continue
            if getattr(p, '_swirl_cx', None) is not None:
                # 绕漩涡中心施加切向加速度（确定性）
                dx = p.x - p._swirl_cx
                dy = p.y - p._swirl_cy
                tl = math.hypot(dx, dy) or 1.0
                swirl = 50.0
                p.vx += (-dy / tl) * swirl * dt
                p.vy += (dx / tl) * swirl * dt
            p.update(dt)
            if p.is_alive():
                alive.append(p)
        self.particles = alive

        # 3) 魔法飞弹：抵达 / 超时后引爆并清理活动效果
        for fx in self._effects:
            if fx['type'] == 'magic_missile' and not fx.get('detonated', False):
                mp = fx.get('missile_p')
                reached = mp is None or not mp.is_alive() or fx['age'] >= fx['duration']
                if reached:
                    fx['detonated'] = True
                    tx, ty = fx['target']
                    self._detonate_missile(tx, ty)

        # 4) 清理已结束的活动效果（飞弹在引爆后保留至 duration 结束再清）
        self._effects = [
            fx for fx in self._effects
            if fx['age'] < fx['duration']
        ]

    def _emit_continuous(self, fx: dict, dt: float):
        t = fx['type']
        x, y = fx['x'], fx['y']
        # 每帧补充数量与 dt 成正比，保证不同帧率下密度一致
        n = max(1, int(20 * dt))
        if t == 'healing':
            for _ in range(n):
                p = Particle(
                    x=x + self._rng.uniform(-40, 40),
                    y=y + self._rng.uniform(-12, 12),
                    vx=self._rng.uniform(-0.6, 0.6),
                    vy=-self._rng.uniform(1.0, 3.0),
                    life=self._rng.uniform(0.8, 1.6), max_life=1.6,
                    size=self._rng.uniform(2.0, 5.0),
                    color=self._rng.choice([self.HEAL, self.HEAL_BRIGHT]),
                    alpha=1.0, mass=0.8,
                )
                self.particles.append(p)
        elif t == 'rune_ring':
            for _ in range(n):
                angle = self._rng.uniform(0, 2 * math.pi)
                radius = fx['data'].get('radius', 70.0)
                px = x + math.cos(angle) * radius
                py = y + math.sin(angle) * radius
                tx_ = -math.sin(angle)
                ty_ = math.cos(angle)
                tangential = self._rng.uniform(20.0, 40.0)
                p = Particle(
                    x=px, y=py,
                    vx=tx_ * tangential, vy=ty_ * tangential,
                    life=self._rng.uniform(0.3, 0.8), max_life=0.8,
                    size=self._rng.uniform(2.0, 4.0),
                    color=self._rng.choice([self.ARCANE, self.ARCANE_BRIGHT]),
                    alpha=1.0, mass=1.0,
                )
                self.particles.append(p)
        elif t == 'void_tendril':
            for _ in range(n):
                angle = self._rng.uniform(0, 2 * math.pi)
                r = self._rng.uniform(10.0, 50.0)
                p = Particle(
                    x=x + math.cos(angle) * r, y=y + math.sin(angle) * r,
                    vx=math.cos(angle) * self._rng.uniform(1.0, 3.0),
                    vy=math.sin(angle) * self._rng.uniform(1.0, 3.0),
                    life=self._rng.uniform(0.6, 1.8), max_life=1.8,
                    size=self._rng.uniform(1.5, 4.0),
                    color=self._rng.choice([self.VOID, self.ARCANE]),
                    alpha=1.0, mass=self._rng.uniform(0.5, 1.2),
                )
                p._swirl_cx = x
                p._swirl_cy = y
                self.particles.append(p)

    def _detonate_missile(self, x: float, y: float):
        for _ in range(24):
            angle = self._rng.uniform(0, 2 * math.pi)
            speed = self._rng.uniform(2.0, 7.0)
            p = Particle(
                x=x, y=y,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                life=self._rng.uniform(0.3, 0.8), max_life=0.8,
                size=self._rng.uniform(2.0, 4.0),
                color=self._rng.choice([self.ARCANE_BRIGHT, self.EMBER]),
                alpha=1.0, mass=1.0,
            )
            self.particles.append(p)

    # ============================================================
    #  渲染 / 查询
    # ============================================================

    def render(self, renderer):
        """渲染魔法粒子（委托给渲染器）。"""
        if hasattr(renderer, 'render'):
            renderer.render(self.particles)

    def render_pygame(self, screen):
        """pygame 快速渲染（无 pygame 时静默跳过）。"""
        try:
            import pygame
            for p in self.particles:
                if not p.is_alive():
                    continue
                alpha = int(max(0, min(255, p.alpha * 255)))
                color = (*p.color[:3], alpha) if len(p.color) == 3 else p.color
                size = max(1, int(p.size))
                pos = (int(p.x), int(p.y))
                if alpha > 10:
                    surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                    pygame.draw.circle(surf, color, (size, size), size)
                    screen.blit(surf, (pos[0] - size, pos[1] - size))
        except ImportError:
            pass

    def list_presets(self) -> dict:
        """返回所有魔法效果元信息。"""
        return dict(self.MAGIC_PRESETS)


def create_magic(width: int = 800, height: int = 600,
                 seed: int = None) -> MagicEffectsManager:
    """创建并返回一个 MagicEffectsManager 实例。"""
    return MagicEffectsManager(width=width, height=height, seed=seed)

"""
天气系统

可切换的天气效果系统，支持：
- 晴天（无粒子）
- 雨天
- 雪天
- 风暴（雨+强风）
- 渐变过渡

使用示例：
    from systems.weather_system import WeatherSystem

    weather = WeatherSystem(width=800, height=600)
    weather.set_weather('rain')
    while running:
        weather.update(dt=0.016)
        weather.render(renderer)
"""

import random
import math

try:
    from core.particle import Particle, ForceField
except ImportError:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.particle import Particle, ForceField


class WeatherSystem:
    """
    天气管理系统

    在不同天气模式间切换，管理所有天气粒子的生命周期。
    """

    # 天气配置预设
    WEATHER_CONFIGS = {
        'clear': {
            'enabled': False,
            'description': '晴天',
        },
        'rain': {
            'enabled': True,
            'particle_type': 'rain',
            'density': 1.0,
            'wind': 1.0,
            'gravity': 25.0,
            'speed_range': (18, 28),
            'life_multiplier': 0.9,
            'size_range': (1, 2),
            'color': (150, 180, 255),
            'spread': 0.08,
            'description': '小雨',
        },
        'heavy_rain': {
            'enabled': True,
            'particle_type': 'rain',
            'density': 2.5,
            'wind': 3.0,
            'gravity': 30.0,
            'speed_range': (22, 35),
            'life_multiplier': 0.8,
            'size_range': (1, 3),
            'color': (130, 160, 235),
            'spread': 0.12,
            'description': '暴雨',
        },
        'snow': {
            'enabled': True,
            'particle_type': 'snow',
            'density': 0.6,
            'wind': 0.5,
            'gravity': 1.5,
            'speed_range': (0.8, 2.5),
            'life_multiplier': 2.5,
            'size_range': (1.5, 4.5),
            'colors': [(255, 255, 255), (230, 240, 255), (200, 220, 255)],
            'spread': math.pi * 0.25,
            'description': '小雪',
        },
        'heavy_snow': {
            'enabled': True,
            'particle_type': 'snow',
            'density': 1.8,
            'wind': 1.5,
            'gravity': 1.0,
            'speed_range': (0.5, 2.0),
            'life_multiplier': 3.5,
            'size_range': (2, 6),
            'colors': [(255, 255, 255), (230, 240, 255), (210, 225, 250)],
            'spread': math.pi * 0.35,
            'description': '大雪',
        },
        'storm': {
            'enabled': True,
            'particle_type': 'mixed',  # 雨为主 + 偶尔雪花
            'density': 2.0,
            'wind': 7.0,
            'gravity': 30.0,
            'speed_range': (20, 38),
            'life_multiplier': 0.7,
            'size_range': (1, 3),
            'color': (120, 150, 220),
            'spread': 0.15,
            'lightning_chance': 0.003,  # 每帧闪电概率
            'description': '风暴',
        },
    }

    def __init__(self,
                 width: int = 800,
                 height: int = 600,
                 initial_weather: str = 'clear'):
        """
        初始化天气系统

        Args:
            width: 场景宽度
            height: 场景高度
            initial_weather: 初始天气类型
        """
        self.width = width
        self.height = height

        self.particles: list = []
        self.emitters: list = []
        self.force_fields: list = []

        self._current_weather = None
        self._target_config = None
        self._transition_progress = 1.0
        self.lightning_active = False
        self.lightning_timer = 0.0

        self.set_weather(initial_weather)

    @property
    def current_weather(self) -> str:
        """当前天气名称"""
        if self._current_weather is None:
            return 'clear'
        return self._current_weather

    @property
    def description(self) -> str:
        """当前天气描述文字"""
        config = self.WEATHER_CONFIGS.get(self._current_weather, {})
        return config.get('description', '')

    @property
    def particle_count(self) -> int:
        return len(self.particles)

    def set_weather(self, weather_name: str, transition_time: float = 0):
        """
        切换天气

        Args:
            weather_name: 天气名称 ('clear', 'rain', 'heavy_rain',
                          'snow', 'heavy_snow', 'storm')
            transition_time: 过渡时间（秒），0 为立即切换
        """
        weather_name = weather_name.lower().strip()
        if weather_name not in self.WEATHER_CONFIGS:
            available = ', '.join(self.WEATHER_CONFIGS.keys())
            raise ValueError(f"未知天气 '{weather_name}'，可选: {available}")

        old_weather = self._current_weather
        self._current_weather = weather_name
        self._target_config = self.WEATHER_CONFIGS[weather_name]

        # 清除旧状态并重建
        self.particles.clear()
        self.emitters.clear()
        self.force_fields.clear()

        if not self._target_config['enabled']:
            return

        self._build_emitters()

    def _build_emitters(self):
        """根据当前配置构建发射器"""
        cfg = self._target_config
        density = cfg['density']

        num_sources = max(5, int(self.width / (80 / density)))

        for i in range(num_sources):
            x = (i + random.uniform(-0.2, 0.2)) * self.width / num_sources

            particle_type = cfg.get('particle_type', 'rain')

            # 风暴模式下混合雨和雪
            if particle_type == 'mixed':
                actual_type = 'snow' if random.random() < 0.08 else 'rain'
            else:
                actual_type = particle_type

            colors = cfg.get('colors', [cfg.get('color', (255, 255, 255))])

            emitter_data = {
                'x': x,
                'y': -5,
                'rate': (25 if actual_type == 'rain' else 8) * density,
                'spread': cfg.get('spread', 0.1),
                'speed_range': cfg['speed_range'],
                'life_base': self.height / cfg['speed_range'][0] * cfg.get('life_multiplier', 1.0),
                'size_range': cfg['size_range'],
                'colors': colors,
                'type': actual_type,
                '_accumulated': 0.0,
            }
            self.emitters.append(emitter_data)

        # 力场
        self.force_fields.append(
            ForceField(field_type='gravity', strength=cfg.get('gravity', 10.0))
        )
        wind = cfg.get('wind', 0)
        if abs(wind) > 0.01:
            self.force_fields.append(
                ForceField(field_type='wind', strength=wind)
            )

    def update(self, dt: float):
        """
        更新天气系统

        Args:
            dt: 时间步长
        """
        cfg = self._target_config
        if cfg and not cfg['enabled']:
            return

        # 发射新粒子
        for em in self.emitters:
            em['_accumulated'] += dt
            rate = em['rate']
            num_to_emit = int(em['_accumulated'] * rate)
            if num_to_emit > 0:
                em['_accumulated'] -= num_to_emit / rate
                for _ in range(num_to_emit):
                    speed = random.uniform(*em['speed_range'])
                    spread = em['spread']
                    angle = random.uniform(-spread, spread)

                    # 雪的飘动感
                    if em['type'] == 'snow':
                        angle += math.pi / 2  # 主要向下
                        speed = random.uniform(*em['speed_range'])

                    p = Particle(
                        x=em['x'] + random.uniform(-10, 10),
                        y=em['y'],
                        vx=math.cos(angle) * speed if em['type'] != 'snow' else random.uniform(-0.5, 0.5),
                        vy=speed if em['type'] != 'snow' else random.uniform(*em['speed_range']),
                        life=random.uniform(em['life_base'] * 0.8, em['life_base']),
                        max_life=em['life_base'],
                        size=random.uniform(*em['size_range']),
                        color=random.choice(em['colors']) if isinstance(em['colors'], list) else em['colors'],
                        alpha=1.0,
                    )
                    self.particles.append(p)

        # 更新粒子
        alive_particles = []
        for p in self.particles:
            if not p.is_alive():
                continue

            # 应用力场
            for ff in self.force_fields:
                fx, fy = ff.get_force(p)
                p.vx += fx * dt
                p.vy += fy * dt

            # 雪花额外水平漂移
            if cfg and cfg.get('particle_type') in ('snow', 'heavy_snow'):
                p.vx += math.sin(p.y * 0.02 + p.x * 0.01) * 0.05

            p.update(dt)

            if p.is_alive():
                alive_particles.append(p)

        self.particles = alive_particles

        # 闪电检测（风暴模式）
        if cfg and cfg.get('lightning_chance'):
            if random.random() < cfg['lightning_chance']:
                self.lightning_active = True
                self.lightning_timer = random.uniform(0.05, 0.15)
            elif self.lightning_active:
                self.lightning_timer -= dt
                if self.lightning_timer <= 0:
                    self.lightning_active = False

    def render(self, renderer):
        """
        渲染天气粒子

        Args:
            renderer: 渲染器实例
        """
        if hasattr(renderer, 'render'):
            renderer.render(self.particles)

    def render_pygame(self, screen):
        """
        pygame 快速渲染方法

        Args:
            screen: pygame Surface
        """
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


def create_weather(width: int = 800, height: int = 600,
                   weather: str = 'clear') -> WeatherSystem:
    """创建并返回一个 WeatherSystem 实例"""
    return WeatherSystem(width=width, height=height, initial_weather=weather)

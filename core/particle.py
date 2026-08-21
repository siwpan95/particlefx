"""
粒子系统核心模块 v3.0 — 全功能增强版

新增特性（v3.0）：
1. 发射形状模块（Shape Module）：point/sphere/cone/circle/rectangle/edge
2. 拖尾模块（Trails）：位置历史队列 + 渲染连线
3. 碰撞模块（Collision）：AABB 检测 + 反弹 + 回调
4. 力场扩展：vortex/turbulence/magnet
5. 事件系统：on_spawn/on_death/on_collision/on_lifetime_tick
6. 光照模块：点光源/方向光/环境光/Blinn-Phong着色
7. LOD 系统：距离相机越远粒子越少
8. 音频联动：粒子事件触发音效
9. 粒子网格体：绑定 Surface/Image 替代纯色圆
10. 纹理动画：精灵表帧动画

设计理念：
- 所有新特性均为可选，不使用则零开销
- 向后兼容：旧代码无需修改即可运行
"""

import numpy as np
import random
import math
from dataclasses import dataclass, field, MISSING
from typing import List, Tuple, Optional, Callable, Dict, Any


# ================================================================
#  曲线/插值辅助函数
# ================================================================

def lerp_color(c1: Tuple[int, int, int],
               c2: Tuple[int, int, int],
               t: float) -> Tuple[int, int, int]:
    """颜色线性插值 (t: 0~1)"""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def evaluate_curve(curve: List[Tuple[float, float]], t: float) -> float:
    """计算曲线在 t 处的值（线性插值）"""
    if not curve:
        return 1.0
    if len(curve) == 1:
        return curve[0][1]
    if t <= curve[0][0]:
        return curve[0][1]
    if t >= curve[-1][0]:
        return curve[-1][1]
    for i in range(len(curve) - 1):
        t0, v0 = curve[i]
        t1, v1 = curve[i + 1]
        if t0 <= t <= t1:
            s = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return v0 + (v1 - v0) * s
    return curve[-1][1]


def evaluate_color_curve(curve: List[Tuple[float, Tuple[int, int, int]]],
                         t: float) -> Tuple[int, int, int]:
    """计算颜色曲线在 t 处的值"""
    if not curve:
        return (255, 255, 255)
    if len(curve) == 1:
        return curve[0][1]
    if t <= curve[0][0]:
        return curve[0][1]
    if t >= curve[-1][0]:
        return curve[-1][1]
    for i in range(len(curve) - 1):
        t0, c0 = curve[i]
        t1, c1 = curve[i + 1]
        if t0 <= t <= t1:
            s = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return lerp_color(c0, c1, s)
    return curve[-1][1]


def evaluate_curve_vel(curve: List[Tuple[float, Tuple[float, float]]],
                       t: float) -> Tuple[float, float]:
    """计算速度倍率曲线在 t 处的值"""
    if not curve:
        return (1.0, 1.0)
    if len(curve) == 1:
        return curve[0][1]
    if t <= curve[0][0]:
        return curve[0][1]
    if t >= curve[-1][0]:
        return curve[-1][1]
    for i in range(len(curve) - 1):
        t0, v0 = curve[i]
        t1, v1 = curve[i + 1]
        if t0 <= t <= t1:
            s = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return (v0[0] + (v1[0] - v0[0]) * s,
                    v0[1] + (v1[1] - v0[1]) * s)
    return curve[-1][1]


# ================================================================
#  发射形状模块（Shape Module）
# ================================================================

@dataclass
class EmitterShape:
    """
    发射器形状 — 控制粒子的初始位置和发射方向
    
    支持的形状：
    - point      : 点发射（默认）
    - sphere     : 球形表面或体内发射
    - cone       : 锥形发射
    - circle     : 圆形边缘或面内发射
    - rectangle  : 矩形区域内发射
    - edge       : 边缘线段发射
    """
    shape_type: str = 'point'  # point | sphere | cone | circle | rectangle | edge
    
    # --- 形状参数 ---
    radius: float = 10.0          # 球/圆锥/圆 的半径
    height: float = 20.0          # 圆锥高度
    width: float = 40.0           # 矩形宽度
    rect_height: float = 30.0     # 矩形高度
    edge_length: float = 30.0     # 边缘长度
    edge_angle: float = 0.0       # 边缘角度（弧度）
    
    # --- 控制参数 ---
    emit_from_shell: bool = False   # True=从表面发射, False=从体积内发射（sphere/circle）
    emit_from_volume: bool = True   # True=从体内发射（仅 sphere/circle 有效）
    randomize_direction: bool = True  # 是否根据形状随机化初始速度方向
    
    def sample_position(self, base_x: float, base_y: float,
                        speed: float, spread: float) -> Tuple[float, float, float, float]:
        """
        采样发射位置和方向
        
        Returns:
            (x, y, vx, vy) : 初始位置和速度分量
        """
        if self.shape_type == 'point':
            return self._sample_point(base_x, base_y, speed, spread)
        elif self.shape_type == 'sphere':
            return self._sample_sphere(base_x, base_y, speed, spread)
        elif self.shape_type == 'cone':
            return self._sample_cone(base_x, base_y, speed, spread)
        elif self.shape_type == 'circle':
            return self._sample_circle(base_x, base_y, speed, spread)
        elif self.shape_type == 'rectangle':
            return self._sample_rectangle(base_x, base_y, speed, spread)
        elif self.shape_type == 'edge':
            return self._sample_edge(base_x, base_y, speed, spread)
        else:
            return self._sample_point(base_x, base_y, speed, spread)

    def _sample_point(self, bx, by, speed, spread):
        """点发射"""
        angle = random.uniform(-spread, spread)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        return bx, by, vx, vy

    def _sample_sphere(self, bx, by, speed, spread):
        """球形发射"""
        r = self.radius
        if self.emit_from_shell:
            # 从球面采样
            angle_2d = random.uniform(0, 2 * math.pi)
            px = bx + r * math.cos(angle_2d)
            py = by + r * math.sin(angle_2d)
        else:
            # 从球内采样（均匀分布）
            angle_2d = random.uniform(0, 2 * math.pi)
            dist = r * math.sqrt(random.uniform(0, 1))  # sqrt 让分布更均匀
            px = bx + dist * math.cos(angle_2d)
            py = by + dist * math.sin(angle_2d)
        
        # 方向：从中心向外辐射（可加 spread 扰动）
        dx = px - bx
        dy = py - by
        dist = math.sqrt(dx*dx + dy*dy) if (dx*dx + dy*dy) > 1e-6 else 1.0
        
        if self.randomize_direction:
            base_angle = math.atan2(dy, dx)
            angle = base_angle + random.uniform(-spread, spread)
        else:
            angle = random.uniform(-spread, spread)
        
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        return px, py, vx, vy

    def _sample_cone(self, bx, by, speed, spread):
        """锥形发射（向上/指定方向的锥体）"""
        # 锥顶在 (bx, by)，开口朝上
        r_at_base = self.radius
        h = self.height
        
        # 在底面上随机取一点
        cone_angle = random.uniform(0, 2 * math.pi)
        r_sample = r_at_base * math.sqrt(random.uniform(0, 1))
        
        px = bx + r_sample * math.cos(cone_angle)
        py = by + h  # 底部（假设锥尖在上）
        
        # 方向：从锥尖向底面扩散
        dx = px - bx
        dy = py - by
        dist = math.sqrt(dx*dx + dy*dy) if (dx*dx + dy*dy) > 1e-6 else 1.0
        
        if self.randomize_direction:
            base_angle = math.atan2(dy, dx)
            angle = base_angle + random.uniform(-spread * 0.5, spread * 0.5)
        else:
            angle = math.pi / 2 + random.uniform(-spread, spread)
        
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        return px, py, vx, vy

    def _sample_circle(self, bx, by, speed, spread):
        """圆形发射（平面内的圆）"""
        r = self.radius
        if self.emit_from_shell:
            angle_2d = random.uniform(0, 2 * math.pi)
            px = bx + r * math.cos(angle_2d)
            py = by + r * math.sin(angle_2d)
        else:
            angle_2d = random.uniform(0, 2 * math.pi)
            dist = r * math.sqrt(random.uniform(0, 1))
            px = bx + dist * math.cos(angle_2d)
            py = by + dist * math.sin(angle_2d)
        
        if self.randomize_direction:
            base_angle = random.uniform(0, 2 * math.pi)
            angle = base_angle + random.uniform(-spread, spread)
        else:
            angle = random.uniform(-spread, spread)
        
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        return px, py, vx, vy

    def _sample_rectangle(self, bx, by, speed, spread):
        """矩形区域发射"""
        hw = self.width / 2
        hh = self.rect_height / 2
        px = bx + random.uniform(-hw, hw)
        py = by + random.uniform(-hh, hh)
        angle = random.uniform(-spread, spread)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        return px, py, vx, vy

    def _sample_edge(self, bx, by, speed, spread):
        """边缘线段发射"""
        half_len = self.edge_length / 2
        cos_a = math.cos(self.edge_angle)
        sin_a = math.sin(self.edge_angle)
        
        offset = random.uniform(-half_len, half_len)
        px = bx + offset * cos_a
        py = by + offset * sin_a
        
        angle = random.uniform(-spread, spread)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        return px, py, vx, vy


# ================================================================
#  碰撞模块（Collision Module）
# ================================================================

@dataclass
class CollisionBox:
    """
    AABB 碰撞盒
    
    用于定义场景中的碰撞边界/障碍物
    """
    x: float       # 左上角 X
    y: float       # 左上角 Y
    width: float   # 宽度
    height: float  # 高度
    restitution: float = 0.5   # 反弹系数 (0=完全吸收, 1=完全反弹)
    friction: float = 0.1      # 摩擦系数 (0=无摩擦, 1=完全停止)
    name: str = ""             # 标识名称
    
    def contains_point(self, px: float, py: float) -> bool:
        """检查点是否在碰撞盒内部"""
        return (self.x <= px <= self.x + self.width and 
                self.y <= py <= self.y + self.height)


@dataclass
class CollisionBoundary:
    """
    世界边界碰撞（屏幕/场景边缘）
    
    mode:
    - 'wrap'     : 环绕（从一边出去另一边进来）
    - 'bounce'   : 反弹
    - 'destroy'  : 消亡
    - 'none'     : 无限制（默认）
    """
    left: float = 0.0
    top: float = 0.0
    right: float = 800.0
    bottom: float = 600.0
    mode: str = 'bounce'   # wrap | bounce | destroy | none
    restitution: float = 0.6
    friction: float = 0.05


# ================================================================
#  Particle  — 单个粒子（v3.0 全功能版）
# ================================================================

@dataclass
class Particle:
    """
    单个粒子（v3.0 全功能版）
    
    新增特性：
    - 拖尾（trail_history）
    - 网格体绑定（mesh/sprite）
    - 纹理动画（texture_frame）
    - 事件标记（collision_data）
    - 光照属性（emissive/light_reactive）
    """

    # --- 基础属性 ---
    x: float
    y: float
    vx: float
    vy: float
    life: float       # 剩余生命值 (0~1，0 = 死亡)
    max_life: float   # 最大生命值（秒）
    size: float       # 当前大小（像素）
    color: Tuple[int, int, int]  # 当前 RGB 颜色
    alpha: float = 1.0  # 透明度 (0~1)

    # --- 旋转 ---
    rotation: float = 0.0
    angular_velocity: float = 0.0

    # --- 质量 ---
    mass: float = 1.0

    # --- 生命周期曲线 ---
    color_over_lifetime: Optional[List[Tuple[float, Tuple[int, int, int]]]] = None
    size_over_lifetime: Optional[List[Tuple[float, float]]] = None
    velocity_over_lifetime: Optional[List[Tuple[float, Tuple[float, float]]]] = None

    # --- 拖尾（新增）---
    trail_enabled: bool = False
    trail_history: List[Tuple[float, float]] = field(default_factory=list)
    trail_max_length: int = 10

    # --- 网格体/精灵（新增）---
    mesh_surface: Any = None         # pygame.Surface 或 PIL Image
    mesh_mode: str = 'circle'        # 'circle' | 'sprite' | 'image'
    
    # --- 纹理动画（新增）---
    texture_frames: Optional[List[Any]] = None  # 帧列表 [Surface, Surface, ...]
    texture_current_frame: int = 0
    texture_fps: float = 12.0
    texture_time_accum: float = 0.0

    # --- 光照（新增）---
    emissive: Tuple[int, int, int] = (0, 0, 0)   # 自发光颜色
    emissive_strength: float = 0.0                # 自发光强度
    light_reactive: bool = True                   # 是否响应外部光照

    # --- 子发射器 & 事件 ---
    _sub_emit_on_death: Optional[Any] = None
    _sub_emit_on_birth: Optional[Any] = None
    
    # --- 碰撞数据（新增）---
    collision_data: Optional[Dict[str, Any]] = None  # 上次碰撞信息

    def is_alive(self) -> bool:
        return self.life > 0.0

    def update(self, dt: float):
        """更新粒子状态（完整管线）"""
        # 1. Velocity over Lifetime
        if self.velocity_over_lifetime:
            t = max(0.0, min(1.0, 1.0 - self.life / self.max_life))
            vx_mult, vy_mult = evaluate_curve_vel(self.velocity_over_lifetime, t)
            self.vx *= (0.95 + 0.05 * vx_mult)
            self.vy *= (0.95 + 0.05 * vy_mult)

        # 2. 更新位置
        self.x += self.vx * dt
        self.y += self.vy * dt

        # 3. 记录拖尾位置
        if self.trail_enabled and self.trail_max_length > 0:
            self.trail_history.append((self.x, self.y))
            if len(self.trail_history) > self.trail_max_length:
                self.trail_history.pop(0)

        # 4. 更新旋转
        self.rotation += self.angular_velocity * dt

        # 5. 更新生命值
        self.life -= dt / self.max_life

        # 6. Color over Lifetime
        if self.color_over_lifetime:
            t = max(0.0, min(1.0, 1.0 - self.life / self.max_life))
            self.color = evaluate_color_curve(self.color_over_lifetime, t)

        # 7. Size over Lifetime
        if self.size_over_lifetime:
            t = max(0.0, min(1.0, 1.0 - self.life / self.max_life))
            self.size = evaluate_curve(self.size_over_lifetime, t)

        # 8. Alpha 淡出
        self.alpha = max(0.0, self.life)

        # 9. 纹理动画帧更新
        if self.texture_frames and len(self.texture_frames) > 1:
            self.texture_time_accum += dt
            frame_interval = 1.0 / self.texture_fps
            while self.texture_time_accum >= frame_interval:
                self.texture_time_accum -= frame_interval
                self.texture_current_frame = (
                    (self.texture_current_frame + 1) % len(self.texture_frames)
                )

    def enable_trail(self, length: int = 10):
        """启用拖尾"""
        self.trail_enabled = True
        self.trail_max_length = length
        self.trail_history = [(self.x, self.y)]

    def get_trail_points(self) -> List[Tuple[float, float]]:
        """获取拖尾坐标列表（含当前位置）"""
        points = list(self.trail_history)
        points.append((self.x, self.y))
        return points

    def get_current_texture(self) -> Any:
        """获取当前纹理帧"""
        if self.texture_frames and 0 <= self.texture_current_frame < len(self.texture_frames):
            return self.texture_frames[self.texture_current_frame]
        return None

    def reset(self, *, x: float, y: float, vx: float, vy: float,
              life: float, max_life: float, size: float,
              color: Tuple[int, int, int], **kwargs) -> 'Particle':
        """
        将粒子重置为初始状态（对象池复用，零分配）。

        所有字段先恢复 dataclass 默认值，再应用传入的新值；
        **kwargs 可覆盖任意可选字段（alpha / rotation / 曲线 / 拖尾等）。
        供 ParticlePool.acquire() 调用，也可单独使用。
        """
        for _f in __import__('dataclasses').fields(type(self)):
            _name = _f.name
            if _name in ('x', 'y', 'vx', 'vy', 'life', 'max_life', 'size', 'color'):
                continue
            if _f.default_factory is not MISSING:
                setattr(self, _name, _f.default_factory())
            elif _f.default is not MISSING:
                setattr(self, _name, _f.default)
            else:
                setattr(self, _name, None)
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.life = life
        self.max_life = max_life
        self.size = size
        self.color = color
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self


# ================================================================
#  ParticlePool  — 粒子对象池（v5.0 新增）
# ================================================================

class ParticlePool:
    """
    粒子对象池 — 复用 Particle 实例，消除高频创建/回收的 GC 压力。

    用法（ParticleSystem 内部自动启用，也可独立使用）：
        pool = ParticlePool(capacity=1000)
        p = pool.acquire(x=..., y=..., vx=..., vy=..., life=..., max_life=..., size=..., color=...)
        # 用完归还
        pool.release(p)

    特性：
    - 容量固定，acquire 在池耗尽时返回 None（调用方自行丢弃）
    - release 时清理拖尾历史 / 纹理帧 / 子发射器等引用型资源，防止内存滞留
    - reset() 基于 dataclass 字段元数据，新增字段自动获得正确默认值
    """

    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self._free: List[Particle] = [
            Particle(x=0.0, y=0.0, vx=0.0, vy=0.0,
                     life=0.0, max_life=1.0, size=1.0,
                     color=(255, 255, 255))
            for _ in range(self.capacity)
        ]
        self._active_count = 0

    @property
    def available(self) -> int:
        """剩余可分配数量"""
        return len(self._free)

    @property
    def active_count(self) -> int:
        """当前已借出（存活）数量"""
        return self._active_count

    def acquire(self, **kwargs) -> Optional[Particle]:
        """
        借出一个粒子。参数同 Particle.reset()。
        池耗尽时返回 None。
        """
        if not self._free:
            return None
        p = self._free.pop()
        p.reset(**kwargs)
        self._active_count += 1
        return p

    def release(self, particle: Particle):
        """归还粒子（清理引用型资源后放回空闲列表）"""
        if particle.trail_history:
            particle.trail_history.clear()
        particle.texture_frames = None
        particle.mesh_surface = None
        particle._sub_emit_on_death = None
        particle._sub_emit_on_birth = None
        particle.collision_data = None
        self._active_count = max(0, self._active_count - 1)
        self._free.append(particle)

    def __len__(self) -> int:
        return self._active_count


# ================================================================
#  力场（增强版：支持 vortex / turbulence / magnet）
# ================================================================

class ForceField:
    """
    力场（v3.0 增强）
    
    类型：
    - gravity    : 重力（全局向下）
    - wind       : 风力（水平）
    - attraction : 点吸引/排斥
    - noise      : 噪声场
    - vortex     : 漩涡力场（新增）
    - turbulence : 湍流场（新增）
    - magnet     : 磁场/极性力场（新增）
    """

    def __init__(self,
                 field_type: str = 'gravity',
                 x: float = 0.0, y: float = 0.0,
                 strength: float = 1.0,
                 # --- 噪声场参数 ---
                 noise_scale: float = 50.0,
                 noise_strength: float = 1.0,
                 noise_octaves: int = 2,
                 # --- 漩涡参数 ---
                 vortex_clockwise: bool = True,
                 vortex_falloff: float = 0.01,
                 # --- 湍流参数 ---
                 turb_scale: float = 30.0,
                 turb_strength: float = 2.0,
                 # --- 磁场参数 ---
                 magnet_polarity: float = 1.0,   # 正=吸引, 负=排斥
                 magnet_range: float = 200.0,
                 ):
        self.field_type = field_type
        self.x = x
        self.y = y
        self.strength = strength
        
        # 噪声参数
        self.noise_scale = noise_scale
        self.noise_strength = noise_strength
        self.noise_octaves = noise_octaves
        
        # 漩涡参数
        self.vortex_clockwise = vortex_clockwise
        self.vortex_falloff = vortex_falloff
        
        # 湍流参数
        self.turb_scale = turb_scale
        self.turb_strength = turb_strength
        
        # 磁场参数
        self.magnet_polarity = magnet_polarity
        self.magnet_range = magnet_range
        
        self._noise_time_offset = random.uniform(0, 1000)

    def get_force(self, particle_or_x, time_or_y: float = 0.0,
                  time: float = 0.0) -> Tuple[float, float]:
        """
        计算受力。
        
        支持两种调用方式：
        - get_force(particle, time)         — 传入 Particle 对象（原有方式）
        - get_force(x, y, time)             — 直接传入坐标（解耦方式，无需 Particle 对象）
        
        示例：
            ff.get_force(particle)
            ff.get_force(particle, t=0.5)
            ff.get_force(100.0, 200.0)
            fx, fy = ff.get_force(rocket['x'], rocket['y'])
        """
        if isinstance(particle_or_x, (int, float)):
            # 坐标模式：get_force(x, y, time)
            x_val = float(particle_or_x)
            y_val = float(time_or_y)
            t_val = time
            # 构造最小协议对象（仅需 x, y, vx, vy, mass）
            class _P:
                __slots__ = ('x', 'y', 'vx', 'vy', 'mass')
                def __init__(self, x, y):
                    self.x = x; self.y = y
                    self.vx = 0.0; self.vy = 0.0; self.mass = 1.0
            p = _P(x_val, y_val)
        else:
            # 粒子对象模式：get_force(particle, time)
            p = particle_or_x
            t_val = time_or_y

        handlers = {
            'gravity': self._force_gravity,
            'wind': self._force_wind,
            'attraction': self._force_attraction,
            'noise': lambda p, t: self._force_noise(p, t),
            'vortex': self._force_vortex,
            'turbulence': self._force_turbulence,
            'magnet': self._force_magnet,
        }
        handler = handlers.get(self.field_type)
        if handler:
            return handler(p, t_val)
        return (0.0, 0.0)

    def _force_gravity(self, p, t):
        return (0.0, self.strength)

    def _force_wind(self, p, t):
        return (self.strength, 0.0)

    def _force_attraction(self, p, t):
        dx = self.x - p.x
        dy = self.y - p.y
        dist_sq = dx * dx + dy * dy
        if dist_sq < 1e-6:
            return (0.0, 0.0)
        dist = math.sqrt(dist_sq)
        force_mag = self.strength / (1.0 + dist * 0.01)
        return (force_mag * dx / dist, force_mag * dy / dist)

    def _force_noise(self, p, t):
        sx = self.noise_scale
        t_val = t + self._noise_time_offset
        fx_total, fy_total = 0.0, 0.0
        for octave in range(max(1, self.noise_octaves)):
            freq = 2.0 ** octave / sx
            amp = self.noise_strength / (2.0 ** octave)
            nx = p.x * freq + t_val * 0.5 + octave * 10.0
            ny = p.y * freq + t_val * 0.3 + octave * 7.0
            fx_total += amp * math.sin(nx) * math.cos(ny * 0.7)
            fy_total += amp * math.cos(ny) * math.sin(nx * 0.7)
        return (fx_total, fy_total)

    def _force_vortex(self, p, t):
        """漩涡力场 — 绕力场中心做圆周运动"""
        dx = p.x - self.x
        dy = p.y - self.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-6:
            return (0.0, 0.0)
        
        # 切向力（垂直于半径方向）
        direction = 1.0 if self.vortex_clockwise else -1.0
        # 切向量：(−dy, dx) 归一化
        tx = -dy / dist
        ty = dx / dist
        
        # 强度随距离衰减
        falloff = math.exp(-dist * self.vortex_falloff)
        mag = self.strength * falloff * direction
        
        return (mag * tx, mag * ty)

    def _force_turbulence(self, p, t):
        """湍流场 — 随机快速变化的方向扰动"""
        sx = self.turb_scale
        t_val = t + self._noise_time_offset * 1.7
        
        # 多层高频噪声产生湍流效果
        fx, fy = 0.0, 0.0
        for i in range(3):
            freq = (3.0 + i * 2.0) / sx
            amp = self.turb_strength / (1.5 ** i)
            phase = t_val * (2.0 + i) + i * 13.37
            
            fx += amp * math.sin(p.x * freq * (1 + i * 0.3) + phase)
            fy += amp * math.cos(p.y * freq * (1 + i * 0.4) + phase * 1.3)
            
            # 加入交叉项让湍流更混沌
            fx += amp * 0.5 * math.cos(p.y * freq * 0.7 + phase * 0.9)
            fy += amp * 0.5 * math.sin(p.x * freq * 0.8 + phase * 1.1)
        
        return (fx, fy)

    def _force_magnet(self, p, t):
        """磁场/极性力场 — 类似磁铁吸引/排斥，有作用范围限制"""
        dx = self.x - p.x
        dy = self.y - p.y
        dist_sq = dx * dx + dy * dy
        dist = math.sqrt(dist_sq)
        
        if dist < 1e-6 or dist > self.magnet_range:
            return (0.0, 0.0)
        
        # 极性控制方向：正=吸引，负=排斥
        direction = self.magnet_polarity
        
        # 平方反比 + 范围衰减
        range_factor = 1.0 - (dist / self.magnet_range)  # 线性衰减到范围边缘
        force_mag = self.strength * direction * range_factor / (1.0 + dist_sq * 0.0001)
        
        return (force_mag * dx / dist, force_mag * dy / dist)

    def set_position(self, x: float, y: float):
        self.x = x
        self.y = y


# ================================================================
#  ParticleAttractor  — 粒子吸引子（v5.0 新增）
# ================================================================

class ParticleAttractor(ForceField):
    """
    粒子吸引子 — 有限范围、可跟随目标的吸引/排斥力场。

    与 ForceField('attraction') 的差异：
    - 有 radius 作用范围（范围外受力为 0）
    - 支持线性 / 反距离两种衰减
    - 支持 follow 回调：每帧自动跟随目标位置（如鼠标、角色、另一个粒子）
    - polarity 为负时变成排斥子

    用法：
        attr = ParticleAttractor(x=400, y=300, strength=8.0, radius=250.0)
        attr.follow = lambda: pygame.mouse.get_pos()   # 跟随鼠标
        system.add_force_field(attr)
        # 每帧（可选）：attr.update(dt)  由 ParticleSystem.update 自动调用
    """

    def __init__(self,
                 x: float = 0.0, y: float = 0.0,
                 strength: float = 5.0,
                 radius: float = 200.0,
                 falloff: str = 'linear',       # 'linear' | 'inverse'
                 polarity: float = 1.0,          # 正=吸引, 负=排斥
                 follow: Optional[Callable[[], Tuple[float, float]]] = None):
        super().__init__(field_type='attraction', x=x, y=y, strength=strength)
        self.radius = max(1e-3, float(radius))
        self.falloff = falloff if falloff in ('linear', 'inverse') else 'linear'
        self.polarity = 1.0 if polarity >= 0 else -1.0
        self.follow = follow

    def update(self, dt: float = 0.0):
        """跟随目标位置（设置 follow 回调后由 ParticleSystem.update 每帧调用）"""
        if self.follow is not None:
            self.x, self.y = self.follow()

    def get_force(self, particle_or_x, time_or_y: float = 0.0,
                  time: float = 0.0) -> Tuple[float, float]:
        """计算受力（支持 Particle 对象或 (x, y) 坐标两种调用方式）"""
        if isinstance(particle_or_x, (int, float)):
            px, py = float(particle_or_x), float(time_or_y)
        else:
            px, py = particle_or_x.x, particle_or_x.y

        dx = self.x - px
        dy = self.y - py
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-6 or dist > self.radius:
            return (0.0, 0.0)

        if self.falloff == 'inverse':
            f = self.strength / (1.0 + dist * 0.05)
        else:
            # 线性衰减：中心最强，边缘归零（过渡更平滑）
            f = self.strength * (1.0 - dist / self.radius)
        f *= self.polarity
        return (f * dx / dist, f * dy / dist)


# ================================================================
#  NoiseField  — 独立噪声场
# ================================================================

class NoiseField:
    """独立噪声场（精细控制版本）"""

    def __init__(self,
                 strength: float = 1.0,
                 scale: float = 50.0,
                 octaves: int = 2,
                 seed: Optional[int] = None):
        self.strength = strength
        self.scale = scale
        self.octaves = max(1, octaves)
        self.seed = seed if seed is not None else random.randint(0, 100000)
        self._time = 0.0

    def update(self, dt: float):
        self._time += dt

    def get_force(self, particle: Particle) -> Tuple[float, float]:
        sx = self.scale
        t = self._time
        fx, fy = 0.0, 0.0
        for octave in range(self.octaves):
            freq = (2.0 ** octave) / sx
            amp = self.strength / (2.0 ** octave)
            phase = self.seed * 0.1 + octave * 31.0
            nx = particle.x * freq + t * 0.4 + phase
            ny = particle.y * freq + t * 0.25 + phase * 0.7
            fx += amp * math.sin(nx + math.cos(ny)) * math.cos(ny * 0.6)
            fy += amp * math.cos(ny + math.sin(nx)) * math.sin(nx * 0.6)
        return (fx, fy)


# ================================================================
#  光照模块（Lighting Module）
# ================================================================

@dataclass
class LightSource:
    """
    光源（用于粒子光照计算）
    
    类型：
    - ambient   : 环境光（均匀照明，无方向）
    - point     : 点光源（从一点向四周发光）
    - directional: 方向光（平行光，如阳光）
    """
    light_type: str = 'ambient'
    
    # --- 通用属性 ---
    color: Tuple[int, int, int] = (255, 255, 255)
    intensity: float = 1.0
    
    # --- 点光源/方向光专用 ---
    x: float = 0.0
    y: float = 0.0
    # 方向光角度（弧度），0=向右, π/2=向下
    direction: float = math.pi / 2
    # 点光源衰减范围
    range_: float = 500.0
    # 衰减类型: 'none' | 'linear' | 'inverse_square'
    attenuation: str = 'inverse_square'

    def get_illumination(self, px: float, py: float) -> Tuple[Tuple[int, int, int], float]:
        """
        计算某点的光照结果
        
        Returns:
            (color, intensity) : 照明后的颜色和强度
        """
        if self.light_type == 'ambient':
            return self.color, self.intensity
        
        elif self.light_type == 'point':
            dx = px - self.x
            dy = py - self.y
            dist = math.sqrt(dx * dx + dy * dy)
            
            if dist > self.range_:
                return (0, 0, 0), 0.0
            
            # 衰减计算
            if self.attenuation == 'none':
                atten = 1.0
            elif self.attenuation == 'linear':
                atten = max(0.0, 1.0 - dist / self.range_)
            else:  # inverse_square
                atten = 1.0 / (1.0 + dist * dist * 0.001)
            
            final_intensity = self.intensity * atten
            
            # 返回带强度的颜色（后续 Blinn-Phong 合成）
            final_color = tuple(int(c * atten) for c in self.color)
            return final_color, final_intensity
        
        elif self.light_type == 'directional':
            # 方向光无衰减
            return self.color, self.intensity
        
        return (0, 0, 0), 0.0


@dataclass
class BlinnPhongParams:
    """
    Blinn-Phong 着色参数
    
    基于 Kajiya 渲染方程的简化模型：
    FinalColor = Ambient + Diffuse + Specular + Emissive
    """
    ambient_color: Tuple[int, int, int] = (40, 40, 50)    # 环境光颜色
    ambient_strength: float = 0.3                          # 环境光强度
    
    diffuse_strength: float = 0.7                          # 漫反射强度（粗糙度相关）
    specular_strength: float = 0.4                         # 高光强度
    shininess: float = 32.0                                # 高光指数（越大越集中）
    
    # 菲尼尔效应（Fresnel）— 视角越斜反射越强
    fresnel_power: float = 3.0                            # 菲尼尔指数
    fresnel_strength: float = 0.2                          # 菲尼尔强度


def apply_lighting_to_particle(
    particle: Particle,
    lights: List[LightSource],
    blinn_phong: Optional[BlinnPhongParams] = None,
    camera_pos: Tuple[float, float] = (400.0, 300.0),
) -> Tuple[int, int, int, float]:
    """
    对单个粒子应用 Blinn-Phong 光照计算
    
    基于 Kajiya 渲染方程简化：
    Lo(p, ωo) = Le(p, ωo) + ∫ fr(p, ωi, ωo) * Li(p, ωi) * (n · ωi) dωi
    
    工程简化为：
    FinalColor = Emissive + Ambient + Σ(Diffuse_i + Specular_i)
    
    Args:
        particle   : 粒子实例
        lights     : 光源列表
        blinn_phong: Blinn-Phong 参数（None 使用默认值）
        camera_pos : 相机位置（用于高光计算）
    
    Returns:
        (r, g, b, alpha) : 最终颜色和透明度
    """
    if blinn_phong is None:
        bp = BlinnPhongParams()
    else:
        bp = blinn_phong
    
    # 基础颜色（粒子当前颜色）
    base_r, base_g, base_b = particle.color
    
    # === 1. 自发光（Emissive）===
    er, eg, eb = particle.emissive
    emissive_str = particle.emissive_strength
    
    # === 2. 环境光（Ambient）===
    ar, ag, ab = bp.ambient_color
    amb_r = ar * bp.ambient_strength
    amb_g = ag * bp.ambient_strength
    amb_b = ab * bp.ambient_strength
    
    # === 3. 各光源贡献（Diffuse + Specular）===
    total_diffuse_r, total_diffuse_g, total_diffuse_b = 0.0, 0.0, 0.0
    total_specular_r, total_specular_g, total_specular_b = 0.0, 0.0, 0.0
    
    for light in lights:
        lc, li = light.get_illumination(particle.x, particle.y)
        lr, lg, lb = lc
        
        if li <= 0.001:
            continue
        
        # ---- Diffuse（漫反射）----
        # 假设粒子法线朝向观察者（2D 简化为正面朝向相机）
        # N · L = cos(θ)，对于 2D 粒子近似为光源方向与视线方向的夹角
        if light.light_type == 'directional':
            # 方向光：用光线方向计算
            lx = math.cos(light.direction)
            ly = math.sin(light.direction)
            ndotl = abs(ly)  # 2D 近似：法线朝上时与方向光的夹角
        elif light.light_type == 'point':
            dx = particle.x - light.x
            dy = particle.y - light.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-6:
                ndotl = 1.0
            else:
                # 粒子法线近似为朝向观察者
                cx, cy = camera_pos
                nx = (cx - particle.x) / max(dist, 1.0) if dist > 1e-6 else 0
                ny = (cy - particle.y) / max(dist, 1.0) if dist > 1e-6 else -1
                ldx = -dx / dist  # 光线指向粒子的方向
                ldy = -dy / dist
                ndotl = max(0, nx * ldx + ny * ldy)
        else:
            ndotl = 1.0  # ambient 不参与漫反射
        
        diff_factor = ndotl * bp.diffuse_strength * li
        total_diffuse_r += lr * diff_factor
        total_diffuse_g += lg * diff_factor
        total_diffuse_b += lb * diff_factor
        
        # ---- Specular（高光 / Blinn-Phong 半向量法）----
        # H = normalize(L + V)  （半向量）
        # spec = pow(max(N·H, shininess))
        cx, cy = camera_pos
        vdx = cx - particle.x
        vdy = cy - particle.y
        v_dist = math.sqrt(vdx*vdx + vdy*vdy)
        if v_dist > 1e-6:
            vx_n = vdx / v_dist
            vy_n = vdy / v_dist
        else:
            vx_n, vy_n = 0.0, -1.0
        
        if light.light_type == 'directional':
            lx_n = math.cos(light.direction)
            ly_n = math.sin(light.direction)
        elif light.light_type == 'point':
            dx = particle.x - light.x
            dy = particle.y - light.y
            d = math.sqrt(dx*dx + dy*dy)
            if d > 1e-6:
                lx_n = -dx / d
                ly_n = -dy / d
            else:
                lx_n, ly_n = 0.0, -1.0
        else:
            lx_n, ly_n = 0.0, 0.0
        
        # 半向量 H
        hx = lx_n + vx_n
        hy = ly_n + vy_n
        h_len = math.sqrt(hx*hx + hy*hy)
        if h_len > 1e-6:
            hx /= h_len
            hy /= h_len
        
        # 粒子法线（朝向相机）
        ndoth = max(0.0, vy_n)  # 简化：N·H ≈ 视线Y分量
        spec = (ndoth ** bp.shininess) * bp.specular_strength * li
        total_specular_r += lr * spec
        total_specular_g += lg * spec
        total_specular_b += lb * spec
    
    # ---- Fresnel（菲尼尔效应）----
    # F = F0 + (1 - F0) * (1 - V·N)^fresnel_power
    cx, cy = camera_pos
    vdx = cx - particle.x
    vdy = cy - particle.y
    v_dist = math.sqrt(vdx*vdx + vdy*vdy)
    if v_dist > 1e-6:
        view_dot_n = abs(vdy) / v_dist  # 简化视角因子
    else:
        view_dot_n = 1.0
    fresnel = bp.fresnel_strength * ((1.0 - view_dot_n) ** bp.fresnel_power)
    
    # === 最终合成 ===
    final_r = int(min(255, base_r * (amb_r + total_diffuse_r) + total_specular_r + er * emissive_str + fresnel * 200))
    final_g = int(min(255, base_g * (amb_g + total_diffuse_g) + total_specular_g + eg * emissive_str + fresnel * 200))
    final_b = int(min(255, base_b * (amb_b + total_diffuse_b) + total_specular_b + eb * emissive_str + fresnel * 200))
    
    return (final_r, final_g, final_b, particle.alpha)


blinn_phong_params = BlinnPhongParams()  # 默认全局参数


# ================================================================
#  Emitter  — 粒子发射器（v3.0 增强版）
# ================================================================

class Emitter:
    """
    粒子发射器（v3.0 增强版）
    
    新增特性：
    - 发射形状（shape: EmitterShape）
    - 拖尾配置（trail_*）
    - 网格体绑定（mesh_*）
    - 纹理动画（texture_*）
    - 光照属性（emissive_*）
    """

    def __init__(self,
                 x: float, y: float,
                 rate: float = 10.0,
                 spread: float = 0.5,
                 direction: float = 0.0,
                 speed: float = 1.0,
                 life: float = 1.0,
                 size: float = 1.0,
                 color: Tuple[int, int, int] = (255, 255, 255),
                 # --- 旋转 & 质量 ---
                 rotation_range: Tuple[float, float] = (0.0, 0.0),
                 angular_velocity_range: Tuple[float, float] = (0.0, 0.0),
                 mass_range: Tuple[float, float] = (1.0, 1.0),
                 # --- 生命周期曲线 ---
                 color_over_lifetime=None,
                 size_over_lifetime=None,
                 velocity_over_lifetime=None,
                 # --- 子发射器 ---
                 sub_emitter_on_death=None,
                 sub_emitter_on_birth=None,
                 # === v3.0 新增 ===
                 # --- 发射形状 ---
                 shape: Optional[EmitterShape] = None,
                 # --- 拖尾 ---
                 trail_enabled: bool = False,
                 trail_length: int = 8,
                 # --- 网格体/精灵 ---
                 mesh_surface: Any = None,
                 mesh_mode: str = 'circle',
                 # --- 纹理动画 ---
                 texture_frames: Optional[List[Any]] = None,
                 texture_fps: float = 12.0,
                 # --- 光照 ---
                 emissive: Tuple[int, int, int] = (0, 0, 0),
                 emissive_strength: float = 0.0,
                 # --- 对象池工厂（v5.0，ParticleSystem 内部自动注入）---
                 particle_factory: Optional[Callable[..., Particle]] = None,
                 ):
        self.x = x
        self.y = y
        self.rate = rate
        self.spread = spread
        self.direction = direction
        self.speed = speed
        self.life = life
        self.size = size
        self.color = color

        self.rotation_range = rotation_range
        self.angular_velocity_range = angular_velocity_range
        self.mass_range = mass_range

        self.color_over_lifetime = color_over_lifetime
        self.size_over_lifetime = size_over_lifetime
        self.velocity_over_lifetime = velocity_over_lifetime

        self.sub_emitter_on_death = sub_emitter_on_death
        self.sub_emitter_on_birth = sub_emitter_on_birth
        
        # === v3.0 新增 ===
        self.shape = shape or EmitterShape('point')
        self.trail_enabled = trail_enabled
        self.trail_length = trail_length
        self.mesh_surface = mesh_surface
        self.mesh_mode = mesh_mode
        self.texture_frames = texture_frames
        self.texture_fps = texture_fps
        self.emissive = emissive
        self.emissive_strength = emissive_strength
        self.particle_factory = particle_factory

        self.accumulated_time = 0.0

    def emit(self, dt: float,
             audio_data: Optional[dict] = None,
             current_time: float = 0.0) -> List[Particle]:
        """发射粒子（支持形状采样）"""
        self.accumulated_time += dt

        current_rate = self.rate
        if audio_data:
            energy = audio_data.get('rms', 1.0)
            current_rate *= (1.0 + energy * 2.0)

        num_to_emit = int(self.accumulated_time * current_rate)
        self.accumulated_time -= num_to_emit / current_rate if current_rate > 0 else 0.0

        new_particles = []
        for _ in range(num_to_emit):
            # 使用形状采样获取位置和方向
            px, py, vx, vy = self.shape.sample_position(
                self.x, self.y, self.speed, self.spread
            )
            
            # 方向偏移：将发射锥旋转到指定方向
            if self.direction != 0.0:
                cos_a = math.cos(self.direction)
                sin_a = math.sin(self.direction)
                vx, vy = vx * cos_a - vy * sin_a, vx * sin_a + vy * cos_a
            
            rotation = random.uniform(*self.rotation_range)
            angular_velocity = random.uniform(*self.angular_velocity_range)
            mass = random.uniform(*self.mass_range)

            p_kwargs = dict(
                x=px, y=py, vx=vx, vy=vy,
                life=1.0, max_life=self.life,
                size=self.size, color=self.color, alpha=1.0,
                rotation=rotation, angular_velocity=angular_velocity, mass=mass,
                color_over_lifetime=self.color_over_lifetime,
                size_over_lifetime=self.size_over_lifetime,
                velocity_over_lifetime=self.velocity_over_lifetime,
                _sub_emit_on_death=self.sub_emitter_on_death,
                _sub_emit_on_birth=self.sub_emitter_on_birth,
                trail_enabled=self.trail_enabled,
                trail_max_length=self.trail_length,
                mesh_surface=self.mesh_surface,
                mesh_mode=self.mesh_mode,
                texture_frames=self.texture_frames,
                texture_fps=self.texture_fps,
                emissive=self.emissive,
                emissive_strength=self.emissive_strength,
            )

            # v5.0：对象池工厂（None = 直接 new，行为与旧版一致）
            if self.particle_factory is not None:
                p = self.particle_factory(**p_kwargs)
                if p is None:
                    continue  # 池耗尽，丢弃本次发射
            else:
                p = Particle(**p_kwargs)
            
            # 启用拖尾
            if self.trail_enabled:
                p.enable_trail(self.trail_length)
            
            new_particles.append(p)

        return new_particles

    def set_position(self, x: float, y: float):
        self.x = x
        self.y = y


# ================================================================
#  ParticleSystem  — 粒子系统管理器（v3.0 全功能版）
# ================================================================

class ParticleSystem:
    """
    粒子系统管理器（v3.0 全功能版）
    
    新增：
    - 碰撞检测（CollisionBox + CollisionBoundary）
    - 事件系统（完整回调链）
    - LOD 系统（距离相机粒子缩减）
    - 光照集成
    - 音频联动
    """

    def __init__(self, max_particles: int = 1000,
                 use_pool: bool = False,
                 use_spatial_hash: bool = False,
                 spatial_hash_cell_size: float = 64.0):
        self.max_particles = max_particles
        self.particles: List[Particle] = []
        self.emitters: List[Emitter] = []
        self.force_fields: List[ForceField] = []
        self.noise_fields: List[NoiseField] = []
        
        # === v5.0：对象池（可选，零开销）===
        self.use_pool = bool(use_pool)
        self._pool: Optional[ParticlePool] = None
        if self.use_pool:
            self._pool = ParticlePool(capacity=max_particles)
        
        # === v5.0：空间哈希（可选，零开销）===
        self.use_spatial_hash = bool(use_spatial_hash)
        self._grid: Optional[SpatialHashGrid] = None
        self._grid_boxes_signature: Optional[tuple] = None
        self._grid_needs_particles = False
        if self.use_spatial_hash:
            from core.spatial_hash import SpatialHashGrid
            self._grid = SpatialHashGrid(cell_size=spatial_hash_cell_size)
        
        # 碰撞系统
        self.collision_boxes: List[CollisionBox] = []
        self.boundary: Optional[CollisionBoundary] = None
        
        # 光照系统
        self.lights: List[LightSource] = []
        self.blinn_phong: BlinnPhongParams = BlinnPhongParams()
        self.camera_pos: Tuple[float, float] = (400.0, 300.0)
        self.lighting_enabled: bool = False
        
        # LOD 系统
        self.lod_enabled: bool = False
        self.lod_distances: List[float] = [100.0, 300.0, 600.0]  # 各级LOD切换距离
        self.lod_multipliers: List[float] = [1.0, 0.5, 0.25, 0.1]  # 各级粒子数量倍率
        
        # 时间管理
        self._time = 0.0
        
        # 子发射器待处理队列
        self._pending_particles: List[Particle] = []
        
        # === 事件系统（完整回调链）===
        self.on_particle_birth: Optional[Callable[[Particle], None]] = None
        self.on_particle_death: Optional[Callable[[Particle], None]] = None
        self.on_particle_collision: Optional[Callable[[Particle, CollisionBox, Dict], None]] = None
        self.on_lifetime_tick: Optional[Callable[[Particle, float], None]] = None
        
        # === 音频联动 ===
        self.audio_enabled: bool = False
        self.audio_birth_sound: Optional[Any] = None
        self.audio_death_sound: Optional[Any] = None
        self.audio_collision_sound: Optional[Any] = None
        
        # === v5.0：粒子间作用力（enable_pairwise_force 开启，默认零开销）===
        self._pairwise_strength: float = 0.0
        self._pairwise_radius: float = 0.0
        self._pairwise_repulsion: bool = False
        
        # === v5.0：粒子光源（enable_particle_lights 开启，默认零开销）===
        self._particle_lights_enabled: bool = False
        self._max_particle_lights: int = 0
        self._min_emissive_strength: float = 0.0
        self.particle_lights: List[LightSource] = []

    # --- 组件添加 ---

    def add_emitter(self, emitter: Emitter):
        # v5.0：启用对象池时注入工厂，发射路径自动复用粒子实例
        if self._pool is not None and emitter.particle_factory is None:
            emitter.particle_factory = self._pool.acquire
        self.emitters.append(emitter)

    def add_force_field(self, force_field: ForceField):
        self.force_fields.append(force_field)

    def add_noise_field(self, noise_field: NoiseField):
        self.noise_fields.append(noise_field)

    def add_collision_box(self, box: CollisionBox):
        self.collision_boxes.append(box)

    def set_boundary(self, boundary: CollisionBoundary):
        self.boundary = boundary

    def add_light(self, light: LightSource):
        self.lights.append(light)
        if not self.lighting_enabled:
            self.lighting_enabled = True

    # --- 碰撞检测 ---

    def _check_collisions(self, p: Particle) -> Optional[Dict[str, Any]]:
        """
        检查并处理粒子碰撞
        
        Returns:
            碰撞信息字典（None 表示无碰撞）
        """
        result = None
        
        # 1. 与碰撞盒检测（v5.0：空间哈希加速，仅检查粒子所在格子的候选盒）
        if self._grid is not None:
            candidates = self._grid.query_point(p.x, p.y)
            if self._grid_needs_particles:
                # 网格中混有粒子与碰撞盒，过滤出碰撞盒
                boxes = [b for b in candidates if isinstance(b, CollisionBox)]
            else:
                boxes = candidates
        else:
            boxes = self.collision_boxes
        for box in boxes:
            if box.contains_point(p.x, p.y):
                # 计算最近边并反弹
                cx = box.x + box.width / 2
                cy = box.y + box.height / 2
                
                # 确定从哪一边进入
                dx_left = abs(p.x - box.x)
                dx_right = abs(p.x - (box.x + box.width))
                dy_top = abs(p.y - box.y)
                dy_bottom = abs(p.y - (box.y + box.height))
                
                min_d = min(dx_left, dx_right, dy_top, dy_bottom)
                
                if min_d == dx_left:
                    p.vx = -p.vx * box.restitution
                    p.x = box.x - 0.1
                elif min_d == dx_right:
                    p.vx = -p.vx * box.restitution
                    p.x = box.x + box.width + 0.1
                elif min_d == dy_top:
                    p.vy = -p.vy * box.restitution
                    p.y = box.y - 0.1
                else:
                    p.vy = -p.vy * box.restitution
                    p.y = box.y + box.height + 0.1
                
                # 应用摩擦力
                p.vx *= (1.0 - box.friction)
                p.vy *= (1.0 - box.friction)
                
                result = {
                    'type': 'box',
                    'name': box.name,
                    'position': (p.x, p.y),
                    'velocity': (p.vx, p.vy),
                    'normal': (dx_left if min_d==dx_left else (-dx_right if min_d==dx_right else 0),
                               dy_top if min_d==dy_top else (-dy_bottom if min_d==dy_bottom else 0)),
                }
                p.collision_data = result
                
                # 触发碰撞事件
                if self.on_particle_collision:
                    self.on_particle_collision(p, box, result)
                break
        
        # 2. 边界检测
        if self.boundary and result is None:
            b = self.boundary
            collided = False
            
            if b.mode == 'bounce':
                if p.x < b.left:
                    p.x = b.left
                    p.vx = -p.vx * b.restitution
                    p.vy *= (1.0 - b.friction)
                    collided = True
                elif p.x > b.right:
                    p.x = b.right
                    p.vx = -p.vx * b.restitution
                    p.vy *= (1.0 - b.friction)
                    collided = True
                if p.y < b.top:
                    p.y = b.top
                    p.vy = -p.vy * b.restitution
                    p.vx *= (1.0 - b.friction)
                    collided = True
                elif p.y > b.bottom:
                    p.y = b.bottom
                    p.vy = -p.vy * b.restitution
                    p.vx *= (1.0 - b.friction)
                    collided = True
                    
            elif b.mode == 'wrap':
                if p.x < b.left:
                    p.x = b.right
                elif p.x > b.right:
                    p.x = b.left
                if p.y < b.top:
                    p.y = b.bottom
                elif p.y > b.bottom:
                    p.y = b.top
                    
            elif b.mode == 'destroy':
                if (p.x < b.left or p.x > b.right or 
                    p.y < b.top or p.y > b.bottom):
                    p.life = 0.0
                    collided = True
            
            if collided:
                result = {'type': 'boundary', 'mode': b.mode}
                p.collision_data = result
        
        return result

    # --- LOD 计算 ---

    def _calc_lod_multiplier(self) -> float:
        """基于相机距离计算 LOD 倍率"""
        if not self.lod_enabled:
            return 1.0
        
        # 取粒子群中心到相机的距离
        if not self.particles:
            return self.lod_multipliers[-1]
        
        cx = sum(p.x for p in self.particles) / len(self.particles)
        cy = sum(p.y for p in self.particles) / len(self.particles)
        dist = math.sqrt((cx - self.camera_pos[0])**2 + (cy - self.camera_pos[1])**2)
        
        for i, lod_dist in enumerate(self.lod_distances):
            if dist < lod_dist:
                return self.lod_multipliers[min(i, len(self.lod_multipliers)-1)]
        return self.lod_multipliers[-1]

    # --- 子发射器触发 ---

    def _trigger_sub_emitters(self, trigger_type: str, parent_particle: Particle):
        system_to_trigger = None
        if trigger_type == 'birth' and parent_particle._sub_emit_on_birth:
            system_to_trigger = parent_particle._sub_emit_on_birth
        elif trigger_type == 'death' and parent_particle._sub_emit_on_death:
            system_to_trigger = parent_particle._sub_emit_on_death

        if system_to_trigger:
            for emitter in system_to_trigger.emitters:
                emitter.set_position(parent_particle.x, parent_particle.y)
                sub_particles = emitter.emit(dt=0.016)
                for sp in sub_particles[:10]:
                    if len(self.particles) + len(self._pending_particles) < self.max_particles:
                        self._pending_particles.append(sp)

    # --- 主更新循环 ---

    def update(self, dt: float, audio_data: Optional[dict] = None):
        """更新粒子系统（完整管线）"""
        self._time += dt
        
        # LOD 倍率
        lod_mult = self._calc_lod_multiplier()
        
        # 更新噪声场时间
        for nf in self.noise_fields:
            nf.update(dt)

        # --- 发射阶段 ---
        for emitter in self.emitters:
            # LOD 减少发射量
            if self.lod_enabled and lod_mult < 1.0 and random.random() > lod_mult:
                continue
            
            new_particles = emitter.emit(dt, audio_data, self._time)
            for p in new_particles:
                if len(self.particles) + len(self._pending_particles) < self.max_particles:
                    # birth 事件
                    if p._sub_emit_on_birth:
                        self._trigger_sub_emitters('birth', p)
                    if self.on_particle_birth:
                        self.on_particle_birth(p)
                    
                    # 音频联动
                    if self.audio_enabled and self.audio_birth_sound:
                        self._play_sound(self.audio_birth_sound, volume=lod_mult)
                    
                    self.particles.append(p)

        # 添加待处理粒子
        self.particles.extend(self._pending_particles)
        self._pending_particles.clear()

        # v5.0：空间哈希重建（仅在启用且脏时）
        if self._grid is not None:
            sig = tuple((id(b), b.x, b.y, b.width, b.height) for b in self.collision_boxes)
            if self._grid_needs_particles or sig != self._grid_boxes_signature:
                self._rebuild_spatial_grid()
                self._grid_boxes_signature = sig

        # --- 更新阶段 ---
        # v5.0：更新带 follow 的吸引子位置（每帧跟踪目标）
        for ff in self.force_fields:
            if isinstance(ff, ParticleAttractor) and ff.follow is not None:
                ff.update(dt)

        alive_particles = []
        for p in self.particles:
            # 预检查：处理上一帧已死亡的粒子
            if not p.is_alive():
                # death 事件
                if p._sub_emit_on_death:
                    self._trigger_sub_emitters('death', p)
                if self.on_particle_death:
                    self.on_particle_death(p)
                if self.audio_enabled and self.audio_death_sound:
                    self._play_sound(self.audio_death_sound, volume=0.3)
                if self._pool is not None:
                    self._pool.release(p)
                continue

            # 应用力场
            for ff in self.force_fields:
                fx, fy = ff.get_force(p, self._time)
                p.vx += (fx / p.mass) * dt
                p.vy += (fy / p.mass) * dt

            # 应用噪声场
            for nf in self.noise_fields:
                fx, fy = nf.get_force(p)
                p.vx += (fx / p.mass) * dt
                p.vy += (fy / p.mass) * dt

            # v5.0：粒子间作用力（未开启时零开销）
            self._apply_pairwise_force_to(p, dt)

            # 碰撞检测
            self._check_collisions(p)

            # 更新粒子状态
            p.update(dt)

            # lifetime_tick 事件（每帧触发，包含即将死亡的粒子）
            if self.on_lifetime_tick:
                self.on_lifetime_tick(p, p.life)

            # 后检查：本帧内死亡（出生即死）的粒子也触发事件
            if not p.is_alive():
                if p._sub_emit_on_death:
                    self._trigger_sub_emitters('death', p)
                if self.on_particle_death:
                    self.on_particle_death(p)
                if self.audio_enabled and self.audio_death_sound:
                    self._play_sound(self.audio_death_sound, volume=0.3)
                if self._pool is not None:
                    self._pool.release(p)
                continue

            if p.is_alive():
                alive_particles.append(p)

        self.particles = alive_particles

        # v5.0：粒子光源更新（未开启时零开销）
        self._update_particle_lights()

    def _play_sound(self, sound, volume: float = 1.0):
        """播放音效（需要 pygame.mixer 已初始化）"""
        try:
            import pygame
            sound.set_volume(volume)
            sound.play()
        except Exception:
            pass  # pygame 未初始化时静默失败

    def render(self, renderer: 'Renderer'):
        """渲染（自动应用光照）"""
        # v5.0：静态光源 + 激活的粒子光源
        active_lights = list(self.lights)
        if self._particle_lights_enabled:
            active_lights += [l for l in self.particle_lights if l.intensity > 0.001]
        if self.lighting_enabled and active_lights:
            # 光照渲染模式：先对每个粒子计算最终颜色
            lit_particles = []
            for p in self.particles:
                r, g, b, a = apply_lighting_to_particle(
                    p, active_lights, self.blinn_phong, self.camera_pos
                )
                # 创建临时粒子副本用于渲染（避免修改原始数据）
                lit_p = Particle(
                    x=p.x, y=p.y, vx=p.vx, vy=p.vy,
                    life=p.life, max_life=p.max_life,
                    size=p.size, color=(r, g, b), alpha=a,
                    rotation=p.rotation, angular_velocity=p.angular_velocity,
                    mass=p.mass,
                    trail_enabled=p.trail_enabled,
                    trail_history=list(p.trail_history),
                    trail_max_length=p.trail_max_length,
                    mesh_surface=p.mesh_surface,
                    mesh_mode=p.mesh_mode,
                )
                lit_particles.append(lit_p)
            renderer.render(lit_particles)
        else:
            renderer.render(self.particles)

    def clear(self):
        # v5.0：对象池模式下归还粒子实例
        if self._pool is not None:
            for p in self.particles:
                self._pool.release(p)
        self.particles.clear()
        self._pending_particles.clear()

    # --- v5.0：空间哈希辅助 ---

    def _rebuild_spatial_grid(self):
        """重建空间哈希：登记碰撞盒（可选：登记粒子）"""
        if self._grid is None:
            return
        self._grid.clear()
        for box in self.collision_boxes:
            self._grid.insert_rect(box, box.x, box.y, box.width, box.height)
        if self._grid_needs_particles:
            for p in self.particles:
                if p.is_alive():
                    self._grid.insert_point(p, p.x, p.y)

    def query_neighbors(self, x: float, y: float, radius: float) -> List[Particle]:
        """
        查询某点半径内的粒子（v5.0）。

        use_spatial_hash=True 时走空间哈希（首次调用后每帧重建粒子格），
        否则退化为线性扫描（结果完全一致）。
        """
        if self._grid is not None:
            if not self._grid_needs_particles:
                self._grid_needs_particles = True
                self._rebuild_spatial_grid()
            found = []
            r2 = radius * radius
            for p in self._grid.query_circle(x, y, radius):
                dx = p.x - x
                dy = p.y - y
                if dx * dx + dy * dy <= r2:
                    found.append(p)
            return found
        r2 = radius * radius
        return [p for p in self.particles
                if (p.x - x) ** 2 + (p.y - y) ** 2 <= r2]

    # --- v5.0：粒子间作用力（pairwise force）---

    def enable_pairwise_force(self, strength: float = 5.0,
                              radius: float = 60.0,
                              repulsion: bool = False):
        """
        开启粒子间相互作用力（吸引或排斥）。

        复杂度：
        - use_spatial_hash=True：O(N·k)，k 为半径内平均邻居数
        - 否则：退化为 O(N²) 线性扫描（小规模可用）
        强度在半径边缘线性衰减到 0，距离 <1px 时忽略（防数值爆炸）。

        Args:
            strength : 力强度（正数）
            radius   : 作用半径（像素）
            repulsion: True = 排斥（负号）
        """
        self._pairwise_strength = float(strength)
        self._pairwise_radius = max(1e-3, float(radius))
        self._pairwise_repulsion = bool(repulsion)
        if self._grid is not None:
            self._grid_needs_particles = True

    def _apply_pairwise_force_to(self, p: Particle, dt: float):
        """对单个粒子应用邻居作用力（空间哈希或线性扫描）"""
        strength = self._pairwise_strength
        radius = self._pairwise_radius
        if strength == 0.0:
            return
        sign = -1.0 if self._pairwise_repulsion else 1.0
        r2 = radius * radius
        fx = fy = 0.0
        if self._grid is not None and self._grid_needs_particles:
            candidates = self._grid.query_circle(p.x, p.y, radius)
        else:
            candidates = self.particles
        for q in candidates:
            if q is p:
                continue
            dx = q.x - p.x
            dy = q.y - p.y
            d2 = dx * dx + dy * dy
            if d2 > r2 or d2 < 1e-6:
                continue
            d = math.sqrt(d2)
            # 线性衰减 + 轻微反距离抑制，防止极端拥挤时数值爆炸
            f = sign * strength * (1.0 - d / radius) / (1.0 + d * 0.01)
            fx += f * dx / d
            fy += f * dy / d
        p.vx += (fx / p.mass) * dt
        p.vy += (fy / p.mass) * dt

    # --- v5.0：粒子光源（emissive particle → 动态点光源）---

    def enable_particle_lights(self, max_lights: int = 8,
                               min_emissive_strength: float = 0.5):
        """
        开启粒子光源：每帧将最亮的自发光粒子作为动态点光源，
        参与 Blinn-Phong 光照管线（自动开启 lighting_enabled）。

        Args:
            max_lights           : 最多注册的点光源数量
            min_emissive_strength: 低于此自发光强度的粒子不参与
        """
        self._particle_lights_enabled = True
        self._max_particle_lights = max(0, int(max_lights))
        self._min_emissive_strength = float(min_emissive_strength)
        self.particle_lights = [
            LightSource(light_type='point', intensity=0.0)
            for _ in range(self._max_particle_lights)
        ]
        if self._max_particle_lights > 0 and not self.lighting_enabled:
            self.lighting_enabled = True

    def _update_particle_lights(self):
        """每帧从存活粒子中挑选最亮的自发光粒子作为点光源"""
        if not self._particle_lights_enabled:
            return
        cands = [p for p in self.particles
                 if p.is_alive() and p.emissive_strength >= self._min_emissive_strength]
        # 亮度 = 自发光强度 × 颜色分量和（近似亮度）
        cands.sort(key=lambda p: p.emissive_strength * sum(p.emissive), reverse=True)
        n = min(self._max_particle_lights, len(cands))
        for i, light in enumerate(self.particle_lights):
            if i < n:
                p = cands[i]
                light.x, light.y = p.x, p.y
                light.color = tuple(min(255, int(c)) for c in p.emissive)
                light.intensity = min(3.0, p.emissive_strength * 0.6)
                light.range_ = 150.0 + p.size * 60.0
            else:
                light.intensity = 0.0

    def get_particle_count(self) -> int:
        return len(self.particles)

    @property
    def time(self) -> float:
        return self._time


# ================================================================
#  Renderer  — 渲染器接口
# ================================================================

class Renderer:
    """渲染器接口"""
    def render(self, particles: List[Particle]):
        raise NotImplementedError("Renderer.render() must be implemented")


# ================================================================
#  曲线构建辅助函数
# ================================================================

def gradient_to_curve(colors: List[Tuple[int, int, int]],
                      num_samples: int = 8) -> List[Tuple[float, Tuple[int, int, int]]]:
    """将颜色列表转换为颜色生命曲线"""
    n = len(colors)
    if n == 0:
        return []
    if n == 1:
        return [(0.0, colors[0])]
    
    curve = []
    for i in range(n - 1):
        t0 = i / (n - 1)
        t1 = (i + 1) / (n - 1)
        curve.append((t0, colors[i]))
        if i < n - 2:
            t_mid = (t0 + t1) / 2
            mid_color = lerp_color(colors[i], colors[i + 1], 0.5)
            curve.append((t_mid, mid_color))
    curve.append((1.0, colors[-1]))
    return curve


def size_curve_constant(base_size: float) -> List[Tuple[float, float]]:
    return [(0.0, base_size), (1.0, base_size)]


def size_curve_fadeout(base_size: float) -> List[Tuple[float, float]]:
    return [(0.0, base_size), (1.0, 0.0)]


def size_curve_pulse(base_size: float,
                     peak_at: float = 0.3,
                     peak_mult: float = 1.5) -> List[Tuple[float, float]]:
    return [
        (0.0, base_size),
        (peak_at, base_size * peak_mult),
        (1.0, base_size * 0.2),
    ]


def velocity_curve_accelerate(mult: float = 1.5) -> List[Tuple[float, Tuple[float, float]]]:
    return [(0.0, (1.0, 1.0)), (1.0, (mult, mult))]


def velocity_curve_decelerate(mult: float = 0.3) -> List[Tuple[float, Tuple[float, float]]]:
    return [(0.0, (1.0, 1.0)), (1.0, (mult, mult))]


# ================================================================
#  便利预设曲线
# ================================================================

FIRE_COLOR_CURVE = gradient_to_curve([
    (255, 255, 240),
    (255, 220, 80),
    (255, 160, 30),
    (255, 80, 10),
    (100, 30, 5),
])

SMOKE_COLOR_CURVE = gradient_to_curve([
    (80, 80, 85),
    (120, 120, 125),
    (160, 160, 165),
    (200, 200, 205),
])

MAGIC_COLOR_CURVE = gradient_to_curve([
    (80, 50, 180),
    (100, 150, 255),
    (180, 200, 255),
    (255, 255, 255),
])

EXPLOSION_SIZE_CURVE = [
    (0.0, 0.5),
    (0.05, 3.0),
    (0.3, 1.5),
    (1.0, 0.0),
]

SPARK_SIZE_CURVE = [
    (0.0, 2.0),
    (0.5, 1.0),
    (1.0, 0.0),
]


# ================================================================
#  快速工厂函数（便捷创建常用组件）
# ================================================================

def create_point_light(x: float, y: float,
                       color: Tuple[int, int, int] = (255, 255, 240),
                       intensity: float = 1.0,
                       range_: float = 400.0) -> LightSource:
    """创建点光源"""
    return LightSource(
        light_type='point', x=x, y=y,
        color=color, intensity=intensity,
        range_=range_, attenuation='inverse_square'
    )


def create_directional_light(direction: float,
                              color: Tuple[int, int, int] = (255, 250, 230),
                              intensity: float = 0.8) -> LightSource:
    """创建方向光（如阳光）"""
    return LightSource(
        light_type='directional',
        direction=direction,
        color=color, intensity=intensity
    )


def create_ambient_light(color: Tuple[int, int, int] = (60, 60, 80),
                         intensity: float = 0.25) -> LightSource:
    """创建环境光"""
    return LightSource(
        light_type='ambient',
        color=color, intensity=intensity
    )


def create_vortex(x: float, y: float,
                  strength: float = 3.0,
                  clockwise: bool = True) -> ForceField:
    """创建漩涡力场"""
    return ForceField(
        field_type='vortex', x=x, y=y,
        strength=strength,
        vortex_clockwise=clockwise,
        vortex_falloff=0.005
    )


def create_turbulence(strength: float = 2.0,
                      scale: float = 40.0) -> ForceField:
    """创建湍流场"""
    return ForceField(
        field_type='turbulence',
        strength=strength,
        turb_scale=scale,
        turb_strength=strength
    )


def create_magnet(x: float, y: float,
                  polarity: float = 1.0,
                  strength: float = 5.0,
                  range_: float = 150.0) -> ForceField:
    """创建磁场（正=吸引，负=排斥）"""
    return ForceField(
        field_type='magnet', x=x, y=y,
        strength=strength,
        magnet_polarity=polarity,
        magnet_range=range_
    )


def create_attractor(x: float = 0.0, y: float = 0.0,
                     strength: float = 5.0,
                     radius: float = 200.0,
                     falloff: str = 'linear',
                     polarity: float = 1.0,
                     follow: Optional[Callable[[], Tuple[float, float]]] = None,
                     ) -> ParticleAttractor:
    """创建粒子吸引子（v5.0：有限范围吸引/排斥，支持 follow 跟随）"""
    return ParticleAttractor(x=x, y=y, strength=strength, radius=radius,
                             falloff=falloff, polarity=polarity, follow=follow)

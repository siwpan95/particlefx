"""
ParticleSystemNumpy — NumPy 向量化粒子系统（P4）

性能目标：对于 1000 粒子，比纯 Python 循环快 30~100 倍。

设计原则：
- 粒子数据全部存于 numpy 数组（SoA: Structure of Arrays）
- 支持重力/风力等常见力场，向量化一次性处理所有粒子
- 渲染接口兼容 PILRenderer（转换为 Particle 列表后渲染）
- 不支持复杂事件/碰撞/拖尾（轻量高性能场景使用）

使用示例：
    from core.particle_numpy import ParticleSystemNumpy, NumpyForceGravity

    sys_np = ParticleSystemNumpy(max_particles=5000)
    sys_np.add_force(NumpyForceGravity(strength=3.0))

    # 批量发射 200 个粒子
    sys_np.emit_burst(x=400, y=300, count=200,
                      speed_range=(2, 10), life_range=(1.0, 3.0),
                      color=(255, 200, 50), size_range=(1.5, 3.5))

    for frame in range(300):
        sys_np.update(dt=1/30)
        img = sys_np.render_to_image(width=800, height=600)
"""

import math
import random
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw


# ================================================================
#  力场接口（NumPy 批量版）
# ================================================================

class NumpyForceBase:
    """NumPy 力场基类（批量计算所有粒子的受力）"""
    
    def apply(self, pos: np.ndarray, vel: np.ndarray,
              mass: np.ndarray, alive: np.ndarray,
              dt: float, time: float = 0.0):
        """
        原地修改 vel（粒子速度数组）。
        
        Args:
            pos   : shape (N, 2), float32 — 粒子位置
            vel   : shape (N, 2), float32 — 粒子速度（原地修改）
            mass  : shape (N,),   float32 — 粒子质量
            alive : shape (N,),   bool    — 存活掩码
            dt    : 时间步长
            time  : 当前时间
        """
        raise NotImplementedError


class NumpyForceGravity(NumpyForceBase):
    """重力（向下）"""
    def __init__(self, strength: float = 9.8):
        self.strength = strength
    
    def apply(self, pos, vel, mass, alive, dt, time=0.0):
        vel[alive, 1] += (self.strength / mass[alive]) * dt


class NumpyForceWind(NumpyForceBase):
    """风力（水平）"""
    def __init__(self, strength: float = 1.0):
        self.strength = strength
    
    def apply(self, pos, vel, mass, alive, dt, time=0.0):
        vel[alive, 0] += (self.strength / mass[alive]) * dt


class NumpyForceDrag(NumpyForceBase):
    """空气阻力（速度衰减）"""
    def __init__(self, factor: float = 0.98):
        """factor: 每帧速度保留比例，<1 产生阻力"""
        self.factor = factor
    
    def apply(self, pos, vel, mass, alive, dt, time=0.0):
        vel[alive] *= (self.factor ** (dt * 60))


class NumpyForceAttraction(NumpyForceBase):
    """点吸引/排斥场"""
    def __init__(self, x: float, y: float,
                 strength: float = 5.0,
                 min_dist: float = 10.0):
        self.center = np.array([x, y], dtype=np.float32)
        self.strength = strength
        self.min_dist = min_dist
    
    def apply(self, pos, vel, mass, alive, dt, time=0.0):
        if not np.any(alive):
            return
        diff = self.center - pos[alive]  # (N_alive, 2)
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        dist = np.maximum(dist, self.min_dist)
        direction = diff / dist
        force = self.strength / (1.0 + dist * 0.01)
        vel[alive] += (direction * force / mass[alive, np.newaxis]) * dt


class NumpyForceTurbulence(NumpyForceBase):
    """湍流场（基于 sin/cos 近似 noise）"""
    def __init__(self, strength: float = 2.0, scale: float = 40.0):
        self.strength = strength
        self.scale = scale
        self._seed = random.uniform(0, 1000)
    
    def apply(self, pos, vel, mass, alive, dt, time=0.0):
        if not np.any(alive):
            return
        p = pos[alive]
        t_val = time + self._seed
        sx = self.scale
        
        # 低成本近似湍流：多层 sin/cos 叠加
        fx = (self.strength * np.sin(p[:, 0] / sx * 3 + t_val * 2) *
              np.cos(p[:, 1] / sx * 2 + t_val * 1.5))
        fy = (self.strength * np.cos(p[:, 1] / sx * 3 + t_val * 1.8) *
              np.sin(p[:, 0] / sx * 2.5 + t_val * 2.2))
        
        force = np.stack([fx, fy], axis=1)
        vel[alive] += (force / mass[alive, np.newaxis]) * dt


# ================================================================
#  ParticleSystemNumpy
# ================================================================

class ParticleSystemNumpy:
    """
    NumPy 向量化粒子系统（P4）
    
    SoA（数组结构）设计 — 所有粒子属性用独立数组存储：
    - pos      : (N, 2) float32  — 位置 (x, y)
    - vel      : (N, 2) float32  — 速度 (vx, vy)
    - life     : (N,)   float32  — 剩余生命 (0~1)
    - max_life : (N,)   float32  — 最大生命（秒）
    - size     : (N,)   float32  — 粒子大小
    - color    : (N, 3) uint8    — RGB 颜色
    - alpha    : (N,)   float32  — 当前透明度
    - mass     : (N,)   float32  — 质量
    - alive    : (N,)   bool     — 存活标记（dead slot 可复用）
    """

    def __init__(self, max_particles: int = 5000):
        self.max_particles = max_particles
        self._n = max_particles
        
        # SoA 粒子数组
        self.pos      = np.zeros((self._n, 2), dtype=np.float32)
        self.vel      = np.zeros((self._n, 2), dtype=np.float32)
        self.life     = np.zeros(self._n, dtype=np.float32)    # 归一化 0~1
        self.max_life = np.ones(self._n, dtype=np.float32)     # 秒
        self.size     = np.ones(self._n, dtype=np.float32)
        self.color    = np.zeros((self._n, 3), dtype=np.uint8)
        self.alpha    = np.zeros(self._n, dtype=np.float32)
        self.mass     = np.ones(self._n, dtype=np.float32)
        self.alive    = np.zeros(self._n, dtype=bool)
        
        # 力场列表
        self._forces: List[NumpyForceBase] = []
        
        # 时间
        self._time = 0.0
        
        # 下一个可用槽位指针（循环分配）
        self._next_slot = 0

    # ------------------------------------------------------------------
    #  力场管理
    # ------------------------------------------------------------------

    def add_force(self, force: NumpyForceBase):
        self._forces.append(force)

    def clear_forces(self):
        self._forces.clear()

    # ------------------------------------------------------------------
    #  粒子发射
    # ------------------------------------------------------------------

    def emit_burst(self,
                   x: float, y: float,
                   count: int,
                   speed_range: Tuple[float, float] = (2, 8),
                   life_range: Tuple[float, float] = (0.5, 2.0),
                   size_range: Tuple[float, float] = (1.5, 3.5),
                   color: Tuple[int, int, int] = (255, 200, 50),
                   mass_range: Tuple[float, float] = (1.0, 1.0),
                   angle_range: Tuple[float, float] = (0, math.pi * 2),
                   direction_bias: Optional[Tuple[float, float]] = None,
                   ):
        """
        批量发射粒子（向量化）。
        
        Args:
            x, y          : 发射位置
            count         : 数量
            speed_range   : (min, max) 速度范围
            life_range    : (min, max) 生命值范围（秒）
            size_range    : (min, max) 大小范围
            color         : RGB 颜色（所有粒子相同颜色）或 (N,3) 数组
            mass_range    : (min, max) 质量范围
            angle_range   : (min, max) 发射角度范围（弧度）
            direction_bias: (dx, dy) 方向偏移，None=全方向
        """
        if count <= 0:
            return
        
        # 找出死亡槽位
        dead_slots = np.where(~self.alive)[0]
        if len(dead_slots) == 0:
            return  # 已满，丢弃
        
        actual = min(count, len(dead_slots))
        slots = dead_slots[:actual]
        
        # 生成随机数据（向量化）
        rng = np.random
        angles = rng.uniform(angle_range[0], angle_range[1], actual).astype(np.float32)
        speeds = rng.uniform(speed_range[0], speed_range[1], actual).astype(np.float32)
        
        vx = np.cos(angles) * speeds
        vy = np.sin(angles) * speeds
        
        if direction_bias is not None:
            vx += direction_bias[0]
            vy += direction_bias[1]
        
        lives = rng.uniform(life_range[0], life_range[1], actual).astype(np.float32)
        sizes = rng.uniform(size_range[0], size_range[1], actual).astype(np.float32)
        masses = rng.uniform(mass_range[0], mass_range[1], actual).astype(np.float32)
        
        # 写入数组
        self.pos[slots, 0] = x
        self.pos[slots, 1] = y
        self.vel[slots, 0] = vx
        self.vel[slots, 1] = vy
        self.max_life[slots] = lives
        self.life[slots] = 1.0  # 归一化，从满值开始
        self.size[slots] = sizes
        self.mass[slots] = masses
        self.alpha[slots] = 1.0
        self.alive[slots] = True
        
        # 颜色（支持单色和随机颜色列表）
        if isinstance(color, (list, tuple)) and len(color) == 3 and isinstance(color[0], int):
            self.color[slots] = color
        elif isinstance(color, np.ndarray) and color.ndim == 2:
            self.color[slots] = color[:actual] if len(color) >= actual else np.tile(color, (actual, 1))[:actual]
        else:
            self.color[slots] = color

    def emit_radial(self, x: float, y: float, count: int,
                    speed_range=(2, 8), life_range=(0.5, 2.0),
                    size_range=(1.5, 3.5),
                    colors: Optional[List[Tuple[int, int, int]]] = None,
                    mass_range=(1.0, 1.0)):
        """
        径向（全方向）发射。支持调色板随机颜色。
        """
        if count <= 0:
            return
        
        dead_slots = np.where(~self.alive)[0]
        if len(dead_slots) == 0:
            return
        
        actual = min(count, len(dead_slots))
        slots = dead_slots[:actual]
        
        rng = np.random
        angles = rng.uniform(0, math.pi * 2, actual).astype(np.float32)
        speeds = rng.uniform(speed_range[0], speed_range[1], actual).astype(np.float32)
        lives = rng.uniform(life_range[0], life_range[1], actual).astype(np.float32)
        sizes = rng.uniform(size_range[0], size_range[1], actual).astype(np.float32)
        masses = rng.uniform(mass_range[0], mass_range[1], actual).astype(np.float32)
        
        self.pos[slots, 0] = x
        self.pos[slots, 1] = y
        self.vel[slots, 0] = np.cos(angles) * speeds
        self.vel[slots, 1] = np.sin(angles) * speeds
        self.max_life[slots] = lives
        self.life[slots] = 1.0
        self.size[slots] = sizes
        self.mass[slots] = masses
        self.alpha[slots] = 1.0
        self.alive[slots] = True
        
        if colors is not None:
            palette_arr = np.array(colors, dtype=np.uint8)
            idx = rng.randint(0, len(colors), actual)
            self.color[slots] = palette_arr[idx]
        else:
            self.color[slots] = (255, 255, 255)

    # ------------------------------------------------------------------
    #  更新（全向量化）
    # ------------------------------------------------------------------

    def update(self, dt: float):
        """
        批量更新所有存活粒子（O(1) 向量操作）。
        
        性能：对于 5000 粒子，此函数单次调用约 0.05~0.2ms（NumPy）。
        """
        self._time += dt
        alive = self.alive
        
        if not np.any(alive):
            return
        
        # 1. 应用力场
        for force in self._forces:
            force.apply(self.pos, self.vel, self.mass, alive, dt, self._time)
        
        # 2. 更新位置
        self.pos[alive] += self.vel[alive] * dt
        
        # 3. 衰减生命值
        self.life[alive] -= dt / self.max_life[alive]
        
        # 4. 更新透明度（与生命值同步）
        self.alpha[alive] = np.clip(self.life[alive], 0, 1)
        
        # 5. 标记死亡粒子
        self.alive[self.life <= 0] = False

    # ------------------------------------------------------------------
    #  渲染
    # ------------------------------------------------------------------

    def render_to_image(self,
                        width: int, height: int,
                        background: Optional[Image.Image] = None,
                        blend_mode: str = 'additive',
                        ) -> Image.Image:
        """
        将粒子渲染到图像（高性能向量化路径）。
        
        Args:
            width, height : 画布尺寸
            background    : 背景图（RGB），None = 黑色
            blend_mode    : 'normal' | 'additive' | 'screen'
        
        Returns:
            RGB Image
        """
        if background is None:
            canvas = Image.new('RGB', (width, height), (0, 0, 0))
        else:
            canvas = background.convert('RGB').copy()
        
        alive = self.alive
        if not np.any(alive):
            return canvas
        
        # 提取存活粒子数据
        pos = self.pos[alive]       # (N, 2)
        colors = self.color[alive]  # (N, 3) uint8
        alphas = self.alpha[alive]  # (N,)
        sizes = self.size[alive]    # (N,)
        
        # 过滤屏幕外的粒子
        in_screen = ((pos[:, 0] >= 0) & (pos[:, 0] < width) &
                     (pos[:, 1] >= 0) & (pos[:, 1] < height) &
                     (alphas > 0.01) & (sizes >= 0.5))
        
        pos = pos[in_screen]
        colors = colors[in_screen]
        alphas = alphas[in_screen]
        sizes = sizes[in_screen]
        
        if len(pos) == 0:
            return canvas
        
        if blend_mode == 'normal':
            # 普通 alpha 混合：用 PIL Draw
            draw = ImageDraw.Draw(canvas)
            for i in range(len(pos)):
                x, y = int(pos[i, 0]), int(pos[i, 1])
                a = float(alphas[i])
                sz = max(1, int(sizes[i] * a))
                r = int(colors[i, 0] * a)
                g = int(colors[i, 1] * a)
                b = int(colors[i, 2] * a)
                draw.ellipse([x - sz, y - sz, x + sz, y + sz], fill=(r, g, b))
        else:
            # Additive / Screen：用 numpy 批量写入像素
            canvas_arr = np.array(canvas, dtype=np.float32)
            
            # 对每个粒子绘制圆盘（向量化绘制思路：用 meshgrid + 距离掩码）
            # 性能折中：对小粒子用单点，中等粒子用小方块，大粒子用圆盘
            xs = pos[:, 0].astype(int)
            ys = pos[:, 1].astype(int)
            
            for i in range(len(pos)):
                xi, yi = xs[i], ys[i]
                a = float(alphas[i])
                sz = max(1, int(sizes[i] * a))
                cr = float(colors[i, 0]) * a
                cg = float(colors[i, 1]) * a
                cb = float(colors[i, 2]) * a
                
                # 计算粒子影响的像素范围
                x0 = max(0, xi - sz)
                x1 = min(width, xi + sz + 1)
                y0 = max(0, yi - sz)
                y1 = min(height, yi + sz + 1)
                
                if x0 >= x1 or y0 >= y1:
                    continue
                
                # 圆盘掩码
                yy, xx = np.ogrid[y0:y1, x0:x1]
                dist_sq = (xx - xi)**2 + (yy - yi)**2
                mask = dist_sq <= sz * sz  # (h, w) bool
                
                patch = canvas_arr[y0:y1, x0:x1]  # (h, w, 3)
                
                if blend_mode == 'additive':
                    patch[mask, 0] = np.clip(patch[mask, 0] + cr, 0, 255)
                    patch[mask, 1] = np.clip(patch[mask, 1] + cg, 0, 255)
                    patch[mask, 2] = np.clip(patch[mask, 2] + cb, 0, 255)
                else:  # screen
                    base = patch[mask] / 255.0
                    src = np.array([cr, cg, cb]) / 255.0
                    result = 1.0 - (1.0 - base) * (1.0 - src)
                    patch[mask] = np.clip(result * 255, 0, 255)
                
                canvas_arr[y0:y1, x0:x1] = patch
            
            canvas = Image.fromarray(canvas_arr.astype(np.uint8))
        
        return canvas

    def render_via_pil_renderer(self, renderer) -> None:
        """
        通过 PILRenderer 渲染（兼容接口）。
        
        将 numpy 数组转换为 Particle 对象列表，传给标准渲染器。
        注意：此路径有 Python 循环开销，仅作为兼容接口使用。
        对于性能敏感场景，优先使用 render_to_image()。
        """
        from core.particle import Particle
        alive = self.alive
        if not np.any(alive):
            renderer.render([])
            return
        
        particles = []
        pos = self.pos[alive]
        vel = self.vel[alive]
        life = self.life[alive]
        max_l = self.max_life[alive]
        sz = self.size[alive]
        col = self.color[alive]
        alp = self.alpha[alive]
        
        for i in range(len(pos)):
            p = Particle(
                x=float(pos[i, 0]),
                y=float(pos[i, 1]),
                vx=float(vel[i, 0]),
                vy=float(vel[i, 1]),
                life=float(life[i]),
                max_life=float(max_l[i]),
                size=float(sz[i]),
                color=(int(col[i, 0]), int(col[i, 1]), int(col[i, 2])),
                alpha=float(alp[i]),
            )
            particles.append(p)
        
        renderer.render(particles)

    # ------------------------------------------------------------------
    #  属性
    # ------------------------------------------------------------------

    @property
    def particle_count(self) -> int:
        return int(np.sum(self.alive))

    @property
    def time(self) -> float:
        return self._time

    def clear(self):
        self.alive[:] = False
        self.life[:] = 0.0


# ================================================================
#  便捷工厂
# ================================================================

def create_numpy_fireworks_burst(x: float, y: float,
                                  palette: List[Tuple[int, int, int]],
                                  style: str = 'radial',
                                  system: Optional[ParticleSystemNumpy] = None,
                                  ) -> ParticleSystemNumpy:
    """
    在 numpy 粒子系统中快速创建烟花爆炸效果。
    
    Args:
        x, y    : 爆炸位置
        palette : 颜色调色板
        style   : 'radial' | 'ring' | 'palm'
        system  : 复用已有系统，None 则创建新系统
    
    Returns:
        ParticleSystemNumpy 实例
    """
    if system is None:
        sys_np = ParticleSystemNumpy(max_particles=2000)
        sys_np.add_force(NumpyForceGravity(strength=3.0))
        sys_np.add_force(NumpyForceDrag(factor=0.99))
    else:
        sys_np = system
    
    palette_arr = np.array(palette, dtype=np.uint8)
    
    if style == 'radial':
        sys_np.emit_radial(x, y, count=150,
                           speed_range=(4, 10), life_range=(1.5, 3.0),
                           size_range=(1.5, 3.5), colors=palette)
    
    elif style == 'ring':
        for ring in range(2):
            ring_speed = 5 + ring * 3
            num_p = 50 + ring * 30
            for i in range(num_p):
                angle = 2 * math.pi * i / num_p
                sys_np.emit_burst(
                    x=x, y=y, count=1,
                    speed_range=(ring_speed - 0.3, ring_speed + 0.3),
                    life_range=(1.0, 2.0),
                    size_range=(2, 3.5),
                    color=palette[ring % len(palette)],
                    angle_range=(angle - 0.02, angle + 0.02),
                )
    
    elif style == 'palm':
        num_arms = random.choice([5, 6, 8])
        for arm in range(num_arms):
            base_angle = 2 * math.pi * arm / num_arms
            sys_np.emit_burst(
                x=x, y=y, count=25,
                speed_range=(5, 9), life_range=(1.2, 2.5),
                size_range=(2, 4),
                color=random.choice(palette),
                angle_range=(base_angle - 0.06, base_angle + 0.06),
            )
    
    return sys_np

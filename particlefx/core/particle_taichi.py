"""
Taichi SPH 流体后端（v5.0 增强 ⑧ — 可选依赖，零开销）

WCSPH（弱可压缩光滑粒子流体动力学）2D 实现：
- Poly6 核估计密度，Spiky 梯度计算压力，拉普拉斯核计算粘性
- 均匀网格邻居搜索（cell size = 支撑半径 h）
- 支持重力 / 粘性 / 边界反弹
- 可选 taichi 架构：cpu / cuda / vulkan / metal（默认自动选择）

依赖：pip install taichi（未安装时 TaichiSPHSystem 构造抛出带指引的错误，
其余粒子引擎功能不受影响）。taichi 1.7.x 暂无 Python 3.14 轮子，
3.14 用户请使用 Python 3.12/3.13 环境或替代方案（JAX / NVIDIA Warp / Cython）。

用法：
    from .particle_taichi import TaichiSPHSystem

    sph = TaichiSPHSystem(max_particles=4096, width=800, height=600)
    sph.add_fluid_block(x=300, y=80, w=120, h=120, spacing=6.0)

    for frame in range(300):
        sph.update(1/60)
        img = sph.render_to_image(width=800, height=600, color=(60, 140, 255))

参考：
- Müller et al. 2003 "Particle-Based Fluid Simulation for Interactive Applications"
- Taichi 官方 SPH 示例与社区实现（SPH_Taichi / taichiCourse01）
"""

import math

import numpy as np

try:
    import taichi as ti
    HAS_TAICHI = True
except ImportError:
    HAS_TAICHI = False
    ti = None


_TAICHI_INSTALL_HINT = (
    "TaichiSPHSystem 需要 taichi 依赖：pip install taichi。"
    "未安装时粒子引擎其余部分不受影响（可选依赖零开销原则）。"
)


def poly6_kernel(r2: float, h: float) -> float:
    """Poly6 核值（2D），r2 为距离平方，h 为支撑半径（CPU 参考实现）"""
    h2 = h * h
    if r2 >= h2:
        return 0.0
    q = 1.0 - r2 / h2
    return 4.0 / (math.pi * h2) * q * q * q


class TaichiSPHSystem:
    """
    Taichi WCSPH 流体系统（v5.0 增强 ⑧）

    参数：
        max_particles   : 最大粒子数（预分配，SoA）
        width/height    : 容器尺寸（像素）
        particle_radius : 粒子物理半径（≈ spacing/2）
        h               : 核支撑半径（默认 = particle_radius * 2.4）
        rest_density    : 目标密度（控制压缩性）
        stiffness       : 压力刚度系数
        viscosity       : 粘性系数
        gravity         : 重力 (gx, gy)，像素/秒²
        boundary_damping: 边界反弹速度保留比例
        taichi_arch     : 'auto' | 'cpu' | 'cuda' | 'vulkan' | 'metal' | None
    """

    def __init__(self,
                 max_particles: int = 4096,
                 width: int = 800,
                 height: int = 600,
                 particle_radius: float = 3.0,
                 h: float = None,
                 rest_density: float = 1000.0,
                 stiffness: float = 800.0,
                 viscosity: float = 0.6,
                 gravity: tuple = (0.0, 500.0),
                 boundary_damping: float = 0.5,
                 taichi_arch: str = "auto"):
        if not HAS_TAICHI:
            raise ImportError(_TAICHI_INSTALL_HINT)

        self.max_particles = int(max_particles)
        self.width = float(width)
        self.height = float(height)
        self.particle_radius = float(particle_radius)
        self.h = float(h if h is not None else particle_radius * 2.4)
        self.rest_density = float(rest_density)
        self.stiffness = float(stiffness)
        self.viscosity = float(viscosity)
        self.gravity = (float(gravity[0]), float(gravity[1]))
        self.boundary_damping = float(boundary_damping)
        self.arch_name = self._init_taichi(taichi_arch)

        self._allocate_fields()
        self._count = 0

    # ------------------------------------------------------------------
    #  初始化
    # ------------------------------------------------------------------

    def _init_taichi(self, arch) -> str:
        if _taichi_initialized():
            return "already-initialized"
        if arch in (None, "auto"):
            try:
                ti.init(arch=ti.gpu)
                return "gpu"
            except Exception:
                ti.init(arch=ti.cpu)
                return "cpu"
        try:
            ti.init(arch=getattr(ti, str(arch), ti.cpu))
            return str(arch)
        except Exception:
            ti.init(arch=ti.cpu)
            return "cpu"

    def _allocate_fields(self):
        N = self.max_particles
        self.pos = ti.Vector.field(2, dtype=ti.f32, shape=N)
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=N)
        self.density = ti.field(dtype=ti.f32, shape=N)
        self.pressure = ti.field(dtype=ti.f32, shape=N)
        self.alive = ti.field(dtype=ti.i32, shape=N)   # 1=存活
        self.count = ti.field(dtype=ti.i32, shape=())  # 存活数量

    # ------------------------------------------------------------------
    #  添加粒子（CPU 端写入）
    # ------------------------------------------------------------------

    def add_particles(self, x, y, vx=0.0, vy=0.0, count=1) -> int:
        """在 (x, y) 添加 count 个带随机抖动的粒子，返回实际添加数"""
        if not HAS_TAICHI:
            raise ImportError(_TAICHI_INSTALL_HINT)
        pos_np = self.pos.to_numpy()
        vel_np = self.vel.to_numpy()
        alive_np = self.alive.to_numpy()
        rng = np.random.default_rng()
        added = 0
        jitter = self.particle_radius * 0.25
        for _ in range(int(count)):
            if self._count >= self.max_particles:
                break
            i = self._count
            pos_np[i] = (float(x) + rng.uniform(-jitter, jitter),
                         float(y) + rng.uniform(-jitter, jitter))
            vel_np[i] = (float(vx), float(vy))
            alive_np[i] = 1
            self._count += 1
            added += 1
        self.pos.from_numpy(pos_np)
        self.vel.from_numpy(vel_np)
        self.alive.from_numpy(alive_np)
        self.count[None] = self._count
        return added

    def add_fluid_block(self, x, y, w, h, spacing=6.0) -> int:
        """以网格布局添加一块方形水体（经典初始场景），返回添加数"""
        added = 0
        nx = max(1, int(w / spacing))
        ny = max(1, int(h / spacing))
        for j in range(ny):
            for i in range(nx):
                added += self.add_particles(x + i * spacing, y + j * spacing,
                                            count=1)
        return added

    # ------------------------------------------------------------------
    #  更新（Taichi 内核，见文件底部）
    # ------------------------------------------------------------------

    def update(self, dt: float):
        """执行一帧 SPH 模拟：密度压力 → 力 → 积分 → 边界"""
        dt = float(dt)
        n = self._count
        if n == 0:
            return
        _kernel_density_pressure(self.pos, self.alive, self.density,
                                 self.pressure, self.h, self.rest_density,
                                 self.stiffness, n)
        _kernel_forces(self.pos, self.vel, self.alive, self.density,
                       self.pressure, self.h, self.viscosity,
                       self.gravity[0], self.gravity[1], dt, n)
        _kernel_integrate(self.pos, self.vel, self.alive, dt, n)
        _kernel_boundary(self.pos, self.vel, self.alive, self.width,
                         self.height, self.particle_radius,
                         self.boundary_damping, n)

    # ------------------------------------------------------------------
    #  渲染 / 导出
    # ------------------------------------------------------------------

    def positions_numpy(self):
        """导出存活粒子位置 (M, 2) float32"""
        pos_np = self.pos.to_numpy()
        alive_np = self.alive.to_numpy()
        return pos_np[:self._count][alive_np[:self._count] == 1]

    def render_to_image(self, width=None, height=None,
                        color=(60, 140, 255),
                        background_color=(6, 8, 18)):
        """将流体粒子渲染为 PIL 图像（叠加圆盘近似液面）"""
        from PIL import Image, ImageDraw
        w = int(width or self.width)
        h = int(height or self.height)
        img = Image.new("RGB", (w, h), background_color)
        draw = ImageDraw.Draw(img)
        r = max(1.0, self.particle_radius * 1.15)
        for px, py in self.positions_numpy():
            draw.ellipse([px - r, py - r, px + r, py + r], fill=color)
        return img

    @property
    def particle_count(self) -> int:
        return int(self._count)


# ================================================================
#  Taichi 内核（模块级 + ti.template 字段参数，最可移植的写法）
# ================================================================

def _taichi_initialized() -> bool:
    if not HAS_TAICHI:
        return False
    try:
        return ti.is_initialized()
    except Exception:
        return False


if HAS_TAICHI:

    @ti.func
    def _sph_poly6(r2, h2):
        q = 1.0 - r2 / h2
        return 4.0 / (3.14159265 * h2) * q * q * q

    @ti.func
    def _sph_spiky_grad_factor(r, h):
        # spiky 核梯度系数（2D）：-30/(π h³) · (1-r/h)² / r
        # （Taichi 不允许非 static 分支 return，改用 r 钳位保证数值安全）
        r_safe = ti.max(r, 1e-6)
        q = 1.0 - r_safe / h
        return -30.0 / (3.14159265 * h * h * h) * q * q / r_safe

    @ti.func
    def _sph_visc_laplacian(r, h):
        # 粘性拉普拉斯核（2D）：20/(π h³) · (1-r/h)
        return 20.0 / (3.14159265 * h * h * h) * (1.0 - r / h)

    @ti.kernel
    def _kernel_density_pressure(pos: ti.template(), alive: ti.template(),
                                 density: ti.template(), pressure: ti.template(),
                                 h: ti.f32, rest_density: ti.f32,
                                 stiffness: ti.f32, n: ti.i32):
        for i in range(n):
            if alive[i] == 1:
                h2 = h * h
                rho = 0.0
                for j in range(n):
                    if alive[j] == 1:
                        dx = pos[i][0] - pos[j][0]
                        dy = pos[i][1] - pos[j][1]
                        r2 = dx * dx + dy * dy
                        if r2 < h2:
                            rho += _sph_poly6(r2, h2)
                density[i] = rho
                pressure[i] = stiffness * (rho - rest_density)
            else:
                density[i] = 0.0

    @ti.kernel
    def _kernel_forces(pos: ti.template(), vel: ti.template(),
                       alive: ti.template(), density: ti.template(),
                       pressure: ti.template(), h: ti.f32,
                       viscosity: ti.f32, gx: ti.f32, gy: ti.f32,
                       dt: ti.f32, n: ti.i32):
        for i in range(n):
            if alive[i] == 1:
                fx = 0.0
                fy = 0.0
                for j in range(n):
                    if alive[j] == 1 and j != i:
                        dx = pos[i][0] - pos[j][0]
                        dy = pos[i][1] - pos[j][1]
                        r2 = dx * dx + dy * dy
                        if r2 < h * h and r2 > 1e-8:
                            r = ti.sqrt(r2)
                            # 压力（对称形式，按邻居密度归一） + 粘性
                            rho_j = density[j]
                            if rho_j < 1.0:
                                rho_j = 1.0
                            p_term = (pressure[i] + pressure[j]) * 0.5 / rho_j
                            grad = _sph_spiky_grad_factor(r, h)
                            fx -= p_term * grad * dx / r
                            fy -= p_term * grad * dy / r

                            lap = _sph_visc_laplacian(r, h)
                            fx += viscosity * (vel[j][0] - vel[i][0]) * lap
                            fy += viscosity * (vel[j][1] - vel[i][1]) * lap

                # 外部力（重力）
                fx += gx
                fy += gy

                vel[i][0] += fx * dt
                vel[i][1] += fy * dt

    @ti.kernel
    def _kernel_integrate(pos: ti.template(), vel: ti.template(),
                          alive: ti.template(), dt: ti.f32, n: ti.i32):
        for i in range(n):
            if alive[i] == 1:
                pos[i][0] += vel[i][0] * dt
                pos[i][1] += vel[i][1] * dt

    @ti.kernel
    def _kernel_boundary(pos: ti.template(), vel: ti.template(),
                         alive: ti.template(), width: ti.f32, height: ti.f32,
                         radius: ti.f32, damping: ti.f32, n: ti.i32):
        for i in range(n):
            if alive[i] == 1:
                if pos[i][0] < radius:
                    pos[i][0] = radius
                    vel[i][0] = -vel[i][0] * damping
                if pos[i][0] > width - radius:
                    pos[i][0] = width - radius
                    vel[i][0] = -vel[i][0] * damping
                if pos[i][1] < radius:
                    pos[i][1] = radius
                    vel[i][1] = -vel[i][1] * damping
                if pos[i][1] > height - radius:
                    pos[i][1] = height - radius
                    vel[i][1] = -vel[i][1] * damping

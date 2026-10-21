"""
GPU 粒子模拟器（v5.0 增强 ⑨ — ModernGL Compute Shader）

把粒子更新（重力 / 风力 / 阻力 / 点吸引子 / 生命衰减）搬进 GPU：
- 粒子数据常驻 SSBO，每帧 dispatch 一个 compute 内核
- 零 CPU 逐粒子循环；CPU 只负责发射与可选回读
- 与 GPUParticleRenderer / UnifiedGPURenderer 组合：
    模拟在 GPU 更新 → 回读坐标渲染，或直接共享缓冲（进阶用法）

依赖：moderngl（未安装时构造抛错，其余功能不受影响）

用法：
    import moderngl
    from .particle_sim_gpu import GPUParticleSimulator

    ctx = moderngl.create_context(standalone=True)
    sim = GPUParticleSimulator(ctx, max_particles=10000,
                               gravity=500.0, wind=0.0, drag=0.99)
    sim.emit_burst(x=400, y=300, count=5000, speed_range=(100, 400),
                   life_range=(1.0, 3.0))

    for frame in range(300):
        sim.update(1/60)
        tuples = sim.to_particle_tuples()  # [(x,y,size,r,g,b,alpha), ...]
        # renderer.render(tuples)
"""

import math
from typing import List, Optional, Tuple

import numpy as np

try:
    import moderngl
    HAS_MODERNGL = True
except ImportError:
    HAS_MODERNGL = False
    moderngl = None


_MODERNGL_INSTALL_HINT = (
    "GPUParticleSimulator 需要 moderngl 依赖：pip install moderngl。"
    "未安装时粒子引擎其余部分不受影响（可选依赖零开销原则）。"
)


# ================================================================
#  Compute Shader
# ================================================================

CS_PARTICLE_UPDATE = """
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer PosBuf     { vec2 pos[]; };
layout(std430, binding = 1) buffer VelBuf     { vec2 vel[]; };
layout(std430, binding = 2) buffer LifeBuf    { float life[]; };
layout(std430, binding = 3) buffer MaxLifeBuf { float max_life[]; };
layout(std430, binding = 4) buffer AliveBuf   { int alive[]; };
layout(std430, binding = 5) buffer ColorBuf   { vec3 color[]; };
layout(std430, binding = 6) buffer SizeBuf    { float size[]; };

uniform float u_dt;
uniform vec2  u_wind_gravity;      // (wind, gravity)
uniform float u_drag;
uniform vec2  u_attractor;         // 吸引子位置
uniform float u_attractor_strength;
uniform float u_attractor_radius;
uniform int   u_alive_count;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(u_alive_count)) return;
    if (alive[idx] == 0) return;

    vec2 p = pos[idx];
    vec2 v = vel[idx];

    // 风力 + 重力
    v.x += u_wind_gravity.x * u_dt;
    v.y += u_wind_gravity.y * u_dt;

    // 阻力（与 NumpyForceDrag 一致：factor ** (dt*60)）
    if (u_drag > 0.0) {
        v *= pow(u_drag, u_dt * 60.0);
    }

    // 点吸引子（线性衰减，半径内生效，与 ParticleAttractor 一致）
    if (u_attractor_strength != 0.0) {
        vec2 d = u_attractor - p;
        float dist = length(d);
        if (dist < u_attractor_radius && dist > 1e-4) {
            float f = u_attractor_strength * (1.0 - dist / u_attractor_radius)
                      / (1.0 + dist * 0.01);
            v += (d / dist) * f * u_dt;
        }
    }

    // 积分
    pos[idx] = p + v * u_dt;
    vel[idx] = v;

    // 生命衰减（与 ParticleSystemNumpy 一致：允许负值，标记死亡）
    life[idx] -= u_dt / max_life[idx];
    if (life[idx] <= 0.0) {
        alive[idx] = 0;
    }
}
"""


# ================================================================
#  GPUParticleSimulator
# ================================================================

class GPUParticleSimulator:
    """ModernGL Compute 全 GPU 粒子模拟器（v5.0 增强 ⑨）"""

    STRIDE = 256  # local_size_x

    def __init__(self,
                 ctx=None,
                 max_particles: int = 10000,
                 wind: float = 0.0,
                 gravity: float = 500.0,
                 drag: float = 0.99,
                 attractor: Optional[Tuple[float, float, float, float]] = None):
        """
        Args:
            ctx          : moderngl 上下文（None 时自动创建 standalone 上下文）
            max_particles: 粒子容量
            wind/gravity : 常量风力/重力
            drag         : 每帧速度保留比例（<1 产生阻力）
            attractor    : (x, y, strength, radius) 点吸引子，None 禁用
        """
        if not HAS_MODERNGL:
            raise ImportError(_MODERNGL_INSTALL_HINT)

        self.max_particles = int(max_particles)
        self.ctx = ctx or moderngl.create_context(standalone=True)
        self.wind = float(wind)
        self.gravity = float(gravity)
        self.drag = float(drag)
        self.attractor = tuple(float(v) for v in attractor) if attractor else None

        self._alive_count = 0
        self._color = np.zeros((self.max_particles, 3), dtype=np.float32)
        self._size = np.ones(self.max_particles, dtype=np.float32) * 4.0

        self._create_buffers()
        self._compile_shader()

    # ------------------------------------------------------------------
    #  GPU 资源
    # ------------------------------------------------------------------

    def _create_buffers(self):
        N = self.max_particles
        self.buf_pos = self.ctx.buffer(reserve=N * 2 * 4)
        self.buf_vel = self.ctx.buffer(reserve=N * 2 * 4)
        self.buf_life = self.ctx.buffer(reserve=N * 4)
        self.buf_max_life = self.ctx.buffer(reserve=N * 4)
        self.buf_alive = self.ctx.buffer(reserve=N * 4)
        self.buf_color = self.ctx.buffer(reserve=N * 3 * 4)
        self.buf_size = self.ctx.buffer(reserve=N * 4)
        for binding, buf in enumerate((self.buf_pos, self.buf_vel, self.buf_life,
                                       self.buf_max_life, self.buf_alive,
                                       self.buf_color, self.buf_size)):
            buf.bind_to_storage_buffer(binding)

    def _compile_shader(self):
        self.cs = self.ctx.compute_shader(CS_PARTICLE_UPDATE)
        self._group_count = (self.max_particles + self.STRIDE - 1) // self.STRIDE

    # ------------------------------------------------------------------
    #  发射
    # ------------------------------------------------------------------

    def emit_burst(self, x: float, y: float, count: int,
                   speed_range: Tuple[float, float] = (50, 200),
                   life_range: Tuple[float, float] = (0.5, 2.0),
                   size_range: Tuple[float, float] = (2.0, 5.0),
                   color: Tuple[int, int, int] = (255, 200, 80),
                   angle_range: Tuple[float, float] = (0, math.pi * 2),
                   direction_bias: Optional[Tuple[float, float]] = None) -> int:
        """批量发射粒子（写入 GPU 缓冲的空闲槽位），返回实际发射数"""
        # 先压缩死亡槽位，释放空间
        self._compact_dead_slots()
        if self._alive_count >= self.max_particles:
            return 0
        actual = min(int(count), self.max_particles - self._alive_count)
        rng = np.random.default_rng()
        angles = rng.uniform(angle_range[0], angle_range[1], actual)
        speeds = rng.uniform(speed_range[0], speed_range[1], actual)
        vx = np.cos(angles) * speeds
        vy = np.sin(angles) * speeds
        if direction_bias is not None:
            vx += direction_bias[0]
            vy += direction_bias[1]

        start = self._alive_count
        pos = np.zeros((actual, 2), dtype=np.float32)
        pos[:, 0] = x
        pos[:, 1] = y
        vel = np.stack([vx, vy], axis=1).astype(np.float32)
        life = np.ones(actual, dtype=np.float32)
        max_life = rng.uniform(life_range[0], life_range[1], actual).astype(np.float32)
        alive = np.ones(actual, dtype=np.int32)
        sizes = rng.uniform(size_range[0], size_range[1], actual).astype(np.float32)
        colors = np.tile(np.array(color, dtype=np.float32) / 255.0, (actual, 1))

        # 写入缓冲（槽位从 alive_count 开始；死亡槽位由 compact 逻辑回收）
        self.buf_pos.write(pos[:actual].tobytes(), offset=start * 2 * 4)
        self.buf_vel.write(vel[:actual].tobytes(), offset=start * 2 * 4)
        self.buf_life.write(life[:actual].tobytes(), offset=start * 4)
        self.buf_max_life.write(max_life[:actual].tobytes(), offset=start * 4)
        self.buf_alive.write(alive[:actual].tobytes(), offset=start * 4)
        self.buf_color.write(colors[:actual].tobytes(), offset=start * 3 * 4)
        self.buf_size.write(sizes[:actual].tobytes(), offset=start * 4)
        self._alive_count += actual
        return actual

    def _compact_dead_slots(self):
        """回读存活标记，将死亡槽位压缩到数组前部（发射前调用）"""
        if self._alive_count == 0:
            return
        alive = np.frombuffer(self.buf_alive.read(self._alive_count * 4),
                              dtype=np.int32).copy()
        keep = alive == 1
        if keep.all():
            return
        # 压缩：回读全部有效数据，按存活顺序重写
        n = self._alive_count
        pos = np.frombuffer(self.buf_pos.read(n * 8), dtype=np.float32).reshape(n, 2)
        vel = np.frombuffer(self.buf_vel.read(n * 8), dtype=np.float32).reshape(n, 2)
        life = np.frombuffer(self.buf_life.read(n * 4), dtype=np.float32)
        max_life = np.frombuffer(self.buf_max_life.read(n * 4), dtype=np.float32)
        size = np.frombuffer(self.buf_size.read(n * 4), dtype=np.float32)
        color = np.frombuffer(self.buf_color.read(n * 12), dtype=np.float32).reshape(n, 3)
        m = int(keep.sum())
        for name, arr, stride in (("pos", pos, 8), ("vel", vel, 8),
                                  ("life", life, 4), ("max_life", max_life, 4),
                                  ("size", size, 4), ("color", color, 12)):
            buf = getattr(self, f"buf_{name}")
            buf.write(arr[keep].tobytes())
            buf.write(b"\x00" * ((n - m) * stride), offset=m * stride)
        self.buf_alive.write(np.ones(m, dtype=np.int32).tobytes())
        self.buf_alive.write(b"\x00" * ((n - m) * 4), offset=m * 4)
        self._alive_count = m

    # ------------------------------------------------------------------
    #  更新
    # ------------------------------------------------------------------

    def update(self, dt: float):
        """GPU 端更新全部存活粒子"""
        if self._alive_count == 0:
            return
        self.cs["u_dt"].value = float(dt)
        self.cs["u_wind_gravity"].value = (self.wind, self.gravity)
        self.cs["u_drag"].value = self.drag
        if self.attractor is not None:
            self.cs["u_attractor"].value = (self.attractor[0], self.attractor[1])
            self.cs["u_attractor_strength"].value = self.attractor[2]
            self.cs["u_attractor_radius"].value = max(1e-3, self.attractor[3])
        else:
            self.cs["u_attractor"].value = (0.0, 0.0)
            self.cs["u_attractor_strength"].value = 0.0
            self.cs["u_attractor_radius"].value = 1.0
        self.cs["u_alive_count"].value = self._alive_count
        self.cs.run(group_x=self._group_count)

    # ------------------------------------------------------------------
    #  回读
    # ------------------------------------------------------------------

    def read_particles(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """回读 (pos, alive_mask, life)，用于外部渲染/统计"""
        n = self._alive_count
        pos = np.frombuffer(self.buf_pos.read(n * 8),
                            dtype=np.float32).reshape(n, 2).copy()
        alive = np.frombuffer(self.buf_alive.read(n * 4),
                              dtype=np.int32).copy()
        life = np.frombuffer(self.buf_life.read(n * 4),
                             dtype=np.float32).copy()
        return pos, alive, life

    def to_particle_tuples(self) -> List[Tuple]:
        """转换为渲染器元组列表 [(x, y, size, r, g, b, alpha), ...]（存活粒子）"""
        n = self._alive_count
        pos = np.frombuffer(self.buf_pos.read(n * 8),
                            dtype=np.float32).reshape(n, 2)
        alive = np.frombuffer(self.buf_alive.read(n * 4), dtype=np.int32)
        life = np.frombuffer(self.buf_life.read(n * 4), dtype=np.float32)
        size = np.frombuffer(self.buf_size.read(n * 4), dtype=np.float32)
        color = np.frombuffer(self.buf_color.read(n * 12),
                              dtype=np.float32).reshape(n, 3)
        max_life = np.frombuffer(self.buf_max_life.read(n * 4), dtype=np.float32)
        result = []
        for i in range(n):
            if alive[i] == 0:
                continue
            alpha = float(max(0.0, min(1.0, life[i])))
            result.append((float(pos[i, 0]), float(pos[i, 1]),
                           float(size[i]),
                           float(color[i, 0]) * 255.0,
                           float(color[i, 1]) * 255.0,
                           float(color[i, 2]) * 255.0,
                           alpha))
        return result

    @property
    def particle_count(self) -> int:
        """存活粒子数（触发一次 alive 回读）"""
        if self._alive_count == 0:
            return 0
        alive = np.frombuffer(self.buf_alive.read(self._alive_count * 4),
                              dtype=np.int32)
        return int((alive == 1).sum())

    def release(self):
        """释放 GPU 资源"""
        for name in ("pos", "vel", "life", "max_life", "alive", "color", "size"):
            buf = getattr(self, f"buf_{name}", None)
            if buf is not None:
                buf.release()

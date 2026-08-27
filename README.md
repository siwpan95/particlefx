# Particle System（独立粒子系统）

## 设计理念

1. **独立性**：不依赖任何项目，纯 Python + NumPy 实现
2. **可复制性**：复制整个目录到目标项目，微调即可使用
3. **简洁性**：只提供核心功能，不添加冗余特性

## 项目结构

```
particle/
├── core/
│   ├── particle.py         # 核心粒子类（Particle/ParticleSystem/Emitter/ForceField/
│   │                       #   ParticleAttractor/ParticlePool/碰撞/光照/LOD/事件）
│   ├── particle_numpy.py   # NumPy 向量化粒子系统（SoA）
│   ├── particle_numba.py   # v5.0：Numba JIT 更新内核（可选依赖）
│   ├── particle_taichi.py  # v5.0：Taichi WCSPH 流体后端（可选依赖）
│   └── spatial_hash.py     # v5.0：均匀网格空间哈希
├── renderers/
│   ├── pygame_renderer.py      # Pygame 渲染器（additive/screen 混合 + Bloom）
│   ├── pil_renderer.py         # PIL 渲染器（离屏/GIF/视频帧/分层渲染）
│   ├── postfx.py               # v5.0：可编程后处理链（Bloom/色调映射/晕影/色差/颗粒）
│   ├── fluid_renderer.py       # CPU 折中版 SSFR 流体渲染
│   ├── particle_renderer_moderngl.py  # GPU 实例化粒子（含软粒子深度淡出）
│   ├── fluid_renderer_modernGL.py     # GPU 统一渲染器（4 种路径 + 软粒子）
│   └── particle_sim_gpu.py     # v5.0：Compute Shader 全 GPU 模拟
├── components/
│   ├── effects.py / effects_factory.py  # 效果工厂（代码 + JSON 声明式）
│   ├── effects_config.json              # 30+ 预设效果配方
│   ├── effects_config.schema.json       # v5.0：配方 JSON Schema
│   └── palettes.json                    # 主题调色板库
├── systems/
│   ├── fireworks_system.py   # 烟花（14 样式 × 10 调色板）
│   ├── weather_system.py     # 天气（6 种）
│   └── combat_effects.py     # 战斗特效（攻击/技能/飘字/死亡）
├── examples/                 # 交互 Demo + 视频生成器 + pygbag Web 入口
├── docs/                     # api.md / effects.md / enhancements.md
└── tests/                    # pytest 全套测试
```

## 核心类

### Particle（粒子）

单个粒子的数据结构：

```python
@dataclass
class Particle:
    x: float  # x 位置
    y: float  # y 位置
    vx: float  # x 速度
    vy: float  # y 速度
    life: float  # 剩余生命值 (0-1)
    max_life: float  # 最大生命值
    size: float  # 大小
    color: Tuple[int, int, int]  # RGB 颜色
    alpha: float = 1.0  # 透明度 (0-1)
```

### ParticleSystem（粒子系统管理器）

管理所有粒子、发射器和力场：

```python
ps = ParticleSystem(max_particles=1000)
ps.add_emitter(emitter)
ps.add_force_field(gravity)
ps.update(dt, audio_data)
ps.render(renderer)
```

### Emitter（发射器）

发射粒子：

```python
emitter = Emitter(
    x=width//2, y=height,
    rate=50.0,  # 发射速率（粒子/秒）
    spread=3.14,  # 扩散角度（弧度）
    speed=200.0,  # 发射速度
    life=2.0,  # 粒子生命值
    size=3.0,  # 粒子大小
    color=(255, 200, 50)  # 粒子颜色
)
```

### ForceField（力场）

模拟重力、风力、吸引力等：

```python
# 重力（向下）
gravity = ForceField(field_type='gravity', strength=100.0)

# 风力（向右）
wind = ForceField(field_type='wind', strength=50.0)

# 吸引力（指向某点）
attractor = ForceField(field_type='attraction', x=400, y=300, strength=200.0)
```

## 使用方法

### 1. 复制到目标项目

```bash
# 复制到 pywallpaper 项目
cp -r particle/ G:/repos/pywallpaper/particle/

# 复制到其他目标项目（按需微调）
cp -r particle/ <你的项目目录>/particle/
```

### 2. 微调适配

根据目标项目的架构进行微调：

- **pywallpaper**：已集成 `particle.py` 原语，可以直接使用
- **draw**：需要创建新的渲染器（`draw/renderers/particle_renderer.py`）

### 3. 使用示例

```python
from particle.core.particle import ParticleSystem, Emitter, ForceField
from particle.renderers.pygame_renderer import PygameRenderer

# 创建粒子系统
ps = ParticleSystem(max_particles=1000)

# 创建发射器
emitter = Emitter(x=400, y=300, rate=50.0, ...)
ps.add_emitter(emitter)

# 创建力场
gravity = ForceField(field_type='gravity', strength=100.0)
ps.add_force_field(gravity)

# 更新和渲染
ps.update(dt)
renderer = PygameRenderer(screen)
ps.render(renderer)
```

## 示例效果

### 烟花（Fireworks）

```bash
cd examples/
python fireworks.py
```

### 其他效果（待创建）

- 雨水（Rain）
- 雪花（Snow）
- 爆炸（Explosion）
- 魔法效果（Magic）

## 音频驱动

粒子系统支持音频驱动：

```python
# 更新时传入音频数据
audio_data = {
    'rms': 0.5,  # 音频 RMS 能量
    'beat': True,  # 是否检测到节拍
    'band_energy': [0.1, 0.2, 0.3]  # 频段能量
}
ps.update(dt, audio_data)
```

在 `Emitter.emit()` 中，可以根据 `audio_data` 调整发射速率：

```python
if audio_data:
    energy = audio_data.get('rms', 1.0)
    current_rate *= (1.0 + energy * 2.0)  # 根据音频能量调整发射速率
```

## 性能优化

### NumPy 向量化

将粒子数据存储在 NumPy 数组中，使用向量化计算：

```python
# TODO: 创建 ParticleSystemNumpy 类
class ParticleSystemNumpy:
    def __init__(self):
        self.positions = np.zeros((max_particles, 2))  # x, y
        self.velocities = np.zeros((max_particles, 2))  # vx, vy
        self.lives = np.zeros(max_particles)  # life
        # ...
```

### Numba JIT 编译

使用 Numba 编译粒子更新循环：

```python
from numba import jit

@jit(nopython=True)
def update_particles(positions, velocities, lives, dt):
    # ...
```

## 依赖

- **必需**：NumPy
- **可选**：Pygame（用于 `pygame_renderer.py`）、Pillow（用于 `pil_renderer.py`）

## v5.0 增强功能（2026）

按"成本从低到高"落地的 9 项增强，全部保持**可选依赖零开销 + 向后兼容**：

| # | 增强 | 入口 | 可选依赖 |
|---|------|------|---------|
| ① | 对象池 + 空间哈希 | `ParticleSystem(use_pool=True, use_spatial_hash=True)` | 无 |
| ② | Numba JIT 更新内核 | `core.particle_numba.NumbaParticleSystem` | numba |
| ③ | 软粒子深度淡出 | `GPUParticleRenderer(enable_depth_fade=True)` | moderngl |
| ④ | 浏览器 Web 导出 | `pygbag --app_name web_main examples` | pygbag |
| ⑤ | 粒子吸引子 + 粒子光源 | `ParticleAttractor` / `enable_pairwise_force` / `enable_particle_lights` | 无 |
| ⑥ | 可编程后处理链 | `renderers.postfx.PostFXChain` | 无 |
| ⑦ | JSON Schema 校验 | `EffectsFactory(validate=True)` | jsonschema(可选) |
| ⑧ | Taichi SPH 流体 | `core.particle_taichi.TaichiSPHSystem` | taichi |
| ⑨ | Compute 全 GPU 模拟 | `renderers.particle_sim_gpu.GPUParticleSimulator` | moderngl |

详见 [docs/enhancements.md](docs/enhancements.md)。

## 许可证

MIT License（可自由复制、修改、使用）

## 作者

- 全哥（独立开发者）
- AI 助手（WorkBuddy）

## 更新日志

- **2026-06-23**：初始版本，创建核心类和示例效果

# Particle System API 参考（增强版 v2.0）

## 核心模块 (`core/particle`)

### 类总览

| 类 | 说明 |
|---|------|
| `Particle` | 单个粒子，支持 Color/Size/Rotation over Lifetime |
| `ParticleSystem` | 粒子系统管理器，支持子发射器、噪声场 |
| `Emitter` | 粒子发射器，支持初始旋转、生命曲线 |
| `ForceField` | 力场（重力/风力/吸引力/噪声场） |
| `NoiseField` | 独立噪声场（Perlin 简化实现） |
| `Renderer` | 渲染器接口基类 |

---

## `Particle` — 单个粒子（增强版）

```python
@dataclass
class Particle:
    x: float          # x 位置
    y: float          # y 位置
    vx: float         # x 方向速度
    vy: float         # y 方向速度
    life: float       # 剩余生命值 (0~1，0=死亡)
    max_life: float   # 最大生命值（秒）
    size: float       # 当前大小（像素）
    color: tuple      # RGB 颜色 (R, G, B)
    alpha: float = 1.0  # 透明度 (0~1)

    # === 新增（均为可选）===
    rotation: float = 0.0          # 当前旋转角度（弧度）
    angular_velocity: float = 0.0   # 角速度（弧度/秒）
    mass: float = 1.0              # 质量（用于力场计算）

    # === 生命周期曲线（可选）===
    color_over_lifetime:  Optional[list]  # 颜色渐变曲线
    size_over_lifetime:  Optional[list]   # 大小曲线
    velocity_over_lifetime: Optional[list]   # 速度倍率曲线

    # === 子发射器（内部使用）===
    _sub_emit_on_death:  Optional[ParticleSystem]
    _sub_emit_on_birth: Optional[ParticleSystem]
```

**新增方法：** 无（所有特性在 `update(dt)` 中自动应用）

### 生命周期曲线格式

**Color over Lifetime：** `[(t0, (r,g,b)), (t1, (r,g,b)), ...]`
- `t` ∈ [0, 1]，0=诞生，1=死亡
- 系统自动在相邻点之间线性插值

**Size over Lifetime：** `[(t0, s0), (t1, s1), ...]`
- 最终粒子大小 = 曲线值（不是倍率）

**Velocity over Lifetime：** `[(t0, (vx_mult, vy_mult)), ...]`
- 速度倍率，1.0 = 保持原速

### 示例

```python
from core.particle import Particle, FIRE_COLOR_CURVE, gradient_to_curve

# 颜色随生命变化（使用预设曲线）
p = Particle(
    x=0, y=0, vx=5, vy=3,
    life=1.0, max_life=2.0,
    size=3.0, color=(255, 200, 50),
    color_over_lifetime=FIRE_COLOR_CURVE,  # 火焰渐变
)

# 大小随生命脉冲（先膨胀后缩小）
p = Particle(
    # ...
    size_over_lifetime=[
        (0.0, 1.0),   # 诞生时正常大小
        (0.2, 3.0),   # 快速膨胀到 3 倍
        (1.0, 0.0),   # 死亡时缩小到 0
    ],
)

# 旋转粒子
p = Particle(
    # ...
    rotation=0.0,           # 初始角度 0°
    angular_velocity=math.pi,  # 每秒旋转 180°
)
```

---

## `ParticleSystem` — 系统管理器（增强版）

```python
system = ParticleSystem(max_particles=1000)

# 添加组件
system.add_emitter(emitter)
system.add_force_field(force_field)
system.add_noise_field(noise_field)  # 新增！

# 每帧调用
system.update(dt=0.016)          # dt 为时间步长（秒）
system.render(renderer)            # 渲染

# 事件回调（新增！）
system.on_particle_birth = callback_fn
system.on_particle_death = callback_fn

# 辅助
system.clear()                     # 清空所有粒子
system.get_particle_count() -> int # 当前粒子数量
system.time  -> float              # 当前系统时间（秒，用于噪声场）
```

### 子发射器（新增特性）

粒子死亡或生成时，自动触发另一个 `ParticleSystem`：

```python
# 创建主粒子系统
main_sys = ParticleSystem(max_particles=500)

# 创建子粒子系统（火花效果）
spark_sys = ParticleSystem(max_particles=200)
spark_emit = Emitter(x=0, y=0, rate=100, ...)
spark_sys.add_emitter(spark_emit)

# 绑定子发射器（粒子死亡时触发）
main_emit = Emitter(
    x=0, y=0, rate=10,
    sub_emitter_on_death=spark_sys,  # 关键！
)
main_sys.add_emitter(main_emit)
```

---

## `Emitter` — 粒子发射器（增强版）

```python
emitter = Emitter(
    x=400, y=300,
    rate=50,                # 发射速率（粒子/秒）
    spread=0.5,             # 扩散角度（弧度）
    speed=5.0,              # 发射速度
    life=1.0,              # 粒子生命值（秒）
    size=2.0,              # 粒子大小
    color=(255, 255, 255), # RGB 颜色

    # === 新增参数 ===
    rotation_range=(0.0, 0.0),          # 初始旋转角度范围（弧度）
    angular_velocity_range=(0.0, 0.0),  # 角速度范围
    mass_range=(1.0, 1.0),           # 质量范围

    # 生命曲线（传递给生成的粒子）
    color_over_lifetime=None,         # 颜色曲线
    size_over_lifetime=None,          # 大小曲线
    velocity_over_lifetime=None,      # 速度倍率曲线

    # 子发射器
    sub_emitter_on_death=None,        # 粒子死亡时触发
    sub_emitter_on_birth=None,        # 粒子生成时触发
)
```

---

## `ForceField` — 力场（增强版）

```python
# 重力场（向下）
gravity = ForceField(field_type='gravity', strength=5.0)

# 风力场（向右）
wind = ForceField(field_type='wind', strength=2.0)

# 吸引力场（指向某点）
attraction = ForceField(
    field_type='attraction',
    x=400, y=300,         # 吸引中心
    strength=3.0,
)

# 噪声场（新增！）
noise = ForceField(
    field_type='noise',
    strength=1.0,           # 噪声强度
    noise_scale=50.0,        # 噪声空间缩放（越大越平滑）
    noise_strength=1.0,      # 噪声强度（独立控制）
    noise_octaves=2,         # 八度数（越大细节越多）
)
```

### 噪声场说明

- 基于多层正弦波叠加，模拟 Perlin 噪声效果
- 随时间自动变化（调用 `system.update(dt)` 时自动推进）
- 适合模拟：火焰摇曳、烟雾飘散、水流湍流

---

## `NoiseField` — 独立噪声场类

```python
from core.particle import NoiseField

nf = NoiseField(
    strength=1.0,    # 噪声强度
    scale=50.0,      # 噪声空间缩放
    octaves=2,        # 八度数
    seed=None,        # 随机种子（None=随机）
)

# 手动推进时间（让噪声"流动"）
nf.update(dt)

# 计算粒子受力
fx, fy = nf.get_force(particle)
```

---

## 曲线构建辅助函数

### `gradient_to_curve(colors, num_samples=8)`

将颜色列表转换为颜色生命曲线。

```python
from core.particle import gradient_to_curve

# 火焰效果：白→黄→橙→暗红
flame_curve = gradient_to_curve([
    (255, 255, 240),  # t=0.0: 白热
    (255, 200, 50),    # t=0.33: 亮黄
    (255, 80, 10),     # t=0.66: 红橙
    (80, 20, 5),       # t=1.0: 暗红（余烬）
])

# 在 Emitter 中使用
emitter = Emitter(
    # ...
    color_over_lifetime=flame_curve,
)
```

### `size_curve_fadeout(base_size)`

大小淡出曲线（诞生时最大，死亡时缩小到 0）。

```python
from core.particle import size_curve_fadeout

emitter = Emitter(
    # ...
    size_over_lifetime=size_curve_fadeout(3.0),  # 基准大小 3.0
)
```

### `size_curve_pulse(base_size, peak_at=0.3, peak_mult=1.5)`

大小脉冲曲线（先膨胀后缩小）。

```python
from core.particle import size_curve_pulse

# 粒子在生命进度 30% 时膨胀到 1.5 倍，然后缩小
emitter = Emitter(
    # ...
    size_over_lifetime=size_curve_pulse(2.0, peak_at=0.3, peak_mult=2.0),
)
```

### `velocity_curve_accelerate(mult=1.5)`

速度加速曲线。

```python
from core.particle import velocity_curve_accelerate

emitter = Emitter(
    # ...
    velocity_over_lifetime=velocity_curve_accelerate(2.0),  # 结束时速度 2 倍
)
```

### `velocity_curve_decelerate(mult=0.3)`

速度减速曲线。

```python
from core.particle import velocity_curve_decelerate

emitter = Emitter(
    # ...
    velocity_over_lifetime=velocity_curve_decelerate(0.5),  # 结束时速度减半
)
```

---

## 预设曲线

模块内置了常用曲线，可直接使用：

```python
from core.particle import (
    FIRE_COLOR_CURVE,   # 火焰颜色渐变
    SMOKE_COLOR_CURVE,  # 烟雾颜色渐变
    MAGIC_COLOR_CURVE,  # 魔法颜色渐变
    EXPLOSION_SIZE_CURVE,  # 爆炸大小曲线
    SPARK_SIZE_CURVE,  # 火花大小曲线
)
```

| 预设 | 类型 | 说明 |
|------|------|------|
| `FIRE_COLOR_CURVE` | 颜色曲线 | 白热→亮黄→橙红→暗红 |
| `SMOKE_COLOR_CURVE` | 颜色曲线 | 深灰→浅灰→透明 |
| `MAGIC_COLOR_CURVE` | 颜色曲线 | 蓝紫→亮蓝→白 |
| `EXPLOSION_SIZE_CURVE` | 大小曲线 | 瞬间膨胀→快速缩小 |
| `SPARK_SIZE_CURVE` | 大小曲线 | 快速缩小到消失 |

---

## 渲染器模块 (`renderers/`)

### Pygame 渲染器（增强版）

```python
from renderers.pygame_renderer import PygameRenderer

renderer = PygameRenderer(screen)  # screen 是 pygame.Surface

# 在主循环中：
system.update(dt)
system.render(renderer)
```

**增强特性：**
- 支持绘制粒子旋转指示线（当 `angular_velocity != 0` 时自动显示）
- 指示线颜色稍暗于粒子颜色，长度 = `size * 1.8`

### PIL 渲染器（增强版）

```python
from renderers.pil_renderer import PILRenderer

renderer = PILRenderer(width=800, height=600)

# 每帧渲染
image = system.render(renderer)  # 返回 Image.Image
final_image = renderer.get_image()
```

**增强特性：**
- 支持绘制粒子旋转指示线（RGBA 模式，透明度随粒子 alpha 变化）

---

## 效果库 (`components/effects`)

详见 [效果使用指南](effects.md)。

### 快速使用

```python
from components.effects import get_effect, list_effects

# 查看可用效果
print(list_effects())
# => ['explosion', 'fire', 'rain', 'snow', 'magic_circle',
#     'fountain', 'smoke', 'sparkle', 'trail']

# 创建效果
system = get_effect('explosion', x=400, y=300, intensity=2.0)

# 或直接导入具体函数
from components.effects import create_fire, create_rain

fire = create_fire(x=100, y=500)
rain_system = create_rain(width=800, height=600)
```

### 效果特性对照表

| 效果 | Color over LT | Size over LT | Rotation | Noise Field |
|------|----------------|---------------|----------|-------------|
| `create_explosion` | ✅ `FIRE_COLOR_CURVE` | ✅ `EXPLOSION_SIZE_CURVE` | ❌ | ❌ |
| `create_fire` | ✅ `FIRE_COLOR_CURVE` | ✅ pulse | ❌ | ✅ |
| `create_smoke` | ✅ `SMOKE_COLOR_CURVE` | ✅ fadeout | ✅ | ✅ |
| `create_snow` | ❌ | ❌ | ✅ | ❌ |
| `create_magic_circle` | ✅ `MAGIC_COLOR_CURVE` | ❌ | ✅ | ❌ |
| `create_fountain` | ✅ water gradient | ❌ | ❌ | ❌ |
| `create_sparkle` | ❌ | ✅ `SPARK_SIZE_CURVE` | ❌ | ❌ |
| `create_trail` | ❌ | ❌ | ❌ | ❌ |

---

## 组合系统 (`systems/`)

### FireworksSystem — 烟花系统

```python
from systems.fireworks_system import FireworksSystem

fw = FireworksSystem(width=800, height=600, auto_launch=True)

while running:
    fw.update(dt)
    fw.render(renderer)

# 手动发射
fw.launch(x=400, palette_name='rainbow', style='chrysanthemum')
```

### WeatherSystem — 天气系统

```python
from systems.weather_system import WeatherSystem

weather = WeatherSystem(width=800, height=600, initial_weather='clear')

# 切换天气
weather.set_weather('rain')        # 小雨
weather.set_weather('storm')       # 风暴
weather.set_weather('heavy_snow')  # 大雪

while running:
    weather.update(dt)
    weather.render(renderer)
```

### CombatEffectsManager — 战斗特效

```python
from systems.combat_effects import CombatEffectsManager

mgr = CombatEffectsManager()

# 攻击命中
mgr.hit_effect(x=200, y=300, effect_type='slash')
mgr.hit_effect(x=200, y=300, effect_type='crit')

# 伤害数字
mgr.floating_damage(x=200, y=280, value=1234, is_crit=True)

# 技能释放
mgr.skill_cast(x=300, y=200, skill_type='fireball')

while running:
    mgr.update(dt)
    mgr.render(renderer)
```

---

## v5.0 新增 API 速查

详见 [enhancements.md](enhancements.md)。全部为可选特性，未启用时零开销。

### 对象池 + 空间哈希

```python
from core.particle import ParticleSystem

ps = ParticleSystem(max_particles=2000,
                    use_pool=True,            # 粒子对象池复用
                    use_spatial_hash=True,     # 碰撞/邻居查询加速
                    spatial_hash_cell_size=64.0)
neighbors = ps.query_neighbors(x=400, y=300, radius=80.0)
```

### 粒子吸引子 / 粒子间作用力 / 粒子光源

```python
from core.particle import ParticleAttractor

# 跟随鼠标的吸引子
attr = ParticleAttractor(x=400, y=300, strength=8.0, radius=250.0,
                         falloff='linear', polarity=1.0,
                         follow=lambda: pygame.mouse.get_pos())
ps.add_force_field(attr)

# 粒子间吸引 / 排斥（空间哈希加速）
ps.enable_pairwise_force(strength=15.0, radius=80.0, repulsion=False)

# 最亮自发光粒子成为动态点光源
ps.enable_particle_lights(max_lights=8, min_emissive_strength=0.5)
```

### Numba / Taichi 后端

```python
from core.particle_numba import NumbaParticleSystem
sys_np = NumbaParticleSystem(max_particles=5000, gravity=3.0, drag=0.99)

from core.particle_taichi import TaichiSPHSystem
sph = TaichiSPHSystem(max_particles=4096, width=800, height=600)
sph.add_fluid_block(x=300, y=80, w=120, h=120, spacing=6.0)
sph.update(1/60)
img = sph.render_to_image(width=800, height=600)
```

### 后处理链

```python
from renderers.postfx import PostFXChain, default_fireworks_chain
from renderers.pil_renderer import LayeredPILRenderer

r = LayeredPILRenderer(width=1280, height=720, bloom=True)
frame = r.render_to_image(particles, background=bg,
                          postfx=default_fireworks_chain())
```

### 软粒子深度淡出 / GPU 模拟

```python
# 软粒子（GPUParticleRenderer 同理）
renderer = UnifiedGPURenderer(screen, mode='hybrid', enable_depth_fade=True)
renderer.enable_depth_fade_with_fluid_depth(fade_range=0.05)

# Compute Shader 全 GPU 模拟
from renderers.particle_sim_gpu import GPUParticleSimulator
sim = GPUParticleSimulator(ctx, max_particles=10000, gravity=500.0)
sim.emit_burst(x=400, y=300, count=5000)
sim.update(1/60)
renderer.render(sim.to_particle_tuples())
```

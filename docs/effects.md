# 效果使用指南

## 一览表

| 效果 | 函数 | 适用场景 |
|------|------|---------|
| 爆炸 | `create_explosion()` | 攻击命中、物体破坏 |
| 火焰 | `create_fire()` | 篝火、燃烧状态、魔法火焰 |
| 雨 | `create_rain()` | 天气背景 |
| 雪 | `create_snow()` | 天气背景 |
| 魔法光环 | `create_magic_circle()` | 技能施法、传送门 |
| 喷泉 | `create_fountain()` | 装饰场景 |
| 烟雾 | `create_smoke()` | 烟熏、爆炸残留、蒸汽 |
| 星光闪烁 | `create_sparkle()` | 拾取物品、UI 强调 |
| 尾迹 | `create_trail()` | 子弹拖尾、角色移动 |

---

## 使用方式

### 方式一：直接调用（推荐）

```python
from components.effects import create_explosion, create_fire

# 创建效果，拿到配置好的 ParticleSystem
explosion = create_explosion(x=400, y=300, intensity=2.0)
fire = create_fire(x=100, y=500)

# 每帧更新和渲染
while running:
    explosion.update(dt)
    fire.update(dt)

    explosion.render(renderer)
    fire.render(renderer)
```

### 方式二：按名称获取

```python
from components.effects import get_effect, list_effects

# 列出所有可用效果
print(list_effects())

# 动态创建
system = get_effect('snow', width=800, height=600, density=1.5)
```

---

## 各效果详细说明

### create_explosion — 爆炸

```python
system = create_explosion(
    x=400, y=300,           # 爆炸中心
    intensity=1.0,          # 强度倍数（粒子数量和大小）
    color_palette=[          # 自定义颜色（可选）
        (255, 200, 50),     # 默认为火焰色调
        (255, 120, 20),
        (255, 60, 10),
    ]
)
```

**组成：** 高速向外喷射的主爆粒子 + 缓慢下落的残骸 + 重力场

### create_fire — 火焰

```python
fire = create_fire(
    x=400, y=500,           # 火焰底部中心
    width=40,               # 火焰宽度
    height=80,              # 火焰高度
    intensity=1.0,          # 强度
)
```

**组成：** 5 层不同颜色/速度的发射器（内白→黄→橙→红→暗红）+ 向上热气流力场

### create_rain / create_snow — 天气

```python
rain = create_rain(
    width=800,              # 场景宽度
    height=600,             # 场景高度
    density=1.0,            # 密度
    wind=2.0,               # 风力（向右）
    color=(150, 180, 255),  # 雨滴颜色
)

snow = create_snow(
    width=800,
    height=600,
    density=0.5,            # 雪通常比雨稀疏
    wind=1.0,
)
```

> **推荐：** 如果需要完整天气管理（切换、渐变），请使用 `systems/weather_system.py` 的 `WeatherSystem`。

### create_magic_circle — 魔法光环

```python
circle = create_magic_circle(
    x=400, y=300,           # 圆心
    radius=80,              # 光环半径
    rotation_speed=1.0,     # 旋转速度
    color=(100, 150, 255),  # 主色调
)
```

**组成：** 3 层圆环发射器 + 中心吸引力场

### create_fountain — 喷泉

```python
fountain = create_fountain(
    x=400, y=550,           # 喷口位置
    height=120,             # 最大喷射高度
    spread=0.4,             # 扩散角度
)
```

**组成：** 主水流 + 两侧溅射水花 + 底部回落 + 重力场

### create_smoke — 烟雾

```python
smoke = create_smoke(
    x=200, y=300,           # 烟源位置
    direction='up',         # 方向: up/down/left/right
    density=1.0,            # 密度
)
```

**特点：** 大粒子、长生命、缓慢扩散、灰色调

### create_sparkle — 星光闪烁

```python
sparkle = create_sparkle(
    x=400, y=300,           # 区域中心
    area_size=60,           # 效果区域大小
    intensity=1.0,          # 强度
)
```

**适用场景：** 物品拾取提示、按钮高亮、宝箱开启等

### create_trail — 尾迹

```python
trail = create_trail(
    x=400, y=300,           # 起点
    direction_x=1.0,        # 运动方向 X
    direction_y=0.0,        # 运动方向 Y
    color=(255, 200, 100),  # 尾迹颜色
    length=1.0,             # 尾迹长度
)

# 在移动循环中持续更新尾迹位置：
trail.emitters[0].set_position(bullet_x, bullet_y)  # 追踪移动物体
```

---

## 性能参考

| 效果 | 典型粒子数 | max_particles |
|------|-----------|---------------|
| explosion (×1) | ~300 | 300 |
| fire (×1) | ~200-400 | 400 |
| rain (×1) | ~200-500 | 500 |
| snow (×0.5) | ~150 | 300 |
| magic_circle | ~120 | 500 |
| fountain | ~250-400 | 600 |
| smoke (×1) | ~150 | 250 |
| sparkle (×1) | ~30-80 | 100 |
| trail (×1) | ~60-200 | 200 |

多个效果同时运行时注意总粒子数。建议在低端设备上降低 `intensity` / `density` 参数。

---

## 自定义颜色调色板

大部分效果接受 `color_palette` 或 `color` 参数。常用调色板参考：

```python
# 火焰色系
FIRE_PALETTE = [
    (255, 255, 150),  # 内核白黄
    (255, 200, 50),   # 亮黄
    (255, 120, 20),   # 橙色
    (255, 60, 0),     # 红橙
]

# 冰霜色系
ICE_PALETTE = [
    (180, 220, 255),  # 冰蓝
    (150, 200, 255),
    (220, 240, 255),
    (100, 180, 255),
]

# 自然绿色
NATURE_GREEN = [
    (50, 255, 100),
    (100, 255, 120),
    (150, 255, 180),
]

# 暗黑紫色（魔法/毒）
DARK_MAGIC = [
    (180, 50, 255),
    (140, 30, 200),
    (100, 20, 160),
]
```

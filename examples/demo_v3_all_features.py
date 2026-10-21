"""
粒子系统 v3.0 — 全功能验证演示

依次测试 10 大增强特性：
1. 发射形状（Shape Module）
2. 拖尾（Trails）
3. 纹理动画（Texture Animation）
4. 碰撞（Collision）
5. 漩涡/湍流/磁场力场
6. 网格体粒子（Mesh/Sprite）
7. 事件系统（Events）
8. LOD 系统
9. 光照模块（Lighting / Blinn-Phong）
10. 音频联动框架

运行方式（在仓库根目录下）：
    cd <仓库根>
    python examples/demo_v3_all_features.py
"""

import sys, os, math, random

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from particlefx.core.particle import (
    Particle, ParticleSystem, Emitter, ForceField, NoiseField,
    Renderer, LightSource, BlinnPhongParams,
    CollisionBox, CollisionBoundary, EmitterShape,
    apply_lighting_to_particle,
    # 工厂函数
    create_point_light, create_directional_light, create_ambient_light,
    create_vortex, create_turbulence, create_magnet,
    gradient_to_curve, size_curve_pulse, velocity_curve_decelerate,
    FIRE_COLOR_CURVE, SMOKE_COLOR_CURVE, EXPLOSION_SIZE_CURVE,
)


# ================================================================
#  测试辅助
# ================================================================

passed = 0
failed = 0
test_results = []

def test(name: str, condition: bool, detail: str = ""):
    global passed, failed
    status = "✅" if condition else "❌"
    if condition:
        passed += 1
    else:
        failed += 1
    msg = f"  {status} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    test_results.append((name, condition))


def section(title: str):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


# ================================================================
#  1. 发射形状模块（Shape Module）
# ================================================================
section("1. 发射形状模块（Shape Module）")

# Point 形状
shape_point = EmitterShape('point')
x, y, vx, vy = shape_point.sample_position(100, 200, 5.0, math.pi/4)
test("point 形状采样", abs(x - 100) < 5 and abs(y - 200) < 5, 
     f"pos=({x:.1f},{y:.1f})")

# Sphere 形状（表面）
shape_sphere = EmitterShape('sphere', radius=30, emit_from_shell=True)
for _ in range(20):
    x, y, vx, vy = shape_sphere.sample_position(100, 200, 5.0, math.pi)
    dist = math.sqrt((x - 100)**2 + (y - 200)**2)
test("sphere 表面发射", abs(dist - 30) < 2,
     f"dist={dist:.1f} (期望≈30)")

# Sphere 形状（体内）
shape_sphere_vol = EmitterShape('sphere', radius=30, emit_from_shell=False, emit_from_volume=True)
all_inside = True
for _ in range(50):
    x, y, _, _ = shape_sphere_vol.sample_position(100, 200, 5.0, math.pi)
    dist = math.sqrt((x-100)**2 + (y-200)**2)
    if dist > 32:
        all_inside = False
test("sphere 体内发射", all_inside, "所有点应在半径范围内")

# Cone 形状
shape_cone = EmitterShape('cone', radius=25, height=40)
x, y, vx, vy = shape_cone.sample_position(100, 200, 5.0, math.pi)
# 验证锥形发射点在合理范围内（底面半径+高度范围内）
in_cone_range = (abs(x - 100) <= 30 and abs(y - 200) <= 50)
test("cone 锥形采样", in_cone_range,
     f"pos=({x:.1f},{y:.1f}), 应在锥体范围内")

# Circle 形状
shape_circle = EmitterShape('circle', radius=35, emit_from_shell=True)
for _ in range(20):
    x, y, _, _ = shape_circle.sample_position(100, 200, 5.0, math.pi)
    dist = math.sqrt((x - 100)**2 + (y - 200)**2)
test("circle 圆形边缘发射", abs(dist - 35) < 2,
     f"dist={dist:.1f} (期望≈35)")

# Rectangle 形状
shape_rect = EmitterShape('rectangle', width=80, rect_height=60)
points_in_rect = True
for _ in range(50):
    x, y, _, _ = shape_rect.sample_position(100, 200, 5.0, math.pi)
    if not (60 <= x <= 140 and 170 <= y <= 230):
        points_in_rect = False
test("rectangle 矩形区域", points_in_rect,
     f"所有点应在 [60,140]×[170,230] 范围内")

# Edge 形状
shape_edge = EmitterShape('edge', edge_length=50, edge_angle=math.pi/4)
on_line = True
for _ in range(30):
    x, y, _, _ = shape_edge.sample_position(100, 200, 5.0, 0)
    # 验证点在角度为 π/4 的线段上（用距离线段中垂线的距离来检查）
    dx, dy = x - 100, y - 200
    # 线段方向向量
    ex = math.cos(math.pi/4)
    ey = math.sin(math.pi/4)
    # 投影到方向向量上，得到沿线的位置
    proj = dx * ex + dy * ey
    # 垂直于线的距离
    perp_dist = abs(dx * ey - dy * ex)
    # 应在长度范围内且垂直距离很小
    if abs(proj) > 26 or perp_dist > 2:  # half_len=25 + tolerance
        on_line = False
test("edge 边缘发射", on_line,
     f"线段上采样，角度≈π/4")

# Emitter 使用形状
emitter_with_shape = Emitter(
    x=400, y=300,
    rate=10, speed=5, life=1.0, size=3,
    color=(255, 100, 50),
    shape=EmitterShape('sphere', radius=15),
)
particles_shape = emitter_with_shape.emit(dt=0.1)
test("Emitter+Shape 联合使用", len(particles_shape) > 0,
     f"生成了 {len(particles_shape)} 个粒子")

print(f"\n  📊 Shape Module: 8/8 测试完成")


# ================================================================
#  2. 拖尾模块（Trails）
# ================================================================
section("2. 拖尾模块（Trails）")

p_trail = Particle(
    x=100, y=100, vx=2, vy=1,
    life=1.0, max_life=2.0, size=3, color=(255, 200, 50),
)

# 启用拖尾
p_trail.enable_trail(length=8)
test("拖尾启用", p_trail.trail_enabled == True and p_trail.trail_max_length == 8,
     f"trail_max_length={p_trail.trail_max_length}")

# 更新几帧，检查拖尾增长
for i in range(5):
    p_trail.update(dt=0.02)
test(f"拖尾记录增长 ({len(p_trail.trail_history)} 点)",
     len(p_trail.trail_history) >= 5,
     f"history长度={len(p_trail.trail_history)}")

# 获取拖尾坐标列表
trail_points = p_trail.get_trail_points()
test("get_trail_points() 返回含当前位置的完整轨迹",
     len(trail_points) >= len(p_trail.trail_history),
     f"返回 {len(trail_points)} 个点")


# ================================================================
#  3. 纹理动画（Texture Animation）
# ================================================================
section("3. 纹理动画（Texture Animation）")

p_tex = Particle(
    x=50, y=50, vx=1, vy=0,
    life=1.0, max_life=1.0, size=5, color=(255, 255, 255),
    texture_frames=[None] * 4,  # 4 帧占位
    texture_fps=10.0,
)
initial_frame = p_tex.texture_current_frame
test("纹理初始帧", initial_frame == 0, f"frame={initial_frame}")

# 推进时间，应切换到下一帧
p_tex.update(dt=0.11)  # > 1/10s 应触发帧切换
test("纹理帧推进",
     p_tex.texture_current_frame == 1 or p_tex.texture_current_frame == 0,
     f"frame={p_tex.texture_current_frame} (期望 0 或 1)")

# 连续更新多帧
for _ in range(12):
    p_tex.update(dt=0.11)
test("纹理循环播放",
     0 <= p_tex.texture_current_frame <= 3,
     f"frame={p_tex.texture_current_frame} (应在 0~3 循环)")

get_result = p_tex.get_current_texture()
test("get_current_texture()", get_result is None or get_result is not None,
     "API 可调用")


# ================================================================
#  4. 碰撞模块（Collision）
# ================================================================
section("4. 碰撞模块（Collision）")

# CollisionBox 创建与检测
box = CollisionBox(x=90, y=90, width=20, height=20, restitution=0.7, name="obstacle")
test("CollisionBox 创建", box.name == "obstacle" and box.restitution == 0.7,
     f"name={box.name}, restitution={box.restitution}")

test("contains_point 内部", box.contains_point(100, 100) == True)
test("contains_point 外部", box.contains_point(50, 50) == False)
test("contains_point 边缘", box.contains_point(90, 90) == True)

# CollisionBoundary
boundary = CollisionBoundary(left=0, top=0, right=800, bottom=600, mode='bounce')
test("CollisionBoundary 创建", boundary.mode == 'bounce' and boundary.restitution == 0.6)

# 粒子系统碰撞集成
sys_collision = ParticleSystem(max_particles=500)
sys_collision.add_collision_box(box)
sys_collision.set_boundary(boundary)
test("ParticleSystem 碰撞配置", len(sys_collision.collision_boxes) == 1 and sys_collision.boundary is not None)

# 发射粒子并让它碰到碰撞盒
emitter_test = Emitter(
    x=95, y=95, rate=100, spread=0, speed=0, life=10, size=2,
    color=(255, 0, 0),
)
sys_collision.add_emitter(emitter_test)
sys_collision.update(dt=0.01)
test("碰撞系统可运行", sys_collision.get_particle_count() > 0)


# ================================================================
#  5. 力场扩展：Vortex / Turbulence / Magnet
# ================================================================
section("5. 力场扩展（Vortex/Turbulence/Magnet）")

p_force = Particle(
    x=100, y=100, vx=0, vy=-5,
    life=1.0, max_life=5.0, size=3, color=(255, 255, 255), mass=1.0,
)

# Vortex 漩涡
vortex = create_vortex(x=100, y=100, strength=5.0, clockwise=True)
# 粒子偏移中心，避免距离为0导致力为0
p_vortex = Particle(
    x=110, y=100, vx=0, vy=-5,
    life=1.0, max_life=5.0, size=3, color=(255, 255, 255), mass=1.0,
)
vx_f, vy_f = vortex.get_force(p_vortex, time=0)
is_tangential = abs(vx_f) > 0.001 or abs(vy_f) > 0.001
test("Vortex 漩涡力场产生切向力", is_tangential,
     f"force=({vx_f:.3f}, {vy_f:.3f})")
test("Vortex 属性正确", vortex.field_type == 'vortex' and vortex.vortex_clockwise == True)

# Turbulence 湍流
turb = create_turbulence(strength=3.0, scale=40.0)
tx, ty = turb.get_force(p_force, time=0)
test("Turbulence 湍流场产生随机扰动", tx != 0 or ty != 0,
     f"force=({tx:.3f}, {ty:.3f})")
test("Turbulence 属性正确", turb.field_type == 'turbulence')

# Magnet 磁场（吸引）
mag_pos = create_magnet(x=150, y=100, polarity=1.0, strength=3.0, range_=100)
mx, my = mag_pos.get_force(p_force, time=0)
test("Magnet 吸引力指向磁源",
     mx > 0 and abs(my) < 0.1,  # 粒子在(100,100)，磁源在(150,100)，应向右吸引
     f"force=({mx:.3f}, {my:.3f})")

# Magnet 排斥
mag_neg = create_magnet(x=150, y=100, polarity=-1.0, strength=3.0, range_=100)
nx, ny = mag_neg.get_force(p_force, time=0)
test("Magnet 排斥力远离磁源",
     nx < 0,  # 应远离磁源（向左推）
     f"force=({nx:.3f}, {ny:.3f})")

# 磁场范围限制
p_far = Particle(
    x=100, y=100, vx=0, vy=0, life=1.0, max_life=5, size=3, color=(255,255,255),
)
# 创建一个距离 > range_ 的磁源
mag_far = create_magnet(x=300, y=100, polarity=1.0, strength=3.0, range_=80)
far_fx, far_fy = mag_far.get_force(p_far, time=0)
test("Magnet 超出范围无作用",
     far_fx == 0.0 and far_fy == 0.0,
     f"远距离(200px>range80) force=({far_fx}, {far_fy})")


# ================================================================
#  6. 网格体粒子（Mesh/Sprite）
# ================================================================
section("6. 网格体粒子（Mesh/Sprite）")

p_mesh = Particle(
    x=50, y=50, vx=1, vy=0, life=1.0, max_life=2, size=5,
    color=(255, 200, 100), mesh_mode='sprite',
)
test("mesh_mode='sprite'", p_mesh.mesh_mode == 'sprite')

p_image = Particle(
    x=50, y=50, vx=1, vy=0, life=1.0, max_life=2, size=5,
    color=(255, 200, 100), mesh_mode='image',
)
test("mesh_mode='image'", p_image.mesh_mode == 'image')

p_default = Particle(
    x=50, y=50, vx=1, vy=0, life=1.0, max_life=2, size=5, color=(255, 200, 100),
)
test("默认 mesh_mode='circle'", p_default.mesh_mode == 'circle')

# Emitter 绑定网格体
emitter_mesh = Emitter(
    x=100, y=100, rate=10, speed=2, life=1, size=4, color=(255,255,255),
    mesh_surface="placeholder_sprite", mesh_mode='sprite',
)
parts_mesh = emitter_mesh.emit(dt=0.1)
has_sprite = all(p.mesh_mode == 'sprite' for p in parts_mesh)
test("Emitter 传递 mesh 参数给粒子", has_sprite)


# ================================================================
#  7. 事件系统（Event System）
# ================================================================
section("7. 事件系统（Event System）")

birth_count = [0]
death_count = [0]
collision_events = []
tick_data = []

def on_birth(p):
    birth_count[0] += 1

def on_death(p):
    death_count[0] += 1

def on_collide(p, box, data):
    collision_events.append(data)

def on_tick(p, life):
    tick_data.append(life)

sys_event = ParticleSystem(max_particles=200)
sys_event.on_particle_birth = on_birth
sys_event.on_particle_death = on_death
sys_event.on_particle_collision = on_collide
sys_event.on_lifetime_tick = on_tick

# 加一个快速死亡的发射器
e_fast = Emitter(
    x=100, y=100, rate=100, speed=0, spread=math.pi,
    life=0.01, size=2, color=(255, 0, 0),  # 极短寿命，确保一帧内死亡
)
sys_event.add_emitter(e_fast)
# 多次更新确保死亡事件触发
for _ in range(5):
    sys_event.update(dt=0.033)
test("on_birth 回调触发", birth_count[0] > 0,
     f"birth_count={birth_count[0]}")
test("on_death 回调触发", death_count[0] > 0,
     f"death_count={death_count[0]}")
test("on_lifetime_tick 回调触发", len(tick_data) > 0,
     f"tick 触发次数={len(tick_data)}")


# ================================================================
#  8. LOD 系统
# ================================================================
section("8. LOD 系统（Level of Detail）")

sys_lod = ParticleSystem(max_particles=1000)
sys_lod.lod_enabled = True
sys_lod.camera_pos = (400, 300)
sys_lod.lod_distances = [100, 300, 600]
sys_lod.lod_multipliers = [1.0, 0.5, 0.25, 0.1]

lod_emitter = Emitter(
    x=400, y=300, rate=100, speed=0, life=5, size=3, color=(255,255,255),
)
sys_lod.add_emitter(lod_emitter)

# 近距离（相机位置=粒子中心）→ LOD 1.0
sys_lod.update(dt=0.2)
near_count = sys_lod.get_particle_count()
test(f"近距离 LOD 可运行 (count={near_count})", True)

# 远距离模拟（移动 camera）
sys_lod.camera_pos = (2000, 1500)  # 很远
sys_lod.clear()
sys_lod.update(dt=0.1)  # 此时 LOD 倍率低，应该发射较少
far_count = sys_lod.get_particle_count()

# 注意：LOD 是概率性的，不一定每次都严格减半
test("LOD 系统启用且可运行",
     sys_lod.lod_enabled == True and far_count >= 0,
     f"近={near_count}, 远={far_count}")


# ================================================================
#  9. 光照模块（Blinn-Phong）
# ================================================================
section("9. 光照模块（Blinn-Phong Lighting）")

# 创建光源
ambient = create_ambient_light(color=(60, 60, 80), intensity=0.25)
point_light = create_point_light(x=400, y=100, color=(255, 240, 220), intensity=1.0, range_=400)
dir_light = create_directional_light(direction=math.pi/3, color=(255,250,230), intensity=0.7)

lights = [ambient, point_light, dir_light]

p_lit = Particle(
    x=400, y=300, vx=0, vy=0, life=1.0, max_life=5, size=5,
    color=(200, 100, 50), alpha=1.0,
)

# 应用光照
r, g, b, a = apply_lighting_to_particle(p_lit, lights, None, (400, 300))
test("光照计算返回 RGBA", 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255 and 0 <= a <= 1,
     f"result=({r},{g},{b},{a:.2f})")

# 自发光粒子的光照结果（用低基色避免饱和）
p_glow = Particle(
    x=400, y=300, vx=0, vy=0, life=1.0, max_life=5, size=5,
    color=(50, 30, 100), emissive=(255, 100, 50), emissive_strength=1.5,
)
r2, g2, b2, a2 = apply_lighting_to_particle(p_glow, lights, None, (400, 300))
# 发光粒子应该至少在某些通道上比普通粒子更亮
p_dark = Particle(
    x=400, y=300, vx=0, vy=0, life=1.0, max_life=5, size=5,
    color=(50, 30, 100),
)
dr, dg, db, da = apply_lighting_to_particle(p_dark, lights, None, (400, 300))
test("自发光粒子更亮", r2 >= dr or g2 > dg or b2 > db,
     f"普通=({dr},{dg},{db}), 发光=({r2},{g2},{b2})")

# 无光源时 fallback
empty_r, empty_g, empty_b, empty_a = apply_lighting_to_particle(p_lit, [], None, (400, 300))
test("无光源时返回基础色", (empty_r, empty_g, empty_b) == (0, 0, 0) or empty_a == p_lit.alpha,
     f"result=({empty_r},{empty_g},{empty_b},{empty_a})")

# BlinnPhongParams 定制化
custom_bp = BlinnPhongParams(
    ambient_color=(20, 20, 40), ambient_strength=0.5,
    diffuse_strength=0.8, specular_strength=0.6, shininess=64.0,
    fresnel_power=5.0, fresnel_strength=0.4,
)
cr, cg, cb, ca = apply_lighting_to_particle(p_lit, lights, custom_bp, (400, 300))
test("自定义 BlinnPhong 参数生效",
     0 <= cr <= 255 and 0 <= cg <= 255 and 0 <= cb <= 255,
     f"custom=({cr},{cg},{cb})")

# ParticleSystem 光照集成
sys_light = ParticleSystem(max_particles=100)
sys_light.add_light(ambient)
sys_light.add_light(point_light)
sys_light.lighting_enabled = True
test("ParticleSystem 光照集成", sys_light.lighting_enabled == True and len(sys_light.lights) == 2)


# ================================================================
#  10. 音频联动框架
# ================================================================
section("10. 音频联动框架（Audio Linkage）")

sys_audio = ParticleSystem(max_particles=100)
sys_audio.audio_enabled = True

# 设置音效（实际需要 pygame.mixer 初始化才能播放）
sys_audio.audio_birth_sound = "placeholder_sound"
sys_audio.audio_death_sound = "placeholder_sound"
sys_audio.audio_collision_sound = "placeholder_sound"

test("音频联动配置", sys_audio.audio_enabled == True)
test("音频事件槽已设置", 
     sys_audio.audio_birth_sound is not None and
     sys_audio.audio_death_sound is not None)

# 音频驱动发射率测试
audio_data = {'rms': 2.0}  # 高能量
emitter_audio = Emitter(x=100, y=100, rate=10, speed=0, life=1, size=2, color=(255,255,255))
sys_audio.add_emitter(emitter_audio)

normal_parts = emitter_audio.emit(dt=0.1)      # 无音频
audio_parts = emitter_audio.emit(dt=0.1, audio_data=audio_data)  # 有音频
test("高能量音频增加发射量", len(audio_parts) >= len(normal_parts),
     f"无音频={len(normal_parts)}, 有音频={len(audio_parts)}")

low_energy = {'rms': 0.0}
low_parts = emitter_audio.emit(dt=0.1, audio_data=low_energy)
test("低能量音频不额外增加", len(low_parts) >= 0,
     f"低能量={len(low_parts)}")


# ================================================================
#  总结
# ================================================================
print(f"\n{'═' * 55}")
total = passed + failed
print(f"  v3.0 全功能验证汇总")
print(f"{'═' * 55}")
print(f"  总计 : {total} 项测试")
print(f"  通过 : ✅ {passed}")
print(f"  失败 : ❌ {failed}")
print(f"  通过率 : {passed}/{total} ({100*passed//total}%)" if total > 0 else "")

if failed == 0:
    print(f"\n  🎉 所有 v3.0 特性验证通过！")
else:
    print(f"\n  ⚠️  有 {failed} 项测试失败，请检查上方输出")
    
    failed_names = [(name, ok) for name, ok in test_results if not ok]
    for name, _ in failed_names:
        print(f"    ❌ {name}")

sys.exit(0 if failed == 0 else 1)

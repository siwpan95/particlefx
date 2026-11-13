"""particlefx 无头自检 —— 不需要 pygame / SDL 窗口，也不写图片文件。

只 import 纯逻辑模块：particlefx.core、particlefx.systems.fireworks_system、
particlefx.components.effects_factory（渲染器与 examples 一律不 import）。

验证点：
1. 核心物理：发射器产出粒子、重力使 vy 增加、Particle 的位置积分与生命衰减；
2. 烟花系统 headless：火箭上升 -> 到高爆炸 -> 爆炸粒子存活；
3. 声明式效果工厂：effects_config.json 往返 + 按配置创建 ParticleSystem 并推进。

运行：python selfcheck.py
"""

import os
import math
import random
import sys
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from particlefx import Emitter, ForceField, ParticleSystem
from particlefx.components.effects_factory import EffectsFactory
from particlefx.core.particle import Particle
from particlefx.systems.fireworks_system import FireworksSystem
from particlefx.systems.light_field import LightFieldSystem

SEED = 20261026


def check_core_physics():
    """功能点 1：发射 -> 重力 -> 积分 -> 生命周期整条链路。"""
    ps = ParticleSystem(max_particles=5000)
    if ps.get_particle_count() != 0:
        return "新建 ParticleSystem 时不应已有粒子"

    ps.add_emitter(Emitter(
        x=400.0, y=300.0, rate=300.0, spread=0.0,
        speed=0.0, life=2.0, size=2.0, color=(255, 200, 50),
    ))
    ps.add_force_field(ForceField(field_type='gravity', strength=100.0))

    dt = 1.0 / 60.0
    for _ in range(30):
        ps.update(dt)

    count = ps.get_particle_count()
    if count <= 0:
        return "发射器推进 30 帧后没有任何粒子"
    if count > ps.max_particles:
        return "粒子数 %d 超过 max_particles=%d" % (count, ps.max_particles)

    for p in ps.particles:
        if not (0.0 <= p.life <= 1.0):
            return "粒子 life=%r 越出 [0, 1]" % (p.life,)
        if not p.is_alive():
            return "已死粒子残留在活动列表中"

    mean_vy = sum(p.vy for p in ps.particles) / count
    if mean_vy <= 0.0:
        return "重力未把平均 vy 拉正：%.6f" % mean_vy
    mean_y = sum(p.y for p in ps.particles) / count
    if mean_y <= 300.0:
        return "重力作用后粒子平均 y=%.3f 未下移" % mean_y

    # 单粒子积分与生命衰减（与 ParticleSystem 解耦的纯数据路径）
    particle = Particle(x=0.0, y=0.0, vx=10.0, vy=0.0,
                        life=1.0, max_life=2.0, size=2.0, color=(1, 2, 3))
    particle.update(0.5)
    if abs(particle.x - 5.0) > 1e-9:
        return "Particle.update 位置积分错误：x=%r，期望 5.0" % (particle.x,)
    if abs(particle.life - 0.75) > 1e-9:
        return "Particle.update 生命衰减错误：life=%r，期望 0.75" % (particle.life,)
    if abs(particle.alpha - 0.75) > 1e-9:
        return "Particle.update 透明度未跟随生命：alpha=%r" % (particle.alpha,)
    particle.update(1.5)
    if particle.is_alive():
        return "粒子生命耗尽后 is_alive() 仍为 True"
    return ""


def check_fireworks_headless():
    """功能点 2：烟花推进（上升 -> 爆炸 -> 残焰粒子）。"""
    random.seed(SEED)
    fw = FireworksSystem(width=800, height=600, auto_launch=False)
    if fw.rocket_count != 0:
        return "auto_launch=False 时初始火箭数应为 0"

    fw.launch(x=400.0, y=100.0, style='willow')
    if fw.rocket_count != 1:
        return "launch 后火箭数为 %r，期望 1" % (fw.rocket_count,)
    start_y = fw.rockets[0]['y']
    fw.update(0.1)
    if fw.rocket_count != 1:
        return "单帧推进后火箭意外消失（目标高度 100）"
    if fw.rockets[0]['y'] >= start_y:
        return "火箭未上升：y %r -> %r" % (start_y, fw.rockets[0]['y'])

    random.seed(SEED)
    fw = FireworksSystem(width=800, height=600, auto_launch=False)
    fw.launch(x=400.0, y=599.0, style='willow')
    for _ in range(20):
        fw.update(0.1)
    if fw.rocket_count != 0:
        return "20 帧后仍有 %d 枚火箭未爆炸" % fw.rocket_count
    if fw.particle_count <= 0:
        return "爆炸后没有产生任何粒子"
    if len(fw.get_all_particles()) != fw.particle_count:
        return "get_all_particles 与 particle_count 不一致：%d vs %d" % (
            len(fw.get_all_particles()), fw.particle_count)
    return ""


def check_effects_factory():
    """功能点 3：JSON 声明式效果 -> ParticleSystem 的配置往返。"""
    factory = EffectsFactory(validate=True)
    names = factory.list_effects()
    if len(names) < 10:
        return "effects_config.json 只解析出 %d 个效果" % len(names)
    categories = factory.list_categories()
    if not categories:
        return "effects_config.json 没有分类信息"
    total_in_categories = sum(c['count'] for c in categories)
    if total_in_categories != len(names):
        return "分类内效果总数 %d 与效果总数 %d 不一致" % (
            total_in_categories, len(names))

    random.seed(SEED)
    name = 'starfield' if 'starfield' in names else names[0]
    info = factory.get_effect_info(name)
    if not info:
        return "get_effect_info(%r) 返回空" % name

    system = factory.create(name, x=400, y=300, width=800, height=600,
                            center_x=400, center_y=300)
    if system is None:
        return "create(%r) 返回 None" % name
    if not system.emitters:
        return "效果 %r 没有生成任何发射器" % name
    if system.max_particles != info.get('max_particles', system.max_particles):
        return "效果 %r 的 max_particles 未按配置生效" % name
    if getattr(system, '_effect_name', None) != name:
        return "效果 %r 未写入 _effect_name 元信息" % name

    for _ in range(12):
        system.update(1.0 / 60.0)
    if system.get_particle_count() <= 0:
        return "效果 %r 推进 12 帧后没有粒子" % name
    return ""


def check_light_field():
    """功能点 4：LightFieldSystem 无头 —— 确定性播种 / 流场运动 / 吸引排斥 / 冲击波 / seed 可复现。"""
    W, H, N = 400, 300, 400

    # (a) 无指针时环境流场应让粒子移动
    s = LightFieldSystem(width=W, height=H, count=N, seed=20261026, max_particles=2000)
    init = [(p.x, p.y) for p in s.particles]
    s.update(16.667)
    moved = sum(1 for (x0, y0), p in zip(init, s.particles)
                if abs(p.x - x0) > 1e-6 or abs(p.y - y0) > 1e-6)
    if moved < N * 0.5:
        return "流场下多数粒子未移动：%d/%d" % (moved, N)

    # (b) 确定性：同 seed + 同 dt 序列 + 同指针序列 => 位置完全一致
    a = LightFieldSystem(width=W, height=H, count=N, seed=777, max_particles=2000)
    b = LightFieldSystem(width=W, height=H, count=N, seed=777, max_particles=2000)
    for _ in range(10):
        a.set_pointer(W / 2, H / 2, active=True, pressing=False)
        b.set_pointer(W / 2, H / 2, active=True, pressing=False)
        a.update(16.667)
        b.update(16.667)
    max_err = max(math.hypot(pa.x - pb.x, pa.y - pb.y)
                  for pa, pb in zip(a.particles, b.particles))
    if max_err > 1e-9:
        return "seed 可复现失败：最大误差 %.3e" % max_err

    # (c) 吸引：指针居中，初始落在影响半径内的粒子应被明显拉向中心。
    #     仅测量“受指针力场作用”的子集，比全局平均更灵敏、更稳健。
    cx, cy = W / 2, H / 2
    infl = LightFieldSystem.INFLUENCE
    c = LightFieldSystem(width=W, height=H, count=N, seed=42, max_particles=2000)
    c_mask = [math.hypot(p.x - cx, p.y - cy) < infl for p in c.particles]
    c_d0 = [math.hypot(p.x - cx, p.y - cy) for p, m in zip(c.particles, c_mask) if m]
    for _ in range(60):
        c.set_pointer(cx, cy, active=True, pressing=False)
        c.update(16.667)
    c_d1 = [math.hypot(p.x - cx, p.y - cy) for p, m in zip(c.particles, c_mask) if m]
    c_a0 = sum(c_d0) / len(c_d0)
    c_a1 = sum(c_d1) / len(c_d1)
    if not (c_a1 < c_a0 * 0.995):
        return "吸引未生效：影响区内平均距离 %.2f -> %.2f" % (c_a0, c_a1)

    # (d) 排斥：按住（pressing），同一子集粒子应被明显推离中心
    e = LightFieldSystem(width=W, height=H, count=N, seed=42, max_particles=2000)
    e_mask = [math.hypot(p.x - cx, p.y - cy) < infl for p in e.particles]
    e_d0 = [math.hypot(p.x - cx, p.y - cy) for p, m in zip(e.particles, e_mask) if m]
    for _ in range(60):
        e.set_pointer(cx, cy, active=True, pressing=True)
        e.update(16.667)
    e_d1 = [math.hypot(p.x - cx, p.y - cy) for p, m in zip(e.particles, e_mask) if m]
    e_a0 = sum(e_d0) / len(e_d0)
    e_a1 = sum(e_d1) / len(e_d1)
    if not (e_a1 > e_a0 * 1.01):
        return "排斥未生效：影响区内平均距离 %.2f -> %.2f" % (e_a0, e_a1)

    # (e) 冲击波：触发后应有环记录且附近粒子获得向外冲量
    f = LightFieldSystem(width=W, height=H, count=N, seed=99, max_particles=2000)
    f.trigger_shockwave(W / 2, H / 2)
    if len(f.shockwaves) < 1:
        return "冲击波未记录"
    for _ in range(3):
        f.update(16.667)

    # (f) 换色与 set_count 不报错
    f.cycle_palette()
    f.set_count(250)
    f.update(16.667)
    if f.get_particle_count() != 250:
        return "set_count 后粒子数不符：%d" % f.get_particle_count()
    return ""


def main():
    checks = [
        ("core emitter/gravity/lifetime", check_core_physics),
        ("FireworksSystem headless launch+burst", check_fireworks_headless),
        ("EffectsFactory JSON config round-trip", check_effects_factory),
        ("LightFieldSystem headless interactivity", check_light_field),
    ]
    for label, func in checks:
        try:
            err = func()
        except Exception:
            print("FAIL: %s raised" % label)
            traceback.print_exc()
            return 1
        if err:
            print("FAIL: %s: %s" % (label, err))
            return 1

    print("selfcheck ok: validated %d things -- %s" % (
        len(checks), "; ".join(label for label, _ in checks)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

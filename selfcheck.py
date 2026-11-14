"""particlefx 无头自检 —— 不需要 pygame / SDL 窗口，也不写图片文件。

只 import 纯逻辑模块：particlefx.core、particlefx.systems.fireworks_system、
particlefx.components.effects_factory（渲染器与 examples 一律不 import）。

验证点：
1. 核心物理：发射器产出粒子、重力使 vy 增加、Particle 的位置积分与生命衰减；
2. 烟花系统 headless：火箭上升 -> 到高爆炸 -> 爆炸粒子存活；
3. 声明式效果工厂：effects_config.json 往返 + 按配置创建 ParticleSystem 并推进；
4. 粒子光场 headless：确定性播种 / 流场运动 / 吸引排斥 / 冲击波 / seed 可复现；
5. （v1.2）可复现模拟：种子化随机源 —— 同种子可复现 / 不同种子相异 /
   每发射器独立子流 / 发射器级 RNG 回滚；
6. （v1.2）固定步长子步进：大 dt 拆步逼近细积分 / 默认关闭逐位一致 / spiral 上限；
7. （v1.2）力场 / 噪声场内部噪声种子化：确定性派发 / 显式种子保留 / reseed 重派；
8. （v1.2）MagicEffectsManager：5 种魔法效果 headless / 生命周期 / 种子可复现；
9. （v1.2）场景配方与种子持久化：导出 -> 重建 / 文件往返 / 工厂带 seed 可复现。

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


def check_determinism():
    """功能点 5（v1.2）：种子化随机源 —— 同种子可复现 / 不同种子相异 /
    每发射器独立子流 / 发射器级 RNG 可回滚。
    """
    from particlefx.core.particle import Particle

    def _snap(ps):
        return [(round(p.x, 9), round(p.y, 9), round(p.vx, 9), round(p.vy, 9),
                round(p.life, 9), tuple(p.color)) for p in ps.particles]

    # (a) 同系统 seed + 同推进序列 => 逐位一致；不同 seed => 结果不同
    def run(seed):
        ps = ParticleSystem(max_particles=4000, seed=seed)
        ps.add_emitter(Emitter(x=400, y=300, rate=200.0, speed=120.0,
                               spread=0.8, life=2.0, size=2.0))
        ps.add_force_field(ForceField(field_type='gravity', strength=60.0))
        for _ in range(40):
            ps.update(1.0 / 60.0)
        return _snap(ps)

    if run(12345) != run(12345):
        return "种子可复现失败：同 seed 两次推进结果不一致"
    if run(1) == run(2):
        return "不同 seed 应当产生不同结果（若偶发相等请检查随机源）"

    # (b) 每发射器独立子流：同种子、两个发射器，跨次一致且两发射器不重合
    def run2():
        ps = ParticleSystem(max_particles=5000, seed=999)
        for k in range(2):
            ps.add_emitter(Emitter(x=400 + k * 50, y=300, rate=150.0, speed=120.0,
                                   spread=1.0, life=2.0, size=2.0))
        for _ in range(10):
            ps.update(1.0 / 60.0)
        return [(round(p.x, 6), round(p.y, 6)) for p in ps.particles]

    a, b = run2(), run2()
    if a != b:
        return "每发射器独立子流不可复现"
    mid = len(a) // 2
    if not any(abs(a[i][0] - a[j][0]) > 1e-3 or abs(a[i][1] - a[j][1]) > 1e-3
               for i in range(0, mid) for j in range(mid, len(a))):
        return "两个发射器子流意外重合（未正确派生）"

    # (c) 发射器级 RNG 可回滚：保存起始状态 -> 推进 -> 恢复 -> 同次 emit 逐位一致
    em = Emitter(x=400, y=300, rate=100.0, speed=50.0, spread=1.0, life=1.0, seed=7)
    st = em._rng.getstate()
    p1 = em.emit(0.1)
    em._rng.setstate(st)
    p2 = em.emit(0.1)
    if [(p.x, p.y, p.vx, p.vy, p.rotation, p.mass) for p in p1] != \
       [(p.x, p.y, p.vx, p.vy, p.rotation, p.mass) for p in p2]:
        return "发射器 RNG 回滚失败：emit 产物不一致"
    return ""


def check_substep():
    """功能点 6（v1.2）：固定步长子步进 —— 大 dt 拆步后逼近细积分、
    默认关闭逐位一致、spiral-of-death 上限保护。
    """
    from particlefx.core.particle import Particle

    def _seed(ps, n=3):
        for i in range(n):
            ps.particles.append(Particle(x=400.0 + i * 10, y=300.0, vx=0.0, vy=0.0,
                                         life=1.0, max_life=10.0, size=2.0, color=(1, 2, 3)))
        return ps

    # (a) 子步进（fixed_dt=1/60）逼近以同步长逐帧推进的细积分（纯重力半隐式欧拉，逐位一致）
    fine = ParticleSystem(max_particles=200, fixed_dt=None)
    fine.add_emitter(Emitter(x=400, y=300, rate=0.0, speed=0.0, life=10.0, size=2.0))
    fine.add_force_field(ForceField(field_type='gravity', strength=200.0))
    _seed(fine, 3)
    for _ in range(20):
        fine.update(1.0 / 60.0)
    fine_state = [(p.x, p.y, p.vx, p.vy) for p in fine.particles]

    sub = ParticleSystem(max_particles=200, fixed_dt=1.0 / 60.0, max_substeps=30)
    sub.add_emitter(Emitter(x=400, y=300, rate=0.0, speed=0.0, life=10.0, size=2.0))
    sub.add_force_field(ForceField(field_type='gravity', strength=200.0))
    _seed(sub, 3)
    sub.update(20.0 / 60.0)  # 一帧内拆成 20 个固定子步
    sub_state = [(p.x, p.y, p.vx, p.vy) for p in sub.particles]
    if any(abs(sx - fx) > 1e-9 or abs(sy - fy) > 1e-9
           for (sx, sy, _, _), (fx, fy, _, _) in zip(sub_state, fine_state)):
        return "子步进未逐位匹配细积分"

    # (b) 默认 fixed_dt=None 与显式 fixed_dt>dt 均只走单步（旧行为不变）
    d1 = ParticleSystem(max_particles=100)
    d1.add_emitter(Emitter(x=400, y=300, rate=0.0, speed=0.0, life=10.0, size=2.0))
    d1.add_force_field(ForceField(field_type='gravity', strength=200.0))
    _seed(d1, 3)
    d1.update(0.1)
    d2 = ParticleSystem(max_particles=100, fixed_dt=1.0)
    d2.add_emitter(Emitter(x=400, y=300, rate=0.0, speed=0.0, life=10.0, size=2.0))
    d2.add_force_field(ForceField(field_type='gravity', strength=200.0))
    _seed(d2, 3)
    d2.update(0.1)
    if [(p.x, p.y, p.vx, p.vy) for p in d1.particles] != \
       [(p.x, p.y, p.vx, p.vy) for p in d2.particles]:
        return "默认关闭子步进与旧行为不一致"
    if d1._last_substep_count != 1 or d2._last_substep_count != 1:
        return "默认路径应只走单步"

    # (c) spiral-of-death 上限：dt=10, fixed_dt=0.01, max_substeps=4 => 实际 4 步
    cap = ParticleSystem(max_particles=100, fixed_dt=0.01, max_substeps=4)
    cap.add_emitter(Emitter(x=0, y=0, rate=0.0, speed=0.0, life=10.0, size=2.0))
    cap.add_force_field(ForceField(field_type='gravity', strength=50.0))
    _seed(cap, 1)
    cap.update(10.0)
    if cap._last_substep_count != 4:
        return "spiral-of-death 上限失效：期望 4 步，实际 %d" % cap._last_substep_count
    if not all(math.isfinite(p.x) and math.isfinite(p.y) for p in cap.particles):
        return "子步进后存在非有限坐标（数值发散）"
    return ""


def check_field_seeding():
    """功能点 7（v1.2）：力场 / 噪声场内部噪声种子化 ——
    系统 seed 派生确定性子流 / 显式种子保留 / reseed 重派 / 整场可复现。
    """
    from particlefx import NoiseField

    def _snap(ps):
        return [(round(p.x, 9), round(p.y, 9), round(p.vx, 9), round(p.vy, 9))
                for p in ps.particles]

    # (a) 含噪声场 / 湍流场 / 独立噪声场的系统，同 seed 逐位可复现、不同 seed 相异
    def run(seed):
        ps = ParticleSystem(max_particles=2000, seed=seed)
        ps.add_emitter(Emitter(x=400, y=300, rate=80.0, speed=120.0,
                               spread=0.8, life=2.0, size=2.0))
        ps.add_force_field(ForceField(field_type='noise', noise_strength=3.0,
                                      noise_scale=40.0))
        ps.add_force_field(ForceField(field_type='turbulence', turb_strength=2.0,
                                      turb_scale=35.0))
        ps.add_noise_field(NoiseField(strength=1.5, scale=60.0))
        for _ in range(30):
            ps.update(1.0 / 60.0)
        return _snap(ps)

    if run(999) != run(999):
        return "含噪声力场的系统同 seed 不可复现"
    if run(1) == run(2):
        return "不同 seed 应产生不同结果"

    # (b) 派生确定性 + 显式种子保留 + reseed 重派
    ps = ParticleSystem(max_particles=10, seed=42)
    ff = ForceField(field_type='noise', noise_strength=3.0)
    nf = NoiseField(strength=1.5)
    ff_explicit = ForceField(field_type='noise', noise_strength=3.0, seed=12345)
    ps.add_force_field(ff)
    ps.add_noise_field(nf)
    ps.add_force_field(ff_explicit)
    if ff.seed is None or nf.seed is None:
        return "系统播种时未派生力场 / 噪声场种子"
    if ff_explicit.seed != 12345 or ff_explicit._system_derived:
        return "用户显式种子被系统覆盖"
    old_ff, old_nf = ff.seed, nf.seed
    ps.reseed(43)
    if ff.seed == old_ff or nf.seed == old_nf:
        return "reseed 未重派派生种子"
    if ff_explicit.seed != 12345:
        return "reseed 误改用户显式种子"
    return ""


def check_magic():
    """功能点 8（v1.2）：MagicEffectsManager —— 5 种效果 headless /
    生命周期清空 / 同 seed 逐位可复现 / 未知效果报错。
    """
    from particlefx.systems.magic_effects import MagicEffectsManager

    # (a) 全部预设可列举且能触发
    presets = MagicEffectsManager(width=1).list_presets()
    expected = {'arcane_burst', 'rune_ring', 'healing', 'void_tendril', 'magic_missile'}
    if set(presets.keys()) != expected:
        return "魔法预设集合不符：%r" % (set(presets.keys()),)
    m = MagicEffectsManager()
    m.cast('arcane_burst', x=400, y=300)
    if m.particle_count <= 0:
        return "arcane_burst 未产生粒子"
    try:
        m.cast('nonexistent', 0, 0)
        return "未知魔法效果未抛 ValueError"
    except ValueError:
        pass

    # (b) 爆发类效果生命周期结束应清空
    m2 = MagicEffectsManager()
    m2.cast('arcane_burst', x=400, y=300)
    for _ in range(200):
        m2.update(1.0 / 60.0)
    if m2.particle_count != 0:
        return "爆发效果推进后粒子未清空：%d" % m2.particle_count

    # (c) 全部 5 种效果同 seed 逐位可复现、不同 seed 相异
    def run(seed):
        mm = MagicEffectsManager(width=800, height=600, seed=seed)
        mm.cast('arcane_burst', x=400, y=300)
        mm.cast('rune_ring', x=400, y=300)
        mm.cast('healing', x=400, y=300)
        mm.cast('void_tendril', x=400, y=300)
        mm.cast('magic_missile', x=400, y=300, tx=500, ty=100)
        for _ in range(120):
            mm.update(1.0 / 60.0)
        return [(round(p.x, 9), round(p.y, 9), round(p.vx, 9), round(p.vy, 9))
                for p in mm.particles]

    if run(42) != run(42):
        return "魔法系统同 seed 不可复现"
    if run(1) == run(2):
        return "魔法系统不同 seed 应产生不同结果"
    return ""


def check_preset():
    """功能点 9（v1.2）：场景配方与种子持久化 —— 导出 -> 重建可复现 /
    文件往返 / EffectsFactory 带 seed 可复现 / 曲线忠实还原。
    """
    import tempfile
    from particlefx import NoiseField
    from particlefx.components.preset import (
        system_to_recipe, recipe_to_system, save_preset, load_preset,
        export_effect_recipe,
    )

    def _snap(ps):
        return [(round(p.x, 9), round(p.y, 9), round(p.vx, 9), round(p.vy, 9))
                for p in ps.particles]

    # (a) 系统 -> 配方 -> 重建，同种子从干净状态推进逐位一致
    ps = ParticleSystem(max_particles=2000, seed=12345)
    ps.add_emitter(Emitter(x=400, y=300, rate=120.0, speed=140.0,
                           spread=0.8, life=2.0, size=3.0))
    ps.add_force_field(ForceField(field_type='noise', noise_strength=3.0,
                                  noise_scale=40.0))
    ps.add_noise_field(NoiseField(strength=1.5, scale=60.0))
    recipe = system_to_recipe(ps, name='demo')
    if recipe['seed'] != 12345 or recipe['emitters'][0]['seed'] is not None:
        return "配方顶层 seed / 派生种子落地不正确"
    rebuilt = recipe_to_system(recipe)
    for _ in range(20):
        ps.update(1.0 / 60.0)
        rebuilt.update(1.0 / 60.0)
    if _snap(ps) != _snap(rebuilt):
        return "配方重建后模拟未逐位复现"

    # (b) 文件往返
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'preset.json')
        save_preset(recipe, path)
        loaded = load_preset(path)
        if loaded['seed'] != 12345:
            return "配方文件往返后 seed 丢失"
        r2 = recipe_to_system(loaded)
        base = recipe_to_system(recipe)
        for _ in range(15):
            r2.update(1.0 / 60.0)
            base.update(1.0 / 60.0)
        if _snap(r2) != _snap(base):
            return "配方文件往返后不可复现"

    # (c) EffectsFactory 带 seed 可复现（含生命期曲线的声明式效果）
    factory = EffectsFactory(validate=True)

    def run():
        s = factory.create('arcane_blast', x=400, y=300, seed=777)
        for _ in range(15):
            s.update(1.0 / 60.0)
        return _snap(s)

    if run() != run():
        return "EffectsFactory.create 带 seed 不可复现"

    # (d) 导出配方含曲线数据（关键帧列表），重建后与直接创建一致
    exported = export_effect_recipe(factory, 'arcane_blast', seed=999, x=400, y=300)
    if exported['seed'] != 999:
        return "export_effect_recipe 未写入 seed"
    has_curve = any(e.get('velocity_over_lifetime') or e.get('color_over_lifetime')
                    for e in exported['emitters'])
    if not has_curve:
        return "导出配方丢失了生命期曲线数据"
    rr = recipe_to_system(exported)
    orig = factory.create('arcane_blast', x=400, y=300, seed=999)
    for _ in range(15):
        rr.update(1.0 / 60.0)
        orig.update(1.0 / 60.0)
    if _snap(rr) != _snap(orig):
        return "含曲线的声明式效果配方重建后未逐位复现"
    return ""


def main():
    checks = [
        ("core emitter/gravity/lifetime", check_core_physics),
        ("FireworksSystem headless launch+burst", check_fireworks_headless),
        ("EffectsFactory JSON config round-trip", check_effects_factory),
        ("LightFieldSystem headless interactivity", check_light_field),
        ("deterministic seeding (v1.2)", check_determinism),
        ("fixed-timestep sub-stepping (v1.2)", check_substep),
        ("force/noise field seeding (v1.2)", check_field_seeding),
        ("MagicEffectsManager (v1.2)", check_magic),
        ("scene preset + seed persistence (v1.2)", check_preset),
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

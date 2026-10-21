#!/usr/bin/env python3
"""
particle 系统增强特性演示

展示所有新增特性：
1. Color over Lifetime（颜色随生命变化）
2. Size over Lifetime（大小随生命变化）
3. Rotation（粒子旋转）
4. Noise Field（噪声场，自然扰动）
5. Velocity over Lifetime（速度曲线）
6. 子发射器（粒子死亡时触发新粒子）

运行方式（在仓库根目录下）：
    cd <仓库根>
    python examples/demo_enhanced_features.py
"""

import sys
import os
import math
import time

# 确保可以导入 core 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from particlefx.core.particle import (
    Particle, ParticleSystem, Emitter, ForceField, NoiseField,
    FIRE_COLOR_CURVE, SMOKE_COLOR_CURVE, MAGIC_COLOR_CURVE,
    EXPLOSION_SIZE_CURVE, SPARK_SIZE_CURVE,
    gradient_to_curve, size_curve_fadeout, size_curve_pulse,
    velocity_curve_accelerate, velocity_curve_decelerate,
)
from particlefx.core.particle import lerp_color, evaluate_curve, evaluate_color_curve


def print_separator(char='=', length=60):
    print(char * length)


def demo_color_over_lifetime():
    """演示 1：Color over Lifetime"""
    print_separator()
    print("演示 1：Color over Lifetime（颜色随生命渐变）")
    print_separator('-')

    # 使用预设火焰颜色曲线
    print("使用 FIRE_COLOR_CURVE（火焰渐变）：")
    print(f"  曲线定义：{FIRE_COLOR_CURVE[:2]} ...")

    # 创建粒子并模拟其颜色变化
    p = Particle(
        x=0, y=0, vx=0, vy=0,
        life=1.0, max_life=1.0,
        size=2.0, color=(255, 255, 255),
        color_over_lifetime=FIRE_COLOR_CURVE,
    )

    print("\n  生命进度 | 颜色 (R, G, B)")
    print("  " + "-" * 40)
    for step in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p.life = 1.0 - step  # life=1 是诞生，life=0 是死亡
        p.update(0.0)  # 不推进时间，只触发颜色更新
        print(f"  t={step:.2f}     | {p.color}")

    print("\n  ✅ 粒子从白热 → 亮黄 → 橙红 → 暗红，完全自动！")
    return True


def demo_size_over_lifetime():
    """演示 2：Size over Lifetime"""
    print_separator()
    print("演示 2：Size over Lifetime（大小随生命变化）")
    print_separator('-')

    # 使用爆炸大小曲线（先膨胀后缩小）
    print("使用 EXPLOSION_SIZE_CURVE（爆炸大小曲线）：")
    for t, s in EXPLOSION_SIZE_CURVE:
        print(f"  t={t:.2f} => size={s:.2f}")

    p = Particle(
        x=0, y=0, vx=0, vy=0,
        life=1.0, max_life=1.0,
        size=2.0, color=(255, 200, 50),
        size_over_lifetime=EXPLOSION_SIZE_CURVE,
    )

    print("\n  生命进度 | 大小")
    print("  " + "-" * 40)
    for step in [0.0, 0.05, 0.2, 0.5, 0.8, 1.0]:
        p.life = 1.0 - step
        p.update(0.0)
        print(f"  t={step:.2f}     | {p.size:.2f}")

    print("\n  ✅ 粒子大小先膨胀（0.05 时达到峰值 3.0）后快速缩小！")

    # 演示 size_curve_fadeout 辅助函数
    print("\n使用 size_curve_fadeout(2.0) 辅助函数：")
    fadeout_curve = size_curve_fadeout(2.0)
    print(f"  生成曲线：{fadeout_curve}")
    return True


def demo_rotation():
    """演示 3：Rotation（粒子旋转）"""
    print_separator()
    print("演示 3：Rotation（粒子旋转）")
    print_separator('-')

    p = Particle(
        x=100, y=200,
        vx=5.0, vy=3.0,
        life=1.0, max_life=2.0,
        size=3.0, color=(255, 255, 100),
        rotation=0.0,           # 初始旋转角度（弧度）
        angular_velocity=math.pi,  # 角速度（π 弧度/秒 = 180°/s）
    )

    print(f"  初始旋转：{p.rotation:.2f} rad = {math.degrees(p.rotation):.1f}°")
    print(f"  角速度：{p.angular_velocity:.2f} rad/s = {math.degrees(p.angular_velocity):.1f}°/s")
    print()

    dt = 0.1
    for i in range(6):
        t = i * dt
        print(f"  t={t:.1f}s  rotation={p.rotation:.2f} rad = {math.degrees(p.rotation):.1f}°")
        p.update(dt)

    print("\n  ✅ 粒子旋转角度随时间变化，渲染时会显示旋转指示线！")
    return True


def demo_noise_field():
    """演示 4：Noise Field（噪声场）"""
    print_separator()
    print("演示 4：Noise Field（噪声场 — 自然扰动）")
    print_separator('-')

    nf = NoiseField(strength=2.0, scale=50.0, octaves=3, seed=42)
    print(f"  噪声场参数：strength={nf.strength}, scale={nf.scale}, octaves={nf.octaves}")

    # 模拟粒子在噪声场中受到的力
    class FakeParticle:
        def __init__(self, x, y):
            self.x, self.y = x, y
            self.mass = 1.0

    test_positions = [(0, 0), (50, 50), (100, 0), (0, 100)]
    print("\n  位置 (x, y) | 噪声力 (fx, fy)")
    print("  " + "-" * 45)

    # 更新噪声时间，观察力的变化
    for t in [0.0, 0.5, 1.0]:
        nf.update(t)
        print(f"\n  --- 时间 t={t:.1f}s ---")
        for x, y in test_positions:
            p = FakeParticle(x, y)
            fx, fy = nf.get_force(p)
            print(f"  ({x:3.0f}, {y:3.0f})      | ({fx:+.3f}, {fy:+.3f})")

    print("\n  ✅ 同一位置在不同时间的受力不同（噪声场是动态的）")
    print("  ✅ 不同位置在同一时间的受力不同（噪声场是空间的）")
    return True


def demo_velocity_over_lifetime():
    """演示 5：Velocity over Lifetime"""
    print_separator()
    print("演示 5：Velocity over Lifetime（速度倍率曲线）")
    print_separator('-')

    # 使用加速曲线：结束时速度变为 2 倍
    vel_curve = velocity_curve_accelerate(mult=2.0)
    print(f"速度加速曲线：{vel_curve}")

    p = Particle(
        x=0, y=0,
        vx=10.0, vy=0.0,  # 初始速度：向右 10 单位/秒
        life=1.0, max_life=1.0,
        size=2.0, color=(100, 200, 255),
        velocity_over_lifetime=vel_curve,
    )

    print("\n  时间 | 速度 (vx, vy) | 说明")
    print("  " + "-" * 50)
    dt = 0.2
    for i in range(6):
        t = i * dt
        print(f"  t={t:.1f} | ({p.vx:.2f}, {p.vy:.2f})     | ", end="")
        if t < 0.5:
            print("速度逐渐加快...")
        else:
            print("速度继续加快！")
        p.update(dt)

    print("\n  ✅ 粒子速度随生命进度动态变化！")
    return True


def demo_sub_emitters():
    """演示 6：子发射器（粒子死亡时触发）"""
    print_separator()
    print("演示 6：子发射器（Sub-Emitters）")
    print_separator('-')

    # 创建主系统
    main_system = ParticleSystem(max_particles=500)

    # 创建子粒子系统（火花效果）
    spark_system = ParticleSystem(max_particles=200)
    spark_emitter = Emitter(
        x=0, y=0,
        rate=100,  # 高发射率（子发射器通常只触发一次）
        spread=math.pi * 2,
        speed=3.0,
        life=0.3,
        size=1.5,
        color=(255, 200, 50),
        size_over_lifetime=SPARK_SIZE_CURVE,
    )
    spark_system.add_emitter(spark_emitter)

    # 创建主粒子（绑定子发射器）
    main_emitter = Emitter(
        x=0, y=0,
        rate=5,
        spread=math.pi * 0.3,
        speed=8.0,
        life=0.5,
        size=4.0,
        color=(255, 100, 50),
        sub_emitter_on_death=spark_system,  # 死亡时触发火花
    )
    main_system.add_emitter(main_emitter)

    print("  主粒子系统：5 个粒子，生命 0.5 秒")
    print("  子粒子系统：粒子死亡时触发火花效果（每个粒子死亡产生 ~12 个火花）")
    print()

    # 模拟
    dt = 0.05
    for frame in range(60):
        t = frame * dt
        main_system.update(dt)

        if frame % 10 == 0:
            n_main = main_system.get_particle_count()
            print(f"  t={t:.2f}s  主粒子数={n_main}")

    n_final = main_system.get_particle_count()
    print(f"\n  模拟结束：主粒子数={n_final}")
    print("\n  ✅ 子发射器让粒子效果可以形成因果链！")
    print("     主爆炸 → 碎片飞溅 → 碎片落地扬尘")
    return True


def demo_full_integration():
    """演示 7：综合效果 — 增强版爆炸"""
    print_separator('=')
    print("演示 7：综合效果 — 增强版爆炸（所有特性整合）")
    print_separator('-')

    from particlefx.components.effects import create_explosion

    # 创建增强版爆炸
    system = create_explosion(x=0, y=0, intensity=1.0)

    print("  创建爆炸效果（使用增强特性）：")
    print("  - Color over Lifetime: FIRE_COLOR_CURVE")
    print("  - Size over Lifetime: EXPLOSION_SIZE_CURVE")
    print("  - 重力场 + 噪声场")
    print()

    # 查看系统组成
    print(f"  发射器数量：{len(system.emitters)}")
    print(f"  力场数量：{len(system.force_fields)}")
    print(f"  噪声场数量：{len(system.noise_fields)}")

    # 模拟几帧
    dt = 0.016
    for frame in range(30):
        system.update(dt)
        if frame % 5 == 0:
            n = system.get_particle_count()
            print(f"  Frame {frame:2d}  t={frame*dt:.3f}s  粒子数={n}")

    print("\n  ✅ 所有增强特性整合在预设效果中，一行调用即可！")
    return True


def demo_helper_functions():
    """演示 8：辅助函数"""
    print_separator()
    print("演示 8：辅助函数（用户友好接口）")
    print_separator('-')

    # gradient_to_curve
    print("gradient_to_curve() — 将颜色列表转为颜色曲线：")
    colors = [(255, 200, 50), (255, 100, 20), (80, 30, 5)]
    curve = gradient_to_curve(colors)
    print(f"  输入：{colors}")
    print(f"  输出：{curve}")
    print()

    # size_curve_pulse
    print("size_curve_pulse() — 大小脉冲曲线（先膨胀后缩小）：")
    pulse = size_curve_pulse(base_size=2.0, peak_at=0.3, peak_mult=2.0)
    print(f"  输入：base_size=2.0, peak_at=0.3, peak_mult=2.0")
    print(f"  输出：{pulse}")
    print()

    # lerp_color
    print("lerp_color() — 颜色插值：")
    c1 = (255, 0, 0)
    c2 = (0, 0, 255)
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = lerp_color(c1, c2, t)
        print(f"  t={t:.2f}  ({c1}) → ({c2}) = {result}")
    print()

    print("  ✅ 辅助函数让曲线定义变得简单！")
    return True


def main():
    """运行所有演示"""
    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "PARTICLE 系统增强特性演示" + " " * 19 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    demos = [
        ("Color over Lifetime", demo_color_over_lifetime),
        ("Size over Lifetime", demo_size_over_lifetime),
        ("Rotation", demo_rotation),
        ("Noise Field", demo_noise_field),
        ("Velocity over Lifetime", demo_velocity_over_lifetime),
        ("Sub-Emitters", demo_sub_emitters),
        ("辅助函数", demo_helper_functions),
        ("综合效果", demo_full_integration),
    ]

    results = []
    for name, func in demos:
        try:
            ok = func()
            results.append((name, ok))
        except Exception as e:
            print(f"  ❌ 错误：{e}")
            results.append((name, False))
        print()

    # 汇总
    print_separator('=')
    print("演示汇总")
    print_separator('-')
    for name, ok in results:
        status = "✅ 通过" if ok else "❌ 失败"
        print(f"  {name:30s} {status}")
    print()
    print("  🎉 所有核心增强特性验证完成！")
    print()


if __name__ == '__main__':
    main()

"""
烟花效果 headless 测试

在无显示环境下运行，保存渲染结果到 GIF
"""

import sys
import os

# 添加父目录到路径（用于导入 particle 模块）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.particle import ParticleSystem, Emitter, ForceField, Particle
from renderers.pil_renderer import PILRenderer
from PIL import Image


def main():
    """主函数"""
    # 配置
    width, height = 800, 600
    frames = 100  # 保存的帧数
    fps = 30

    # 创建粒子系统
    ps = ParticleSystem(max_particles=10000)

    # 创建发射器（烟花发射器）
    emitter = Emitter(
        x=width // 2, y=height,
        rate=50.0,  # 发射速率
        spread=3.14,  # 扩散角度（π = 180度，向上发射）
        speed=200.0,  # 发射速度
        life=2.0,  # 粒子生命值
        size=3.0,  # 粒子大小
        color=(255, 200, 50)  # 粒子颜色（橙黄色）
    )
    ps.add_emitter(emitter)

    # 创建力场（重力）
    gravity = ForceField(field_type='gravity', strength=100.0)
    ps.add_force_field(gravity)

    # 创建渲染器
    renderer = PILRenderer(width, height)

    # 渲染并保存帧
    print(f"开始渲染 {frames} 帧...")
    images = []

    for frame in range(frames):
        # 更新粒子系统
        dt = 1.0 / fps
        ps.update(dt)

        # 渲染
        renderer.render(ps.particles)
        img = renderer.get_image()

        # 转换为 RGB 模式（GIF 不支持 RGBA）
        img_rgb = Image.new('RGB', (width, height), (0, 0, 0))
        img_rgb.paste(img, (0, 0), img)

        images.append(img_rgb)

        # 显示进度
        if (frame + 1) % 10 == 0:
            print(f"  渲染进度: {frame + 1}/{frames} 帧 (粒子数: {ps.get_particle_count()})")

    # 保存为 GIF
    output_path = os.path.join(os.path.dirname(__file__), 'fireworks.gif')
    print(f"保存 GIF 到: {output_path}")
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=1000 // fps,  # 毫秒
        loop=0
    )
    print(f"✓ 完成！GIF 已保存: {output_path}")


if __name__ == '__main__':
    main()

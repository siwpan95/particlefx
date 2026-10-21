"""
烟花效果示例

使用粒子系统创建烟花效果
"""

import pygame
import sys
import os

# 添加父目录到路径（用于导入 particle 模块）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from particlefx.core.particle import ParticleSystem, Emitter, ForceField, Particle
from particlefx.renderers.pygame_renderer import PygameRenderer


def main():
    """主函数"""
    # 初始化 Pygame
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Particle System - Fireworks Example")

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
    renderer = PygameRenderer(screen)

    # 主循环
    clock = pygame.time.Clock()
    running = True
    dt = 0.0

    while running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # 更新粒子系统
        ps.update(dt)

        # 渲染
        screen.fill((0, 0, 0))  # 黑色背景
        ps.render(renderer)
        pygame.display.flip()

        # 控制帧率
        dt = clock.tick(60) / 1000.0  # 转换为秒

        # 显示粒子数量
        pygame.display.set_caption(f"Particle System - Fireworks Example (Particles: {ps.get_particle_count()})")

    # 退出
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()

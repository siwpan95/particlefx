"""
pygbag Web 导出入口（v5.0 增强 ④）— 浏览器中运行粒子系统

pygbag 将 pygame-ce 应用打包为 WebAssembly，在浏览器直接运行：
    pip install pygbag
    pygbag --app_name web_main examples

本地桌面环境也可直接运行（回退为普通 pygame 循环）：
    python examples/web_main.py

pygbag 约定：
- 主循环必须是 async 函数，每帧 await asyncio.sleep(0) 让出控制权
- 无需任何平台判断即可跨桌面/浏览器运行（pygame-ce 内部处理 emscripten）
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame

from systems.fireworks_system import FireworksSystem
from renderers.pygame_renderer import PygameRenderer


async def main():
    """迷你烟花秀：FireworksSystem + PygameRenderer（Bloom 开）"""
    pygame.init()
    width, height = 800, 600
    flags = pygame.RESIZABLE
    screen = pygame.display.set_mode((width, height), flags)
    pygame.display.set_caption("Particle System — Web Demo (pygbag)")

    fw = FireworksSystem(width=width, height=height,
                         auto_launch=True, launch_interval=0.9,
                         max_active=6, max_particles=3000)
    renderer = PygameRenderer(screen, blend_mode='additive',
                              bloom_enabled=True, bloom_intensity=0.6)
    clock = pygame.time.Clock()
    running = True

    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)  # 上限 50ms，防后台恢复跳帧

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # 点击手动发射
                mx, my = pygame.mouse.get_pos()
                fw.launch(x=mx, y=my)
            elif event.type == pygame.VIDEORESIZE:
                width, height = event.w, event.h
                screen = pygame.display.set_mode((width, height), flags)
                renderer.screen = screen
                fw.width, fw.height = width, height

        screen.fill((4, 4, 16))
        fw.update(dt)
        fw.render(renderer)
        renderer.apply_bloom()
        pygame.display.flip()

        # pygbag 约定：每帧让出控制权给浏览器事件循环
        await asyncio.sleep(0)

    pygame.quit()


if __name__ == "__main__":
    asyncio.run(main())

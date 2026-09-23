"""
流体渲染器 Demo — 展示 FluidRenderer 的流体视觉效果

运行（在仓库根目录下）：
    python examples/demo_fluid.py
"""

import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import numpy as np
import random
import math

from renderers.fluid_renderer import FluidRenderer


# ================================================================
#  鼠标交互流体
# ================================================================

class MouseFluidDemo:
    """
    鼠标拖拽产生流体效果。
    
    鼠标按下时，在鼠标位置持续发射粒子，
    用 FluidRenderer 渲染为连续液面。
    """
    
    def __init__(self, width=900, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Fluid Renderer Demo — 鼠标拖拽产生流体，ESC退出")
        self.clock = pygame.time.Clock()
        
        # 流体渲染器
        self.fluid_renderer = FluidRenderer(
            width=width,
            height=height,
            particle_radius=20,
            smooth_sigma=5.0,
            threshold=0.25,
            fluid_color=(50, 180, 255),
            ambient=0.15,
            diffuse=0.6,
            specular=0.8,
            shininess=32.0,
            additive_blend=True,
        )
        
        # 粒子系统（NumPy）
        self.max_particles = 3000
        self.positions = np.zeros((self.max_particles, 2), dtype=np.float32)
        self.velocities = np.zeros((self.max_particles, 2), dtype=np.float32)
        self.sizes = np.ones(self.max_particles, dtype=np.float32) * 15.0
        self.alphas = np.zeros(self.max_particles, dtype=np.float32)
        self.colors = np.zeros((self.max_particles, 3), dtype=np.uint8)
        self.life = np.zeros(self.max_particles, dtype=np.float32)  # 归一化 0~1
        self.max_life = np.ones(self.max_particles, dtype=np.float32)
        self.alive = np.zeros(self.max_particles, dtype=bool)
        
        self._next_slot = 0
        
        # 背景色
        self.bg_color = (10, 15, 25)
        
        # 调色板
        self.palette = [
            (50, 180, 255),   # 蓝色
            (100, 220, 200),  # 青色
            (150, 100, 255),  # 紫色
            (255, 100, 150),  # 粉色
            (255, 200, 50),   # 黄色
        ]
        
        # 鼠标状态
        self.mouse_down = False
        self.mouse_pos = (0, 0)
        self.last_mouse_pos = (0, 0)
        
        # FPS 显示
        self.font = pygame.font.Font(None, 24)
        
        # 粒子计数显示
        self.particle_count = 0
    
    def emit_at(self, x, y, count=20):
        """在指定位置发射粒子"""
        dead = np.where(~self.alive)[0]
        if len(dead) == 0:
            return
        
        actual = min(count, len(dead))
        slots = dead[:actual]
        
        # 随机速度（从鼠标运动方向偏移）
        mouse_dx = self.mouse_pos[0] - self.last_mouse_pos[0]
        mouse_dy = self.mouse_pos[1] - self.last_mouse_pos[1]
        mouse_speed = math.sqrt(mouse_dx**2 + mouse_dy**2) * 0.1
        
        angles = np.random.uniform(0, math.pi * 2, actual).astype(np.float32)
        speeds = np.random.uniform(0.5, 3.0 + mouse_speed, actual).astype(np.float32)
        
        self.positions[slots, 0] = x + np.random.uniform(-5, 5, actual)
        self.positions[slots, 1] = y + np.random.uniform(-5, 5, actual)
        self.velocities[slots, 0] = np.cos(angles) * speeds + mouse_dx * 0.05
        self.velocities[slots, 1] = np.sin(angles) * speeds + mouse_dy * 0.05
        self.sizes[slots] = np.random.uniform(8, 20, actual)
        self.alphas[slots] = 1.0
        self.life[slots] = 1.0
        self.max_life[slots] = np.random.uniform(2.0, 4.0, actual)
        self.colors[slots] = random.choice(self.palette)
        self.alive[slots] = True
    
    def update(self, dt):
        """更新粒子"""
        alive = self.alive
        
        if not np.any(alive):
            self.particle_count = 0
            return
        
        # 更新位置
        self.positions[alive] += self.velocities[alive] * dt * 60
        
        # 重力
        self.velocities[alive, 1] += 3.0 * dt * 60
        
        # 阻尼
        self.velocities[alive] *= 0.99
        
        # 更新生命值
        self.life[alive] -= dt / self.max_life[alive]
        self.alphas[alive] = np.clip(self.life[alive], 0, 1)
        
        # 标记死亡
        dead_mask = self.life <= 0
        self.alive[dead_mask] = False
        
        # 边界反弹
        mask_x_low = (self.positions[:, 0] < 0) & alive
        mask_x_high = (self.positions[:, 0] >= self.width) & alive
        mask_y_low = (self.positions[:, 1] < 0) & alive
        mask_y_high = (self.positions[:, 1] >= self.height) & alive
        
        self.velocities[mask_x_low, 0] *= -0.5
        self.velocities[mask_x_high, 0] *= -0.5
        self.velocities[mask_y_low, 1] *= -0.5
        self.velocities[mask_y_high, 1] *= -0.5
        
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0, self.width - 1)
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0, self.height - 1)
        
        self.particle_count = int(np.sum(self.alive))
    
    def render(self):
        """渲染"""
        # 收集存活粒子数据
        alive = self.alive
        if np.any(alive):
            pos = self.positions[alive]
            colors = self.colors[alive]
            sizes = self.sizes[alive]
            alphas = self.alphas[alive]
        else:
            pos = np.zeros((0, 2), dtype=np.float32)
            colors = None
            sizes = None
            alphas = None
        
        # 用 FluidRenderer 渲染
        img_array = self.fluid_renderer.render(pos, colors, sizes, alphas)
        
        # 显示
        pygame.surfarray.blit_array(self.screen, img_array.swapaxes(0, 1))
    
    def draw_ui(self, fps):
        """绘制 UI（FPS、粒子数）"""
        fps_text = self.font.render(f"FPS: {fps:.0f}", True, (200, 200, 200))
        count_text = self.font.render(f"Particles: {self.particle_count}", True, (200, 200, 200))
        hint_text = self.font.render("Drag mouse to create fluid", True, (150, 150, 150))
        
        self.screen.blit(fps_text, (10, 10))
        self.screen.blit(count_text, (10, 35))
        self.screen.blit(hint_text, (10, self.height - 30))
    
    def run(self):
        """主循环"""
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_c:
                        # 清除所有粒子
                        self.alive[:] = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # 左键
                        self.mouse_down = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.mouse_down = False
            
            # 更新鼠标位置
            self.last_mouse_pos = self.mouse_pos
            self.mouse_pos = pygame.mouse.get_pos()
            
            # 鼠标按下时发射粒子
            if self.mouse_down:
                self.emit_at(self.mouse_pos[0], self.mouse_pos[1], count=30)
            
            # 随机自动发射（让画面更丰富）
            if random.random() < 0.05:
                x = random.uniform(50, self.width - 50)
                y = random.uniform(50, self.height - 50)
                self.emit_at(x, y, count=10)
            
            # 更新
            self.update(dt)
            
            # 渲染
            self.render()
            
            # UI
            self.draw_ui(self.clock.get_fps())
            
            pygame.display.flip()
        
        pygame.quit()


# ================================================================
#  多流体颜色 Demo
# ================================================================

class MultiColorFluidDemo:
    """
    多种颜色的流体混合效果。
    
    不同颜色的流体粒子混合时，
    会呈现颜色融合的视觉效果。
    """
    
    def __init__(self, width=900, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Multi-Color Fluid Demo — 空格键切换颜色，ESC退出")
        self.clock = pygame.time.Clock()
        
        # 多个流体渲染器（每个颜色一个）
        self.colors = [
            (50, 180, 255),   # 蓝
            (255, 100, 150),  # 粉
            (100, 220, 100),  # 绿
            (255, 200, 50),   # 黄
            (150, 100, 255),  # 紫
        ]
        self.color_names = ['Blue', 'Pink', 'Green', 'Yellow', 'Purple']
        self.current_color = 0
        
        self.fluid_renderer = FluidRenderer(
            width=width,
            height=height,
            particle_radius=20,
            smooth_sigma=5.0,
            threshold=0.25,
            fluid_color=self.colors[self.current_color],
            additive_blend=True,
        )
        
        # 粒子系统（复用 MouseFluidDemo 的逻辑）
        self.max_particles = 5000
        self.positions = np.zeros((self.max_particles, 2), dtype=np.float32)
        self.velocities = np.zeros((self.max_particles, 2), dtype=np.float32)
        self.sizes = np.ones(self.max_particles, dtype=np.float32) * 15.0
        self.alphas = np.zeros(self.max_particles, dtype=np.float32)
        self.colors_arr = np.zeros((self.max_particles, 3), dtype=np.uint8)
        self.life = np.zeros(self.max_particles, dtype=np.float32)
        self.max_life = np.ones(self.max_particles, dtype=np.float32)
        self.alive = np.zeros(self.max_particles, dtype=bool)
        
        self.particle_count = 0
        self.font = pygame.font.Font(None, 24)
    
    def emit_at(self, x, y, count=20):
        """发射粒子（用当前选中的颜色）"""
        dead = np.where(~self.alive)[0]
        if len(dead) == 0:
            return
        
        actual = min(count, len(dead))
        slots = dead[:actual]
        
        angles = np.random.uniform(0, math.pi * 2, actual).astype(np.float32)
        speeds = np.random.uniform(1.0, 4.0, actual).astype(np.float32)
        
        self.positions[slots, 0] = x
        self.positions[slots, 1] = y
        self.velocities[slots, 0] = np.cos(angles) * speeds
        self.velocities[slots, 1] = np.sin(angles) * speeds
        self.sizes[slots] = np.random.uniform(8, 20, actual)
        self.alphas[slots] = 1.0
        self.life[slots] = 1.0
        self.max_life[slots] = np.random.uniform(2.0, 4.0, actual)
        self.colors_arr[slots] = self.colors[self.current_color]
        self.alive[slots] = True
    
    def update(self, dt):
        """更新粒子"""
        alive = self.alive
        
        if np.any(alive):
            # 更新位置
            self.positions[alive] += self.velocities[alive] * dt * 60
            
            # 重力 + 阻尼
            self.velocities[alive, 1] += 2.0 * dt * 60
            self.velocities[alive] *= 0.99
            
            # 生命值
            self.life[alive] -= dt / self.max_life[alive]
            self.alphas[alive] = np.clip(self.life[alive], 0, 1)
            
            # 标记死亡
            self.alive[self.life <= 0] = False
            
            # 边界
            self.positions[:, 0] = np.clip(self.positions[:, 0], 0, self.width - 1)
            self.positions[:, 1] = np.clip(self.positions[:, 1], 0, self.height - 1)
            
            # 边界反弹
            mask_x = (self.positions[:, 0] <= 0) | (self.positions[:, 0] >= self.width - 1)
            mask_y = (self.positions[:, 1] <= 0) | (self.positions[:, 1] >= self.height - 1)
            self.velocities[mask_x, 0] *= -0.5
            self.velocities[mask_y, 1] *= -0.5
        
        self.particle_count = int(np.sum(self.alive))
    
    def render(self):
        """渲染"""
        alive = self.alive
        if np.any(alive):
            pos = self.positions[alive]
            colors = self.colors_arr[alive]
            sizes = self.sizes[alive]
            alphas = self.alphas[alive]
        else:
            pos = np.zeros((0, 2), dtype=np.float32)
            colors = None
            sizes = None
            alphas = None
        
        img_array = self.fluid_renderer.render(pos, colors, sizes, alphas)
        pygame.surfarray.blit_array(self.screen, img_array.swapaxes(0, 1))
    
    def run(self):
        """主循环"""
        # 初始发射一些粒子
        for _ in range(10):
            x = random.uniform(100, self.width - 100)
            y = random.uniform(100, self.height - 100)
            self.emit_at(x, y, count=50)
        
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # 切换颜色
                        self.current_color = (self.current_color + 1) % len(self.colors)
                        self.fluid_renderer.fluid_color = np.array(
                            self.colors[self.current_color], dtype=np.float32
                        )
                    elif event.key == pygame.K_c:
                        self.alive[:] = False
            
            # 持续发射
            if self.particle_count < 2000:
                x = random.uniform(50, self.width - 50)
                y = random.uniform(50, self.height - 50)
                self.emit_at(x, y, count=5)
            
            self.update(dt)
            self.render()
            
            # UI
            fps = self.clock.get_fps()
            fps_text = self.font.render(f"FPS: {fps:.0f} | Color: {self.color_names[self.current_color]} | SPACE to switch", 
                                       True, (200, 200, 200))
            count_text = self.font.render(f"Particles: {self.particle_count}", True, (200, 200, 200))
            self.screen.blit(fps_text, (10, 10))
            self.screen.blit(count_text, (10, 35))
            
            pygame.display.flip()
        
        pygame.quit()


# ================================================================
#  主入口
# ================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Fluid Renderer Demo')
    parser.add_argument('--mode', choices=['mouse', 'multicolor'], default='mouse',
                        help='Demo mode: mouse (interactive) or multicolor')
    args = parser.parse_args()
    
    if args.mode == 'mouse':
        demo = MouseFluidDemo()
        demo.run()
    else:
        demo = MultiColorFluidDemo()
        demo.run()

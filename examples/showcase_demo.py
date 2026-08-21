"""
粒子效果展示 Demo v4.0

展示增强渲染系统 + 30 种新粒子效果

特性：
- PygameRenderer v4.0：additive/screen 混合 + Bloom 后处理
- 6 大分类 × 30 种预设效果（JSON 驱动）
- 键盘切换分类和效果，鼠标点击发射
- FPS 计数器

操作：
  1-6 : 切换分类（宇宙/魔法/科技/自然/庆典/战斗）
  SPACE : 下一个效果
  BACKSPACE : 上一个效果
  B : 切换 Bloom 开/关
  M : 循环混合模式（normal → additive → screen）
  LMB : 发射效果
  RMB : 清空
  ESC : 退出
"""

import sys
import os
import math
import random
import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.particle import Particle, ParticleSystem
from renderers.pygame_renderer import PygameRenderer
from components.effects_factory import EffectsFactory, create_effect
from components.effects import (
    create_explosion, create_fire, create_magic_circle,
    create_sparkle, create_smoke, create_fountain,
    create_vortex_effect, create_collision_sparks,
)


# ============================================================
#  配置
# ============================================================

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 800
BG_COLOR = (8, 8, 20)

CATEGORY_NAMES = {
    'cosmic': '宇宙/太空',
    'magic': '魔法/奇幻',
    'tech': '科技/科幻',
    'nature': '自然/环境',
    'celebration': '庆典/UI',
    'combat': '战斗/破坏',
}

CATEGORY_COLORS = {
    'cosmic': (180, 120, 255),
    'magic': (150, 80, 255),
    'tech': (80, 180, 255),
    'nature': (80, 255, 120),
    'celebration': (255, 200, 50),
    'combat': (255, 80, 80),
}


# ============================================================
#  主 Demo 类
# ============================================================

class ShowcaseDemo:
    def __init__(self, screen):
        self.screen = screen
        self.w, self.h = screen.get_size()

        # 效果工厂
        self.factory = EffectsFactory()
        self.all_categories = list(CATEGORY_NAMES.keys())
        self.current_cat_idx = 0
        self.current_effect_idx = 0
        self.current_effect_name = ''

        # 渲染器
        self.blend_mode = 'additive'
        self.bloom_enabled = True
        self.renderer = PygameRenderer(
            screen,
            blend_mode=self.blend_mode,
            bloom_enabled=self.bloom_enabled,
            bloom_threshold=80,
            bloom_intensity=0.6,
            bloom_sigma=10.0,
            bloom_radius=15,
            bloom_downsample=2,
            glow_enabled=True,
            default_glow_radius_mult=3.0,
        )

        # 活跃效果系统
        self.active_systems: list = []

        # 字体
        self.font = pygame.font.Font(None, 22)
        self.font_large = pygame.font.Font(None, 32)
        self.font_title = pygame.font.Font(None, 48)

        # 统计
        self.clock = pygame.time.Clock()
        self.fps = 0

        # 自动演示
        self.auto_demo_timer = 0
        self.auto_demo_interval = 2.0  # 每2秒自动发射

        print(f"[ShowcaseDemo] 初始化完成，{len(self.factory.list_effects())} 个效果可用")

    def get_category(self):
        return self.all_categories[self.current_cat_idx]

    def get_effect_names(self):
        return self.factory.list_effects(self.get_category())

    def get_current_effect(self):
        names = self.get_effect_names()
        if not names:
            return None
        self.current_effect_idx %= len(names)
        return names[self.current_effect_idx]

    def launch_effect(self, x: float, y: float):
        """在指定位置发射当前效果"""
        cat = self.get_category()
        name = self.get_current_effect()
        if not name:
            return

        system = create_effect(
            name,
            x=x, y=y,
            center_x=x, center_y=y,
            width=self.w, height=self.h,
            direction_x=random.uniform(-1, 1),
            direction_y=-1,
        )
        if system:
            system._spawn_time = pygame.time.get_ticks()
            system._max_lifetime = 5000  # 5秒最大生命周期
            self.active_systems.append(system)
            self.current_effect_name = name

    def launch_builtin(self, x: float, y: float):
        """发射内置效果（来自 effects.py）"""
        builtins = [
            lambda px, py: create_explosion(px, py, intensity=1.2),
            lambda px, py: create_magic_circle(px, py, radius=80, color=(150, 100, 255)),
            lambda px, py: create_sparkle(px, py, intensity=0.8),
            lambda px, py: create_vortex_effect(px, py, intensity=1.0),
        ]
        system = random.choice(builtins)(x, y)
        if system:
            system._spawn_time = pygame.time.get_ticks()
            system._max_lifetime = 4000
            self.active_systems.append(system)

    def clear_all(self):
        self.active_systems = []

    def update(self, dt: float):
        # 清理过期系统
        now = pygame.time.get_ticks()
        self.active_systems = [
            s for s in self.active_systems
            if len(s.particles) > 0 and
            (now - getattr(s, '_spawn_time', now)) < getattr(s, '_max_lifetime', 8000)
        ]

        # 更新所有系统
        for system in self.active_systems:
            system.update(dt)

        # 自动演示
        self.auto_demo_timer += dt
        if self.auto_demo_timer >= self.auto_demo_interval:
            self.auto_demo_timer = 0
            self.launch_effect(
                random.uniform(self.w * 0.2, self.w * 0.8),
                random.uniform(self.h * 0.2, self.h * 0.7),
            )

    def render(self):
        # 清屏
        self.screen.fill(BG_COLOR)

        # 收集所有活跃粒子
        all_particles = []
        for system in self.active_systems:
            all_particles.extend(system.particles)

        # 渲染
        self.renderer.render(all_particles)
        self.renderer.apply_bloom()

        # UI
        self._draw_ui()

    def _draw_ui(self):
        cat = self.get_category()
        effect_names = self.get_effect_names()
        idx = self.current_effect_idx % max(1, len(effect_names))

        # 标题
        title = self.font_title.render(f"粒子效果展示 v4.0", True, (255, 255, 255))
        self.screen.blit(title, (20, 15))

        # 分类标签
        x_pos = 20
        for i, c in enumerate(self.all_categories):
            color = CATEGORY_COLORS[c] if i == self.current_cat_idx else (100, 100, 100)
            label = f"[{i+1}] {CATEGORY_NAMES[c]}"
            txt = self.font.render(label, True, color)
            self.screen.blit(txt, (x_pos, 65))
            x_pos += 160

        # 当前效果
        if len(effect_names) > 0:
            name = effect_names[idx]
            idx_text = f"({idx + 1}/{len(effect_names)})"
            eff_txt = self.font_large.render(
                f"当前: {name} {idx_text}", True, (255, 255, 100)
            )
            self.screen.blit(eff_txt, (20, 95))

        # 状态栏
        fps_txt = self.font.render(f"FPS: {self.clock.get_fps():.0f}", True, (100, 255, 100))
        self.screen.blit(fps_txt, (self.w - 130, 15))

        blend_txt = self.font.render(
            f"混合: {self.blend_mode}", True, (150, 220, 255)
        )
        self.screen.blit(blend_txt, (self.w - 130, 40))

        bloom_txt = self.font.render(
            f"Bloom: {'ON' if self.bloom_enabled else 'OFF'}",
            True, (255, 200, 100) if self.bloom_enabled else (100, 100, 100)
        )
        self.screen.blit(bloom_txt, (self.w - 130, 65))

        particle_count = sum(len(s.particles) for s in self.active_systems)
        count_txt = self.font.render(
            f"粒子数: {particle_count}", True, (200, 200, 200)
        )
        self.screen.blit(count_txt, (self.w - 130, 90))

        # 操作提示
        hints = [
            "SPACE/BACKSPACE: 切换效果  |  M: 循环混合模式",
            "B: 切换 Bloom  |  LMB: 发射  |  RMB: 清屏",
            "1-6: 切换分类  |  ESC: 退出",
        ]
        for i, hint in enumerate(hints):
            txt = self.font.render(hint, True, (140, 140, 160))
            self.screen.blit(txt, (20, self.h - 80 + i * 22))

    def toggle_bloom(self):
        self.bloom_enabled = not self.bloom_enabled
        self.renderer.bloom_enabled = self.bloom_enabled

    def cycle_blend_mode(self):
        modes = ['normal', 'additive', 'screen']
        idx = modes.index(self.blend_mode)
        self.blend_mode = modes[(idx + 1) % len(modes)]
        # 重建 renderer
        self.renderer = PygameRenderer(
            self.screen,
            blend_mode=self.blend_mode,
            bloom_enabled=self.bloom_enabled,
            bloom_threshold=self.renderer.bloom_threshold,
            bloom_intensity=self.renderer.bloom_intensity,
            bloom_sigma=self.renderer.bloom_sigma,
            bloom_radius=self.renderer.bloom_radius,
            bloom_downsample=self.renderer.bloom_downsample,
            glow_enabled=self.renderer.glow_enabled,
        )

    def run(self):
        running = True
        mouse_down = False

        while running:
            dt = min(self.clock.tick(60) / 1000.0, 0.05)  # 上限 0.05s

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.current_effect_idx += 1
                        self.current_effect_name = ''
                    elif event.key == pygame.K_BACKSPACE:
                        self.current_effect_idx -= 1
                        self.current_effect_name = ''
                    elif event.key == pygame.K_b:
                        self.toggle_bloom()
                    elif event.key == pygame.K_m:
                        self.cycle_blend_mode()
                    elif event.key >= pygame.K_1 and event.key <= pygame.K_6:
                        self.current_cat_idx = event.key - pygame.K_1
                        self.current_effect_idx = 0
                        self.current_effect_name = ''
                        self.clear_all()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # LMB
                        mouse_down = True
                        self.launch_effect(*event.pos)
                    elif event.button == 3:  # RMB
                        self.clear_all()
                        self.launch_builtin(*event.pos)

            # 持续按住鼠标发射
            if mouse_down and pygame.mouse.get_pressed()[0]:
                mx, my = pygame.mouse.get_pos()
                self.launch_effect(mx, my)
            else:
                mouse_down = False

            self.update(dt)
            self.render()
            pygame.display.flip()

        pygame.quit()


# ============================================================
#  入口
# ============================================================

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("粒子效果展示 v4.0 — 30 种预设 + Bloom + Additive Blending")
    pygame.mouse.set_visible(True)

    demo = ShowcaseDemo(screen)
    demo.run()


if __name__ == '__main__':
    main()

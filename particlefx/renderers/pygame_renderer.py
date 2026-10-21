"""
Pygame 渲染器 v4.0 — 增强版

新增（v4.0）：
- 混合模式：ADDITIVE（BLEND_RGBA_ADD）/ SCREEN / NORMAL
- Bloom 后处理：多级降采样 + 分离式高斯模糊 + 叠加合成
- 粒子发光：每个粒子可选 glow_radius + glow_color
- 拖尾优化：内部用 collections.deque 替代 list.pop(0)

用法：
  renderer = PygameRenderer(screen, blend_mode='additive', bloom_enabled=True)
  renderer.render(particles)
  renderer.apply_bloom()  # 在 render 之后调用（可选）
"""

import pygame
import math
import numpy as np
from collections import deque
from ..core.particle import Particle, Renderer

# ================================================================
#  Bloom 后处理工具函数
# ================================================================

def _gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    """生成一维高斯核"""
    rng = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (rng / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def _blur_separable(arr: np.ndarray, sigma: float, radius: int) -> np.ndarray:
    """分离式高斯模糊（水平+垂直），纯 NumPy 实现"""
    kernel = _gaussian_kernel_1d(sigma, radius)
    # 水平 pass
    h, w, c = arr.shape
    result = np.zeros_like(arr)
    for i in range(-radius, radius + 1):
        shifted = np.roll(arr, i, axis=1)
        result += shifted * kernel[i + radius]
    # 垂直 pass
    arr = result.copy()
    result = np.zeros_like(arr)
    for i in range(-radius, radius + 1):
        shifted = np.roll(arr, i, axis=0)
        result += shifted * kernel[i + radius]
    return result


def _apply_bloom_pygame(surface: pygame.Surface,
                         threshold: int = 120,
                         intensity: float = 0.5,
                         sigma: float = 8.0,
                         blur_radius: int = 12,
                         downsample: int = 2) -> pygame.Surface:
    """
    对 Pygame Surface 应用 Bloom 后处理。
    
    管线：
      1. 提取亮部（threshold 以上像素）
      2. 降采样（加速模糊）
      3. 分离式高斯模糊
      4. 升采样回原分辨率
      5. BLEND_RGB_ADD 叠加回原图
    """
    w, h = surface.get_size()
    ds_w, ds_h = max(1, w // downsample), max(1, h // downsample)

    # Step 1: 降采样
    small = pygame.transform.scale(surface, (ds_w, ds_h))

    # Step 2: 转 numpy 提取亮部
    arr = pygame.surfarray.array3d(small).astype(np.float32)
    luminance = arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114
    mask = luminance > threshold
    bright = np.zeros_like(arr)
    bright[mask] = arr[mask]

    # Step 3: 分离式高斯模糊
    blurred = _blur_separable(bright, sigma / downsample, max(2, blur_radius // downsample))
    blurred = np.clip(blurred * intensity, 0, 255).astype(np.uint8)

    # Step 4: 升采样
    bloom_surf = pygame.surfarray.make_surface(blurred)
    bloom_surf = pygame.transform.smoothscale(bloom_surf, (w, h))

    # Step 5: ADD 叠加回原 surface
    result = surface.copy()
    result.blit(bloom_surf, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    return result


# ================================================================
#  PygameRenderer v4.0
# ================================================================

class PygameRenderer(Renderer):
    """Pygame 渲染器 v4.0 — 支持混合模式与 Bloom 后处理"""

    BLEND_NORMAL = 'normal'
    BLEND_ADDITIVE = 'additive'
    BLEND_SCREEN = 'screen'

    def __init__(self, screen: pygame.Surface,
                 blend_mode: str = 'normal',
                 trail_width: int = 1,
                 trail_alpha_decay: float = 0.7,
                 show_rotation_indicator: bool = True,
                 # Bloom 参数
                 bloom_enabled: bool = False,
                 bloom_threshold: int = 120,
                 bloom_intensity: float = 0.5,
                 bloom_sigma: float = 8.0,
                 bloom_radius: int = 12,
                 bloom_downsample: int = 2,
                 # 粒子发光
                 glow_enabled: bool = True,
                 default_glow_radius_mult: float = 2.5):
        """
        Args:
            screen                   : Pygame 屏幕表面
            blend_mode               : 'normal' | 'additive' | 'screen'
            trail_width              : 拖尾线宽
            trail_alpha_decay        : 拖尾透明度衰减系数
            show_rotation_indicator  : 是否显示旋转指示线
            bloom_enabled            : 是否启用 Bloom 后处理
            bloom_threshold          : 亮部提取阈值 (0-255)
            bloom_intensity          : Bloom 强度倍率
            bloom_sigma              : 高斯模糊 sigma
            bloom_radius             : 高斯模糊半径
            bloom_downsample         : 降采样倍率
            glow_enabled             : 是否绘制粒子发光光晕
            default_glow_radius_mult : 默认发光半径 = size * 此倍率
        """
        self.screen = screen
        self.blend_mode = blend_mode
        self.trail_width = trail_width
        self.trail_alpha_decay = trail_alpha_decay
        self.show_rotation_indicator = show_rotation_indicator

        # Bloom
        self.bloom_enabled = bloom_enabled
        self.bloom_threshold = bloom_threshold
        self.bloom_intensity = bloom_intensity
        self.bloom_sigma = bloom_sigma
        self.bloom_radius = bloom_radius
        self.bloom_downsample = bloom_downsample

        # 粒子发光
        self.glow_enabled = glow_enabled
        self.default_glow_radius_mult = default_glow_radius_mult

        # 中间 Surface（用于混合模式）
        self._blend_surface: pygame.Surface = None
        self._bloom_applied_this_frame: bool = False

    def _ensure_blend_surface(self, w: int, h: int):
        """确保中间混合 surface 存在且尺寸匹配"""
        if self._blend_surface is None or self._blend_surface.get_size() != (w, h):
            if self.blend_mode == self.BLEND_ADDITIVE:
                # additive 混合需要黑色透明底
                self._blend_surface = pygame.Surface((w, h), pygame.SRCALPHA)
                self._blend_surface.fill((0, 0, 0, 0))
            elif self.blend_mode == self.BLEND_SCREEN:
                # screen 混合渲染到临时 RGBA surface
                self._blend_surface = pygame.Surface((w, h), pygame.SRCALPHA)
                self._blend_surface.fill((0, 0, 0, 0))
            else:
                self._blend_surface = None

    def render(self, particles: list):
        """
        渲染粒子列表。
        当 blend_mode 为 'additive'/'screen' 时，粒子先渲染到中间 surface，
        然后通过指定混合模式 blit 到屏幕。
        """
        w, h = self.screen.get_size()

        if self.blend_mode in (self.BLEND_ADDITIVE, self.BLEND_SCREEN):
            self._ensure_blend_surface(w, h)
            # 清空中 surface
            self._blend_surface.fill((0, 0, 0, 0))
            draw_target = self._blend_surface
        else:
            draw_target = self.screen

        # --- 粒子渲染 ---
        for p in particles:
            if not p.is_alive():
                continue

            # 拖尾
            if p.trail_enabled and len(p.trail_history) >= 2:
                self._render_trail(p, draw_target)

            x = int(p.x)
            y = int(p.y)
            alpha = min(1.0, max(0.0, p.alpha))

            # 发光光晕（在主体下方）
            if self.glow_enabled:
                self._render_glow(p, x, y, alpha, draw_target)

            # 主体
            if p.mesh_mode == 'sprite' and p.mesh_surface is not None:
                self._render_sprite(p, x, y, alpha, draw_target)
            elif p.mesh_mode == 'image' and p.get_current_texture() is not None:
                self._render_texture_frame(p, x, y, alpha, draw_target)
            else:
                self._render_circle(p, x, y, alpha, draw_target)

            # 旋转指示线
            if (self.show_rotation_indicator and
                (p.angular_velocity != 0.0 or p.rotation != 0.0) and
                p.mesh_mode == 'circle'):
                self._render_rotation_indicator(p, x, y, draw_target)

        # --- 混合到屏幕 ---
        if self.blend_mode == self.BLEND_ADDITIVE:
            self.screen.blit(self._blend_surface, (0, 0),
                             special_flags=pygame.BLEND_RGBA_ADD)
        elif self.blend_mode == self.BLEND_SCREEN:
            self._blend_screen_to_screen()

        self._bloom_applied_this_frame = False

    def apply_bloom(self):
        """
        对当前屏幕应用 Bloom 后处理。
        应在 render() 之后、pygame.display.flip() 之前调用。
        """
        if not self.bloom_enabled:
            return
        if self._bloom_applied_this_frame:
            return

        bloomed = _apply_bloom_pygame(
            self.screen,
            threshold=self.bloom_threshold,
            intensity=self.bloom_intensity,
            sigma=self.bloom_sigma,
            blur_radius=self.bloom_radius,
            downsample=self.bloom_downsample,
        )
        self.screen.blit(bloomed, (0, 0))
        self._bloom_applied_this_frame = True

    def render_with_bloom(self, particles: list):
        """
        便捷方法：render + apply_bloom 一步完成。
        """
        self.render(particles)
        self.apply_bloom()

    # ================================================================
    #  内部渲染方法
    # ================================================================

    def _render_glow(self, p: Particle, x: int, y: int, alpha: float, target: pygame.Surface):
        """渲染粒子发光光晕"""
        glow_radius = getattr(p, 'glow_radius', None)
        if glow_radius is None or glow_radius <= 0:
            # 使用默认发光半径
            glow_radius = p.size * self.default_glow_radius_mult

        glow_alpha = alpha * 0.25
        if glow_alpha <= 0.01:
            return

        glow_color_rgb = getattr(p, 'glow_color', p.color[:3])
        # 创建带透明度渐变的临时 surface
        r = int(max(1, glow_radius))
        if r > 200:
            r = 200  # 性能保护
        size_s = r * 2 + 4
        glow_surf = pygame.Surface((size_s, size_s), pygame.SRCALPHA)

        # 多层绘制实现径向渐变
        steps = max(1, min(r, 5))
        for i in range(steps, 0, -1):
            ri = int(r * i / steps)
            ai = int(glow_alpha * 255 * (i / steps) ** 2)
            if ai <= 0:
                break
            color = (*glow_color_rgb, ai)
            cx = size_s // 2
            cy = size_s // 2
            pygame.draw.circle(glow_surf, color, (cx, cy), ri)

        target.blit(glow_surf, (x - size_s // 2, y - size_s // 2),
                    special_flags=pygame.BLEND_RGBA_ADD)

    def _render_circle(self, p: Particle, x: int, y: int, alpha: float, target: pygame.Surface):
        """渲染圆形粒子"""
        radius = int(max(1, p.size * alpha))
        color = (
            int(min(255, p.color[0])),
            int(min(255, p.color[1])),
            int(min(255, p.color[2])),
            int(alpha * 255),
        )
        if radius > 0:
            if self.blend_mode == self.BLEND_ADDITIVE:
                # 使用带 alpha 的 surface + BLEND_RGBA_ADD
                s = radius * 2 + 4
                circ = pygame.Surface((s, s), pygame.SRCALPHA)
                pygame.draw.circle(circ, color, (s // 2, s // 2), radius)
                target.blit(circ, (x - s // 2, y - s // 2),
                            special_flags=pygame.BLEND_RGBA_ADD)
            else:
                # 直接绘制（normal/screen 模式先绘到 target）
                pygame.draw.circle(target, color[:3], (x, y), radius)

    def _render_sprite(self, p: Particle, x: int, y: int, alpha: float, target: pygame.Surface):
        """渲染精灵/Surface 网格体"""
        surf = p.mesh_surface
        if not isinstance(surf, pygame.Surface):
            return

        size = int(max(2, p.size))
        try:
            scaled = pygame.transform.scale(surf, (size * 2, size * 2))
        except Exception:
            scaled = surf

        mod_color = (
            int(min(255, p.color[0])),
            int(min(255, p.color[1])),
            int(min(255, p.color[2])),
        )
        scaled.fill((*mod_color, int(alpha * 255)), special_flags=pygame.BLEND_RGB_MULT)

        rect = scaled.get_rect(center=(x, y))
        if self.blend_mode == self.BLEND_ADDITIVE:
            target.blit(scaled, rect, special_flags=pygame.BLEND_RGBA_ADD)
        else:
            target.blit(scaled, rect)

    def _render_texture_frame(self, p: Particle, x: int, y: int, alpha: float, target: pygame.Surface):
        """渲染纹理动画当前帧"""
        frame = p.get_current_texture()
        if not isinstance(frame, pygame.Surface):
            return

        size = int(max(2, p.size))
        try:
            scaled = pygame.transform.scale(frame, (size * 2, size * 2))
        except Exception:
            scaled = frame

        # 应用透明度
        if alpha < 1.0:
            scaled.set_alpha(int(alpha * 255))

        rect = scaled.get_rect(center=(x, y))
        if self.blend_mode == self.BLEND_ADDITIVE:
            target.blit(scaled, rect, special_flags=pygame.BLEND_RGBA_ADD)
        else:
            target.blit(scaled, rect)

    def _render_trail(self, p: Particle, target: pygame.Surface):
        """渲染拖尾"""
        points = p.get_trail_points()
        if len(points) < 2:
            return

        n = len(points)
        for i in range(n - 1):
            t = i / max(1, n - 1)
            trail_alpha = t ** self.trail_alpha_decay * min(1.0, p.alpha)
            if trail_alpha <= 0.01:
                continue

            r = int(min(255, p.color[0] * trail_alpha))
            g = int(min(255, p.color[1] * trail_alpha))
            b = int(min(255, p.color[2] * trail_alpha))

            p1 = (int(points[i][0]), int(points[i][1]))
            p2 = (int(points[i + 1][0]), int(points[i + 1][1]))

            width = max(1, int(self.trail_width * t))

            if self.blend_mode == self.BLEND_ADDITIVE:
                # 创建带 alpha 的线条 surface
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                line_len = math.sqrt(dx * dx + dy * dy)
                if line_len < 1:
                    continue
                line_surf = pygame.Surface((int(line_len) + width * 2 + 4,
                                            width * 2 + 4), pygame.SRCALPHA)
                mid_y = line_surf.get_height() // 2
                line_color = (r, g, b, int(trail_alpha * 255))
                pygame.draw.line(line_surf, line_color,
                                 (2, mid_y), (int(line_len) + 2, mid_y), width)
                # 旋转到正确角度
                angle = math.degrees(math.atan2(dy, dx))
                rotated = pygame.transform.rotate(line_surf, -angle)
                target.blit(rotated,
                            (p1[0] - rotated.get_width() // 2,
                             p1[1] - rotated.get_height() // 2),
                            special_flags=pygame.BLEND_RGBA_ADD)
            else:
                pygame.draw.line(target, (r, g, b), p1, p2, width)

    def _render_rotation_indicator(self, p: Particle, x: int, y: int, target: pygame.Surface):
        """渲染旋转指示线"""
        radius = max(int(p.size * min(1.0, p.alpha)), 1)
        line_len = radius * 1.8
        end_x = x + math.cos(p.rotation) * line_len
        end_y = y + math.sin(p.rotation) * line_len

        alpha = min(1.0, p.alpha)
        color = tuple(int(c * alpha) for c in p.color[:3])
        line_color = tuple(max(0, c - 50) for c in color)

        pygame.draw.line(target, line_color, (x, y), (int(end_x), int(end_y)), 1)

    def _blend_screen_to_screen(self):
        """
        Screen 混合模式：将中间 surface 以 screen 方式叠加到屏幕。
        screen(a, b) = 1 - (1 - a) * (1 - b) = a + b - a * b
        """
        if self._blend_surface is None:
            return

        # 读取两个 surface 为 numpy 数组
        src_arr = pygame.surfarray.array3d(self._blend_surface).astype(np.float32) / 255.0
        dst_arr = pygame.surfarray.array3d(self.screen).astype(np.float32) / 255.0

        # Screen blend
        result = 1.0 - (1.0 - src_arr) * (1.0 - dst_arr)
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)

        # 写回屏幕
        pygame.surfarray.blit_array(self.screen, result)

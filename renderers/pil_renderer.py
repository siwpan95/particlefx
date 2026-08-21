"""
PIL 渲染器 v4.0 — 全功能增强版

v4.0 新增特性：
- P2: Additive / Screen 混合模式（烟花/魔法效果专用）
- P5: 内置 Bloom 后处理（apply_bloom / post_process_bloom）
- P6: LayeredPILRenderer 分层渲染（glow + main 独立合成）
- P7: Catmull-Rom 平滑拖尾曲线（可选）

向后兼容：旧代码无需修改即可运行
"""

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import List, Optional, Tuple
from core.particle import Particle, Renderer


# ================================================================
#  辅助：Catmull-Rom 样条插值（P7）
# ================================================================

def _catmull_rom_points(points: List[Tuple[float, float]],
                        num_segments: int = 4) -> List[Tuple[int, int]]:
    """
    Catmull-Rom 样条：将折线插值为平滑曲线点列表
    
    Args:
        points      : 控制点列表 [(x,y), ...]
        num_segments: 每段细分数量
    Returns:
        插值后的点列表
    """
    if len(points) < 2:
        return [(int(p[0]), int(p[1])) for p in points]
    
    result = []
    
    # 首尾各重复一次（Catmull-Rom 边界处理）
    pts = [points[0]] + list(points) + [points[-1]]
    
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i-1], pts[i], pts[i+1], pts[i+2]
        
        for j in range(num_segments):
            t = j / num_segments
            t2 = t * t
            t3 = t2 * t
            
            # Catmull-Rom 基函数
            f0 = -0.5*t3 + t2 - 0.5*t
            f1 = 1.5*t3 - 2.5*t2 + 1.0
            f2 = -1.5*t3 + 2.0*t2 + 0.5*t
            f3 = 0.5*t3 - 0.5*t2
            
            x = f0*p0[0] + f1*p1[0] + f2*p2[0] + f3*p3[0]
            y = f0*p0[1] + f1*p1[1] + f2*p2[1] + f3*p3[1]
            result.append((int(x), int(y)))
    
    result.append((int(points[-1][0]), int(points[-1][1])))
    return result


# ================================================================
#  辅助：混合模式合成（P2）
# ================================================================

def _blend_additive(base_arr: np.ndarray,
                    src_arr: np.ndarray,
                    alpha: float) -> np.ndarray:
    """
    Additive 混合：base + src * alpha（结果截断到 0~255）
    适用于烟花、激光、魔法等发光效果 — 粒子越叠加越亮
    """
    blended = base_arr.astype(np.float32) + src_arr.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _blend_screen(base_arr: np.ndarray,
                  src_arr: np.ndarray,
                  alpha: float) -> np.ndarray:
    """
    Screen 混合：1 - (1-base)(1-src*alpha)（结果截断到 0~255）
    适用于烟花 — 比 Additive 柔和，高亮处不会过曝
    """
    base_f = base_arr.astype(np.float32) / 255.0
    src_f = src_arr.astype(np.float32) / 255.0 * alpha
    result = 1.0 - (1.0 - base_f) * (1.0 - src_f)
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


# ================================================================
#  PILRenderer v4.0
# ================================================================

class PILRenderer(Renderer):
    """
    PIL 渲染器 v4.0
    
    新增参数（v4.0）：
        blend_mode         : 'normal' | 'additive' | 'screen'
        smooth_trail       : True = 用 Catmull-Rom 平滑拖尾（P7）
        trail_segments     : 每段拖尾细分数（smooth_trail=True 时有效）
        glow_radius        : 粒子光晕半径，0 = 不绘制光晕
        glow_strength      : 光晕强度（0.0~1.0）
    """

    def __init__(self, image: Image.Image,
                 trail_width: int = 1,
                 trail_alpha_decay: float = 0.7,
                 show_rotation_indicator: bool = True,
                 # v4.0 新增
                 blend_mode: str = 'normal',
                 smooth_trail: bool = True,
                 trail_segments: int = 4,
                 glow_radius: int = 0,
                 glow_strength: float = 0.4,
                 ):
        """
        Args:
            image                  : PIL Image 对象（RGB 或 RGBA）
            trail_width            : 拖尾线宽
            trail_alpha_decay      : 拖尾透明度衰减系数
            show_rotation_indicator: 是否显示旋转指示线
            blend_mode             : 混合模式 ('normal'|'additive'|'screen')
            smooth_trail           : 是否用 Catmull-Rom 平滑拖尾曲线
            trail_segments         : Catmull-Rom 每段细分数
            glow_radius            : 粒子光晕半径（0 = 不绘制）
            glow_strength          : 光晕强度
        """
        self.image = image
        self.draw = ImageDraw.Draw(image)
        self.trail_width = trail_width
        self.trail_alpha_decay = trail_alpha_decay
        self.show_rotation_indicator = show_rotation_indicator
        self.blend_mode = blend_mode
        self.smooth_trail = smooth_trail
        self.trail_segments = trail_segments
        self.glow_radius = glow_radius
        self.glow_strength = glow_strength

    def update_image(self, image: Image.Image):
        """更换绑定的图像（用于逐帧渲染）"""
        self.image = image
        self.draw = ImageDraw.Draw(image)

    # ------------------------------------------------------------------
    #  主渲染入口
    # ------------------------------------------------------------------

    def render(self, particles: List[Particle]):
        """渲染粒子列表"""
        if self.blend_mode == 'normal':
            self._render_normal(particles)
        else:
            self._render_blended(particles)

    def _render_normal(self, particles: List[Particle]):
        """标准 alpha 混合渲染（原有逻辑）"""
        self.draw = ImageDraw.Draw(self.image)
        for p in particles:
            if not p.is_alive():
                continue
            if p.trail_enabled and len(p.trail_history) >= 2:
                self._render_trail(p)
            x, y = int(p.x), int(p.y)
            alpha = p.alpha
            if (p.mesh_mode == 'image' and
                    p.mesh_surface is not None and
                    isinstance(p.mesh_surface, Image.Image)):
                self._render_image_mesh(p, x, y, alpha)
            elif (p.get_current_texture() is not None and
                  isinstance(p.get_current_texture(), Image.Image)):
                self._render_texture_frame(p, x, y, alpha)
            else:
                self._render_circle(p, x, y, alpha)
            if (self.show_rotation_indicator and
                    (p.angular_velocity != 0.0 or p.rotation != 0.0) and
                    p.mesh_mode == 'circle'):
                self._render_rotation_indicator(p, x, y)

    def _render_blended(self, particles: List[Particle]):
        """
        Additive / Screen 混合渲染（P2）
        
        策略：在透明 RGBA 图层上绘制粒子，然后用指定混合模式合成到主图。
        """
        w, h = self.image.size
        # 粒子绘制层（RGBA 透明背景）
        layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer)

        for p in particles:
            if not p.is_alive():
                continue
            x, y = int(p.x), int(p.y)
            alpha_val = max(0.0, min(1.0, p.alpha))

            if p.trail_enabled and len(p.trail_history) >= 2:
                self._render_trail_to(p, layer_draw)

            r = int(p.color[0] * alpha_val)
            g = int(p.color[1] * alpha_val)
            b = int(p.color[2] * alpha_val)
            sz = max(1, int(p.size * alpha_val))

            if self.glow_radius > 0:
                gsz = sz + self.glow_radius
                glow_alpha = int(alpha_val * 80)
                layer_draw.ellipse(
                    [x - gsz, y - gsz, x + gsz, y + gsz],
                    fill=(r // 3, g // 3, b // 3, glow_alpha)
                )

            layer_draw.ellipse(
                [x - sz, y - sz, x + sz, y + sz],
                fill=(r, g, b, int(alpha_val * 255))
            )

        # 合成到主图
        base = self.image.convert('RGB')
        base_arr = np.array(base, dtype=np.float32)
        layer_rgb = np.array(layer.convert('RGB'), dtype=np.float32)
        layer_alpha = np.array(layer.split()[3], dtype=np.float32) / 255.0
        # 扩展 alpha 为 3 通道
        alpha3 = np.stack([layer_alpha] * 3, axis=2)

        if self.blend_mode == 'additive':
            result_arr = _blend_additive(base_arr, layer_rgb, 1.0)
            # 仅在有粒子的地方做 additive（其他地方保持原样）
            result_arr = base_arr * (1 - alpha3) + result_arr * alpha3
        else:  # screen
            result_arr = _blend_screen(base_arr, layer_rgb, 1.0)
            result_arr = base_arr * (1 - alpha3) + result_arr * alpha3

        result = Image.fromarray(np.clip(result_arr, 0, 255).astype(np.uint8))
        # 把合成结果写回原图
        self.image.paste(result)
        self.draw = ImageDraw.Draw(self.image)

    # ------------------------------------------------------------------
    #  P5 — Bloom 后处理
    # ------------------------------------------------------------------

    def apply_bloom(self,
                    radius: int = 12,
                    strength: float = 0.5,
                    threshold: int = 160) -> Image.Image:
        """
        对当前绑定图像做 Bloom 辉光处理（Screen blend）。
        
        原地修改 self.image 并返回处理后的图像。
        
        Args:
            radius    : 高斯模糊半径（越大辉光扩散越广）
            strength  : 亮部提取强度（0.0~1.0）
            threshold : 亮部阈值（0~255，越低越多区域发光）
        
        Returns:
            处理后的 PIL Image（已同步到 self.image）
        """
        result = post_process_bloom(self.image, radius, strength, threshold)
        self.image.paste(result)
        self.draw = ImageDraw.Draw(self.image)
        return self.image

    # ------------------------------------------------------------------
    #  原有渲染方法（兼容保留）
    # ------------------------------------------------------------------

    def _render_circle(self, p: Particle, x: int, y: int, alpha: float):
        radius = max(1, int(p.size * alpha))
        color = (
            int(p.color[0] * alpha),
            int(p.color[1] * alpha),
            int(p.color[2] * alpha),
        )
        self.draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color
        )

    def _render_image_mesh(self, p: Particle, x: int, y: int, alpha: float):
        """渲染 PIL Image 网格体"""
        img = p.mesh_surface
        size = max(4, int(p.size * 2))
        try:
            scaled = img.resize((size, size), Image.LANCZOS)
        except Exception:
            scaled = img
        if scaled.mode == 'RGBA':
            r, g, b, a = scaled.split()
            a = a.point(lambda i: int(i * alpha))
            scaled = Image.merge('RGBA', (r, g, b, a))
        box = (x - size // 2, y - size // 2)
        try:
            self.image.paste(scaled, box, scaled if scaled.mode == 'RGBA' else None)
        except Exception:
            pass

    def _render_texture_frame(self, p: Particle, x: int, y: int, alpha: float):
        """渲染纹理动画帧"""
        frame = p.get_current_texture()
        if not isinstance(frame, Image.Image):
            return
        size = max(4, int(p.size * 2))
        try:
            scaled = frame.resize((size, size), Image.LANCZOS)
        except Exception:
            scaled = frame
        box = (x - size // 2, y - size // 2)
        try:
            self.image.paste(scaled, box, scaled if scaled.mode == 'RGBA' else None)
        except Exception:
            pass

    def _render_trail(self, p: Particle):
        """渲染拖尾（法线模式）"""
        points = p.get_trail_points()
        self._draw_trail_on(points, p, self.draw)

    def _render_trail_to(self, p: Particle, draw: ImageDraw.Draw):
        """渲染拖尾到指定 draw 上下文（混合模式用）"""
        points = p.get_trail_points()
        self._draw_trail_on(points, p, draw)

    def _draw_trail_on(self, points: List[Tuple],
                       p: Particle, draw: ImageDraw.Draw):
        """拖尾绘制核心（P7：可选 Catmull-Rom 平滑）"""
        if len(points) < 2:
            return
        
        if self.smooth_trail and len(points) >= 3:
            # Catmull-Rom 平滑曲线插值
            smooth_pts = _catmull_rom_points(points, self.trail_segments)
        else:
            smooth_pts = [(int(pt[0]), int(pt[1])) for pt in points]
        
        n = len(smooth_pts)
        for i in range(n - 1):
            t = i / max(n - 1, 1)
            trail_alpha = (t ** self.trail_alpha_decay) * p.alpha
            
            r = int(p.color[0] * trail_alpha)
            g_c = int(p.color[1] * trail_alpha)
            b = int(p.color[2] * trail_alpha)
            
            width = max(1, int(self.trail_width * t + 1))
            draw.line(
                [smooth_pts[i], smooth_pts[i + 1]],
                fill=(r, g_c, b),
                width=width
            )

    def _render_rotation_indicator(self, p: Particle, x: int, y: int):
        radius = max(int(p.size * p.alpha), 1)
        line_len = radius * 1.8
        end_x = x + math.cos(p.rotation) * line_len
        end_y = y + math.sin(p.rotation) * line_len
        alpha = p.alpha
        color = tuple(int(c * alpha) for c in p.color[:3])
        line_color = tuple(max(0, c - 50) for c in color)
        self.draw.line(
            [(x, y), (int(end_x), int(end_y))],
            fill=line_color, width=1
        )


# ================================================================
#  P5 — 独立 Bloom 后处理函数（可在外部直接调用）
# ================================================================

def post_process_bloom(img: Image.Image,
                       radius: int = 12,
                       strength: float = 0.5,
                       threshold: int = 160) -> Image.Image:
    """
    对 PIL Image 做 Bloom 辉光后处理（Screen blend）。
    
    算法：
    1. 提取亮度 > threshold 的区域
    2. 高斯模糊（radius）
    3. Screen blend 叠加到原图
    
    Args:
        img       : 输入图像（RGB）
        radius    : 高斯模糊半径
        strength  : 亮部提取强度（0.0~1.0）
        threshold : 亮部阈值（0~255）
    
    Returns:
        处理后的新图像（不修改原图）
    """
    arr = np.array(img.convert('RGB'), dtype=np.float32)
    
    # 提取亮部
    bright_mask = arr.max(axis=2, keepdims=True) > threshold
    bright = arr * bright_mask * strength
    bright = np.clip(bright, 0, 255).astype(np.uint8)
    bright_img = Image.fromarray(bright)
    
    # 高斯模糊
    blurred = bright_img.filter(ImageFilter.GaussianBlur(radius=radius))
    
    # Screen blend：1 - (1-base)(1-bloom)
    base = arr / 255.0
    bloom = np.array(blurred, dtype=np.float32) / 255.0
    result = 1.0 - (1.0 - base) * (1.0 - bloom)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result)


# ================================================================
#  P6 — LayeredPILRenderer（分层渲染器）
# ================================================================

class LayeredPILRenderer(Renderer):
    """
    分层 PIL 渲染器（P6）
    
    将粒子分为两层渲染：
    - glow_layer  : 发光层（Screen blend，低亮度辉光）
    - main_layer  : 主粒子层（Additive blend，高亮核心）
    
    两层独立合成到背景后，可选追加 Bloom 后处理。
    
    适用于高质量烟花、魔法效果等需要发光分离的场景。
    
    用法::
    
        renderer = LayeredPILRenderer(width=1280, height=720,
                                      bloom=True, bloom_radius=14)
        bg_image = make_background(...)
        result = renderer.render_to_image(particles, background=bg_image)
    """

    def __init__(self,
                 width: int,
                 height: int,
                 bloom: bool = True,
                 bloom_radius: int = 12,
                 bloom_strength: float = 0.5,
                 bloom_threshold: int = 150,
                 glow_scale: float = 2.0,
                 glow_alpha: float = 0.35,
                 smooth_trail: bool = True,
                 ):
        """
        Args:
            width, height   : 画布尺寸
            bloom           : 是否追加 Bloom 后处理
            bloom_radius    : Bloom 半径
            bloom_strength  : Bloom 强度
            bloom_threshold : Bloom 亮部阈值
            glow_scale      : glow_layer 粒子尺寸倍率（>1 = 更大的光晕）
            glow_alpha      : glow_layer 整体透明度
            smooth_trail    : 是否平滑拖尾
        """
        self.width = width
        self.height = height
        self.bloom = bloom
        self.bloom_radius = bloom_radius
        self.bloom_strength = bloom_strength
        self.bloom_threshold = bloom_threshold
        self.glow_scale = glow_scale
        self.glow_alpha = glow_alpha
        self.smooth_trail = smooth_trail

    def render(self, particles: List[Particle]):
        """
        实现 Renderer 接口（简化版，不带背景）。
        建议使用 render_to_image() 以获得完整功能。
        """
        w, h = self.width, self.height
        blank = Image.new('RGB', (w, h), (0, 0, 0))
        return self.render_to_image(particles, background=blank)

    def render_to_image(self,
                        particles: List[Particle],
                        background: Optional[Image.Image] = None,
                        postfx=None,
                        ) -> Image.Image:
        """
        渲染粒子到图像（完整管线）。
        
        Args:
            particles  : 粒子列表
            background : 背景图（RGB），None 则用黑色背景
            postfx     : 可选 PostFXChain 后处理链（v5.0，在 Bloom 之后执行）
        
        Returns:
            合成后的 RGB 图像
        """
        w, h = self.width, self.height

        if background is None:
            canvas = Image.new('RGB', (w, h), (0, 0, 0))
        else:
            canvas = background.convert('RGB').copy()

        # —— 第一层：glow 层（Screen 合成，大圆柔光）——
        glow_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_layer)
        for p in particles:
            if not p.is_alive():
                continue
            x, y = int(p.x), int(p.y)
            a = p.alpha * self.glow_alpha
            sz = max(2, int(p.size * a * self.glow_scale))
            r = int(p.color[0] * a)
            g = int(p.color[1] * a)
            b = int(p.color[2] * a)
            glow_draw.ellipse(
                [x - sz, y - sz, x + sz, y + sz],
                fill=(r, g, b, int(a * 120))
            )

        # glow 层 Screen 合成
        canvas_arr = np.array(canvas, dtype=np.float32)
        glow_rgb = np.array(glow_layer.convert('RGB'), dtype=np.float32)
        glow_alpha_arr = np.array(glow_layer.split()[3], dtype=np.float32) / 255.0
        alpha3 = np.stack([glow_alpha_arr] * 3, axis=2)
        screen = 1.0 - (1.0 - canvas_arr / 255.0) * (1.0 - glow_rgb / 255.0)
        canvas_arr = canvas_arr / 255.0 * (1 - alpha3) + screen * alpha3
        canvas_arr = np.clip(canvas_arr * 255, 0, 255).astype(np.uint8)
        canvas = Image.fromarray(canvas_arr)

        # —— 第二层：main 粒子层（Additive 合成）——
        main_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        main_draw = ImageDraw.Draw(main_layer)
        for p in particles:
            if not p.is_alive():
                continue
            x, y = int(p.x), int(p.y)
            a = p.alpha
            sz = max(1, int(p.size * a))
            r = int(p.color[0] * a)
            g = int(p.color[1] * a)
            b_c = int(p.color[2] * a)

            # 拖尾
            if p.trail_enabled and len(p.trail_history) >= 2:
                pts = p.get_trail_points()
                if self.smooth_trail and len(pts) >= 3:
                    pts = _catmull_rom_points(pts, 4)
                else:
                    pts = [(int(pt[0]), int(pt[1])) for pt in pts]
                n = len(pts)
                for i in range(n - 1):
                    t = i / max(n - 1, 1)
                    ta = t ** 0.7 * a
                    main_draw.line(
                        [pts[i], pts[i+1]],
                        fill=(int(p.color[0]*ta), int(p.color[1]*ta),
                              int(p.color[2]*ta), int(ta * 200)),
                        width=max(1, int(t + 1))
                    )

            main_draw.ellipse(
                [x - sz, y - sz, x + sz, y + sz],
                fill=(r, g, b_c, int(a * 255))
            )

        # main 层 Additive 合成
        canvas_arr = np.array(canvas, dtype=np.float32)
        main_rgb = np.array(main_layer.convert('RGB'), dtype=np.float32)
        main_alpha_arr = np.array(main_layer.split()[3], dtype=np.float32) / 255.0
        alpha3 = np.stack([main_alpha_arr] * 3, axis=2)
        additive = np.clip(canvas_arr + main_rgb, 0, 255)
        canvas_arr = canvas_arr * (1 - alpha3) + additive * alpha3
        canvas_arr = np.clip(canvas_arr, 0, 255).astype(np.uint8)
        canvas = Image.fromarray(canvas_arr)

        # —— Bloom 后处理（可选）——
        if self.bloom:
            canvas = post_process_bloom(canvas,
                                        radius=self.bloom_radius,
                                        strength=self.bloom_strength,
                                        threshold=self.bloom_threshold)

        # —— v5.0：可编程后处理链（PostFXChain，可选）——
        if postfx is not None:
            canvas = postfx.apply(canvas)

        return canvas

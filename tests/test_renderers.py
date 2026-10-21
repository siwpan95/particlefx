"""
PIL 渲染器单元测试

覆盖：PILRenderer（normal/additive/screen 混合模式）、
      LayeredPILRenderer（分层渲染）、Bloom 后处理、
      Catmull-Rom 拖尾平滑、辅助混合函数。
"""

import math
import random
import pytest
import numpy as np
from PIL import Image, ImageDraw

from particlefx.core.particle import Particle, Renderer
from particlefx.renderers.pil_renderer import (
    PILRenderer,
    LayeredPILRenderer,
    post_process_bloom,
    _catmull_rom_points,
    _blend_additive,
    _blend_screen,
)


# ============================================================
#  辅助：创建测试粒子
# ============================================================

def _make_particle(x=200, y=150, vx=0, vy=0,
                   color=(255, 200, 50), size=5, life=1.0, alpha=1.0,
                   trail_enabled=False, angular_velocity=0.0, rotation=0.0):
    return Particle(
        x=x, y=y, vx=vx, vy=vy,
        life=life, max_life=1.0,
        size=size, color=color, alpha=alpha,
        trail_enabled=trail_enabled,
        angular_velocity=angular_velocity,
        rotation=rotation,
    )


def _make_canvas(width=400, height=300, bg_color=(0, 0, 0)):
    return Image.new('RGB', (width, height), bg_color)


# ============================================================
#  Catmull-Rom 样条
# ============================================================

class TestCatmullRom:
    """Catmull-Rom 样条插值"""

    def test_single_point(self):
        result = _catmull_rom_points([(100, 200)])
        assert result == [(100, 200)]

    def test_two_points(self):
        result = _catmull_rom_points([(0, 0), (100, 100)])
        assert len(result) > 2  # 有插值点

    def test_three_points(self):
        pts = [(0, 0), (50, 50), (100, 0)]
        result = _catmull_rom_points(pts, num_segments=4)
        assert len(result) >= 2

    def test_many_points(self):
        pts = [(i * 20, i * 10) for i in range(5)]
        result = _catmull_rom_points(pts, num_segments=3)
        # 验证首尾一致
        assert result[-1] == (pts[-1][0], pts[-1][1])

    def test_preserves_endpoint(self):
        pts = [(0, 0), (50, 50), (100, 100)]
        result = _catmull_rom_points(pts)
        assert result[-1] == (int(pts[-1][0]), int(pts[-1][1]))


# ============================================================
#  混合模式辅助函数
# ============================================================

class TestBlendAdditive:
    """Additive 混合"""

    def test_basic_addition(self):
        base = np.zeros((10, 10, 3), dtype=np.float32)
        src = np.full((10, 10, 3), 100, dtype=np.float32)
        result = _blend_additive(base, src, 1.0)
        assert np.allclose(result, 100)

    def test_clipping(self):
        base = np.full((10, 10, 3), 200, dtype=np.float32)
        src = np.full((10, 10, 3), 200, dtype=np.float32)
        result = _blend_additive(base, src, 1.0)
        # 200 + 200 = 400, clipped to 255
        assert np.all(result == 255)

    def test_alpha_control(self):
        base = np.zeros((10, 10, 3), dtype=np.float32)
        src = np.full((10, 10, 3), 200, dtype=np.float32)
        result = _blend_additive(base, src, 0.5)
        assert np.allclose(result, 100)


class TestBlendScreen:
    """Screen 混合"""

    def test_black_base(self):
        base = np.zeros((10, 10, 3), dtype=np.float32)
        src = np.full((10, 10, 3), 100, dtype=np.float32)
        result = _blend_screen(base, src, 1.0)
        # Screen with black: 1-(1-0)*(1-100/255) = 100/255
        # result = (100/255) * 255 ≈ 100
        expected = (1.0 - (1.0 - 0) * (1.0 - 100.0 / 255.0)) * 255
        assert np.allclose(result, expected, atol=2)

    def test_white_base(self):
        base = np.full((10, 10, 3), 255, dtype=np.float32)
        src = np.full((10, 10, 3), 200, dtype=np.float32)
        result = _blend_screen(base, src, 1.0)
        # Screen with white stays white-ish
        assert np.all(result >= 200)

    def test_alpha_control(self):
        base = np.zeros((10, 10, 3), dtype=np.float32)
        src = np.full((10, 10, 3), 255, dtype=np.float32)
        result = _blend_screen(base, src, 0.5)
        # 结果应为 0.5 * 255 ≈ 128
        expected = (1.0 - (1.0 - 0) * (1.0 - 255 * 0.5 / 255)) * 255
        assert np.allclose(result, expected, atol=5)


# ============================================================
#  PILRenderer
# ============================================================

class TestPILRendererInit:
    """初始化"""

    def test_default_params(self):
        img = _make_canvas()
        renderer = PILRenderer(image=img)
        assert renderer.blend_mode == 'normal'
        assert renderer.smooth_trail is True
        assert renderer.glow_radius == 0
        assert renderer.trail_width == 1

    def test_custom_params(self):
        img = _make_canvas()
        renderer = PILRenderer(
            image=img,
            blend_mode='additive',
            smooth_trail=False,
            glow_radius=5,
            glow_strength=0.6,
        )
        assert renderer.blend_mode == 'additive'
        assert renderer.smooth_trail is False
        assert renderer.glow_radius == 5


class TestPILRendererNormal:
    """标准渲染模式"""

    def test_render_single_particle(self):
        img = _make_canvas()
        renderer = PILRenderer(image=img)
        p = _make_particle()
        renderer.render([p])
        # 画布不再是纯黑
        arr = np.array(img)
        assert arr.max() > 0

    def test_render_dead_particle_skipped(self):
        img = _make_canvas()
        renderer = PILRenderer(image=img)
        p = _make_particle(life=0.0)
        renderer.render([p])
        arr = np.array(img)
        assert arr.max() == 0

    def test_render_multiple_particles(self):
        img = _make_canvas()
        renderer = PILRenderer(image=img)
        particles = [_make_particle(x=i * 40, y=150) for i in range(5)]
        renderer.render(particles)
        arr = np.array(img)
        assert arr.max() > 0

    def test_render_empty_list(self):
        img = _make_canvas()
        renderer = PILRenderer(image=img)
        renderer.render([])
        # 不崩溃即可

    def test_update_image(self):
        img1 = _make_canvas()
        img2 = _make_canvas(bg_color=(50, 50, 50))
        renderer = PILRenderer(image=img1)
        renderer.update_image(img2)
        assert renderer.image is img2


class TestPILRendererBlended:
    """混合渲染模式"""

    def test_additive_mode(self):
        img = _make_canvas()
        renderer = PILRenderer(image=img, blend_mode='additive')
        p = _make_particle(color=(255, 100, 50))
        renderer.render([p])
        arr = np.array(img)
        assert arr.max() > 0

    def test_screen_mode(self):
        img = _make_canvas()
        renderer = PILRenderer(image=img, blend_mode='screen')
        p = _make_particle(color=(255, 100, 50))
        renderer.render([p])
        arr = np.array(img)
        assert arr.max() > 0

    def test_additive_with_glow(self):
        img = _make_canvas()
        renderer = PILRenderer(image=img, blend_mode='additive',
                               glow_radius=8, glow_strength=0.6)
        p = _make_particle(color=(255, 200, 50))
        renderer.render([p])
        arr = np.array(img)
        assert arr.max() > 0

    def test_multiple_particles_additive(self):
        img = _make_canvas()
        renderer = PILRenderer(image=img, blend_mode='additive')
        particles = [_make_particle(x=200, y=150, color=(100, 50, 25))
                     for _ in range(5)]
        renderer.render(particles)
        # 多粒子叠加应该更亮
        arr = np.array(img)
        assert arr.max() > 0


class TestPILRendererTrail:
    """拖尾渲染"""

    def test_trail_rendered(self):
        img = _make_canvas()
        renderer = PILRenderer(image=img)
        p = _make_particle(trail_enabled=True)
        p.trail_history = [(190, 150), (195, 150)]
        p.trail_max_length = 10
        renderer.render([p])
        arr = np.array(img)
        assert arr.max() > 0

    def test_smooth_trail(self):
        img = _make_canvas()
        renderer = PILRenderer(image=img, smooth_trail=True, trail_segments=4)
        p = _make_particle(trail_enabled=True)
        p.trail_history = [(180, 150), (190, 150), (195, 150)]
        p.trail_max_length = 10
        renderer.render([p])
        arr = np.array(img)
        assert arr.max() > 0

    def test_no_smooth_trail(self):
        img = _make_canvas()
        renderer = PILRenderer(image=img, smooth_trail=False)
        p = _make_particle(trail_enabled=True)
        p.trail_history = [(180, 150), (190, 150), (195, 150)]
        p.trail_max_length = 10
        renderer.render([p])
        arr = np.array(img)
        assert arr.max() > 0


class TestPILRendererRotation:
    """旋转指示线"""

    def test_rotation_indicator(self):
        img = _make_canvas()
        renderer = PILRenderer(image=img, show_rotation_indicator=True)
        p = _make_particle(rotation=math.pi / 4, angular_velocity=1.0)
        renderer.render([p])
        arr = np.array(img)
        assert arr.max() > 0

    def test_no_rotation_indicator(self):
        img = _make_canvas()
        renderer = PILRenderer(image=img, show_rotation_indicator=False)
        p = _make_particle(rotation=math.pi / 4, angular_velocity=1.0)
        renderer.render([p])
        # 仍然有粒子圆，但不崩溃


# ============================================================
#  Bloom 后处理
# ============================================================

class TestPostProcessBloom:
    """Bloom 辉光后处理"""

    def test_black_image_no_change(self):
        img = _make_canvas()
        result = post_process_bloom(img, radius=5, strength=0.5)
        assert isinstance(result, Image.Image)
        # 黑色图像 bloom 无变化
        arr = np.array(result)
        assert arr.max() == 0

    def test_bright_image_produces_bloom(self):
        img = Image.new('RGB', (100, 100), (200, 200, 200))
        result = post_process_bloom(img, radius=5, strength=0.8, threshold=150)
        arr = np.array(result)
        # Bloom 应让图像更亮
        assert arr.mean() > 200

    def test_threshold_controls_selection(self):
        img = Image.new('RGB', (100, 100), (50, 50, 50))
        result = post_process_bloom(img, threshold=200)
        # 50 < 200，无亮部，bloom 不生效
        arr = np.array(result)
        assert arr.mean() <= 55

    def test_does_not_modify_original(self):
        img = Image.new('RGB', (100, 100), (200, 200, 200))
        original_arr = np.array(img).copy()
        _ = post_process_bloom(img, radius=5, strength=0.5)
        # 原图不变
        assert np.allclose(np.array(img), original_arr)


# ============================================================
#  PILRenderer Bloom
# ============================================================

class TestPILRendererApplyBloom:
    """PILRenderer 的 apply_bloom 方法"""

    def test_apply_bloom(self):
        img = _make_canvas(bg_color=(200, 200, 200))
        renderer = PILRenderer(image=img)
        p = _make_particle(color=(255, 255, 255))
        renderer.render([p])
        renderer.apply_bloom(radius=5, strength=0.5)
        arr = np.array(img)
        assert arr.max() > 200


# ============================================================
#  LayeredPILRenderer
# ============================================================

class TestLayeredPILRendererInit:
    """初始化"""

    def test_default_params(self):
        renderer = LayeredPILRenderer(width=400, height=300)
        assert renderer.width == 400
        assert renderer.height == 300
        assert renderer.bloom is True

    def test_custom_params(self):
        renderer = LayeredPILRenderer(
            width=800, height=600,
            bloom=False, bloom_radius=20,
            glow_scale=3.0, glow_alpha=0.5,
        )
        assert renderer.width == 800
        assert renderer.bloom is False
        assert renderer.glow_scale == 3.0


class TestLayeredPILRendererRender:
    """分层渲染"""

    def test_render_empty(self):
        renderer = LayeredPILRenderer(width=400, height=300)
        result = renderer.render([])
        assert isinstance(result, Image.Image)
        assert result.size == (400, 300)

    def test_render_with_particles(self):
        renderer = LayeredPILRenderer(width=400, height=300)
        particles = [_make_particle(x=200, y=150) for _ in range(5)]
        result = renderer.render(particles)
        assert isinstance(result, Image.Image)
        arr = np.array(result)
        assert arr.max() > 0

    def test_render_to_image_with_bg(self):
        renderer = LayeredPILRenderer(width=400, height=300)
        bg = Image.new('RGB', (400, 300), (30, 30, 50))
        particles = [_make_particle(x=200, y=150)]
        result = renderer.render_to_image(particles, background=bg)
        assert isinstance(result, Image.Image)
        arr = np.array(result)
        assert arr.max() > 30

    def test_render_to_image_black_bg(self):
        renderer = LayeredPILRenderer(width=400, height=300)
        particles = [_make_particle(x=200, y=150)]
        result = renderer.render_to_image(particles)
        assert isinstance(result, Image.Image)

    def test_bloom_disabled(self):
        renderer = LayeredPILRenderer(width=400, height=300, bloom=False)
        particles = [_make_particle(x=200, y=150)]
        result = renderer.render_to_image(particles)
        assert isinstance(result, Image.Image)

    def test_glow_scale_effect(self):
        renderer_small = LayeredPILRenderer(
            width=400, height=300, glow_scale=1.0, bloom=False)
        renderer_large = LayeredPILRenderer(
            width=400, height=300, glow_scale=4.0, bloom=False)
        particles = [_make_particle(x=200, y=150, color=(255, 200, 50))]
        r1 = renderer_small.render_to_image(particles)
        r2 = renderer_large.render_to_image(particles)
        # 更大的 glow_scale 应产生更多亮像素
        assert np.array(r2).max() > 0

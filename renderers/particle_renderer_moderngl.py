"""
GPU 粒子渲染器 v1.0 — ModernGL 加速版

特性：
- 实例化渲染（Instanced Rendering）：单次 Draw Call 渲染数千粒子
- 点精灵（Point Sprite）：GPU 自动将点渲染为面向相机的四边形
- 软粒子片段着色器：圆形渐变 + 发光效果
- 混合模式：ADDITIVE（glBlendFunc(GL_SRC_ALPHA, GL_ONE)）
- 内置 Bloom 后处理：Ping-Pong FBO + 分离式高斯模糊
- 兼容 Pygame OPENGL 窗口

用法：
  import pygame
  from renderers.particle_renderer_moderngl import GPUParticleRenderer

  screen = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
  renderer = GPUParticleRenderer(screen, max_particles=10000)

  while True:
      particles = [...]  # 粒子列表: [(x,y,size,r,g,b,alpha), ...]
      renderer.render(particles, apply_bloom=True)
      pygame.display.flip()
"""

import numpy as np
import moderngl
import pygame
from typing import List, Tuple, Optional

# ================================================================
#  GLSL Shader — 实例化点精灵粒子
# ================================================================

VS_PARTICLE = """
#version 330 core

layout (location = 0) in vec2 in_pos;           // 点精灵顶点（始终为 0,0）
layout (location = 1) in vec2 in_offset;         // 实例偏移（粒子位置）
layout (location = 2) in float in_size;          // 实例大小
layout (location = 3) in vec3 in_color;          // 实例颜色
layout (location = 4) in float in_alpha;         // 实例透明度

uniform mat4 u_projection;
uniform float u_point_size_scale;

out vec3 v_color;
out float v_alpha;
out float v_size;

void main()
{
    // 计算屏幕空间偏移：size 范围内的偏移
    vec2 offset = in_pos * in_size * u_point_size_scale;
    vec4 world_pos = vec4(in_offset + offset, 0.0, 1.0);
    gl_Position = u_projection * world_pos;

    v_color = in_color;
    v_alpha = in_alpha;
    v_size = in_size;
    gl_PointSize = in_size * u_point_size_scale * 2.0;
}
"""

FS_PARTICLE_SOFT = """
#version 330 core

in vec3 v_color;
in float v_alpha;
in float v_size;

out vec4 frag_color;

void main()
{
    // 点精灵坐标：gl_PointCoord 范围 [0,1]
    vec2 uv = gl_PointCoord - 0.5;
    float dist = length(uv) * 2.0;  // 0=中心, 1=边缘

    // 软圆形：中心 alpha=1.0，边缘 alpha=0.0
    float falloff = 1.0 - smoothstep(0.0, 1.0, dist);

    // 二次方让边缘更柔和
    falloff = falloff * falloff;

    float alpha = v_alpha * falloff;

    // 发光增强：中心附近额外加亮
    float glow = exp(-dist * dist * 4.0) * 0.3;
    alpha = min(1.0, alpha + glow);

    if (alpha < 0.01)
        discard;

    frag_color = vec4(v_color * (1.0 + glow), alpha);
}
"""

FS_PARTICLE_SOFT_DEPTH = """
#version 330 core

in vec3 v_color;
in float v_alpha;
in float v_size;

uniform sampler2D u_scene_depth;      // 场景深度纹理（NDC 0~1，非线性）
uniform float u_depth_fade_range;     // 软粒子淡出距离（NDC 单位）
uniform vec2 u_screen_size;           // 渲染分辨率（用于 gl_FragCoord 归一化）

out vec4 frag_color;

void main()
{
    vec2 uv = gl_PointCoord - 0.5;
    float dist = length(uv) * 2.0;
    float falloff = 1.0 - smoothstep(0.0, 1.0, dist);
    falloff = falloff * falloff;

    float alpha = v_alpha * falloff;
    float glow = exp(-dist * dist * 4.0) * 0.3;
    alpha = min(1.0, alpha + glow);

    // 软粒子：粒子深度与场景深度接近时淡出，消除与几何体的穿模硬边。
    // scene_depth - particle_depth ∈ [0, range] 时 alpha 平滑衰减到 0。
    vec2 tex_uv = gl_FragCoord.xy / u_screen_size;
    float scene_depth = texture(u_scene_depth, tex_uv).r;
    float particle_depth = gl_FragCoord.z;
    float depth_diff = scene_depth - particle_depth;
    float fade = smoothstep(0.0, u_depth_fade_range, depth_diff);
    alpha *= fade;

    if (alpha < 0.01)
        discard;

    frag_color = vec4(v_color * (1.0 + glow), alpha);
}
"""

FS_PARTICLE_RING = """
#version 330 core

in vec3 v_color;
in float v_alpha;
in float v_size;

out vec4 frag_color;

void main()
{
    vec2 uv = gl_PointCoord - 0.5;
    float dist = length(uv) * 2.0;  // 0=中心, 1=边缘

    // 环形粒子（中心空心）
    float inner = smoothstep(0.3, 0.5, dist);
    float outer = 1.0 - smoothstep(0.7, 1.0, dist);
    float ring = inner * outer;

    float alpha = v_alpha * ring;
    if (alpha < 0.01)
        discard;

    frag_color = vec4(v_color, alpha);
}
"""

# ================================================================
#  Bloom Shader
# ================================================================

VS_QUAD = """
#version 330 core
in vec2 in_vert;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    v_uv = in_vert * 0.5 + 0.5;
}
"""

FS_BRIGHT_PASS = """
#version 330 core
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_scene;
uniform float u_threshold;
void main() {
    vec3 color = texture(u_scene, v_uv).rgb;
    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
    if (brightness > u_threshold)
        frag_color = vec4(color, 1.0);
    else
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
}
"""

FS_BLUR = """
#version 330 core
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_image;
uniform bool u_horizontal;

uniform float weight[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main() {
    vec2 tex_offset = 1.0 / textureSize(u_image, 0);
    vec3 result = texture(u_image, v_uv).rgb * weight[0];

    if (u_horizontal) {
        for (int i = 1; i < 5; ++i) {
            result += texture(u_image, v_uv + vec2(tex_offset.x * i, 0.0)).rgb * weight[i];
            result += texture(u_image, v_uv - vec2(tex_offset.x * i, 0.0)).rgb * weight[i];
        }
    } else {
        for (int i = 1; i < 5; ++i) {
            result += texture(u_image, v_uv + vec2(0.0, tex_offset.y * i)).rgb * weight[i];
            result += texture(u_image, v_uv - vec2(0.0, tex_offset.y * i)).rgb * weight[i];
        }
    }
    frag_color = vec4(result, 1.0);
}
"""

FS_COMPOSITE = """
#version 330 core
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_exposure;
void main() {
    vec3 hdr = texture(u_scene, v_uv).rgb;
    vec3 bloom = texture(u_bloom, v_uv).rgb;
    hdr += bloom;
    // Reinhard tone mapping
    vec3 mapped = hdr / (hdr + vec3(1.0));
    // Gamma
    mapped = pow(mapped, vec3(1.0 / 2.2));
    frag_color = vec4(mapped, 1.0);
}
"""


# ================================================================
#  GPUParticleRenderer
# ================================================================

class GPUParticleRenderer:
    """GPU 粒子渲染器 — ModernGL 实例化渲染 + Bloom 后处理"""

    PARTICLE_VERTICES = np.array([
        -0.5, -0.5,
         0.5, -0.5,
        -0.5,  0.5,
        -0.5,  0.5,
         0.5, -0.5,
         0.5,  0.5,
    ], dtype=np.float32)

    FULLSCREEN_QUAD = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
        -1.0,  1.0,
         1.0, -1.0,
         1.0,  1.0,
    ], dtype=np.float32)

    def __init__(self, screen: pygame.Surface,
                 max_particles: int = 10000,
                 particle_style: str = 'soft',
                 point_size_scale: float = 1.0,
                 # Bloom 参数
                 bloom_enabled: bool = True,
                 bloom_threshold: float = 0.6,
                 bloom_blur_passes: int = 10,
                 bloom_exposure: float = 1.0,
                 # 渲染参数
                 background_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.05, 1.0),
                 # v5.0：软粒子深度淡出
                 enable_depth_fade: bool = False,
                 depth_fade_range: float = 0.05):
        """
        Args:
            screen              : Pygame OPENGL 窗口 surface
            max_particles       : 最大粒子数（预分配 VBO）
            particle_style      : 'soft'（软圆形）或 'ring'（环形）
            point_size_scale    : 点精灵大小缩放
            bloom_enabled       : 是否启用 Bloom 后处理
            bloom_threshold     : 亮部提取阈值 (0.0-1.0)
            bloom_blur_passes   : 高斯模糊迭代次数（2的倍数，如10=5次水平+5次垂直）
            bloom_exposure      : 色调映射曝光度
            background_color    : 清屏颜色 (r, g, b, a)
        """
        self.screen = screen
        self.width = screen.get_width()
        self.height = screen.get_height()
        self.max_particles = max_particles
        self.point_size_scale = point_size_scale
        self.background_color = background_color

        # Bloom
        self.bloom_enabled = bloom_enabled
        self.bloom_threshold = bloom_threshold
        self.bloom_blur_passes = bloom_blur_passes
        self.bloom_exposure = bloom_exposure

        # v5.0：软粒子深度淡出（需外部传入场景深度纹理）
        self.enable_depth_fade = bool(enable_depth_fade) and particle_style == 'soft'
        self.depth_fade_range = float(depth_fade_range)
        self.depth_texture = None  # 通过 set_scene_depth() 注入

        # ModernGL context
        self.ctx = moderngl.create_context()

        # 编译 shader（软粒子深度淡出用专用变体）
        if self.enable_depth_fade:
            vs_style = FS_PARTICLE_SOFT_DEPTH
        else:
            vs_style = FS_PARTICLE_SOFT if particle_style == 'soft' else FS_PARTICLE_RING
        self.prog_particle = self.ctx.program(
            vertex_shader=VS_PARTICLE,
            fragment_shader=vs_style,
        )

        if self.bloom_enabled:
            self.prog_bright = self.ctx.program(
                vertex_shader=VS_QUAD,
                fragment_shader=FS_BRIGHT_PASS,
            )
            self.prog_blur = self.ctx.program(
                vertex_shader=VS_QUAD,
                fragment_shader=FS_BLUR,
            )
            self.prog_composite = self.ctx.program(
                vertex_shader=VS_QUAD,
                fragment_shader=FS_COMPOSITE,
            )

        # --- 几何体缓冲 ---
        # 粒子实例数据 VBO（每帧更新）
        self.instance_vbo = self.ctx.buffer(
            reserve=max_particles * (2 + 1 + 3 + 1) * 4  # offset(2) + size(1) + color(3) + alpha(1) = 7 floats
        )
        self.instance_vbo.bind_to_storage_buffer(0)

        # 粒子顶点 VBO（四边形）
        self.vbo_quad = self.ctx.buffer(self.PARTICLE_VERTICES)

        # VAO for particles
        self.vao_particle = self.ctx.vertex_array(
            self.prog_particle,
            [
                (self.vbo_quad, '2f', 'in_pos'),
                (self.instance_vbo, '2f 1f 3f 1f/i', 'in_offset', 'in_size', 'in_color', 'in_alpha'),
            ]
        )

        # 全屏四边形
        self.vbo_fullscreen = self.ctx.buffer(self.FULLSCREEN_QUAD)
        if self.bloom_enabled:
            self.vao_fullscreen = self.ctx.vertex_array(
                self.prog_composite,
                [(self.vbo_fullscreen, '2f', 'in_vert')]
            )

        # --- 帧缓冲（用于 Bloom） ---
        if self.bloom_enabled:
            self._setup_bloom_fbos()

        # --- 投影矩阵 ---
        self._update_projection()

        # 启用混合
        self.ctx.enable(moderngl.BLEND)

    def _update_projection(self):
        """更新正交投影矩阵（左上角原点，y 向下）"""
        w, h = self.width, self.height
        # OpenGL 坐标系 y 向上，pygame y 向下，需要翻转
        proj = np.array([
            [2.0 / w, 0.0,      0.0, -1.0],
            [0.0,     -2.0 / h, 0.0,  1.0],
            [0.0,      0.0,     1.0,  0.0],
            [0.0,      0.0,     0.0,  1.0],
        ], dtype=np.float32)
        self.prog_particle['u_projection'].write(proj.tobytes())
        self.prog_particle['u_point_size_scale'].value = self.point_size_scale

    def _setup_bloom_fbos(self):
        """创建 Bloom 所需的帧缓冲和纹理"""
        w, h = self.width, self.height

        # 场景帧缓冲（带颜色附件）
        self.scene_tex = self.ctx.texture((w, h), 4, dtype='f1')
        self.scene_fbo = self.ctx.framebuffer(color_attachments=[self.scene_tex])

        # Ping-Pong 模糊帧缓冲
        self.ping_tex = self.ctx.texture((w, h), 4, dtype='f1')
        self.pong_tex = self.ctx.texture((w, h), 4, dtype='f1')
        self.ping_fbo = self.ctx.framebuffer(color_attachments=[self.ping_tex])
        self.pong_fbo = self.ctx.framebuffer(color_attachments=[self.pong_tex])

    def resize(self, width: int, height: int):
        """调整渲染分辨率"""
        self.width = width
        self.height = height
        self._update_projection()
        if self.bloom_enabled:
            self._setup_bloom_fbos()

    def render(self, particles: List[Tuple],
               apply_bloom: bool = True):
        """
        渲染粒子列表。

        Args:
            particles   : 粒子数据列表，每个元素为 (x, y, size, r, g, b, alpha)
                          或 (x, y, size, r, g, b, alpha, glow_radius)
            apply_bloom : 是否应用 Bloom 后处理
        """
        if not particles:
            self.ctx.clear(*self.background_color)
            return

        n = min(len(particles), self.max_particles)

        # 构建实例数据
        instance_data = np.zeros((self.max_particles, 7), dtype=np.float32)
        for i in range(n):
            p = particles[i]
            x, y = float(p[0]), float(p[1])
            sz = float(p[2]) if len(p) > 2 else 10.0
            r = float(p[3]) / 255.0 if len(p) > 3 else 1.0
            g = float(p[4]) / 255.0 if len(p) > 4 else 1.0
            b = float(p[5]) / 255.0 if len(p) > 5 else 1.0
            a = float(p[6]) if len(p) > 6 else 1.0
            instance_data[i] = [x, y, sz, r, g, b, a]

        self.instance_vbo.write(instance_data.tobytes())

        if apply_bloom and self.bloom_enabled:
            # --- Pass 1: 渲染粒子到场景 FBO ---
            self.scene_fbo.use()
            self.scene_fbo.clear(*self.background_color)

            # 设置 additive blending
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE

            self._bind_depth_fade()
            self.vao_particle.render(instances=n)

            # --- Pass 2: 亮部提取 ---
            bright_output = self.ping_fbo
            bright_output.use()
            bright_output.clear(0, 0, 0, 0)
            self.scene_tex.use(location=0)
            self.prog_bright['u_scene'].value = 0
            self.prog_bright['u_threshold'].value = self.bloom_threshold
            self.vao_fullscreen.render(mode=moderngl.TRIANGLES)

            # --- Pass 3-12: Ping-Pong 高斯模糊 ---
            horizontal = True
            first = True
            for i in range(self.bloom_blur_passes):
                if horizontal:
                    target = self.pong_fbo
                else:
                    target = self.ping_fbo

                target.use()
                target.clear(0, 0, 0, 0)

                if first:
                    self.ping_tex.use(location=0)
                elif horizontal:
                    self.ping_tex.use(location=0)
                else:
                    self.pong_tex.use(location=0)

                self.prog_blur['u_image'].value = 0
                self.prog_blur['u_horizontal'].value = horizontal
                self.vao_fullscreen.render(mode=moderngl.TRIANGLES)

                horizontal = not horizontal
                first = False

            # --- Pass 13: 合成 ---
            self.ctx.screen.use()
            self.ctx.screen.clear(*self.background_color)

            self.scene_tex.use(location=0)
            # 最后一个 blur pass 的输出在 ping/pong 纹理中
            blur_tex = self.pong_tex if self.bloom_blur_passes % 2 == 1 else self.ping_tex
            blur_tex.use(location=1)

            self.prog_composite['u_scene'].value = 0
            self.prog_composite['u_bloom'].value = 1
            self.prog_composite['u_exposure'].value = self.bloom_exposure

            self.vao_fullscreen.render(mode=moderngl.TRIANGLES)

        else:
            # --- 无 Bloom: 直接渲染到屏幕 ---
            self.ctx.screen.use()
            self.ctx.screen.clear(*self.background_color)

            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
            self._bind_depth_fade()
            self.vao_particle.render(instances=n)

        # 恢复默认混合
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    # ------------------------------------------------------------------
    #  v5.0：软粒子深度淡出
    # ------------------------------------------------------------------

    def set_scene_depth(self, depth_texture, fade_range: Optional[float] = None):
        """
        注入场景深度纹理（moderngl.Texture，单通道 R，NDC 0~1 深度）。

        软粒子原理：片元着色器比较粒子深度 gl_FragCoord.z 与场景深度，
        差值小于 depth_fade_range 时按比例淡出 alpha，消除粒子与几何体的穿模硬边。
        深度纹理通常来自场景深度预渲染 pass（如流体 SSFR 的深度贴图）。

        Args:
            depth_texture : moderngl 深度纹理（单通道）
            fade_range    : 淡出距离（NDC 单位，默认使用构造参数）
        """
        self.depth_texture = depth_texture
        if fade_range is not None:
            self.depth_fade_range = float(fade_range)

    def _bind_depth_fade(self):
        """在粒子绘制前绑定深度纹理与软粒子 uniform（未配置时无操作）"""
        if not (self.enable_depth_fade and self.depth_texture is not None):
            return
        if 'u_scene_depth' not in self.prog_particle:
            return
        self.depth_texture.use(location=1)
        self.prog_particle['u_scene_depth'].value = 1
        self.prog_particle['u_depth_fade_range'].value = self.depth_fade_range
        self.prog_particle['u_screen_size'].value = (float(self.width), float(self.height))

    def render_simple(self, particles: List[Tuple]):
        """简单快速渲染（无 Bloom）"""
        self.render(particles, apply_bloom=False)

    def release(self):
        """释放 GPU 资源"""
        self.vao_particle.release()
        self.vbo_quad.release()
        self.instance_vbo.release()
        if hasattr(self, 'vao_fullscreen'):
            self.vao_fullscreen.release()
        if hasattr(self, 'scene_fbo'):
            self.scene_fbo.release()
        if hasattr(self, 'ping_fbo'):
            self.ping_fbo.release()
        if hasattr(self, 'pong_fbo'):
            self.pong_fbo.release()

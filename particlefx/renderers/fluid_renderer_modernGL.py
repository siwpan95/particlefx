"""
ModernGL 统一 GPU 渲染器 v5.0 — 粒子/流体二合一

统一了 fluid_renderer_modernGL.py (v1)、v2.py、v3.py 的功能。

渲染路径：
  - 'fluid_ssfr'   : 屏幕空间流体渲染（Point Sprite深度+双边滤波+法线光照）
  - 'fluid_metaball': Metaball密度场（Shader内计算，效果更平滑，兼容性更好）
  - 'particles'     : 实例化粒子渲染（软圆形/环形 + 可选Bloom）
  - 'hybrid'        : 实例化粒子 + 流体合批（性能优先场景）

Bloom 后处理：Ping-Pong FBO + 分离式高斯模糊（所有路径共享）

用法：
  import pygame
  from .fluid_renderer_moderngl import UnifiedGPURenderer
  
  screen = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
  renderer = UnifiedGPURenderer(screen, mode='particles', max_particles=10000)
  renderer.render(particles)
"""

import numpy as np
import moderngl
import pygame
import math
from typing import Optional, Tuple, List


# ================================================================
#  Shader 源码库
# ================================================================

# === 全屏四边形（共享） ===
VS_QUAD = """
#version 330 core
in vec2 in_vert;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    v_uv = in_vert * 0.5 + 0.5;
}
"""

VS_QUAD_460 = """
#version 460 core
out vec2 v_uv;
void main() {
    vec2 positions[] = vec2[](vec2(-1,-1), vec2(1,-1), vec2(-1,1), vec2(1,1));
    vec2 uvs[] = vec2[](vec2(0,0), vec2(1,0), vec2(0,1), vec2(1,1));
    gl_Position = vec4(positions[gl_VertexID], 0, 1);
    v_uv = uvs[gl_VertexID];
}
"""

# === 流体 SSFR Shader (v1) ===
VS_DEPTH = """
#version 330 core
layout (location = 0) in vec2 in_pos;
layout (location = 1) in float in_size;
uniform mat4 u_mvp;
uniform float u_point_scale;
out float v_depth;

void main() {
    vec3 pos = vec3(in_pos, 0.0);
    gl_Position = u_mvp * vec4(pos, 1.0);
    gl_PointSize = max(1.0, u_point_scale * in_size / gl_Position.w);
    v_depth = (gl_Position.z / gl_Position.w + 1.0) * 0.5;
}
"""

FS_DEPTH = """
#version 330 core
layout (location = 0) out float out_depth;
in float v_depth;
void main() {
    vec2 center = gl_PointCoord * 2.0 - 1.0;
    if (dot(center, center) > 1.0) discard;
    out_depth = v_depth;
}
"""

FS_BILATERAL = """
#version 330 core
in vec2 v_uv;
layout (location = 0) out float out_depth;
uniform sampler2D u_tex;
uniform int u_horizontal;
uniform float u_sigma_space;
uniform float u_sigma_range;
uniform int u_kernel;
void main() {
    ivec2 size = textureSize(u_tex, 0);
    vec2 texel = 1.0 / vec2(size);
    float center = texture(u_tex, v_uv).r;
    if (center < 0.001) { out_depth = 0.0; return; }
    float sum = 0.0, w_total = 0.0;
    int hk = u_kernel / 2;
    for (int i = -hk; i <= hk; i++) {
        vec2 off = (u_horizontal == 1) ? vec2(float(i)*texel.x, 0) : vec2(0, float(i)*texel.y);
        float s = texture(u_tex, v_uv + off).r;
        float ws = exp(-0.5*float(i*i)/(u_sigma_space*u_sigma_space));
        float wr = exp(-0.5*(s-center)*(s-center)/(u_sigma_range*u_sigma_range));
        float w = ws * wr;
        sum += s * w; w_total += w;
    }
    out_depth = (w_total > 0.0) ? (sum / w_total) : center;
}
"""

FS_LIGHTING = """
#version 330 core
in vec2 v_uv;
layout (location = 0) out vec4 out_color;
uniform sampler2D u_depth;
uniform sampler2D u_bg;
uniform vec3 u_fluid_color;
uniform vec3 u_light_dir;
uniform float u_ambient;
uniform float u_diffuse;
uniform float u_specular;
uniform float u_shininess;
uniform float u_refract;
void main() {
    float depth = texture(u_depth, v_uv).r;
    if (depth < 0.01) { out_color = vec4(0.0); return; }
    ivec2 size = textureSize(u_depth, 0);
    vec2 texel = 1.0 / vec2(size);
    float dL = texture(u_depth, v_uv + vec2(-texel.x, 0)).r;
    float dR = texture(u_depth, v_uv + vec2( texel.x, 0)).r;
    float dD = texture(u_depth, v_uv + vec2(0, -texel.y)).r;
    float dU = texture(u_depth, v_uv + vec2(0,  texel.y)).r;
    float dx = (dR - dL) / (2.0 * texel.x);
    float dy = (dU - dD) / (2.0 * texel.y);
    vec3 N = normalize(vec3(-dx, -dy, 1.0));
    vec2 ref_uv = clamp(v_uv + N.xy * u_refract, 0.0, 1.0);
    vec3 bg = texture(u_bg, ref_uv).rgb;
    vec3 atten = exp(-(1.0 - u_fluid_color) * depth * 2.0);
    vec3 refracted = bg * atten;
    vec3 L = normalize(u_light_dir);
    float NdotL = max(dot(N, L), 0.0);
    vec3 V = vec3(0, 0, 1);
    vec3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    vec3 color = refracted * (u_ambient + u_diffuse * NdotL) * u_fluid_color;
    color += u_specular * pow(NdotH, u_shininess);
    out_color = vec4(color, clamp(depth * 1.5, 0.0, 0.85));
}
"""

# === Metaball 流体 Shader (v2/v3) ===
FS_METABALL = """
#version 330 core
in vec2 v_uv;
layout (location = 0) out vec4 out_result;
uniform sampler2D u_particle_tex;
uniform int u_num_particles;
uniform vec2 u_resolution;
uniform float u_smooth_radius;
void main() {
    vec2 pixel_pos = v_uv * u_resolution;
    float density = 0.0, thickness = 0.0;
    for (int i = 0; i < 2000; i++) {
        if (i >= u_num_particles) break;
        vec4 p = texelFetch(u_particle_tex, ivec2(i, 0), 0);
        vec2 p_pos = p.xy * u_resolution;
        float r = p.z * u_smooth_radius;
        vec2 d = pixel_pos - p_pos;
        float d2 = dot(d, d);
        float r2 = r * r;
        if (d2 < r2) {
            float c = exp(-3.0 * d2 / r2);
            density += c;
            thickness += sqrt(1.0 - d2 / r2);
        }
    }
    out_result = vec4(clamp(density, 0, 1), clamp(thickness, 0, 1), 0, 1);
}
"""

FS_METABALL_SMOOTH = """
#version 330 core
in vec2 v_uv;
layout (location = 0) out float out_density;
uniform sampler2D u_density_tex;
uniform float u_sigma_space;
uniform float u_sigma_range;
uniform int u_kernel;
void main() {
    ivec2 size = textureSize(u_density_tex, 0);
    vec2 texel = 1.0 / vec2(size);
    float center = texture(u_density_tex, v_uv).r;
    if (center < 0.001) { out_density = 0.0; return; }
    float sum = 0.0, w = 0.0;
    int hk = u_kernel / 2;
    for (int y = -hk; y <= hk; y++)
    for (int x = -hk; x <= hk; x++) {
        vec2 off = vec2(x, y) * texel;
        float s = texture(u_density_tex, v_uv + off).r;
        float ws = exp(-0.5*float(x*x+y*y)/(u_sigma_space*u_sigma_space));
        float wr = exp(-0.5*(s-center)*(s-center)/(u_sigma_range*u_sigma_range));
        float w2 = ws * wr;
        sum += s * w2; w += w2;
    }
    out_density = (w > 0) ? sum/w : center;
}
"""

# === 实例化粒子 Shader ===
VS_PARTICLE_INST = """
#version 330 core
layout (location = 0) in vec2 in_vert;
layout (location = 1) in vec2 in_offset;
layout (location = 2) in float in_size;
layout (location = 3) in vec3 in_color;
layout (location = 4) in float in_alpha;
uniform mat4 u_projection;
uniform float u_scale;
out vec3 v_color;
out float v_alpha;
out float v_size;
void main() {
    vec2 off = in_vert * in_size * u_scale;
    gl_Position = u_projection * vec4(in_offset + off, 0, 1);
    v_color = in_color;
    v_alpha = in_alpha;
    v_size = in_size;
    gl_PointSize = in_size * u_scale * 2.0;
}
"""

FS_PARTICLE_SOFT = """
#version 330 core
in vec3 v_color;
in float v_alpha;
out vec4 frag_color;

// v5.0：软粒子深度淡出（默认关闭，uniform 分支零开销）
uniform sampler2D u_scene_depth;
uniform float u_depth_fade_range;
uniform vec2 u_screen_size;
uniform bool u_depth_fade_enabled;

void main() {
    vec2 uv = gl_PointCoord - 0.5;
    float dist = length(uv) * 2.0;
    float f = (1.0 - smoothstep(0.0, 1.0, dist));
    f = f * f;
    float glow = exp(-dist * dist * 4.0) * 0.3;
    float a = min(1.0, v_alpha * f + glow);

    // 软粒子：与场景深度接近时淡出（消除穿模硬边）
    if (u_depth_fade_enabled) {
        vec2 tex_uv = gl_FragCoord.xy / u_screen_size;
        float scene_depth = texture(u_scene_depth, tex_uv).r;
        float depth_diff = scene_depth - gl_FragCoord.z;
        a *= smoothstep(0.0, u_depth_fade_range, depth_diff);
    }

    if (a < 0.01) discard;
    frag_color = vec4(v_color * (1.0 + glow), a);
}
"""

FS_PARTICLE_RING = """
#version 330 core
in vec3 v_color;
in float v_alpha;
out vec4 frag_color;
void main() {
    vec2 uv = gl_PointCoord - 0.5;
    float dist = length(uv) * 2.0;
    float ring = smoothstep(0.3, 0.5, dist) * (1.0 - smoothstep(0.7, 1.0, dist));
    float a = v_alpha * ring;
    if (a < 0.01) discard;
    frag_color = vec4(v_color, a);
}
"""

# === Bloom 后处理 Shader ===
FS_BRIGHT = """
#version 330 core
in vec2 v_uv;
out vec4 frag_color;
uniform sampler2D u_scene;
uniform float u_threshold;
void main() {
    vec3 c = texture(u_scene, v_uv).rgb;
    float b = dot(c, vec3(0.2126, 0.7152, 0.0722));
    frag_color = (b > u_threshold) ? vec4(c, 1.0) : vec4(0, 0, 0, 1.0);
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
    vec2 to = 1.0 / textureSize(u_image, 0);
    vec3 r = texture(u_image, v_uv).rgb * weight[0];
    for (int i = 1; i < 5; i++) {
        vec2 off = u_horizontal ? vec2(to.x * i, 0) : vec2(0, to.y * i);
        r += texture(u_image, v_uv + off).rgb * weight[i];
        r += texture(u_image, v_uv - off).rgb * weight[i];
    }
    frag_color = vec4(r, 1.0);
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
    vec3 hdr = texture(u_scene, v_uv).rgb + texture(u_bloom, v_uv).rgb;
    vec3 mapped = hdr / (hdr + vec3(1.0));
    mapped = pow(mapped, vec3(1.0 / 2.2));
    frag_color = vec4(mapped, 1.0);
}
"""


# ================================================================
#  UnifiedGPURenderer
# ================================================================

class UnifiedGPURenderer:
    """
    ModernGL 统一 GPU 渲染器 v5.0

    模式：
        'particles'     — 实例化粒子渲染（带 Bloom）
        'fluid_ssfr'    — 屏幕空间流体（Point Sprite + 双边滤波）
        'fluid_metaball'— Metaball 密度场流体（Shader内计算，兼容性最好）
        'hybrid'        — 粒子 + 流体合批渲染

    用法：
        pygame.init()
        screen = pygame.display.set_mode((w, h), pygame.OPENGL | pygame.DOUBLEBUF)
        renderer = UnifiedGPURenderer(screen, mode='particles', max_particles=10000)
        renderer.render(particles_data)
    """

    MODES = ('particles', 'fluid_ssfr', 'fluid_metaball', 'hybrid')

    FULLSCREEN_VERTICES = np.array([
        -1, -1,  1, -1, -1,  1,
        -1,  1,  1, -1,  1,  1,
    ], dtype=np.float32)

    PARTICLE_VERTICES = np.array([
        -0.5, -0.5,  0.5, -0.5, -0.5,  0.5,
        -0.5,  0.5,  0.5, -0.5,  0.5,  0.5,
    ], dtype=np.float32)

    def __init__(self,
                 screen: Optional[pygame.Surface] = None,
                 mode: str = 'particles',
                 width: int = 800,
                 height: int = 600,
                 max_particles: int = 10000,
                 # 粒子参数
                 particle_style: str = 'soft',
                 point_scale: float = 1.0,
                 # 流体参数
                 fluid_color: Tuple[float, float, float] = (0.2, 0.7, 1.0),
                 # Bloom 参数
                 bloom_enabled: bool = False,
                 bloom_threshold: float = 0.6,
                 bloom_blur_passes: int = 10,
                 bloom_exposure: float = 1.0,
                 # 通用
                 background_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.05, 1.0),
                 standalone: bool = False,
                 # v5.0：软粒子深度淡出
                 enable_depth_fade: bool = False,
                 depth_fade_range: float = 0.05,
                 ):
        self.screen = screen
        self.mode = mode
        self.width = width
        self.height = height
        self.max_particles = max_particles
        self.point_scale = point_scale
        self.fluid_color = np.array(fluid_color, dtype=np.float32)
        self.background_color = background_color

        # Bloom
        self.bloom_enabled = bloom_enabled
        self.bloom_threshold = bloom_threshold
        self.bloom_blur_passes = bloom_blur_passes
        self.bloom_exposure = bloom_exposure

        # v5.0：软粒子深度淡出（默认关闭）
        self.enable_depth_fade = bool(enable_depth_fade)
        self.depth_fade_range = float(depth_fade_range)
        self.depth_fade_texture = None

        # Context
        if standalone:
            self.ctx = moderngl.create_context(standalone=True)
        elif screen is not None:
            self.ctx = moderngl.get_context()
        else:
            self.ctx = moderngl.create_context(standalone=True)

        print(f"[UnifiedGPURenderer] GPU: {self.ctx.info.get('GL_RENDERER', 'unknown')}")
        print(f"[UnifiedGPURenderer] Mode: {mode}")

        self._compile_all_shaders()
        self._create_geometry()
        self._create_fbos()
        self._update_projection()

        self._particle_count = 0

    def _compile_all_shaders(self):
        """编译当前模式需要的所有 Shader"""
        m = self.mode

        # 全屏 quad shader
        self.prog_quad = self.ctx.program(vertex_shader=VS_QUAD, fragment_shader="""
            #version 330 core
            out vec4 frag; void main() { frag = vec4(1); }
        """)

        if m in ('particles', 'hybrid'):
            self.prog_particle_inst = self.ctx.program(
                vertex_shader=VS_PARTICLE_INST,
                fragment_shader=FS_PARTICLE_SOFT,
            )
            # v5.0：软粒子 uniform 初始化（默认关闭，零行为变化）
            if 'u_depth_fade_enabled' in self.prog_particle_inst:
                self.prog_particle_inst['u_depth_fade_enabled'].value = False
                self.prog_particle_inst['u_depth_fade_range'].value = self.depth_fade_range
                self.prog_particle_inst['u_screen_size'].value = (float(self.width), float(self.height))
                self.prog_particle_inst['u_scene_depth'].value = 0

        if m == 'fluid_ssfr':
            self.prog_depth = self.ctx.program(vertex_shader=VS_DEPTH, fragment_shader=FS_DEPTH)
            self.prog_bilateral = self.ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_BILATERAL)
            self.prog_lighting = self.ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_LIGHTING)

        if m == 'fluid_metaball':
            self.prog_metaball = self.ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_METABALL)
            self.prog_metaball_smooth = self.ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_METABALL_SMOOTH)
            self.prog_lighting = self.ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_LIGHTING)

        if m == 'hybrid':
            self.prog_metaball = self.ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_METABALL)
            self.prog_bilateral = self.ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_BILATERAL)
            self.prog_lighting = self.ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_LIGHTING)

        # Bloom shader（所有模式共享）
        if self.bloom_enabled:
            self.prog_bright = self.ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_BRIGHT)
            self.prog_blur = self.ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_BLUR)
            self.prog_composite = self.ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_COMPOSITE)

    def _create_geometry(self):
        """创建几何体缓冲"""
        self.vbo_fullscreen = self.ctx.buffer(self.FULLSCREEN_VERTICES)

        # 全屏 VAO（用于后处理）
        if self.bloom_enabled:
            self.vao_fullscreen = self.ctx.vertex_array(
                self.prog_composite, [(self.vbo_fullscreen, '2f', 'in_vert')]
            )

        # 粒子 VAO
        if self.mode in ('particles', 'hybrid'):
            self.vbo_particle_quad = self.ctx.buffer(self.PARTICLE_VERTICES)
            self.vbo_particle_inst = self.ctx.buffer(
                reserve=self.max_particles * 28  # 7 floats * 4 bytes
            )
            self.vao_particle = self.ctx.vertex_array(
                self.prog_particle_inst,
                [
                    (self.vbo_particle_quad, '2f', 'in_vert'),
                    (self.vbo_particle_inst, '2f 1f 3f 1f/i', 'in_offset', 'in_size', 'in_color', 'in_alpha'),
                ]
            )

        # 流体 VAO
        if self.mode == 'fluid_ssfr':
            self.vbo_fluid_pos = self.ctx.buffer(reserve=self.max_particles * 8)
            self.vbo_fluid_size = self.ctx.buffer(reserve=self.max_particles * 4)
            self.vao_fluid = self.ctx.vertex_array(
                self.prog_depth,
                [(self.vbo_fluid_pos, '2f', 'in_pos'), (self.vbo_fluid_size, '1f', 'in_size')]
            )

    def _create_fbos(self):
        """创建帧缓冲"""
        w, h = self.width, self.height

        # 场景纹理
        self.scene_tex = self.ctx.texture((w, h), 4, dtype='f4')
        self.scene_fbo = self.ctx.framebuffer(color_attachments=[self.scene_tex])

        # 流体纹理
        if self.mode in ('fluid_ssfr', 'fluid_metaball', 'hybrid'):
            self.tex_depth = self.ctx.texture((w, h), 1, dtype='f4')
            self.tex_temp = self.ctx.texture((w, h), 1, dtype='f4')
            self.tex_smoothed = self.ctx.texture((w, h), 1, dtype='f4')
            self.tex_output = self.ctx.texture((w, h), 4, dtype='f4')
            self.tex_bg = self.ctx.texture((w, h), 3, dtype='f4')

            self.fbo_depth = self.ctx.framebuffer(color_attachments=[self.tex_depth])
            self.fbo_temp = self.ctx.framebuffer(color_attachments=[self.tex_temp])
            self.fbo_smoothed = self.ctx.framebuffer(color_attachments=[self.tex_smoothed])
            self.fbo_output = self.ctx.framebuffer(color_attachments=[self.tex_output])

        # Metaball 特殊纹理
        if self.mode == 'fluid_metaball':
            self.tex_density = self.ctx.texture((w, h), 4, dtype='f4')
            self.fbo_density = self.ctx.framebuffer(color_attachments=[self.tex_density])
            self.particle_tex = self.ctx.texture((self.max_particles, 1), 4, dtype='f4')

        # Bloom FBO
        if self.bloom_enabled:
            self.ping_tex = self.ctx.texture((w, h), 4, dtype='f4')
            self.pong_tex = self.ctx.texture((w, h), 4, dtype='f4')
            self.ping_fbo = self.ctx.framebuffer(color_attachments=[self.ping_tex])
            self.pong_fbo = self.ctx.framebuffer(color_attachments=[self.pong_tex])

    def _update_projection(self):
        """更新正交投影矩阵"""
        w, h = self.width, self.height
        proj = np.array([
            [2/w, 0, 0, -1],
            [0, -2/h, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        if hasattr(self, 'prog_particle_inst'):
            self.prog_particle_inst['u_projection'].write(proj.tobytes())
            self.prog_particle_inst['u_scale'].value = self.point_scale

    # ================================================================
    #  渲染入口
    # ================================================================

    def render(self,
               particles: List[Tuple] = None,
               fluid_positions: np.ndarray = None,
               fluid_sizes: np.ndarray = None,
               background: np.ndarray = None,
               apply_bloom: bool = None,
               ) -> Optional[np.ndarray]:
        """
        统一渲染入口。

        Args:
            particles      : 粒子数据 [(x,y,size,r,g,b,alpha), ...]
            fluid_positions: 流体粒子位置 (N,2)
            fluid_sizes    : 流体粒子大小 (N,)
            background     : 背景图像 (H,W,3)
            apply_bloom    : 是否应用Bloom（默认跟随实例配置）
        Returns:
            (H,W,4) float32 RGBA 或 None（直接渲染到屏幕）
        """
        if apply_bloom is None:
            apply_bloom = self.bloom_enabled

        if background is not None and self.mode in ('fluid_ssfr', 'fluid_metaball', 'hybrid'):
            self._set_background(background)

        if self.mode == 'particles':
            self._render_particles(particles, apply_bloom)
        elif self.mode == 'fluid_ssfr':
            self._render_fluid_ssfr(fluid_positions, fluid_sizes)
        elif self.mode == 'fluid_metaball':
            return self._render_fluid_metaball(fluid_positions, fluid_sizes)
        elif self.mode == 'hybrid':
            self._render_hybrid(particles, fluid_positions, fluid_sizes, apply_bloom)

        if apply_bloom and self.mode in ('particles', 'hybrid'):
            self._apply_bloom()

        return None

    # ================================================================
    #  粒子渲染路径
    # ================================================================

    def _render_particles(self, particles, apply_bloom):
        n = min(len(particles) if particles else 0, self.max_particles)
        if n == 0:
            self.ctx.screen.clear(*self.background_color)
            return

        # 构建实例数据
        data = np.zeros((self.max_particles, 7), dtype=np.float32)
        for i in range(n):
            p = particles[i]
            data[i] = [
                float(p[0]), float(p[1]),                         # x, y
                float(p[2]) if len(p) > 2 else 10.0,              # size
                float(p[3])/255.0 if len(p) > 3 else 1.0,        # r
                float(p[4])/255.0 if len(p) > 4 else 1.0,        # g
                float(p[5])/255.0 if len(p) > 5 else 1.0,        # b
                float(p[6]) if len(p) > 6 else 1.0,              # alpha
            ]
        self.vbo_particle_inst.write(data.tobytes())

        if apply_bloom:
            self.scene_fbo.use()
            self.scene_fbo.clear(*self.background_color)
        else:
            self.ctx.screen.use()
            self.ctx.screen.clear(*self.background_color)

        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self._bind_depth_fade()
        self.vao_particle.render(instances=n)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    # ------------------------------------------------------------------
    #  v5.0：软粒子深度淡出
    # ------------------------------------------------------------------

    def set_scene_depth(self, depth_texture, fade_range: Optional[float] = None):
        """
        注入场景深度纹理（单通道 R，NDC 0~1 深度）。

        软粒子原理：片元着色器比较粒子深度与场景深度，
        差值小于 depth_fade_range 时按比例淡出 alpha（消除穿模硬边）。
        未注入纹理时深度淡出不生效（默认 uniform 分支关闭）。
        """
        self.depth_fade_texture = depth_texture
        if fade_range is not None:
            self.depth_fade_range = float(fade_range)

    def enable_depth_fade_with_fluid_depth(self, fade_range: Optional[float] = None):
        """
        便捷方法：将流体 SSFR 深度贴图（self.tex_depth）作为场景深度，
        供 hybrid 模式下粒子与流体表面之间的软粒子淡出。

        需在 _create_fbos 之后调用（tex_depth 存在时才可用）。
        """
        if not hasattr(self, 'tex_depth'):
            raise RuntimeError("当前模式没有流体深度贴图（需 fluid_ssfr/hybrid 模式）")
        self.set_scene_depth(self.tex_depth, fade_range)

    def _bind_depth_fade(self):
        """粒子绘制前绑定深度纹理与软粒子 uniform（未配置时无操作）"""
        prog = getattr(self, 'prog_particle_inst', None)
        if prog is None or 'u_scene_depth' not in prog:
            return
        if self.enable_depth_fade and self.depth_fade_texture is not None:
            self.depth_fade_texture.use(location=1)
            prog['u_scene_depth'].value = 1
            prog['u_depth_fade_enabled'].value = True
            prog['u_depth_fade_range'].value = self.depth_fade_range
            prog['u_screen_size'].value = (float(self.width), float(self.height))
        else:
            prog['u_depth_fade_enabled'].value = False

    # ================================================================
    #  流体 SSFR 渲染路径
    # ================================================================

    def _render_fluid_ssfr(self, positions, sizes):
        n = len(positions) if positions is not None else 0
        if n == 0: return

        # MVP 矩阵
        fov = math.radians(60)
        aspect = self.width / self.height
        near, far = 0.1, 1000.0
        f = 1.0 / math.tan(fov / 2)
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0,0] = f/aspect; proj[1,1] = f
        proj[2,2] = (far+near)/(near-far); proj[2,3] = 2*far*near/(near-far)
        proj[3,2] = -1
        mvp = proj.copy()

        pos2d = positions[:n].astype(np.float32)
        self.vbo_fluid_pos.write(pos2d.tobytes())
        if sizes is not None:
            sz = sizes[:n].astype(np.float32)
        else:
            sz = np.full(n, 10.0, dtype=np.float32)
        self.vbo_fluid_size.write(sz.tobytes())

        self.ctx.disable(moderngl.BLEND)

        # Pass 1: 深度写入
        self.fbo_depth.use()
        self.ctx.clear(0, 0, 0, 0)
        self.prog_depth['u_mvp'].write(mvp.tobytes())
        self.prog_depth['u_point_scale'].value = self.width / 800.0
        self.vao_fluid.render(moderngl.POINTS, n)

        # Pass 2: 双边滤波（水平+垂直）
        self.fbo_temp.use()
        self.ctx.clear(0, 0, 0, 0)
        self._setup_bilateral(self.prog_bilateral, self.tex_depth, horizontal=True)
        self._render_quad(self.prog_bilateral)

        self.fbo_smoothed.use()
        self.ctx.clear(0, 0, 0, 0)
        self._setup_bilateral(self.prog_bilateral, self.tex_temp, horizontal=False)
        self._render_quad(self.prog_bilateral)

        # Pass 3: 光照
        self.fbo_output.use()
        self.ctx.clear(0, 0, 0, 0)
        self.prog_lighting['u_depth'].value = 0
        self.tex_smoothed.use(0)
        self.prog_lighting['u_bg'].value = 1
        self.tex_bg.use(1)
        self.prog_lighting['u_fluid_color'].value = tuple(self.fluid_color)
        self.prog_lighting['u_light_dir'].value = (0.3, 0.5, 1.0)
        self.prog_lighting['u_ambient'].value = 0.15
        self.prog_lighting['u_diffuse'].value = 0.6
        self.prog_lighting['u_specular'].value = 0.8
        self.prog_lighting['u_shininess'].value = 32.0
        self.prog_lighting['u_refract'].value = 0.015
        self._render_quad(self.prog_lighting)

        # 输出到屏幕
        self._output_to_screen(self.fbo_output, self.tex_output)

    # ================================================================
    #  流体 Metaball 渲染路径
    # ================================================================

    def _render_fluid_metaball(self, positions, sizes):
        n = len(positions) if positions is not None else 0
        if n == 0:
            return np.zeros((self.height, self.width, 4), dtype=np.float32)

        # 上传粒子数据到纹理（复用 _create_fbos 中预分配的纹理，避免泄漏）
        data = np.zeros((n, 4), dtype=np.float32)
        data[:, 0] = positions[:n, 0] / self.width
        data[:, 1] = positions[:n, 1] / self.height
        data[:, 2] = sizes[:n] if sizes is not None else 10.0

        if n <= self.max_particles:
            self.particle_tex.write(data.tobytes())
        else:
            # 粒子数超过预分配纹理容量时才重建（极端情况）
            self.particle_tex.release()
            self.particle_tex = self.ctx.texture((n, 1), 4, dtype='f4', data=data.tobytes())

        # Pass 1: Metaball 密度场
        self.fbo_density.use()
        self.ctx.clear(0, 0, 0, 0)
        self.prog_metaball['u_particle_tex'].value = 0
        self.particle_tex.use(0)
        self.prog_metaball['u_num_particles'].value = n
        self.prog_metaball['u_resolution'].value = (float(self.width), float(self.height))
        self.prog_metaball['u_smooth_radius'].value = 1.0
        self._render_quad(self.prog_metaball)

        # Pass 2: 平滑
        self.fbo_smoothed.use()
        self.ctx.clear(0, 0, 0, 0)
        self.prog_metaball_smooth['u_density_tex'].value = 0
        self.tex_density.use(0)
        self.prog_metaball_smooth['u_sigma_space'].value = 2.0
        self.prog_metaball_smooth['u_sigma_range'].value = 0.1
        self.prog_metaball_smooth['u_kernel'].value = 11
        self._render_quad(self.prog_metaball_smooth)

        # Pass 3: 光照
        self.fbo_output.use()
        self.ctx.clear(0, 0, 0, 0)
        self.prog_lighting['u_depth'].value = 0
        self.tex_smoothed.use(0)
        self.prog_lighting['u_bg'].value = 1
        self.tex_bg.use(1)
        self.prog_lighting['u_fluid_color'].value = tuple(self.fluid_color)
        self.prog_lighting['u_light_dir'].value = (0.3, 0.5, 1.0)
        self.prog_lighting['u_ambient'].value = 0.15
        self.prog_lighting['u_diffuse'].value = 0.6
        self.prog_lighting['u_specular'].value = 0.8
        self.prog_lighting['u_shininess'].value = 32.0
        self.prog_lighting['u_refract'].value = 0.015
        self._render_quad(self.prog_lighting)

        data = np.frombuffer(self.tex_output.read(), dtype=np.float32)
        return data.reshape((self.height, self.width, 4))

    # ================================================================
    #  混合渲染路径
    # ================================================================

    def _render_hybrid(self, particles, fluid_positions, fluid_sizes, apply_bloom):
        # 先渲染流体，再叠加粒子
        if fluid_positions is not None:
            self._render_fluid_ssfr(fluid_positions, fluid_sizes)
        if particles:
            self._render_particles(particles, apply_bloom=False)

    # ================================================================
    #  Bloom 后处理
    # ================================================================

    def _apply_bloom(self):
        if not self.bloom_enabled:
            return

        source_tex = self.scene_tex if hasattr(self, 'scene_tex') else None
        if source_tex is None:
            return

        # 亮部提取 → ping
        self.ping_fbo.use()
        self.ping_fbo.clear(0, 0, 0, 0)
        source_tex.use(0)
        self.prog_bright['u_scene'].value = 0
        self.prog_bright['u_threshold'].value = self.bloom_threshold
        self._render_quad(self.prog_bright)

        # Ping-Pong 高斯模糊
        horizontal = True
        first = True
        for i in range(self.bloom_blur_passes):
            target = self.pong_fbo if horizontal else self.ping_fbo
            src = self.ping_tex if first or not horizontal else self.pong_tex
            if not first and horizontal:
                src = self.ping_tex

            target.use()
            target.clear(0, 0, 0, 0)
            src.use(0)
            self.prog_blur['u_image'].value = 0
            self.prog_blur['u_horizontal'].value = horizontal
            self._render_quad(self.prog_blur)

            horizontal = not horizontal
            first = False

        # 合成
        self.ctx.screen.use()
        source_tex.use(0)
        blur_tex = self.pong_tex if self.bloom_blur_passes % 2 == 1 else self.ping_tex
        blur_tex.use(1)
        self.prog_composite['u_scene'].value = 0
        self.prog_composite['u_bloom'].value = 1
        self.prog_composite['u_exposure'].value = self.bloom_exposure
        self._render_quad(self.prog_composite)

    # ================================================================
    #  工具方法
    # ================================================================

    def _setup_bilateral(self, prog, tex, horizontal=True):
        prog['u_tex'].value = 0
        tex.use(0)
        prog['u_horizontal'].value = 1 if horizontal else 0
        prog['u_sigma_space'].value = 2.0
        prog['u_sigma_range'].value = 0.1
        prog['u_kernel'].value = 11

    def _render_quad(self, prog):
        vao = self.ctx.vertex_array(prog, [(self.vbo_fullscreen, '2f', 'in_vert')])
        vao.render(moderngl.TRIANGLES)

    def _output_to_screen(self, fbo, tex):
        """将 FBO 内容输出到屏幕"""
        data = np.frombuffer(tex.read(), dtype=np.float32).reshape(self.height, self.width, 4)
        surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        data_u8 = np.clip(data * 255, 0, 255).astype(np.uint8)
        pygame.surfarray.blit_array(surf, data_u8.swapaxes(0, 1)[:, :, :3])

        if self.screen is not None:
            self.screen.fill((0, 0, 0))
            self.screen.blit(surf, (0, 0))
            pygame.display.flip()

    def _set_background(self, bg: np.ndarray):
        if bg.dtype != np.float32:
            bg = bg.astype(np.float32) / 255.0
        if bg.shape[0] != self.height or bg.shape[1] != self.width:
            surf = pygame.surfarray.make_surface((bg*255).astype(np.uint8).swapaxes(0,1))
            surf = pygame.transform.scale(surf, (self.width, self.height))
            bg = pygame.surfarray.array3d(surf).swapaxes(0,1).astype(np.float32)/255.0
        self.tex_bg.write(bg.tobytes())

    def resize(self, width: int, height: int):
        self.width = width
        self.height = height
        self._update_projection()
        self._create_fbos()

    def release(self):
        """释放 GPU 资源"""
        try:
            if hasattr(self, 'ctx'):
                self.ctx.release()
        except Exception:
            pass

    def __del__(self):
        self.release()

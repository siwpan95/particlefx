"""
光照碰撞秀视频生成器 v1.0

展示：
  Blinn-Phong 光照系统（点光源/方向光/环境光/自发光粒子）
  3种碰撞边界模式：bounce(反弹) / wrap(环绕) / destroy(销毁)
  EmitterShape 发射形状（sphere/cone/rectangle/circle/edge）

编排：
  0-30s:  bounce 模式 + 点光源旋转
  30-60s: wrap 模式 + 方向光变化
  60-80s: destroy 模式 + 自发光粒子对比
  80-100s: EmitterShape 展示（sphere/cone/rectangle）
  100-120s: 光照开关对比（关/开交替）

运行（在仓库根目录下）：
    cd <仓库根>
    python examples/make_lighting_collision_video.py

依赖：
  ffmpeg, Pillow, numpy
"""

import sys, os, math, random, time, wave, subprocess, tempfile, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 2026-09-23 去本地化：仓根改由本文件位置推导（原 D:/allin/tools/particle 已不存在）
PARTICLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARTICLE_ROOT)

from core.particle import (
    Particle, ParticleSystem, Emitter, ForceField,
    EmitterShape, CollisionBoundary, CollisionBox,
    LightSource, BlinnPhongParams,
    create_point_light, create_directional_light, create_ambient_light,
)
from renderers.pil_renderer import PILRenderer, post_process_bloom

# ================================================================
#  视频参数
# ================================================================

WIDTH, HEIGHT = 1280, 720
FPS = 30
DURATION = 120
TOTAL_FRAMES = FPS * DURATION
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(tempfile.gettempdir(), "_lighting_frames")
FRAMES_DIR = TEMP_DIR

# 分段编排
SEGMENTS = [
    # (start_sec, end_sec, label, boundary_mode, lighting_config)
    (0,   30,  "Bounce + Point Light",    'bounce',
     {'ambient': (20, 15, 30, 0.2), 'point': (640, 200, (255, 240, 200), 1.5, 400)}),
    (30,  60,  "Wrap + Directional Light", 'wrap',
     {'ambient': (15, 10, 20, 0.15), 'directional': ((255, 230, 180), 0.8, (-1, -0.5))}),
    (60,  80,  "Destroy + Emissive",       'destroy',
     {'ambient': (10, 8, 15, 0.1), 'point': (640, 360, (255, 200, 100), 2.0, 500),
      'emissive_particles': True}),
    (80,  100, "Emitter Shapes",           'bounce',
     {'ambient': (20, 15, 25, 0.2), 'point': (640, 300, (200, 180, 255), 1.2, 350)}),
    (100, 120, "Lighting ON/OFF",          'bounce',
     {'toggle_lighting': True}),
]


# ================================================================
#  音效合成
# ================================================================

def synthesize_lighting_audio(duration_sec: float = DURATION,
                               sample_rate: int = 44100) -> np.ndarray:
    total_samples = int(duration_sec * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float64)
    rng = np.random.default_rng(42)

    # 神秘氛围音
    t = np.arange(total_samples) / sample_rate
    ambient = np.sin(2*math.pi*55*t) * 0.01
    ambient += np.sin(2*math.pi*82*t) * 0.005
    mod = 0.5 + 0.5 * np.sin(2*math.pi*0.1*t)
    audio[:] += ambient * mod

    # 碰撞音效（轻柔）
    for t_sec in [2, 5, 8, 12, 18, 22, 28]:
        s = int(t_sec * sample_rate)
        dur = min(int(0.1*sample_rate), total_samples - s)
        if dur <= 0: continue
        noise = rng.uniform(-1, 1, dur) * np.exp(-np.linspace(0, 15, dur))
        audio[s:s+dur] += noise * 0.03

    peak = max(np.abs(audio).max(), 1e-9)
    return (audio / peak * 0.8 * 32767).astype(np.int16)


def write_wav(path, samples, sr=44100):
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


# ================================================================
#  主渲染流程
# ================================================================

def render_frames():
    os.makedirs(FRAMES_DIR, exist_ok=True)

    print(f"{'='*60}")
    print(f"  光照碰撞秀视频生成器 v1.0")
    print(f"  {WIDTH}x{HEIGHT} @ {FPS}fps × {DURATION}s")
    print(f"{'='*60}")

    dt = 1.0 / FPS
    ps = ParticleSystem(max_particles=800)

    renderer = PILRenderer(
        image=Image.new('RGB', (WIDTH, HEIGHT)),
        blend_mode='additive',
        glow_radius=3,
        glow_strength=0.4,
    )

    current_label = ""
    current_seg_idx = -1
    lighting_enabled = True

    print(f"开始渲染 {TOTAL_FRAMES} 帧...")

    for frame_idx in range(TOTAL_FRAMES):
        t_sec = frame_idx / FPS

        # 确定分段
        seg_idx = 0
        for i, (start, end, label, bmode, lcfg) in enumerate(SEGMENTS):
            if start <= t_sec < end:
                seg_idx = i
                break

        if seg_idx != current_seg_idx:
            current_seg_idx = seg_idx
            _, _, current_label, boundary_mode, light_cfg = SEGMENTS[seg_idx]

            # 重建系统
            ps = ParticleSystem(max_particles=800)
            ps.lighting_enabled = True
            ps.blinn_phong = BlinnPhongParams(
                ambient_strength=0.2, diffuse_strength=0.6,
                specular_strength=0.8, shininess=48.0,
                fresnel_power=4.0, fresnel_strength=0.35,
            )

            # 碰撞边界
            if boundary_mode == 'bounce':
                ps.set_boundary(CollisionBoundary(
                    left=80, top=80, right=WIDTH-80, bottom=HEIGHT-80,
                    mode='bounce', restitution=0.85, friction=0.1))
            elif boundary_mode == 'wrap':
                ps.set_boundary(CollisionBoundary(
                    left=0, top=0, right=WIDTH, bottom=HEIGHT,
                    mode='wrap'))
            elif boundary_mode == 'destroy':
                ps.set_boundary(CollisionBoundary(
                    left=100, top=100, right=WIDTH-100, bottom=HEIGHT-100,
                    mode='destroy'))

            # 力场
            ps.add_force_field(ForceField(field_type='gravity', strength=3.0))

            # 光照
            ps.lights = []
            if 'ambient' in light_cfg:
                r, g, b, intensity = light_cfg['ambient']
                ps.add_light(create_ambient_light(color=(r, g, b), intensity=intensity))

            # 发射器
            if seg_idx == 3:  # Emitter Shapes
                shapes = [
                    ('sphere', EmitterShape('sphere', radius=50, emit_from_volume=True)),
                    ('cone', EmitterShape('cone', radius=30, height=80)),
                    ('rectangle', EmitterShape('rectangle', width=60, rect_height=40)),
                ]
                for sname, shape in shapes:
                    em = Emitter(
                        x=WIDTH//3 + (shapes.index((sname, shape)) * WIDTH//3) if len(shapes) > 1 else WIDTH//2,
                        y=HEIGHT//2,
                        rate=30, spread=math.pi*0.5, speed=5,
                        life=3.0, size=3,
                        color=(200, 180, 255),
                        shape=shape,
                    )
                    ps.add_emitter(em)
                current_label += f" ({', '.join([s for s, _ in shapes])})"

            elif light_cfg.get('emissive_particles'):
                # 自发光粒子发射器
                em = Emitter(
                    x=WIDTH//2, y=HEIGHT//2,
                    rate=40, spread=math.pi*2, speed=8,
                    life=2.0, size=4,
                    color=(255, 200, 100),
                    emissive=(255, 180, 80), emissive_strength=1.5,
                )
                ps.add_emitter(em)
                # 普通粒子对比
                em2 = Emitter(
                    x=WIDTH//2 - 200, y=HEIGHT//2,
                    rate=30, spread=math.pi*2, speed=6,
                    life=2.0, size=3,
                    color=(100, 150, 255),
                )
                ps.add_emitter(em2)

            elif light_cfg.get('toggle_lighting'):
                # 简单发射器
                em = Emitter(
                    x=WIDTH//2, y=HEIGHT//2,
                    rate=50, spread=math.pi*2, speed=6,
                    life=3.0, size=3,
                    color=(200, 180, 255),
                )
                ps.add_emitter(em)

            else:
                # 普通发射器
                em = Emitter(
                    x=WIDTH//2, y=HEIGHT//2 - 100,
                    rate=40, spread=math.pi*0.8, speed=5,
                    life=3.0, size=3,
                    color=(200, 180, 255),
                )
                ps.add_emitter(em)

            print(f"  [{t_sec:.1f}s] 段切换 → {current_label}")

        # 动态点光源位置（旋转）
        for i, (start, end, label, bmode, lcfg) in enumerate(SEGMENTS):
            if start <= t_sec < end:
                if 'point' in lcfg:
                    px, py_base, color, intensity, range_ = lcfg['point']
                    # 点光源绕场景旋转
                    angle = t_sec * 0.5
                    lx = int(px + 200 * math.cos(angle))
                    ly = int(py_base + 100 * math.sin(angle))
                    ps.lights = [l for l in ps.lights if not hasattr(l, 'position')]
                    ps.add_light(create_point_light(x=lx, y=ly,
                                                     color=color, intensity=intensity, range_=range_))

                if 'directional' in lcfg:
                    color, intensity, direction = lcfg['directional']
                    # 方向随时间变化
                    dx = direction[0] + 0.3 * math.sin(t_sec * 0.2)
                    dy = direction[1] + 0.2 * math.cos(t_sec * 0.3)
                    ps.lights = [l for l in ps.lights if not hasattr(l, 'direction')]
                    ps.add_light(create_directional_light(color=color, intensity=intensity,
                                                          direction=(dx, dy)))

                # 光照开关切换（每5秒）
                if lcfg.get('toggle_lighting'):
                    cycle = int((t_sec - start) / 5)
                    lighting_enabled = (cycle % 2 == 0)
                    ps.lighting_enabled = lighting_enabled
                    if lighting_enabled:
                        current_label = "Lighting ON"
                    else:
                        current_label = "Lighting OFF"
                break

        # 更新
        ps.update(dt)

        # 渲染
        bg = Image.new('RGB', (WIDTH, HEIGHT), (8, 6, 12))

        # 绘制碰撞边界可视化
        if ps.boundary:
            draw = ImageDraw.Draw(bg)
            b = ps.boundary
            draw.rectangle([b.left, b.top, b.right, b.bottom],
                           outline=(60, 55, 70), width=1)
            # 边界模式标签
            draw.text((b.right+5, b.top), f"mode: {b.mode}", fill=(80, 75, 90))

        renderer.update_image(bg)
        renderer.render(ps.particles)

        result = post_process_bloom(renderer.image, radius=8, strength=0.3, threshold=120)

        # 标签
        draw = ImageDraw.Draw(result)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()

        draw.rectangle([10, 10, 350, 38], fill=(0, 0, 0))
        draw.text((20, 15), current_label, fill=(220, 220, 240), font=font)

        # 粒子数+光照状态
        draw.text((WIDTH-200, 10),
                  f"P:{ps.get_particle_count()}  Light:{'ON' if ps.lighting_enabled else 'OFF'}",
                  fill=(150, 150, 170), font=font)

        frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_idx:05d}.jpg")
        result.save(frame_path, quality=92)

        if frame_idx % 60 == 0:
            pct = frame_idx / TOTAL_FRAMES * 100
            print(f"  [{pct:5.1f}%] 帧 {frame_idx}/{TOTAL_FRAMES} — {current_label}")

    print(f"帧渲染完成！共 {TOTAL_FRAMES} 帧")


def encode_video():
    print("\n编码视频...")
    audio_samples = synthesize_lighting_audio()
    audio_wav = os.path.join(TEMP_DIR, "lighting_audio.wav")
    write_wav(audio_wav, audio_samples)

    pattern = os.path.join(FRAMES_DIR, "frame_%05d.jpg")
    video_tmp = os.path.join(TEMP_DIR, "lighting_noaudio.mp4")
    video_out = os.path.join(OUTPUT_DIR, "lighting_collision_show.mp4")

    cmd = ["ffmpeg", "-y", "-framerate", str(FPS), "-i", pattern,
           "-vcodec", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", video_tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("视频编码失败")

    cmd = ["ffmpeg", "-y", "-i", video_tmp, "-i", audio_wav,
           "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
           "-shortest", video_out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("音视频合并失败")
    print(f"  输出: {video_out}")
    return video_out


def cleanup():
    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)


if __name__ == "__main__":
    t0 = time.time()
    render_frames()
    output = encode_video()
    cleanup()
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  完成! 总耗时: {elapsed:.1f}s")
    print(f"  输出: {output}")
    print(f"{'='*60}")

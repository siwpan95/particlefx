"""
渲染器对比秀视频生成器 v1.0

展示同一爆炸效果在不同渲染后端的视觉差异：
  PIL: normal / additive / screen + bloom
  Pygame v4: additive+bloom（如果可用）
  CPU Fluid: metaball 融合液面（如果可用）

分屏对比模式，直观展示多渲染后端架构。

运行（在仓库根目录下）：
    cd <仓库根>
    python examples/make_renderer_compare_video.py

依赖：
  ffmpeg, Pillow, numpy
"""

import sys, os, math, random, time, wave, subprocess, tempfile, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# 2026-09-23 去本地化：仓根改由本文件位置推导（原 D:/allin/tools/particle 已不存在）
PARTICLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARTICLE_ROOT)

from core.particle import Particle, ParticleSystem, Emitter, ForceField
from components.effects import create_explosion
from renderers.pil_renderer import PILRenderer, post_process_bloom

# 尝试导入 FluidRenderer
try:
    from renderers.fluid_renderer import FluidRenderer
    HAS_FLUID = True
except ImportError:
    HAS_FLUID = False

# ================================================================
#  视频参数
# ================================================================

WIDTH, HEIGHT = 1280, 720
HALF_W = WIDTH // 2
FPS = 30
DURATION = 90  # 1.5分钟
TOTAL_FRAMES = FPS * DURATION
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(tempfile.gettempdir(), "_renderer_frames")
FRAMES_DIR = TEMP_DIR

# 分段编排
SEGMENTS = [
    # (start_sec, end_sec, left_label, left_mode, right_label, right_mode)
    (0,   15,  'PIL Normal',       'normal',   'PIL Additive',     'additive'),
    (15,  30,  'PIL Additive',     'additive', 'PIL Screen',       'screen'),
    (30,  50,  'PIL Additive',     'additive', 'PIL Additive+Bloom','additive_bloom'),
    (50,  70,  'PIL Screen+Bloom', 'screen_bloom', 'Fluid Metaball', 'fluid'),
    (70,  90,  'PIL Additive+Bloom','additive_bloom', 'PIL Screen+Bloom', 'screen_bloom'),
]


# ================================================================
#  音效合成
# ================================================================

def synthesize_compare_audio(duration_sec: float = DURATION,
                              sample_rate: int = 44100) -> np.ndarray:
    total_samples = int(duration_sec * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float64)
    rng = np.random.default_rng(42)

    # 环境音
    t = np.arange(total_samples) / sample_rate
    ambient = np.sin(2*math.pi*80*t) * 0.01 + rng.uniform(-1, 1, total_samples) * 0.003
    for _ in range(5):
        ambient = np.convolve(ambient, np.ones(8)/8, mode='same')
    audio[:] += ambient[:total_samples]

    # 定期爆炸音效（配合爆炸触发）
    for t_sec in [0, 15, 30, 50, 70]:
        s = int(t_sec * sample_rate)
        dur = min(int(0.4*sample_rate), total_samples - s)
        if dur <= 0: continue
        t_arr = np.arange(dur) / sample_rate
        env = np.exp(-np.linspace(0, 8, dur))
        boom = np.sin(2*math.pi*60*t_arr) * env * 0.4
        noise = rng.uniform(-1, 1, dur) * env * 0.2
        audio[s:s+dur] += boom + noise

    peak = max(np.abs(audio).max(), 1e-9)
    return (audio / peak * 0.8 * 32767).astype(np.int16)


def write_wav(path, samples, sr=44100):
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


# ================================================================
#  渲染函数
# ================================================================

def render_half(particles, bg_color, mode, w, h):
    """渲染半个画面"""
    bg = Image.new('RGB', (w, h), bg_color)
    renderer = PILRenderer(image=bg, blend_mode='normal', glow_radius=2, glow_strength=0.3)

    if mode == 'normal':
        renderer.blend_mode = 'normal'
        renderer.render(particles)

    elif mode == 'additive':
        renderer.blend_mode = 'additive'
        renderer.render(particles)

    elif mode == 'screen':
        renderer.blend_mode = 'screen'
        renderer.render(particles)

    elif mode == 'additive_bloom':
        renderer.blend_mode = 'additive'
        renderer.render(particles)
        bg = post_process_bloom(renderer.image, radius=12, strength=0.5, threshold=130)

    elif mode == 'screen_bloom':
        renderer.blend_mode = 'screen'
        renderer.render(particles)
        bg = post_process_bloom(renderer.image, radius=12, strength=0.5, threshold=130)

    elif mode == 'fluid':
        if HAS_FLUID:
            try:
                fluid_r = FluidRenderer(width=w, height=h, particle_radius=15,
                                         smooth_sigma=4.0, threshold=0.25,
                                         fluid_color=(50, 180, 255),
                                         additive_blend=True)
                active_p = [p for p in particles if p.is_alive()]
                if active_p:
                    positions = np.array([[p.x, p.y] for p in active_p], dtype=np.float32)
                    sizes = np.array([p.size for p in active_p], dtype=np.float32)
                    img_arr = fluid_r.render(positions, sizes, background=np.array(bg))
                    bg = Image.fromarray(np.clip(img_arr[:,:,:3]*255, 0, 255).astype(np.uint8)).convert('RGB')
                # 渲染普通粒子叠加
                renderer.update_image(bg)
                renderer.blend_mode = 'additive'
                renderer.render(particles)
                bg = renderer.image
            except Exception:
                # 失败则回退到 additive
                renderer.blend_mode = 'additive'
                renderer.render(particles)
                bg = post_process_bloom(renderer.image, radius=8, strength=0.3, threshold=120)
        else:
            # 无 FluidRenderer，显示提示文字
            renderer.blend_mode = 'additive'
            renderer.render(particles)
            bg = renderer.image
            draw = ImageDraw.Draw(bg)
            draw.text((20, h//2), "Fluid Renderer\nNot Available", fill=(200, 200, 200))

    else:
        renderer.render(particles)

    return bg


# ================================================================
#  主渲染流程
# ================================================================

def render_frames():
    os.makedirs(FRAMES_DIR, exist_ok=True)

    print(f"{'='*60}")
    print(f"  渲染器对比秀视频生成器 v1.0")
    print(f"  {WIDTH}x{HEIGHT} @ {FPS}fps × {DURATION}s")
    print(f"  FluidRenderer: {'可用' if HAS_FLUID else '不可用'}")
    print(f"{'='*60}")

    dt = 1.0 / FPS
    current_segment_idx = -1
    left_label, left_mode = '', 'normal'
    right_label, right_mode = '', 'additive'
    explosion_sys = None
    need_new_explosion = True

    print(f"开始渲染 {TOTAL_FRAMES} 帧...")

    for frame_idx in range(TOTAL_FRAMES):
        t_sec = frame_idx / FPS

        # 确定当前分段
        seg_idx = 0
        for i, (start, end, ll, lm, rl, rm) in enumerate(SEGMENTS):
            if start <= t_sec < end:
                seg_idx = i
                break

        if seg_idx != current_segment_idx:
            current_segment_idx = seg_idx
            _, _, left_label, left_mode, right_label, right_mode = SEGMENTS[seg_idx]
            need_new_explosion = True
            print(f"  [{t_sec:.1f}s] 段切换 → 左:{left_label} | 右:{right_label}")

        # 新爆炸
        if need_new_explosion:
            explosion_sys = create_explosion(x=HALF_W//2 + 10, y=HEIGHT//2, intensity=1.5)
            need_new_explosion = False

        # 更新
        explosion_sys.update(dt)

        # 粒子数不够时重新触发爆炸
        if explosion_sys.get_particle_count() < 5 and t_sec > 5:
            need_new_explosion = True

        # 左半渲染
        left_bg_color = (8, 6, 12)
        left_img = render_half(explosion_sys.particles, left_bg_color,
                                left_mode, HALF_W, HEIGHT)

        # 右半渲染（粒子位置偏移到右半区域）
        right_particles = []
        for p in explosion_sys.particles:
            rp = Particle(
                x=p.x + HALF_W, y=p.y,
                vx=p.vx, vy=p.vy,
                life=p.life, max_life=p.max_life,
                size=p.size, color=p.color, alpha=p.alpha,
            )
            right_particles.append(rp)

        right_bg_color = (8, 6, 12)
        right_img = render_half(right_particles, right_bg_color,
                                right_mode, HALF_W, HEIGHT)

        # 合并两半画面
        result = Image.new('RGB', (WIDTH, HEIGHT))
        result.paste(left_img, (0, 0))
        result.paste(right_img, (HALF_W, 0))

        # 中间分隔线
        draw = ImageDraw.Draw(result)
        draw.line([(HALF_W, 0), (HALF_W, HEIGHT)], fill=(60, 60, 70), width=2)

        # 标签
        try:
            font = ImageFont.truetype("arial.ttf", 18)
            small_font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
            small_font = font

        draw.rectangle([10, 10, HALF_W-10, 38], fill=(0, 0, 0))
        draw.text((20, 15), left_label, fill=(220, 220, 240), font=font)

        draw.rectangle([HALF_W+10, 10, WIDTH-10, 38], fill=(0, 0, 0))
        draw.text((HALF_W+20, 15), right_label, fill=(220, 220, 240), font=font)

        draw.text((WIDTH-150, HEIGHT-40),
                  f"Particles: {explosion_sys.get_particle_count()}",
                  fill=(150, 150, 170), font=small_font)

        # 保存帧
        frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_idx:05d}.jpg")
        result.save(frame_path, quality=92)

        if frame_idx % 30 == 0:
            pct = frame_idx / TOTAL_FRAMES * 100
            print(f"  [{pct:5.1f}%] 帧 {frame_idx}/{TOTAL_FRAMES}")

    print(f"帧渲染完成！共 {TOTAL_FRAMES} 帧")


def encode_video():
    print("\n编码视频...")
    audio_samples = synthesize_compare_audio()
    audio_wav = os.path.join(TEMP_DIR, "renderer_audio.wav")
    write_wav(audio_wav, audio_samples)

    pattern = os.path.join(FRAMES_DIR, "frame_%05d.jpg")
    video_tmp = os.path.join(TEMP_DIR, "renderer_noaudio.mp4")
    video_out = os.path.join(OUTPUT_DIR, "renderer_compare.mp4")

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

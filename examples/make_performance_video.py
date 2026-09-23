"""
性能极限秀视频生成器 v1.0

展示：
  NumPy SoA 向量化粒子系统性能
  500 → 1000 → 2000 → 5000 → 10000 粒子实时帧率
  Python ParticleSystem vs NumPy ParticleSystemNumpy 对比
  GPU 渲染器（如果可用）

画面左角实时显示 FPS 和粒子数量。

运行（在仓库根目录下）：
    cd <仓库根>
    python examples/make_performance_video.py

依赖：
  ffmpeg, Pillow, numpy
"""

import sys, os, math, random, time, wave, subprocess, tempfile, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 2026-09-23 去本地化：仓根改由本文件位置推导（原 D:/allin/tools/particle 已不存在）
PARTICLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARTICLE_ROOT)

from core.particle import ParticleSystem, Emitter, ForceField
from core.particle_numpy import (
    ParticleSystemNumpy, NumpyForceGravity, NumpyForceDrag, NumpyForceTurbulence,
)
from renderers.pil_renderer import PILRenderer, post_process_bloom

# ================================================================
#  视频参数
# ================================================================

WIDTH, HEIGHT = 1280, 720
FPS = 30
DURATION = 120  # 2分钟
TOTAL_FRAMES = FPS * DURATION
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(tempfile.gettempdir(), "_perf_frames")
FRAMES_DIR = TEMP_DIR

# 性能阶梯编排（秒 → 粒子数, 系统 类型）
PERF_STEPS = [
    (0,   15,  500,   'python'),
    (15,  30,  1000,  'python'),
    (30,  45,  2000,  'python'),
    (45,  60,  500,   'numpy'),
    (60,  75,  1000,  'numpy'),
    (75,  90,  2000,  'numpy'),
    (90,  105, 5000,  'numpy'),
    (105, 120, 10000, 'numpy'),
]


# ================================================================
#  音效合成
# ================================================================

def synthesize_perf_audio(duration_sec: float = DURATION,
                           sample_rate: int = 44100) -> np.ndarray:
    total_samples = int(duration_sec * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float64)
    rng = np.random.default_rng(42)

    # 持续低频氛围
    t = np.arange(total_samples) / sample_rate
    hum = np.sin(2*math.pi*45*t) * 0.008 + np.sin(2*math.pi*70*t) * 0.004
    mod = 0.5 + 0.5 * np.sin(2*math.pi*0.08*t)
    audio[:] += hum * mod

    # 阶梯切换音效（短促提示音）
    for start, end, count, stype in PERF_STEPS:
        s = int(start * sample_rate)
        dur = min(int(0.15*sample_rate), total_samples - s)
        if dur <= 0: continue
        t_arr = np.arange(dur) / sample_rate
        freq = 800 if stype == 'numpy' else 500
        env = np.exp(-np.linspace(0, 10, dur))
        audio[s:s+dur] += np.sin(2*math.pi*freq*t_arr) * env * 0.15

    peak = max(np.abs(audio).max(), 1e-9)
    return (audio / peak * 0.8 * 32767).astype(np.int16)


def write_wav(path, samples, sr=44100):
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


# ================================================================
#  性能计量
# ================================================================

class PerfCounter:
    """实时 FPS 计量器"""
    def __init__(self):
        self.frame_times = []
        self.fps = 0.0

    def tick(self, elapsed_ms: float):
        self.frame_times.append(elapsed_ms)
        # 保留最近 30 帧计算平均 FPS
        if len(self.frame_times) > 30:
            self.frame_times = self.frame_times[-30:]
        avg_ms = sum(self.frame_times) / len(self.frame_times)
        self.fps = 1000.0 / avg_ms if avg_ms > 0 else 0


# ================================================================
#  主渲染流程
# ================================================================

def render_frames():
    os.makedirs(FRAMES_DIR, exist_ok=True)

    print(f"{'='*60}")
    print(f"  性能极限秀视频生成器 v1.0")
    print(f"  {WIDTH}x{HEIGHT} @ {FPS}fps × {DURATION}s")
    print(f"{'='*60}")

    dt = 1.0 / FPS
    perf = PerfCounter()
    current_step_idx = -1
    current_sys = None
    current_sys_type = 'python'
    current_count = 500
    current_label = ""

    # NumPy 渲染用 PILRenderer（兼容接口）
    renderer = PILRenderer(
        image=Image.new('RGB', (WIDTH, HEIGHT)),
        blend_mode='additive',
        glow_radius=2,
        glow_strength=0.3,
    )

    print(f"开始渲染 {TOTAL_FRAMES} 帧...")

    for frame_idx in range(TOTAL_FRAMES):
        t0_frame = time.time()
        t_sec = frame_idx / FPS

        # 确定当前阶梯
        step_idx = 0
        for i, (start, end, count, stype) in enumerate(PERF_STEPS):
            if start <= t_sec < end:
                step_idx = i
                break

        if step_idx != current_step_idx:
            current_step_idx = step_idx
            _, _, current_count, current_sys_type = PERF_STEPS[step_idx]
            current_label = f"{current_sys_type.upper()} | {current_count} particles"

            # 创建新系统
            if current_sys_type == 'python':
                current_sys = ParticleSystem(max_particles=current_count)
                em = Emitter(x=WIDTH//2, y=HEIGHT//2 - 50,
                            rate=current_count*2, spread=math.pi*2,
                            speed=5, life=5.0, size=3,
                            color=(200, 180, 255))
                current_sys.add_emitter(em)
                current_sys.add_force_field(ForceField(field_type='gravity', strength=3.0))
                # 预热
                current_sys.update(dt)

            elif current_sys_type == 'numpy':
                current_sys = ParticleSystemNumpy(max_particles=current_count)
                current_sys.add_force(NumpyForceGravity(strength=3.0))
                current_sys.add_force(NumpyForceDrag(factor=0.99))
                current_sys.add_force(NumpyForceTurbulence(strength=2.0, scale=40.0))
                current_sys.emit_radial(x=WIDTH//2, y=HEIGHT//2 - 50,
                                       count=current_count,
                                       speed_range=(2, 8),
                                       life_range=(3.0, 5.0),
                                       size_range=(2, 4),
                                       colors=[(200, 180, 255), (150, 200, 255), (100, 255, 200)])
            print(f"  [{t_sec:.1f}s] 阶梯切换 → {current_label}")

        # 更新
        if current_sys_type == 'python':
            current_sys.update(dt)
            particles = current_sys.particles
            p_count = current_sys.get_particle_count()
        else:
            current_sys.update(dt)
            # NumPy → 用 render_to_image
            p_count = current_sys.particle_count

        # 渲染
        elapsed_frame_ms = (time.time() - t0_frame) * 1000
        perf.tick(elapsed_frame_ms)

        bg = Image.new('RGB', (WIDTH, HEIGHT), (8, 6, 12))

        if current_sys_type == 'python':
            renderer.update_image(bg)
            renderer.render(particles)
            result = post_process_bloom(renderer.image, radius=6, strength=0.25, threshold=120)
        else:
            # NumPy 直接渲染
            np_img = current_sys.render_to_image(WIDTH, HEIGHT, blend_mode='additive')
            result = np_img

        # 性能信息叠加
        draw = ImageDraw.Draw(result)
        try:
            big_font = ImageFont.truetype("arial.ttf", 36)
            font = ImageFont.truetype("arial.ttf", 20)
            small_font = ImageFont.truetype("arial.ttf", 14)
        except:
            big_font = ImageFont.load_default()
            font = small_font = big_font

        # FPS 大字
        fps_color = (255, 80, 80) if perf.fps < 20 else (80, 255, 80) if perf.fps > 45 else (255, 220, 80)
        draw.rectangle([10, 10, 200, 55], fill=(0, 0, 0))
        draw.text((20, 15), f"FPS: {perf.fps:.1f}", fill=fps_color, font=big_font)

        # 系统信息
        draw.rectangle([10, 60, 300, 120], fill=(0, 0, 0))
        draw.text((20, 65), f"System: {current_sys_type.upper()}", fill=(220, 220, 240), font=font)
        draw.text((20, 90), f"Particles: {p_count} / {current_count}", fill=(180, 180, 200), font=font)

        # 性能对比提示
        if current_sys_type == 'numpy' and current_count >= 2000:
            draw.rectangle([WIDTH-300, 10, WIDTH-10, 50], fill=(0, 0, 0))
            draw.text((WIDTH-290, 15), "NumPy SoA Vectorized", fill=(100, 255, 150), font=font)

        elif current_sys_type == 'python' and current_count >= 1000:
            draw.rectangle([WIDTH-280, 10, WIDTH-10, 50], fill=(0, 0, 0))
            draw.text((WIDTH-270, 15), "Python OOP Loop", fill=(255, 150, 100), font=font)

        # 阶梯进度条
        progress = (t_sec % 15) / 15
        bar_x1 = 10 + int((WIDTH - 20) * progress)
        if bar_x1 > 10:
            draw.rectangle([10, HEIGHT-30, bar_x1, HEIGHT-20],
                           fill=(80, 200, 120))
        draw.rectangle([10, HEIGHT-30, WIDTH-10, HEIGHT-20],
                       outline=(60, 60, 70))

        frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_idx:05d}.jpg")
        result.save(frame_path, quality=92)

        if frame_idx % 30 == 0:
            pct = frame_idx / TOTAL_FRAMES * 100
            print(f"  [{pct:5.1f}%] 帧 {frame_idx}/{TOTAL_FRAMES} — "
                  f"{current_label} FPS:{perf.fps:.1f}")

    print(f"帧渲染完成！共 {TOTAL_FRAMES} 帧")


def encode_video():
    print("\n编码视频...")
    audio_samples = synthesize_perf_audio()
    audio_wav = os.path.join(TEMP_DIR, "perf_audio.wav")
    write_wav(audio_wav, audio_samples)

    pattern = os.path.join(FRAMES_DIR, "frame_%05d.jpg")
    video_tmp = os.path.join(TEMP_DIR, "perf_noaudio.mp4")
    video_out = os.path.join(OUTPUT_DIR, "performance_show.mp4")

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

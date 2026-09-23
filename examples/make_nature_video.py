"""
自然场景秀视频生成器 v1.0

展示 13 个工厂函数效果逐一串联：
  explosion / fire / rain / snow / magic_circle / fountain
  smoke / sparkle / trail / vortex / collision_sparks
  lighted_firework / magnet_field

每个效果展示 12-15s，带标签文字标注效果名称。
最后 20s 多效果叠加夜景。

运行（在仓库根目录下）：
    cd <仓库根>
    python examples/make_nature_video.py

依赖：
  ffmpeg, Pillow, numpy

音效：从环境变量 SOUNDS_DIR（或 SFX_ROOT）指向的音效库根目录加载；
      未设置时回落到 <仓库根>/sfx/sounds，缺文件会逐条提示但不中断。
"""

import sys, os, math, random, time, subprocess, tempfile, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 2026-09-28 去本地化：仓根改由本文件位置推导（原 D:/allin/tools/particle 已不存在）
PARTICLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 2026-09-28 去本地化：音效库根改由 环境变量 SOUNDS_DIR/SFX_ROOT → 仓内 sfx/sounds 推导
# （原 D:/allin/tools/sfx/sounds 已不存在）
SFX_ROOT = (os.environ.get("SFX_ROOT") or os.environ.get("SOUNDS_DIR")
            or os.path.join(PARTICLE_ROOT, "sfx", "sounds"))
sys.path.insert(0, PARTICLE_ROOT)

from core.particle import ParticleSystem, Emitter, ForceField, NoiseField
from components.effects import (
    create_explosion, create_fire, create_rain, create_snow,
    create_magic_circle, create_fountain, create_smoke, create_sparkle,
    create_trail, create_vortex_effect, create_collision_sparks,
    create_lighted_firework, create_magnet_field,
)
from renderers.pil_renderer import PILRenderer, LayeredPILRenderer, post_process_bloom
from examples.sfx_mixer import load_audio, mix_timeline, write_wav as sfx_write_wav

# ================================================================
#  视频参数
# ================================================================

WIDTH, HEIGHT = 1280, 720
FPS = 30
EFFECT_DURATION = 12  # 每个效果展示秒数
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(tempfile.gettempdir(), "_nature_frames")
FRAMES_DIR = TEMP_DIR

# 效果编排列表（名称, 工厂函数, 场景背景色, 参数）
EFFECTS_LIST = [
    ('Explosion',    create_explosion,      (15, 12, 20),  {'x': 640, 'y': 360, 'intensity': 1.5}),
    ('Fire',         create_fire,           (10, 8, 15),   {'x': 640, 'y': 550, 'width': 60, 'height': 100}),
    ('Rain',         create_rain,           (30, 35, 50),  {'width': WIDTH, 'height': HEIGHT, 'density': 1.5, 'wind': 2.0}),
    ('Snow',         create_snow,           (40, 45, 55),  {'width': WIDTH, 'height': HEIGHT, 'density': 1.0}),
    ('Magic Circle', create_magic_circle,   (8, 6, 15),   {'x': 640, 'y': 360, 'radius': 100}),
    ('Fountain',     create_fountain,       (10, 15, 25),  {'x': 640, 'y': 550, 'height': 150}),
    ('Smoke',        create_smoke,          (15, 15, 18),  {'x': 640, 'y': 450, 'direction': 'up', 'density': 1.5}),
    ('Sparkle',      create_sparkle,        (5, 3, 10),    {'x': 640, 'y': 360, 'area_size': 100, 'intensity': 2.0}),
    ('Trail',        create_trail,          (5, 3, 8),     {'x': 640, 'y': 360, 'direction_x': 1.0, 'direction_y': -0.5}),
    ('Vortex',       create_vortex_effect,  (8, 5, 15),   {'x': 640, 'y': 360, 'radius': 80, 'strength': 5.0}),
    ('Collision Sparks', create_collision_sparks, (12, 10, 18), {'x': 640, 'y': 360, 'width': 300, 'height': 200, 'intensity': 1.5}),
    ('Lighted Firework', create_lighted_firework, (5, 3, 10), {'x': 640, 'y': 360, 'intensity': 1.5}),
    ('Magnet Field', create_magnet_field,   (5, 3, 10),   {'x_center': 640, 'y_center': 360, 'attract_x': 500, 'attract_y': 360, 'repel_x': 780, 'repel_y': 360}),
]

# 综合叠加场景（最后20s）
FINALE_DURATION = 20


# ================================================================
#  音效合成
# ================================================================

def build_nature_audio(total_duration: float, sample_rate: int = 44100) -> np.ndarray:
    """用真实音效文件构建自然场景音轨"""
    print("  加载真实自然音效...")
    if not os.path.isdir(SFX_ROOT):
        print(f"    ! 音效库目录不存在：{SFX_ROOT}")
        print("      请设置环境变量 SOUNDS_DIR（或 SFX_ROOT）指向音效库根目录；"
              "缺失音效将静音处理。")

    sfx = {}
    load_map = {
        'explosion':  os.path.join(SFX_ROOT, 'smallexplosion1.mp3'),
        'explod':     os.path.join(SFX_ROOT, 'explod02.wav'),
        'campfire':   os.path.join(SFX_ROOT, 'campfire-1.mp3'),
        'fire':       os.path.join(SFX_ROOT, 'fire-1.mp3'),
        'rain':       os.path.join(SFX_ROOT, 'light-rain-01.mp3'),
        'blizzard':   os.path.join(SFX_ROOT, 'mixkit_blizzard_wind.wav'),
        'magic':      os.path.join(SFX_ROOT, 'magic-wand-01.mp3'),
        'splash':     os.path.join(SFX_ROOT, 'water-splash-1.mp3'),
        'river':      os.path.join(SFX_ROOT, 'river-2.mp3'),
        'wind':       os.path.join(SFX_ROOT, 'forest-wind-in-the-trees.mp3'),
        'birds':      os.path.join(SFX_ROOT, 'bird-01.mp3'),
        'whoosh':     os.path.join(SFX_ROOT, 'whoosh-02.mp3'),
        'firecracker': os.path.join(SFX_ROOT, 'firecracker-gun-4.mp3'),
        'magnet':     os.path.join(SFX_ROOT, 'magnet.mp3'),
    }
    for key, path in load_map.items():
        if os.path.exists(path):
            sfx[key] = load_audio(path, target_sr=sample_rate)
            print(f"    ✓ {key}")
        else:
            print(f"    ✗ {key}: 文件不存在")
            sfx[key] = np.zeros(0, dtype=np.float64)

    effect_starts = [i * EFFECT_DURATION for i in range(len(EFFECTS_LIST))]
    finale_start = len(EFFECTS_LIST) * EFFECT_DURATION

    events = []

    # 效果 → 音效映射
    effect_sfx = {
        'Explosion':        [('explosion', 0.0, 1.0), ('explod', 0.1, 0.8)],
        'Fire':             [('campfire', 0.0, 0.5)],
        'Rain':             [('rain', 0.0, 0.4)],
        'Snow':             [('blizzard', 0.0, 0.4)],
        'Magic Circle':     [('magic', 0.0, 0.6)],
        'Fountain':         [('splash', 2.0, 0.7), ('river', 0.0, 0.25)],
        'Smoke':            [('wind', 0.0, 0.15)],
        'Sparkle':          [('magic', 0.0, 0.5)],
        'Trail':            [('whoosh', 0.0, 0.4)],
        'Vortex':           [('whoosh', 0.0, 0.5)],
        'Collision Sparks': [('explosion', 0.0, 0.6)],
        'Lighted Firework': [('firecracker', 0.0, 0.7)],
        'Magnet Field':     [('magnet', 0.0, 0.4)],
    }

    for i, (name, _, _, _) in enumerate(EFFECTS_LIST):
        t_start = effect_starts[i]
        if name in effect_sfx:
            for sfx_key, offset, vol in effect_sfx[name]:
                t = t_start + offset
                if sfx_key in sfx and len(sfx[sfx_key]) > 0:
                    events.append((sfx[sfx_key], t, vol, 0.5, 1.0))

    # 环境底噪（鸟鸣 + 微风循环）
    for t_env in range(0, int(total_duration), 20):
        if 'birds' in sfx and len(sfx['birds']) > 0:
            events.append((sfx['birds'], t_env, 0.08, 2.0, 2.0))

    # 终场爆炸
    if 'explosion' in sfx and len(sfx['explosion']) > 0:
        events.append((sfx['explosion'], finale_start + 5, 1.2, 0.0, 1.0))

    print(f"  混合 {len(events)} 个音效事件...")
    mixed = mix_timeline(events, total_duration, sample_rate)
    return mixed


def synthesize_nature_audio(total_duration, sample_rate=44100):
    """[已废弃]"""
    return build_nature_audio(total_duration, sample_rate)


# ================================================================
#  主渲染流程
# ================================================================

def render_frames():
    os.makedirs(FRAMES_DIR, exist_ok=True)

    total_duration = len(EFFECTS_LIST) * EFFECT_DURATION + FINALE_DURATION
    total_frames = int(FPS * total_duration)

    print(f"{'='*60}")
    print(f"  自然场景秀视频生成器 v1.0")
    print(f"  {WIDTH}x{HEIGHT} @ {FPS}fps × {total_duration}s")
    print(f"  {len(EFFECTS_LIST)} 个效果 × {EFFECT_DURATION}s + {FINALE_DURATION}s 综合叠加")
    print(f"{'='*60}")

    renderer = PILRenderer(
        image=Image.new('RGB', (WIDTH, HEIGHT)),
        blend_mode='additive',
        glow_radius=3,
        glow_strength=0.4,
    )

    current_system = None
    current_name = ""
    current_bg = (15, 12, 20)
    dt = 1.0 / FPS

    print(f"开始渲染 {total_frames} 帧...")

    for frame_idx in range(total_frames):
        t_sec = frame_idx / FPS

        # 确定当前效果
        effect_idx = int(t_sec / EFFECT_DURATION)
        if effect_idx < len(EFFECTS_LIST):
            name, factory, bg_color, kwargs = EFFECTS_LIST[effect_idx]

            # 效果切换
            if current_name != name:
                current_system = factory(**kwargs)
                current_name = name
                current_bg = bg_color
                print(f"  [{t_sec:.1f}s] 效果切换 → {name}")

            # 更新粒子系统
            current_system.update(dt)

            # 渲染
            bg = Image.new('RGB', (WIDTH, HEIGHT), current_bg)
            renderer.update_image(bg)
            renderer.render(current_system.particles if hasattr(current_system, 'particles') else [])
            result = post_process_bloom(renderer.image, radius=8, strength=0.35, threshold=110)

        else:
            # 综合叠加段：多效果同时展示
            if current_name != "Finale":
                # 创建多个效果叠加
                fire_sys = create_fire(x=300, y=550)
                smoke_sys = create_smoke(x=300, y=450)
                sparkle_sys = create_sparkle(x=900, y=300, intensity=2.0)
                fountain_sys = create_fountain(x=900, y=550, height=120)
                current_name = "Finale"
                current_bg = (10, 8, 15)
                # 预热
                for _ in range(10):
                    fire_sys.update(dt); smoke_sys.update(dt)
                    sparkle_sys.update(dt); fountain_sys.update(dt)

            fire_sys.update(dt); smoke_sys.update(dt)
            sparkle_sys.update(dt); fountain_sys.update(dt)

            bg = Image.new('RGB', (WIDTH, HEIGHT), current_bg)
            renderer.update_image(bg)
            # 合并所有粒子
            all_p = (fire_sys.particles + smoke_sys.particles +
                     sparkle_sys.particles + fountain_sys.particles)
            renderer.render(all_p)
            result = post_process_bloom(renderer.image, radius=12, strength=0.5, threshold=100)
            name = "All Effects Combined"

        # 添加效果名称标签
        draw = ImageDraw.Draw(result)
        try:
            font = ImageFont.truetype("arial.ttf", 22)
        except:
            font = ImageFont.load_default()

        draw.rectangle([30, HEIGHT-55, 300, HEIGHT-25], fill=(0, 0, 0))
        draw.text((40, HEIGHT-50), name, fill=(220, 220, 240), font=font)

        p_count = current_system.get_particle_count() if current_system else 0
        draw.text((WIDTH-160, HEIGHT-50), f"Particles: {p_count}", fill=(160, 160, 180))

        # 保存帧
        frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_idx:05d}.jpg")
        result.save(frame_path, quality=92)

        if frame_idx % 60 == 0:
            pct = frame_idx / total_frames * 100
            print(f"  [{pct:5.1f}%] 帧 {frame_idx}/{total_frames} — {name}")

    print(f"帧渲染完成！共 {total_frames} 帧")
    return total_frames


def encode_video(total_frames):
    print("\n编码视频...")
    total_duration = len(EFFECTS_LIST) * EFFECT_DURATION + FINALE_DURATION
    audio = build_nature_audio(total_duration)
    audio_wav = os.path.join(TEMP_DIR, "nature_audio.wav")
    sfx_write_wav(audio_wav, audio)

    pattern = os.path.join(FRAMES_DIR, "frame_%05d.jpg")
    video_tmp = os.path.join(TEMP_DIR, "nature_noaudio.mp4")
    video_out = os.path.join(OUTPUT_DIR, "nature_show.mp4")

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
    total_frames = render_frames()
    output = encode_video(total_frames)
    cleanup()
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  完成! 总耗时: {elapsed:.1f}s")
    print(f"  输出: {output}")
    print(f"{'='*60}")

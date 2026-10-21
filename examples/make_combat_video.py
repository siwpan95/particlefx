"""
战斗特效秀视频生成器 v1.0

展示 CombatEffectsManager：
  5种攻击(slash/impact/pierce/blunt/crit)
  + 6种技能(fireball/ice_storm/lightning/heal/shield/aoe_circle)
  + 飘浮伤害数字 + 死亡消散 + 升级光环

编排：模拟一段 RPG 战斗场景序列

运行（在仓库根目录下）：
    cd <仓库根>
    python examples/make_combat_video.py

依赖：
  ffmpeg, Pillow, numpy

音效：从环境变量 SOUNDS_DIR（或 SFX_ROOT）指向的音效库根目录加载；
      未设置时回落到 <仓库根>/sfx/sounds，缺文件会逐条提示但不中断。
"""

import sys, os, math, random, time, subprocess, tempfile, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 2026-09-23 去本地化：仓根改由本文件位置推导（原路径已不存在）
PARTICLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 2026-09-23 去本地化：音效库根改由 环境变量 SOUNDS_DIR/SFX_ROOT → 仓内 sfx/sounds 推导
# （原路径已不存在）
SFX_ROOT = (os.environ.get("SFX_ROOT") or os.environ.get("SOUNDS_DIR")
            or os.path.join(PARTICLE_ROOT, "sfx", "sounds"))
sys.path.insert(0, PARTICLE_ROOT)

from particlefx.systems.combat_effects import CombatEffectsManager
from particlefx.core.particle import Particle, ForceField
from particlefx.renderers.pil_renderer import PILRenderer, post_process_bloom
from examples.sfx_mixer import load_audio, mix_timeline, write_wav as sfx_write_wav

# ================================================================
#  视频参数
# ================================================================

WIDTH, HEIGHT = 1280, 720
FPS = 30
DURATION = 120  # 2分钟
TOTAL_FRAMES = FPS * DURATION
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(tempfile.gettempdir(), "_combat_frames")
FRAMES_DIR = TEMP_DIR

# 战斗编排（帧索引 → 触发事件）
# 格式: (frame, event_type, x, y, **kwargs)
COMBAT_SCHEDULE = [
    # 攻击阶段 0-30s
    (30,   'hit', 500, 360, {'effect_type': 'slash', 'direction': 0.3}),
    (90,   'damage', 500, 360, {'value': 128, 'is_crit': False}),
    (150,  'hit', 480, 340, {'effect_type': 'impact'}),
    (210,  'damage', 480, 340, {'value': 256}),
    (300,  'hit', 520, 350, {'effect_type': 'pierce', 'direction': -0.5}),
    (360,  'damage', 520, 350, {'value': 512}),
    (450,  'hit', 500, 360, {'effect_type': 'blunt'}),
    (510,  'damage', 500, 360, {'value': 384}),
    # 暴击 30s处
    (600,  'hit', 500, 350, {'effect_type': 'crit'}),
    (620,  'damage', 500, 350, {'value': 9999, 'is_crit': True}),
    # 技能阶段 25-70s
    (750,  'skill', 500, 360, {'skill_type': 'fireball'}),
    (770,  'damage', 500, 360, {'value': 3000, 'color': (255, 120, 20)}),
    (900,  'skill', 500, 360, {'skill_type': 'ice_storm'}),
    (920,  'damage', 500, 360, {'value': 2500, 'color': (100, 180, 255)}),
    (1050, 'skill', 500, 360, {'skill_type': 'lightning'}),
    (1070, 'damage', 500, 360, {'value': 4000, 'color': (180, 180, 255)}),
    (1200, 'skill', 500, 360, {'skill_type': 'heal'}),
    (1220, 'damage', 500, 360, {'value': 5000, 'is_heal': True}),
    (1350, 'skill', 500, 360, {'skill_type': 'shield'}),
    (1500, 'skill', 500, 360, {'skill_type': 'aoe_circle', 'duration': 1.0}),
    (1520, 'damage', 500, 360, {'value': 8000, 'color': (255, 100, 100)}),
    # 结局 70-120s
    (2100, 'death', 500, 360, {}),
    (2400, 'level_up', 500, 360, {}),
]

EVENT_LABELS = {
    'hit': {'slash': 'Slash', 'impact': 'Impact', 'pierce': 'Pierce',
            'blunt': 'Blunt', 'crit': 'CRITICAL HIT'},
    'skill': {'fireball': 'Fireball', 'ice_storm': 'Ice Storm',
              'lightning': 'Lightning', 'heal': 'Heal',
              'shield': 'Shield', 'aoe_circle': 'AOE Circle'},
    'damage': 'Damage Number',
    'death': 'Death Fadeout',
    'level_up': 'Level Up!',
}


# ================================================================
#  音效合成
# ================================================================

def build_combat_audio(duration_sec: float = DURATION,
                       sample_rate: int = 44100) -> np.ndarray:
    """用真实音效文件构建战斗音轨"""
    print("  加载真实战斗音效...")
    if not os.path.isdir(SFX_ROOT):
        print(f"    ! 音效库目录不存在：{SFX_ROOT}")
        print("      请设置环境变量 SOUNDS_DIR（或 SFX_ROOT）指向音效库根目录；"
              "缺失音效将静音处理。")

    sfx = {}
    load_map = {
        'katana':    os.path.join(SFX_ROOT, 'katana1.mp3'),
        'explosion': os.path.join(SFX_ROOT, 'smallexplosion1.mp3'),
        'explod':    os.path.join(SFX_ROOT, 'explod02.wav'),
        'magic':     os.path.join(SFX_ROOT, 'magic-wand-01.mp3'),
        'whoosh':    os.path.join(SFX_ROOT, 'whoosh-02.mp3'),
        'death':     os.path.join(SFX_ROOT,
            'sonic_death_sound_effect_sound_effect_mp3_download_free_-_quick_sounds.mp3'),
        'bell':      os.path.join(SFX_ROOT, 'bell01.wav'),
        'heal':      os.path.join(SFX_ROOT, 'dream-harp-02.mp3'),
        'fire':      os.path.join(SFX_ROOT, 'firecracker-gun-4.mp3'),
        'thunder':   os.path.join(SFX_ROOT, 'thunder7.wav'),
        'shield':    os.path.join(SFX_ROOT, 'a-shite-in-shining-armor.mp3'),
    }
    for key, path in load_map.items():
        if os.path.exists(path):
            sfx[key] = load_audio(path, target_sr=sample_rate)
            print(f"    ✓ {key}")
        else:
            print(f"    ✗ {key}: 文件不存在")
            sfx[key] = np.zeros(0, dtype=np.float64)

    events = []

    # 音效映射：event → 使用的音效文件
    def add_sfx(key, t_sec, vol=1.0):
        if key in sfx and len(sfx[key]) > 0:
            events.append((sfx[key], t_sec, vol, 0.0, 0.15))

    # 按战斗编排添加音效
    for frame, event, x, y, kwargs in COMBAT_SCHEDULE:
        t = frame / FPS

        if event == 'hit':
            et = kwargs.get('effect_type', 'slash')
            sfx_map = {
                'slash':  ('katana', 1.0),
                'impact': ('explosion', 0.8),
                'pierce': ('whoosh', 0.7),
                'blunt':  ('explod', 0.8),
                'crit':   ('katana', 1.5),
            }
            if et in sfx_map:
                k, v = sfx_map[et]
                add_sfx(k, t, v)
            if et == 'crit':
                add_sfx('explosion', t + 0.1, 1.0)

        elif event == 'skill':
            st = kwargs.get('skill_type', 'fireball')
            sfx_map = {
                'fireball':   ('fire', 0.9),
                'ice_storm':  ('magic', 0.7),
                'lightning':  ('thunder', 0.8),
                'heal':       ('heal', 0.6),
                'shield':     ('shield', 0.5),
                'aoe_circle': ('explosion', 1.2),
            }
            if st in sfx_map:
                k, v = sfx_map[st]
                add_sfx(k, t, v)

        elif event == 'death':
            add_sfx('death', t, 0.8)

        elif event == 'level_up':
            add_sfx('bell', t, 0.8)
            # 多加几个 bell 产生上行音阶效果
            add_sfx('bell', t + 0.25, 0.6)
            add_sfx('bell', t + 0.5, 0.4)
            add_sfx('bell', t + 0.7, 0.35)

    print(f"  混合 {len(events)} 个音效事件...")
    mixed = mix_timeline(events, duration_sec, sample_rate)
    return mixed


def synthesize_combat_audio(duration_sec=DURATION, sample_rate=44100):
    """[已废弃]"""
    return build_combat_audio(duration_sec, sample_rate)


# ================================================================
#  场景绘制
# ================================================================

def draw_battle_scene(img):
    """绘制战斗场景背景"""
    draw = ImageDraw.Draw(img)
    # 暗色战场地面
    for y in range(HEIGHT):
        t = y / HEIGHT
        r = int(20 + 15 * t)
        g = int(15 + 10 * t)
        b = int(25 + 20 * t)
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))

    # 地面纹理线
    ground_y = HEIGHT - 80
    draw.line([(0, ground_y), (WIDTH, ground_y)], fill=(40, 35, 45), width=2)
    for x in range(0, WIDTH, 40):
        draw.line([(x, ground_y), (x+20, ground_y+5)], fill=(35, 30, 40))

    # 远处山脉剪影
    mountains = [(0, 180), (80, 100), (200, 140), (350, 80),
                 (500, 120), (650, 90), (800, 110), (950, 70),
                 (1100, 100), (1280, 150)]
    for i in range(len(mountains)-1):
        x1, y1 = mountains[i]
        x2, y2 = mountains[i+1]
        draw.polygon([(x1, ground_y-120+y1), (x2, ground_y-120+y2),
                      (x2, ground_y), (x1, ground_y)],
                     fill=(18, 18, 22))

    # 角色位置标记（地面圆圈）
    cx, cy = 500, ground_y - 30
    draw.ellipse([cx-25, cy-8, cx+25, cy+8], fill=(35, 30, 40), outline=(60, 55, 65))

    return img


def draw_floating_texts(img, combat_mgr):
    """绘制飘浮伤害数字"""
    draw = ImageDraw.Draw(img)
    for ft in combat_mgr.floating_text_data:
        text = ft['text']
        x = int(ft['x'])
        y = int(ft['y'])
        color = ft['color']
        alpha = ft['alpha']
        scale = ft['scale']
        is_crit = ft['is_crit']

        # 字号：暴击更大
        font_size = int(24 * scale * alpha)
        if font_size < 8:
            continue

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (IOError, OSError):
            try:
                font = ImageFont.load_default(size=font_size)
            except Exception:
                font = ImageFont.load_default()

        # 阴影
        shadow_color = (0, 0, 0)
        draw.text((x+2, y+2), text, fill=shadow_color, font=font)

        # 主文字
        r = int(color[0] * alpha)
        g = int(color[1] * alpha)
        b = int(color[2] * alpha)
        draw.text((x, y), text, fill=(r, g, b), font=font)

        # 暴击额外描边
        if is_crit and alpha > 0.5:
            draw.text((x-1, y), text, fill=(255, 255, 200), font=font)


# ================================================================
#  主渲染流程
# ================================================================

def render_frames():
    os.makedirs(FRAMES_DIR, exist_ok=True)

    print(f"{'='*60}")
    print(f"  战斗特效秀视频生成器 v1.0")
    print(f"  {WIDTH}x{HEIGHT} @ {FPS}fps × {DURATION}s")
    print(f"{'='*60}")

    combat_mgr = CombatEffectsManager()
    renderer = PILRenderer(
        image=Image.new('RGB', (WIDTH, HEIGHT)),
        blend_mode='additive',
        glow_radius=3,
        glow_strength=0.4,
    )

    dt = 1.0 / FPS
    current_label = "Ready..."
    label_timer = 0.0

    print(f"开始渲染 {TOTAL_FRAMES} 帧...")

    for frame_idx in range(TOTAL_FRAMES):
        t_sec = frame_idx / FPS

        # 触发编排事件
        for sched_frame, event, x, y, kwargs in COMBAT_SCHEDULE:
            if frame_idx == sched_frame:
                if event == 'hit':
                    combat_mgr.hit_effect(x, y, **kwargs)
                    et = kwargs.get('effect_type', 'slash')
                    current_label = EVENT_LABELS['hit'].get(et, et)
                    label_timer = 2.0
                    print(f"  [{t_sec:.1f}s] {event}: {et}")
                elif event == 'skill':
                    combat_mgr.skill_cast(x, y, **kwargs)
                    st = kwargs.get('skill_type', 'fireball')
                    current_label = EVENT_LABELS['skill'].get(st, st)
                    label_timer = 2.5
                    print(f"  [{t_sec:.1f}s] {event}: {st}")
                elif event == 'damage':
                    combat_mgr.floating_damage(x, y, **kwargs)
                    val = kwargs.get('value', 0)
                    current_label = f"-{val}" if not kwargs.get('is_heal') else f"+{val}"
                    label_timer = 1.5
                elif event == 'death':
                    combat_mgr.death_fadeout(x, y)
                    current_label = "Death Fadeout"
                    label_timer = 3.0
                    print(f"  [{t_sec:.1f}s] death_fadeout")
                elif event == 'level_up':
                    combat_mgr.level_up(x, y)
                    current_label = "Level Up!"
                    label_timer = 3.0
                    print(f"  [{t_sec:.1f}s] level_up")

        # 更新
        combat_mgr.update(dt)
        label_timer -= dt

        # 渲染背景
        bg = Image.new('RGB', (WIDTH, HEIGHT), (25, 20, 30))
        bg = draw_battle_scene(bg)

        # 渲染粒子
        renderer.update_image(bg.copy())
        renderer.render(combat_mgr.particles)

        # Bloom
        result = post_process_bloom(renderer.image, radius=10, strength=0.4, threshold=100)

        # 飘浮文字
        draw_floating_texts(result, combat_mgr)

        # 标签
        draw = ImageDraw.Draw(result)
        if label_timer > 0:
            alpha = min(1.0, label_timer / 0.5)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            draw.rectangle([30, 30, 250, 60], fill=(0, 0, 0))
            draw.text((40, 35), current_label, fill=(220, 220, 240), font=font)

        # 粒子数
        draw.text((WIDTH-150, 30),
                  f"Particles: {combat_mgr.particle_count}",
                  fill=(150, 150, 170))

        # 保存帧
        frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_idx:05d}.jpg")
        result.save(frame_path, quality=92)

        if frame_idx % 60 == 0:
            pct = frame_idx / TOTAL_FRAMES * 100
            print(f"  [{pct:5.1f}%] 帧 {frame_idx}/{TOTAL_FRAMES} "
                  f"— 粒子:{combat_mgr.particle_count}")

    print(f"帧渲染完成！共 {TOTAL_FRAMES} 帧")


def encode_video():
    print("\n编码视频...")
    audio = build_combat_audio()
    audio_wav = os.path.join(TEMP_DIR, "combat_audio.wav")
    sfx_write_wav(audio_wav, audio)

    pattern = os.path.join(FRAMES_DIR, "frame_%05d.jpg")
    video_tmp = os.path.join(TEMP_DIR, "combat_noaudio.mp4")
    video_out = os.path.join(OUTPUT_DIR, "combat_show.mp4")

    cmd = ["ffmpeg", "-y", "-framerate", str(FPS), "-i", pattern,
           "-vcodec", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", video_tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg err:", r.stderr[-1500:])
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

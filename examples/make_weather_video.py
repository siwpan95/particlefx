"""
天气模拟秀视频生成器 v1.0

展示 WeatherSystem 6种天气渐变切换：
  clear → rain → heavy_rain → storm → snow → heavy_snow → clear

特性：
  - 天气切换过渡动画
  - 暴风雨闪电闪烁
  - 风向变化 + 雪花飘动
  - 动态天空背景（日→夜→阴→晴）
  - 音效混合（雨声/雷声/风声）

运行（在仓库根目录下）：
    cd <仓库根>
    python examples/make_weather_video.py

依赖：
  ffmpeg, Pillow, numpy

音效：从环境变量 SOUNDS_DIR（或 SFX_ROOT）指向的音效库根目录加载；
      未设置时回落到 <仓库根>/sfx/sounds，缺文件会逐条提示但不中断。
"""

import sys, os, math, random, time, subprocess, tempfile, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# 2026-09-23 去本地化：仓根改由本文件位置推导（原 D:/allin/tools/particle 已不存在）
PARTICLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 2026-09-23 去本地化：音效库根改由 环境变量 SOUNDS_DIR/SFX_ROOT → 仓内 sfx/sounds 推导
# （原 D:/allin/tools/sfx/sounds 已不存在）
SFX_ROOT = (os.environ.get("SFX_ROOT") or os.environ.get("SOUNDS_DIR")
            or os.path.join(PARTICLE_ROOT, "sfx", "sounds"))
sys.path.insert(0, PARTICLE_ROOT)

from systems.weather_system import WeatherSystem
from core.particle import Particle, ForceField
from renderers.pil_renderer import PILRenderer, LayeredPILRenderer, post_process_bloom
from examples.sfx_mixer import load_audio, mix_timeline, write_wav as sfx_write_wav

# ================================================================
#  视频参数
# ================================================================

WIDTH, HEIGHT = 1280, 720
FPS = 30
DURATION = 150  # 2.5分钟
TOTAL_FRAMES = FPS * DURATION
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(tempfile.gettempdir(), "_weather_frames")
FRAMES_DIR = TEMP_DIR

# 天气切换编排（秒 → 天气类型）
WEATHER_SCHEDULE = [
    (0,   'clear'),       # 0-5s: 晴空
    (5,   'rain'),        # 5-25s: 小雨渐入
    (25,  'heavy_rain'),  # 25-55s: 暴雨
    (55,  'storm'),       # 55-85s: 风暴+闪电
    (85,  'snow'),        # 85-110s: 雪花飘落
    (110, 'heavy_snow'),  # 110-130s: 大雪纷飞
    (130, 'clear'),       # 130-150s: 回归晴空
]

WEATHER_LABELS = {
    'clear':       'Clear Sky',
    'rain':        'Light Rain',
    'heavy_rain':  'Heavy Rain',
    'storm':       'Thunderstorm',
    'snow':        'Snowfall',
    'heavy_snow':  'Heavy Snow',
}

WEATHER_BG_COLORS = {
    'clear':       ((135, 206, 235), (70, 130, 180)),      # 天蓝渐变
    'rain':        ((80, 100, 130),  (40, 60, 80)),         # 雨天灰蓝
    'heavy_rain':  ((50, 60, 80),   (25, 35, 50)),          # 暴雨暗灰
    'storm':       ((30, 30, 50),   (15, 15, 25)),          # 风暴深黑
    'snow':        ((180, 195, 215), (100, 110, 130)),      # 雪天亮灰
    'heavy_snow':  ((200, 210, 230), (150, 160, 180)),      # 大雪白灰
}


# ================================================================
#  音效合成
# ================================================================

def build_weather_audio(duration_sec: float = DURATION,
                        sample_rate: int = 44100) -> np.ndarray:
    """用真实音效文件构建天气音轨（替代合成音频）"""
    print("  加载真实音效文件...")
    if not os.path.isdir(SFX_ROOT):
        print(f"    ! 音效库目录不存在：{SFX_ROOT}")
        print("      请设置环境变量 SOUNDS_DIR（或 SFX_ROOT）指向音效库根目录；"
              "缺失音效将静音处理。")

    # 加载所有需要的音效
    sfx = {}
    load_map = {
        'rain':      os.path.join(SFX_ROOT, 'light-rain-01.mp3'),
        'storm':     os.path.join(SFX_ROOT, 'storm.mp3'),
        'thunder':   os.path.join(SFX_ROOT, 'thunder7.wav'),
        'wind':      os.path.join(SFX_ROOT, 'forest-wind-in-the-trees.mp3'),
        'blizzard':  os.path.join(SFX_ROOT, 'mixkit_blizzard_wind.wav'),
        'birds':     os.path.join(SFX_ROOT, 'bird-01.mp3'),
    }
    for key, path in load_map.items():
        if os.path.exists(path):
            sfx[key] = load_audio(path, target_sr=sample_rate)
            print(f"    ✓ {key}: {path.split(chr(92))[-1]}")
        else:
            print(f"    ✗ {key}: 文件不存在 ({path})")
            sfx[key] = np.zeros(0, dtype=np.float64)

    # 构建时间轴声音事件列表: (audio, start_sec, volume, fade_in, fade_out)
    events = []

    # --- 晴空段: 鸟鸣 ---
    if len(sfx.get('birds', [])) > 0:
        events.append((sfx['birds'], 0.0, 0.25, 1.0, 2.0))
        events.append((sfx['birds'], 135.0, 0.25, 2.0, 1.0))

    # --- 雨声 5-85s (从小雨→暴雨→风暴) ---
    if len(sfx.get('rain', [])) > 0:
        # 循环使用雨声文件（如果是短文件则重复）
        # light-rain-01.mp3 如果不够长，下面用 mix_timeline 自动循环处理
        events.append((sfx['rain'], 5.0, 0.35, 3.0, 0.0))
        events.append((sfx['rain'], 25.0, 0.55, 0.0, 0.0))
        events.append((sfx['rain'], 55.0, 0.65, 0.0, 2.0))

    # --- 风声 5-85s ---
    if len(sfx.get('wind', [])) > 0:
        events.append((sfx['wind'], 5.0, 0.25, 3.0, 0.0))
        events.append((sfx['wind'], 55.0, 0.45, 0.0, 2.0))

    # --- 风暴 55-85s (30s, 源文件只有2.7s需要循环) ---
    if len(sfx.get('storm', [])) > 0:
        events.append((sfx['storm'], 55.0, 0.55, 2.0, 3.0, 30.0))

    # --- 雷声 (在风暴段中穿插) ---
    thunder_times = [60, 66, 72, 77, 80]
    for t in thunder_times:
        if len(sfx.get('thunder', [])) > 0:
            events.append((sfx['thunder'], t, 1.2, 0.0, 0.3))

    # --- 雪声 85-130s (45s, 源文件4.5s需要循环) ---
    if len(sfx.get('blizzard', [])) > 0:
        events.append((sfx['blizzard'], 85.0, 0.5, 2.0, 0.0, 45.0))
        events.append((sfx['blizzard'], 110.0, 0.6, 0.0, 3.0, 20.0))
    # 雪天微风
    if len(sfx.get('wind', [])) > 0:
        events.append((sfx['wind'], 85.0, 0.15, 3.0, 2.0))

    print(f"  混合 {len(events)} 个音效层...")
    mixed = mix_timeline(events, duration_sec, sample_rate)

    # 扩展：对短音效做自动循环
    # (已在 _loop_audio 中处理，这里直接返回)
    return mixed


# 废弃旧函数别名保留兼容（实际不再使用合成）
def synthesize_weather_audio(duration_sec=DURATION, sample_rate=44100):
    """[已废弃] 使用 build_weather_audio 替代"""
    return build_weather_audio(duration_sec, sample_rate)


# ================================================================
#  动态天空背景
# ================================================================

class DynamicSkyWeather:
    """天气场景动态天空背景"""

    def __init__(self, width, height):
        self.w = width
        self.h = height
        # 星星
        self.stars = []
        rng = random.Random(42)
        for _ in range(200):
            self.stars.append((
                rng.randint(0, width),
                rng.randint(0, int(height*0.6)),
                rng.randint(80, 255),
                rng.uniform(0, 2*math.pi),
                rng.uniform(0.5, 2.0),
            ))

    def render(self, current_weather, frame_idx, lightning_active=False):
        """渲染背景天空"""
        top_color, bot_color = WEATHER_BG_COLORS.get(
            current_weather, ((100, 150, 200), (50, 80, 100)))

        img = Image.new('RGB', (self.w, self.h))
        draw = ImageDraw.Draw(img)

        # 渐变天空
        for y in range(self.h):
            t = y / self.h
            r = int(top_color[0] + (bot_color[0] - top_color[0]) * t)
            g = int(top_color[1] + (bot_color[1] - top_color[1]) * t)
            b = int(top_color[2] + (bot_color[2] - top_color[2]) * t)
            draw.line([(0, y), (self.w, y)], fill=(r, g, b))

        # 晴天/雪天显示星星
        if current_weather in ('clear', 'snow', 'heavy_snow'):
            for sx, sy, br, ph, sp in self.stars:
                flicker = 0.6 + 0.4 * math.sin(ph + frame_idx * 0.03 * sp)
                a = max(0.2, min(1.0, flicker))
                c = int(br * a)
                sz = max(1, int(1.5 * a))
                if br > 180:
                    draw.ellipse([sx-sz, sy-sz, sx+sz, sy+sz], fill=(c, c, min(255, c+20)))
                else:
                    draw.point((sx, sy), fill=(c, c, c))

        # 闪电全屏白闪
        if lightning_active:
            flash = Image.new('RGB', (self.w, self.h), (240, 240, 255))
            img = Image.blend(img, flash, 0.35)

        return img


# ================================================================
#  地面剪影
# ================================================================

def draw_ground_silhouette(img, weather_type):
    """绘制地面建筑/树木剪影"""
    draw = ImageDraw.Draw(img)
    h = img.height
    w = img.width
    ground_y = h - 40

    # 地面
    draw.rectangle([0, ground_y, w, h], fill=(20, 25, 30))

    # 建筑剪影
    rng = random.Random(7)
    buildings = [(50, 80), (180, 120), (320, 60), (500, 100),
                 (680, 70), (820, 90), (1000, 110), (1150, 50)]
    for bx, bh in buildings:
        bw = rng.randint(60, 100)
        top_y = ground_y - bh
        draw.rectangle([bx, top_y, bx+bw, ground_y], fill=(15, 18, 22))
        # 窗户灯光（晴天/雪天才有）
        if weather_type in ('clear', 'snow', 'heavy_snow'):
            for wy in range(top_y + 10, ground_y - 10, 15):
                for wx in range(bx + 8, bx + bw - 8, 12):
                    if rng.random() < 0.6:
                        draw.rectangle([wx, wy, wx+6, wy+4], fill=(200, 180, 120))

    # 树木
    trees = [(100, 50), (400, 40), (700, 55), (950, 35), (1200, 45)]
    for tx, th in trees:
        # 树干
        draw.rectangle([tx-3, ground_y-th, tx+3, ground_y], fill=(25, 30, 20))
        # 树冠
        draw.ellipse([tx-th//2, ground_y-th-15, tx+th//2, ground_y-th+5],
                     fill=(18, 22, 16))

    return img


# ================================================================
#  主渲染流程
# ================================================================

def get_weather_at_time(t_sec):
    """根据时间获取当前天气类型"""
    current = 'clear'
    for sched_t, weather in WEATHER_SCHEDULE:
        if t_sec >= sched_t:
            current = weather
    return current


def render_frames():
    os.makedirs(FRAMES_DIR, exist_ok=True)

    print(f"{'='*60}")
    print(f"  天气模拟秀视频生成器 v1.0")
    print(f"  {WIDTH}x{HEIGHT} @ {FPS}fps × {DURATION}s")
    print(f"{'='*60}")

    # 初始化
    weather_sys = WeatherSystem(width=WIDTH, height=HEIGHT, initial_weather='clear')
    sky = DynamicSkyWeather(WIDTH, HEIGHT)

    renderer = PILRenderer(
        image=Image.new('RGB', (WIDTH, HEIGHT)),
        blend_mode='additive',
        glow_radius=2,
        glow_strength=0.3,
    )

    current_weather = 'clear'
    dt = 1.0 / FPS

    print(f"开始渲染 {TOTAL_FRAMES} 帧...")

    for frame_idx in range(TOTAL_FRAMES):
        t_sec = frame_idx / FPS

        # 天气切换
        target_weather = get_weather_at_time(t_sec)
        if target_weather != current_weather:
            weather_sys.set_weather(target_weather)
            current_weather = target_weather
            print(f"  [{t_sec:.1f}s] 天气切换 → {target_weather}")

        # 更新天气系统
        weather_sys.update(dt)

        # 渲染背景天空
        bg = sky.render(current_weather, frame_idx, weather_sys.lightning_active)

        # 渲染地面剪影
        bg = draw_ground_silhouette(bg, current_weather)

        # 渲染粒子
        renderer.update_image(bg.copy())
        renderer.render(weather_sys.particles)

        # Bloom（风暴/大雨/大雪时加强）
        if current_weather in ('storm', 'heavy_rain', 'heavy_snow'):
            result = post_process_bloom(renderer.image, radius=8, strength=0.3, threshold=120)
        else:
            result = renderer.image

        # 添加天气标签文字
        draw = ImageDraw.Draw(result)
        label = WEATHER_LABELS.get(current_weather, current_weather)
        # 半透明标签背景
        draw.rectangle([30, HEIGHT-65, 250, HEIGHT-30], fill=(0, 0, 0, 128))
        try:
            draw.text((40, HEIGHT-60), label, fill=(220, 220, 240))
        except Exception:
            draw.text((40, HEIGHT-60), label, fill=(220, 220, 240))

        # 粒子数信息
        draw.text((WIDTH-140, HEIGHT-60),
                  f"Particles: {weather_sys.particle_count}",
                  fill=(180, 180, 200))

        # 保存帧
        frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_idx:05d}.jpg")
        result.save(frame_path, quality=92)

        if frame_idx % 60 == 0:
            pct = frame_idx / TOTAL_FRAMES * 100
            print(f"  [{pct:5.1f}%] 帧 {frame_idx}/{TOTAL_FRAMES} "
                  f"— {current_weather}, 粒子:{weather_sys.particle_count}")

    print(f"帧渲染完成！共 {TOTAL_FRAMES} 帧")


def encode_video():
    """ffmpeg 编码"""
    print("\n编码视频...")

    # 用真实音效构建音轨
    audio = build_weather_audio()
    audio_wav = os.path.join(TEMP_DIR, "weather_audio.wav")
    sfx_write_wav(audio_wav, audio)

    pattern = os.path.join(FRAMES_DIR, "frame_%05d.jpg")
    video_tmp = os.path.join(TEMP_DIR, "weather_noaudio.mp4")
    video_out = os.path.join(OUTPUT_DIR, "weather_show.mp4")

    # 编码无音频视频
    cmd = ["ffmpeg", "-y", "-framerate", str(FPS), "-i", pattern,
           "-vcodec", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", video_tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg err:", r.stderr[-1500:])
        raise RuntimeError("视频编码失败")
    print(f"  视频编码完成: {video_tmp}")

    # 合并音频
    cmd = ["ffmpeg", "-y", "-i", video_tmp, "-i", audio_wav,
           "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
           "-shortest", video_out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg err:", r.stderr[-1500:])
        raise RuntimeError("音视频合并失败")
    print(f"  最终输出: {video_out}")
    return video_out


def cleanup():
    print("清理临时文件...")
    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    print("清理完成")


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

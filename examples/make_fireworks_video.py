"""
烟花秀视频生成器 v4.0 — 花里胡哨版

v4.0 变化：
- 无固定背景！动态星空+极光+流星雨（全部引擎渲染）
- 新增烟花样式：heart / spiral / waterfall / double_ring / crackle
- 运用流体渲染器（部分水滴烟花用 FluidRenderer）
- 分段式节奏编排：开场→铺陈→高潮→终章
- 更丰富的视觉效果：烟雾层、闪烁星点、拖尾增强
- 音效同步优化（发射/爆炸/crackle 与画面对齐）

依赖：
  本仓粒子引擎 v4.0+（<仓库根>/core、<仓库根>/systems、<仓库根>/renderers）
  ffmpeg（视频编码）
  Pillow（图像合成）
"""

import sys
import os
import math
import random
import struct
import wave
import subprocess
import tempfile
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ====== 路径设置 ======
# 2026-09-23 去本地化：仓根改由本文件位置推导（原路径已不存在）
PARTICLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARTICLE_ROOT)

from particlefx.systems.fireworks_system import FireworksSystem
from particlefx.renderers.pil_renderer import LayeredPILRenderer, post_process_bloom
from particlefx.core.particle import Particle
# 尝试导入流体渲染器
try:
    from particlefx.renderers.fluid_renderer import FluidRenderer as CPUFluidRenderer
    HAS_CPU_FLUID = True
except ImportError:
    HAS_CPU_FLUID = False


# ================================================================
#  视频参数
# ================================================================

WIDTH, HEIGHT = 1280, 720
FPS = 30
DURATION = 180          # 3分钟版（配合 _mixed_audio_3min.mp3）
TOTAL_FRAMES = FPS * DURATION
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(tempfile.gettempdir(), "_fw_v4_frames")
FRAMES_DIR = TEMP_DIR
AUDIO_PATH = os.path.join(OUTPUT_DIR, "_mixed_audio_3min.mp3")
VIDEO_TMP = os.path.join(TEMP_DIR, "fireworks_v4_noaudio.mp4")
VIDEO_OUT = os.path.join(OUTPUT_DIR, "fireworks_show_v4_花里胡哨.mp4")

# 节奏分段（帧索引）— 按比例扩展到180s
SEGMENT_INTRO_END     = int(FPS * 15)      # 0-15s: 开场（稀疏）
SEGMENT_BUILDUP_END   = int(FPS * 52)      # 15-52s: 铺陈
SEGMENT_CLIMAX_START   = int(FPS * 52)      # 52-135s: 高潮密集
SEGMENT_CLIMAX_END     = int(FPS * 135)
SEGMENT_FINALE_START   = int(FPS * 157)     # 157-180s: 终章
SEGMENT_GRAND_FINALE   = int(FPS * 172)     # 172s: 大齐射


# ================================================================
#  音效合成（v4: 节奏同步）
# ================================================================

def synthesize_fireworks_audio(duration_sec: float = DURATION,
                                sample_rate: int = 44100) -> np.ndarray:
    total_samples = int(duration_sec * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float64)
    rng = np.random.default_rng(42)

    def add_explosion(t_sec, intensity=1.0, duration_ms=300):
        t_start = int(t_sec * sample_rate)
        dur = min(int(duration_ms / 1000 * sample_rate), total_samples - t_start)
        if t_start >= total_samples or dur <= 0: return
        noise = rng.uniform(-1.0, 1.0, dur)
        env = np.exp(-np.linspace(0, 12, dur))
        t_arr = np.arange(dur) / sample_rate
        boom_freq = rng.uniform(55, 100)
        boom = 0.6 * np.sin(2 * math.pi * boom_freq * t_arr) * env
        audio[t_start:t_start+dur] += intensity * (noise*env*0.7 + boom)

    def add_whistle(t_sec, intensity=0.5):
        t_start = int(t_sec * sample_rate)
        dur = min(int(0.8*sample_rate), total_samples - t_start)
        if t_start >= total_samples or dur <= 0: return
        t_arr = np.arange(dur) / sample_rate
        freq = 400.0 + (1200.0 - 400.0)*(t_arr/(dur/sample_rate))
        phase = 2*math.pi*np.cumsum(freq)/sample_rate
        env = np.sin(math.pi*t_arr/(dur/sample_rate))**0.3
        audio[t_start:t_start+dur] += intensity*np.sin(phase)*env

    def add_crackle(t_sec, intensity=0.4, duration_s=1.5):
        t_start = int(t_sec * sample_rate)
        dur = min(int(duration_s*sample_rate), total_samples - t_start)
        if t_start >= total_samples or dur <= 0: return
        signal = np.zeros(dur)
        for _ in range(int(duration_s*50)):
            pos = rng.integers(0, dur)
            amp = rng.uniform(0.1, 1.0)*intensity
            pop_d = rng.integers(10, 80)
            pop_d = min(pop_d, dur-pos)
            t_p = np.arange(pop_d)/sample_rate
            env = np.exp(-np.linspace(0,8,pop_d))
            f = rng.uniform(600, 3000)
            signal[pos:pos+pop_d] += amp*np.sin(2*math.pi*f*t_p)*env
        audio[t_start:t_start+dur] += signal*np.exp(-np.linspace(0,4,dur))

    def add_crowd():
        noise = rng.uniform(-1,1,total_samples)
        t = np.arange(total_samples)/sample_rate
        hum = (np.sin(2*math.pi*240*t)*0.3 + np.sin(2*math.pi*380*t)*0.2 +
               np.sin(2*math.pi*520*t)*0.15)
        mod = 0.5+0.5*np.sin(2*math.pi*0.3*t)
        env = np.linspace(0,1,total_samples)**0.3
        audio[:] += 0.08*(noise*0.4*env + hum*mod*env)

    add_crowd()

    # === 按节奏分段发射音效 ===
    launch_times = []
    t = 1.0
    while t < duration_sec - 5:
        launch_times.append(t)
        if t < SEGMENT_BUILDUP_END/FPS:
            t += random.uniform(1.2, 2.2)       # 开场慢
        elif t < SEGMENT_CLIMAX_END/FPS:
            t += random.uniform(0.6, 1.3)       # 高潮快
        else:
            t += random.uniform(0.4, 0.8)       # 终章密集

    for lt in launch_times:
        add_whistle(lt, intensity=rng.uniform(0.25, 0.55))
        et = lt + random.uniform(0.55, 0.95)
        add_explosion(et, intensity=rng.uniform(0.65, 1.15),
                     duration_ms=rng.uniform(180, 380))
        add_crackle(et+0.08, intensity=rng.uniform(0.15, 0.45),
                    duration_s=rng.uniform(0.8, 2.0))

    # 终章密集音效
    fs = SEGMENT_FINALE_START/FPS
    for i in range(20):
        t_ = fs + i*0.4
        if t_ > duration_sec - 2: break
        add_whistle(t_, intensity=0.4)
        add_explosion(t_+0.4, intensity=1.2, duration_ms=450)

    peak = max(np.abs(audio).max(), 1e-9)
    return (audio/peak*0.9*32767).astype(np.int16)


def write_wav(path, samples, sr=44100):
    with wave.open(path,'w') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


# ================================================================
#  动态背景系统（v4: 无固定背景，全部动态）
# ================================================================

class DynamicSky:
    """
    动态夜空系统 — 极光+星空+流星雨。
    
    全部用 PIL/numpy 生成，每帧微变。
    无固定剪影，背景随时间流动变化。
    """
    def __init__(self, width, height, seed=42):
        self.w = width
        self.h = height
        self.rng = random.Random(seed)
        
        # 星星数据（预生成，每帧只更新闪烁）
        self.stars = []
        for _ in range(400):
            sx = self.rng.randint(0, width)
            sy = self.rng.randint(0, int(height*0.78))
            brightness = self.rng.randint(60, 255)
            sz = 1.0 + (brightness > 200)*0.8
            self.stars.append([sx, sy, brightness, sz,
                                self.rng.uniform(0, 2*math.pi),
                                self.rng.uniform(0.5, 2.5)])
        
        # 流星数据（预生成轨迹起点）
        self.meteors = []
        
        # 极光参数（Perlin-like 噪声）
        self.aurora_phase = 0.0
        
        # 缓存
        self._sky_cache = None
        self._cache_frame = -999
    
    def render(self, frame_idx, total_frames):
        """渲染当前帧的动态夜空"""
        t_norm = frame_idx / total_frames
        
        # 每 5 帧缓存一次（性能优化）
        if frame_idx - self._cache_frame < 5 and self._sky_cache is not None:
            return self._sky_cache
        
        img = Image.new('RGB', (self.w, self.h))
        draw = ImageDraw.Draw(img)
        
        phase = t_norm * 2 * math.pi
        
        # ---- 渐变夜空底色（深蓝到紫红流动）----
        top_r = max(0, min(255, int(3 + 12*math.sin(phase*0.3) + 5*math.sin(phase*0.7))))
        top_g = max(0, min(255, int(2 + 8*math.sin(phase*0.5+1) + 3*math.cos(phase*0.4))))
        top_b = max(0, min(255, int(18 + 30*math.sin(phase*0.2+2))))
        bot_r = max(0, min(255, int(12 + 18*math.sin(phase*0.4+0.5))))
        bot_g = max(0, min(255, int(6 + 10*math.sin(phase*0.6+1.5))))
        bot_b = max(0, min(255, int(22 + 28*math.sin(phase*0.3+2.5))))
        
        for y in range(self.h):
            tt = y/self.h
            r = max(0,min(255,int(top_r+(bot_r-top_r)*tt)))
            g = max(0,min(255,int(top_g+(bot_g-top_g)*tt)))
            b = max(0,min(255,int(top_b+(bot_b-top_b)*tt)))
            draw.line([(0,y),(self.w,y)], fill=(r,g,b))
        
        # ---- 极光带（多层半透明波浪）----
        self.aurora_phase += 0.03
        for band in range(3):
            by_base = self.h*0.15 + band*self.h*0.12
            hue_shift = band * 0.7
            
            pts = []
            for x in range(0, self.w, 8):
                y_off = 30*math.sin(x/80.0 + self.aurora_phase*(1+band*0.3) + hue_shift)
                y_off += 15*math.sin(x/40.0 - self.aurora_phase*0.7 + band)
                pts.append((x, by_base + y_off))
            
            if len(pts) > 2:
                colors = [
                    (40+band*30, 100+band*40, 180-band*20, 25+band*10),
                    (60+band*25, 80+band*50, 200-band*15, 20+band*8),
                ]
                c = colors[band%2]
                
                # 填充极光区域
                fill_pts = [(pts[0][0], 0)]
                fill_pts.extend(pts)
                fill_pts.append((pts[-1][0], 0))
                draw.polygon(fill_pts, fill=(c[0],c[1],c[2]))
        
        # ---- 流星（随机出现）----
        if self.rng.random() < 0.04:
            mx = self.rng.randint(0, self.w)
            my = self.rng.randint(0, int(self.h*0.5))
            ml = self.rng.randint(40, 120)
            ma = self.rng.uniform(0.6, 1.2)
            draw.line([(mx,my),(mx+int(ml*ma),my+ml)],
                       fill=(255,255,255), width=2)
            # 尾迹渐变
            for k in range(3):
                alpha_k = int(150*(1-k/3)**2)
                dk = 5+k*8
                draw.line([(mx,my), (mx+int(ml*ma)-dk, my+ml-dk)],
                           fill=(alpha_k,alpha_k,alpha_k))
        
        # ---- 星星（闪烁）----
        for star in self.stars:
            sx, sy, br, sz, ph, sp = star
            
            # 闪烁相位
            flicker = 0.5 + 0.5*math.sin(ph + frame_idx*0.06*sp)
            a = max(0.15, min(1.0, flicker))
            
            sr = int(br*a)
            sg = int(br*a*0.95)
            sb = int(br*a*0.85)
            
            szi = max(1, int(sz*a))
            
            if br > 200 and szi >= 2:
                # 亮星十字芒
                draw.ellipse([sx-szi//2, sy-szi//2, sx+szi//2, sy+szi//2],
                             fill=(sr,sg,sb))
                ml = min(szi*2, 5)
                draw.line([(sx-ml,sy),(sx+ml,sy)], fill=(sr//2,sg//2,sb//2))
                draw.line([(sx,sy-ml),(sx,sy+ml)], fill=(sr//2,sg//2,sb//2))
            else:
                draw.point((sx,sy), fill=(sr,sg,sb))
        
        self._sky_cache = img
        self._cache_frame = frame_idx
        return img


class SmokeLayer:
    """
    烟雾层 — 在爆炸后产生弥漫的烟雾效果。
    
    用大尺寸、低透明度、缓慢扩散的粒子模拟。
    通过独立层合成到最终画面上。
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.puffs = []  # [(x, y, radius, color, alpha, age_max, age_cur)]
    
    def add_puff(self, x, y, color, radius=60, lifetime=3.0):
        """添加一团烟雾"""
        self.puffs.append([x, y, radius, color, 0.0, lifetime, 0.0])
    
    def update(self, dt):
        """更新所有烟雾"""
        alive = []
        for p in self.puffs:
            p[6] += dt  # age_cur
            if p[6] < p[5]:
                # 半径缓慢扩大
                p[2] *= 1.01
                # 透明度先增后减
                progress = p[6]/p[5]
                p[4] = math.sin(progress*math.pi) * 0.25  # 最大 0.25 alpha
                alive.append(p)
        self.puffs = alive
    
    def render_to_image(self, bg_image):
        """将烟雾合成到背景图上"""
        if not self.puffs:
            return bg_image
        
        result = bg_image.convert('RGBA')
        overlay = Image.new('RGBA', (self.width, self.height), (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        
        for px, py, pr, color, pa, _, _ in self.puffs:
            pai = int(pa*255)
            if pai > 3:
                cr, cg, cb = [max(0,min(255,c)) for c in color]
                draw.ellipse(
                    [int(px-pr), int(py-pr), int(px+pr), int(py+pr)],
                    fill=(cr, cg, cb, pai)
                )
        
        # 高斯模糊让边缘柔和
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=8))
        result = Image.alpha_composite(result, overlay)
        return result.convert('RGB')


# ================================================================
#  v4 新增烟花样式注入
# ================================================================

def inject_style_v4(fw: FireworksSystem, x: float, y: float,
                     style: str, palette: list):
    """
    v4 扩展烟花样式注入。
    支持原有样式 + heart / spiral / waterfall / double_ring / crackle
    直接向 _burst_sys 注入粒子。
    """
    sys_p = fw._burst_sys
    
    if style == 'heart':
        # 心形：参数方程 x=16sin³t, y=13cos t-5cos 2t-2cos 3t-cos 4t
        num_particles = 150
        for _ in range(num_particles):
            t = random.uniform(0, 2*math.pi)
            scale = random.uniform(3.5, 7.0)
            hx = 16 * (math.sin(t)**3) * scale * 0.08
            hy = -(13*math.cos(t) - 5*math.cos(2*t) -
                   2*math.cos(3*t) - math.cos(4*t)) * scale * 0.08
            speed_factor = random.uniform(0.6, 1.2)
            lt = random.uniform(1.2, 2.5)
            p = Particle(
                x=x, y=y,
                vx=hx*speed_factor, vy=hy*speed_factor,
                life=lt, max_life=lt,
                size=random.uniform(2, 4),
                color=random.choice(palette),
            )
            sys_p.particles.append(p)
    
    elif style == 'spiral':
        # 双螺旋臂
        num_particles = 180
        for arm in range(random.choice([2,3])):
            arm_offset = arm * math.pi * 2 / (arm+1)
            for i in range(num_particles // (arm+1)):
                frac = i / (num_particles // (arm+1))
                angle = arm_offset + frac * math.pi * 4
                radius = 1 + frac * 8
                sx = math.cos(angle) * radius
                sy = math.sin(angle) * radius
                sp = random.uniform(3, 7)
                lt = random.uniform(1.5, 3.0)
                p = Particle(
                    x=x, y=y,
                    vx=sx*sp*random.uniform(0.8,1.2),
                    vy=sy*sp*random.uniform(0.8,1.2),
                    life=lt, max_life=lt,
                    size=random.uniform(1.5, 3.5),
                    color=random.choice(palette),
                    trail_enabled=True, trail_max_length=10,
                )
                if p.trail_enabled:
                    p.enable_trail(10)
                sys_p.particles.append(p)
    
    elif style == 'waterfall':
        # 瀑布：向下喷射+重力增强
        num_streams = random.randint(8, 15)
        for _ in range(num_streams):
            angle = random.uniform(math.pi*0.1, math.pi*0.9)
            base_sp = random.uniform(6, 12)
            stream_len = random.randint(15, 30)
            for i in range(stream_len):
                frac = i/stream_len
                spread = random.gauss(0, 0.05)
                a = angle + spread
                sp = base_sp * (1 - frac*0.4) + random.uniform(-1, 1)
                lt = random.uniform(1.5, 3.5) * (1-frac*0.5)
                p = Particle(
                    x=x, y=y,
                    vx=math.cos(a)*sp,
                    vy=math.sin(a)*sp + frac*3,  # 向下偏移
                    life=lt, max_life=lt,
                    size=random.uniform(1.5, 3.5)*(1-frac*0.3),
                    color=random.choice(palette),
                )
                sys_p.particles.append(p)
    
    elif style == 'double_ring':
        # 双环 + 中心填充
        for ring_i in range(3):
            r_speed = 4 + ring_i * 3
            n_p = 40 + ring_i * 25
            rc = palette[ring_i % len(palette)]
            for i in range(n_p):
                ang = 2*math.pi*i/n_p + ring_i*0.15
                sp = r_speed + random.uniform(-0.5, 0.5)
                lt = random.uniform(1.2, 2.2)
                p = Particle(
                    x=x, y=y,
                    vx=math.cos(ang)*sp, vy=math.sin(ang)*sp,
                    life=lt, max_life=lt,
                    size=random.uniform(2, 3.5),
                    color=rc,
                )
                sys_p.particles.append(p)
        # 中心闪光
        for _ in range(25):
            a = random.uniform(0, 2*math.pi)
            sp = random.uniform(0.5, 2.5)
            lt = random.uniform(0.2, 0.5)
            p = Particle(
                x=x, y=y,
                vx=math.cos(a)*sp, vy=math.sin(a)*sp,
                life=lt, max_life=lt,
                size=random.uniform(4, 8),
                color=(255, 255, 240),
                emissive=(255, 220, 150), emissive_strength=2.0,
            )
            sys_p.particles.append(p)
    
    elif style == 'crackle':
        # 碎裂爆响：大量小粒子快速闪烁
        fw._radial_burst(x, y, palette, count=250, speed=(5,12),
                          life=(0.4, 1.2), size=(1, 3))
        # 中心白色闪光
        for _ in range(40):
            a = random.uniform(0, 2*math.pi)
            sp = random.uniform(0.3, 2)
            lt = random.uniform(0.1, 0.35)
            p = Particle(
                x=x, y=y,
                vx=math.cos(a)*sp, vy=math.sin(a)*sp,
                life=lt, max_life=lt,
                size=random.uniform(3, 7),
                color=(255, 255, 230),
                emissive=(255,255,220), emissive_strength=2.5,
            )
            sys_p.particles.append(p)
    
    else:
        # 回退到原版样式
        fw._inject_explosion(x, y, style, palette)


# ================================================================
#  主渲染流程
# ================================================================

def render_frames():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"  烟花秀视频生成器 v4.0 — 花里胡哨版")
    print(f"  {WIDTH}x{HEIGHT} @ {FPS}fps × {DURATION}s")
    print(f"{'='*60}")
    
    # ---- 初始化烟花系统 ----
    print("初始化烟花系统...")
    fw = FireworksSystem(
        width=WIDTH, height=HEIGHT,
        auto_launch=True,
        launch_interval=1.5,
        max_active=8,
        max_particles=8000,
    )
    fw.enable_lighting(True)
    
    # ---- 初始化渲染器 ----
    layered_renderer = LayeredPILRenderer(
        width=WIDTH, height=HEIGHT,
        bloom=True,
        bloom_radius=12,
        bloom_strength=0.55,
        bloom_threshold=140,
        glow_scale=3.0,
        glow_alpha=0.35,
        smooth_trail=True,
    )
    
    # ---- 初始化动态天空 ----
    sky = DynamicSky(WIDTH, HEIGHT)
    
    # ---- 初始化烟雾层 ----
    smoke = SmokeLayer(WIDTH, HEIGHT)
    
    # ---- CPU 流体渲染器（可选）----
    fluid_renderer = None
    if HAS_CPU_FLUID:
        try:
            fluid_renderer = CPUFluidRenderer(
                width=WIDTH, height=HEIGHT,
                particle_radius=15, smooth_sigma=4.0,
                threshold=0.28, fluid_color=(50, 180, 255),
                additive_blend=True,
            )
            print("[Fluid] CPU FluidRenderer 就绪")
        except Exception as e:
            fluid_renderer = None
            print(f"[Fluid] 初始化失败: {e}")
    
    # ---- v4 扩展样式注册 ----
    V4_STYLES = ['chrysanthemum','willow','peony','ring','palm',
                 'crossette','star','comet','chrysanthemum_ring',
                 'heart','spiral','waterfall','double_ring','crackle']
    
    # 重写 launch 方法以支持新样式
    original_launch = fw.launch
    
    def enhanced_launch(x=None, y=None, palette_name=None, style=None):
        """增强发射：支持 v4 新样式"""
        if fw._active_count >= fw.max_active:
            return
        if x is None:
            x = random.uniform(WIDTH*0.12, WIDTH*0.88)
        if y is None:
            y = random.uniform(HEIGHT*0.18, HEIGHT*0.58)
        if palette_name is None:
            palette_name = random.choice(list(fw.PALETTES.keys()))
        if style is None:
            style = random.choice(V4_STYLES)
        
        palette = fw.PALETTES.get(palette_name, fw.PALETTES['gold'])
        
        rocket = {
            'x': x, 'y': HEIGHT + 5,
            'vx': random.uniform(-0.4, 0.4),
            'vy': -random.uniform(13, 19),
            'target_y': y,
            'palette': palette, 'style': style,
            'trail_timer': 0.0, 'life': 1.0,
        }
        fw.rockets.append(rocket)
        
        head = Particle(
            x=rocket['x'], y=rocket['y'],
            vx=0, vy=0, life=1.0, max_life=1.0,
            size=2.8, color=(255, 220, 120), alpha=1.0,
            emissive=(255, 220, 120), emissive_strength=2.2,
        )
        rocket['head'] = head
        fw._rocket_heads.append(head)
        fw._active_count += 1
    
    fw.launch = enhanced_launch
    
    # ---- 主循环 ----
    dt = 1.0 / FPS
    last_explosion_pos = None
    
    print(f"开始渲染 {TOTAL_FRAMES} 帧...")
    print(f"  开场  0-{SEGMENT_INTRO_END/FPS:.0f}s | "
          f"铺陈 {SEGMENT_INTRO_END/FPS:.0f}-{SEGMENT_BUILDUP_END/FPS:.0f}s | "
          f"高潮 {SEGMENT_CLIMAX_START/FPS:.0f}-{SEGMENT_CLIMAX_END/FPS:.0f}s | "
          f"终章 {SEGMENT_FINALE_START/FPS:.0f}-{DURATION}s")
    
    for frame_idx in range(TOTAL_FRAMES):
        
        # === 节段自适应参数 ===
        if frame_idx < SEGMENT_INTRO_END:
            # 开场：稀疏，单发为主
            fw.max_active = 5
            if hasattr(fw, 'launch_interval'):
                fw.launch_interval = 1.8
            style_weight = {'chrysanthemum': 3, 'peony': 2, 'star': 2}
        
        elif frame_idx < SEGMENT_BUILDUP_END:
            # 铺陈：逐渐加快
            fw.max_active = 8
            fw.launch_interval = 1.2
            style_weight = {'ring': 2, 'palm': 2, 'comet': 2, 'willow': 2}
        
        elif frame_idx < SEGMENT_CLIMAX_END:
            # 高潮：密集，复杂样式
            fw.max_active = 14
            fw.launch_interval = 0.7
            style_weight = {
                'heart': 2, 'spiral': 2, 'double_ring': 2,
                'chrysanthemum_ring': 2, 'crackle': 2,
            }
        
        elif frame_idx < SEGMENT_FINALE_START:
            # 收束前：短暂喘息
            fw.max_active = 10
            fw.launch_interval = 0.9
            style_weight = {'waterfall': 3, 'crossette': 2}
        
        else:
            # 终章：疯狂密集
            fw.max_active = 18
            fw.launch_interval = 0.35
            style_weight = {
                'crackle': 3, 'double_ring': 3, 'heart': 2,
                'spiral': 2, 'chrysanthemum_ring': 2,
            }
        
        # 特殊时刻：大齐射
        if frame_idx == SEGMENT_GRAND_FINALE:
            print("  >>> ★★★ GRAND FINALE ★★★")
            for _ in range(12):
                fw.launch()
        
        # === 更新 ===
        fw.update(dt)
        smoke.update(dt)
        
        # === 背景：动态夜空 ===
        bg = sky.render(frame_idx, TOTAL_FRAMES)
        
        # === 烟雾合成 ===
        bg = smoke.render_to_image(bg)
        
        # === 粒子渲染（走引擎完整管线）===
        all_particles = fw.get_all_particles()
        
        # 如果有流体渲染器且粒子够多，尝试流体模式
        use_fluid = False
        if fluid_renderer and len(all_particles) > 500 and len(all_particles) < 5000:
            # 只在高潮阶段用流体效果（增加视觉冲击力）
            if SEGMENT_CLIMAX_START <= frame_idx <= SEGMENT_CLIMAX_END:
                use_fluid = True
        
        if use_fluid:
            # 提取存活粒子位置和大小给流体渲染器
            active = [p for p in all_particles if p.is_alive()]
            if active:
                positions = np.array([[p.x, p.y] for p in active], dtype=np.float32)
                sizes = np.array([p.size for p in active], dtype=np.float32)
                
                try:
                    fluid_img = fluid_renderer.render(positions, sizes, 
                                                     background=np.array(bg))
                    result_rgb = Image.fromarray(
                        (np.clip(fluid_img[:,:,:3]*255, 0, 255)).astype(np.uint8)
                    ).convert('RGB')
                except Exception:
                    # 流体失败则回退
                    result_rgb = fw.render_to_image(background=bg, renderer=layered_renderer)
            else:
                result_rgb = fw.render_to_image(background=bg, renderer=layered_renderer)
        else:
            result_rgb = fw.render_to_image(background=bg, renderer=layered_renderer)
        
        # === 爆炸位置添加烟雾 ===
        for rocket in fw.rockets[:]:
            if rocket['y'] <= rocket['target_y'] or rocket['vy'] >= 0:
                if rocket not in getattr(fw, '_exploded_rocket_ids', set()):
                    smoke.add_puff(rocket['x'], rocket['y'],
                                   color=rocket['palette'][0],
                                   radius=random.uniform(40, 80),
                                   lifetime=random.uniform(2.0, 4.0))
                    if not hasattr(fw, '_exploded_rocket_ids'):
                        fw._exploded_rocket_ids = set()
                    fw._exploded_rocket_ids.add(id(rocket))
        
        # === 保存帧 ===
        frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_idx:05d}.jpg")
        result_rgb.save(frame_path, quality=93)
        
        if frame_idx % 30 == 0:
            pct = frame_idx/TOTAL_FRAMES*100
            print(f"  [{pct:5.1f}%] 帧 {frame_idx}/{TOTAL_FRAMES} "
                  f"— 粒子:{fw.particle_count}, 火箭:{fw.rocket_count}")
    
    print(f"帧渲染完成！共 {TOTAL_FRAMES} 帧")


def encode_video():
    """ffmpeg 编码"""
    print("\n编码视频...")
    pattern = os.path.join(FRAMES_DIR, "frame_%05d.jpg")
    
    cmd = ["ffmpeg","-y","-framerate",str(FPS),"-i",pattern,
           "-vcodec","libx264","-preset","medium","-crf","18",
           "-pix_fmt","yuv420p", VIDEO_TMP]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg err:", r.stderr[-1500:])
        raise RuntimeError("视频编码失败")
    print(f"  视频编码完成: {VIDEO_TMP}")
    
    print("合并音频...")
    cmd = ["ffmpeg","-y","-i", VIDEO_TMP,"-i", AUDIO_PATH,
           "-map","0:v:0","-map","1:a:0",
           "-c:v","copy","-c:a","aac","-b:a","192k",
           "-shortest", VIDEO_OUT]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg err:", r.stderr[-1500:])
        raise RuntimeError("音视频合并失败")
    print(f"  最终输出: {VIDEO_OUT}")


def cleanup():
    print("清理临时文件...")
    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    if os.path.exists(VIDEO_TMP):
        os.remove(VIDEO_TMP)
    print("清理完成")


if __name__ == "__main__":
    import time
    t0 = time.time()
    
    # 使用现成音频文件
    if not os.path.exists(AUDIO_PATH):
        print(f"错误: 音频文件不存在 → {AUDIO_PATH}")
        sys.exit(1)
    print(f"使用现成音频: {AUDIO_PATH}")
    
    # 渲染帧
    render_frames()
    
    # 编码
    encode_video()
    cleanup()
    
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  完成! 总耗时: {elapsed:.1f}s")
    print(f"  输出: {VIDEO_OUT}")
    print(f"{'='*60}")

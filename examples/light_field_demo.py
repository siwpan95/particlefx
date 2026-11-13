"""
交互式粒子光场 · pygame 实时 Demo
===================================

把 ``particle-light-field.html`` 的交互模型用 particlefx 原生复现：

  鼠标移动            -> 吸引
  按住鼠标（左键）    -> 排斥
  点击（左键按下）     -> 冲击波
  空格                -> 切换配色（Aurora / Ember / Toxic，平滑过渡）
  R                  -> 用同一 seed 重置（画面可复现）
  P                  -> 暂停 / 继续
  S                  -> 导出当前帧 PNG（需 Pillow）
  [ / ]              -> 减少 / 增加粒子数

运行：
    pip install "particlefx[pygame,pil]"
    python light_field_demo.py                 # 随机 seed
    python light_field_demo.py --seed 12345    # 固定 seed，画面可复现
    python light_field_demo.py --count 2000 --width 1280 --height 720
"""

import sys
import math
import random

try:
    import pygame
except ImportError:
    sys.stderr.write(
        "缺少 pygame。请先安装：pip install \"particlefx[pygame,pil]\"\n"
    )
    raise

from particlefx import LightFieldSystem


def parse_args(argv):
    args = {"seed": None, "count": 1500, "width": 960, "height": 600,
            "max_particles": 3000}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--seed" and i + 1 < len(argv):
            args["seed"] = int(argv[i + 1]); i += 2
        elif a == "--count" and i + 1 < len(argv):
            args["count"] = int(argv[i + 1]); i += 2
        elif a == "--width" and i + 1 < len(argv):
            args["width"] = int(argv[i + 1]); i += 2
        elif a == "--height" and i + 1 < len(argv):
            args["height"] = int(argv[i + 1]); i += 2
        else:
            i += 1
    return args


def main():
    args = parse_args(sys.argv[1:])
    seed = args["seed"] if args["seed"] is not None else random.randint(0, 2**31)
    W, H = args["width"], args["height"]

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("particlefx · 交互式粒子光场")

    system = LightFieldSystem(
        width=W, height=H,
        count=args["count"],
        seed=seed,
        max_particles=args["max_particles"],
    )

    # 持久 surface：每帧半透明覆盖实现拖尾衰减
    surf = pygame.Surface((W, H), pygame.SRCALPHA)

    try:
        from particlefx.renderers import PygameRenderer
    except ImportError:
        sys.stderr.write("缺少 pygame 渲染器支持，无法运行 Demo。\n")
        return

    renderer = PygameRenderer(
        surf,
        blend_mode='additive',
        bloom_enabled=True,
        bloom_threshold=110,
        bloom_intensity=0.55,
        bloom_sigma=8.0,
        bloom_radius=12,
        bloom_downsample=2,
        glow_enabled=True,
        default_glow_radius_mult=2.5,
    )

    font = pygame.font.SysFont(None, 18)
    clock = pygame.time.Clock()
    paused = False
    running = True

    def bg_rgba(alpha):
        r, g, b = system.current_bg()
        return (r, g, b, int(alpha * 255))

    while running:
        dt = clock.tick(60)
        if dt > 100:
            dt = 100  # 防止后台切回大跳变

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k == pygame.K_SPACE:
                    system.cycle_palette()
                elif k == pygame.K_r:
                    system.reset()
                elif k == pygame.K_p:
                    paused = not paused
                    system.paused = paused
                elif k == pygame.K_s:
                    try:
                        fname = "lightfield_seed%d_%d.png" % (system.seed,
                                                             pygame.time.get_ticks())
                        system.save_image(fname)
                        print("已保存:", fname)
                    except Exception as e:
                        print("保存失败（需 Pillow）:", e)
                elif k == pygame.K_LEFTBRACKET:
                    system.set_count(max(200, system.get_particle_count() - 200))
                elif k == pygame.K_RIGHTBRACKET:
                    system.set_count(min(system.max_particles,
                                         system.get_particle_count() + 200))
                elif k == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                system.set_pointer(event.pos[0], event.pos[1], active=True,
                                  pressing=True)
                system.trigger_shockwave(event.pos[0], event.pos[1])
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                system.set_pointer(event.pos[0], event.pos[1], active=True,
                                  pressing=False)
            elif event.type == pygame.MOUSEMOTION:
                system.set_pointer(event.pos[0], event.pos[1], active=True,
                                  pressing=pygame.mouse.get_pressed()[0])

        # 拖尾：半透明背景覆盖（source-over），逐步衰减上一帧
        fade = pygame.Surface((W, H), pygame.SRCALPHA)
        fade.fill(bg_rgba(system.FADE_ALPHA))
        surf.blit(fade, (0, 0))

        if not paused:
            system.update(dt)

        renderer.render(system.particles)
        renderer.apply_bloom()

        # 叠加冲击波可见环
        for sw in system.shockwaves:
            pygame.draw.circle(
                surf, (180, 220, 255),
                (int(sw["x"]), int(sw["y"])), int(sw["r"]),
                2,
            )

        screen.blit(surf, (0, 0))

        # HUD
        fps = clock.get_fps()
        lines = [
            "鼠标移动=吸引  按住=排斥  点击=冲击波",
            "空格=换色  R=重置  P=暂停  S=保存  [ ]=粒子数",
            "seed=%d  配色=%s  粒子=%d  %.0f FPS%s" % (
                system.seed, system.current_palette_name(),
                system.get_particle_count(), fps,
                "  [暂停]" if paused else ""),
        ]
        y = 10
        for line in lines:
            txt = font.render(line, True, (223, 231, 255))
            screen.blit(txt, (12, y))
            y += 22

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()

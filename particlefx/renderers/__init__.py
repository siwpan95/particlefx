"""渲染器模块 v4.0

各渲染器按其第三方依赖**独立可选**：缺依赖的那个不导出，其余照常可用。
这样 `pip install particlefx`（仅 numpy）后 `import particlefx.renderers` 不会炸，
而 `from particlefx.renderers import PygameRenderer` 在缺 pygame 时给出明确报错。

可用性探测：
    from particlefx.renderers import AVAILABLE
    print(AVAILABLE)          # {'PygameRenderer', 'PILRenderer', ...}
"""

__all__ = []

# ---- 核心渲染器（无第三方依赖或仅 numpy）----
from .fluid_renderer import FluidRenderer
__all__.append('FluidRenderer')

# ---- pygame 渲染器（需 pygame-ce）----
try:
    from .pygame_renderer import PygameRenderer
    __all__.append('PygameRenderer')
except ImportError:  # pragma: no cover - 取决于环境是否装了 pygame
    pass

# ---- PIL 渲染器与后处理链（需 Pillow）----
try:
    from .pil_renderer import PILRenderer
    __all__.append('PILRenderer')
except ImportError:  # pragma: no cover
    pass

try:
    from .postfx import (
        PostFXChain, PostFXStep,
        BloomStep, TonemapStep, VignetteStep,
        ChromaticAberrationStep, GrainStep,
        default_fireworks_chain, default_clean_chain,
    )
    __all__ += [
        'PostFXChain', 'PostFXStep',
        'BloomStep', 'TonemapStep', 'VignetteStep',
        'ChromaticAberrationStep', 'GrainStep',
        'default_fireworks_chain', 'default_clean_chain',
    ]
except ImportError:  # pragma: no cover
    pass

# ---- ModernGL GPU 渲染器（需 moderngl + pygame）----
try:
    from .fluid_renderer_moderngl import UnifiedGPURenderer as ModernGLFluidRenderer
    from .particle_renderer_moderngl import GPUParticleRenderer
    __all__ += ['ModernGLFluidRenderer', 'GPUParticleRenderer']
except ImportError:  # pragma: no cover
    pass

#: 当前环境下实际可用的渲染器名集合。
#: 注：FluidRenderer 的主体是纯 numpy 计算，仅 ``render_to_surface`` 需要 pygame，
#: 故它在无 pygame 时仍被导出（用到该方法时才需装 pygame-ce）。
AVAILABLE = frozenset(__all__)

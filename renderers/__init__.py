"""渲染器模块 v4.0"""
from renderers.pygame_renderer import PygameRenderer
from renderers.pil_renderer import PILRenderer
from renderers.fluid_renderer import FluidRenderer
from renderers.postfx import (
    PostFXChain, PostFXStep,
    BloomStep, TonemapStep, VignetteStep,
    ChromaticAberrationStep, GrainStep,
    default_fireworks_chain, default_clean_chain,
)

# ModernGL GPU 渲染器
try:
    from renderers.fluid_renderer_moderngl import UnifiedGPURenderer as ModernGLFluidRenderer
    from renderers.particle_renderer_moderngl import GPUParticleRenderer
    __all__ = [
        'PygameRenderer',
        'PILRenderer',
        'FluidRenderer',
        'ModernGLFluidRenderer',
        'GPUParticleRenderer',
    ]
except ImportError:
    __all__ = [
        'PygameRenderer',
        'PILRenderer',
        'FluidRenderer',
    ]

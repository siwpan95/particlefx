"""
particlefx — 独立粒子系统库

纯 Python + NumPy 实现的粒子系统，附带多渲染器（pygame / PIL / ModernGL）、
效果工厂（30+ 预设配方）、天气与战斗特效系统、Taichi WCSPH 流体后端与
Compute Shader 全 GPU 模拟。

快速上手：
    from particlefx import ParticleSystem, Emitter, ForceField
    from particlefx.renderers import PygameRenderer

    ps = ParticleSystem(max_particles=1000)
    ps.add_emitter(Emitter(x=400, y=300, rate=50.0))
    ps.add_force_field(ForceField(field_type='gravity', strength=100.0))

子包：
    particlefx.core        核心粒子类 / NumPy 向量化 / Numba JIT / Taichi SPH / 空间哈希
    particlefx.renderers   pygame / PIL / ModernGL / 流体 / 后处理 / GPU 模拟
    particlefx.components  效果工厂与声明式配方（JSON）
    particlefx.systems     烟花 / 天气 / 战斗特效

许可证：MIT
"""

from .core import (
    Particle,
    ParticlePool,
    ParticleSystem,
    Emitter,
    EmitterShape,
    ForceField,
    NoiseField,
    ParticleAttractor,
    CollisionBox,
    CollisionBoundary,
    LightSource,
    BlinnPhongParams,
    Renderer,
    SpatialHashGrid,
    create_attractor,
    ParticleSystemNumpy,
)

__version__ = "1.0.0"

__all__ = [
    '__version__',
    # 核心
    'Particle',
    'ParticlePool',
    'ParticleSystem',
    'Emitter',
    'EmitterShape',
    'ForceField',
    'NoiseField',
    'ParticleAttractor',
    'CollisionBox',
    'CollisionBoundary',
    'LightSource',
    'BlinnPhongParams',
    'Renderer',
    'SpatialHashGrid',
    'create_attractor',
    'ParticleSystemNumpy',
]


def available_backends():
    """返回当前环境下可用的加速后端名集合。

    仅装了 numpy 时返回 ``{'ParticleSystemNumpy'}``；装了 numba / taichi 会
    追加对应实现。用于运行时判断「这台机器能不能用某条快路径」。

    Returns:
        set[str]: 后端类名集合
    """
    from .core import AVAILABLE_BACKENDS
    return set(AVAILABLE_BACKENDS)

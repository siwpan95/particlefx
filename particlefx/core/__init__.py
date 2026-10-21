"""粒子系统核心模块

含四层实现，**逐层可选**：

- 纯 Python 层（无额外依赖，始终可用）：``ParticleSystem``
- NumPy 向量化层（需 numpy）：``ParticleSystemNumpy``
- Numba JIT 层（需 numba）：``NumbaParticleSystem``
- Taichi WCSPH 流体层（需 taichi）：``TaichiSPHSystem``

未安装对应可选依赖时，该层的类不在 ``__all__`` 中（导入本模块不会报错），
可用 ``particlefx.core.AVAILABLE_BACKENDS`` 查看当前环境实际支持的后端。
"""
from .particle import (
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
    create_attractor,
)
from .spatial_hash import SpatialHashGrid

__all__ = [
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
]

# ---- NumPy 向量化层（numpy 是核心依赖，始终可用）----
from .particle_numpy import (
    ParticleSystemNumpy,
    NumpyForceBase,
    NumpyForceGravity,
    NumpyForceWind,
    NumpyForceDrag,
    NumpyForceAttraction,
    NumpyForceTurbulence,
)
__all__ += [
    'ParticleSystemNumpy',
    'NumpyForceBase',
    'NumpyForceGravity',
    'NumpyForceWind',
    'NumpyForceDrag',
    'NumpyForceAttraction',
    'NumpyForceTurbulence',
]

# ---- Numba JIT 层（需 numba）----
try:
    from .particle_numba import NumbaParticleSystem, HAS_NUMBA
    __all__.append('NumbaParticleSystem')
except ImportError:  # pragma: no cover - 取决于环境是否装了 numba
    HAS_NUMBA = False

# ---- Taichi WCSPH 流体层（需 taichi）----
try:
    from .particle_taichi import TaichiSPHSystem, HAS_TAICHI
    __all__.append('TaichiSPHSystem')
except ImportError:  # pragma: no cover
    HAS_TAICHI = False

#: 当前环境下**真正可运行**的后端实现名。
#: 注：NumbaParticleSystem / TaichiSPHSystem 在缺依赖时仍可导入（降级为纯 Python 实现），
#: 故此处依据 HAS_NUMBA / HAS_TAICHI 标志位判定，而非依据是否导入成功。
AVAILABLE_BACKENDS = frozenset(
    n for n in ('ParticleSystem', 'ParticleSystemNumpy')
) | ({'NumbaParticleSystem'} if HAS_NUMBA else set()) | ({'TaichiSPHSystem'} if HAS_TAICHI else set())

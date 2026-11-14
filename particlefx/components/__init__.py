"""粒子效果组件包"""
from .effects import (
    create_explosion,
    create_fire,
    create_rain,
    create_snow,
    create_magic_circle,
    create_fountain,
    create_smoke,
    create_sparkle,
    create_trail,
    get_effect,
    list_effects,
    EFFECT_REGISTRY,
)
from .effects_factory import (
    EffectsFactory,
    get_factory,
    create_effect,
    list_all_effects,
)
from .preset import (
    system_to_recipe,
    recipe_to_system,
    save_preset,
    load_preset,
    export_effect_recipe,
)

__all__ = [
    'create_explosion',
    'create_fire',
    'create_rain',
    'create_snow',
    'create_magic_circle',
    'create_fountain',
    'create_smoke',
    'create_sparkle',
    'create_trail',
    'get_effect',
    'list_effects',
    'EFFECT_REGISTRY',
    # 声明式效果工厂（JSON 配方驱动）
    'EffectsFactory',
    'get_factory',
    'create_effect',
    'list_all_effects',
    'system_to_recipe',
    'recipe_to_system',
    'save_preset',
    'load_preset',
    'export_effect_recipe',
]

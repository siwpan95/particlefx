"""
场景配方与种子持久化（v1.2 新增）

把 ``ParticleSystem`` 的「结构 + 参数 + 系统种子」序列化为一份可分享的 JSON 配方，
用于「创意库」式的内容沉淀与复现：

- :func:`system_to_recipe` ：把当前 ``ParticleSystem`` 导出为配方字典；
- :func:`recipe_to_system` ：由配方重建完全一致的 ``ParticleSystem``（同种子 -> 逐位复现）；
- :func:`save_preset` / :func:`load_preset` ：配方 <-> 文件；
- :func:`export_effect_recipe` ：把 EffectsFactory 的某个声明式效果导出为配方（带种子）。

设计要点（与 v1.2 可复现内核一致）：
- 配方顶层只保存「系统种子」；由系统派生的发射器 / 力场 / 噪声场种子在重建时
  由系统重新派生，因此一份配方 + 一个种子即可完整复现场景，且天然可分享。
- 用户显式指定的发射器 / 力场种子会被原样保留。
"""

import json
import os
from typing import Dict, Any, Optional

from ..core.particle import (
    ParticleSystem, Emitter, EmitterShape, ForceField, NoiseField,
)


def _shape_to_dict(shape: EmitterShape) -> Dict[str, Any]:
    return {
        'shape_type': shape.shape_type,
        'radius': shape.radius,
        'height': shape.height,
        'width': shape.width,
        'rect_height': shape.rect_height,
        'edge_length': shape.edge_length,
        'edge_angle': shape.edge_angle,
        'emit_from_shell': shape.emit_from_shell,
        'emit_from_volume': shape.emit_from_volume,
        'randomize_direction': shape.randomize_direction,
    }


def _shape_from_dict(d: Dict[str, Any]) -> EmitterShape:
    s = EmitterShape(d.get('shape_type', 'point'))
    for k, v in d.items():
        if k != 'shape_type' and hasattr(s, k):
            setattr(s, k, v)
    return s


def _to_jsonable(v):
    """递归把关键帧曲线里的 tuple 转成 list，得到纯 JSON 友好结构。"""
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    return v


def _serialize_curve(curve):
    """生命期曲线是 ``List[Tuple[float, ...]]`` 关键帧数据，可直接序列化（忠实还原）。

    这些曲线并非 callable：``gradient_to_curve`` / ``velocity_curve_accelerate`` 等
    工厂返回的都是关键帧列表，evaluate_curve* 对 list / tuple 通用，因此逐位还原无损。
    """
    if curve is None:
        return None
    if isinstance(curve, str):
        # 兼容极少数把曲线名直接存进字段的场景（重建时按无效曲线忽略）
        return curve
    return _to_jsonable(curve)


def _deserialize_curve(value):
    """由 JSON 数据重建生命期曲线；字符串名称（无法还原为数据）按无效忽略。"""
    if isinstance(value, str):
        return None
    return value


def _emitter_to_dict(em: Emitter) -> Dict[str, Any]:
    # 派生种子不落地（重建时由系统重新派生）；显式种子原样保留。
    seed = em.seed if not getattr(em, '_system_derived', False) else None
    return {
        'x': em.x, 'y': em.y,
        'rate': em.rate, 'spread': em.spread, 'direction': em.direction,
        'speed': em.speed, 'life': em.life, 'size': em.size,
        'color': list(em.color) if not isinstance(em.color, str) else em.color,
        'rotation_range': list(em.rotation_range),
        'angular_velocity_range': list(em.angular_velocity_range),
        'mass_range': list(em.mass_range),
        'color_over_lifetime': _serialize_curve(em.color_over_lifetime),
        'size_over_lifetime': _serialize_curve(em.size_over_lifetime),
        'velocity_over_lifetime': _serialize_curve(em.velocity_over_lifetime),
        'emissive': list(em.emissive) if em.emissive is not None else None,
        'emissive_strength': em.emissive_strength,
        'trail_enabled': em.trail_enabled,
        'trail_length': em.trail_length,
        'shape': _shape_to_dict(em.shape),
        'seed': seed,
    }


def _force_field_to_dict(ff: ForceField) -> Dict[str, Any]:
    seed = ff.seed if not getattr(ff, '_system_derived', False) else None
    return {
        'field_type': ff.field_type,
        'x': ff.x, 'y': ff.y, 'strength': ff.strength,
        'noise_scale': ff.noise_scale, 'noise_strength': ff.noise_strength,
        'noise_octaves': ff.noise_octaves,
        'vortex_clockwise': ff.vortex_clockwise, 'vortex_falloff': ff.vortex_falloff,
        'turb_scale': ff.turb_scale, 'turb_strength': ff.turb_strength,
        'magnet_polarity': ff.magnet_polarity, 'magnet_range': ff.magnet_range,
        'seed': seed,
    }


def _noise_field_to_dict(nf: NoiseField) -> Dict[str, Any]:
    seed = nf.seed if not getattr(nf, '_system_derived', False) else None
    return {
        'strength': nf.strength, 'scale': nf.scale,
        'octaves': nf.octaves, 'seed': seed,
    }


def system_to_recipe(ps: ParticleSystem, name: str = 'preset') -> Dict[str, Any]:
    """把 ``ParticleSystem`` 导出为可序列化的配方字典。

    顶层仅保存系统种子；由系统派生的子流种子在重建时重新派生。
    """
    emitters = [_emitter_to_dict(em) for em in ps.emitters]
    force_fields = [_force_field_to_dict(ff) for ff in ps.force_fields]
    noise_fields = [_noise_field_to_dict(nf) for nf in ps.noise_fields]
    return {
        'version': 1,
        'name': name,
        'max_particles': ps.max_particles,
        'seed': ps._seed,
        'fixed_dt': ps.fixed_dt,
        'max_substeps': ps.max_substeps,
        'emitters': emitters,
        'force_fields': force_fields,
        'noise_fields': noise_fields,
    }


def recipe_to_system(recipe: Dict[str, Any]) -> ParticleSystem:
    """由配方字典重建 ``ParticleSystem``（同种子 -> 逐位复现）。"""
    ps = ParticleSystem(
        max_particles=recipe.get('max_particles', 1000),
        seed=recipe.get('seed'),
        fixed_dt=recipe.get('fixed_dt'),
        max_substeps=recipe.get('max_substeps', 100),
    )
    for ed in recipe.get('emitters', []):
        shape = _shape_from_dict(ed.get('shape', {}))
        em = Emitter(
            x=ed['x'], y=ed['y'],
            rate=ed.get('rate', 1.0), spread=ed.get('spread', 0.5),
            direction=ed.get('direction', 0.0), speed=ed.get('speed', 10.0),
            life=ed.get('life', 1.0), size=ed.get('size', 5.0),
            color=tuple(ed['color']) if isinstance(ed.get('color'), list) else ed.get('color', (255, 255, 255)),
            rotation_range=tuple(ed.get('rotation_range', (0, 0))),
            angular_velocity_range=tuple(ed.get('angular_velocity_range', (0, 0))),
            mass_range=tuple(ed.get('mass_range', (1, 1))),
            color_over_lifetime=_deserialize_curve(ed.get('color_over_lifetime')),
            size_over_lifetime=_deserialize_curve(ed.get('size_over_lifetime')),
            velocity_over_lifetime=_deserialize_curve(ed.get('velocity_over_lifetime')),
            trail_enabled=ed.get('trail_enabled', False),
            trail_length=ed.get('trail_length', 10),
            emissive=tuple(ed['emissive']) if isinstance(ed.get('emissive'), list) else ed.get('emissive'),
            emissive_strength=ed.get('emissive_strength', 1.0),
            shape=shape,
            seed=ed.get('seed'),
        )
        ps.add_emitter(em)
    for ffd in recipe.get('force_fields', []):
        ps.add_force_field(ForceField(
            field_type=ffd.get('field_type', 'gravity'),
            x=ffd.get('x', 0.0), y=ffd.get('y', 0.0),
            strength=ffd.get('strength', 1.0),
            noise_scale=ffd.get('noise_scale', 50.0),
            noise_strength=ffd.get('noise_strength', 1.0),
            noise_octaves=ffd.get('noise_octaves', 2),
            vortex_clockwise=ffd.get('vortex_clockwise', True),
            vortex_falloff=ffd.get('vortex_falloff', 0.01),
            turb_scale=ffd.get('turb_scale', 30.0),
            turb_strength=ffd.get('turb_strength', 2.0),
            magnet_polarity=ffd.get('magnet_polarity', 1.0),
            magnet_range=ffd.get('magnet_range', 200.0),
            seed=ffd.get('seed'),
        ))
    for nfd in recipe.get('noise_fields', []):
        ps.add_noise_field(NoiseField(
            strength=nfd.get('strength', 1.0),
            scale=nfd.get('scale', 50.0),
            octaves=nfd.get('octaves', 2),
            seed=nfd.get('seed'),
        ))
    return ps


def save_preset(recipe: Dict[str, Any], path: str) -> str:
    """把配方字典写入 JSON 文件，返回路径。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(recipe, f, ensure_ascii=False, indent=2)
    return path


def load_preset(path: str) -> Dict[str, Any]:
    """从 JSON 文件读取配方字典。"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def export_effect_recipe(factory, name: str, seed: int = None, **params) -> Dict[str, Any]:
    """把 EffectsFactory 的某个声明式效果导出为带种子的配方（便于分享 / 复现）。

    Args:
        factory: EffectsFactory 实例。
        name: 效果名（如 'nebula_burst'）。
        seed: 可选系统种子；给定 -> 配方可复现。
        **params: 运行时参数（x, y, ...）。
    """
    params = dict(params)
    if seed is not None:
        params.setdefault('seed', seed)
    system = factory.create(name, **params)
    if system is None:
        raise ValueError(f"EffectsFactory 无法创建效果: {name}")
    return system_to_recipe(system, name=name)

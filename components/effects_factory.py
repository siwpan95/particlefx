"""
效果工厂 v4.0 — JSON 驱动

读取 effects_config.json，将声明式配方转换为 ParticleSystem 实例。
不改动 effects.py 的已有工厂函数，新增效果只需修改 JSON。

用法：
  from components.effects_factory import EffectsFactory
  factory = EffectsFactory()
  system = factory.create('nebula_burst', x=400, y=300)
"""

import json
import math
import os
import random
from typing import Dict, List, Optional, Any, Tuple

# v5.0：jsonschema 为可选依赖（未安装时回退到内置轻量校验）
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

try:
    from core.particle import (
        ParticleSystem, Emitter, ForceField, EmitterShape,
        FIRE_COLOR_CURVE, SMOKE_COLOR_CURVE, MAGIC_COLOR_CURVE,
        gradient_to_curve, size_curve_fadeout, size_curve_pulse,
        velocity_curve_accelerate, velocity_curve_decelerate,
        create_vortex, create_turbulence, create_magnet,
    )
except ImportError:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.particle import (
        ParticleSystem, Emitter, ForceField, EmitterShape,
        FIRE_COLOR_CURVE, SMOKE_COLOR_CURVE, MAGIC_COLOR_CURVE,
        gradient_to_curve, size_curve_fadeout, size_curve_pulse,
        velocity_curve_accelerate, velocity_curve_decelerate,
        create_vortex, create_turbulence, create_magnet,
    )

# ============================================================
#  资源加载
# ============================================================

def _load_json(filename: str) -> dict:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _validate_config_structure(config: dict) -> List[str]:
    """
    内置轻量结构校验（jsonschema 未安装时的回退，v5.0 增强 ⑦）。

    校验：根节点 / categories / effects 结构、每个效果的 emitters 与
    force_fields 的必要字段。返回错误信息列表（空 = 通过）。
    """
    errors: List[str] = []
    if not isinstance(config, dict):
        return ["配置根节点必须是对象"]
    for req in ("effects", "categories"):
        if req not in config:
            errors.append(f"缺少必需字段: {req}")
    categories = config.get("categories", {})
    if not isinstance(categories, dict):
        errors.append("categories 必须是对象")
    effects = config.get("effects", {})
    if not isinstance(effects, dict):
        errors.append("effects 必须是对象")
    else:
        for name, cfg in effects.items():
            if not isinstance(cfg, dict):
                errors.append(f"效果 {name!r} 必须是对象")
                continue
            if "emitters" not in cfg:
                errors.append(f"效果 {name!r} 缺少 emitters")
            elif not isinstance(cfg["emitters"], list) or not cfg["emitters"]:
                errors.append(f"效果 {name!r} 的 emitters 必须是非空数组")
            for i, em in enumerate(cfg.get("emitters", []) if isinstance(cfg.get("emitters"), list) else []):
                if not isinstance(em, dict):
                    errors.append(f"效果 {name!r} emitter[{i}] 必须是对象")
                    continue
                for req in ("x", "y"):
                    if req not in em:
                        errors.append(f"效果 {name!r} emitter[{i}] 缺少 {req}")
                shape = em.get("shape", "point")
                if shape not in ("point", "sphere", "cone", "circle", "rectangle", "edge"):
                    errors.append(f"效果 {name!r} emitter[{i}] 未知形状: {shape!r}")
            for i, ff in enumerate(cfg.get("force_fields", []) if isinstance(cfg.get("force_fields"), list) else []):
                if not isinstance(ff, dict):
                    errors.append(f"效果 {name!r} force_field[{i}] 必须是对象")
                    continue
                if "type" not in ff:
                    errors.append(f"效果 {name!r} force_field[{i}] 缺少 type")
                elif ff["type"] not in ("gravity", "wind", "vortex", "turbulence",
                                        "noise", "attraction", "magnet"):
                    errors.append(f"效果 {name!r} force_field[{i}] 未知类型: {ff['type']!r}")
    return errors


class EffectsFactory:
    """
    声明式粒子效果工厂。
    
    用法：
      factory = EffectsFactory()
      
      # 列出所有可用效果
      print(factory.list_effects())
      
      # 按分类列出
      print(factory.list_effects('cosmic'))
      
      # 创建效果
      nebula = factory.create('nebula_burst', x=400, y=300)
      
      # 创建效果（批量预设参数）
      effects = factory.create_all('cosmic', x=400, y=300)
    """

    def __init__(self, validate: bool = True, strict: bool = False):
        """
        Args:
            validate : 是否在加载时校验 effects_config.json（v5.0 增强 ⑦）
            strict   : True = 校验失败抛 ValueError；False = 打印警告继续
        """
        self._config = _load_json('effects_config.json')
        self._palettes = _load_json('palettes.json')
        if validate:
            self._validate(strict)
        self._effects = self._config['effects']
        self._categories = self._config['categories']

    def _validate(self, strict: bool):
        """校验配置结构（jsonschema 可用时执行完整 Schema 校验）"""
        errors = _validate_config_structure(self._config)
        if HAS_JSONSCHEMA:
            schema = _load_json('effects_config.schema.json')
            validator = jsonschema.Draft7Validator(schema)
            for err in sorted(validator.iter_errors(self._config),
                              key=lambda e: list(e.path)):
                loc = "/".join(str(p) for p in err.path) or "(root)"
                errors.append(f"jsonschema[{loc}]: {err.message}")
        if errors:
            msg = "effects_config.json 校验失败:\n  " + "\n  ".join(errors[:20])
            if strict:
                raise ValueError(msg)
            print(f"[EffectsFactory] 警告: {msg}")

    # ============================================================
    #  查询
    # ============================================================

    def list_effects(self, category: str = None) -> List[str]:
        """列出可用效果名称"""
        if category:
            return list(self._categories.get(category, {}).get('effects', []))
        return list(self._effects.keys())

    def list_categories(self) -> List[Dict]:
        """列出所有分类"""
        return [
            {'id': k, 'label': v['label'], 'count': len(v['effects'])}
            for k, v in self._categories.items()
        ]

    def get_effect_info(self, name: str) -> Optional[Dict]:
        """获取效果配置信息"""
        return self._effects.get(name)

    # ============================================================
    #  创建
    # ============================================================

    def create(self, name: str, **params) -> Optional[ParticleSystem]:
        """
        创建单个效果。

        Args:
            name    : 效果名称（如 'nebula_burst'）
            **params: 运行时参数（x, y, width, height, direction, intensity）
        
        Returns:
            ParticleSystem 实例，或 None
        """
        cfg = self._effects.get(name)
        if cfg is None:
            print(f"[EffectsFactory] 未知效果: {name}")
            return None

        request_params = dict(params)

        # 解析调色板
        palette_name = cfg.get('palette')
        palette = self._resolve_palette(palette_name)

        # 创建系统
        max_particles = cfg.get('max_particles', 300)
        system = ParticleSystem(max_particles=max_particles)

        # 创建发射器
        for em_cfg in cfg.get('emitters', []):
            emitter = self._build_emitter(em_cfg, palette, request_params)
            if emitter:
                system.add_emitter(emitter)

        # 创建力场
        for ff_cfg in cfg.get('force_fields', []):
            ff = self._build_force_field(ff_cfg, request_params)
            if ff:
                system.add_force_field(ff)

        # 附加元信息
        system._effect_name = name
        system._effect_category = cfg.get('category', '')
        system._blend_mode = cfg.get('blend_mode', 'normal')
        system._bloom_recommended = cfg.get('bloom', False)

        return system

    def create_all(self, category: str, **params) -> Dict[str, ParticleSystem]:
        """批量创建某个分类下的全部效果"""
        return {
            name: self.create(name, **params)
            for name in self.list_effects(category)
        }

    # ============================================================
    #  内部构建
    # ============================================================

    def _resolve_palette(self, palette_ref: str) -> List[Tuple[int, int, int]]:
        """解析调色板引用 'category.name'"""
        if not palette_ref:
            return None
        if isinstance(palette_ref, list):
            return palette_ref

        parts = palette_ref.split('.')
        if len(parts) == 2:
            cat, name = parts
            return self._palettes.get(cat, {}).get(name)
        # 尝试在所有分类中查找
        for cat, palettes in self._palettes.items():
            if palette_ref in palettes:
                return palettes[palette_ref]
        return None

    def _resolve_coord(self, value: Any, params: dict, default: float = 0.0) -> float:
        """解析坐标值，支持变量替换"""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            v = value
            # 变量替换
            for k, val in params.items():
                v = v.replace(f'${k}', str(val))
            # 简单表达式求值 (+)
            try:
                return float(eval(v))
            except Exception:
                return default
        if isinstance(value, list):
            return random.uniform(value[0], value[1])
        return default

    def _resolve_special(self, value: str, params: dict) -> Any:
        """解析特殊值：切线方向、螺旋等"""
        if value == '$tangent':
            angle = params.get('angle', 0)
            return (math.cos(angle + math.pi / 2), math.sin(angle + math.pi / 2))
        if value == '$normal_outward':
            angle = params.get('angle', 0)
            return (math.cos(angle), math.sin(angle))
        if value == '$spiral_inward':
            # 这个由 vortex force field 处理
            return None
        if value == '$spiral_upward':
            return None
        if value == '$direction':
            dx = params.get('direction_x', 0)
            dy = params.get('direction_y', -1)
            return (dx, dy)
        return value

    def _build_emitter(self, cfg: dict, palette: list, params: dict) -> Optional[Emitter]:
        """从 JSON 配置构建 Emitter（映射到 Emitter 的实际 API）"""
        x = self._resolve_coord(cfg.get('x', 0), params)
        y = self._resolve_coord(cfg.get('y', 0), params)

        # 速度/生命/大小（随机从范围取单值，Emitter 不支持 range）
        def _rand(val, default=10.0):
            if isinstance(val, list):
                return random.uniform(val[0], val[1])
            return float(val) if val else default

        # 方向（角度，弧度）
        dir_cfg = cfg.get('direction')
        direction = 0.0
        if dir_cfg:
            resolved = self._resolve_special(dir_cfg, params)
            if resolved and isinstance(resolved, tuple):
                direction = math.atan2(resolved[1], resolved[0])
            elif isinstance(dir_cfg, list):
                direction = _rand(dir_cfg, 0)
            else:
                direction = 0.0

        # Burst 模式：rate 设为 0，另有 burst_count
        rate = cfg.get('rate', 0)
        if rate == 'burst':
            burst_count = cfg.get('burst_count', 100)
            rate = burst_count * 60  # 一次性大量发射

        emitter = Emitter(
            x=x, y=y,
            rate=rate,
            spread=cfg.get('spread', 0.5),
            direction=direction,
            speed=_rand(cfg.get('speed', 100)),
            life=_rand(cfg.get('life', 1.5)),
            size=_rand(cfg.get('size', 8)),
            color=(255, 255, 255),  # 稍后覆盖
        )

        # 旋转
        rot = cfg.get('rotation')
        if rot and isinstance(rot, list):
            emitter.rotation_range = (rot[0], rot[1])

        angular = cfg.get('angular_velocity')
        if angular and isinstance(angular, list):
            emitter.angular_velocity_range = (angular[0], angular[1])

        # 颜色处理
        color_ref = cfg.get('color')
        color_lt = cfg.get('color_over_lifetime')

        if isinstance(color_lt, str):
            ref_palette = self._resolve_palette(color_lt)
            if ref_palette:
                emitter.color_over_lifetime = gradient_to_curve(ref_palette)
            elif color_lt == 'fade_out':
                emitter.color = random.choice(palette) if palette else (255, 255, 255)
        elif palette:
            emitter.color = random.choice(palette)

        if color_ref and isinstance(color_ref, list):
            emitter.color = tuple(color_ref)

        # 大小曲线
        size_lt = cfg.get('size_over_lifetime')
        base_sz = emitter.size if emitter.size > 0 else 10
        if size_lt == 'fade_out':
            emitter.size_over_lifetime = size_curve_fadeout(base_sz)
        elif isinstance(size_lt, str) and ':' in size_lt:
            parts = size_lt.split(':')
            if parts[0] == 'pulse':
                emitter.size_over_lifetime = size_curve_pulse(
                    base_sz,
                    peak_at=0.3,
                    peak_mult=float(parts[1]) if len(parts) > 1 else 1.5
                )

        # 速度曲线
        vel_lt = cfg.get('velocity_over_lifetime')
        if vel_lt == 'accelerate':
            emitter.velocity_over_lifetime = velocity_curve_accelerate()
        elif vel_lt == 'decelerate':
            emitter.velocity_over_lifetime = velocity_curve_decelerate()

        # 发射形状
        shape_cfg = cfg.get('shape', 'point')
        if shape_cfg != 'point':
            shape = EmitterShape()
            shape.shape_type = shape_cfg
            if shape_cfg == 'sphere':
                shape.radius = cfg.get('shape_radius', 30)
            elif shape_cfg == 'circle':
                shape.radius = cfg.get('shape_radius', 50)
                # shape_thickness > 0 时从圆环边缘发射
                thickness = cfg.get('shape_thickness', 0)
                shape.emit_from_shell = thickness > 0
            elif shape_cfg == 'cone':
                shape.radius = cfg.get('shape_radius', 80)
                # shape_angle 是圆锥开口角度，据此计算圆锥高度
                cone_angle = cfg.get('shape_angle', 1.0)
                half_angle = max(cone_angle * 0.5, 0.01)
                shape.height = shape.radius / math.tan(half_angle)
            elif shape_cfg == 'rectangle':
                shape.width = cfg.get('shape_width', 100)
                shape.rect_height = cfg.get('shape_height', 100)
            elif shape_cfg == 'edge':
                # JSON 提供两个端点 (x1,y1)-(x2,y2)，转换为 EmitterShape 的
                # edge_length + edge_angle 模型（以发射器位置为中心）
                x1 = self._resolve_coord(cfg.get('shape_x1', 0), params)
                y1 = self._resolve_coord(cfg.get('shape_y1', 0), params)
                x2 = self._resolve_coord(cfg.get('shape_x2', 100), params)
                y2 = self._resolve_coord(cfg.get('shape_y2', 100), params)
                dx = x2 - x1
                dy = y2 - y1
                shape.edge_length = math.sqrt(dx * dx + dy * dy)
                shape.edge_angle = math.atan2(dy, dx)
            emitter.shape = shape

        # 拖尾
        if cfg.get('trail_enabled'):
            emitter.trail_enabled = True
            emitter.trail_length = cfg.get('trail_length', 10)

        # 发光
        if cfg.get('emissive'):
            emissive_color = cfg.get('emissive_color', [255, 255, 255])
            emitter.emissive = tuple(emissive_color)
            emitter.emissive_strength = cfg.get('emissive_strength', 1.0)

        return emitter

    def _build_force_field(self, cfg: dict, params: dict) -> Optional[ForceField]:
        """从 JSON 配置构建 ForceField"""
        ftype = cfg.get('type', 'gravity')
        strength = cfg.get('strength', 10.0)
        x = self._resolve_coord(cfg.get('x', 0), params)
        y = self._resolve_coord(cfg.get('y', 0), params)

        if ftype == 'gravity':
            ff = ForceField(field_type='gravity', strength=strength)
        elif ftype == 'wind':
            ff = ForceField(field_type='wind', strength=strength)
        elif ftype == 'vortex':
            ff = create_vortex(
                x=x, y=y,
                strength=strength,
            )
        elif ftype == 'turbulence':
            ff = create_turbulence(
                strength=strength,
                scale=cfg.get('scale', 50)
            )
        elif ftype == 'noise':
            ff = ForceField(
                field_type='noise',
                strength=strength,
                noise_strength=strength,
                noise_scale=cfg.get('scale', 50)
            )
        elif ftype == 'attraction':
            ff = ForceField(
                field_type='attraction',
                x=x, y=y,
                strength=strength,
            )
        elif ftype == 'magnet':
            ff = create_magnet(
                x=x, y=y,
                strength=strength,
                range_=cfg.get('radius', 200)
            )
        else:
            return None

        return ff


# ============================================================
#  便捷函数
# ============================================================

_factory_instance = None

def get_factory() -> EffectsFactory:
    """获取全局工厂单例"""
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = EffectsFactory()
    return _factory_instance


def create_effect(name: str, **params) -> Optional[ParticleSystem]:
    """便捷创建函数"""
    return get_factory().create(name, **params)


def list_all_effects() -> List[str]:
    return get_factory().list_effects()

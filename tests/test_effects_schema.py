"""
JSON Schema 校验测试（v5.0 增强 ⑦）
"""
import json
import pathlib

import pytest

from components.effects_factory import (
    EffectsFactory, _validate_config_structure, _load_json, HAS_JSONSCHEMA,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent


class TestSchemaFile:
    def test_schema_exists_and_valid_json(self):
        p = ROOT / "components" / "effects_config.schema.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["type"] == "object"
        assert "definitions" in data
        assert "emitter" in data["definitions"]
        assert "force_field" in data["definitions"]

    def test_schema_requires_emitters(self):
        p = ROOT / "components" / "effects_config.schema.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        effect = data["definitions"]["effect"]
        assert "emitters" in effect.get("required", [])


class TestBuiltinValidator:
    def test_current_config_passes(self):
        """现有 30 个效果的配置必须通过内置校验"""
        cfg = _load_json("effects_config.json")
        errors = _validate_config_structure(cfg)
        assert errors == [], "\n".join(errors)

    def test_missing_emitters_detected(self):
        cfg = {"categories": {}, "effects": {"bad": {"category": "x"}}}
        errors = _validate_config_structure(cfg)
        assert any("缺少 emitters" in e for e in errors)

    def test_bad_shape_detected(self):
        cfg = {"categories": {},
               "effects": {"bad": {"emitters": [{"x": 0, "y": 0, "shape": "hexagon"}]}}}
        errors = _validate_config_structure(cfg)
        assert any("未知形状" in e for e in errors)

    def test_bad_force_field_type_detected(self):
        cfg = {"categories": {},
               "effects": {"bad": {"emitters": [{"x": 0, "y": 0}],
                                   "force_fields": [{"type": "teleport"}]}}}
        errors = _validate_config_structure(cfg)
        assert any("未知类型" in e for e in errors)

    def test_missing_root_keys_detected(self):
        assert _validate_config_structure({}) != []
        assert _validate_config_structure([]) == ["配置根节点必须是对象"]


class TestFactoryValidation:
    def test_factory_loads_with_validation(self):
        f = EffectsFactory(validate=True)
        assert len(f.list_effects()) == 31

    def test_strict_mode_rejects_broken_config(self, monkeypatch):
        # 直接替换配置加载函数：坏配置 → strict 模式必须抛 ValueError
        import components.effects_factory as ef
        broken = {"categories": {}, "effects": {"bad": {}}}
        real_palettes = ef._load_json("palettes.json")

        def fake_load(filename):
            if filename.endswith("effects_config.json"):
                return broken
            return real_palettes
        monkeypatch.setattr(ef, "_load_json", fake_load)
        with pytest.raises(ValueError):
            EffectsFactory(validate=True, strict=True)

    def test_non_strict_prints_warning(self, capsys):
        # 校验通过时不应有任何警告输出
        EffectsFactory(validate=True)
        captured = capsys.readouterr()
        assert "警告" not in captured.out

    def test_singleton_still_works(self):
        from components.effects_factory import get_factory
        f1 = get_factory()
        f2 = get_factory()
        assert f1 is f2
        assert f1.list_effects()

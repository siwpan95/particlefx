"""场景配方与种子持久化测试 —— v1.2 preset 模块 + EffectsFactory 可复现。"""

import json
import os
import tempfile

from particlefx import ParticleSystem, Emitter, ForceField, NoiseField
from particlefx.components.preset import (
    system_to_recipe, recipe_to_system, save_preset, load_preset,
    export_effect_recipe,
)
from particlefx.components.effects_factory import EffectsFactory


def _snapshot(ps):
    return [
        (round(p.x, 9), round(p.y, 9), round(p.vx, 9), round(p.vy, 9))
        for p in ps.particles
    ]


def _build_seeded_system():
    ps = ParticleSystem(max_particles=2000, seed=12345)
    ps.add_emitter(Emitter(x=400, y=300, rate=120.0, speed=140.0,
                           spread=0.8, life=2.0, size=3.0))
    ps.add_force_field(ForceField(field_type='noise', noise_strength=3.0,
                                  noise_scale=40.0))
    ps.add_noise_field(NoiseField(strength=1.5, scale=60.0))
    return ps


def test_system_to_recipe_reproducible_roundtrip():
    """导出配方 -> 重建 -> 同种子逐位复现（从干净状态推进，因累计时间不在配方内）。"""
    ps = _build_seeded_system()
    recipe = system_to_recipe(ps, name='demo')
    # 派生种子不落地（应为 None），顶层保存系统种子
    assert recipe['seed'] == 12345
    assert recipe['emitters'][0]['seed'] is None
    assert recipe['force_fields'][0]['seed'] is None
    assert recipe['noise_fields'][0]['seed'] is None

    rebuilt = recipe_to_system(recipe)
    # 两者都从干净状态推进相同序列 -> 逐位一致
    for _ in range(20):
        ps.update(1 / 60)
        rebuilt.update(1 / 60)
    assert _snapshot(rebuilt) == _snapshot(ps)


def test_preset_file_roundtrip():
    """save_preset -> load_preset -> 重建 可复现（从干净状态推进）。"""
    ps = _build_seeded_system()
    recipe = system_to_recipe(ps)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'preset.json')
        save_preset(recipe, path)
        assert os.path.exists(path)
        loaded = load_preset(path)
        assert loaded['seed'] == 12345
        rebuilt = recipe_to_system(loaded)
        for _ in range(15):
            ps.update(1 / 60)
            rebuilt.update(1 / 60)
        assert _snapshot(rebuilt) == _snapshot(ps)


def test_unseeded_recipe_does_not_force_determinism():
    """未播种系统的配方重建后不应保证一致（保留旧行为）。"""
    ps = ParticleSystem(max_particles=500)
    ps.add_emitter(Emitter(x=0, y=0, rate=80.0, speed=50.0, spread=1.0, life=1.0))
    recipe = system_to_recipe(ps)
    assert recipe['seed'] is None


def test_factory_create_is_reproducible_with_seed():
    """EffectsFactory.create 带 seed -> 两次创建推进逐位一致。"""
    fx = EffectsFactory(validate=False)

    def run():
        sys = fx.create('arcane_blast', x=400, y=300, seed=777)
        for _ in range(20):
            sys.update(1 / 60)
        return _snapshot(sys)

    assert run() == run()


def test_factory_seed_from_config_default():
    """效果配置里的默认 seed 也能让 create 可复现。"""
    fx = EffectsFactory(validate=False)
    # 临时给一个效果注入默认 seed，验证从配置读取
    fx._effects['arcane_blast'] = dict(fx._effects['arcane_blast'])
    fx._effects['arcane_blast']['seed'] = 555
    a = fx.create('arcane_blast', x=400, y=300)
    b = fx.create('arcane_blast', x=400, y=300)
    for _ in range(20):
        a.update(1 / 60); b.update(1 / 60)
    assert _snapshot(a) == _snapshot(b)


def test_export_effect_recipe_has_seed():
    """export_effect_recipe 导出带种子的配方。"""
    fx = EffectsFactory(validate=False)
    recipe = export_effect_recipe(fx, 'arcane_blast', seed=999, x=400, y=300)
    assert recipe['seed'] == 999
    assert 'emitters' in recipe and recipe['emitters']
    # 重建可复现
    rebuilt = recipe_to_system(recipe)
    orig = fx.create('arcane_blast', x=400, y=300, seed=999)
    for _ in range(15):
        orig.update(1 / 60); rebuilt.update(1 / 60)
    assert _snapshot(rebuilt) == _snapshot(orig)

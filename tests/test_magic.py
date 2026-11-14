"""魔法效果系统测试 —— v1.2 MagicEffectsManager（确定性 / 生命周期 / 接口）。"""

from particlefx.systems.magic_effects import MagicEffectsManager


def _snapshot(mgr):
    return [
        (round(p.x, 9), round(p.y, 9), round(p.vx, 9), round(p.vy, 9))
        for p in mgr.particles
    ]


def _full_run(seed):
    m = MagicEffectsManager(width=800, height=600, seed=seed)
    m.cast('arcane_burst', x=400, y=300)
    m.cast('rune_ring', x=400, y=300)
    m.cast('healing', x=400, y=300)
    m.cast('void_tendril', x=400, y=300)
    m.cast('magic_missile', x=400, y=300, tx=500, ty=100)
    for _ in range(120):
        m.update(1 / 60)
    return _snapshot(m)


def test_all_presets_listed():
    presets = MagicEffectsManager(width=1).list_presets()
    assert set(presets.keys()) == {
        'arcane_burst', 'rune_ring', 'healing', 'void_tendril', 'magic_missile'
    }


def test_cast_creates_particles():
    m = MagicEffectsManager()
    assert m.particle_count == 0
    m.cast('arcane_burst', x=400, y=300)
    assert m.particle_count > 0
    # update 推进后粒子最终衰减清空（爆发类）
    for _ in range(200):
        m.update(1 / 60)
    assert m.particle_count == 0


def test_unknown_effect_raises():
    import pytest
    with pytest.raises(ValueError):
        MagicEffectsManager().cast('nonexistent', 0, 0)


def test_seed_reproducible():
    """相同 seed + 相同施法序列 => 逐位一致。"""
    a = _full_run(42)
    b = _full_run(42)
    assert a == b
    assert len(a) > 0


def test_different_seeds_differ():
    assert _full_run(1) != _full_run(2)


def test_factory_lists_magic_recipes():
    """EffectsFactory 已具备魔法类声明式配方（README 原 '待创建' 已落地）。"""
    from particlefx.components.effects_factory import EffectsFactory
    fx = EffectsFactory(validate=False)
    magic = fx.list_effects('magic')
    assert 'arcane_blast' in magic
    # 能经 JSON 配置声明式创建
    sys = fx.create('arcane_blast', x=400, y=300)
    assert sys is not None
    sys.update(1 / 60)
    assert sys.get_particle_count() > 0

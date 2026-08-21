"""
EffectsFactory 单元测试

覆盖：JSON 加载、效果创建、EmitterShape 属性映射、
      调色板解析、力场构建、edge 端点转换。
"""

import math
import pytest
from components.effects_factory import EffectsFactory, create_effect, get_factory
from core.particle import EmitterShape


@pytest.fixture
def factory():
    return EffectsFactory()


class TestFactoryBasics:
    def test_list_effects(self, factory):
        effects = factory.list_effects()
        assert len(effects) > 0

    def test_list_categories(self, factory):
        cats = factory.list_categories()
        assert len(cats) > 0
        assert all('id' in c and 'label' in c and 'count' in c for c in cats)

    def test_get_effect_info(self, factory):
        effects = factory.list_effects()
        info = factory.get_effect_info(effects[0])
        assert info is not None
        assert 'emitters' in info

    def test_unknown_effect(self, factory):
        result = factory.create('nonexistent_effect')
        assert result is None


class TestCreateEffect:
    def test_create_returns_system(self, factory):
        effects = factory.list_effects()
        system = factory.create(effects[0], x=400, y=300)
        assert system is not None
        assert hasattr(system, 'update')
        assert hasattr(system, 'render')

    def test_create_all_in_category(self, factory):
        cats = factory.list_categories()
        if cats:
            results = factory.create_all(cats[0]['id'], x=400, y=300)
            assert len(results) > 0
            for name, sys in results.items():
                assert sys is not None

    def test_create_with_params(self, factory):
        """测试参数替换 $x, $y 等"""
        system = factory.create(factory.list_effects()[0],
                                x=250, y=350, width=800, height=600)
        assert system is not None


class TestEmitterShapeMapping:
    """验证 effects_factory 正确映射 JSON shape 配置到 EmitterShape 属性"""

    def test_sphere_shape_uses_radius(self, factory):
        """sphere_radius 应映射到 EmitterShape.radius"""
        # 找一个使用 sphere 形状的效果
        for name in factory.list_effects():
            cfg = factory.get_effect_info(name)
            for em_cfg in cfg.get('emitters', []):
                if em_cfg.get('shape') == 'sphere':
                    system = factory.create(name, x=400, y=300)
                    assert system is not None
                    # 检查发射器形状
                    for emitter in system.emitters:
                        if emitter.shape.shape_type == 'sphere':
                            expected = em_cfg.get('shape_radius', 30)
                            assert emitter.shape.radius == expected
                    return
        pytest.skip("没有使用 sphere 形状的效果")

    def test_circle_shape_uses_radius(self, factory):
        """circle_radius 应映射到 EmitterShape.radius"""
        for name in factory.list_effects():
            cfg = factory.get_effect_info(name)
            for em_cfg in cfg.get('emitters', []):
                if em_cfg.get('shape') == 'circle':
                    system = factory.create(name, x=400, y=300)
                    for emitter in system.emitters:
                        if emitter.shape.shape_type == 'circle':
                            expected = em_cfg.get('shape_radius', 50)
                            assert emitter.shape.radius == expected
                    return
        pytest.skip("没有使用 circle 形状的效果")

    def test_rectangle_shape_uses_width_and_height(self, factory):
        """rect_width 应映射到 EmitterShape.width, shape_height 到 rect_height"""
        for name in factory.list_effects():
            cfg = factory.get_effect_info(name)
            for em_cfg in cfg.get('emitters', []):
                if em_cfg.get('shape') == 'rectangle':
                    system = factory.create(name, x=400, y=300)
                    for emitter in system.emitters:
                        if emitter.shape.shape_type == 'rectangle':
                            assert emitter.shape.width == em_cfg.get('shape_width', 100)
                            assert emitter.shape.rect_height == em_cfg.get('shape_height', 100)
                    return
        pytest.skip("没有使用 rectangle 形状的效果")

    def test_edge_shape_converts_endpoints(self, factory):
        """edge 端点 (x1,y1)-(x2,y2) 应转换为 edge_length + edge_angle"""
        for name in factory.list_effects():
            cfg = factory.get_effect_info(name)
            for em_cfg in cfg.get('emitters', []):
                if em_cfg.get('shape') == 'edge':
                    system = factory.create(name, x=400, y=300, width=800, height=600)
                    for emitter in system.emitters:
                        if emitter.shape.shape_type == 'edge':
                            # edge_length 应该 > 0
                            assert emitter.shape.edge_length > 0
                            # edge_angle 应该是有效角度
                            assert -math.pi <= emitter.shape.edge_angle <= math.pi
                    return
        pytest.skip("没有使用 edge 形状的效果")

    def test_cone_shape_uses_radius(self, factory):
        """cone_radius 应映射到 EmitterShape.radius"""
        for name in factory.list_effects():
            cfg = factory.get_effect_info(name)
            for em_cfg in cfg.get('emitters', []):
                if em_cfg.get('shape') == 'cone':
                    system = factory.create(name, x=400, y=300)
                    for emitter in system.emitters:
                        if emitter.shape.shape_type == 'cone':
                            expected = em_cfg.get('shape_radius', 80)
                            assert emitter.shape.radius == expected
                            # height 应该根据 angle 计算出来
                            assert emitter.shape.height > 0
                    return
        pytest.skip("没有使用 cone 形状的效果")

    def test_no_unused_attributes(self, factory):
        """确保工厂没有设置 EmitterShape 上不存在的属性"""
        for name in factory.list_effects():
            system = factory.create(name, x=400, y=300)
            if system is None:
                continue
            for emitter in system.emitters:
                shape = emitter.shape
                # EmitterShape dataclass 的合法字段
                valid_fields = {
                    'shape_type', 'radius', 'height', 'width',
                    'rect_height', 'edge_length', 'edge_angle',
                    'emit_from_shell', 'emit_from_volume',
                    'randomize_direction',
                }
                for attr in vars(shape):
                    assert attr in valid_fields, \
                        f"效果 {name} 的 EmitterShape 有非法属性: {attr}"


class TestSingleton:
    def test_get_factory_singleton(self):
        f1 = get_factory()
        f2 = get_factory()
        assert f1 is f2

    def test_create_effect_convenience(self):
        effects = get_factory().list_effects()
        system = create_effect(effects[0], x=100, y=100)
        assert system is not None

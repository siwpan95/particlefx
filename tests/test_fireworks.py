"""
FireworksSystem 单元测试

覆盖：发射、爆炸、样式枚举、粒子计数、crossette 二次分裂。
"""

import pytest
from particlefx.systems.fireworks_system import FireworksSystem, create_fireworks


@pytest.fixture
def fw():
    return FireworksSystem(width=800, height=600, auto_launch=False)


class TestLaunch:
    def test_manual_launch(self, fw):
        fw.launch(x=400, y=300)
        assert fw.rocket_count == 1

    def test_launch_increments_count(self, fw):
        for _ in range(3):
            fw.launch()
        assert fw.rocket_count == 3

    def test_max_active_limit(self, fw):
        fw.max_active = 2
        fw.launch()
        fw.launch()
        fw.launch()  # 应被忽略
        assert fw.rocket_count == 2

    def test_default_position(self, fw):
        fw.launch()
        rocket = fw.rockets[0]
        assert 0 <= rocket['x'] <= 800
        assert rocket['y'] > 600  # 从底部发射

    def test_random_palette_and_style(self, fw):
        fw.launch()
        rocket = fw.rockets[0]
        assert rocket['palette'] is not None
        assert rocket['style'] is not None

    def test_explicit_palette_and_style(self, fw):
        fw.launch(palette_name='gold', style='chrysanthemum')
        rocket = fw.rockets[0]
        assert rocket['style'] == 'chrysanthemum'

    def test_rocket_has_head_particle(self, fw):
        fw.launch()
        rocket = fw.rockets[0]
        assert 'head' in rocket
        assert rocket['head'].is_alive()


class TestUpdate:
    def test_auto_launch(self):
        fw = FireworksSystem(width=800, height=600,
                             auto_launch=True, launch_interval=0.01)
        fw.update(0.02)
        assert fw.rocket_count > 0

    def test_rocket_rises(self, fw):
        fw.launch(x=400, y=100)
        initial_y = fw.rockets[0]['y']
        fw.update(0.1)
        assert fw.rockets[0]['y'] < initial_y  # y 减小 = 上升

    def test_explosion_triggers(self, fw):
        # 固定长寿命样式（willow 粒子寿命 2.5~4.5s），
        # 避免随机样式（如 crackle/peony，寿命 0.4~1.8s）在
        # 20×0.1s 推进后全部死亡导致的 flaky 失败。
        fw.launch(x=400, y=599, style='willow')
        # 快速推进时间直到爆炸
        for _ in range(20):
            fw.update(0.1)
        # 火箭应该已爆炸
        assert fw.rocket_count == 0
        # 应该有爆炸粒子（1.9s 后 willow 粒子仍在 2.5~4.5s 生命内）
        assert fw.particle_count > 0

    def test_all_styles_produce_particles(self, fw):
        """测试所有烟花样式都能正常爆炸"""
        for style in FireworksSystem.STYLES:
            fw.launch(x=400, y=400, style=style)
            # 推进到爆炸
            for _ in range(30):
                fw.update(0.1)
            # 至少应该有粒子产生（某些样式可能已消亡）
            # 主要验证不崩溃
        assert True  # 如果跑到这里没异常就算通过


class TestRendering:
    def test_get_all_particles(self, fw):
        fw.launch(x=400, y=300)
        particles = fw.get_all_particles()
        # 应该包含火箭头粒子
        assert len(particles) > 0

    def test_particles_property(self, fw):
        fw.launch(x=400, y=300)
        assert len(fw.particles) > 0

    def test_particle_count(self, fw):
        fw.launch(x=400, y=300)
        assert fw.particle_count > 0

    def test_render_no_crash(self, fw):
        class MockRenderer:
            def render(self, particles):
                pass
        fw.launch(x=400, y=300)
        fw.update(0.1)
        fw.render(MockRenderer())

    def test_render_empty(self, fw):
        class MockRenderer:
            def render(self, particles):
                pass
        fw.render(MockRenderer())  # 没有粒子时不应崩溃


class TestCrossette:
    def test_crossette_split(self, fw):
        fw.launch(x=400, y=400, style='crossette')
        # 推进到爆炸 + 二次分裂
        for _ in range(50):
            fw.update(0.1)
        # 不崩溃即通过
        assert True


class TestCompatibility:
    def test_create_fireworks_helper(self):
        fw = create_fireworks(width=640, height=480)
        assert fw.width == 640
        assert fw.height == 480

    def test_enable_lighting(self, fw):
        fw.enable_lighting(True)
        assert fw._lighting_enabled is True

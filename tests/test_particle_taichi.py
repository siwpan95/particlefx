"""
Taichi SPH 后端测试（v5.0 增强 ⑧）

taichi 未安装时：验证零开销降级（清晰报错 + 标志位 + 模块可编译）；
taichi 可用时：完整功能测试（添加粒子 / 模拟 / 导出 / 渲染）。
"""
import math

import pytest

import particlefx.core.particle_taichi as pta


class TestFallbackWithoutTaichi:
    def test_flag(self):
        if pta.HAS_TAICHI:
            pytest.skip("taichi 已安装")
        assert pta.ti is None

    def test_constructor_raises_helpful_error(self):
        if pta.HAS_TAICHI:
            pytest.skip("taichi 已安装")
        with pytest.raises(ImportError) as e:
            pta.TaichiSPHSystem()
        assert "pip install taichi" in str(e.value)

    def test_module_compiles(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parent.parent
               / "particlefx" / "core" / "particle_taichi.py").read_text(encoding="utf-8")
        compile(src, "core/particle_taichi.py", "exec")


class TestKernelMath:
    def test_poly6_cpu_reference(self):
        """Poly6 核 CPU 参考实现：支撑半径内非零、边缘连续、归一近似"""
        h = 1.0
        assert pta.poly6_kernel(0.0, h) > 0
        assert pta.poly6_kernel(h * h, h) == 0.0       # 边界处为 0
        assert pta.poly6_kernel(h * h + 1e-6, h) == 0.0
        # 单调递减
        assert pta.poly6_kernel(0.0, h) > pta.poly6_kernel(0.5 * h * h, h)


@pytest.mark.skipif(not pta.HAS_TAICHI, reason="taichi 未安装")
class TestTaichiSPHSystem:
    def test_init_and_add_particles(self):
        sph = pta.TaichiSPHSystem(max_particles=512, width=200, height=200)
        added = sph.add_particles(100, 50, count=10)
        assert added == 10
        assert sph.particle_count == 10

    def test_add_fluid_block(self):
        sph = pta.TaichiSPHSystem(max_particles=2048, width=200, height=200)
        n = sph.add_fluid_block(60, 30, 40, 40, spacing=8.0)
        assert 20 <= n <= 40
        assert sph.particle_count == n

    def test_update_moves_particles_downward(self):
        sph = pta.TaichiSPHSystem(max_particles=512, width=200, height=200,
                                  gravity=(0.0, 300.0))
        sph.add_fluid_block(90, 20, 30, 30, spacing=6.0)
        y_before = sph.positions_numpy()[:, 1].mean()
        for _ in range(30):
            sph.update(1 / 60)
        y_after = sph.positions_numpy()[:, 1].mean()
        assert y_after > y_before  # 重力作用下整体下移

    def test_boundary_contains_particles(self):
        sph = pta.TaichiSPHSystem(max_particles=512, width=200, height=200)
        sph.add_particles(100, 100, vy=500.0, count=20)
        for _ in range(120):
            sph.update(1 / 60)
        pos = sph.positions_numpy()
        assert pos[:, 0].min() >= 0.0
        assert pos[:, 1].min() >= 0.0
        assert pos[:, 0].max() <= 200.0
        assert pos[:, 1].max() <= 200.0

    def test_render_to_image(self):
        sph = pta.TaichiSPHSystem(max_particles=512, width=128, height=96)
        sph.add_particles(64, 48, count=30)
        sph.update(1 / 60)
        img = sph.render_to_image(width=128, height=96)
        assert img.size == (128, 96)
        assert img.mode == "RGB"

    def test_density_pressure_plausible(self):
        sph = pta.TaichiSPHSystem(max_particles=512, width=200, height=200)
        sph.add_fluid_block(90, 90, 30, 30, spacing=6.0)
        sph.update(1 / 60)
        d = sph.density.to_numpy()[:sph.particle_count]
        assert (d > 0).all()  # 所有存活粒子密度为正

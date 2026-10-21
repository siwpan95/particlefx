"""
ModernGL compute 全 GPU 模拟测试（v5.0 增强 ⑨）

无 GPU 环境：源码结构验证（着色器存在 / API 存在 / 语法可编译）；
moderngl 可用时：构造 + 发射 + 更新 + 回读全链路。
"""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = (ROOT / "particlefx" / "renderers" / "particle_sim_gpu.py").read_text(encoding="utf-8")


class TestSourceStructure:
    def test_compute_shader_declared(self):
        assert "CS_PARTICLE_UPDATE" in SRC
        assert "#version 430 core" in SRC
        assert "layout(local_size_x = 256) in" in SRC
        assert "gl_GlobalInvocationID" in SRC

    def test_shader_has_expected_bindings(self):
        for binding in range(7):
            assert f"binding = {binding}" in SRC

    def test_api_present(self):
        assert "class GPUParticleSimulator" in SRC
        for method in ("emit_burst", "update", "read_particles",
                       "to_particle_tuples", "release"):
            assert f"def {method}" in SRC

    def test_module_compiles(self):
        compile(SRC, "renderers/particle_sim_gpu.py", "exec")

    def test_fallback_flag(self):
        import particlefx.renderers.particle_sim_gpu as psg
        if not psg.HAS_MODERNGL:
            assert psg.moderngl is None
            with pytest.raises(ImportError) as e:
                psg.GPUParticleSimulator()
            assert "pip install moderngl" in str(e.value)


@pytest.mark.skipif(not __import__("importlib").util.find_spec("moderngl"),
                    reason="moderngl 未安装")
class TestGPUParticleSimulator:
    def _make(self, **kw):
        import moderngl
        from particlefx.renderers.particle_sim_gpu import GPUParticleSimulator
        ctx = moderngl.create_context(standalone=True, require=430)
        return GPUParticleSimulator(ctx=ctx, **kw)

    def test_emit_and_count(self):
        sim = self._make(max_particles=1024)
        n = sim.emit_burst(x=100, y=100, count=100)
        assert n == 100
        assert sim.particle_count == 100

    def test_update_moves_downward(self):
        sim = self._make(max_particles=1024, gravity=500.0)
        sim.emit_burst(x=100, y=50, count=64,
                       speed_range=(0.0, 0.0), life_range=(5.0, 5.0))
        pos0, _, _ = sim.read_particles()
        y0 = pos0[:, 1].mean()
        for _ in range(30):
            sim.update(1 / 60)
        pos1, _, _ = sim.read_particles()
        y1 = pos1[:, 1].mean()
        assert y1 > y0  # 重力下落

    def test_lifetime_kills_particles(self):
        sim = self._make(max_particles=512)
        sim.emit_burst(x=100, y=100, count=200,
                       life_range=(0.1, 0.2))
        for _ in range(60):
            sim.update(1 / 60)
        assert sim.particle_count == 0

    def test_to_particle_tuples_format(self):
        sim = self._make(max_particles=512)
        sim.emit_burst(x=100, y=100, count=10, life_range=(2.0, 2.0))
        sim.update(1 / 60)
        tuples = sim.to_particle_tuples()
        assert len(tuples) == 10
        for t in tuples:
            assert len(t) == 7  # (x, y, size, r, g, b, alpha)
            assert 0.0 <= t[6] <= 1.0

    def test_attractor_pulls(self):
        sim = self._make(max_particles=512, gravity=0.0,
                         attractor=(0, 100, 800.0, 400.0))
        sim.emit_burst(x=100, y=100, count=32,
                       speed_range=(0.0, 0.0), life_range=(5.0, 5.0))
        for _ in range(30):
            sim.update(1 / 60)
        pos, _, _ = sim.read_particles()
        dx = pos[:, 0]  # 粒子 x=100 处，吸引子在 x=0
        assert dx.mean() < 100.0  # 被吸引向左移动

    def test_release(self):
        sim = self._make(max_particles=64)
        sim.emit_burst(x=0, y=0, count=8)
        sim.release()  # 不崩溃

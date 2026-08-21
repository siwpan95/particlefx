"""
软粒子深度淡出测试（v5.0 增强 ③）

无 GPU 环境下验证：
- 着色器变体与 uniform 声明的存在性
- 渲染器接口（构造参数 / set_scene_depth）的存在性
- 模块语法可编译
moderngl 可用时额外验证构造参数注入。
"""
import inspect
import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GPUFILE = ROOT / "renderers" / "particle_renderer_moderngl.py"
UNIFIEDFILE = ROOT / "renderers" / "fluid_renderer_modernGL.py"


def _src(path):
    return path.read_text(encoding="utf-8")


class TestDepthFadeShaders:
    def test_gpu_renderer_has_depth_shader_variant(self):
        src = _src(GPUFILE)
        assert "FS_PARTICLE_SOFT_DEPTH" in src
        assert "uniform sampler2D u_scene_depth" in src
        assert "u_depth_fade_range" in src
        assert "u_screen_size" in src
        # 软粒子核心：场景深度与粒子深度差值驱动淡出
        assert "smoothstep(0.0, u_depth_fade_range, depth_diff)" in src

    def test_unified_renderer_shader_has_optional_depth_fade(self):
        src = _src(UNIFIEDFILE)
        assert "u_depth_fade_enabled" in src
        assert "u_scene_depth" in src
        # 关闭时零行为变化（uniform 分支）
        assert "if (u_depth_fade_enabled)" in src

    def test_gpu_renderer_api(self):
        src = _src(GPUFILE)
        assert "def set_scene_depth(self" in src
        assert "def _bind_depth_fade(self" in src
        assert "enable_depth_fade: bool = False" in src
        assert "depth_fade_range: float = 0.05" in src

    def test_unified_renderer_api(self):
        src = _src(UNIFIEDFILE)
        assert "def set_scene_depth(self" in src
        assert "def enable_depth_fade_with_fluid_depth(self" in src
        assert "def _bind_depth_fade(self" in src
        assert "enable_depth_fade: bool = False" in src

    def test_modules_compile(self):
        compile(_src(GPUFILE), str(GPUFILE), "exec")
        compile(_src(UNIFIEDFILE), str(UNIFIEDFILE), "exec")


@pytest.mark.skipif(not __import__("importlib").util.find_spec("moderngl"),
                    reason="moderngl 未安装")
class TestDepthFadeWithModernGL:
    def test_gpu_renderer_signature(self):
        from renderers.particle_renderer_moderngl import GPUParticleRenderer
        sig = inspect.signature(GPUParticleRenderer.__init__)
        assert "enable_depth_fade" in sig.parameters
        assert "depth_fade_range" in sig.parameters

    def test_unified_renderer_signature(self):
        from renderers.fluid_renderer_modernGL import UnifiedGPURenderer
        sig = inspect.signature(UnifiedGPURenderer.__init__)
        assert "enable_depth_fade" in sig.parameters
        assert "depth_fade_range" in sig.parameters

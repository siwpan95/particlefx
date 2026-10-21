"""
pygbag Web 导出入口测试（v5.0 增强 ④）

验证：
- web_main.py 遵循 pygbag 约定（async main + 每帧 asyncio.sleep(0)）
- 模块可导入且 main 为协程函数
- 渲染器 API 调用与 PygameRenderer 实际签名一致
"""
import asyncio
import importlib.util
import inspect
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_module():
    path = ROOT / "examples" / "web_main.py"
    spec = importlib.util.spec_from_file_location("web_main_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


class TestWebMain:
    def test_source_follows_pygbag_convention(self):
        _, path = _load_module()
        src = path.read_text(encoding="utf-8")
        assert "async def main" in src
        assert "await asyncio.sleep(0)" in src
        assert "asyncio.run(main())" in src

    def test_main_is_coroutine_function(self):
        mod, _ = _load_module()
        assert inspect.iscoroutinefunction(mod.main)

    def test_renderer_kwargs_match_api(self):
        """web_main.py 使用的 PygameRenderer 参数必须真实存在"""
        pygame = pytest.importorskip("pygame")
        from particlefx.renderers.pygame_renderer import PygameRenderer
        sig = inspect.signature(PygameRenderer.__init__)
        for kw in ("blend_mode", "bloom_enabled", "bloom_intensity"):
            assert kw in sig.parameters, f"PygameRenderer 缺少参数 {kw}"

    def test_imports_resolve(self):
        mod, _ = _load_module()
        assert mod.FireworksSystem is not None
        assert mod.PygameRenderer is not None

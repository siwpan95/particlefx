"""
pytest 配置 — 将项目根目录加入 sys.path，
使 `from particlefx.core.particle import ...` 等导入在测试中直接可用。
"""
import os, sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

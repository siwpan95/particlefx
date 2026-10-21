# Changelog

本项目所有值得注意的变更都会记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.0.0] - 2026-10-21

### Added

- **发布至 PyPI**：发行名 `particlefx`，导入名 `particlefx`（无连字符差异）。
  - <https://pypi.org/project/particlefx/>
  - 安装：`pip install particlefx`（仅需 numpy）；可选依赖组 `pygame` / `pil` /
    `numba` / `taichi` / `gpu` / `schema` / `all`。
- 新增 `pyproject.toml`（setuptools 后端），sdist + wheel 双产物。
- 新增顶层 `particlefx/__init__.py`：一键导出核心 API，并提供
  `particlefx.available_backends()` 查询当前环境可运行的加速后端。
- `particlefx.core.AVAILABLE_BACKENDS` / `particlefx.renderers.AVAILABLE`：
  运行时探测本机实际可用的后端与渲染器。

### Changed

- **源码重组为命名空间包**：原顶层 `core/` `renderers/` `systems/` `components/`
  移入 `particlefx/` 下，包内改用相对导入。
  动机：原顶层名（`core` / `renderers` / `systems` / `components`）过于通用，
  安装到用户 site-packages 会污染命名空间并易与他人包冲突。
- 导入路径变更（破坏性）：`from core.particle import ...` →
  `from particlefx import ...` 或 `from particlefx.core.particle import ...`。
- 渲染器与可选依赖解耦：各渲染器按依赖**独立可选**导入。原先
  `import particlefx.renderers` 会因缺 pygame 直接失败，现在缺依赖的那个不导出，
  其余照常可用（`pip install particlefx` 仅装 numpy 即可导入全部子包）。
- 清理死代码：移除 5 处 `except ImportError` + `sys.path.insert` 回退块
  （包内相对导入不会失败，回退已无意义）。
- README 增加 PyPI 徽章、安装章节，并显式写明「发行名 vs 导入名」。

### Fixed

- `core.particle_numpy` 与 `renderers.fluid_renderer` 的第三方依赖由顶层 import
  改为按需延迟导入，核心包不再隐式依赖 Pillow / pygame。
- 效果工厂的 JSON 配方（`effects_config.json` / `palettes.json`）显式声明为
  package-data，保证安装后 `EffectsFactory` 能正常加载。

## [1.0.0-portable] - 2026-08-27

### Added

- 首个发行版：便携源码 zip（剔除 .venv/构建产物/输出物，依赖见 README）。
- 发布面：gitee release `v1.0.0`（粒子系统库 v1.0.0）。

### Removed

- 仓库 `siwpan/particle` 已删除并重建为 `siwpan/particlefx`（Gitee 不支持直接改
  仓库 URL 路径，且删库后无重定向，旧地址返回 404）。

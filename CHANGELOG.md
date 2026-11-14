# Changelog

本项目所有值得注意的变更都会记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.2.0] - 2026-11-14

### Added

- **可复现（确定性）模拟**：`ParticleSystem(seed=...)` 与 `Emitter(seed=...)`
  支持种子化随机源；系统级种子按发射器「添加顺序」派发独立确定性子流，使
  「系统 seed + 发射器添加顺序 + 同推进序列」即可逐位复现整场模拟。新增便捷方法
  `reseed()` / `save_rng_state()` / `restore_rng_state()`（回滚 / 快照，依赖
  Python `random.Random` 可序列化状态）。未传 `seed` 时回退全局 `random`，与旧版逐位一致。
- **固定步长子步进（数值稳定）**：`ParticleSystem(fixed_dt=..., max_substeps=...)`
  在 `dt > fixed_dt` 时自动把一帧拆成若干固定子步推进（半隐式欧拉稳定性），
  子步数受 `max_substeps` 上限保护（spiral-of-death 防护）。默认 `fixed_dt=None`
  时退化为旧版单步积分，行为逐位一致。
- **共享确定性随机模块**：新增 `particlefx/core/random.py`（`mulberry32` /
  `hash_seed` / `derive_seed`），`light_field.py` 改为复用，消除与 Web 端的重复实现。
- **全系统噪声种子化**：`ForceField` 增加 `seed` 参数（确定性派发噪声 / 湍流时间偏移）；
  系统播种时为 `ForceField` / `NoiseField` 按添加序号派生确定性子流，`reseed()` 一并重派；
  用户显式 `seed` 始终保留。至此含噪声场的整场模拟也完全可复现。
- **魔法效果系统（补齐原「待创建」扩展点）**：新增
  `particlefx/systems/magic_effects.py` 的 `MagicEffectsManager` / `create_magic(...)`，
  提供 5 种可种子复现的魔法效果（奥术爆发 / 符文环 / 治愈微光 / 虚空触须 / 魔法飞弹），
  并导出到 `particlefx.systems` 与包根。（`effects_config.json` 的 `magic` 分类下另有
  6 个声明式魔法配方，此前已存在。）
- **场景配方 + seed 持久化与分享**：新增 `particlefx/components/preset.py`
  （`system_to_recipe` / `recipe_to_system` / `save_preset` / `load_preset` /
  `export_effect_recipe`）；配方记录「结构 + 参数 + 系统种子」，生命期曲线（关键帧数据）
  导出 / 重建无损，一份配方 + 一个种子即可完整复现场景。
- 新增 `tests/test_determinism.py` / `tests/test_substep.py` / `tests/test_magic.py` /
  `tests/test_preset.py`，以及 `selfcheck.py` 的 5 项 v1.2 无头校验（共 9 项）。

### Changed

- `EmitterShape.sample_position` 及全部 `_sample_*` 增加可选 `rng` 参数；发射路径
  （形状采样 + 旋转 / 角速度 / 质量）与 LOD 随机门槛改用系统 / 发射器种子化随机源。
- `ParticleSystem.update()` 内部重构为 `_step()` + 子步进包装；公开 `update()` 对外契约不变。
- `EffectsFactory.create(..., seed=..., fixed_dt=..., max_substeps=...)`：构建期随机
  （speed/life/size 范围抽样、坐标列表、配色）也走种子化随机源，使声明式配方带 seed 时
  可完整复现；`effects_config.schema.json` 增加 `seed` / `fixed_dt` / `max_substeps`
  可选字段，`seed=None` 时仍回退全局 `random`（旧行为不变）。
- README 更正过时的「其他效果（待创建）」清单，补充魔法效果与场景配方说明。

## [1.1.1] - 2026-11-13

### Changed

- **打包元数据改用 PEP 639 许可表达式**：`project.license` 由已废弃的 TOML 表
  `{file = "LICENSE"}` 改为 SPDX 字符串 `"MIT"`，并用 `project.license-files = ["LICENSE"]`
  声明许可证文件；`build-system.requires` 提升至 `setuptools>=77.0`（PEP 639 支持）。
  目的：消除 setuptools 的 `project.license` 弃用告警（原定 2027-02 前必须迁移）。
  功能无变化。

## [1.1.0] - 2026-11-13

### Added

- **粒子光场系统 `LightFieldSystem`**（`particlefx/systems/light_field.py`）：把
  `examples/web/light_field.html` 那套零依赖 Canvas 交互模型移植为 particlefx 原生
  Python 系统，二者共享同一套交互语义与配色常量
  （移动=吸引 / 按住=排斥 / 点击=冲击波 / `?seed=` 可复现 /
  Aurora·Ember·Toxic 三套配色平滑过渡）。物理为逐粒子纯 Python 积分（复用 `Particle`），
  渲染交给库内渲染器（`PygameRenderer(additive+bloom)` 实时 / `LayeredPILRenderer` 离线导出 PNG）。
  - `create_light_field(...)` 快捷构造。
  - `selfcheck.py::check_light_field()` 无头校验：流场运动 / seed 可复现 / 吸引排斥 / 冲击波 / 换色与 `set_count`。
- **Web 参考实现收录**：`examples/web/light_field.html`（零依赖 Canvas2D，双击即开），
  与 Python 端为「同源双实现」，文件头已注明 provenance（共享交互语义与配色常量）。
- **光场交互 Demo**：`examples/light_field_demo.py`（pygame 实时），键位对齐 Web 版
  （移动=吸引 / 按住=排斥 / 点击=冲击波 / 空格=换色 / R=重置 / P=暂停 / S=存 PNG / `[` `]`=增减粒子）。
- **API 文档**：`docs/api.md` 新增 `LightFieldSystem` 段落（构造 / 交互 API 表 / 可复现性与配色说明）。

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

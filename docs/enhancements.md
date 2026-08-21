# v5.0 增强功能落地说明

本文档记录按"成本从低到高"顺序落地的 9 项增强（全部遵循
**可选依赖零开销 + 向后兼容**原则，未启用时行为与旧版完全一致）。

## ① 对象池 + 空间哈希（低成本，纯 Python）

- **ParticlePool**（core/particle.py）：固定容量粒子对象池，acquire/release
  复用 Particle 实例，消除高频创建/回收的 GC 压力；基于 dataclass 字段元数据
  的 Particle.reset() 保证新增字段自动获得正确默认值。
- **SpatialHashGrid**（core/spatial_hash.py）：均匀网格空间哈希，支持
  点/矩形对象登记、去重查询与相邻对象对遍历。
- **集成**：ParticleSystem(use_pool=True, use_spatial_hash=True)；
  query_neighbors(x, y, radius) 供邻居查询。碰撞检测在网格启用时只检查
  粒子所在格子的候选碰撞盒，结果与线性扫描**逐位一致**（有对照测试保证）。

```python
ps = ParticleSystem(max_particles=2000, use_pool=True, use_spatial_hash=True)
```

## ② Numba 可选 JIT 内核（低成本）

core/particle_numba.py — NumbaParticleSystem（继承 ParticleSystemNumpy）：

- numba 可用且未添加自定义力场 → jit 内核更新（重力/风力/阻力常量），
  预期 30~100 倍加速（原 README 的 TODO 落地）
- 否则 → 向量化回退路径，与 ParticleSystemNumpy + NumpyForce* 力场
  **逐位一致**（life 负值语义等细节均有对照测试）
- numba 0.67+ 原生支持 Python 3.14（cp314 轮子）；
  实测 5000 粒子单帧 21µs，比 NumPy 向量化路径快 13.6 倍

```python
from core.particle_numba import NumbaParticleSystem
sys_np = NumbaParticleSystem(max_particles=5000, gravity=3.0, wind=0.0, drag=0.99)
```

## ③ 软粒子深度淡出（低成本，ModernGL 着色器）

粒子与场景几何体穿模时产生硬边的通用解法：

- GPUParticleRenderer(enable_depth_fade=True) + renderer.set_scene_depth(depth_texture)：
  专用 FS_PARTICLE_SOFT_DEPTH 着色器按 scene_depth - gl_FragCoord.z 平滑淡出 alpha
- UnifiedGPURenderer：粒子着色器内置可选 uniform 分支（默认关闭零开销），
  set_scene_depth() 注入深度纹理；enable_depth_fade_with_fluid_depth()
  直接复用流体 SSFR 的深度贴图（hybrid 模式）
- 深度纹理通常来自场景深度预渲染 pass（流体 SSFR / 自建深度 FBO）

## ④ pygbag Web 导出（低成本）

examples/web_main.py：遵循 pygbag 约定（async 主循环 + 每帧
await asyncio.sleep(0)）的迷你烟花秀，桌面/浏览器双端可运行：

```bash
pip install pygbag
pygbag --app_name web_main examples
```

## ⑤ 粒子吸引子 + 粒子光源（中成本）

- **ParticleAttractor**（core/particle.py）：有限范围吸引/排斥力场，
  线性/反距离衰减、极性、follow 回调（跟随鼠标/角色/粒子，每帧自动更新）；
  create_attractor() 工厂。
- **粒子间作用力**：system.enable_pairwise_force(strength, radius, repulsion)
  —— 空间哈希加速 O(N·k) 的粒子间吸引/排斥，未启用哈希时退化为线性扫描。
- **粒子光源**：system.enable_particle_lights(max_lights, min_emissive_strength)
  —— 每帧自动挑选最亮的自发光粒子注册为动态点光源，参与 Blinn-Phong 管线。

## ⑥ 可编程后处理链 PostFX（中成本）

renderers/postfx.py — 声明式后处理链（PIL/numpy 实现，无额外依赖）：

- 内置步骤：BloomStep / TonemapStep（Reinhard、ACES）/ VignetteStep /
  ChromaticAberrationStep / GrainStep
- PostFXChain 顺序执行，支持 dict/JSON 序列化与预设
  （default_fireworks_chain / default_clean_chain）
- 集成：LayeredPILRenderer.render_to_image(particles, postfx=chain)

## ⑦ effects_config JSON Schema 校验（中成本）

- components/effects_config.schema.json：draft-07 Schema（effects/categories/
  emitters/force_fields 结构、形状与力场枚举）
- EffectsFactory(validate=True, strict=False)：jsonschema 可用时执行完整
  Schema 校验；未安装时回退内置轻量校验（缺字段/坏形状/坏力场类型检测）；
  strict=True 时校验失败抛 ValueError

## ⑧ Taichi + SPH 后端（中成本）

core/particle_taichi.py — WCSPH 2D 流体（Müller 2003）：

- Poly6 密度核 + Spiky 压力梯度 + 粘性拉普拉斯核
- add_particles / add_fluid_block 初始化，update(dt) 全 Taichi 内核
  （密度压力 → 力 → 积分 → 边界反弹）
- positions_numpy() / render_to_image() 导出渲染
- 未安装 taichi 时构造抛出带安装指引的 ImportError（其余功能不受影响）；
  taichi 1.7.x 暂无 Python 3.14 轮子（替代方案见文末）

## ⑨ ModernGL Compute 全 GPU 模拟（高成本）

renderers/particle_sim_gpu.py — 粒子数据常驻 SSBO，compute shader 逐帧更新：

- 内核：风力/重力/阻力/点吸引子/生命衰减（与 CPU 端数学一致）
- emit_burst 写入空闲槽位（自动压缩死亡槽位）、update(dt) 一次 dispatch、
  read_particles() / to_particle_tuples() 回读渲染
- 需要 OpenGL 4.3+（#version 430 core）；与 GPUParticleRenderer/
  UnifiedGPURenderer 组合使用

## 测试与回归

所有增强均有独立测试文件（tests/test_spatial_hash_pool.py、
test_particle_numba.py、test_depth_fade.py、test_web_main.py、
test_attractor_lights.py、test_postfx.py、test_effects_schema.py、
test_particle_taichi.py、test_gpu_sim.py），可选依赖缺失时自动 skip，
降级路径全覆盖。另修复了 test_explosion_triggers 的随机样式 flaky 失败。

---

## Python 3.14 环境结论与替代方案（2026 实测）

依据 PyPI 轮子元数据 + 本机实测（Python 3.14.7 / Windows）：

| 依赖 | Python 3.14 支持 | 结论 |
|------|-----------------|------|
| numba 0.67.0 | ✅ 有 cp314 轮子 | **直接 pip install numba 即可**；实测 5000 粒子 JIT 21µs/帧（13.6×） |
| moderngl 5.12.0 | ❌ 依赖 glcontext 3.0.0 无 cp314 轮子 | 需 MSVC 编译，或 3.12/3.13 venv（glcontext 有 cp312/cp313 轮子） |
| taichi 1.7.4 | ❌ 无 cp314 轮子 | 3.12/3.13 venv，或用下述替代方案 |
| warp-lang 1.16.0 | ❌ 无 cp314 轮子（且需 NVIDIA GPU） | 3.12/3.13 + NVIDIA GPU |
| jaxlib 0.11.1 | ✅ 有 cp314 轮子 | CPU 可跑；SPH 需自行实现（配 jax-md 邻居列表） |
| cython 3.2.9 | ✅ 有 cp314 轮子 | 手写 C 扩展内核（PySPH 本身无 cp314 轮子） |

### numba 的替代（如无需时）

- 项目自带 NumPy 向量化后端（particle_numpy）与对象池/空间哈希，
  零依赖即可获得不错性能
- JAX（jaxlib 0.11.1 支持 3.14）：函数式数组计算，CPU/GPU 同构，
  适合将更新写成 jit 函数；生态有 jax-md 提供邻居搜索
- Cython 3.2（支持 3.14）：最接近 C 的手写内核方案
- mypyc / CPython 3.14 自带 JIT（PEP 744，实验性）：纯解释路径的免费小幅加速

### taichi（SPH）的替代（3.14 上）

1. **Python 3.12/3.13 venv 装 taichi**（最省事）：本机已建 .venv312
   （uv 管理的 CPython 3.12.13 + taichi 1.7.4）
2. **JAX 手写 SPH**：jaxlib 有 cp314 轮子，用 jax-md 的 cell-list 邻居搜索，
   压力/粘性核与 TaichiSPHSystem 数学相同，jit 后 CPU/GPU 同跑
3. **NVIDIA Warp**：SPH 邻居搜索开箱即用（sphWarpCore），但当前无 cp314
   轮子且需 NVIDIA GPU → 配合 3.12/3.13 使用
4. **项目内置回退**：ParticleSystemNumpy + SpatialHashGrid 的 pairwise
   力已能近似有限流体效果（无 SPH 压力项）

### 本机 GPU/JIT 验证环境（已实测 ✅）

.venv312（Python 3.12.13 + moderngl 5.12 + taichi 1.7.4 + numba 0.67）：

    .venv312/Scripts/python.exe -m pytest -q

实测结果（本机 NVIDIA GPU / CUDA）：

- 全量 497 passed / 2 skipped：ModernGL compute 全 GPU 模拟、
  软粒子深度淡出、Taichi SPH（CUDA 后端）、Numba JIT 全部真实执行
- 3.14 主环境 485 passed / 14 skipped（仅 moderngl/taichi 相关项跳过；
  numba 0.67 在 3.14 上直接可用，实测 5000 粒子 21µs/帧 = 13.6× NumPy）
- moderngl 在 3.14 上唯一障碍是 glcontext 无 cp314 轮子；
  如需 3.14 原生支持：安装 MSVC Build Tools 后
  pip install moderngl 会从源码编译 glcontext


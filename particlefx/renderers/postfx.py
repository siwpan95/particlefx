"""
可编程后处理链 PostFX（v5.0 增强 ⑥）

将后处理步骤（Bloom / 色调映射 / 晕影 / 色差 / 颗粒）声明式组合为一条链，
按顺序应用到 PIL 图像上。支持 dict/JSON 序列化，便于与 effects_config.json
同级配置。

设计：
- 每个步骤独立实现 apply(image) -> image，不修改输入
- PostFXChain 按顺序执行，可嵌套导出为 JSON
- 全部基于 numpy + PIL，无额外依赖
- 与 LayeredPILRenderer.render_to_image(postfx=...) 集成

用法：
    from .postfx import PostFXChain, BloomStep, TonemapStep, VignetteStep

    chain = PostFXChain([
        BloomStep(radius=12, strength=0.6, threshold=150),
        TonemapStep(mode='aces', exposure=1.2),
        VignetteStep(strength=0.35),
    ])
    out = chain.apply(image)

    # JSON 序列化 / 反序列化
    data = chain.to_dict()
    chain2 = PostFXChain.from_dict(data)
    chain.save_json("postfx.json")
    chain3 = PostFXChain.load_json("postfx.json")
"""

import json
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageFilter

from .pil_renderer import post_process_bloom  # 复用现有 Bloom 实现


# ================================================================
#  步骤基类与内置步骤
# ================================================================

class PostFXStep:
    """后处理步骤基类"""

    name: str = "step"

    def apply(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PostFXStep":
        raise NotImplementedError


class BloomStep(PostFXStep):
    """Bloom 辉光（复用 pil_renderer.post_process_bloom）"""

    name = "bloom"

    def __init__(self, radius: int = 12, strength: float = 0.5,
                 threshold: int = 160):
        self.radius = int(radius)
        self.strength = float(strength)
        self.threshold = int(threshold)

    def apply(self, image: Image.Image) -> Image.Image:
        return post_process_bloom(image, radius=self.radius,
                                  strength=self.strength,
                                  threshold=self.threshold)

    def to_dict(self):
        return {"type": self.name, "radius": self.radius,
                "strength": self.strength, "threshold": self.threshold}

    @classmethod
    def from_dict(cls, data):
        return cls(radius=data.get("radius", 12),
                   strength=data.get("strength", 0.5),
                   threshold=data.get("threshold", 160))


def _tonemap_numpy(arr: np.ndarray, mode: str, exposure: float) -> np.ndarray:
    """色调映射：reinhard / aces（Narkowicz 拟合）"""
    x = np.clip(arr * exposure, 0.0, None)
    if mode == "reinhard":
        mapped = x / (x + 1.0)
    elif mode == "aces":
        mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
        mapped = np.clip(mapped, 0.0, 1.0)
    else:
        mapped = x
    # 近似 gamma 2.2
    return np.power(mapped, 1.0 / 2.2)


class TonemapStep(PostFXStep):
    """色调映射 + gamma（防 additive 混合过曝）"""

    name = "tonemap"

    def __init__(self, mode: str = "reinhard", exposure: float = 1.0):
        self.mode = mode if mode in ("reinhard", "aces") else "reinhard"
        self.exposure = float(exposure)

    def apply(self, image: Image.Image) -> Image.Image:
        arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        mapped = _tonemap_numpy(arr, self.mode, self.exposure)
        return Image.fromarray(np.clip(mapped * 255, 0, 255).astype(np.uint8))

    def to_dict(self):
        return {"type": self.name, "mode": self.mode,
                "exposure": self.exposure}

    @classmethod
    def from_dict(cls, data):
        return cls(mode=data.get("mode", "reinhard"),
                   exposure=data.get("exposure", 1.0))


class VignetteStep(PostFXStep):
    """晕影：四周压暗，聚焦中心"""

    name = "vignette"

    def __init__(self, strength: float = 0.4, power: float = 2.2):
        self.strength = float(strength)
        self.power = float(power)

    def apply(self, image: Image.Image) -> Image.Image:
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        h, w = arr.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        d = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
        d = np.clip(d / np.sqrt(2.0), 0.0, 1.0)
        mask = 1.0 - self.strength * np.power(d, self.power)
        mask = np.clip(mask, 0.0, 1.0)[:, :, None]
        return Image.fromarray(np.clip(arr * mask, 0, 255).astype(np.uint8))

    def to_dict(self):
        return {"type": self.name, "strength": self.strength,
                "power": self.power}

    @classmethod
    def from_dict(cls, data):
        return cls(strength=data.get("strength", 0.4),
                   power=data.get("power", 2.2))


class ChromaticAberrationStep(PostFXStep):
    """色差：红/蓝通道轻微径向偏移（镜头感）"""

    name = "chromatic_aberration"

    def __init__(self, offset: float = 1.5):
        self.offset = max(0.0, float(offset))

    def apply(self, image: Image.Image) -> Image.Image:
        if self.offset <= 0:
            return image.copy()
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        h, w = arr.shape[:2]
        off = self.offset
        r = np.roll(arr[:, :, 0], shift=-int(off), axis=1)
        b = np.roll(arr[:, :, 2], shift=int(off), axis=1)
        out = arr.copy()
        out[:, :, 0] = r
        out[:, :, 2] = b
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    def to_dict(self):
        return {"type": self.name, "offset": self.offset}

    @classmethod
    def from_dict(cls, data):
        return cls(offset=data.get("offset", 1.5))


class GrainStep(PostFXStep):
    """胶片颗粒（加性噪声，可固定种子复现）"""

    name = "grain"

    def __init__(self, amount: float = 6.0, seed: Optional[int] = None):
        self.amount = max(0.0, float(amount))
        self.seed = seed

    def apply(self, image: Image.Image) -> Image.Image:
        if self.amount <= 0:
            return image.copy()
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0.0, self.amount, arr.shape)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    def to_dict(self):
        return {"type": self.name, "amount": self.amount, "seed": self.seed}

    @classmethod
    def from_dict(cls, data):
        return cls(amount=data.get("amount", 6.0), seed=data.get("seed"))


# ================================================================
#  步骤注册表（JSON 反序列化用）
# ================================================================

STEP_REGISTRY = {
    cls.name: cls for cls in (
        BloomStep, TonemapStep, VignetteStep,
        ChromaticAberrationStep, GrainStep,
    )
}


# ================================================================
#  PostFXChain
# ================================================================

class PostFXChain:
    """按顺序执行的后处理链（v5.0 增强 ⑥）"""

    def __init__(self, steps: Optional[List[PostFXStep]] = None):
        self.steps: List[PostFXStep] = list(steps or [])

    # --- 构建 ---

    def add(self, step: PostFXStep) -> "PostFXChain":
        self.steps.append(step)
        return self

    # --- 执行 ---

    def apply(self, image: Image.Image) -> Image.Image:
        """按顺序应用全部步骤（不修改输入）"""
        img = image
        for step in self.steps:
            img = step.apply(img)
        return img

    def __len__(self):
        return len(self.steps)

    def __repr__(self):
        return f"PostFXChain({[s.name for s in self.steps]})"

    # --- JSON 序列化 ---

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": [s.to_dict() for s in self.steps]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PostFXChain":
        chain = cls()
        for item in data.get("steps", []):
            step_type = item.get("type")
            step_cls = STEP_REGISTRY.get(step_type)
            if step_cls is None:
                raise ValueError(f"未知后处理步骤类型: {step_type!r}")
            chain.add(step_cls.from_dict(item))
        return chain

    @classmethod
    def from_json(cls, text: str) -> "PostFXChain":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load_json(cls, path: str) -> "PostFXChain":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())


# ================================================================
#  预设
# ================================================================

def default_fireworks_chain() -> PostFXChain:
    """烟花/魔法场景推荐链：Bloom → ACES 色调映射 → 轻微晕影"""
    return PostFXChain([
        BloomStep(radius=14, strength=0.7, threshold=140),
        TonemapStep(mode="aces", exposure=1.15),
        VignetteStep(strength=0.25, power=2.5),
    ])


def default_clean_chain() -> PostFXChain:
    """普通场景轻量链：仅色调映射防过曝"""
    return PostFXChain([
        TonemapStep(mode="reinhard", exposure=1.0),
    ])

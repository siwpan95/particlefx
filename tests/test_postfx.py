"""
可编程后处理链 PostFX 测试（v5.0 增强 ⑥）
"""
import numpy as np
import pytest
from PIL import Image

from particlefx.renderers.postfx import (
    PostFXChain, PostFXStep, BloomStep, TonemapStep, VignetteStep,
    ChromaticAberrationStep, GrainStep,
    default_fireworks_chain, default_clean_chain, STEP_REGISTRY,
)


def _img(w=64, h=48):
    """生成带明显暗角/亮中心的渐变测试图"""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    val = np.clip(255 * (1.0 - d), 0, 255).astype(np.uint8)
    arr[..., 0] = val
    arr[..., 1] = val // 2
    arr[..., 2] = val // 4
    return Image.fromarray(arr)


class TestSteps:
    def test_bloom_step(self):
        out = BloomStep(radius=6, strength=0.5, threshold=120).apply(_img())
        assert out.size == (64, 48)
        a = np.array(_img(), dtype=np.float32)
        b = np.array(out, dtype=np.float32)
        assert b.sum() >= a.sum() * 0.9  # Bloom 不应显著压暗

    def test_tonemap_reinhard_clamps(self):
        img = Image.new("RGB", (16, 16), (255, 255, 255))
        out = TonemapStep(mode="reinhard").apply(img)
        arr = np.array(out)
        assert arr.max() <= 255
        assert arr.min() >= 0

    def test_tonemap_aces_darkens_highlights(self):
        img = Image.new("RGB", (16, 16), (255, 255, 255))
        aces = TonemapStep(mode="aces", exposure=1.0).apply(img)
        assert np.array(aces).mean() < 255  # 纯白被色调映射压下来

    def test_vignette_darkens_corners(self):
        out = VignetteStep(strength=0.6, power=2.0).apply(_img())
        arr = np.array(out, dtype=np.float32)
        corner = arr[0:4, 0:4].mean()
        center = arr[20:28, 28:36].mean()
        assert corner < center

    def test_chromatic_aberration_shifts_channels(self):
        # 用横向渐变图：R 通道逐列变化，偏移后与 G 通道不再一致
        w, h = 32, 32
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[..., 0] = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
        arr[..., 1] = 100
        arr[..., 2] = 100
        img = Image.fromarray(arr)
        out = ChromaticAberrationStep(offset=2.0).apply(img)
        out_arr = np.array(out)
        # 边缘列：R 与 G 不再相等（R 发生了偏移）
        assert not np.array_equal(out_arr[:, 0, 0], out_arr[:, 0, 1])
        assert not np.array_equal(out_arr[:, -1, 0], out_arr[:, -1, 1])

    def test_grain_adds_noise_and_seed_reproducible(self):
        img = Image.new("RGB", (16, 16), (128, 128, 128))
        a = np.array(GrainStep(amount=8.0, seed=42).apply(img))
        b = np.array(GrainStep(amount=8.0, seed=42).apply(img))
        assert np.array_equal(a, b)  # 同种子结果一致
        assert a.std() > 0  # 有噪声

    def test_chain_applies_in_order(self):
        calls = []
        class Spy(PostFXStep):
            name = "spy"
            def __init__(self, tag):
                self.tag = tag
            def apply(self, image):
                calls.append(self.tag)
                return image
            def to_dict(self):
                return {"type": self.name, "tag": self.tag}
            @classmethod
            def from_dict(cls, d):
                return cls(d["tag"])
        chain = PostFXChain([Spy("a"), Spy("b")])
        chain.apply(_img())
        assert calls == ["a", "b"]

    def test_chain_does_not_mutate_input(self):
        img = _img()
        before = np.array(img).copy()
        chain = default_fireworks_chain()
        chain.apply(img)
        assert np.array_equal(np.array(img), before)


class TestSerialization:
    def test_roundtrip(self):
        chain = default_fireworks_chain()
        chain.add(ChromaticAberrationStep(offset=1.2))
        data = chain.to_dict()
        assert data["steps"][0]["type"] == "bloom"
        chain2 = PostFXChain.from_dict(data)
        assert [s.name for s in chain2.steps] == [s.name for s in chain.steps]
        out1 = chain.apply(_img())
        out2 = chain2.apply(_img())
        assert np.array_equal(np.array(out1), np.array(out2))

    def test_json_roundtrip(self, tmp_path):
        chain = default_fireworks_chain()
        p = tmp_path / "postfx.json"
        chain.save_json(str(p))
        chain2 = PostFXChain.load_json(str(p))
        assert [s.name for s in chain2.steps] == [s.name for s in chain.steps]

    def test_unknown_step_raises(self):
        with pytest.raises(ValueError):
            PostFXChain.from_dict({"steps": [{"type": "not_a_step"}]})

    def test_registry_covers_builtins(self):
        for name in ("bloom", "tonemap", "vignette",
                     "chromatic_aberration", "grain"):
            assert name in STEP_REGISTRY


class TestPresets:
    def test_fireworks_chain(self):
        chain = default_fireworks_chain()
        assert [s.name for s in chain.steps] == ["bloom", "tonemap", "vignette"]
        out = chain.apply(_img())
        assert out.size == (64, 48)

    def test_clean_chain(self):
        chain = default_clean_chain()
        assert [s.name for s in chain.steps] == ["tonemap"]


class TestRendererIntegration:
    def test_layered_renderer_accepts_postfx(self):
        from particlefx.core.particle import Particle
        from particlefx.renderers.pil_renderer import LayeredPILRenderer
        r = LayeredPILRenderer(width=128, height=96)
        p = Particle(x=64, y=48, vx=0, vy=0, life=1.0, max_life=2.0,
                     size=6.0, color=(255, 200, 80), alpha=1.0)
        plain = r.render_to_image([p])
        styled = r.render_to_image([p], postfx=default_fireworks_chain())
        assert plain.size == styled.size == (128, 96)
        assert not np.array_equal(np.array(plain), np.array(styled))  # 链生效

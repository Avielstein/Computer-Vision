"""Thin adapter around the upstream ``lingbot_vision`` inference API.

This wrapper hides the couple of sharp edges in the raw API so the rest of
the subproject can think in terms of *dense feature grids*:

- the small ViT-S/16 variant is the default (runs on CPU / Apple Silicon);
- ``extract_patch_tokens`` wants a **string** device, so we keep one around;
- callers usually want features shaped ``[h, w, C]`` (a spatial grid), not the
  flat ``[1, h*w, C]`` token sequence the backbone returns.

Everything downstream (object discovery, tracking, navigation cues) consumes
:meth:`LBVBackbone.dense_features`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Resolve the upstream `lingbot_vision` package. Prefer a pip-installed copy,
# but fall back to the vendored clone under third_party/ so the sandbox runs
# under any interpreter (pyenv, conda, VSCode's Run button) without needing
# `pip install -e` to have targeted that specific interpreter.
try:  # noqa: SIM105
    import lingbot_vision  # noqa: F401
except ModuleNotFoundError:
    _vendored = Path(__file__).resolve().parents[1] / "third_party" / "lingbot-vision"
    if (_vendored / "lingbot_vision").is_dir():
        sys.path.insert(0, str(_vendored))
    else:
        raise ModuleNotFoundError(
            "lingbot_vision not found. Run ./setup.sh from the LingBotVisionNav "
            "directory to clone it into third_party/ and install dependencies."
        )

from lingbot_vision import (
    extract_patch_tokens,
    load_image,
    load_pretrained_backbone,
)
from lingbot_vision.preprocess import IMAGENET_MEAN, IMAGENET_STD, _snap


@dataclass
class DenseFeatures:
    """Result of running the backbone on one image."""

    tokens: torch.Tensor  # [h*w, C] float32 patch tokens (on CPU)
    grid: np.ndarray  # [h, w, C] float32 spatial feature map
    rgb: np.ndarray  # [H, W, 3] uint8 image the tokens were computed from
    hw: tuple[int, int]  # (h, w) patch-grid dims
    embed_dim: int  # C

    @property
    def normalized_grid(self) -> np.ndarray:
        """L2-normalized ``[h, w, C]`` grid (cosine-similarity ready)."""
        g = self.grid.reshape(-1, self.embed_dim)
        g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)
        return g.reshape(*self.hw, self.embed_dim)


class LBVBackbone:
    """Frozen LingBot-Vision backbone + dense feature extraction."""

    def __init__(self, variant: str = "small", device: str | None = None, dtype: str = "auto"):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # CPU + bf16/fp16 autocast is unsupported / slow; force fp32 off-GPU.
        if dtype == "auto" and not device.startswith("cuda"):
            dtype = "fp32"
        self.device = device
        self.model, self.embed_dim = load_pretrained_backbone(
            variant=variant, device=device, dtype=dtype
        )
        # Resolve the concrete torch dtype the model ended up in.
        self.torch_dtype = next(self.model.parameters()).dtype
        self.patch_size = int(self.model.patch_size)
        self.variant = variant

    # -- preprocessing -----------------------------------------------------
    def _prep_array(self, rgb: np.ndarray, size: int, mode: str):
        """Preprocess an in-memory RGB uint8 array the same way ``load_image`` does."""
        size = _snap(size, self.patch_size)
        pil = Image.fromarray(rgb).convert("RGB")
        if mode == "square":
            crop = pil.resize((size, size), resample=Image.BILINEAR)
        elif mode == "shortest":
            w0, h0 = pil.size
            if w0 < h0:
                new_w, new_h = size, int(round(size * h0 / w0))
            else:
                new_h, new_w = size, int(round(size * w0 / h0))
            resized = pil.resize((new_w, new_h), resample=Image.BILINEAR)
            left, top = (new_w - size) // 2, (new_h - size) // 2
            crop = resized.crop((left, top, left + size, top + size))
        else:
            raise ValueError(f"unknown resize mode: {mode!r}")
        img_rgb = np.asarray(crop, dtype=np.uint8)
        img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
        img_t = img_t.permute(2, 0, 1).unsqueeze(0)
        img_norm = (img_t - IMAGENET_MEAN) / IMAGENET_STD
        return img_norm, img_rgb

    # -- inference ---------------------------------------------------------
    def dense_features(self, image, size: int = 512, mode: str = "square") -> DenseFeatures:
        """Run the backbone on ``image`` (path | PIL | np.uint8 array).

        Returns a :class:`DenseFeatures` with a ``[h, w, C]`` feature grid.
        """
        if isinstance(image, (str, Path)):
            img_norm, img_rgb, _ = load_image(
                str(image), size=size, patch_size=self.patch_size, mode=mode
            )
        else:
            if isinstance(image, Image.Image):
                image = np.asarray(image.convert("RGB"), dtype=np.uint8)
            img_norm, img_rgb = self._prep_array(np.asarray(image), size=size, mode=mode)

        tokens, (h, w) = extract_patch_tokens(self.model, img_norm, self.device, self.torch_dtype)
        tokens = tokens[0].detach().float().cpu()  # [h*w, C]
        grid = tokens.numpy().reshape(h, w, self.embed_dim)
        return DenseFeatures(
            tokens=tokens, grid=grid, rgb=img_rgb, hw=(h, w), embed_dim=self.embed_dim
        )


def load(variant: str = "small", device: str | None = None, dtype: str = "auto") -> LBVBackbone:
    """Convenience constructor: ``bb = lbv_nav.load()``."""
    return LBVBackbone(variant=variant, device=device, dtype=dtype)

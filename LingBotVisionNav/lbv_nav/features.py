"""Feature-map utilities shared by the experiments.

Everything here operates on the ``[h, w, C]`` dense grid produced by
:class:`lbv_nav.backbone.DenseFeatures` (or its L2-normalized variant).
"""

from __future__ import annotations

import numpy as np


def pca_rgb(grid: np.ndarray) -> np.ndarray:
    """Project a ``[h, w, C]`` feature grid to a ``[h, w, 3]`` uint8 PCA image.

    Mirrors the upstream demo: mean-center, take the top-3 principal
    directions, percentile-stretch each channel to [0, 1].
    """
    h, w, c = grid.shape
    x = grid.reshape(-1, c).astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    rgb = (x @ vt[:3].T).reshape(h, w, 3)
    lo = np.percentile(rgb, 1, axis=(0, 1), keepdims=True)
    hi = np.percentile(rgb, 99, axis=(0, 1), keepdims=True)
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)


def cosine_heatmap(norm_grid: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    """Cosine similarity of every patch in ``norm_grid`` to ``query_vec``.

    ``norm_grid`` must be L2-normalized ``[h, w, C]``; returns ``[h, w]`` in
    roughly [-1, 1].
    """
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    return norm_grid @ q


def kmeans_segments(grid: np.ndarray, k: int = 6, seed: int = 0) -> np.ndarray:
    """Unsupervised segmentation of the feature grid into ``k`` clusters.

    Returns a ``[h, w]`` int label map. Useful as training-free object /
    region proposals: boundary-centric features cluster cleanly along object
    edges.
    """
    from sklearn.cluster import KMeans

    h, w, c = grid.shape
    x = grid.reshape(-1, c).astype(np.float32)
    labels = KMeans(n_clusters=k, random_state=seed, n_init=4).fit_predict(x)
    return labels.reshape(h, w)


def upsample(map_2d: np.ndarray, out_hw: tuple[int, int], smooth: bool = True) -> np.ndarray:
    """Resize a low-res patch-grid map up to image resolution.

    ``smooth=True`` uses bilinear (good for heatmaps); ``False`` uses nearest
    (good for integer label maps).
    """
    import cv2

    H, W = out_hw
    interp = cv2.INTER_LINEAR if smooth else cv2.INTER_NEAREST
    return cv2.resize(map_2d.astype(np.float32), (W, H), interpolation=interp)


def overlay_heatmap(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend a [h,w] or [H,W] heatmap (any range) over an RGB uint8 image."""
    import cv2

    H, W = rgb.shape[:2]
    if heat.shape[:2] != (H, W):
        heat = upsample(heat, (H, W), smooth=True)
    hn = (heat - heat.min()) / (heat.ptp() + 1e-8)
    cmap = cv2.applyColorMap((hn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
    return (alpha * cmap + (1 - alpha) * rgb).astype(np.uint8)


def label_map_to_rgb(labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Colorize an integer label map with a fixed random palette."""
    rng = np.random.default_rng(seed)
    k = int(labels.max()) + 1
    palette = rng.integers(40, 255, size=(k, 3), dtype=np.uint8)
    return palette[labels]

"""Object discovery / detection from frozen dense features (no detector head).

Two training-free modes:

- :func:`query_detect` -- one-shot detection: point at an object in the image
  (or hand in a descriptor), get a similarity mask + bounding boxes for every
  region that matches it. This is the open-vocabulary-ish "find more things
  like this" primitive.
- :func:`cluster_objects` -- fully unsupervised: KMeans over patch features
  yields region proposals; connected components of each cluster become boxes.
"""

from __future__ import annotations

import numpy as np

from .backbone import DenseFeatures
from .features import cosine_heatmap, kmeans_segments


def _grid_boxes(mask_grid: np.ndarray, img_hw, min_patches: int = 3):
    """Connected components of a boolean patch-grid mask -> pixel-space boxes."""
    import cv2

    h, w = mask_grid.shape
    H, W = img_hw
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_grid.astype(np.uint8), connectivity=8
    )
    sx, sy = W / w, H / h
    boxes = []
    for i in range(1, n):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_patches:
            continue
        x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        boxes.append(
            {
                "xyxy": (int(x * sx), int(y * sy), int((x + bw) * sx), int((y + bh) * sy)),
                "area_patches": int(area),
            }
        )
    return sorted(boxes, key=lambda b: -b["area_patches"])


def query_detect(
    feats: DenseFeatures,
    query_xy: tuple[int, int] | None = None,
    query_vec: np.ndarray | None = None,
    sim_thresh: float = 0.5,
    min_patches: int = 3,
):
    """Detect regions matching a query descriptor.

    Provide either ``query_xy`` (an (x, y) pixel in the resized image) or an
    explicit ``query_vec``. Returns dict with the similarity heatmap, boolean
    mask, and pixel-space bounding boxes.
    """
    ngrid = feats.normalized_grid
    h, w = feats.hw
    H, W = feats.rgb.shape[:2]

    if query_vec is None:
        if query_xy is None:
            raise ValueError("pass either query_xy or query_vec")
        gx = min(w - 1, max(0, int(query_xy[0] / W * w)))
        gy = min(h - 1, max(0, int(query_xy[1] / H * h)))
        query_vec = ngrid[gy, gx]

    heat = cosine_heatmap(ngrid, query_vec)
    mask = heat >= sim_thresh
    boxes = _grid_boxes(mask, (H, W), min_patches=min_patches)
    return {"heat": heat, "mask": mask, "boxes": boxes, "hw": (h, w)}


def cluster_objects(feats: DenseFeatures, k: int = 6, min_patches: int = 4, seed: int = 0):
    """Unsupervised object proposals via KMeans over patch features.

    Returns dict with the ``[h, w]`` label map and, per cluster, its bounding
    boxes (connected components in pixel space).
    """
    labels = kmeans_segments(feats.grid, k=k, seed=seed)
    H, W = feats.rgb.shape[:2]
    per_cluster = {}
    for c in range(k):
        boxes = _grid_boxes(labels == c, (H, W), min_patches=min_patches)
        if boxes:
            per_cluster[c] = boxes
    return {"labels": labels, "clusters": per_cluster, "hw": feats.hw}


def draw_boxes(rgb: np.ndarray, boxes, color=(0, 255, 0), thickness: int = 2):
    """Draw a list of ``{'xyxy': (...)}`` boxes onto a copy of ``rgb``."""
    import cv2

    out = rgb.copy()
    for b in boxes:
        x0, y0, x1, y1 = b["xyxy"]
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
    return out

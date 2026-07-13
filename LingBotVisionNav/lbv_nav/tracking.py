"""Training-free object tracking via dense-feature token matching.

LingBot-Vision's boundary-centric features are spatially structured, so an
object selected in frame 0 can be re-localized in later frames purely by
nearest-neighbor matching in feature space -- no tracker training, no motion
model. This is the "video object segmentation with training-free token
matching" use case from the model card, reduced to a point/box tracker.
"""

from __future__ import annotations

import numpy as np

from .backbone import LBVBackbone
from .features import cosine_heatmap


def _grid_xy_from_pixel(px, py, img_hw, grid_hw):
    """Map an image-space pixel to a (row, col) index in the patch grid."""
    H, W = img_hw
    h, w = grid_hw
    gx = min(w - 1, max(0, int(px / W * w)))
    gy = min(h - 1, max(0, int(py / H * h)))
    return gy, gx


def track_point(
    backbone: LBVBackbone,
    frames: list,
    init_xy: tuple[int, int],
    size: int = 512,
    mode: str = "square",
    sim_thresh: float = 0.5,
):
    """Track a query point across ``frames`` by dense-feature matching.

    ``init_xy`` is the (x, y) pixel of the target in the *first* frame (in the
    resized ``size`` x ``size`` space). Returns a list of per-frame dicts with
    the peak-match pixel, its similarity, and the full similarity heatmap.

    The query descriptor is refreshed each frame from the best match (simple
    template update) so slow appearance changes are tolerated.
    """
    results = []
    query = None
    for i, frame in enumerate(frames):
        feats = backbone.dense_features(frame, size=size, mode=mode)
        ngrid = feats.normalized_grid  # [h, w, C]
        h, w = feats.hw
        H, W = feats.rgb.shape[:2]

        if query is None:
            gy, gx = _grid_xy_from_pixel(init_xy[0], init_xy[1], (H, W), (h, w))
            query = ngrid[gy, gx].copy()

        heat = cosine_heatmap(ngrid, query)  # [h, w]
        gy, gx = np.unravel_index(int(np.argmax(heat)), heat.shape)
        sim = float(heat[gy, gx])
        # peak patch center -> pixel
        px = int((gx + 0.5) / w * W)
        py = int((gy + 0.5) / h * H)

        found = sim >= sim_thresh
        if found:  # template update only on confident matches
            query = 0.7 * query + 0.3 * ngrid[gy, gx]
            query /= np.linalg.norm(query) + 1e-8

        results.append(
            {
                "frame": i,
                "xy": (px, py),
                "sim": sim,
                "found": found,
                "heat": heat,
                "rgb": feats.rgb,
                "hw": (h, w),
            }
        )
    return results

"""Navigation-oriented spatial cues from dense features.

LingBot-Vision is a 2D image backbone -- it is the *initializer* for
LingBot-Depth 2.0, which lifts these features to metric 3D. This module stays
honest about that: it derives cheap, geometry-free navigation cues from the
2D feature grid, and marks the clean seam where a depth model plugs in.

Cues implemented:

- ``free_space_mask``: a feature-similarity heuristic for "ground / traversable
  vs. obstacle". We assume the bottom-center patches of a forward-facing camera
  look at the floor, use them as a reference descriptor, and flag patches whose
  features diverge from it as obstacles.
- ``obstacle_columns``: collapses the mask into a per-column nearest-obstacle
  profile -- a 1D "how far can I go straight ahead in this direction" signal,
  the kind of thing a local planner consumes.
"""

from __future__ import annotations

import numpy as np

from .backbone import DenseFeatures
from .features import cosine_heatmap


def free_space_mask(
    feats: DenseFeatures,
    ref_frac: float = 0.15,
    sim_thresh: float = 0.6,
):
    """Heuristic traversable-ground mask from the feature grid.

    Samples a reference descriptor from the bottom-center ``ref_frac`` band of
    the grid (assumed floor for a forward-facing camera), then labels each
    patch traversable if its cosine similarity to that reference exceeds
    ``sim_thresh``.

    Returns ``(free[h, w] bool, sim[h, w] float)``.
    """
    ngrid = feats.normalized_grid
    h, w = feats.hw
    band = max(1, int(h * ref_frac))
    c0, c1 = int(w * 0.3), int(w * 0.7)
    ref = ngrid[h - band :, c0:c1].reshape(-1, feats.embed_dim).mean(axis=0)
    ref /= np.linalg.norm(ref) + 1e-8
    sim = cosine_heatmap(ngrid, ref)
    return sim >= sim_thresh, sim


def obstacle_columns(free: np.ndarray) -> np.ndarray:
    """Per-column free-space depth from a ``[h, w]`` traversability mask.

    For each column, walks up from the bottom of the frame and counts how many
    consecutive rows are traversable before the first obstacle. Normalized to
    [0, 1] (1 == free all the way up). This is a simple forward-clearance
    profile across the field of view.
    """
    h, w = free.shape
    clearance = np.zeros(w, dtype=np.float32)
    for x in range(w):
        run = 0
        for y in range(h - 1, -1, -1):
            if free[y, x]:
                run += 1
            else:
                break
        clearance[x] = run / h
    return clearance


def steer_suggestion(clearance: np.ndarray) -> dict:
    """Toy planner: pick the FOV direction with the most forward clearance.

    Returns the argmax column (normalized to [-1, 1], negative = left) and a
    coarse action label. Purely illustrative of how the cue feeds a planner.
    """
    w = len(clearance)
    # smooth so we prefer wide open corridors, not single lucky columns
    k = max(1, w // 8)
    kernel = np.ones(k) / k
    smoothed = np.convolve(clearance, kernel, mode="same")
    best = int(np.argmax(smoothed))
    offset = (best - (w - 1) / 2) / ((w - 1) / 2)  # -1 (left) .. +1 (right)
    if smoothed[best] < 0.25:
        action = "blocked"
    elif abs(offset) < 0.2:
        action = "forward"
    elif offset < 0:
        action = "turn_left"
    else:
        action = "turn_right"
    return {"offset": float(offset), "action": action, "clearance": float(smoothed[best])}

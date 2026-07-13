"""LingBot-Vision navigation sandbox.

Small helper layer on top of the frozen LingBot-Vision ViT backbone for
exploring dense-feature use cases relevant to robot navigation: object
discovery, training-free tracking, and spatial / traversability cues.

    import lbv_nav
    bb = lbv_nav.load("small")           # frozen ViT-S/16, CPU-friendly
    feats = bb.dense_features("img.png")  # [h, w, C] boundary-centric grid
"""

from .backbone import DenseFeatures, LBVBackbone, load
from . import features

__all__ = ["load", "LBVBackbone", "DenseFeatures", "features"]

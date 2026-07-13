"""LingBot-Vision navigation-sandbox demos.

One CLI over the frozen ViT-S/16 backbone. Every subcommand writes a PNG (or
GIF) panel into ``outputs/``.

    python demos.py pca      --input data/example.png
    python demos.py detect   --input data/example.png --xy 256 256
    python demos.py cluster  --input data/example.png --k 6
    python demos.py nav      --input data/example.png
    python demos.py track    --frames data/frames --xy 256 256

Defaults target CPU / Apple Silicon with the small variant. Pass
``--variant base|large|giant`` and ``--device cuda`` on a GPU box.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import lbv_nav
from lbv_nav import features as F
from lbv_nav import navigation as NAV
from lbv_nav import object_discovery as OD
from lbv_nav import tracking as TR

OUT = Path(__file__).parent / "outputs"


def _save(name: str, rgb: np.ndarray):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"[demos] wrote {path}")
    return path


def _panel(*imgs, labels=None):
    """Horizontally concatenate same-height RGB images with optional labels."""
    H = min(im.shape[0] for im in imgs)
    resized = []
    for i, im in enumerate(imgs):
        r = cv2.resize(im, (int(im.shape[1] * H / im.shape[0]), H))
        if labels:
            cv2.rectangle(r, (0, 0), (r.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(r, labels[i], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        resized.append(r)
    return np.concatenate(resized, axis=1)


def cmd_pca(bb, args):
    feats = bb.dense_features(args.input, size=args.size, mode=args.mode)
    pca = F.upsample(F.pca_rgb(feats.grid).astype(np.float32), feats.rgb.shape[:2], smooth=False).astype(np.uint8)
    _save("pca.png", _panel(feats.rgb, pca, labels=[f"input {feats.rgb.shape[0]}px", f"patch PCA {feats.hw[0]}x{feats.hw[1]}"]))


def cmd_detect(bb, args):
    feats = bb.dense_features(args.input, size=args.size, mode=args.mode)
    res = OD.query_detect(feats, query_xy=tuple(args.xy), sim_thresh=args.thresh)
    heat = F.overlay_heatmap(feats.rgb, res["heat"])
    boxed = OD.draw_boxes(feats.rgb, res["boxes"])
    cv2.drawMarker(heat, tuple(args.xy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
    print(f"[demos] {len(res['boxes'])} region(s) match query at {tuple(args.xy)}")
    _save("detect.png", _panel(heat, boxed, labels=["query similarity", f"{len(res['boxes'])} boxes"]))


def cmd_cluster(bb, args):
    feats = bb.dense_features(args.input, size=args.size, mode=args.mode)
    res = OD.cluster_objects(feats, k=args.k)
    seg = F.upsample(res["labels"].astype(np.float32), feats.rgb.shape[:2], smooth=False).astype(int)
    seg_rgb = F.label_map_to_rgb(seg)
    blend = (0.5 * seg_rgb + 0.5 * feats.rgb).astype(np.uint8)
    boxed = feats.rgb.copy()
    palette = F.label_map_to_rgb(np.arange(args.k))
    for c, boxes in res["clusters"].items():
        boxed = OD.draw_boxes(boxed, boxes, color=tuple(int(v) for v in palette[c]))
    print(f"[demos] {args.k} clusters -> {sum(len(b) for b in res['clusters'].values())} proposal boxes")
    _save("cluster.png", _panel(blend, boxed, labels=[f"k={args.k} segments", "proposals"]))


def cmd_nav(bb, args):
    feats = bb.dense_features(args.input, size=args.size, mode=args.mode)
    free, sim = NAV.free_space_mask(feats, sim_thresh=args.thresh)
    clearance = NAV.obstacle_columns(free)
    plan = NAV.steer_suggestion(clearance)
    H, W = feats.rgb.shape[:2]
    free_up = F.upsample(free.astype(np.float32), (H, W), smooth=False)
    overlay = feats.rgb.copy()
    green = np.zeros_like(overlay); green[..., 1] = 255
    m = free_up > 0.5
    overlay[m] = (0.5 * overlay[m] + 0.5 * green[m]).astype(np.uint8)
    # draw clearance profile + steer arrow
    for x in range(W):
        cx = int(x / W * len(clearance))
        y = int(H - clearance[min(cx, len(clearance) - 1)] * H)
        cv2.line(overlay, (x, H), (x, y), (255, 255, 0), 1)
    tip = int((plan["offset"] * 0.5 + 0.5) * W)
    cv2.arrowedLine(overlay, (W // 2, H - 10), (tip, H - 60), (0, 0, 255), 3, tipLength=0.3)
    print(f"[demos] nav plan: action={plan['action']} offset={plan['offset']:+.2f} clearance={plan['clearance']:.2f}")
    _save("nav.png", _panel(feats.rgb, overlay, labels=["input", f"free-space | {plan['action']}"]))


def cmd_track(bb, args):
    import glob

    paths = sorted(glob.glob(str(Path(args.frames) / "*")))
    paths = [p for p in paths if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    if not paths:
        raise FileNotFoundError(f"no frames in {args.frames}")
    res = TR.track_point(bb, paths, tuple(args.xy), size=args.size, mode=args.mode, sim_thresh=args.thresh)
    frames_out = []
    for r in res:
        img = r["rgb"].copy()
        color = (0, 255, 0) if r["found"] else (0, 0, 255)
        cv2.circle(img, r["xy"], 10, color, 3)
        cv2.putText(img, f"f{r['frame']} sim={r['sim']:.2f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        frames_out.append(img)
    try:
        import imageio.v2 as imageio

        OUT.mkdir(parents=True, exist_ok=True)
        gif = OUT / "track.gif"
        imageio.mimsave(str(gif), frames_out, duration=0.2)
        print(f"[demos] wrote {gif} ({len(frames_out)} frames)")
    except Exception as e:  # noqa: BLE001
        print(f"[demos] gif skipped ({e}); writing strip instead")
    _save("track.png", _panel(*frames_out[: min(5, len(frames_out))]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="small", choices=["small", "base", "large", "giant"])
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--mode", default="square", choices=["square", "shortest"])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pca", help="dense-feature PCA visualization")
    p.set_defaults(fn=cmd_pca, input="data/example.png")
    p.add_argument("--input", default="data/example.png")

    p = sub.add_parser("detect", help="one-shot query detection")
    p.set_defaults(fn=cmd_detect)
    p.add_argument("--input", default="data/example.png")
    p.add_argument("--xy", type=int, nargs=2, default=[256, 256])
    p.add_argument("--thresh", type=float, default=0.5)

    p = sub.add_parser("cluster", help="unsupervised object proposals (KMeans)")
    p.set_defaults(fn=cmd_cluster)
    p.add_argument("--input", default="data/example.png")
    p.add_argument("--k", type=int, default=6)

    p = sub.add_parser("nav", help="free-space / traversability cue + steer")
    p.set_defaults(fn=cmd_nav)
    p.add_argument("--input", default="data/example.png")
    p.add_argument("--thresh", type=float, default=0.6)

    p = sub.add_parser("track", help="training-free point tracking across frames")
    p.set_defaults(fn=cmd_track)
    p.add_argument("--frames", default="data/frames")
    p.add_argument("--xy", type=int, nargs=2, default=[256, 256])
    p.add_argument("--thresh", type=float, default=0.5)

    args = ap.parse_args()
    print(f"[demos] loading LingBot-Vision variant={args.variant} device={args.device or 'auto'} ...")
    bb = lbv_nav.load(variant=args.variant, device=args.device)
    print(f"[demos] loaded: patch_size={bb.patch_size} embed_dim={bb.embed_dim} device={bb.device} dtype={bb.torch_dtype}")
    args.fn(bb, args)


if __name__ == "__main__":
    main()

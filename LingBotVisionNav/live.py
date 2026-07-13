"""Live webcam demo for the LingBot-Vision navigation sandbox.

Runs the frozen ViT-S/16 backbone on your webcam feed. Inference runs on a
background thread so the window and keyboard stay responsive even though the
model only manages a few frames per second on CPU.

    python live.py                 # default: PCA feature view
    python live.py --size 256      # smaller = faster
    python live.py --cam 1         # pick a different camera index

Keys (focus the video window):
    1 pca      dense-feature PCA blended over the frame
    2 detect   click a pixel -> highlight everything with similar features
    3 nav      free-space / traversability overlay + steer arrow
    4 track    click an object -> follow it across frames (template update)
    r          reset the query point
    q / ESC    quit

macOS note: the terminal / VSCode running this needs Camera permission
(System Settings -> Privacy & Security -> Camera). GUI calls (imshow/waitKey)
stay on the main thread, as macOS requires.
"""

from __future__ import annotations

import argparse
import threading
import time

import cv2
import numpy as np

import lbv_nav
from lbv_nav import features as F
from lbv_nav import navigation as NAV
from lbv_nav import object_discovery as OD

# Shared state between the main (GUI) thread and the inference worker.
STATE = {
    "query_xy": None,
    "query_vec": None,
    "mode": "pca",
    "latest_frame": None,  # RGB uint8 from the camera (written by main)
    "latest_disp": None,   # BGR uint8 to show (written by worker)
    "infer_fps": 0.0,
    "running": True,
}
LOCK = threading.Lock()
MODE_KEYS = {ord("1"): "pca", ord("2"): "detect", ord("3"): "nav", ord("4"): "track"}
LEGEND = {
    "pca": "PCA: similar stuff = similar color; watch edges pop",
    "detect": "DETECT: click something -> matches light up",
    "nav": "NAV: point at a FLOOR/hallway, not your face",
    "track": "TRACK: click a target -> circle follows it",
}


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        with LOCK:
            STATE["query_xy"] = (x, y)
            STATE["query_vec"] = None  # recompute descriptor on next frame


def _render(feats):
    """Build a BGR display image for the current mode."""
    rgb = feats.rgb
    mode = STATE["mode"]

    if mode == "pca":
        pca = F.pca_rgb(feats.grid)
        pca = F.upsample(pca.astype(np.float32), rgb.shape[:2], smooth=True).astype(np.uint8)
        out = (0.5 * rgb + 0.5 * pca).astype(np.uint8)

    elif mode == "detect":
        qxy = STATE["query_xy"]
        if qxy is None:
            out = rgb.copy()
        else:
            res = OD.query_detect(feats, query_xy=qxy, sim_thresh=0.5)
            out = F.overlay_heatmap(rgb, res["heat"], alpha=0.5)
            out = OD.draw_boxes(out, res["boxes"], color=(0, 255, 0), thickness=2)
            cv2.drawMarker(out, qxy, (255, 255, 255), cv2.MARKER_CROSS, 18, 2)

    elif mode == "nav":
        free, _ = NAV.free_space_mask(feats, sim_thresh=0.6)
        clearance = NAV.obstacle_columns(free)
        plan = NAV.steer_suggestion(clearance)
        H, W = rgb.shape[:2]
        free_up = F.upsample(free.astype(np.float32), (H, W), smooth=False) > 0.5
        out = rgb.copy()
        green = np.zeros_like(out); green[..., 1] = 255
        out[free_up] = (0.55 * out[free_up] + 0.45 * green[free_up]).astype(np.uint8)
        tip = int((plan["offset"] * 0.5 + 0.5) * W)
        cv2.arrowedLine(out, (W // 2, H - 12), (tip, H - 70), (0, 0, 255), 4, tipLength=0.3)
        cv2.putText(out, plan["action"], (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    elif mode == "track":
        out = rgb.copy()
        qxy = STATE["query_xy"]
        if qxy is not None:
            ngrid = feats.normalized_grid
            h, w = feats.hw
            H, W = rgb.shape[:2]
            if STATE["query_vec"] is None:
                gx = min(w - 1, int(qxy[0] / W * w))
                gy = min(h - 1, int(qxy[1] / H * h))
                STATE["query_vec"] = ngrid[gy, gx].copy()
            heat = F.cosine_heatmap(ngrid, STATE["query_vec"])
            gy, gx = np.unravel_index(int(np.argmax(heat)), heat.shape)
            sim = float(heat[gy, gx])
            px, py = int((gx + 0.5) / w * W), int((gy + 0.5) / h * H)
            STATE["query_xy"] = (px, py)
            color = (0, 255, 0) if sim >= 0.5 else (0, 0, 255)
            cv2.circle(out, (px, py), 12, color, 3)
            cv2.putText(out, f"sim={sim:.2f}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if sim >= 0.5:
                STATE["query_vec"] = 0.8 * STATE["query_vec"] + 0.2 * ngrid[gy, gx]
                STATE["query_vec"] /= np.linalg.norm(STATE["query_vec"]) + 1e-8
    else:
        out = rgb.copy()

    # on-screen legend + prompt
    if STATE["mode"] in ("detect", "track") and STATE["query_xy"] is None:
        cv2.putText(out, "click in the window", (10, out.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(out, LEGEND[mode], (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _worker(bb, size):
    """Continuously run inference on the latest camera frame."""
    last = time.time()
    while STATE["running"]:
        with LOCK:
            frame = STATE["latest_frame"]
        if frame is None:
            time.sleep(0.005)
            continue
        feats = bb.dense_features(frame, size=size, mode="square")
        disp = _render(feats)
        now = time.time()
        with LOCK:
            STATE["latest_disp"] = disp
            STATE["infer_fps"] = 0.9 * STATE["infer_fps"] + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="small", choices=["small", "base", "large", "giant"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--size", type=int, default=384, help="processing resolution (/16); lower = faster")
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    args = ap.parse_args()

    print(f"[live] loading LingBot-Vision variant={args.variant} ...")
    bb = lbv_nav.load(variant=args.variant, device=args.device)
    print(f"[live] loaded: patch_size={bb.patch_size} embed_dim={bb.embed_dim} device={bb.device}")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"[live] could not open camera {args.cam} (check Camera permission on macOS)")

    win = "LingBot-Vision live  [1]pca [2]detect [3]nav [4]track  r=reset q=quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _on_mouse)

    worker = threading.Thread(target=_worker, args=(bb, args.size), daemon=True)
    worker.start()

    print("[live] running. Focus the window; press q or ESC to quit.")
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("[live] frame grab failed; stopping.")
            break
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with LOCK:
            STATE["latest_frame"] = rgb
            disp = STATE["latest_disp"]
            ifps = STATE["infer_fps"]

        if disp is None:  # first frames before the worker has produced anything
            disp = frame_bgr.copy()
            cv2.putText(disp, "warming up...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            disp = disp.copy()
            cv2.putText(disp, f"{STATE['mode']}  {ifps:4.1f} fps", (disp.shape[1] - 190, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(win, disp)

        key = cv2.waitKey(1) & 0xFF  # main loop spins at camera rate -> snappy keys
        if key in (ord("q"), 27):
            break
        if key in MODE_KEYS:
            STATE["mode"] = MODE_KEYS[key]
            print(f"[live] mode -> {STATE['mode']}")
        elif key == ord("r"):
            with LOCK:
                STATE["query_xy"], STATE["query_vec"] = None, None

    STATE["running"] = False
    worker.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Live demo for the LingBot-Vision navigation sandbox.

Runs the frozen ViT-S/16 backbone on a live *or* recorded feed. Inference runs
on a background thread so the window and keyboard stay responsive even though
the model only manages a few frames per second on CPU.

The input is a flexible frame source (see ``lbv_nav/source.py``):

    python live.py                            # webcam index 0 (default)
    python live.py --source 1                 # a different camera index
    python live.py --source clip.mp4          # a recorded video file
    python live.py --source clip.mp4 --loop   # ...replaying at end
    python live.py --source screen            # whole primary screen
    python live.py --source screen:2          # a specific monitor
    python live.py --source screen:0,0,640,480  # a screen region (x,y,w,h)
    python live.py --region 0,0,640,480       # region via explicit flag
    python live.py --size 256                 # smaller = faster

Screen capture is a handy way to run detection/tracking on anything you can put
on screen (e.g. a YouTube clip playing in a browser) without committing to a
file format.

Keys (focus the video window):
    1 pca      dense-feature PCA blended over the frame
    2 detect   click a pixel -> highlight everything with similar features
    3 track    click an object -> follow it across frames (template update)
    r          reset the query point
    space      start / stop recording an MP4 into outputs/
    q / ESC    quit

macOS note: the terminal / VSCode running this needs Camera permission for
webcam sources and Screen Recording permission for screen sources (System
Settings -> Privacy & Security). GUI calls (imshow/waitKey) stay on the main
thread, as macOS requires.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).parent / "outputs"

import lbv_nav
from lbv_nav import features as F
from lbv_nav import object_discovery as OD
from lbv_nav.source import open_source, parse_region

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
MODE_KEYS = {ord("1"): "pca", ord("2"): "detect", ord("3"): "track"}
LEGEND = {
    "pca": "PCA: similar stuff = similar color; watch edges pop",
    "detect": "DETECT: click something -> matches light up",
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
    ap.add_argument(
        "--source",
        default="0",
        help="0/1=webcam index, path.mp4=video file, screen / screen:N / screen:x,y,w,h",
    )
    ap.add_argument("--cam", type=int, default=None, help="camera index (back-compat alias for --source)")
    ap.add_argument("--monitor", type=int, default=None, help="screen source: monitor index (1=primary)")
    ap.add_argument("--region", default=None, help="screen source: capture region 'x,y,w,h'")
    ap.add_argument("--loop", action="store_true", help="video file: replay when it ends")
    ap.add_argument("--fps", type=float, default=None, help="video file: override playback fps")
    args = ap.parse_args()

    print(f"[live] loading LingBot-Vision variant={args.variant} ...")
    bb = lbv_nav.load(variant=args.variant, device=args.device)
    print(f"[live] loaded: patch_size={bb.patch_size} embed_dim={bb.embed_dim} device={bb.device}")

    spec = str(args.cam) if args.cam is not None else args.source
    src = open_source(spec, monitor=args.monitor, region=parse_region(args.region), loop=args.loop)
    print(f"[live] source: kind={src.kind} live={src.is_live} fps={src.native_fps} size={src.size}")

    win = "LingBot-Vision  [1]pca [2]detect [3]track  r=reset SPACE=rec q=quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _on_mouse)

    worker = threading.Thread(target=_worker, args=(bb, args.size), daemon=True)
    worker.start()

    writer = None          # cv2.VideoWriter while recording, else None
    rec_path = None
    rec_frames = 0
    disp_fps = 20.0        # measured display rate, used as the recording fps
    last = time.time()

    print("[live] running. Focus the window; keys: 1-3 modes, r reset, SPACE record, q quit.")
    while True:
        rgb = src.read()
        if rgb is None:  # only non-live (file) sources reach genuine end-of-stream
            print("[live] end of stream; stopping.")
            break
        with LOCK:
            STATE["latest_frame"] = rgb
            disp = STATE["latest_disp"]
            ifps = STATE["infer_fps"]

        now = time.time()
        disp_fps = 0.9 * disp_fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now

        warming = disp is None
        if warming:  # first frames before the worker has produced anything
            disp = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(disp, "warming up...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            disp = disp.copy()
            cv2.putText(disp, f"{STATE['mode']}  {ifps:4.1f} fps", (disp.shape[1] - 190, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Write the clean annotated frame (no REC badge) to the recording.
        if writer is not None:
            writer.write(disp)
            rec_frames += 1

        # Draw the REC badge on the shown frame only.
        shown = disp
        if writer is not None:
            shown = disp.copy()
            cv2.circle(shown, (20, disp.shape[0] - 20), 8, (0, 0, 255), -1)
            cv2.putText(shown, f"REC {rec_frames}", (36, disp.shape[0] - 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(win, shown)

        # Live sources spin at capture rate (snappy keys). File sources are paced
        # toward their native fps so playback isn't a fast-forward -- but via
        # waitKey (which still pumps the GUI + returns keys), never time.sleep.
        if src.is_live:
            wait_ms = 1
        else:
            target = 1.0 / (args.fps or src.native_fps or 30.0)
            wait_ms = max(1, int((target - (time.time() - now)) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in MODE_KEYS:
            STATE["mode"] = MODE_KEYS[key]
            print(f"[live] mode -> {STATE['mode']}")
        elif key == ord("r"):
            with LOCK:
                STATE["query_xy"], STATE["query_vec"] = None, None
        elif key == ord(" "):  # space toggles recording
            if writer is None:
                if warming:
                    print("[live] still warming up; try again in a second")
                else:
                    OUT.mkdir(parents=True, exist_ok=True)
                    rec_path = OUT / f"live_{STATE['mode']}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    h, w = disp.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    # For files, use the true native fps so the clip's duration
                    # matches the source; for live feeds use the measured rate.
                    rec_fps = round(src.native_fps, 1) if (not src.is_live and src.native_fps) else round(disp_fps, 1)
                    writer = cv2.VideoWriter(str(rec_path), fourcc, rec_fps, (w, h))
                    rec_frames = 0
                    print(f"[live] ● recording -> {rec_path} @ {rec_fps:.0f} fps")
            else:
                writer.release()
                print(f"[live] ■ saved {rec_path} ({rec_frames} frames)")
                writer, rec_path, rec_frames = None, None, 0

    if writer is not None:
        writer.release()
        print(f"[live] ■ saved {rec_path} ({rec_frames} frames)")
    STATE["running"] = False
    worker.join(timeout=1.0)
    src.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

LingBotVisionNav
================

A sandbox for probing **[LingBot-Vision](https://github.com/robbyant/lingbot-vision)** —
Robbyant / Ant Group's open-source *boundary-centric* vision foundation model —
for capabilities relevant to **robot navigation**: object detection, tracking,
and spatial / traversability cues.

## Why this model

Most vision backbones (CLIP, classic ViTs) are trained for *semantic
invariance* and throw away fine-grained spatial structure. LingBot-Vision
inverts that: it is pretrained with **masked boundary modeling** on ~161M
images, so its frozen dense patch features stay *spatially structured* while
remaining semantically rich. That makes the features unusually good for dense
prediction — segmentation, depth, video object segmentation — which is exactly
what navigation stacks need. It is also the initializer for LingBot-Depth 2.0,
the seam where these 2D features get lifted to metric 3D.

Weights ship Apache-2.0 in four sizes (`vit-small/base/large/giant`, ViT-S/16
→ 1.1B-param ViT-g/16). This sandbox defaults to **ViT-S/16**, which runs on
CPU / Apple Silicon.

## What's here

| Capability | Module | Idea |
|---|---|---|
| Dense-feature PCA viz | `lbv_nav/features.py` | Sanity-check what the backbone "sees" |
| One-shot object detection | `lbv_nav/object_discovery.py` | Point at a thing → find all regions like it (cosine match → boxes) |
| Unsupervised object proposals | `lbv_nav/object_discovery.py` | KMeans over patch features → region proposals, no labels |
| Training-free tracking | `lbv_nav/tracking.py` | Re-localize a point across frames by feature matching (no tracker training) |

> A `navigation.py` free-space / steering experiment also lives in the package
> but is unwired from the demos for now.

Everything consumes one primitive — a `[h, w, C]` dense feature grid from
`lbv_nav.load(...).dense_features(image)`.

## Setup

```bash
cd LingBotVisionNav
./setup.sh          # clones upstream backbone into third_party/, installs deps
```

Model weights (~86 MB for small) auto-download from Hugging Face on first run
and cache under `~/.cache/huggingface`. Requires Python ≥ 3.10, torch ≥ 2.0.

## Usage

```bash
python demos.py pca      --input data/example.png
python demos.py detect   --input data/example.png --xy 256 256    # click a pixel
python demos.py cluster  --input data/example.png --k 6
python demos.py track    --frames data/frames --xy 256 256         # a folder of frames
```

All demos write panels to `outputs/`. Add `--variant base|large|giant` and
`--device cuda` on a GPU box; `--size` controls input resolution (snapped to a
multiple of the patch size, 16).

### Live webcam

Run any mode on your webcam in real time:

```bash
python live.py                 # PCA feature view (default)
python live.py --size 256      # smaller = faster
python live.py --cam 1         # different camera index
```

Keys (focus the video window): `1` pca · `2` detect (click an object) · `3`
track (click a target) · `r` reset · `space` start/stop recording · `q`/ESC
quit.

Press `space` to record the annotated view to a timestamped MP4 in `outputs/`
(e.g. `outputs/live_track_20260713_143022.mp4`); press `space` again to stop. A
red REC badge shows while recording (it is not baked into the saved file).

CPU throughput of the small variant on this machine: **~14 fps @ size 256**,
~6 fps @ 384, ~3 fps @ 512. Lower `--size` for smoother live video; a CUDA GPU
is much faster. On macOS the terminal / VSCode needs Camera permission
(System Settings → Privacy & Security → Camera).

Programmatic use:

```python
import lbv_nav
from lbv_nav import object_discovery as OD

bb = lbv_nav.load("small")                 # frozen ViT-S/16
feats = bb.dense_features("data/example.png")   # DenseFeatures: grid [h, w, C]
res = OD.query_detect(feats, query_xy=(256, 256))
print(len(res["boxes"]), "matching regions")
```

## Roadmap notes

- **Multi-object tracking**: `tracking.py` currently tracks a single point with
  a simple template update; extend to per-object mask propagation for MOT.
- **Navigation (parked for now)**: `lbv_nav/navigation.py` holds an early
  free-space / steering experiment, unwired from the demos. Reviving it and
  pairing the frozen features with **LingBot-Depth 2.0** for metric 3D is the
  natural path back to the "Nav" in the project name.

## Data

`data/example.png` is the upstream sample image (Apache-2.0). Drop your own
images in `data/` and a sequence of frames in `data/frames/` for the tracking
demo. Nothing large is committed (see repo-root `.gitignore`).

## Credits

Backbone & weights © Robbyant (灵波科技), Apache-2.0 —
paper *"Vision Pretraining for Dense Spatial Perception"* (arXiv:2607.05247),
[github.com/robbyant/lingbot-vision](https://github.com/robbyant/lingbot-vision),
[HF collection](https://huggingface.co/collections/robbyant/lingbot-vision).
This sandbox only adds thin wrappers and demos.

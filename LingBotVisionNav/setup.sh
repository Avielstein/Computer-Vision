#!/usr/bin/env bash
# Set up the LingBotVisionNav sandbox: fetch the upstream backbone package and
# its Python deps. Model weights (~86 MB for small) auto-download from Hugging
# Face on first run of demos.py and are cached under ~/.cache/huggingface.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY="$HERE/third_party"
REPO="$THIRD_PARTY/lingbot-vision"

mkdir -p "$THIRD_PARTY"

if [ ! -d "$REPO/.git" ]; then
  echo "[setup] cloning robbyant/lingbot-vision ..."
  git clone --depth 1 https://github.com/robbyant/lingbot-vision.git "$REPO"
else
  echo "[setup] lingbot-vision already cloned; pulling latest ..."
  git -C "$REPO" pull --ff-only || true
fi

echo "[setup] installing python dependencies ..."
python3 -m pip install -r "$HERE/requirements.txt"

echo "[setup] installing lingbot-vision (editable) ..."
python3 -m pip install -e "$REPO"

echo "[setup] done. Try:  cd LingBotVisionNav && python demos.py pca --input data/example.png"

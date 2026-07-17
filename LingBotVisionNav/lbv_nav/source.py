"""Flexible frame sources for the live sandbox.

One small interface -- :class:`FrameSource` -- that yields **RGB uint8** frames
from any of:

- a webcam / capture device (integer index),
- a local video file (``.mp4`` / ``.mov`` / ...),
- the screen (a whole monitor or a bounding-box region), via ``mss``.

``live.py`` consumes these behind :func:`open_source` so switching between live
and recorded input is just a CLI flag. Everything downstream already speaks RGB
numpy arrays (``LBVBackbone.dense_features`` accepts them), so nothing else has
to change.

    src = open_source("0")                 # webcam index 0
    src = open_source("clip.mp4")          # video file
    src = open_source("screen")            # whole primary screen
    src = open_source("screen:0,0,640,480")  # a region (x,y,w,h)

    while True:
        rgb = src.read()      # HxWx3 uint8, or None at end-of-file
        if rgb is None:
            break
        ...
    src.close()
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def parse_region(text: str | None) -> tuple[int, int, int, int] | None:
    """Parse an ``"x,y,w,h"`` string into a 4-tuple, or return None."""
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError(f"region must be 'x,y,w,h', got {text!r}")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


class FrameSource:
    """Base interface. Subclasses yield RGB uint8 ``HxWx3`` frames."""

    kind: str = "base"
    is_live: bool = True
    native_fps: float | None = None
    size: tuple[int, int] = (0, 0)  # (width, height)

    def read(self) -> np.ndarray | None:
        """Return the next RGB frame, or ``None`` at genuine end-of-stream.

        Live sources (camera/screen) never return ``None`` during normal
        operation; a hard failure raises instead.
        """
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _CaptureSource(FrameSource):
    """Shared cv2.VideoCapture plumbing for camera and file sources."""

    def __init__(self, target):
        self._cap = cv2.VideoCapture(target)
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.native_fps = fps if fps and fps > 0 else None
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size = (w, h)

    def _grab(self) -> np.ndarray | None:
        ok, frame_bgr = self._cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self._cap.release()


class CameraSource(_CaptureSource):
    """A live webcam / capture device by integer index."""

    kind = "camera"
    is_live = True

    def __init__(self, index: int):
        super().__init__(index)
        if not self._cap.isOpened():
            raise SystemExit(
                f"[source] could not open camera {index} "
                "(check Camera permission: System Settings -> Privacy & Security -> Camera)"
            )

    def read(self) -> np.ndarray | None:
        rgb = self._grab()
        if rgb is None:
            # A live camera shouldn't EOF; surface it as an error, matching
            # live.py's previous "frame grab failed; stopping" behavior.
            raise RuntimeError("[source] camera frame grab failed")
        return rgb


class VideoFileSource(_CaptureSource):
    """A recorded video file, optionally looping at end-of-stream."""

    kind = "file"
    is_live = False

    def __init__(self, path: str, loop: bool = False):
        if not Path(path).exists():
            raise SystemExit(f"[source] video file not found: {path}")
        super().__init__(path)
        if not self._cap.isOpened():
            raise SystemExit(f"[source] could not open video file: {path}")
        self._loop = loop

    def read(self) -> np.ndarray | None:
        rgb = self._grab()
        if rgb is None and self._loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            rgb = self._grab()
        return rgb


class ScreenSource(FrameSource):
    """Screen capture (a monitor or a region) via ``mss``.

    Thread-safety: an ``mss.mss()`` instance must be created and used on the
    *same* thread. ``live.py`` calls :meth:`read` on its main capture thread, so
    we lazily build the instance on the first ``read`` -- do NOT move this into
    ``__init__``, which may run on a different thread.
    """

    kind = "screen"
    is_live = True
    native_fps = None

    def __init__(self, monitor: int | None = None, region: tuple[int, int, int, int] | None = None):
        self._monitor = monitor
        self._region = region
        self._sct = None  # created lazily on first read (same-thread requirement)
        self._bbox: dict | None = None
        if region is not None:
            x, y, w, h = region
            self.size = (w, h)

    def _ensure(self) -> None:
        if self._sct is not None:
            return
        try:
            import mss  # imported here so the dep is only needed for screen capture
        except ModuleNotFoundError as e:
            raise SystemExit(
                "[source] screen capture needs the 'mss' package: pip install mss"
            ) from e
        try:
            self._sct = mss.mss()
            if self._region is not None:
                x, y, w, h = self._region
                self._bbox = {"left": x, "top": y, "width": w, "height": h}
            else:
                # mss monitors are 1-indexed: [0] is the union of all, [1] the primary.
                idx = self._monitor if self._monitor is not None else 1
                mon = self._sct.monitors[idx]
                self._bbox = mon
                self.size = (mon["width"], mon["height"])
            # Prime one grab so a permission failure surfaces immediately.
            self._sct.grab(self._bbox)
        except IndexError as e:
            raise SystemExit(
                f"[source] no such monitor {self._monitor} "
                f"(available: 1..{len(self._sct.monitors) - 1})"
            ) from e
        except Exception as e:  # noqa: BLE001 -- surface as a clear, actionable message
            raise SystemExit(
                "[source] screen capture failed. On macOS grant Screen Recording "
                "permission (System Settings -> Privacy & Security -> Screen Recording) "
                f"to the terminal / app running this. ({e})"
            ) from e

    def read(self) -> np.ndarray | None:
        self._ensure()
        raw = self._sct.grab(self._bbox)  # BGRA
        arr = np.asarray(raw)
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()


def open_source(
    spec: str | int | None,
    monitor: int | None = None,
    region: tuple[int, int, int, int] | None = None,
    loop: bool = False,
) -> FrameSource:
    """Open the right :class:`FrameSource` for ``spec``.

    - int or all-digit string (``"0"``) -> :class:`CameraSource`
    - ``"screen"`` / ``"screen:N"`` / ``"screen:x,y,w,h"`` -> :class:`ScreenSource`
      (explicit ``monitor`` / ``region`` args override the encoded form)
    - anything else -> :class:`VideoFileSource` (treated as a file path)
    """
    if spec is None:
        spec = "0"
    if isinstance(spec, int):
        return CameraSource(spec)

    text = str(spec).strip()

    if text.isdigit():
        return CameraSource(int(text))

    if text == "screen" or text.startswith("screen:"):
        enc_monitor: int | None = None
        enc_region: tuple[int, int, int, int] | None = None
        if text.startswith("screen:"):
            rest = text[len("screen:"):]
            if "," in rest:
                enc_region = parse_region(rest)
            elif rest:
                enc_monitor = int(rest)
        # Explicit CLI flags win over the string-encoded values.
        return ScreenSource(
            monitor=monitor if monitor is not None else enc_monitor,
            region=region if region is not None else enc_region,
        )

    return VideoFileSource(text, loop=loop)

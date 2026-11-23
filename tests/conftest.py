"""Shared test helpers: fake detectors, a fake PyAutoGUI and frame-context builders.

Nothing here needs a camera, a display or the optional AI dependencies.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from open_ai_hci.sensors._mediapipe import RawHand  # noqa: E402
from open_ai_hci.types import CursorState, FaceData, FrameContext, HandData  # noqa: E402

W, H = 1280, 720


def px_to_norm(px: float, size: int) -> float:
    """Normalised coordinate whose int-truncated pixel is exactly ``px`` (centre of the pixel)."""
    return (px + 0.5) / size


def hand_landmarks(
    index_px: tuple[float, float],
    thumb_px: tuple[float, float],
    w: int = W,
    h: int = H,
    middle_px: tuple[float, float] | None = None,
) -> np.ndarray:
    lm = np.zeros((21, 3))
    lm[:, 0], lm[:, 1] = 0.5, 0.5
    lm[0] = (0.5, 0.9, 0.0)  # wrist
    lm[9] = (0.5, 0.6, 0.0)  # middle MCP (gives the feature scale something to chew on)
    lm[8] = (px_to_norm(index_px[0], w), px_to_norm(index_px[1], h), 0.0)
    lm[4] = (px_to_norm(thumb_px[0], w), px_to_norm(thumb_px[1], h), 0.0)
    # middle fingertip: by default parked far away from any thumb position used in the tests
    mid = middle_px if middle_px is not None else (w * 0.95, h * 0.05)
    lm[12] = (px_to_norm(mid[0], w), px_to_norm(mid[1], h), 0.0)
    return lm


def palm_landmarks() -> np.ndarray:
    """An open palm: wrist low, all four fingertips well beyond their PIP joints (thumb away from the index)."""
    lm = np.zeros((21, 3))
    lm[:, 0], lm[:, 1] = 0.5, 0.5
    lm[0] = (0.5, 0.9, 0.0)
    lm[9] = (0.5, 0.6, 0.0)
    for k, (tip, pip) in enumerate(((8, 6), (12, 10), (16, 14), (20, 18))):
        x = 0.40 + 0.07 * k
        lm[pip] = (x, 0.55, 0.0)
        lm[tip] = (x, 0.25, 0.0)
    lm[4] = (0.15, 0.6, 0.0)  # thumb far from the fingertips
    return lm


def fist_landmarks() -> np.ndarray:
    """A fist: every fingertip nearer to the wrist than its PIP joint; thumb far from the fingertips."""
    lm = palm_landmarks()
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        lm[tip] = (lm[pip][0], 0.75, 0.0)
    lm[4] = (0.9, 0.9, 0.0)
    return lm


def make_palm(handedness="Right") -> HandData:
    return HandData(0, handedness, 0.99, palm_landmarks(), W, H)


def make_fist(handedness="Right") -> HandData:
    return HandData(0, handedness, 0.99, fist_landmarks(), W, H)


def make_hand(index_px=(600, 300), thumb_px=(600, 400), handedness="Right", w: int = W, h: int = H, middle_px=None) -> HandData:
    return HandData(0, handedness, 0.99, hand_landmarks(index_px, thumb_px, w, h, middle_px), w, h)


def make_face(nose_px=(640, 360), w: int = W, h: int = H) -> FaceData:
    lm = np.full((478, 3), 0.5)
    lm[1] = (px_to_norm(nose_px[0], w), px_to_norm(nose_px[1], h), 0.0)
    return FaceData(0, lm, w, h)


def make_ctx(
    hands=(), faces=(), t: float = 0.0, cursor: CursorState | None = None, w: int = W, h: int = H, frame_index: int = 1
) -> FrameContext:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    return FrameContext(
        frame=frame, rgb=frame.copy(), width=w, height=h, frame_index=frame_index, timestamp=t,
        hands=list(hands), faces=list(faces), cursor=cursor,
    )


class FakeHandBackend:
    """Scripted ``detect()``: pops one entry per call (``None`` = no hand, or (index_px, thumb_px, handedness))."""

    def __init__(self, script):
        self.script = list(script)
        self.closed = False

    def detect(self, rgb, ts_ms):
        item = self.script.pop(0) if self.script else None
        if item is None:
            return []
        index_px, thumb_px = item[0], item[1]
        handed = item[2] if len(item) > 2 else "Right"
        h, w = rgb.shape[:2]
        return [RawHand(handed, 0.99, hand_landmarks(index_px, thumb_px, w, h))]

    def close(self):
        self.closed = True


class FakeFaceBackend:
    """Scripted ``detect()``: one entry per call (``None`` = no face, else nose pixel)."""

    def __init__(self, script):
        self.script = list(script)

    def detect(self, rgb, ts_ms):
        item = self.script.pop(0) if self.script else None
        if item is None:
            return []
        h, w = rgb.shape[:2]
        return [make_face(item, w, h).landmarks]

    def close(self):
        pass


class FakePyAutoGUI:
    """Records calls; mimics the parts of PyAutoGUI the actions use."""

    class FailSafeException(Exception):
        pass

    def __init__(self, size=(1920, 1080)):
        self.FAILSAFE = True
        self.PAUSE = 0.0
        self._size = size
        self.calls: list[tuple[str, tuple, dict]] = []
        self.raise_on: str | None = None

    def size(self):
        return self._size

    def _rec(self, name, *a, **k):
        if self.raise_on == name:
            raise self.FailSafeException("fail-safe triggered")
        self.calls.append((name, a, k))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: self._rec(name, *a, **k)

    def names(self):
        return [c[0] for c in self.calls]


@pytest.fixture
def fake_pag():
    return FakePyAutoGUI()


@pytest.fixture
def bare_engine():
    """An engine with an empty pipeline: just a bus and a config to attach components to."""
    from open_ai_hci import HCIConfig, HCIEngine

    cfg = HCIConfig()
    cfg.screen.width, cfg.screen.height = 1920, 1080
    return HCIEngine(cfg, sensors=[], processors=[], actions=[], raise_errors=True)


@pytest.fixture
def collect():
    """``events, handler = collect()``; subscribe ``handler`` to anything."""

    def factory():
        events: list[Any] = []
        return events, events.append

    return factory


def names(events) -> list[str]:
    return [e.name for e in events]


ns = SimpleNamespace  # tiny convenience for ad-hoc fakes

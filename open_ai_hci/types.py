"""Shared data types: per-frame context, sensor outputs and landmark indices.

All coordinates stored in ``landmarks`` arrays are *normalised* image
coordinates (``x`` and ``y`` in ``[0, 1]``, ``z`` relative depth). Pixel
coordinates are derived on demand through :meth:`HandData.px` and friends,
using ``int(norm * size)`` truncation exactly like the original prototype.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

import numpy as np

__all__ = [
    "HandLandmark",
    "HAND_CONNECTIONS",
    "POSE_CONNECTIONS",
    "FACE_NOSE_TIP",
    "HandData",
    "HeadPose",
    "FaceData",
    "PoseData",
    "CursorState",
    "FrameContext",
    "jsonable",
]

FACE_NOSE_TIP = 1  # FaceMesh landmark used for head flicks


class HandLandmark(IntEnum):
    """The 21 MediaPipe hand landmarks."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)


def _px(landmarks: np.ndarray, idx: int, width: int, height: int) -> tuple[int, int]:
    lm = landmarks[int(idx)]
    return int(lm[0] * width), int(lm[1] * height)


@dataclass
class HandData:
    """One tracked hand.

    Attributes:
        index: Position in the detector output (0 = primary hand after sorting).
        handedness: ``"Left"`` or ``"Right"`` *as seen by the user*. MediaPipe
            assumes a mirrored (selfie) image, which is what the engine feeds it.
        score: Handedness classification confidence.
        landmarks: ``(21, 3)`` float64 array of normalised landmarks.
        frame_width: Width in pixels of the frame the landmarks belong to.
        frame_height: Height in pixels of that frame.
    """

    index: int
    handedness: str
    score: float
    landmarks: np.ndarray
    frame_width: int
    frame_height: int

    def px(self, idx: int) -> tuple[int, int]:
        """Pixel coordinates (truncated ints) of landmark ``idx``."""
        return _px(self.landmarks, idx, self.frame_width, self.frame_height)

    @property
    def index_tip(self) -> tuple[int, int]:
        return self.px(HandLandmark.INDEX_TIP)

    @property
    def thumb_tip(self) -> tuple[int, int]:
        return self.px(HandLandmark.THUMB_TIP)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "handedness": self.handedness,
            "score": self.score,
            "landmarks": self.landmarks,
        }


@dataclass
class HeadPose:
    """Head orientation in degrees.

    ``yaw > 0``: face turns towards the right of the image.
    ``pitch > 0``: face tilts up. ``roll > 0``: head tilts clockwise on screen.
    """

    yaw: float
    pitch: float
    roll: float


@dataclass
class FaceData:
    """One tracked face (MediaPipe FaceMesh / FaceLandmarker topology)."""

    index: int
    landmarks: np.ndarray
    frame_width: int
    frame_height: int
    ear_left: float | None = None
    ear_right: float | None = None
    head_pose: HeadPose | None = None

    def px(self, idx: int) -> tuple[int, int]:
        return _px(self.landmarks, idx, self.frame_width, self.frame_height)

    @property
    def nose(self) -> tuple[int, int]:
        """Pixel position of the nose tip (landmark 1)."""
        return self.px(FACE_NOSE_TIP)

    @property
    def ear(self) -> float | None:
        """Mean eye aspect ratio, if blink analysis is enabled."""
        if self.ear_left is None or self.ear_right is None:
            return None
        return (self.ear_left + self.ear_right) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "nose": self.nose,
            "ear_left": self.ear_left,
            "ear_right": self.ear_right,
            "head_pose": self.head_pose,
        }


@dataclass
class PoseData:
    """Full-body pose (33 MediaPipe pose landmarks)."""

    index: int
    landmarks: np.ndarray
    visibility: np.ndarray
    frame_width: int
    frame_height: int

    def px(self, idx: int) -> tuple[int, int]:
        return _px(self.landmarks, idx, self.frame_width, self.frame_height)

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "landmarks": self.landmarks, "visibility": self.visibility}


@dataclass
class CursorState:
    """Smoothed screen-space cursor produced by the smoothing processor.

    ``active`` is ``False`` when the hand is currently lost and ``x``/``y`` are
    the last known position.
    """

    x: float
    y: float
    raw_x: float
    raw_y: float
    active: bool = True


@dataclass
class FrameContext:
    """Everything known about one processed frame.

    Sensors fill ``hands`` / ``faces`` / ``poses``; processors fill ``cursor``,
    ``pinch_status`` and friends; the HUD and actions read from it.
    """

    frame: np.ndarray  # BGR, already mirrored; the HUD draws on this
    rgb: np.ndarray  # RGB copy used by detectors
    width: int
    height: int
    frame_index: int
    timestamp: float  # time.monotonic() seconds
    hands: list[HandData] = field(default_factory=list)
    faces: list[FaceData] = field(default_factory=list)
    poses: list[PoseData] = field(default_factory=list)
    cursor: CursorState | None = None
    pinch_status: str = "DISCONNECTED"
    pinch_distance: float | None = None
    scroll_mode: bool = False
    paused: bool = False  # cursor control paused by the pause gesture / engine.pause()
    gesture: str | None = None
    gesture_confidence: float = 0.0
    key: int = -1  # last cv2.waitKey() code (HUD mode only), -1 if none
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def hand(self) -> HandData | None:
        """Primary hand, or ``None``."""
        return self.hands[0] if self.hands else None

    @property
    def face(self) -> FaceData | None:
        """Primary face, or ``None``."""
        return self.faces[0] if self.faces else None

    @property
    def pose(self) -> PoseData | None:
        return self.poses[0] if self.poses else None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly summary (never includes image buffers)."""
        return {
            "frame": self.frame_index,
            "width": self.width,
            "height": self.height,
            "hands": self.hands,
            "faces": self.faces,
            "poses": self.poses,
            "cursor": self.cursor,
            "pinch_status": self.pinch_status,
            "pinch_distance": self.pinch_distance,
            "scroll_mode": self.scroll_mode,
            "gesture": self.gesture,
        }


def jsonable(value: Any, precision: int = 4) -> Any:
    """Recursively convert ``value`` into plain JSON types.

    Handles NumPy arrays/scalars, dataclasses, enums, objects with ``to_dict``,
    and replaces NaN/inf with ``None``.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Enum):
        return jsonable(value.value, precision)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return round(f, precision) if math.isfinite(f) else None
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist(), precision)
    if isinstance(value, dict):
        return {str(k): jsonable(v, precision) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v, precision) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return jsonable(value.to_dict(), precision)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name), precision) for f in dataclasses.fields(value)}
    return str(value)

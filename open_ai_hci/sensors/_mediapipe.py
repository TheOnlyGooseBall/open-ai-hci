"""MediaPipe backends for the hand / face / pose sensors.

Two interchangeable implementations exist for each detector:

* **legacy**: the classic ``mp.solutions.*`` API. Models are bundled in the
  wheel. This is what the original prototype used. MediaPipe wheels from
  0.10.30 on no longer ship it.
* **tasks**: the MediaPipe Tasks API (``HandLandmarker`` & co.), which needs
  ``.task`` model files (see :mod:`open_ai_hci._models`).

``MediaPipeConfig.backend = "auto"`` picks *legacy* when it exists, otherwise
*tasks*. Sensors only talk to the small ``detect(rgb, timestamp_ms)`` interface
defined here, so the rest of the framework is backend-agnostic.

MediaPipe is imported lazily (inside the factories), so importing
``open_ai_hci`` stays fast and works on machines without a camera stack.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._models import ensure_model
from ..config import FaceConfig, HandConfig, MediaPipeConfig, PoseConfig

__all__ = [
    "RawHand",
    "RawPose",
    "resolve_backend",
    "create_hand_backend",
    "create_face_backend",
    "create_pose_backend",
]

logger = logging.getLogger("open_ai_hci")


@dataclass
class RawHand:
    label: str
    score: float
    landmarks: np.ndarray  # (21, 3) float64, normalised


@dataclass
class RawPose:
    landmarks: np.ndarray  # (33, 3)
    visibility: np.ndarray  # (33,)


class _Backend(ABC):
    @abstractmethod
    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list[Any]: ...

    def close(self) -> None:  # pragma: no cover - trivial
        pass


def _import_mediapipe() -> Any:
    try:
        import mediapipe as mp  # noqa: WPS433 - deliberate lazy import
    except ImportError as exc:  # pragma: no cover - environment specific
        raise ImportError(
            "MediaPipe is required for camera tracking. Install it with: pip install mediapipe"
        ) from exc
    return mp


def resolve_backend(mp_cfg: MediaPipeConfig) -> str:
    """Return ``"legacy"`` or ``"tasks"`` for the installed MediaPipe."""
    mp = _import_mediapipe()
    has_legacy = hasattr(mp, "solutions")
    if mp_cfg.backend == "legacy":
        if not has_legacy:
            raise RuntimeError(
                f"MediaPipe {getattr(mp, '__version__', '?')} no longer ships the legacy mp.solutions API. "
                "Use mediapipe.backend='tasks' (or 'auto'), or install mediapipe<=0.10.21."
            )
        return "legacy"
    if mp_cfg.backend == "tasks":
        return "tasks"
    return "legacy" if has_legacy else "tasks"


def _lm_array(landmarks: Any) -> np.ndarray:
    return np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float64)


class _Monotonic:
    """Tasks VIDEO mode needs strictly increasing integer millisecond timestamps."""

    def __init__(self) -> None:
        self._last = -1

    def __call__(self, ts_ms: int) -> int:
        self._last = max(self._last + 1, int(ts_ms))
        return self._last


# ------------------------------------------------------------------------------ legacy
class _LegacyHands(_Backend):
    def __init__(self, cfg: HandConfig) -> None:
        mp = _import_mediapipe()
        self._det = mp.solutions.hands.Hands(
            max_num_hands=cfg.max_hands,
            min_detection_confidence=cfg.detection_confidence,
            min_tracking_confidence=cfg.tracking_confidence,
            model_complexity=cfg.model_complexity,
        )

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list[RawHand]:
        res = self._det.process(rgb)
        out: list[RawHand] = []
        if res.multi_hand_landmarks:
            handed = res.multi_handedness or []
            for i, lms in enumerate(res.multi_hand_landmarks):
                label, score = "Unknown", 0.0
                if i < len(handed):
                    cls = handed[i].classification[0]
                    label, score = cls.label, float(cls.score)
                out.append(RawHand(label, score, _lm_array(lms.landmark)))
        return out

    def close(self) -> None:
        self._det.close()


class _LegacyFace(_Backend):
    def __init__(self, cfg: FaceConfig) -> None:
        mp = _import_mediapipe()
        self._det = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=cfg.max_faces,
            min_detection_confidence=cfg.detection_confidence,
            min_tracking_confidence=cfg.tracking_confidence,
        )

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list[np.ndarray]:
        res = self._det.process(rgb)
        if not res.multi_face_landmarks:
            return []
        return [_lm_array(f.landmark) for f in res.multi_face_landmarks]

    def close(self) -> None:
        self._det.close()


class _LegacyPose(_Backend):
    def __init__(self, cfg: PoseConfig) -> None:
        mp = _import_mediapipe()
        self._det = mp.solutions.pose.Pose(
            model_complexity=cfg.model_complexity,
            min_detection_confidence=cfg.detection_confidence,
            min_tracking_confidence=cfg.tracking_confidence,
        )

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list[RawPose]:
        res = self._det.process(rgb)
        if not res.pose_landmarks:
            return []
        lms = res.pose_landmarks.landmark
        return [RawPose(_lm_array(lms), np.array([lm.visibility for lm in lms], dtype=np.float64))]

    def close(self) -> None:
        self._det.close()


# ------------------------------------------------------------------------------- tasks
def _tasks_modules() -> tuple[Any, Any, Any]:
    mp = _import_mediapipe()
    from mediapipe.tasks import python as mp_python  # noqa: WPS433
    from mediapipe.tasks.python import vision  # noqa: WPS433

    return mp, mp_python, vision


def _mp_image(mp: Any, rgb: np.ndarray) -> Any:
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


class _TasksHands(_Backend):
    def __init__(self, cfg: HandConfig, mp_cfg: MediaPipeConfig) -> None:
        self._mp, mp_python, vision = _tasks_modules()
        path = ensure_model("hand", mp_cfg.hand_model_path, mp_cfg.model_dir, mp_cfg.auto_download)
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.max_hands,
            min_hand_detection_confidence=cfg.detection_confidence,
            min_hand_presence_confidence=cfg.tracking_confidence,
            min_tracking_confidence=cfg.tracking_confidence,
        )
        self._det = vision.HandLandmarker.create_from_options(options)
        self._ts = _Monotonic()

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list[RawHand]:
        res = self._det.detect_for_video(_mp_image(self._mp, rgb), self._ts(timestamp_ms))
        out: list[RawHand] = []
        for i, lms in enumerate(res.hand_landmarks or []):
            label, score = "Unknown", 0.0
            if res.handedness and i < len(res.handedness) and res.handedness[i]:
                cat = res.handedness[i][0]
                label, score = cat.category_name, float(cat.score)
            out.append(RawHand(label, score, _lm_array(lms)))
        return out

    def close(self) -> None:
        self._det.close()


class _TasksFace(_Backend):
    def __init__(self, cfg: FaceConfig, mp_cfg: MediaPipeConfig) -> None:
        self._mp, mp_python, vision = _tasks_modules()
        path = ensure_model("face", mp_cfg.face_model_path, mp_cfg.model_dir, mp_cfg.auto_download)
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=cfg.max_faces,
            min_face_detection_confidence=cfg.detection_confidence,
            min_face_presence_confidence=cfg.tracking_confidence,
            min_tracking_confidence=cfg.tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._det = vision.FaceLandmarker.create_from_options(options)
        self._ts = _Monotonic()

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list[np.ndarray]:
        res = self._det.detect_for_video(_mp_image(self._mp, rgb), self._ts(timestamp_ms))
        return [_lm_array(f) for f in (res.face_landmarks or [])]

    def close(self) -> None:
        self._det.close()


class _TasksPose(_Backend):
    _VARIANT = {0: "pose_lite", 1: "pose_full", 2: "pose_heavy"}

    def __init__(self, cfg: PoseConfig, mp_cfg: MediaPipeConfig) -> None:
        self._mp, mp_python, vision = _tasks_modules()
        kind = self._VARIANT.get(cfg.model_complexity, "pose_full")
        path = ensure_model(kind, mp_cfg.pose_model_path, mp_cfg.model_dir, mp_cfg.auto_download)
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=cfg.max_poses,
            min_pose_detection_confidence=cfg.detection_confidence,
            min_pose_presence_confidence=cfg.tracking_confidence,
            min_tracking_confidence=cfg.tracking_confidence,
        )
        self._det = vision.PoseLandmarker.create_from_options(options)
        self._ts = _Monotonic()

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list[RawPose]:
        res = self._det.detect_for_video(_mp_image(self._mp, rgb), self._ts(timestamp_ms))
        out: list[RawPose] = []
        for lms in res.pose_landmarks or []:
            vis = np.array([getattr(lm, "visibility", 0.0) or 0.0 for lm in lms], dtype=np.float64)
            out.append(RawPose(_lm_array(lms), vis))
        return out

    def close(self) -> None:
        self._det.close()


# ------------------------------------------------------------------------------ factories
def create_hand_backend(cfg: HandConfig, mp_cfg: MediaPipeConfig) -> _Backend:
    kind = resolve_backend(mp_cfg)
    logger.info("Hand tracking backend: %s", kind)
    return _LegacyHands(cfg) if kind == "legacy" else _TasksHands(cfg, mp_cfg)


def create_face_backend(cfg: FaceConfig, mp_cfg: MediaPipeConfig) -> _Backend:
    kind = resolve_backend(mp_cfg)
    logger.info("Face tracking backend: %s", kind)
    return _LegacyFace(cfg) if kind == "legacy" else _TasksFace(cfg, mp_cfg)


def create_pose_backend(cfg: PoseConfig, mp_cfg: MediaPipeConfig) -> _Backend:
    kind = resolve_backend(mp_cfg)
    logger.info("Pose tracking backend: %s", kind)
    return _LegacyPose(cfg) if kind == "legacy" else _TasksPose(cfg, mp_cfg)

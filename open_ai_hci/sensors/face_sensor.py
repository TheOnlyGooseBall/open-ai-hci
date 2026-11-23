"""MediaPipe face sensor: nose tracking, blink (EAR) and head pose."""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from ..config import FaceConfig
from ..types import FaceData, FrameContext, HeadPose
from ._mediapipe import create_face_backend
from .base import BaseSensor

__all__ = ["FaceSensor", "eye_aspect_ratio", "estimate_head_pose", "head_angles_from_rotation"]

# FaceMesh landmark indices (shared by FaceMesh and the Tasks FaceLandmarker).
RIGHT_EYE = (33, 160, 158, 133, 153, 144)  # subject's right eye: p1, p2, p3, p4, p5, p6
LEFT_EYE = (362, 385, 387, 263, 373, 380)

# solvePnP correspondences: nose tip, chin, outer eye corners, mouth corners.
HEAD_POSE_LANDMARKS = (1, 152, 33, 263, 61, 291)
# Generic head model (millimetres) in camera-style axes: x right, y down, z away from the camera.
_HEAD_MODEL = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip
        (0.0, 330.0, 65.0),  # chin
        (-225.0, -170.0, 135.0),  # landmark 33  (image-left eye corner)
        (225.0, -170.0, 135.0),  # landmark 263 (image-right eye corner)
        (-150.0, 150.0, 125.0),  # landmark 61  (image-left mouth corner)
        (150.0, 150.0, 125.0),  # landmark 291 (image-right mouth corner)
    ],
    dtype=np.float64,
)


def eye_aspect_ratio(points: np.ndarray) -> float:
    """Eye aspect ratio from six eye points ``p1..p6`` (pixel coordinates, shape ``(6, 2)``).

    ``EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)``. Roughly 0.25-0.35 for an open
    eye and below ~0.2 when closed.
    """
    p = np.asarray(points, dtype=np.float64)
    horizontal = np.linalg.norm(p[0] - p[3])
    if horizontal < 1e-9:
        return 0.0
    return float((np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2.0 * horizontal))


def head_angles_from_rotation(rotation: np.ndarray) -> HeadPose:
    """Convert a head->camera rotation matrix to yaw/pitch/roll in degrees.

    Uses the direction the nose points (``-z`` of the head frame) for yaw and
    pitch and the head's right axis for roll, which avoids Euler-angle
    ambiguities. Conventions: yaw > 0 turns right in the image, pitch > 0 looks
    up, roll > 0 tilts clockwise on screen.
    """
    r = np.asarray(rotation, dtype=np.float64)
    forward = r @ np.array([0.0, 0.0, -1.0])
    right = r @ np.array([1.0, 0.0, 0.0])
    yaw = math.degrees(math.atan2(forward[0], -forward[2]))
    pitch = math.degrees(math.atan2(-forward[1], math.hypot(forward[0], forward[2])))
    roll = math.degrees(math.atan2(right[1], right[0]))
    return HeadPose(yaw=yaw, pitch=pitch, roll=roll)


def estimate_head_pose(points_px: np.ndarray, width: int, height: int) -> HeadPose | None:
    """Estimate head pose from the six :data:`HEAD_POSE_LANDMARKS` (pixel coords, ``(6, 2)``).

    Returns ``None`` if the PnP solve fails or yields a pose behind the camera.
    """
    image_pts = np.asarray(points_px, dtype=np.float64).reshape(6, 2)
    focal = float(width)
    camera = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(_HEAD_MODEL, image_pts, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or tvec[2, 0] <= 0:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    return head_angles_from_rotation(rotation)


class FaceSensor(BaseSensor):
    """Tracks faces with FaceMesh and derives optional blink / head-pose signals.

    Always provides the 468+ normalised landmarks (``FaceData.nose`` is landmark
    1, used for the head-flick gesture). With ``FaceConfig.enable_blink`` it also
    fills ``ear_left`` / ``ear_right``; with ``enable_head_pose`` it fills
    ``head_pose`` (yaw/pitch/roll in degrees).

    Emits ``face_detected`` / ``face_lost``.
    """

    name = "face"

    def __init__(self, config: FaceConfig | None = None, *, backend: Any = None) -> None:
        super().__init__()
        self._cfg = config
        self._backend = backend

    @property
    def cfg(self) -> FaceConfig:
        return self._cfg if self._cfg is not None else self.config.face

    def setup(self) -> None:
        if self._backend is None:
            self._backend = create_face_backend(self.cfg, self.config.mediapipe)

    def process(self, ctx: FrameContext) -> None:
        cfg = self.cfg
        faces: list[FaceData] = []
        for i, landmarks in enumerate(self._backend.detect(ctx.rgb, int(ctx.timestamp * 1000))):
            face = FaceData(index=i, landmarks=landmarks, frame_width=ctx.width, frame_height=ctx.height)
            if cfg.enable_blink:
                face.ear_right = eye_aspect_ratio(self._pixels(face, RIGHT_EYE))
                face.ear_left = eye_aspect_ratio(self._pixels(face, LEFT_EYE))
            if cfg.enable_head_pose:
                face.head_pose = estimate_head_pose(self._pixels(face, HEAD_POSE_LANDMARKS), ctx.width, ctx.height)
            faces.append(face)
        ctx.faces = faces
        self.emit_presence("face", bool(faces), count=len(faces))

    @staticmethod
    def _pixels(face: FaceData, indices: tuple[int, ...]) -> np.ndarray:
        pts = face.landmarks[list(indices), :2]
        return pts * np.array([face.frame_width, face.frame_height], dtype=np.float64)

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None

"""Landmark feature extraction shared by the recorder tool and the AI classifiers.

Keeping this in one dependency-free module guarantees that the features a model
is *trained* on (``tools/record_gestures.py``) are bit-for-bit what it sees at
*inference* time (:class:`~open_ai_hci.processors.ai.ONNXGestureClassifier`).
"""
from __future__ import annotations

import numpy as np

__all__ = ["FEATURE_DIM", "normalize_hand", "landmarks_to_features"]

FEATURE_DIM = 63  # 21 landmarks x (x, y, z)

_WRIST = 0
_MIDDLE_MCP = 9


def normalize_hand(landmarks: np.ndarray, handedness: str | None = None, mirror_left: bool = True) -> np.ndarray:
    """Make hand landmarks translation-, scale- and (optionally) handedness-invariant.

    Steps: subtract the wrist, divide by the wrist -> middle-finger-MCP distance
    (hand size), and, when ``handedness == "Left"`` and ``mirror_left`` is set,
    negate x so every hand looks like a right hand.

    Args:
        landmarks: ``(21, 3)`` normalised landmarks.

    Returns:
        ``(21, 3)`` float32 array. Degenerate input (all points identical) comes back as zeros.
    """
    lm = np.asarray(landmarks, dtype=np.float64).reshape(21, 3)
    rel = lm - lm[_WRIST]
    scale = float(np.linalg.norm(rel[_MIDDLE_MCP, :2]))
    if scale < 1e-6:
        return np.zeros((21, 3), dtype=np.float32)
    rel = rel / scale
    if mirror_left and handedness == "Left":
        rel[:, 0] = -rel[:, 0]
    return rel.astype(np.float32)


def landmarks_to_features(landmarks: np.ndarray, handedness: str | None = None, mirror_left: bool = True) -> np.ndarray:
    """Flat ``(63,)`` float32 feature vector for a classifier (see :func:`normalize_hand`)."""
    return normalize_hand(landmarks, handedness, mirror_left).reshape(-1)

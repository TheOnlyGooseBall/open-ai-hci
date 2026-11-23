import math

import cv2
import numpy as np
import pytest
from conftest import hand_landmarks

from open_ai_hci.features import FEATURE_DIM, landmarks_to_features, normalize_hand
from open_ai_hci.sensors.face_sensor import (
    _HEAD_MODEL,
    estimate_head_pose,
    eye_aspect_ratio,
    head_angles_from_rotation,
)


# ------------------------------------------------------------------------------- blink
def _eye(open_ratio: float) -> np.ndarray:
    """Six eye points (p1..p6) with a given vertical opening."""
    w = 30.0
    v = open_ratio * w / 2.0
    return np.array([(0, 0), (w * 0.3, -v), (w * 0.7, -v), (w, 0), (w * 0.7, v), (w * 0.3, v)], dtype=float)


def test_eye_aspect_ratio_open_vs_closed():
    assert eye_aspect_ratio(_eye(0.30)) == pytest.approx(0.30)
    assert eye_aspect_ratio(_eye(0.05)) < 0.21 < eye_aspect_ratio(_eye(0.30))
    assert eye_aspect_ratio(np.zeros((6, 2))) == 0.0  # degenerate: no division by zero


# ---------------------------------------------------------------------------- head pose
def _rot(axis: str, deg: float) -> np.ndarray:
    rvec = np.zeros(3)
    rvec["xyz".index(axis)] = math.radians(deg)
    return cv2.Rodrigues(rvec)[0]


def test_head_angles_signs_and_magnitudes():
    p = head_angles_from_rotation(np.eye(3))
    assert (p.yaw, p.pitch, p.roll) == pytest.approx((0, 0, 0), abs=1e-9)
    assert head_angles_from_rotation(_rot("y", -20)).yaw == pytest.approx(20)  # nose towards image right
    assert head_angles_from_rotation(_rot("y", 20)).yaw == pytest.approx(-20)
    assert head_angles_from_rotation(_rot("x", -15)).pitch == pytest.approx(15)  # looking up
    assert head_angles_from_rotation(_rot("z", 10)).roll == pytest.approx(10)  # clockwise on screen


def test_head_pose_recovered_from_projected_landmarks():
    w, h = 1280, 720
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=float)
    for yaw, pitch, roll in [(0, 0, 0), (-25, 0, 0), (20, 10, 0), (0, -15, 0), (10, 5, 12)]:
        rotation = _rot("z", roll) @ _rot("x", -pitch) @ _rot("y", -yaw)
        truth = head_angles_from_rotation(rotation)  # ground truth in the same convention
        pts, _ = cv2.projectPoints(_HEAD_MODEL, cv2.Rodrigues(rotation)[0], np.array([[0.0], [0.0], [1500.0]]), cam, np.zeros(4))
        pose = estimate_head_pose(pts.reshape(6, 2), w, h)
        assert pose is not None
        assert pose.yaw == pytest.approx(truth.yaw, abs=3.0)
        assert pose.pitch == pytest.approx(truth.pitch, abs=3.0)
        assert pose.roll == pytest.approx(truth.roll, abs=3.0)
        assert pose.yaw == pytest.approx(yaw, abs=5.0) and pose.pitch == pytest.approx(pitch, abs=5.0)


# ----------------------------------------------------------------------------- features
def test_features_are_translation_and_scale_invariant():
    lm = hand_landmarks((600, 300), (560, 340))
    lm[:, :2] += np.random.default_rng(0).normal(0, 0.05, (21, 2))
    base = landmarks_to_features(lm)
    assert base.shape == (FEATURE_DIM,) and base.dtype == np.float32
    shifted = lm.copy()
    shifted[:, :2] += (0.12, -0.05)
    assert landmarks_to_features(shifted) == pytest.approx(base, abs=1e-5)
    scaled = lm.copy()
    scaled[:, :2] = (scaled[:, :2] - lm[0, :2]) * 1.7 + lm[0, :2]
    assert landmarks_to_features(scaled) == pytest.approx(base, abs=1e-4)


def test_left_hand_is_mirrored_to_look_like_a_right_hand():
    lm = hand_landmarks((600, 300), (560, 340))
    right = normalize_hand(lm, "Right")
    left_view = lm.copy()
    left_view[:, 0] = 1.0 - left_view[:, 0]  # a real left hand is the mirror image
    assert normalize_hand(left_view, "Left") == pytest.approx(right, abs=1e-5)
    assert not np.allclose(normalize_hand(left_view, "Left", mirror_left=False), right)


def test_degenerate_hand_gives_zeros_not_nan():
    assert not np.any(normalize_hand(np.full((21, 3), 0.5)))

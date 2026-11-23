"""Optional plugin: classify hand poses with your own trained ``.onnx`` model.

Install the runtime with ``pip install "open-ai-hci[ai]"``. Without it the core
framework is unaffected: this module imports fine, and the classifier simply
disables itself (with a clear log message) unless ``strict=True``.

Model contract
--------------
* Input: float32 ``(1, 63)``: the output of
  :func:`open_ai_hci.features.landmarks_to_features` (the same features
  ``tools/record_gestures.py`` records).
* Output: float32 ``(1, n_classes)`` probabilities *or* raw logits (softmax is
  applied automatically when the row does not already sum to ~1). Models with several
  outputs (e.g. scikit-learn exports return ``[label, probabilities]``) are handled: the
  first floating-point output with more than one value is used, or name it with ``output_name``.
* Labels: pass ``labels=[...]``, or place ``<model>.labels.json`` (a JSON list)
  or ``<model>.labels.txt`` (one label per line) next to the model, or store a
  comma-separated ``labels`` entry in the ONNX ``metadata_props``.

Events: ``gesture`` (``gesture``, ``confidence``, ``previous``, ``handedness``) when a
new label becomes stable, ``gesture_<name>`` with the same payload, and
``gesture_end`` when it stops (hand lost, low confidence or ``background_label``).
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..._optional import MissingOptionalDependency, install_hint, require
from ...config import OnnxConfig
from ...features import FEATURE_DIM, landmarks_to_features
from ...types import FrameContext
from ..base import BaseProcessor

__all__ = ["ONNXGestureClassifier", "softmax"]

logger = logging.getLogger("open_ai_hci")


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    z = np.asarray(x, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _pick_scores(outputs: Sequence[Any]) -> np.ndarray:
    """Pick the probabilities/logits row out of an ONNX model's outputs.

    Many exporters return ``[predicted_label, probabilities]``. The scores are the first
    floating-point output with more than one value; if none qualifies, the first output is used.
    """
    arrays = [np.asarray(o) for o in outputs]
    for arr in arrays:
        if np.issubdtype(arr.dtype, np.floating) and arr.size > 1:
            return arr.astype(np.float64).reshape(-1)
    return arrays[0].astype(np.float64).reshape(-1)


class ONNXGestureClassifier(BaseProcessor):
    """Custom neural gesture classifier running on ONNX Runtime.

    Args:
        model_path: Path to the ``.onnx`` file.
        labels: Class names (index = model output index). See module docs for auto-discovery.
        min_confidence: A prediction below this is treated as "no gesture".
        stable_frames: A label must win this many consecutive processed frames to fire.
        background_label: Label (e.g. ``"none"``) that means "nothing recognised".
        mirror_left: Mirror left-hand landmarks so one model serves both hands.
        providers: ONNX Runtime execution providers (default: CPU).
        output_name: Name of the output tensor holding probabilities/logits. Default: auto-detect.
        strict: Raise on missing runtime / model instead of disabling the plugin.
        session: Inject a ready-made object with ``run(None, {name: array})``
            and ``get_inputs()`` (tests, custom runtimes). Skips loading ``model_path``.
    """

    name = "onnx_classifier"

    def __init__(
        self,
        model_path: str | Path | None = None,
        labels: Sequence[str] | None = None,
        *,
        min_confidence: float = 0.6,
        stable_frames: int = 3,
        background_label: str | None = "none",
        mirror_left: bool = True,
        providers: Sequence[str] | None = None,
        output_name: str | None = None,
        strict: bool = False,
        session: Any = None,
    ) -> None:
        if model_path is None and session is None:
            raise ValueError("Provide model_path (or session=)")
        if stable_frames < 1:
            raise ValueError("stable_frames must be >= 1")
        self.model_path = Path(model_path) if model_path is not None else None
        self.labels: list[str] | None = list(labels) if labels is not None else None
        self.min_confidence = min_confidence
        self.stable_frames = stable_frames
        self.background_label = background_label
        self.mirror_left = mirror_left
        self.providers = list(providers) if providers else ["CPUExecutionProvider"]
        self.output_name = output_name
        self.strict = strict
        self._session: Any = session
        self._input_name: str | None = None
        self.enabled = True
        # debounce state
        self._current: str | None = None
        self._candidate: str | None = None
        self._candidate_count = 0

    @classmethod
    def from_config(cls, cfg: OnnxConfig, **kwargs: Any) -> ONNXGestureClassifier:
        """Build from :class:`~open_ai_hci.config.OnnxConfig`; the instance then follows ``engine.config.onnx`` live."""
        clf = cls(
            cfg.model_path,
            min_confidence=cfg.min_confidence,
            stable_frames=cfg.stable_frames,
            background_label=cfg.background_label or None,
            **kwargs,
        )
        clf._follow_config = True
        return clf

    _follow_config = False

    def reconfigure(self) -> None:
        """Re-read confidence / debounce settings from ``engine.config.onnx`` (only for :meth:`from_config` instances)."""
        if not self._follow_config or self.engine is None:
            return
        cfg = self.engine.config.onnx
        self.min_confidence = cfg.min_confidence
        self.stable_frames = max(1, cfg.stable_frames)
        self.background_label = cfg.background_label or None

    # ------------------------------------------------------------------ lifecycle
    def setup(self) -> None:
        try:
            if self._session is None:
                ort = require("onnxruntime", "ONNXGestureClassifier")
                assert self.model_path is not None
                if not self.model_path.is_file():
                    raise FileNotFoundError(f"ONNX gesture model not found: {self.model_path}")
                self._session = ort.InferenceSession(str(self.model_path), providers=self.providers)
                if self.labels is None:
                    self.labels = self._discover_labels()
            self._input_name = self._session.get_inputs()[0].name
        except (MissingOptionalDependency, FileNotFoundError) as exc:
            if self.strict:
                raise
            self.enabled = False
            logger.warning("ONNXGestureClassifier disabled: %s", exc)
            if isinstance(exc, MissingOptionalDependency):
                logger.warning("To enable it: %s", install_hint("onnxruntime"))

    def _discover_labels(self) -> list[str] | None:
        assert self.model_path is not None and self._session is not None
        json_path = self.model_path.with_suffix(".labels.json")
        txt_path = self.model_path.with_suffix(".labels.txt")
        if json_path.is_file():
            return [str(x) for x in json.loads(json_path.read_text(encoding="utf-8"))]
        if txt_path.is_file():
            return [ln.strip() for ln in txt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        try:
            meta = self._session.get_modelmeta().custom_metadata_map
            if "labels" in meta:
                return [x.strip() for x in meta["labels"].split(",") if x.strip()]
        except Exception:  # pragma: no cover - runtime specific
            pass
        return None

    # --------------------------------------------------------------------- frames
    def predict(self, landmarks: np.ndarray, handedness: str | None = None) -> tuple[str, float]:
        """Classify one hand. Returns ``(label, confidence)``; usable without an engine."""
        if self._session is None or self._input_name is None:
            self.setup()
        if not self.enabled:
            raise RuntimeError("ONNXGestureClassifier is disabled (see log for the reason)")
        feats = landmarks_to_features(landmarks, handedness, self.mirror_left).reshape(1, FEATURE_DIM)
        wanted = [self.output_name] if self.output_name else None
        out = _pick_scores(self._session.run(wanted, {self._input_name: feats}))
        probs = out if abs(float(out.sum()) - 1.0) < 1e-3 and out.min() >= 0 else softmax(out)
        idx = int(np.argmax(probs))
        label = self.labels[idx] if self.labels and idx < len(self.labels) else str(idx)
        return label, float(probs[idx])

    def process(self, ctx: FrameContext) -> None:
        if not self.enabled:
            return
        hand = ctx.hand
        if hand is None:
            self._update(None, 0.0, ctx, None)
            return
        label, conf = self.predict(hand.landmarks, hand.handedness)
        if conf < self.min_confidence or label == self.background_label:
            self._update(None, conf, ctx, hand.handedness)
        else:
            self._update(label, conf, ctx, hand.handedness)

    def _update(self, label: str | None, conf: float, ctx: FrameContext, handedness: str | None) -> None:
        if label == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate, self._candidate_count = label, 1

        if self._candidate_count >= self.stable_frames and label != self._current:
            previous, self._current = self._current, label
            if label is None:
                self.emit("gesture_end", gesture=previous)
            else:
                payload = dict(gesture=label, confidence=conf, previous=previous, handedness=handedness)
                self.emit("gesture", **payload)
                self.emit(f"gesture_{label}", **payload)
        ctx.gesture = self._current
        ctx.gesture_confidence = conf if self._current is not None else 0.0

    def close(self) -> None:
        self._session = None

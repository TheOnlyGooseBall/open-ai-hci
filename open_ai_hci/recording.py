"""Record labelled hand landmarks and load them back for training.

Used by ``open-ai-hci record`` (``open_ai_hci/record.py``); importable on its own::

    from open_ai_hci.recording import GestureRecorder, load_dataset

    with GestureRecorder("data/gestures.csv") as rec:
        rec.add("fist", hand_landmarks, handedness="Right")

    X, y, hands = load_dataset("data/gestures.csv")   # X: (n, 63) features

Formats are chosen by file extension: ``.csv`` (default), ``.jsonl`` (one JSON object per
line) or ``.json`` (a single array).

Since v0.2.0 the recorder stores the **normalised, scale-invariant** features by default: the
21 landmarks relative to the wrist, divided by the hand size (wrist to middle-finger knuckle),
with left hands mirrored (:func:`open_ai_hci.features.normalize_hand`). That is exactly what
the ONNX classifier computes at inference time, so a model trained on the CSV sees identical
inputs, and the file can be shared without leaking where in the frame (or how big) the hand was.
Pass ``normalize=False`` (CLI: ``--raw``) to keep the raw MediaPipe coordinates instead.
"""
from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .features import landmarks_to_features, normalize_hand
from .types import HandData

__all__ = ["COLUMNS", "FEATURE_COLUMNS", "GestureRecorder", "load_dataset"]

#: Raw-landmark CSV header (``--raw``).
COLUMNS: list[str] = ["label", "handedness", "timestamp"] + [f"{axis}{i}" for i in range(21) for axis in "xyz"]
#: Normalised-feature CSV header (the default): ``nx0, ny0, nz0, ... nz20``.
FEATURE_COLUMNS: list[str] = ["label", "handedness", "timestamp"] + [f"n{axis}{i}" for i in range(21) for axis in "xyz"]


class GestureRecorder:
    """Append-only writer for labelled hand samples.

    Args:
        path: Output file (``.csv``, ``.jsonl`` or ``.json``). Existing files are appended to.
        fmt: Force ``"csv"``, ``"jsonl"`` or ``"json"`` instead of inferring from the extension.
        normalize: Store scale-invariant, wrist-relative features (default) instead of raw landmarks.
        mirror_left: With ``normalize``, mirror left hands so one model serves both hands.
    """

    def __init__(self, path: str | Path, fmt: str | None = None, *, normalize: bool = True, mirror_left: bool = True) -> None:
        self.normalize, self.mirror_left = normalize, mirror_left
        self.columns = FEATURE_COLUMNS if normalize else COLUMNS
        self.path = Path(path)
        self.fmt = (fmt or {".jsonl": "jsonl", ".json": "json"}.get(self.path.suffix.lower(), "csv")).lower()
        if self.fmt not in ("csv", "jsonl", "json"):
            raise ValueError("fmt must be csv, jsonl or json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.counts: Counter[str] = Counter()
        self._fh: Any = None
        self._writer: Any = None
        self._json_rows: list[dict[str, Any]] = []
        self._open()

    # ----------------------------------------------------------------------- open
    def _open(self) -> None:
        exists = self.path.is_file() and self.path.stat().st_size > 0
        if self.fmt == "csv":
            if exists:
                with self.path.open(newline="", encoding="utf-8") as fh:
                    header = next(csv.reader(fh), [])
                if header != self.columns:
                    kind = "raw landmarks" if header == COLUMNS else "normalised features" if header == FEATURE_COLUMNS else "an unknown layout"
                    raise ValueError(
                        f"{self.path} has an incompatible header ({kind}); refusing to append. "
                        f"Use a new file, or {'drop --raw' if header == FEATURE_COLUMNS else 'pass --raw'} to match it."
                    )
            self._fh = self.path.open("a", newline="", encoding="utf-8")
            self._writer = csv.writer(self._fh)
            if not exists:
                self._writer.writerow(self.columns)
        elif self.fmt == "jsonl":
            self._fh = self.path.open("a", encoding="utf-8")
        else:  # json: keep rows in memory, write the whole array on close
            if exists:
                self._json_rows = list(json.loads(self.path.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------------ add
    def add(
        self,
        label: str,
        landmarks: np.ndarray | HandData,
        handedness: str | None = None,
        timestamp: float | None = None,
    ) -> bool:
        """Record one sample. ``landmarks`` is a ``(21, 3)`` array or a :class:`HandData`.

        Returns ``False`` (and records nothing) for a degenerate hand whose size is zero.
        """
        if isinstance(landmarks, HandData):
            handedness = handedness or landmarks.handedness
            landmarks = landmarks.landmarks
        lm = np.asarray(landmarks, dtype=np.float64).reshape(21, 3)
        ts = time.time() if timestamp is None else timestamp
        hand = handedness or "Unknown"
        if self.normalize:
            lm = normalize_hand(lm, hand, self.mirror_left).astype(np.float64)
            if not lm.any():
                return False
        if self.fmt == "csv":
            self._writer.writerow([label, hand, f"{ts:.3f}"] + [f"{v:.6f}" for v in lm.reshape(-1)])
        else:
            row = {"label": label, "handedness": hand, "timestamp": round(ts, 3), "landmarks": np.round(lm, 6).tolist()}
            if self.normalize:
                row["normalized"] = True
            if self.fmt == "jsonl":
                self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")
            else:
                self._json_rows.append(row)
        self.counts[label] += 1
        return True

    def __len__(self) -> int:
        return sum(self.counts.values())

    # ---------------------------------------------------------------------- close
    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        if self.fmt == "json":
            self.path.write_text(json.dumps(self._json_rows, indent=None, separators=(",", ":")), encoding="utf-8")
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> GestureRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def load_dataset(
    path: str | Path, *, normalize: bool = True, mirror_left: bool = True
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load a recording.

    Files written with the default (normalised) recorder are returned as they are: their rows are
    already the classifier features. Raw files (``--raw`` / ``normalize=False``) are normalised here
    when ``normalize=True`` (default), or returned as flattened raw landmarks otherwise.

    Returns:
        ``(X, labels, handedness)`` where ``X`` has shape ``(n, 63)`` (float32).
    """
    p = Path(path)
    suffix = p.suffix.lower()
    rows: list[tuple[str, str, np.ndarray, bool]] = []  # label, hand, values, already normalised
    if suffix == ".json":
        for r in json.loads(p.read_text(encoding="utf-8")):
            rows.append((r["label"], r["handedness"], np.asarray(r["landmarks"], dtype=np.float64), bool(r.get("normalized"))))
    elif suffix == ".jsonl":
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append((r["label"], r["handedness"], np.asarray(r["landmarks"], dtype=np.float64), bool(r.get("normalized"))))
    else:
        with p.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            is_features = bool(reader.fieldnames) and "nx0" in reader.fieldnames
            cols = FEATURE_COLUMNS[3:] if is_features else COLUMNS[3:]
            for r in reader:
                lm = np.array([float(r[c]) for c in cols], dtype=np.float64)
                rows.append((r["label"], r["handedness"], lm, is_features))
    if not rows:
        return np.zeros((0, 63), dtype=np.float32), [], []
    labels = [r[0] for r in rows]
    hands = [r[1] for r in rows]
    if normalize:
        x = np.stack([r[2].reshape(-1) if r[3] else landmarks_to_features(r[2], r[1], mirror_left) for r in rows])
    else:
        x = np.stack([r[2].reshape(-1) for r in rows])
    return x.astype(np.float32), labels, hands

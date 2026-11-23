"""Train a custom gesture classifier from a recording and export it to ONNX.

::

    open-ai-hci record --label right_click --samples 150
    open-ai-hci record --label none --samples 300           # relaxed poses: what NOT to fire on
    open-ai-hci train --data data/gestures.csv --out models/gestures.onnx

or from Python::

    from open_ai_hci.training import train_model
    result = train_model("data/gestures.csv", "models/gestures.onnx")
    print(result.summary())

The model is a small multi-layer perceptron (``--model mlp``; ``logreg`` and ``forest`` are also
available) trained on the 63 scale-invariant landmark features of :mod:`open_ai_hci.features`. The export
is a plain ONNX graph with probabilities as output, plus ``<model>.labels.json``. Both are picked up by
:class:`~open_ai_hci.processors.ai.ONNXGestureClassifier`. A single-sample inference takes well under a
millisecond with ONNX Runtime, so the 60+ FPS tracking loop is never slowed down; :func:`train_model`
measures this and reports it.

Needs ``pip install "open-ai-hci[train]"`` (scikit-learn + skl2onnx) and ``[ai]`` (onnxruntime).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._optional import MissingOptionalDependency, require
from .features import FEATURE_DIM
from .recording import load_dataset

__all__ = ["TrainResult", "train_model", "benchmark_onnx", "build_parser", "main", "MODEL_KINDS"]

MODEL_KINDS = ("mlp", "logreg", "forest")
TARGET_FPS = 60.0


@dataclass
class TrainResult:
    """What :func:`train_model` produced."""

    model_path: Path
    labels_path: Path
    labels: list[str]
    class_counts: dict[str, int]
    accuracy: float  # on the held-out split (or on the training data if there was too little data to split)
    evaluated_on: str  # "held-out" | "training data"
    per_class_recall: dict[str, float] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)
    inference_ms: float | None = None  # mean single-sample ONNX Runtime latency
    warnings: list[str] = field(default_factory=list)

    @property
    def inference_fps(self) -> float | None:
        return 1000.0 / self.inference_ms if self.inference_ms else None

    def summary(self) -> str:
        lines = [f"Trained on {sum(self.class_counts.values())} samples: " + ", ".join(f"{k}={v}" for k, v in self.class_counts.items())]
        lines.append(f"Accuracy ({self.evaluated_on}): {self.accuracy:.1%}")
        if self.per_class_recall:
            lines.append("Recall per class: " + ", ".join(f"{k}={v:.0%}" for k, v in self.per_class_recall.items()))
        if self.inference_ms is not None:
            fps = self.inference_fps or 0.0
            verdict = f"OK for {TARGET_FPS:.0f}+ FPS" if fps >= TARGET_FPS else f"BELOW {TARGET_FPS:.0f} FPS"
            lines.append(f"ONNX Runtime inference: {self.inference_ms:.3f} ms/sample (~{fps:,.0f} per second): {verdict}")
        lines += [f"Warning: {w}" for w in self.warnings]
        lines += [f"Model:  {self.model_path}", f"Labels: {self.labels_path}"]
        return "\n".join(lines)


def _make_model(kind: str, seed: int) -> Any:
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=seed)
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=2000, random_state=seed)
    if kind == "forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=100, max_depth=12, random_state=seed)
    raise ValueError(f"Unknown model kind {kind!r}; choose one of {', '.join(MODEL_KINDS)}")


def benchmark_onnx(model_path: str | Path, runs: int = 300) -> float:
    """Mean milliseconds for one single-sample inference through ONNX Runtime (after a warm-up)."""
    ort = require("onnxruntime", "Benchmarking the exported model")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    x = np.zeros((1, FEATURE_DIM), dtype=np.float32)
    for _ in range(10):
        session.run(None, {name: x})
    t0 = time.perf_counter()
    for _ in range(runs):
        session.run(None, {name: x})
    return (time.perf_counter() - t0) * 1000.0 / runs


def train_model(
    data: str | Path,
    out: str | Path = "models/gestures.onnx",
    *,
    model: str = "mlp",
    test_size: float = 0.2,
    seed: int = 0,
    min_samples_per_class: int = 10,
    benchmark: bool = True,
) -> TrainResult:
    """Train on a recording and write ``out`` (``.onnx``) plus ``out`` with the suffix ``.labels.json``.

    Args:
        data: File written by ``open-ai-hci record`` (``.csv`` / ``.jsonl`` / ``.json``).
        out: Output ``.onnx`` path (folders are created).
        model: ``"mlp"`` (default), ``"logreg"`` or ``"forest"``.
        test_size: Fraction held out for the accuracy estimate (``0`` trains on everything).
        seed: Random seed for the split and the model.
        min_samples_per_class: Refuse to train when a class has fewer samples than this.
        benchmark: Measure ONNX Runtime latency after exporting (needs ``onnxruntime``).

    Raises:
        MissingOptionalDependency: scikit-learn / skl2onnx are not installed (the message has the pip command).
        ValueError: unusable data (fewer than two classes, too few samples, unknown model).
    """
    require("sklearn", "Training a gesture model")
    skl2onnx = require("skl2onnx", "Exporting a gesture model to ONNX")
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split

    if model not in MODEL_KINDS:
        raise ValueError(f"Unknown model kind {model!r}; choose one of {', '.join(MODEL_KINDS)}")
    data_path = Path(data)
    if not data_path.is_file():
        raise ValueError(f"Training data not found: {data_path}. Record some first: open-ai-hci record --label NAME")
    X, y_text, _ = load_dataset(data_path)
    counts = Counter(y_text)
    labels = sorted(counts)
    if len(labels) < 2:
        raise ValueError(
            f"Need at least two classes to train, found {len(labels)} ({', '.join(labels) or 'none'}). "
            "Record a second label, e.g. `open-ai-hci record --label none` with relaxed hand poses."
        )
    few = {k: v for k, v in counts.items() if v < min_samples_per_class}
    if few:
        raise ValueError(f"Too few samples for {few} (need at least {min_samples_per_class} per class). Record more.")
    warnings: list[str] = []
    if "none" not in counts:
        warnings.append("no 'none' class: the model will fire on every hand pose. Record relaxed poses with --label none.")
    if max(counts.values()) > 5 * min(counts.values()):
        warnings.append(f"imbalanced classes {dict(counts)}; record similar amounts per label for best results.")

    index = {name: i for i, name in enumerate(labels)}
    y = np.array([index[v] for v in y_text])
    evaluated_on = "held-out"
    can_split = test_size > 0 and min(counts.values()) * test_size >= 1.0 and len(y) * test_size >= len(labels)
    if can_split:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    else:
        X_tr, X_te, y_tr, y_te = X, X, y, y
        evaluated_on = "training data"
        if test_size > 0:
            warnings.append("too little data for a held-out split; the accuracy is measured on the training data.")

    clf = _make_model(model, seed).fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    accuracy = float((pred == y_te).mean())
    cm = confusion_matrix(y_te, pred, labels=list(range(len(labels))))
    recall = {labels[i]: float(cm[i, i] / cm[i].sum()) for i in range(len(labels)) if cm[i].sum() > 0}
    if accuracy < 0.9:
        warnings.append(f"accuracy {accuracy:.0%} is low: record more varied samples (angles, distances, both hands).")

    onnx_model = skl2onnx.to_onnx(clf, X[:1].astype(np.float32), options={id(clf): {"zipmap": False}}, target_opset=15)
    out_path = Path(out)
    if out_path.suffix.lower() != ".onnx":
        out_path = out_path.with_suffix(".onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(onnx_model.SerializeToString())
    labels_path = out_path.with_suffix(".labels.json")  # where ONNXGestureClassifier looks for it
    labels_path.write_text(json.dumps(labels), encoding="utf-8")

    ms: float | None = None
    if benchmark:
        try:
            ms = benchmark_onnx(out_path)
        except MissingOptionalDependency:
            warnings.append("onnxruntime is not installed, so the inference speed was not measured (pip install 'open-ai-hci[ai]').")
        if ms is not None and 1000.0 / ms < TARGET_FPS:
            warnings.append(f"inference is slower than {TARGET_FPS:.0f} FPS; try --model logreg.")
    return TrainResult(
        out_path, labels_path, labels, dict(counts), accuracy, evaluated_on, recall, cm.tolist(), ms, warnings
    )


def build_parser(prog: str = "open-ai-hci train") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/gestures.csv", help="recording from `open-ai-hci record` (default data/gestures.csv)")
    p.add_argument("--out", default="models/gestures.onnx", help="output .onnx path (default models/gestures.onnx)")
    p.add_argument("--model", choices=MODEL_KINDS, default="mlp", help="classifier type (default mlp)")
    p.add_argument("--test-size", type=float, default=0.2, help="held-out fraction for the accuracy estimate (0 = train on all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-samples", type=int, default=10, help="minimum samples per class")
    p.add_argument("--no-benchmark", action="store_true", help="skip the ONNX Runtime speed test")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = train_model(
            args.data, args.out, model=args.model, test_size=args.test_size, seed=args.seed,
            min_samples_per_class=args.min_samples, benchmark=not args.no_benchmark,
        )
    except MissingOptionalDependency as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(result.summary())
    print(f"\nUse it:  ONNXGestureClassifier('{result.model_path}')   or the GUI's Custom ONNX AI Model tab.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

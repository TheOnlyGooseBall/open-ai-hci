"""``open-ai-hci train``: record -> train -> ONNX -> classifier, plus error handling and the CLI."""
import json

import numpy as np
import pytest

from open_ai_hci._optional import MissingOptionalDependency, is_available
from open_ai_hci.processors.ai import ONNXGestureClassifier
from open_ai_hci.recording import GestureRecorder
from open_ai_hci.training import MODEL_KINDS, benchmark_onnx, main, train_model

needs_ml = pytest.mark.skipif(
    not (is_available("sklearn") and is_available("skl2onnx") and is_available("onnxruntime")),
    reason="scikit-learn / skl2onnx / onnxruntime not installed",
)


def make_pose(kind, hand, rng):
    lm = np.zeros((21, 3))
    lm[0] = (0.5, 0.9, 0)
    reach = {"open": 0.35, "fist": 0.12, "peace": 0.25}[kind]
    for f, mcp in enumerate((5, 9, 13, 17)):
        r = reach if kind != "peace" or f < 2 else 0.1  # peace: only index + middle extended
        for j in range(4):
            lm[mcp + j] = (0.4 + f * 0.07, 0.65 - r * (j + 1) / 4, 0)
    lm += rng.normal(0, 0.004, lm.shape)
    if hand == "Left":
        lm[:, 0] = 1.0 - lm[:, 0]
    return lm


def record(path, kinds=("open", "fist"), n=60, seed=1, **kw):
    rng = np.random.default_rng(seed)
    with GestureRecorder(path, **kw) as rec:
        for i in range(n):
            hand = "Left" if i % 2 else "Right"
            for k in kinds:
                rec.add(k, make_pose(k, hand, rng), hand)
    return rng


@needs_ml
@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_train_exports_a_model_the_classifier_can_use_for_both_hands(tmp_path, kind):
    data = tmp_path / "gestures.csv"
    rng = record(data, ("open", "fist", "peace"))
    result = train_model(data, tmp_path / "models" / "gestures.onnx", model=kind, seed=0)
    assert result.model_path.is_file() and result.labels_path.name == "gestures.labels.json"
    assert json.loads(result.labels_path.read_text()) == ["fist", "open", "peace"] == result.labels
    assert result.class_counts == {"open": 60, "fist": 60, "peace": 60} and result.evaluated_on == "held-out"
    assert result.accuracy > 0.95 and all(v > 0.9 for v in result.per_class_recall.values())
    assert np.array(result.confusion).shape == (3, 3)
    clf = ONNXGestureClassifier(result.model_path)
    clf.setup()
    assert clf.enabled and clf.labels == result.labels
    for k in ("open", "fist", "peace"):
        for hand in ("Right", "Left"):
            label, conf = clf.predict(make_pose(k, hand, rng), hand)
            assert label == k and conf > 0.6, (kind, k, hand, label, conf)


@needs_ml
def test_inference_is_fast_enough_for_a_60_fps_loop(tmp_path):
    data = tmp_path / "g.csv"
    record(data)
    result = train_model(data, tmp_path / "m.onnx")
    assert result.inference_ms is not None and result.inference_fps >= 60  # the tracking loop budget is ~16 ms
    assert result.inference_ms < 5.0  # in practice far below 1 ms; generous for slow CI machines
    assert "OK for 60+ FPS" in result.summary()
    assert benchmark_onnx(result.model_path, runs=20) > 0


@needs_ml
def test_training_works_on_raw_recordings_and_other_formats(tmp_path):
    for name, kw in (("raw.csv", {"normalize": False}), ("n.jsonl", {}), ("n.json", {})):
        record(tmp_path / name, **kw)
        result = train_model(tmp_path / name, tmp_path / f"{name}.onnx", benchmark=False)
        assert result.accuracy > 0.95 and result.inference_ms is None


@needs_ml
def test_output_path_gets_the_onnx_suffix_and_small_datasets_skip_the_split(tmp_path):
    data = tmp_path / "tiny.csv"
    record(data, n=4)  # 4 samples per class: too few for a stratified held-out split at 20 %
    result = train_model(data, tmp_path / "out" / "mymodel", min_samples_per_class=3, benchmark=False)
    assert result.model_path.name == "mymodel.onnx" and result.evaluated_on == "training data"
    assert any("held-out" in w for w in result.warnings)


@needs_ml
def test_warnings_for_missing_none_class_and_imbalance(tmp_path):
    data = tmp_path / "g.csv"
    rng = np.random.default_rng(2)
    with GestureRecorder(data) as rec:
        for _ in range(120):
            rec.add("open", make_pose("open", "Right", rng), "Right")
        for _ in range(15):
            rec.add("fist", make_pose("fist", "Right", rng), "Right")
    result = train_model(data, tmp_path / "m.onnx", benchmark=False)
    text = " ".join(result.warnings)
    assert "'none' class" in text and "imbalanced" in text


@needs_ml
def test_a_none_class_silences_the_warning(tmp_path):
    data = tmp_path / "g.csv"
    record(data, ("open",), n=30)
    rng = np.random.default_rng(3)
    with GestureRecorder(data) as rec:
        for _ in range(30):
            rec.add("none", make_pose("fist", "Right", rng), "Right")
    result = train_model(data, tmp_path / "m.onnx", benchmark=False)
    assert not any("'none' class" in w for w in result.warnings)


@needs_ml
def test_unusable_data_gives_actionable_errors(tmp_path):
    with pytest.raises(ValueError, match="not found.*open-ai-hci record"):
        train_model(tmp_path / "missing.csv", tmp_path / "m.onnx")
    one = tmp_path / "one.csv"
    record(one, ("open",), n=20)
    with pytest.raises(ValueError, match="at least two classes"):
        train_model(one, tmp_path / "m.onnx")
    few = tmp_path / "few.csv"
    record(few, n=4)
    with pytest.raises(ValueError, match="Too few samples"):
        train_model(few, tmp_path / "m.onnx")
    with pytest.raises(ValueError, match="Unknown model"):
        train_model(few, tmp_path / "m.onnx", model="transformer")


def test_missing_scikit_learn_gives_the_install_hint(tmp_path, monkeypatch):
    import open_ai_hci.training as mod

    def fake_require(module, feature):
        raise MissingOptionalDependency(f"{feature} requires '{module}'. Install it with:  pip install \"open-ai-hci[train]\"")

    monkeypatch.setattr(mod, "require", fake_require)
    with pytest.raises(MissingOptionalDependency, match=r"open-ai-hci\[train\]"):
        train_model(tmp_path / "x.csv", tmp_path / "m.onnx")
    assert main(["--data", str(tmp_path / "x.csv")]) == 1


@needs_ml
def test_train_cli_prints_a_report_and_returns_error_codes(tmp_path, capsys):
    data = tmp_path / "g.csv"
    record(data)
    out = tmp_path / "models" / "g.onnx"
    assert main(["--data", str(data), "--out", str(out), "--model", "logreg", "--test-size", "0.25"]) == 0
    text = capsys.readouterr().out
    assert "Accuracy (held-out)" in text and "ONNX Runtime inference" in text and "ONNXGestureClassifier" in text and out.is_file()
    assert main(["--data", str(tmp_path / "nope.csv")]) == 2
    assert "error:" in capsys.readouterr().err


@needs_ml
def test_open_ai_hci_train_subcommand_dispatches(tmp_path, capsys):
    from open_ai_hci.__main__ import main as cli

    data = tmp_path / "g.csv"
    record(data)
    assert cli(["train", "--data", str(data), "--out", str(tmp_path / "x.onnx"), "--no-benchmark"]) == 0
    assert (tmp_path / "x.onnx").is_file() and (tmp_path / "x.labels.json").is_file()
    capsys.readouterr()

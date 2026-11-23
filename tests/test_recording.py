import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import hand_landmarks, make_hand

from open_ai_hci.features import landmarks_to_features
from open_ai_hci.recording import COLUMNS, FEATURE_COLUMNS, GestureRecorder, load_dataset

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("name", ["g.csv", "g.jsonl", "g.json"])
def test_round_trip_in_every_format(tmp_path, name):
    path = tmp_path / name
    lm_a, lm_b = hand_landmarks((600, 300), (560, 340)), hand_landmarks((700, 200), (690, 260))
    with GestureRecorder(path) as rec:
        rec.add("fist", lm_a, "Right")
        rec.add("open", make_hand((700, 200), (690, 260), handedness="Left"))
        assert len(rec) == 2 and rec.counts == {"fist": 1, "open": 1}
    X, y, hands = load_dataset(path)
    assert y == ["fist", "open"] and hands == ["Right", "Left"]
    assert X.shape == (2, 63) and X.dtype == np.float32
    assert X[0] == pytest.approx(landmarks_to_features(lm_a, "Right"), abs=1e-5)
    assert X[1] == pytest.approx(landmarks_to_features(lm_b, "Left"), abs=1e-5)  # same features the classifier sees
    # the stored rows ARE the features: asking for "raw" cannot undo the normalisation
    assert load_dataset(path, normalize=False)[0] == pytest.approx(X, abs=1e-6)


@pytest.mark.parametrize("name", ["g.csv", "g.jsonl", "g.json"])
def test_raw_mode_keeps_raw_landmarks_and_normalises_on_load(tmp_path, name):
    path = tmp_path / name
    lm = hand_landmarks((600, 300), (560, 340))
    with GestureRecorder(path, normalize=False) as rec:
        rec.add("fist", lm, "Right")
    assert load_dataset(path, normalize=False)[0][0] == pytest.approx(lm.reshape(-1), abs=1e-5)
    assert load_dataset(path)[0][0] == pytest.approx(landmarks_to_features(lm, "Right"), abs=1e-5)


def test_normalised_rows_are_scale_and_translation_invariant_and_mirror_left_hands(tmp_path):
    base = hand_landmarks((600, 300), (560, 340))
    moved_and_scaled = (base - base[0]) * 0.5 + np.array([0.2, 0.3, 0.0])  # smaller hand somewhere else in the frame
    with GestureRecorder(tmp_path / "g.csv") as rec:
        rec.add("a", base, "Right")
        rec.add("a", moved_and_scaled, "Right")
        rec.add("a", np.column_stack([1.0 - base[:, 0], base[:, 1], base[:, 2]]), "Left")  # the mirror image, labelled Left
    X, _, _ = load_dataset(tmp_path / "g.csv")
    assert X[1] == pytest.approx(X[0], abs=1e-5)
    assert X[2] == pytest.approx(X[0], abs=1e-5)
    assert X[0].reshape(21, 3)[0] == pytest.approx([0, 0, 0]) and np.linalg.norm(X[0].reshape(21, 3)[9, :2]) == pytest.approx(1.0, abs=1e-5)


def test_degenerate_hands_are_not_recorded(tmp_path):
    with GestureRecorder(tmp_path / "g.csv") as rec:
        assert rec.add("a", np.zeros((21, 3)), "Right") is False
        assert rec.add("a", hand_landmarks((600, 300), (560, 340)), "Right") is True
        assert len(rec) == 1


def test_mixing_raw_and_normalised_files_is_refused_with_a_hint(tmp_path):
    path = tmp_path / "g.csv"
    with GestureRecorder(path, normalize=False) as rec:
        rec.add("a", hand_landmarks((1, 1), (2, 2)))
    with pytest.raises(ValueError, match="raw landmarks.*--raw|--raw.*raw landmarks"):
        GestureRecorder(path)
    other = tmp_path / "n.csv"
    GestureRecorder(other).close()
    with pytest.raises(ValueError, match="drop --raw"):
        GestureRecorder(other, normalize=False)


@pytest.mark.parametrize("name", ["g.csv", "g.jsonl", "g.json"])
def test_appends_to_existing_files(tmp_path, name):
    path = tmp_path / name
    for label in ("a", "b"):
        with GestureRecorder(path) as rec:
            rec.add(label, hand_landmarks((1, 1), (2, 2)))
    assert load_dataset(path)[1] == ["a", "b"]


def test_csv_layout_and_header_guard(tmp_path):
    path = tmp_path / "g.csv"
    with GestureRecorder(path) as rec:
        rec.add("x", hand_landmarks((10, 10), (20, 20)))
    lines = path.read_text().splitlines()
    assert lines[0].split(",") == FEATURE_COLUMNS and len(lines) == 2 and len(FEATURE_COLUMNS) == 66
    assert lines[0].startswith("label,handedness,timestamp,nx0,ny0,nz0,nx1") and COLUMNS[3] == "x0"
    path.write_text("wrong,header\n1,2\n")
    with pytest.raises(ValueError, match="incompatible"):
        GestureRecorder(path)


def test_invalid_format_and_empty_dataset(tmp_path):
    with pytest.raises(ValueError):
        GestureRecorder(tmp_path / "x.csv", fmt="xml")
    GestureRecorder(tmp_path / "empty.csv").close()
    X, y, h = load_dataset(tmp_path / "empty.csv")
    assert X.shape == (0, 63) and y == [] and h == []


def test_jsonl_is_one_object_per_line(tmp_path):
    path = tmp_path / "g.jsonl"
    with GestureRecorder(path) as rec:
        rec.add("x", hand_landmarks((10, 10), (20, 20)))
        rec.add("y", hand_landmarks((10, 10), (20, 20)))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["label"] for r in rows] == ["x", "y"] and len(rows[0]["landmarks"]) == 21


def test_cli_records_from_a_video_without_a_window(tmp_path, monkeypatch):
    """End-to-end: tools/record_gestures.py --auto on a synthetic video, with a fake hand detector."""
    import cv2

    video = tmp_path / "clip.avi"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
    for _ in range(12):
        vw.write(np.zeros((240, 320, 3), np.uint8))
    vw.release()

    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import record_gestures
    finally:
        sys.path.pop(0)
    from open_ai_hci import record as record_module

    assert record_gestures.main is record_module.main  # the tools/ script is a thin wrapper

    from conftest import FakeHandBackend

    import open_ai_hci.sensors.hand_sensor as hs

    monkeypatch.setattr(hs, "create_hand_backend", lambda cfg, mp_cfg: FakeHandBackend([((100, 100), (120, 130))] * 50))
    out = tmp_path / "out.csv"
    code = record_gestures.main(["--label", "pinch", "--source", str(video), "--auto", "--samples", "5", "--out", str(out)])
    assert code == 0
    X, y, _ = load_dataset(out)
    assert y == ["pinch"] * 5 and X.shape == (5, 63)


# ------------------------------------------------------------------------ open-ai-hci record
@pytest.fixture
def clip(tmp_path):
    import cv2

    video = tmp_path / "clip.avi"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
    for _ in range(80):
        vw.write(np.zeros((240, 320, 3), np.uint8))
    vw.release()
    return str(video)


@pytest.fixture
def fake_hand(monkeypatch):
    from conftest import FakeHandBackend

    import open_ai_hci.sensors.hand_sensor as hs

    monkeypatch.setattr(hs, "create_hand_backend", lambda cfg, mp_cfg: FakeHandBackend([((100, 100), (120, 130))] * 500))


def test_record_cli_stops_at_the_requested_sample_count_and_stores_normalised_rows(tmp_path, clip, fake_hand, capsys):
    from open_ai_hci.record import main

    out = tmp_path / "gestures.csv"
    assert main(["--label", "right_click", "--samples", "7", "--source", clip, "--auto", "--out", str(out)]) == 0
    assert out.read_text().splitlines()[0].startswith("label,handedness,timestamp,nx0")
    X, y, _ = load_dataset(out)
    assert y == ["right_click"] * 7 and X.shape == (7, 63)
    text = capsys.readouterr().out
    assert "Saved 7 new samples" in text and "open-ai-hci train" in text


def test_record_cli_auto_mode_with_several_labels_records_each_in_turn(tmp_path, clip, fake_hand):
    from open_ai_hci.record import main

    out = tmp_path / "g.csv"
    assert main(["--labels", "a,b", "--samples", "4", "--source", clip, "--auto", "--out", str(out)]) == 0
    assert load_dataset(out)[1] == ["a"] * 4 + ["b"] * 4


def test_record_cli_raw_flag_and_header_mismatch_error(tmp_path, clip, fake_hand, capsys):
    from open_ai_hci.record import main

    out = tmp_path / "raw.csv"
    assert main(["--label", "x", "--samples", "3", "--source", clip, "--auto", "--raw", "--out", str(out)]) == 0
    assert out.read_text().splitlines()[0].startswith("label,handedness,timestamp,x0,y0,z0")
    assert main(["--label", "x", "--samples", "3", "--source", clip, "--auto", "--out", str(out)]) == 2  # normalised vs raw file
    assert "incompatible header" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["--labels", "a,a"], ["--labels", ",".join("abcdefghij")], ["--label", "a", "--samples", "0"]])
def test_record_cli_rejects_bad_arguments(argv, tmp_path):
    from open_ai_hci.record import main

    assert main([*argv, "--out", str(tmp_path / "x.csv")]) == 2


def test_record_cli_reports_an_unavailable_camera_cleanly(tmp_path, fake_hand, capsys):
    from open_ai_hci.record import main

    assert main(["--label", "a", "--source", str(tmp_path / "missing.mp4"), "--auto", "--out", str(tmp_path / "x.csv")]) == 1
    err = capsys.readouterr().err
    assert "Could not open camera" in err and "doctor" in err


def test_record_cli_interactive_keys_countdown_and_auto_finish(tmp_path, clip, fake_hand, monkeypatch):
    """Window mode with scripted key presses: SPACE starts recording, it stops itself at --samples, then the run ends."""
    import cv2

    import open_ai_hci.record as rec_mod

    keys = iter([-1, ord(" ")] + [-1] * 200)
    shown = []
    monkeypatch.setattr(cv2, "imshow", lambda name, frame: shown.append(name))
    monkeypatch.setattr(cv2, "waitKey", lambda ms: next(keys))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    out = tmp_path / "i.csv"
    assert rec_mod.main(["--label", "peace", "--samples", "6", "--countdown", "0", "--source", clip, "--out", str(out)]) == 0
    assert load_dataset(out)[1] == ["peace"] * 6 and shown and shown[0] == rec_mod.WINDOW


def test_record_cli_quit_key_saves_what_was_recorded(tmp_path, clip, fake_hand, monkeypatch):
    import cv2

    import open_ai_hci.record as rec_mod

    keys = iter([ord(" "), -1, -1, -1, ord("q")])
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "waitKey", lambda ms: next(keys))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    out = tmp_path / "q.csv"
    assert rec_mod.main(["--labels", "a,b", "--samples", "50", "--countdown", "0", "--source", clip, "--out", str(out)]) == 0
    assert 1 <= len(load_dataset(out)[1]) < 50

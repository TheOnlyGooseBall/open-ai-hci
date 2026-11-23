"""MediaPipe backend selection, model download logic and the Tasks-API adapters.

The Tasks adapters are exercised against the *real installed* MediaPipe classes (so a wrong option
name fails here), with only the model-loading call faked, because the ``.task`` files are large
downloads that CI machines may not have.
"""
import types
import urllib.error

import numpy as np
import pytest

from open_ai_hci import FaceConfig, HandConfig, MediaPipeConfig, PoseConfig, _models
from open_ai_hci._models import MODEL_URLS, ModelUnavailableError, ensure_model
from open_ai_hci.sensors import _mediapipe as backend

mp = pytest.importorskip("mediapipe")
HAS_TASKS = pytest.importorskip("mediapipe.tasks.python.vision") is not None


# ------------------------------------------------------------------ backend selection
def test_auto_picks_the_api_that_exists():
    expected = "legacy" if hasattr(mp, "solutions") else "tasks"
    assert backend.resolve_backend(MediaPipeConfig()) == expected
    assert backend.resolve_backend(MediaPipeConfig(backend="tasks")) == "tasks"


def test_forcing_legacy_on_a_mediapipe_without_it_explains_the_fix(monkeypatch):
    monkeypatch.delattr(mp, "solutions", raising=False)
    with pytest.raises(RuntimeError, match="no longer ships the legacy"):
        backend.resolve_backend(MediaPipeConfig(backend="legacy"))
    assert backend.resolve_backend(MediaPipeConfig()) == "tasks"


# -------------------------------------------------------------------------- models
@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OPEN_AI_HCI_HOME", str(tmp_path / "home"))
    return tmp_path / "models"


def test_explicit_path_is_used_or_rejected(tmp_path):
    f = tmp_path / "my.task"
    f.write_bytes(b"x" * 10)
    assert ensure_model("hand", str(f)) == f
    with pytest.raises(ModelUnavailableError, match="not found"):
        ensure_model("hand", str(tmp_path / "nope.task"))
    with pytest.raises(ValueError):
        ensure_model("unknown_kind")


def test_cache_dir_resolution(monkeypatch, tmp_path):
    monkeypatch.setenv("OPEN_AI_HCI_HOME", str(tmp_path))
    assert _models.cache_dir() == tmp_path / "models"
    assert _models.cache_dir(str(tmp_path / "x")) == tmp_path / "x"


def test_cached_model_is_reused_without_network(model_dir, monkeypatch):
    model_dir.mkdir(parents=True)
    cached = model_dir / "hand_landmarker.task"
    cached.write_bytes(b"x" * 200_000)
    monkeypatch.setattr(_models.urllib.request, "urlopen", lambda *a, **k: pytest.fail("must not download"))
    assert ensure_model("hand", model_dir=str(model_dir)) == cached


def test_download_writes_atomically(model_dir, monkeypatch):
    import io

    payload = b"m" * 300_000

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    seen = []
    monkeypatch.setattr(_models.urllib.request, "urlopen", lambda url, timeout: (seen.append(url), Resp(payload))[1])
    path = ensure_model("face", model_dir=str(model_dir))
    assert path.read_bytes() == payload and seen == [MODEL_URLS["face"]]
    assert not list(model_dir.glob("*.part"))  # temp file renamed away


@pytest.mark.parametrize("failure", [urllib.error.URLError("offline"), OSError("disk full")])
def test_download_failure_gives_manual_instructions(model_dir, monkeypatch, failure):
    def boom(*a, **k):
        raise failure

    monkeypatch.setattr(_models.urllib.request, "urlopen", boom)
    with pytest.raises(ModelUnavailableError) as info:
        ensure_model("hand", model_dir=str(model_dir))
    msg = str(info.value)
    assert MODEL_URLS["hand"] in msg and "hand_model_path" in msg
    assert not list(model_dir.glob("*.part"))


def test_tiny_downloads_are_rejected_as_error_pages(model_dir, monkeypatch):
    import io

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(_models.urllib.request, "urlopen", lambda url, timeout: Resp(b"<html>403</html>"))
    with pytest.raises(ModelUnavailableError, match="implausibly small"):
        ensure_model("hand", model_dir=str(model_dir))
    assert not (model_dir / "hand_landmarker.task").exists()


def test_auto_download_off_fails_fast_with_instructions(model_dir):
    with pytest.raises(ModelUnavailableError, match="auto_download is off"):
        ensure_model("hand", model_dir=str(model_dir), auto_download=False)


# ------------------------------------------------------------------- tasks adapters
def lm(x, y, z=0.0, visibility=None):
    d = types.SimpleNamespace(x=x, y=y, z=z)
    if visibility is not None:
        d.visibility = visibility
    return d


@pytest.fixture
def tasks(monkeypatch, tmp_path):
    """Real MediaPipe option classes; the landmarker itself is replaced by a recorder."""
    from mediapipe.tasks.python import vision

    dummy = tmp_path / "dummy.task"
    dummy.write_bytes(b"0")
    monkeypatch.setattr(backend, "ensure_model", lambda kind, *a, **k: dummy)
    created = {}

    def install(cls_name, result):
        class Fake:
            def __init__(self, options):
                self.options = options
                self.closed = False

            def _run(self, image, ts):
                created["image"], created["ts"] = image, ts
                return result

            detect_for_video = _run

            def close(self):
                self.closed = True

        monkeypatch.setattr(getattr(vision, cls_name), "create_from_options", classmethod(lambda cls, options: Fake(options)))

    return install, vision


def frame(h=48, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_tasks_hand_adapter_builds_valid_options_and_converts_results(tasks):
    install, vision = tasks
    cat = types.SimpleNamespace(category_name="Left", score=0.93)
    result = types.SimpleNamespace(
        hand_landmarks=[[lm(i / 100, i / 200, 0.01 * i) for i in range(21)]], handedness=[[cat]]
    )
    install("HandLandmarker", result)
    cfg = HandConfig(max_hands=2, detection_confidence=0.6, tracking_confidence=0.55)
    det = backend._TasksHands(cfg, MediaPipeConfig(backend="tasks"))
    opts = det._det.options
    assert opts.num_hands == 2 and opts.min_hand_detection_confidence == 0.6
    assert opts.running_mode == vision.RunningMode.VIDEO
    hands = det.detect(frame(), 100)
    assert len(hands) == 1 and hands[0].label == "Left" and hands[0].score == pytest.approx(0.93)
    assert hands[0].landmarks.shape == (21, 3) and hands[0].landmarks[8] == pytest.approx([0.08, 0.04, 0.08])
    det.close()
    assert det._det.closed


def test_tasks_adapters_use_strictly_increasing_timestamps(tasks):
    install, _ = tasks
    install("HandLandmarker", types.SimpleNamespace(hand_landmarks=[], handedness=[]))
    det = backend._TasksHands(HandConfig(), MediaPipeConfig(backend="tasks"))
    seen = []
    orig = det._det.detect_for_video
    det._det.detect_for_video = lambda image, ts: (seen.append(ts), orig(image, ts))[1]
    for ts in (1000, 1000, 990, 1005):  # repeated / going backwards (paused video, clock jitter)
        det.detect(frame(), ts)
    assert seen == sorted(set(seen)) and len(seen) == 4


def test_tasks_face_adapter(tasks):
    install, _ = tasks
    install("FaceLandmarker", types.SimpleNamespace(face_landmarks=[[lm(0.5, 0.5)] * 478, [lm(0.1, 0.1)] * 478]))
    det = backend._TasksFace(FaceConfig(max_faces=2), MediaPipeConfig(backend="tasks"))
    assert det._det.options.num_faces == 2
    faces = det.detect(frame(), 1)
    assert len(faces) == 2 and faces[0].shape == (478, 3)


def test_tasks_pose_adapter_picks_the_model_variant_and_reads_visibility(tasks, monkeypatch):
    install, _ = tasks
    kinds = []
    monkeypatch.setattr(backend, "ensure_model", lambda kind, *a, **k: (kinds.append(kind), tasks_dummy)[1])
    tasks_dummy = __import__("pathlib").Path(__file__)  # any existing file; the fake landmarker never reads it
    install("PoseLandmarker", types.SimpleNamespace(pose_landmarks=[[lm(0.1, 0.2, 0.0, visibility=0.9)] * 33]))
    det = backend._TasksPose(PoseConfig(model_complexity=2), MediaPipeConfig(backend="tasks"))
    poses = det.detect(frame(), 1)
    assert kinds == ["pose_heavy"]
    assert poses[0].landmarks.shape == (33, 3) and poses[0].visibility == pytest.approx([0.9] * 33)


def test_hand_sensor_end_to_end_through_the_real_tasks_adapter(tasks, bare_engine):
    """HandSensor -> real factory -> Tasks adapter -> HandData, as it runs in production."""
    from conftest import make_ctx

    from open_ai_hci import HandSensor

    install, _ = tasks
    cat = types.SimpleNamespace(category_name="Right", score=0.9)
    install("HandLandmarker", types.SimpleNamespace(hand_landmarks=[[lm(0.5, 0.25)] * 21], handedness=[[cat]]))
    bare_engine.config.mediapipe.backend = "tasks"
    sensor = bare_engine.add_sensor(HandSensor())
    sensor.setup()
    ctx = make_ctx(w=64, h=48)
    ctx.rgb = np.zeros((48, 64, 3), np.uint8)
    events = []
    bare_engine.on("*", events.append)
    sensor.process(ctx)
    assert ctx.hand.handedness == "Right" and ctx.hand.index_tip == (32, 12)
    assert [e.name for e in events] == ["hand_detected"]
    sensor.close()


def test_model_missing_surfaces_as_a_clear_error_from_the_sensor(bare_engine, model_dir):
    from open_ai_hci import HandSensor

    bare_engine.config.mediapipe.backend = "tasks"
    bare_engine.config.mediapipe.auto_download = False
    bare_engine.config.mediapipe.model_dir = str(model_dir)
    sensor = bare_engine.add_sensor(HandSensor())
    with pytest.raises(ModelUnavailableError, match="hand_model_path"):
        sensor.setup()

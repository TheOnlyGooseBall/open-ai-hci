import json
import time

import numpy as np
import pytest
from conftest import hand_landmarks, make_ctx, make_hand, names

from open_ai_hci import Intent, MultimodalAgent, ONNXGestureClassifier
from open_ai_hci._optional import MissingOptionalDependency, install_hint, is_available, require
from open_ai_hci.features import landmarks_to_features
from open_ai_hci.processors.ai import parse_intent
from open_ai_hci.processors.ai.onnx_classifier import softmax
from open_ai_hci.types import CursorState


# ------------------------------------------------------------------ optional import helper
def test_require_gives_an_actionable_message():
    with pytest.raises(MissingOptionalDependency) as info:
        require("definitely_not_installed_pkg", "My feature")
    msg = str(info.value)
    assert "My feature" in msg and "definitely_not_installed_pkg" in msg and "pip install" in msg
    assert isinstance(info.value, ImportError)
    assert "open-ai-hci[ai]" in install_hint("onnxruntime") and "open-ai-hci[voice]" in install_hint("faster_whisper")
    assert is_available("numpy") and not is_available("definitely_not_installed_pkg")


def test_importing_the_ai_package_needs_no_ai_dependencies():
    import subprocess
    import sys

    code = (
        "import sys\n"
        "for m in ('onnxruntime','torch','websockets','faster_whisper','sounddevice'): sys.modules[m] = None\n"
        "import open_ai_hci\n"
        "from open_ai_hci import ONNXGestureClassifier, MultimodalAgent, WebSocketBridge\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(__import__("pathlib").Path(__file__).parent.parent))
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


# ------------------------------------------------------------------------------ intents
@pytest.mark.parametrize(
    "text,name,args",
    [
        ("Click", "click", {}),
        ("  left click here!  ", "click", {}),
        ("Double-click", "double_click", {}),
        ("right click", "right_click", {}),
        ("Scroll down.", "scroll_down", {"amount": 5}),
        ("scroll up", "scroll_up", {"amount": 5}),
        ("copy that", "hotkey", {"keys": "ctrl+c"}),
        ("select all", "hotkey", {"keys": "ctrl+a"}),
        ("press Enter", "press", {"key": "enter"}),
        ("type Hello, World", "type", {"text": "hello world"}),
        ("stop listening", "pause_listening", {}),
    ],
)
def test_parse_intent_known_phrases(text, name, args):
    intent = parse_intent(text)
    assert intent is not None and intent.name == name and intent.args == args and intent.text == text.strip()


@pytest.mark.parametrize("text", ["", "   ", "what a lovely day", "please click the thing on the left", "clicker"])
def test_parse_intent_ignores_everything_else(text):
    assert parse_intent(text) is None  # stray conversation must never move the mouse


def test_parse_intent_custom_phrase_table():
    table = [(r"^open (?P<app>\w+)$", "launch", {"how": "voice"})]
    i = parse_intent("Open Firefox", table)
    assert (i.name, i.args) == ("launch", {"how": "voice", "app": "firefox"})
    assert parse_intent("click", table) is None


# --------------------------------------------------------------------------- the agent
@pytest.fixture
def agent(bare_engine):
    a = bare_engine.add_processor(MultimodalAgent())
    a.setup()
    return a


def cursor_ctx(**kw):
    return make_ctx(cursor=CursorState(11.0, 22.0, 11.0, 22.0), **kw)


def test_voice_text_becomes_an_intent_with_gesture_context(bare_engine, agent):
    events = []
    bare_engine.on("*", events.append)
    agent.submit_text("click")
    ctx = cursor_ctx(hands=[make_hand(handedness="Left")])
    ctx.pinch_status, ctx.gesture = "HOLDING CLICK", "fist"
    agent.process(ctx)
    assert names(events) == ["voice_command", "intent"]
    intent = events[1]
    assert intent.intent == "click" and intent.origin == "voice" and intent.confidence == 1.0
    assert intent.context == {"x": 11.0, "y": 22.0, "pinching": True, "gesture": "fist", "handedness": "Left"}
    assert events[0].text == "click"


def test_unrecognised_speech_raises_an_alert_not_an_intent(bare_engine, agent):
    events = []
    bare_engine.on("*", events.append)
    agent.submit_text("the weather is nice")
    agent.process(cursor_ctx())
    assert names(events) == ["voice_command", "alert"] and "no command" in events[1].text


def test_pause_and_resume_listening(bare_engine, agent):
    events = []
    bare_engine.on("intent", events.append)
    for phrase in ["stop listening", "click", "start listening", "click"]:
        agent.submit_text(phrase)
    agent.process(cursor_ctx())
    assert [e.intent for e in events] == ["click"]  # only the post-resume click went through


def test_require_hand_for_and_fusers(bare_engine):
    seen, alerts = [], []
    bare_engine.on("intent", seen.append)
    bare_engine.on("alert", alerts.append)

    def drag_to_click(intent: Intent, ctx):
        if intent.name == "scroll_down" and ctx.pinch_status == "HOLDING CLICK":
            return None  # veto: don't scroll mid-drag
        return intent

    a = bare_engine.add_processor(MultimodalAgent(require_hand_for=["click"], fusers=[drag_to_click]))
    a.setup()
    a.submit_text("click")
    a.process(cursor_ctx())  # no hand
    assert seen == [] and "needs a visible hand" in alerts[-1].text

    a.submit_text("click")
    a.process(cursor_ctx(hands=[make_hand()]))
    assert [e.intent for e in seen] == ["click"]

    a.submit_text("scroll down")
    ctx = cursor_ctx()
    ctx.pinch_status = "HOLDING CLICK"
    a.process(ctx)
    assert len(seen) == 1  # vetoed by the fuser


def test_custom_parser(bare_engine):
    seen = []
    bare_engine.on("intent", seen.append)
    a = bare_engine.add_processor(
        MultimodalAgent(intent_parser=lambda t: Intent("custom", {"len": len(t)}, text=t, confidence=0.5) if "go" in t else None)
    )
    a.setup()
    a.submit_text("let's go")
    a.process(cursor_ctx())
    assert seen[0].intent == "custom" and seen[0].args == {"len": 8} and seen[0].confidence == 0.5


def test_vlm_runs_in_the_background_and_can_produce_intents(bare_engine):
    events, calls = [], []

    def fake_vlm(frame, prompt):
        calls.append((frame.shape, prompt))
        time.sleep(0.05)
        return "There is a dialog with an OK button. click"

    a = bare_engine.add_processor(
        MultimodalAgent(vlm=fake_vlm, vlm_intent_parser=lambda text: Intent("click", {}, text=text) if "OK button" in text else None)
    )
    a.setup()
    bare_engine.on("*", events.append)
    a.ask("What is on screen?")
    t0 = time.monotonic()
    a.process(cursor_ctx())  # schedules the call; must not block for the model
    assert time.monotonic() - t0 < 0.04
    deadline = time.monotonic() + 3
    while "intent" not in names(events) and time.monotonic() < deadline:
        a.process(cursor_ctx())
        time.sleep(0.01)
    a.close()
    assert calls == [((720, 1280, 3), "What is on screen?")]
    result = next(e for e in events if e.name == "vlm_result")
    assert result.prompt == "What is on screen?" and "OK button" in result.text
    intent = next(e for e in events if e.name == "intent")
    assert intent.intent == "click" and intent.origin == "vlm"


def test_vlm_errors_are_reported_not_raised(bare_engine):
    events = []
    a = bare_engine.add_processor(MultimodalAgent(vlm=lambda f, p: 1 / 0, vlm_intent_parser=lambda t: Intent("click")))
    a.setup()
    bare_engine.on("*", events.append)
    a.ask()
    deadline = time.monotonic() + 3
    while "vlm_result" not in names(events) and time.monotonic() < deadline:
        a.process(cursor_ctx())
        time.sleep(0.01)
    a.close()
    assert events[-1].name == "vlm_result" and events[-1].text.startswith("[error]")
    assert "intent" not in names(events)  # an error string is never parsed into an action


def test_ask_requires_a_vlm(agent):
    with pytest.raises(RuntimeError, match="vlm"):
        agent.ask()


def test_voice_dependencies_missing_degrades_gracefully(bare_engine, monkeypatch, caplog):
    import open_ai_hci.processors.ai.multimodal_agent as mod

    def fake_require(module, feature):
        raise MissingOptionalDependency(f"{feature} requires {module}")

    monkeypatch.setattr(mod, "require", fake_require)
    a = bare_engine.add_processor(MultimodalAgent(voice=True))
    a.setup()  # must not raise
    assert a._voice is None and "Voice commands disabled" in caplog.text and "open-ai-hci[voice]" in caplog.text
    a.submit_text("click")  # typed/other-STT input still works
    seen = []
    bare_engine.on("intent", seen.append)
    a.process(cursor_ctx())
    assert len(seen) == 1
    with pytest.raises(MissingOptionalDependency):
        bare_engine.add_processor(MultimodalAgent(voice=True, strict=True)).setup()


def test_voice_listener_pipeline_with_a_custom_transcriber():
    """VAD + transcription loop, driven by a fake sounddevice module (no microphone needed)."""
    import queue
    import threading
    import types

    from open_ai_hci.processors.ai import VoiceListener

    sr = VoiceListener.SAMPLE_RATE
    block = int(sr * 0.05)
    loud = (np.sin(np.linspace(0, 200, block)) * 0.3).astype(np.float32)
    quiet = np.zeros(block, dtype=np.float32)
    script = [quiet] * 3 + [loud] * 12 + [quiet] * 20  # one utterance followed by silence

    class FakeStream:
        def __init__(self, callback, **kw):
            self.callback = callback

        def __enter__(self):
            def feed():
                for chunk in script:
                    self.callback(chunk.reshape(-1, 1), block, None, None)
                    time.sleep(0.002)

            threading.Thread(target=feed, daemon=True).start()
            return self

        def __exit__(self, *a):
            return False

    fake_sd = types.SimpleNamespace(InputStream=lambda **kw: FakeStream(kw["callback"]))
    got = queue.Queue()
    lengths = []

    def transcribe(audio):
        lengths.append(len(audio) / sr)
        return "scroll down"

    vl = VoiceListener(got.put, transcriber=transcribe, silence_s=0.5)
    vl._stop.clear()
    t = threading.Thread(target=vl._run, args=(fake_sd,), daemon=True)
    t.start()
    try:
        assert got.get(timeout=5) == "scroll down"
        assert 0.6 <= lengths[0] <= 1.5  # ~12 loud blocks + trailing silence
    finally:
        vl._stop.set()
        t.join(3)


# ------------------------------------------------------------------------------- ONNX
class FakeSession:
    """Mimics onnxruntime.InferenceSession for a model driven by one scripted output per call.

    ``label_first=True`` mimics scikit-learn exports, which return ``[label, probabilities]``.
    """

    def __init__(self, outputs, label_first=False):
        self.outputs = list(outputs)
        self.inputs, self.requested = [], []
        self.label_first = label_first

    def get_inputs(self):
        return [type("I", (), {"name": "input"})()]

    def run(self, output_names, feed):
        self.inputs.append(feed["input"])
        self.requested.append(output_names)
        probs = np.asarray([self.outputs.pop(0)], dtype=np.float32)
        if self.label_first:
            return [np.asarray([int(probs.argmax())], dtype=np.int64), probs]
        return [probs]


def test_softmax_is_stable_and_normalised():
    p = softmax(np.array([1000.0, 1001.0, 999.0]))
    assert p.sum() == pytest.approx(1.0) and np.all(np.isfinite(p)) and p.argmax() == 1


def test_classifier_debounces_and_emits_gesture_events(bare_engine):
    labels = ["none", "fist", "open"]
    probs = [
        [0.9, 0.05, 0.05],  # none
        [0.1, 0.8, 0.1],  # fist x1 (not yet stable)
        [0.1, 0.8, 0.1],  # fist x2 -> fires with stable_frames=2
        [0.1, 0.8, 0.1],  # fist x3 (no repeat)
        [0.1, 0.1, 0.8],  # open x1
        [0.1, 0.1, 0.8],  # open x2 -> fires, previous=fist
        [0.4, 0.3, 0.3],  # low confidence
        [0.4, 0.3, 0.3],  # -> gesture_end
    ]
    session = FakeSession(probs)
    clf = bare_engine.add_processor(ONNXGestureClassifier(session=session, labels=labels, min_confidence=0.6, stable_frames=2))
    clf.setup()
    events = []
    bare_engine.on("*", events.append)
    ctxs = []
    for _ in probs:
        ctx = make_ctx(hands=[make_hand(handedness="Left")])
        clf.process(ctx)
        ctxs.append(ctx)
    assert names(events) == ["gesture", "gesture_fist", "gesture", "gesture_open", "gesture_end"]
    first, second = events[0], events[2]
    assert first.gesture == "fist" and first.previous is None and first.confidence == pytest.approx(0.8)
    assert second.gesture == "open" and second.previous == "fist" and second.handedness == "Left"
    assert events[-1].gesture == "open"
    assert [c.gesture for c in ctxs] == [None, None, "fist", "fist", "fist", "open", "open", None]
    assert session.inputs[0].shape == (1, 63) and session.inputs[0].dtype == np.float32


def test_classifier_hand_loss_ends_the_gesture(bare_engine):
    clf = bare_engine.add_processor(
        ONNXGestureClassifier(session=FakeSession([[0.1, 0.9]] * 2), labels=["none", "fist"], stable_frames=1)
    )
    clf.setup()
    events = []
    bare_engine.on("*", events.append)
    clf.process(make_ctx(hands=[make_hand()]))
    clf.process(make_ctx(hands=[]))
    assert names(events) == ["gesture", "gesture_fist", "gesture_end"]


def test_classifier_disables_itself_when_onnxruntime_is_missing(bare_engine, monkeypatch, caplog, tmp_path):
    import open_ai_hci.processors.ai.onnx_classifier as mod

    def fake_require(module, feature):
        raise MissingOptionalDependency(f"{feature} requires the optional package 'onnxruntime'")

    monkeypatch.setattr(mod, "require", fake_require)
    clf = bare_engine.add_processor(ONNXGestureClassifier(tmp_path / "m.onnx"))
    clf.setup()  # no exception: the core keeps working
    assert clf.enabled is False and "open-ai-hci[ai]" in caplog.text
    ctx = make_ctx(hands=[make_hand()])
    clf.process(ctx)  # silently a no-op
    assert ctx.gesture is None
    with pytest.raises(MissingOptionalDependency):
        bare_engine.add_processor(ONNXGestureClassifier(tmp_path / "m.onnx", strict=True)).setup()


@pytest.mark.skipif(not (is_available("onnx") and is_available("onnxruntime")), reason="onnx/onnxruntime not installed")
def test_real_onnx_model_round_trip(bare_engine, tmp_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    W = rng.normal(size=(63, 3)).astype(np.float32)
    b = rng.normal(size=(3,)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "W"], ["mm"]),
            helper.make_node("Add", ["mm", "b"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["probs"], axis=1),
        ],
        "tiny_gesture_net",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 63])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, [1, 3])],
        [numpy_helper.from_array(W, "W"), numpy_helper.from_array(b, "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "gesture.onnx"
    onnx.save(model, str(path))
    (tmp_path / "gesture.labels.json").write_text(json.dumps(["none", "fist", "open"]))

    clf = bare_engine.add_processor(ONNXGestureClassifier(path, stable_frames=1, min_confidence=0.0))
    clf.setup()
    assert clf.enabled and clf.labels == ["none", "fist", "open"]

    lm = hand_landmarks((600, 300), (560, 340))
    feats = landmarks_to_features(lm, "Right")
    expected = softmax(feats @ W + b)
    label, conf = clf.predict(lm, "Right")
    assert label == ["none", "fist", "open"][int(expected.argmax())]
    assert conf == pytest.approx(float(expected.max()), abs=1e-5)

    events = []
    bare_engine.on("gesture", events.append)
    clf.process(make_ctx(hands=[make_hand()]))
    assert len(events) <= 1  # fires when the label is not the background class


def test_classifier_requires_a_model_or_session():
    with pytest.raises(ValueError):
        ONNXGestureClassifier()
    with pytest.raises(ValueError):
        ONNXGestureClassifier(session=FakeSession([]), stable_frames=0)


def test_models_with_a_label_output_before_the_probabilities_are_handled(bare_engine):
    """scikit-learn / skl2onnx style: outputs are [label (int64), probabilities (float)]."""
    session = FakeSession([[0.1, 0.9], [0.8, 0.2]], label_first=True)
    clf = bare_engine.add_processor(ONNXGestureClassifier(session=session, labels=["none", "fist"], min_confidence=0.0, stable_frames=1, background_label=None))
    clf.setup()
    assert clf.predict(hand_landmarks((600, 300), (560, 340))) == ("fist", pytest.approx(0.9))
    assert clf.predict(hand_landmarks((600, 300), (560, 340))) == ("none", pytest.approx(0.8))


def test_output_name_is_forwarded_to_the_runtime():
    session = FakeSession([[0.2, 0.8]])
    clf = ONNXGestureClassifier(session=session, labels=["a", "b"], output_name="probs")
    clf.setup()
    clf.predict(hand_landmarks((600, 300), (560, 340)))
    assert session.requested == [["probs"]]


@pytest.mark.skipif(not (is_available("sklearn") and is_available("skl2onnx") and is_available("onnxruntime")), reason="scikit-learn/skl2onnx not installed")
def test_readme_training_recipe_end_to_end(tmp_path):
    """record -> load_dataset -> train sklearn -> export ONNX -> ONNXGestureClassifier, for both hands."""
    from skl2onnx import to_onnx
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    from open_ai_hci.recording import GestureRecorder, load_dataset

    rng = np.random.default_rng(1)

    def pose(kind, hand):
        lm = np.zeros((21, 3))
        lm[0] = (0.5, 0.9, 0)
        reach = 0.35 if kind == "open" else 0.12
        for f, mcp in enumerate((5, 9, 13, 17)):
            for j in range(4):
                lm[mcp + j] = (0.4 + f * 0.07, 0.65 - reach * (j + 1) / 4, 0)
        lm += rng.normal(0, 0.004, lm.shape)
        if hand == "Left":  # a real left hand is the mirror image
            lm[:, 0] = 1.0 - lm[:, 0]
        return lm

    with GestureRecorder(tmp_path / "data.csv") as rec:
        for i in range(120):
            hand = "Left" if i % 2 else "Right"
            rec.add("open", pose("open", hand), hand)
            rec.add("fist", pose("fist", hand), hand)

    X, y, _ = load_dataset(tmp_path / "data.csv")
    labels = sorted(set(y))
    y_idx = np.array([labels.index(v) for v in y])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_idx, test_size=0.2, random_state=0)
    model = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=0).fit(X_tr, y_tr)
    assert model.score(X_te, y_te) > 0.95
    onnx_model = to_onnx(model, X[:1], options={id(model): {"zipmap": False}}, target_opset=15)
    (tmp_path / "gesture.onnx").write_bytes(onnx_model.SerializeToString())
    (tmp_path / "gesture.labels.json").write_text(json.dumps(labels))

    clf = ONNXGestureClassifier(tmp_path / "gesture.onnx")
    clf.setup()
    assert clf.enabled and clf.labels == ["fist", "open"]
    for kind in ("open", "fist"):
        for hand in ("Right", "Left"):  # one model serves both hands thanks to mirroring
            label, conf = clf.predict(pose(kind, hand), hand)
            assert label == kind and conf > 0.8, (kind, hand, label, conf)

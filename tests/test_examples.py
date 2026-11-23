"""The shipped examples must import cleanly and do what their docstrings say."""
import importlib.util
import threading
import time
from pathlib import Path

import pytest

from open_ai_hci import WebSocketBridge
from open_ai_hci._optional import is_available

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), EXAMPLES / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_01_spells_out_the_prototype_thresholds():
    cfg = load("01_mouse_hud_demo.py").config
    assert (cfg.camera.width, cfg.camera.height, cfg.camera.frame_skip) == (1280, 720, 2)
    assert (cfg.smoothing.kind, cfg.smoothing.alpha) == ("ema", 0.20)
    assert cfg.hand.pinch_threshold_px == 45
    assert (cfg.face.flick_threshold_px, cfg.face.flick_cooldown_s) == (28, 1.2)
    assert cfg.mouse.failsafe is True
    assert (cfg.hand.right_pinch_enabled, cfg.hand.pause_gesture, cfg.hand.pause_hold_s) == (True, "open_palm", 1.0)
    assert cfg.face.flick_enabled is False and cfg.mouse.right_click_on_flick is False  # v0.2: no accidental flick clicks
    cfg.validate()


def test_example_02_wires_callbacks_and_hotkeys():
    hci = load("02_custom_callbacks.py").hci
    assert hci.config.face.flick_enabled and [s.name for s in hci.sensors] == ["hand", "face"]
    for name in ("pinch_start", "pinch_end", "drag_end", "head_flick", "celebrate", "hand_detected", "hand_lost", "right_click", "control_paused"):
        assert hci.bus.has_subscribers(name), name
    assert hci.actions[0].name == "os_keyboard" and hci.actions[0].dry_run is True
    assert not any(a.name == "os_mouse" for a in hci.actions)  # the demo never touches the real cursor
    got = []
    hci.on("celebrate", got.append)
    hci.emit("head_flick", direction="left", dx=-40.0, dy=0.0, x=1, y=1)
    assert got and got[0].reason == "flick"


@pytest.mark.skipif(not is_available("websockets"), reason="websockets not installed")
def test_example_03_client_receives_what_the_bridge_sends(bare_engine, capsys):
    example = load("03_websocket_stream.py")
    bridge = bare_engine.add_action(WebSocketBridge(port=0))
    bare_engine.setup()
    client = threading.Thread(target=lambda: _quiet(example.run_client, bridge.url), daemon=True)
    client.start()
    try:
        deadline = time.monotonic() + 5
        seen = ""
        while "pinch_start" not in seen and time.monotonic() < deadline:
            bare_engine.emit("pinch_start", x=1.0, y=2.0, distance=3.0, handedness="Right")
            time.sleep(0.05)
            seen += capsys.readouterr().out
        assert "connected;" in seen and "pinch_start" in seen
    finally:
        bare_engine.close()
        client.join(5)


def _quiet(fn, *args):
    try:
        fn(*args)
    except Exception:  # connection closed by the server shutting down
        pass


def test_example_04_wires_the_onnx_gesture_pipeline(bare_engine):
    from types import SimpleNamespace

    import numpy as np
    from conftest import FakePyAutoGUI, make_ctx, make_hand

    from open_ai_hci import ONNXGestureClassifier, OSMouseAction

    example = load("04_ai_custom_gestures.py")

    class Session:  # always answers "right_click" with high confidence
        def get_inputs(self):
            return [SimpleNamespace(name="x")]

        def run(self, names, feed):
            return [np.array([[0.02, 0.98]], dtype=np.float32)]

    clf = ONNXGestureClassifier(session=Session(), labels=["none", "right_click"], min_confidence=0.7, stable_frames=1)
    hci = example.build_engine("models/none.onnx", live=False, classifier=clf)
    assert hci.config.onnx.enabled and hci.get(ONNXGestureClassifier) is clf and not hci.config.mouse.enabled
    kb = [a for a in hci.actions if a.name == "os_keyboard"][0]
    assert kb.dry_run is True and "gesture_peace" in kb.bindings
    # a recognised right_click gesture becomes a right_click event (and clicks, once the mouse is live)
    events = []
    hci.on("right_click", events.append)
    pag = FakePyAutoGUI()
    hci.add_action(OSMouseAction(pyautogui_module=pag))
    hci.sensors.clear()  # no camera / MediaPipe model download in the test
    hci.config.screen.width, hci.config.screen.height = 1920, 1080
    hci.setup()
    ctx = make_ctx(hands=[make_hand()])
    clf.process(ctx)
    assert ctx.gesture == "right_click" and [e.via for e in events] == ["ai_gesture"] and pag.names() == ["click"]
    hci.close()


def test_example_05_builds_a_dry_run_assistant_that_talks_to_a_mocked_llm():
    import json

    from conftest import make_ctx
    from test_llm import FakeHTTP, actions_json, openai_reply

    from open_ai_hci import LLMRouter

    example = load("05_llm_multimodal_assistant.py")
    args = example.argparse.Namespace(
        provider="lmstudio", model="qwen2.5", base_url=None, signature="Best, Sam", voice=False, live=False, no_hand=True
    )
    http = FakeHTTP(openai_reply(actions_json({"action": "type", "text": "Best, Sam"})))
    hci = example.build_engine(args, http=http, env={})
    assert hci.config.llm.dry_run is True and hci.actions == [] and hci.get(LLMRouter) is not None
    hci.setup()
    seen = []
    hci.on("intent", seen.append)
    hci.get(LLMRouter).submit("type my email signature")
    assert hci.get(LLMRouter).wait_idle(5)
    hci.process_frame(make_ctx().frame)
    assert [(e.intent, e.dry_run) for e in seen] == [("type", True)]
    assert "Best, Sam" in json.dumps(http.requests[0]["json"]["messages"][0]["content"])
    hci.close()


def test_example_06_launches_the_gui_entry_point():
    module = load("06_desktop_gui.py")
    from open_ai_hci.ui.gui import main

    assert module.main is main

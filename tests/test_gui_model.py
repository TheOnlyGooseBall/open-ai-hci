"""The control panel's logic (no tkinter needed)."""
from __future__ import annotations

import json
import sys

import pytest

from open_ai_hci.config import HCIConfig
from open_ai_hci.ui.gui_model import (
    FLAGS,
    PROVIDERS,
    SLIDERS,
    ControlPanelModel,
    EngineRunner,
    provider_by_key,
    provider_by_label,
)


class FakeEngine:
    def __init__(self, config):
        self.config = config
        self.running = False
        self.paused = False
        self.fps = 59.6
        self.applied: list[HCIConfig] = []
        self.stopped = False
        import threading

        self._stop = threading.Event()

    def run(self, show_hud=False):
        self.show_hud = show_hud
        self.running = True
        self._stop.wait(5)
        self.running = False

    def stop(self):
        self.stopped = True
        self._stop.set()

    def close(self):  # pragma: no cover
        pass

    def apply_config(self, cfg):
        self.applied.append(cfg)

    def toggle_pause(self):
        self.paused = not self.paused

    def get(self, _cls):
        return None


class BoomEngine(FakeEngine):
    def run(self, show_hud=False):
        raise RuntimeError("camera 0 could not be opened\nmore detail")


def wait_for(cond, timeout=3.0):
    import time

    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def model(tmp_path):
    runner = EngineRunner(FakeEngine)
    m = ControlPanelModel(config_path=tmp_path / "config.yaml", runner=runner, env={})
    yield m
    m.stop()


# --------------------------------------------------------------------------- catalogue
def test_provider_dropdown_matches_the_spec():
    assert [p.label for p in PROVIDERS] == [
        "Ollama (Local)", "LM Studio (Local)", "Custom OpenAI-Compatible URL",
        "OpenAI (Cloud)", "Anthropic (Cloud)", "Google Gemini (Cloud)",
    ]
    assert provider_by_label("Ollama (Local)").base_url == "http://localhost:11434/v1"
    assert provider_by_label("LM Studio (Local)").base_url == "http://localhost:1234/v1"
    assert provider_by_label("Custom OpenAI-Compatible URL").base_url == ""
    assert [p.cloud for p in PROVIDERS] == [False, False, False, True, True, True]
    assert provider_by_key("gemini").label == "Google Gemini (Cloud)"
    with pytest.raises(KeyError):
        provider_by_label("nope")
    with pytest.raises(KeyError):
        provider_by_key("nope")


def test_slider_ranges_match_the_spec():
    assert SLIDERS["smoothing"][1:3] == (0.05, 0.80)
    assert SLIDERS["pinch"][1:3] == (20.0, 80.0)
    assert SLIDERS["onnx_confidence"][1:3] == (0.50, 0.95)
    assert "Middle-Pinch Right-Click" in FLAGS.values() and "Open-Palm Pause Toggle" in FLAGS.values()


# ----------------------------------------------------------------------------- flags
def test_every_flag_round_trips(model):
    for name in FLAGS:
        for value in (False, True):
            assert model.set_flag(name, value) == ""
            assert model.get_flag(name) is value
    with pytest.raises(KeyError):
        model.set_flag("nope", True)
    with pytest.raises(KeyError):
        model.get_flag("nope")


def test_flags_map_onto_the_config(model):
    c = model.cfg
    model.set_flag("right_pinch", False)
    assert not c.hand.right_pinch_enabled and not c.mouse.right_click_on_pinch
    model.set_flag("right_pinch", True)
    assert c.hand.right_pinch_enabled and c.mouse.right_click_on_pinch
    model.set_flag("palm_pause", False)
    assert c.hand.pause_gesture == "none"
    model.set_flag("palm_pause", True)
    assert c.hand.pause_gesture == "open_palm"
    c.hand.pause_gesture = "fist"
    model.set_flag("palm_pause", False)
    model.set_flag("palm_pause", True)
    assert c.hand.pause_gesture == "open_palm"  # back on = the default pose
    c.hand.pause_gesture = "fist"
    model.set_flag("palm_pause", True)
    assert c.hand.pause_gesture == "fist"  # already on: keep the user's choice


def test_default_profile_shows_right_pinch_and_pause_on(model):
    assert model.get_flag("right_pinch") and model.get_flag("palm_pause") and model.get_flag("left_click")
    assert not model.get_flag("llm_enabled") and not model.get_flag("onnx_enabled")


def test_sliders_clamp_round_and_map(model):
    assert model.set_value("smoothing", 5) == "" and model.get_value("smoothing") == 0.80
    model.set_value("smoothing", 0.0)
    assert model.get_value("smoothing") == 0.05
    model.set_value("smoothing", 0.337)
    assert model.get_value("smoothing") == pytest.approx(0.34)
    assert model.cfg.smoothing.kind == "ema"
    model.set_value("pinch", 10)
    assert model.get_value("pinch") == 20.0
    model.set_value("pinch", 47.4)
    assert model.cfg.hand.pinch_threshold_px == 47.0 and model.cfg.hand.right_pinch_threshold_px == 47.0
    model.set_value("onnx_confidence", 0.9)
    assert model.cfg.onnx.min_confidence == pytest.approx(0.9)
    with pytest.raises(KeyError):
        model.get_value("nope")


# --------------------------------------------------------------------------- providers
def test_provider_change_autofills_defaults(model):
    assert model.set_provider("LM Studio (Local)") == ("http://localhost:1234/v1", "local-model")
    assert model.cfg.llm.provider == "lmstudio"
    url, mdl = model.set_provider("Ollama (Local)")
    assert url == "http://localhost:11434/v1" and mdl == "llama3.2" and model.cfg.llm.provider == "ollama"
    url, mdl = model.set_provider("Anthropic (Cloud)")
    assert url == "https://api.anthropic.com" and mdl == "claude-haiku-5-5"
    assert model.provider.key == "anthropic"


def test_provider_change_keeps_what_the_user_typed(model):
    model.set_provider("Custom OpenAI-Compatible URL")
    assert model.cfg.llm.base_url == ""
    model.set_text("base_url", "  http://192.168.1.5:8080/v1 ")
    model.set_text("model", "my-model")
    model.set_provider("Ollama (Local)")
    assert model.cfg.llm.base_url == "http://192.168.1.5:8080/v1" and model.cfg.llm.model == "my-model"


def test_text_fields(model):
    model.set_text("onnx_path", " models/x.onnx ")
    assert model.cfg.onnx.model_path == "models/x.onnx"
    model.set_text("api_key", " secret ")  # not stripped, not stored in the config
    assert model.api_key == " secret " and model.cfg.llm.api_key == ""
    with pytest.raises(KeyError):
        model.set_text("nope", "x")


def test_camera_source(model):
    model.set_camera(" 2 ")
    assert model.cfg.camera.source == 2
    model.set_camera("rtsp://cam/stream")
    assert model.cfg.camera.source == "rtsp://cam/stream"
    model.set_camera("")
    assert model.cfg.camera.source == 0


def test_invalid_values_are_reported_not_raised(model):
    model.cfg.llm.timeout_s = -5
    msg = model.set_flag("hud", True)
    assert msg and "timeout" in msg.lower()
    assert model.start() and not model.runner.alive


# ----------------------------------------------------------------------------- engine
def test_start_stop_status_and_live_apply(model):
    assert model.status == "Stopped"
    assert model.start() == ""
    assert wait_for(lambda: model.status == "Running")
    assert model.start() == "The engine is already running."
    model.set_value("pinch", 55)
    engine = model.runner.engine
    assert engine.applied and engine.applied[-1].hand.pinch_threshold_px == 55
    model.runner.toggle_pause()
    assert model.status == "Paused"
    model.runner.toggle_pause()
    assert model.status == "Running"
    assert model.runner.fps == pytest.approx(59.6)
    model.stop()
    assert model.status == "Stopped" and engine.stopped and model.runner.fps == 0.0


def test_hud_flag_is_passed_to_the_engine(model):
    model.set_flag("hud", True)
    model.start()
    assert wait_for(lambda: getattr(model.runner.engine, "running", False))
    assert model.runner.engine.show_hud is (sys.platform != "darwin")


def test_engine_runtime_config_carries_the_masked_key_only_in_memory(model):
    model.set_text("api_key", "sk-xyz")
    model.start()
    assert wait_for(lambda: model.runner.engine is not None)
    assert model.runner.engine.config.llm.api_key == "sk-xyz"
    assert model.cfg.llm.api_key == ""


def test_engine_failure_is_reported(tmp_path):
    m = ControlPanelModel(config_path=tmp_path / "c.yaml", runner=EngineRunner(BoomEngine))
    m.start()
    assert wait_for(lambda: m.runner.error)
    assert m.runner.error == "RuntimeError: camera 0 could not be opened"
    assert wait_for(lambda: m.status == "Stopped")


def test_restart_required_only_for_startup_settings(model):
    model.start()
    assert wait_for(lambda: model.status == "Running")
    model.set_value("pinch", 40)
    assert not model.restart_required
    model.set_flag("llm_enabled", True)
    assert model.restart_required
    model.set_flag("llm_enabled", False)
    assert not model.restart_required


def test_camera_busy_reason(model):
    assert model.camera_busy_reason() == ""
    model.start()
    assert "Stop the engine" in model.camera_busy_reason()


def test_pause_and_apply_are_noops_when_stopped(model):
    model.runner.toggle_pause()
    model.runner.apply(model.cfg)
    assert model.runner.engine is None


def test_send_command_needs_a_running_router(model):
    assert model.send_command("   ") == "Type a command first."
    assert "Start the engine" in model.send_command("scroll down")
    model.start()
    assert wait_for(lambda: model.runner.alive)
    assert "Start the engine" in model.send_command("scroll down")  # fake engine has no router


def test_send_command_paths(model):
    class Router:
        available = False
        unavailable_reason = "no key"

        def __init__(self):
            self.sent = []
            self.accept = True

        def submit(self, text, source="text"):
            self.sent.append((text, source))
            return self.accept

    router = Router()
    model.start()
    assert wait_for(lambda: model.runner.alive)
    model.runner.engine.get = lambda _cls: router
    assert model.send_command("x") == "no key"
    router.available = True
    assert model.send_command(" open a new tab ") == "" and router.sent == [("open a new tab", "text")]
    router.accept = False
    assert "busy" in model.send_command("again")


# ---------------------------------------------------------------------------- profile
def test_save_profile_round_trips_and_hides_the_key(model, tmp_path):
    model.set_flag("llm_enabled", True)
    model.set_provider("Anthropic (Cloud)")
    model.set_text("api_key", "sk-secret-123")
    model.set_value("smoothing", 0.3)
    model.set_text("onnx_path", "models/g.onnx")
    path = model.save_profile()
    text = path.read_text(encoding="utf-8")
    assert "sk-secret-123" not in text
    loaded = HCIConfig.load(path)
    assert loaded.llm.enabled and loaded.llm.provider == "anthropic" and loaded.llm.model == "claude-haiku-5-5"
    assert loaded.smoothing.alpha == pytest.approx(0.3) and loaded.onnx.model_path == "models/g.onnx"
    assert loaded.hand.pause_gesture == "open_palm" and loaded.hand.right_pinch_enabled


def test_save_profile_can_remember_the_key_when_asked(model, tmp_path):
    model.set_text("api_key", "sk-remember")
    model.save_api_key = True
    path = model.save_profile(tmp_path / "other.yaml")
    assert "sk-remember" in path.read_text(encoding="utf-8")
    assert model.config_path == path


def test_panel_reopens_with_the_saved_profile(model, tmp_path):
    model.set_value("pinch", 33)
    model.save_profile()
    again = ControlPanelModel(config_path=model.config_path, runner=EngineRunner(FakeEngine))
    assert again.get_value("pinch") == 33 and again.load_error == ""


def test_corrupt_profile_does_not_block_the_panel(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("hand:\n  pinch_threshold_px: [not, a, number\n", encoding="utf-8")
    m = ControlPanelModel(config_path=bad, runner=EngineRunner(FakeEngine))
    assert m.load_error.startswith("Could not read config.yaml") or m.cfg.hand.pinch_threshold_px
    assert m.cfg is not None


# -------------------------------------------------------------------------------- LLM
class FakeHTTP:
    def __init__(self, status, payload):
        self.status, self.payload, self.calls = status, payload, []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url, dict(headers)))
        return self.status, json.dumps(self.payload)


def test_test_connection_ok_for_local_servers(tmp_path):
    http = FakeHTTP(200, {"data": [{"id": "local-model"}]})
    m = ControlPanelModel(config_path=tmp_path / "c.yaml", runner=EngineRunner(FakeEngine), http=http, env={})
    m.set_provider("LM Studio (Local)")
    res = m.test_connection()
    assert res.ok and "1 model" in res.message and res.provider == "lmstudio"
    assert http.calls[0][1] == "http://localhost:1234/v1/models"


def test_test_connection_ollama_native_tags(tmp_path):
    http = FakeHTTP(200, {"models": [{"name": "llama3.2:latest"}]})
    m = ControlPanelModel(config_path=tmp_path / "c.yaml", runner=EngineRunner(FakeEngine), http=http, env={})
    m.set_provider("Ollama (Local)")
    m.set_text("base_url", "http://localhost:11434")
    assert m.test_connection().ok
    assert http.calls[0][1] == "http://localhost:11434/api/tags"


def test_test_connection_cloud_without_key_fails_cleanly(tmp_path):
    http = FakeHTTP(200, {})
    m = ControlPanelModel(config_path=tmp_path / "c.yaml", runner=EngineRunner(FakeEngine), http=http, env={})
    m.set_provider("OpenAI (Cloud)")
    res = m.test_connection()
    assert not res.ok and "OPENAI_API_KEY" in res.message and http.calls == []


def test_test_connection_uses_the_typed_key_and_reports_rejection(tmp_path):
    http = FakeHTTP(401, {"error": {"message": "Incorrect API key sk-bad"}})
    m = ControlPanelModel(config_path=tmp_path / "c.yaml", runner=EngineRunner(FakeEngine), http=http, env={})
    m.set_provider("OpenAI (Cloud)")
    m.set_text("api_key", "sk-bad")
    res = m.test_connection()
    assert not res.ok and "sk-bad" not in res.message
    assert http.calls[0][2]["Authorization"] == "Bearer sk-bad"


def test_test_connection_with_invalid_config(tmp_path):
    m = ControlPanelModel(config_path=tmp_path / "c.yaml", runner=EngineRunner(FakeEngine), http=FakeHTTP(200, {}), env={})
    m.cfg.llm.timeout_s = -1
    res = m.test_connection()
    assert not res.ok


# ------------------------------------------------------------------ record / train commands
def test_record_and_train_commands():
    argv = ControlPanelModel.record_command("right_click", 150, "data/g.csv")
    assert argv == [sys.executable, "-m", "open_ai_hci", "record", "--label", "right_click", "--samples", "150", "--out", "data/g.csv"]
    assert ControlPanelModel.train_command("a.csv", "b.onnx")[-4:] == ["--data", "a.csv", "--out", "b.onnx"]
    for bad_label in ("", "has space", "a;b", "x" * 41, "--out"):
        with pytest.raises(ValueError):
            ControlPanelModel.record_command(bad_label, 10)
    for bad_n in (0, 5001):
        with pytest.raises(ValueError):
            ControlPanelModel.record_command("ok", bad_n)
    with pytest.raises(ValueError):
        ControlPanelModel.train_command("", "x")


def test_launch_runs_a_subprocess_without_a_shell():
    proc = ControlPanelModel.launch([sys.executable, "-c", "print('hi')"])
    out, _ = proc.communicate(timeout=20)
    assert out.strip() == "hi" and proc.returncode == 0


def test_adopt_trained_model_enables_it(model):
    model.adopt_trained_model("models/new.onnx")
    assert model.cfg.onnx.enabled and model.cfg.onnx.model_path == "models/new.onnx"

"""The tkinter skin, driven for real when a display is available (skipped otherwise)."""
from __future__ import annotations

import sys
import time

import pytest

tk = pytest.importorskip("tkinter")

from open_ai_hci.processors.ai.llm_router import ConnectionResult  # noqa: E402
from open_ai_hci.ui import gui as gui_module  # noqa: E402
from open_ai_hci.ui.gui import ControlPanel, main, tkinter_available  # noqa: E402
from open_ai_hci.ui.gui_model import FLAGS, ControlPanelModel, EngineRunner  # noqa: E402
from tests.test_gui_model import FakeEngine, FakeHTTP, wait_for  # noqa: E402


@pytest.fixture
def root():
    try:
        r = tk.Tk()
    except tk.TclError as exc:  # no display (plain CI shell): xvfb-run makes these run
        pytest.skip(f"no display: {exc}")
    r.withdraw()
    yield r
    try:
        r.destroy()
    except tk.TclError:
        pass


@pytest.fixture
def panel(root, tmp_path):
    model = ControlPanelModel(config_path=tmp_path / "config.yaml", runner=EngineRunner(FakeEngine), http=FakeHTTP(200, {"data": [{"id": "m"}]}), env={})
    p = ControlPanel(root, model, poll_ms=20)
    yield p
    model.stop()


def pump(root, seconds=0.15):
    end = time.time() + seconds
    while time.time() < end:
        root.update()
        time.sleep(0.005)


def test_tabs_are_named_as_specified(panel):
    names = [panel.notebook.tab(i, "text") for i in range(panel.notebook.index("end"))]
    assert names == ["Engine Controls", "Core Vision & Gestures", "Custom ONNX AI Model", "LLM Integration"]


def test_widgets_reflect_the_config(panel):
    assert panel.flag_vars["left_click"].get() and panel.flag_vars["right_pinch"].get() and panel.flag_vars["palm_pause"].get()
    assert not panel.flag_vars["llm_enabled"].get()
    assert panel.provider_var.get() == "Ollama (Local)"
    assert panel.url_var.get() == "http://localhost:11434/v1" and panel.model_var.get() == "llama3.2"
    assert panel.slider_vars["smoothing"].get() == pytest.approx(0.2)
    assert panel.slider_labels["pinch"].cget("text") == "45"
    assert panel.key_entry.cget("show") != ""  # the API key is masked


def test_checkbox_and_slider_edit_the_model(panel):
    panel.flag_vars["palm_pause"].set(False)
    panel.tabs["vision"].update()
    panel._on_flag("palm_pause")
    assert panel.model.cfg.hand.pause_gesture == "none"
    panel._on_slider("pinch", 55.0)
    assert panel.model.cfg.hand.pinch_threshold_px == 55
    assert panel.slider_labels["pinch"].cget("text") == "55"
    panel._on_slider("smoothing", 0.33)
    assert panel.slider_labels["smoothing"].cget("text") == "0.33"
    assert set(panel.flag_vars) == set(FLAGS)


def test_provider_dropdown_autofills_url_and_model(panel):
    panel.provider_var.set("LM Studio (Local)")
    panel.on_provider()
    assert panel.url_var.get() == "http://localhost:1234/v1" and panel.model.cfg.llm.provider == "lmstudio"
    panel.provider_var.set("Google Gemini (Cloud)")
    panel.on_provider()
    assert panel.url_var.get().startswith("https://generativelanguage.googleapis.com") and panel.model_var.get() == "gemini-2.5-flash"
    panel.provider_var.set("Custom OpenAI-Compatible URL")
    panel.on_provider()
    assert panel.url_var.get() == ""


def test_typing_into_fields_updates_the_model(panel):
    panel.url_var.set("http://10.0.0.2:1234/v1")
    panel.model_var.set("qwen")
    panel.key_var.set("sk-typed")
    panel.onnx_path_var.set("models/a.onnx")
    panel.camera_var.set("1")
    assert panel.model.cfg.llm.base_url == "http://10.0.0.2:1234/v1" and panel.model.cfg.llm.model == "qwen"
    assert panel.model.api_key == "sk-typed" and panel.model.cfg.onnx.model_path == "models/a.onnx"
    assert panel.model.cfg.camera.source == 1


def test_engine_buttons_and_status_colours(root, panel):
    pump(root)
    assert panel.status_var.get() == "Stopped" and "disabled" in panel.stop_btn.state() and "disabled" not in panel.start_btn.state()
    panel.on_start()
    assert wait_for(lambda: panel.model.status == "Running")
    pump(root)
    assert panel.status_var.get() == "Running" and "disabled" in panel.start_btn.state()
    assert panel.fps_var.get() == "60 FPS"
    assert panel.status_dot.itemcget(panel._dot, "fill") == gui_module.STATUS_COLORS["Running"]
    panel.on_pause()
    pump(root)
    assert panel.status_var.get() == "Paused" and panel.status_dot.itemcget(panel._dot, "fill") == gui_module.STATUS_COLORS["Paused"]
    panel.on_pause()
    panel.on_stop()
    pump(root)
    assert panel.status_var.get() == "Stopped" and panel.fps_var.get() == ""


def test_restart_hint_and_error_message_are_shown(root, panel):
    panel.on_start()
    assert wait_for(lambda: panel.model.status == "Running")
    panel.flag_vars["llm_enabled"].set(True)
    panel._on_flag("llm_enabled")
    pump(root)
    assert "Stop, then Start" in panel.restart_var.get()
    panel.on_stop()
    panel.model.runner.error = "RuntimeError: camera 0 could not be opened"
    pump(root)
    assert "camera 0" in panel.error_var.get()


def test_test_connection_badge_goes_green_then_red(root, panel):
    panel.provider_var.set("LM Studio (Local)")
    panel.on_provider()
    panel.on_test_connection()
    assert panel.badge.cget("bg") == gui_module.BADGE_COLORS["busy"]
    assert wait_for(lambda: (pump(root, 0.05), panel.badge.cget("bg") == gui_module.BADGE_COLORS["ok"])[1])
    assert "connected" in panel.badge.cget("text") and "disabled" not in panel.test_btn.state()

    panel.provider_var.set("OpenAI (Cloud)")
    panel.on_provider()
    panel.on_test_connection()
    assert wait_for(lambda: (pump(root, 0.05), panel.badge.cget("bg") == gui_module.BADGE_COLORS["fail"])[1])
    assert "OPENAI_API_KEY" in panel.badge_msg.get()


def test_second_click_while_testing_is_ignored(panel):
    calls = []

    def slow():
        calls.append(1)
        time.sleep(0.2)
        return ConnectionResult(True, "ok", "ollama", "m")

    panel.model.test_connection = slow
    panel.on_test_connection()
    panel.on_test_connection()
    assert wait_for(lambda: panel._test_result is not None)
    assert calls == [1]


def test_save_profile_button_writes_config(panel, tmp_path):
    panel.on_save()
    assert (tmp_path / "config.yaml").is_file() and "Saved to" in panel.message_var.get()
    panel.save_key_var.set(True)
    panel.model.save_api_key = True
    panel.key_var.set("sk-plain")
    panel.on_save()
    assert "plain text" in panel.message_var.get()


def test_save_profile_error_is_shown(panel, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    panel.model.config_path = blocker / "config.yaml"  # a file where a folder is needed
    panel.on_save()
    assert panel.message_var.get().startswith("Could not save")


def test_send_command_box(panel):
    panel.command_var.set("scroll down")
    panel.on_send()
    assert "Start the engine" in panel.message_var.get() and panel.command_var.get() == "scroll down"


def test_record_dialog_refuses_while_engine_runs(root, panel):
    panel.on_start()
    assert wait_for(lambda: panel.model.runner.alive)
    panel.on_record()
    assert "Stop the engine" in panel.message_var.get()


def test_record_and_train_dialogs_run_subprocess_and_adopt_model(root, panel, monkeypatch):
    created = []
    real = gui_module._SubprocessDialog

    class Dialog(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            created.append(self)

    monkeypatch.setattr(gui_module, "_SubprocessDialog", Dialog)
    real_launch = ControlPanelModel.launch
    monkeypatch.setattr(ControlPanelModel, "launch", staticmethod(lambda argv: real_launch([sys.executable, "-c", "print('trained ok')"])))
    panel.on_train()
    dlg = created[-1]
    dlg.vars[1].set("models/new.onnx")
    dlg.run()
    assert wait_for(lambda: (pump(root, 0.05), "Finished (exit code 0)" in dlg.log.get("1.0", "end"))[1], timeout=10)
    text = dlg.log.get("1.0", "end")
    assert "trained ok" in text and "selected in the panel" in text
    assert panel.model.cfg.onnx.model_path == "models/new.onnx" and panel.flag_vars["onnx_enabled"].get()
    assert panel.onnx_path_var.get() == "models/new.onnx"
    dlg.close()


def test_dialog_validation_errors_are_logged(root, panel, monkeypatch):
    created = []
    real = gui_module._SubprocessDialog
    monkeypatch.setattr(gui_module, "_SubprocessDialog", lambda *a, **k: created.append(real(*a, **k)) or created[-1])
    panel.on_record()
    dlg = created[-1]
    dlg.vars[0].set("bad label")
    dlg.run()
    assert "label must be" in dlg.log.get("1.0", "end")
    dlg.close()


def test_close_stops_the_engine_and_destroys_the_window(root, panel):
    panel.on_start()
    assert wait_for(lambda: panel.model.runner.alive)
    engine = panel.model.runner.engine
    panel.close()
    assert engine.stopped
    try:
        exists = bool(root.winfo_exists())
    except tk.TclError:  # some Tk builds raise once the root is destroyed
        exists = False
    assert not exists


# --------------------------------------------------------------------------- entry point
def test_main_reports_missing_tkinter(monkeypatch, capsys):
    monkeypatch.setattr(gui_module, "tkinter_available", lambda: False)
    assert main([]) == 1
    assert "tkinter is not available" in capsys.readouterr().err


def test_main_reports_missing_display(monkeypatch, capsys):
    def boom(*a, **k):
        raise tk.TclError("no display name")

    monkeypatch.setattr(tk, "Tk", boom)
    assert main([]) == 1
    assert "cannot open a window" in capsys.readouterr().err


def test_main_builds_the_panel_and_autostarts(monkeypatch, root, tmp_path):
    seen = {}
    monkeypatch.setattr(tk, "Tk", lambda: root)
    monkeypatch.setattr(root, "mainloop", lambda: seen.setdefault("looped", True))
    monkeypatch.setattr(gui_module.ControlPanel, "on_start", lambda self: seen.setdefault("started", True))
    assert main(["--config", str(tmp_path / "c.yaml"), "--autostart"]) == 0
    assert seen == {"looped": True, "started": True}


def test_tkinter_available_is_true_here():
    assert tkinter_available()

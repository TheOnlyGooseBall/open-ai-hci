"""VoiceIntentProcessor (rules / LLM / rules_then_llm) and the config-driven default pipeline."""
import json

import pytest
from conftest import FakePyAutoGUI, make_ctx, names
from test_llm import FakeHTTP, actions_json, openai_reply

from open_ai_hci import (
    HCIConfig,
    HCIEngine,
    LLMConfig,
    LLMRouter,
    MouseConfig,
    OnnxConfig,
    OSKeyboardAction,
    OSMouseAction,
    VoiceConfig,
)
from open_ai_hci.processors.ai import ONNXGestureClassifier, VoiceIntentProcessor


def build(mode, replies=(), *, with_router=True, pag=None):
    cfg = HCIConfig(mouse=MouseConfig(enabled=False), llm=LLMConfig(provider="lmstudio"))
    cfg.screen.width, cfg.screen.height = 1920, 1080
    http = FakeHTTP(*replies) if replies else FakeHTTP(openai_reply(actions_json()))
    procs = []
    router = None
    if with_router:
        router = LLMRouter(http=http, env={})
        procs.append(router)
    voice = VoiceIntentProcessor(mode, voice=False)
    procs.append(voice)
    actions = [OSMouseAction(pyautogui_module=pag), OSKeyboardAction(pyautogui_module=pag)] if pag else []
    engine = HCIEngine(cfg, sensors=[], processors=procs, actions=actions, raise_errors=True)
    engine.setup()
    events = []
    engine.on("*", events.append)
    return engine, voice, router, http, events


def frame(engine):
    engine.process_frame(make_ctx().frame)


def test_rules_mode_uses_the_phrase_table_and_ignores_everything_else():
    pag = FakePyAutoGUI()
    engine, voice, router, http, events = build("rules", pag=pag)
    voice.submit_text("new tab")
    voice.submit_text("what is the meaning of life")
    frame(engine)
    assert pag.calls == [("hotkey", ("ctrl", "t"), {})]
    assert http.requests == []  # the LLM is never contacted in rules mode
    assert names(events).count("voice_command") == 2
    assert any(e.name == "alert" and "no command" in e.text for e in events)
    engine.close()


def test_llm_mode_sends_everything_to_the_model_even_known_phrases():
    pag = FakePyAutoGUI()
    engine, voice, router, http, events = build(
        "llm", [openai_reply(actions_json({"action": "hotkey", "keys": ["ctrl", "t"]}))], pag=pag
    )
    voice.submit_text("new tab")  # a rules phrase, but llm mode does not use the table
    frame(engine)
    assert router.wait_idle(5)
    assert len(http.requests) == 1 and "new tab" in http.requests[0]["json"]["messages"][1]["content"]
    frame(engine)
    assert pag.calls == [("hotkey", ("ctrl", "t"), {})]
    assert any(e.name == "intent" and e.origin == "llm" for e in events)
    engine.close()


def test_rules_then_llm_only_asks_the_model_when_the_rules_do_not_match():
    pag = FakePyAutoGUI()
    engine, voice, router, http, events = build("rules_then_llm", [openai_reply(actions_json({"action": "scroll", "amount": -300}))], pag=pag)
    voice.submit_text("click")  # rules
    frame(engine)
    assert http.requests == [] and pag.names() == ["click"]
    voice.submit_text("scroll down slowly please")  # not in the table -> LLM
    frame(engine)
    assert router.wait_idle(5)
    frame(engine)
    assert len(http.requests) == 1 and pag.calls[-1] == ("scroll", (-3,), {})
    assert any(e.name == "alert" and e.text.startswith("thinking") for e in events)
    engine.close()


def test_text_input_works_without_a_microphone_and_tags_the_origin():
    engine, voice, router, http, events = build("rules")
    voice.submit_text("copy that", source="text")
    frame(engine)
    intent = next(e for e in events if e.name == "intent")
    assert intent.intent == "hotkey" and intent.args == {"keys": "ctrl+c"} and intent.origin == "text"
    engine.close()


def test_pause_listening_stops_both_rules_and_llm():
    pag = FakePyAutoGUI()
    engine, voice, router, http, events = build("rules_then_llm", pag=pag)
    voice.submit_text("stop listening")
    voice.submit_text("click")
    voice.submit_text("anything for the llm")
    frame(engine)
    assert pag.calls == [] and http.requests == []
    voice.submit_text("start listening")
    voice.submit_text("click")
    frame(engine)
    assert pag.names() == ["click"]
    engine.close()


def test_llm_modes_degrade_to_rules_without_a_router(caplog):
    pag = FakePyAutoGUI()
    with caplog.at_level("WARNING"):
        engine, voice, _, _, events = build("llm", with_router=False, pag=pag)
    assert "needs an LLMRouter" in caplog.text
    voice.submit_text("click")  # no router: the rules still work
    voice.submit_text("do something clever")
    frame(engine)
    assert pag.names() == ["click"] and any("no command" in e.text for e in events if e.name == "alert")
    engine.close()


def test_a_down_llm_surfaces_as_llm_error_and_never_crashes():
    engine, voice, router, http, events = build("llm", [(500, {"error": "down"})])
    voice.submit_text("do something")
    frame(engine)
    assert router.wait_idle(5)
    frame(engine)
    assert "llm_error" in names(events)
    engine.close()


def test_custom_phrase_table_and_invalid_mode():
    voice = VoiceIntentProcessor("rules", voice=False, phrases=[(r"^zap$", "click", {})])
    assert voice.intent_parser("zap").name == "click" and voice.intent_parser("click") is None
    with pytest.raises(ValueError, match="mode must be"):
        VoiceIntentProcessor("telepathy")


def test_mode_defaults_to_the_voice_config(bare_engine):
    bare_engine.config.voice.mode = "llm"
    voice = bare_engine.add_processor(VoiceIntentProcessor(voice=False))
    assert voice.mode == "llm"
    assert VoiceIntentProcessor("rules", voice=False).mode == "rules"


def test_missing_voice_dependencies_degrade_with_a_hint(monkeypatch, caplog, bare_engine):
    import open_ai_hci.processors.ai.multimodal_agent as ma
    from open_ai_hci._optional import MissingOptionalDependency

    def boom(self):
        raise MissingOptionalDependency("voice needs faster_whisper")

    monkeypatch.setattr(ma.VoiceListener, "start", boom)
    voice = bare_engine.add_processor(VoiceIntentProcessor("rules"))
    with caplog.at_level("WARNING"):
        voice.setup()
    assert "Voice commands disabled" in caplog.text and voice._voice is None
    voice.submit_text("click")  # typing commands still work


def test_microphone_listener_is_built_from_voice_config(bare_engine, monkeypatch):
    import open_ai_hci.processors.ai.multimodal_agent as ma

    seen = {}
    monkeypatch.setattr(ma.VoiceListener, "start", lambda self: seen.update(model=self.model_size, lang=self.language, th=self.energy_threshold))
    bare_engine.config.voice = VoiceConfig(enabled=True, model_size="base.en", language=None, energy_threshold=0.03)
    voice = bare_engine.add_processor(VoiceIntentProcessor("rules"))
    voice.setup()
    assert seen == {"model": "base.en", "lang": None, "th": 0.03}
    assert voice._voice.on_text == voice.submit_text


# ------------------------------------------------------------------ default pipeline from config
def test_default_pipeline_adds_ai_layers_from_the_config(tmp_path):
    base = HCIEngine(HCIConfig())
    assert [p.name for p in base.processors] == ["smoothing", "gesture_rules"] and [a.name for a in base.actions] == ["os_mouse"]

    cfg = HCIConfig()
    cfg.onnx = OnnxConfig(enabled=True, model_path=str(tmp_path / "m.onnx"), min_confidence=0.8, stable_frames=5, background_label="idle")
    cfg.llm.enabled = True
    cfg.voice.enabled = True
    engine = HCIEngine(cfg)
    assert [p.name for p in engine.processors] == ["smoothing", "gesture_rules", "onnx_classifier", "llm_router", "voice_intent"]
    assert [a.name for a in engine.actions] == ["os_mouse", "os_keyboard"]
    clf = engine.get(ONNXGestureClassifier)
    assert (clf.min_confidence, clf.stable_frames, clf.background_label) == (0.8, 5, "idle")

    no_model = HCIConfig()
    no_model.onnx.enabled = True  # enabled but no path yet: skipped rather than crashing
    assert "onnx_classifier" not in [p.name for p in HCIEngine(no_model).processors]


def test_apply_config_updates_the_onnx_classifier_live(tmp_path):
    cfg = HCIConfig()
    cfg.onnx = OnnxConfig(enabled=True, model_path=str(tmp_path / "m.onnx"))
    engine = HCIEngine(cfg)
    new = HCIConfig.from_dict(cfg.to_dict())
    new.onnx.min_confidence, new.onnx.stable_frames = 0.9, 7
    engine.apply_config(new)
    clf = engine.get(ONNXGestureClassifier)
    assert (clf.min_confidence, clf.stable_frames) == (0.9, 7)


def test_a_gesture_from_an_onnx_model_triggers_a_right_click_end_to_end():
    pag = FakePyAutoGUI()
    cfg = HCIConfig()
    cfg.screen.width, cfg.screen.height = 1920, 1080
    engine = HCIEngine(cfg, sensors=[], actions=[OSMouseAction(pyautogui_module=pag)], raise_errors=True)
    engine.setup()
    engine.emit("gesture_right_click", gesture="right_click", confidence=0.95, previous=None, handedness="Right")
    assert pag.calls == [("click", (), {"button": "right"})]
    engine.close()


def test_engine_pause_api_and_toggle():
    engine = HCIEngine(HCIConfig(mouse=MouseConfig(enabled=False)), sensors=[], raise_errors=True)
    events = []
    engine.on("*", events.append)
    assert engine.paused is False
    engine.pause()
    assert engine.paused is True
    assert engine.toggle_pause() is False
    assert [e.name for e in events if e.name.startswith("control")] == ["control_paused", "control_resumed"]
    bare = HCIEngine(HCIConfig(), sensors=[], processors=[], actions=[])
    bare.pause()  # no rules processor: harmless
    assert bare.paused is False


def test_paused_engine_stops_cursor_events_but_keeps_processing():
    from conftest import FakeHandBackend

    from open_ai_hci import HandSensor

    pag = FakePyAutoGUI()
    cfg = HCIConfig()
    cfg.screen.width, cfg.screen.height = 1920, 1080
    script = [((640, 360), (640, 460))] * 4
    engine = HCIEngine(cfg, sensors=[HandSensor(backend=FakeHandBackend(script))], actions=[OSMouseAction(pyautogui_module=pag)], raise_errors=True)
    blank = make_ctx().frame
    engine.process_frame(blank)
    assert pag.names() == ["moveTo"]
    engine.pause()
    engine.process_frame(blank)
    engine.process_frame(blank)
    assert pag.names() == ["moveTo"]  # nothing while paused
    engine.resume()
    engine.process_frame(blank)
    assert pag.names() == ["moveTo", "moveTo"]
    engine.close()


def test_apply_config_swaps_settings_and_notifies_components():
    pag = FakePyAutoGUI()
    cfg = HCIConfig()
    cfg.screen.width, cfg.screen.height = 1920, 1080
    engine = HCIEngine(cfg, sensors=[], actions=[OSMouseAction(pyautogui_module=pag)], raise_errors=True)
    engine.setup()
    new = HCIConfig.from_dict(cfg.to_dict())
    new.mouse.right_click_on_pinch = False
    new.smoothing.alpha = 0.5
    new.hand.pinch_threshold_px = 33
    engine.apply_config(new)
    assert engine.config is new and engine.get(OSMouseAction).cfg.right_click_on_pinch is False
    engine.emit("right_click", x=1, y=1, via="x")
    assert pag.calls == []
    bad = HCIConfig()
    bad.hand.pause_gesture = "wave"
    with pytest.raises(ValueError, match="pause_gesture"):
        engine.apply_config(bad)
    assert engine.config is new  # a rejected config leaves the running one untouched
    engine.close()
    json.dumps(new.to_dict())  # (and the new sections serialise)

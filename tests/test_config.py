import json

import pytest

from open_ai_hci import PROFILES, HCIConfig
from open_ai_hci._optional import is_available


def test_defaults_are_the_prototype_values():
    c = HCIConfig()
    assert (c.camera.width, c.camera.height, c.camera.frame_skip, c.camera.mirror) == (1280, 720, 2, True)
    assert (c.smoothing.kind, c.smoothing.alpha) == ("ema", 0.20)
    assert c.hand.max_hands == 1 and c.hand.pinch_threshold_px == 45
    assert (c.hand.detection_confidence, c.hand.tracking_confidence) == (0.70, 0.70)
    assert (c.face.flick_threshold_px, c.face.flick_axis_ratio, c.face.flick_cooldown_s) == (28, 3, 1.2)
    assert c.face.detection_confidence == 0.70 and c.face.max_faces == 1
    assert c.mouse.failsafe is True and c.mouse.pause == 0.0
    assert c.hud.window_name == "Natural Camera Frame Active"
    assert c.hud.title == "System: Hand Tracker & Clicks Active"
    c.validate()


def test_profiles_apply_on_top_of_defaults():
    precise = HCIConfig.from_profile("precise")
    assert precise.smoothing.kind == "one_euro" and precise.camera.frame_skip == 1
    assert precise.hand.pinch_threshold_px == 45  # untouched
    acc = HCIConfig.from_profile("accessibility")
    assert acc.hand.pinch_threshold_px == 60 and acc.hand.right_pinch_threshold_px == 60 and acc.hand.pause_hold_s == 1.5
    for name in PROFILES:
        HCIConfig.from_profile(name).validate()
    with pytest.raises(ValueError, match="Unknown profile"):
        HCIConfig.from_profile("nope")


def test_profiles_do_not_leak_between_instances():
    a = HCIConfig.from_profile("fast")
    a.camera.width = 1
    assert PROFILES["fast"]["camera"]["width"] == 640
    assert HCIConfig.from_profile("fast").camera.width == 640


def test_from_dict_partial_and_unknown_keys(caplog):
    c = HCIConfig.from_dict({"hand": {"pinch_threshold_px": 50}, "bogus": 1, "face": {"nope": 2}})
    assert c.hand.pinch_threshold_px == 50 and c.hand.max_hands == 1
    assert "bogus" in caplog.text and "nope" in caplog.text
    with pytest.raises(TypeError):
        HCIConfig.from_dict({"hand": 5})


def test_json_round_trip(tmp_path):
    c = HCIConfig.from_profile("accessibility")
    c.camera.source = "video.mp4"
    path = c.save(tmp_path / "sub" / "profile.json")
    assert json.loads(path.read_text())["hand"]["pinch_threshold_px"] == 60
    assert HCIConfig.load(path) == c


@pytest.mark.skipif(not is_available("yaml"), reason="PyYAML not installed")
def test_yaml_round_trip_and_partial_file(tmp_path):
    c = HCIConfig.from_profile("fast")
    assert HCIConfig.load(c.save(tmp_path / "p.yaml")) == c
    (tmp_path / "partial.yml").write_text("hand:\n  pinch_threshold_px: 52\n")
    assert HCIConfig.load(tmp_path / "partial.yml").hand.pinch_threshold_px == 52


def test_yaml_round_trip_without_pyyaml(tmp_path, monkeypatch):
    import open_ai_hci._yaml as ymod

    monkeypatch.setattr(ymod, "is_available", lambda name: False)  # force the built-in reader / writer
    cfg = HCIConfig()
    cfg.llm.user_context = {"email_signature": "Best, Sam # not a comment: really"}
    cfg.llm.api_key = "sk-test#1"
    cfg.hud.quit_keys = ["q", "x"]
    cfg.camera.source = "rtsp://cam/stream"
    path = cfg.save(tmp_path / "config.yaml")
    text = path.read_text(encoding="utf-8")
    assert "hand:" in text and "pinch_threshold_px: 45.0" in text and "quit_keys: [\"q\", \"x\"]" in text
    assert HCIConfig.load(path).to_dict() == cfg.to_dict()


def test_builtin_yaml_reads_comments_quotes_and_partial_files(tmp_path, monkeypatch):
    import open_ai_hci._yaml as ymod

    monkeypatch.setattr(ymod, "is_available", lambda name: False)
    (tmp_path / "c.yaml").write_text(
        "# my profile\nhand:\n  pinch_threshold_px: 52  # looser\n  pause_gesture: 'fist'\nllm:\n  enabled: yes\n"
        "  model: \"qwen2.5\"\n  allowed_actions: [\"click\", 'hotkey']\nscreen: {}\n",
        encoding="utf-8",
    )
    cfg = HCIConfig.load(tmp_path / "c.yaml")
    assert cfg.hand.pinch_threshold_px == 52 and cfg.hand.pause_gesture == "fist"
    assert cfg.llm.enabled is True and cfg.llm.model == "qwen2.5" and cfg.llm.allowed_actions == ["click", "hotkey"]


@pytest.mark.parametrize(
    "text,match",
    [("- a\n- b\n", "block lists"), ("just words\n", "key: value"), ("a:\n\tb: 1\n", "tabs"), ("a: [1, 2\n", "flow value")],
)
def test_builtin_yaml_rejects_unsupported_syntax(monkeypatch, text, match):
    import open_ai_hci._yaml as ymod

    monkeypatch.setattr(ymod, "is_available", lambda name: False)
    with pytest.raises(ValueError, match=match):
        ymod.load_yaml(text)


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda c: setattr(c.camera, "frame_skip", 0), "frame_skip"),
        (lambda c: setattr(c.hand, "pinch_threshold_px", -1), "pinch_threshold_px"),
        (lambda c: setattr(c.hand, "detection_confidence", 1.5), "detection_confidence"),
        (lambda c: setattr(c.smoothing, "kind", "magic"), "smoothing.kind"),
        (lambda c: setattr(c.smoothing, "alpha", 0), "alpha"),
        (lambda c: setattr(c.mediapipe, "backend", "x"), "backend"),
        (lambda c: setattr(c.screen, "width", 100), "screen.width"),
    ],
)
def test_validation_rejects_bad_values(mutate, fragment):
    c = HCIConfig()
    mutate(c)
    with pytest.raises(ValueError, match=fragment):
        c.validate()


def test_with_overrides_returns_a_validated_copy():
    base = HCIConfig()
    c = base.with_overrides(**{"hand.pinch_threshold_px": 50, "smoothing.kind": "kalman"})
    assert c.hand.pinch_threshold_px == 50 and base.hand.pinch_threshold_px == 45
    with pytest.raises(AttributeError):
        base.with_overrides(**{"hand.nonexistent": 1})
    with pytest.raises(ValueError):
        base.with_overrides(**{"camera.frame_skip": 0})

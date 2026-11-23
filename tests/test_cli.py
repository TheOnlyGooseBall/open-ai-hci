import pytest

from open_ai_hci import HCIConfig, HCIEngine
from open_ai_hci.__main__ import main


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0 and "open-ai-hci" in capsys.readouterr().out


def test_doctor_reports_core_and_optional_dependencies(capsys):
    code = main(["doctor"])
    out = capsys.readouterr().out
    assert code in (0, 1)  # 1 only when e.g. there is no display on this machine
    for needle in ("opencv-python", "numpy", "mediapipe", "Optional extras", "onnxruntime", "websockets"):
        assert needle in out


def test_run_passes_profile_camera_and_flags_to_the_engine(monkeypatch):
    seen = {}

    def fake_run(self, show_hud=None, **kw):
        seen.update(show_hud=show_hud, cfg=self.config)

    monkeypatch.setattr(HCIEngine, "run", fake_run)
    assert main(["--profile", "precise", "--camera", "2", "--no-mouse", "--no-hud"]) == 0
    cfg = seen["cfg"]
    assert seen["show_hud"] is False and cfg.smoothing.kind == "one_euro"
    assert cfg.camera.source == 2 and cfg.mouse.enabled is False
    assert main(["--camera", "clip.mp4"]) == 0 and seen["cfg"].camera.source == "clip.mp4" and seen["show_hud"] is True


def test_config_file_is_loaded(monkeypatch, tmp_path):
    path = HCIConfig.from_profile("accessibility").save(tmp_path / "p.json")
    seen = {}
    monkeypatch.setattr(HCIEngine, "run", lambda self, show_hud=None, **kw: seen.setdefault("cfg", self.config))
    assert main(["--config", str(path)]) == 0
    assert seen["cfg"].hand.pinch_threshold_px == 60


def test_setup_problems_become_a_friendly_error_not_a_traceback(monkeypatch, capsys):
    def boom(self, show_hud=None, **kw):
        raise RuntimeError("Could not open camera/video source 0.")

    monkeypatch.setattr(HCIEngine, "run", boom)
    assert main(["--no-hud"]) == 1
    err = capsys.readouterr().err
    assert "error: Could not open camera" in err and "doctor" in err

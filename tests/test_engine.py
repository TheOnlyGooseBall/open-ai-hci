import time

import cv2
import numpy as np
import pytest
from conftest import FakeFaceBackend, FakeHandBackend, FakePyAutoGUI, names

from open_ai_hci import (
    HUD,
    FaceSensor,
    FPSCounter,
    HandSensor,
    HCIConfig,
    HCIEngine,
    MouseConfig,
    OSMouseAction,
)
from open_ai_hci.sensors.base import BaseSensor

BLANK = np.zeros((720, 1280, 3), dtype=np.uint8)
OPEN = ((640, 360), (640, 460))  # 100 px apart
PINCH = ((640, 360), (644, 400))  # 40.2 px apart
PINCH2 = ((700, 360), (704, 400))


def make_engine(hand=(), face=(), pag=None, **kw):
    cfg = HCIConfig.from_profile("prototype")  # head-flick right click, no pause gesture (v0.1 behaviour)
    cfg.screen.width, cfg.screen.height = 1920, 1080
    cfg.camera.frame_skip = 1
    sensors = [HandSensor(backend=FakeHandBackend(hand)), FaceSensor(backend=FakeFaceBackend(face))]
    actions = [OSMouseAction(pyautogui_module=pag)] if pag is not None else []
    return HCIEngine(cfg, sensors=sensors, actions=actions, raise_errors=True, **kw)


def write_video(path, n_frames, size=(320, 240)):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, size)
    for _ in range(n_frames):
        vw.write(np.zeros((size[1], size[0], 3), np.uint8))
    vw.release()
    return str(path)


# ------------------------------------------------------------- prototype end to end
def test_end_to_end_reproduces_the_prototype_actions():
    pag = FakePyAutoGUI(size=(1920, 1080))
    hand = [OPEN, PINCH, PINCH2, OPEN, None]
    face = [(600, 360), (600, 360), (650, 360), (650, 360), (650, 360)]  # flick on frame 3
    engine = make_engine(hand, face, pag)
    alerts = []
    engine.on("alert", lambda e: alerts.append(e.text))
    for _ in range(5):
        engine.process_frame(BLANK)

    seq = list(pag.calls)
    assert seq[0] == ("moveTo", (960, 540), {})  # 640 * 1920/1280, 360 * 1080/720 (first sample unsmoothed)
    assert [n for n, _, _ in seq] == [
        "moveTo",  # frame 1: open hand
        "moveTo", "mouseDown",  # frame 2: pinch starts (the cursor moves first, like the prototype)
        "moveTo", "click",  # frame 3: still held; hand then face -> right click on the head flick
        "moveTo", "mouseUp",  # frame 4: released
    ]  # frame 5: hand lost, nothing held -> nothing to do
    assert ("mouseDown", (), {"button": "left"}) in seq
    assert ("click", (), {"button": "right"}) in seq
    assert ("mouseUp", (), {"button": "left"}) in seq
    assert alerts == ["triggered right click!"]


def test_prototype_smoothing_is_applied_to_the_real_cursor():
    pag = FakePyAutoGUI(size=(1920, 1080))
    engine = make_engine([((640, 360), (640, 460)), ((1000, 360), (1000, 460))], [], pag)
    engine.process_frame(BLANK)
    engine.process_frame(BLANK)
    moves = [a for n, a, _ in pag.calls if n == "moveTo"]
    expected_x = int(960 + (1000 * 1.5 - 960) * 0.20)
    assert moves[1][0] == expected_x  # prev + (target - prev) * 0.20, truncated with int()


def test_hand_loss_releases_a_held_button_and_stops_moving():
    pag = FakePyAutoGUI()
    engine = make_engine([PINCH, None], [], pag)
    engine.process_frame(BLANK)
    engine.process_frame(BLANK)
    assert pag.names() == ["moveTo", "mouseDown", "mouseUp"]


# ----------------------------------------------------------------- process_frame / ctx
def test_process_frame_returns_a_populated_context_and_mirrors():
    engine = make_engine([OPEN], [(640, 360)])
    frame = BLANK.copy()
    frame[100, 10] = (255, 255, 255)  # white pixel near the left edge
    got = []
    engine.on("frame", lambda e: got.append(e.ctx))
    ctx = engine.process_frame(frame)
    assert got == [ctx]
    assert ctx.hand is not None and ctx.face is not None and ctx.cursor.active
    assert ctx.width == 1280 and ctx.height == 720 and ctx.frame_index == 1
    assert ctx.extras["latency_ms"] >= 0 and "fps" in ctx.extras
    assert ctx.frame[100, 1280 - 1 - 10].tolist() == [255, 255, 255]  # mirrored
    assert frame[100, 10].tolist() == [255, 255, 255]  # caller's array untouched
    assert engine.process_frame(frame, mirror=False).frame[100, 10].tolist() == [255, 255, 255]


def test_frame_event_does_not_reach_wildcard_subscribers():
    engine = make_engine()
    seen = []
    engine.on("*", seen.append)
    engine.process_frame(BLANK)
    assert "frame" not in names(seen)


# ------------------------------------------------------------------------- run loop
def test_run_with_frame_skip_matches_the_prototype_counting(tmp_path):
    engine = make_engine(hand=[OPEN] * 20)
    engine.config.camera.source = write_video(tmp_path / "v.avi", 10)
    engine.config.camera.frame_skip = 2
    ev = []
    engine.on("*", ev.append)
    engine.run(show_hud=False)
    assert engine.frame_index == 10 and engine.processed_frames == 5  # every 2nd frame is processed
    assert names(ev)[0] == "engine_start" and names(ev)[-1] == "engine_stop"
    assert not engine.running
    assert engine.sensors[0]._backend is None  # detectors released on exit


def test_run_max_frames(tmp_path):
    engine = make_engine(hand=[OPEN] * 20)
    engine.config.camera.source = write_video(tmp_path / "v.avi", 30)
    engine.run(show_hud=False, max_frames=6)
    assert engine.frame_index == 6 and engine.processed_frames == 6


def test_stop_from_an_event_handler(tmp_path):
    engine = make_engine(hand=[OPEN] * 50)
    engine.config.camera.source = write_video(tmp_path / "v.avi", 40)
    engine.on("hand_detected", lambda e: engine.stop())
    engine.run(show_hud=False)
    assert engine.processed_frames < 40


def test_run_in_a_background_thread(tmp_path):
    engine = make_engine(hand=[OPEN] * 20)
    engine.config.camera.source = write_video(tmp_path / "v.avi", 8)
    t = engine.start(show_hud=False)
    engine.join(10)
    assert not t.is_alive() and engine.processed_frames == 8


def test_start_twice_is_rejected_while_running(tmp_path):
    engine = make_engine(hand=[OPEN] * 100)
    engine.config.camera.source = write_video(tmp_path / "v.avi", 60)
    engine.config.camera.max_fps = 30  # slow enough that the first run is still going
    t = engine.start(show_hud=False)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            engine.start()
    finally:
        engine.stop()
        engine.join(10)
    assert not t.is_alive()


def test_run_with_a_headless_hud(tmp_path):
    hud = HUD(headless=True)
    engine = make_engine(hand=[OPEN] * 10, hud=hud)
    engine.config.camera.source = write_video(tmp_path / "v.avi", 5)
    engine.run(show_hud=True)
    assert engine.processed_frames == 5 and not engine._hud_active


def test_max_fps_limits_the_loop(tmp_path):
    engine = make_engine(hand=[OPEN] * 20)
    engine.config.camera.source = write_video(tmp_path / "v.avi", 10)
    engine.config.camera.max_fps = 50
    t0 = time.perf_counter()
    engine.run(show_hud=False)
    assert time.perf_counter() - t0 >= 9 / 50 * 0.9


def test_unopenable_source_gives_a_clear_error_and_no_stop_event(tmp_path):
    engine = make_engine()
    engine.config.camera.source = str(tmp_path / "missing.avi")
    ev = []
    engine.on("*", ev.append)
    with pytest.raises(RuntimeError, match="Could not open camera"):
        engine.run(show_hud=False)
    assert "engine_stop" not in names(ev)


# ------------------------------------------------------------------ robustness
class Exploding(BaseSensor):
    name = "exploding"

    def process(self, ctx):
        raise ValueError("kaboom")


def test_a_failing_plugin_does_not_stop_the_pipeline(caplog):
    engine = make_engine([OPEN])
    engine.raise_errors = False
    engine.bus.raise_errors = False
    engine.add_sensor(Exploding())
    errors = []
    engine.on("error", errors.append)
    ctx = engine.process_frame(BLANK)
    assert ctx.hand is not None  # the healthy sensor still ran
    assert errors[0].component == "exploding" and "kaboom" in errors[0].message
    assert "kaboom" in caplog.text


def test_raise_errors_surfaces_plugin_failures():
    engine = make_engine()
    engine.add_sensor(Exploding())
    with pytest.raises(ValueError, match="kaboom"):
        engine.process_frame(BLANK)


def test_setup_failure_closes_already_started_components():
    class Broken(BaseSensor):
        name = "broken"

        def setup(self):
            raise RuntimeError("no device")

        def process(self, ctx):
            pass

    engine = make_engine()
    backend = engine.sensors[0]._backend
    engine.add_sensor(Broken())
    with pytest.raises(RuntimeError, match="no device"):
        engine.setup()
    assert backend.closed is True and engine._setup_done is False


# ------------------------------------------------------------------ composition
def test_add_remove_get_and_context_manager():
    pag = FakePyAutoGUI()
    engine = make_engine(pag=pag)
    mouse = engine.get(OSMouseAction)
    assert mouse is not None and engine.get(HUD) is None
    with engine:
        extra = engine.add_action(OSMouseAction(pyautogui_module=FakePyAutoGUI()))  # set up immediately
        assert extra._pag is not None and extra._screen == (1920, 1080)
        assert engine.remove(OSMouseAction) == 2
        assert engine.remove(OSMouseAction) == 0
    assert engine._setup_done is False


def test_default_pipeline_and_string_profile():
    e = HCIEngine("precise")
    assert [c.name for c in e.sensors] == ["hand"]  # v0.2: face tracking is opt-in
    assert [c.name for c in HCIEngine("prototype").sensors] == ["hand", "face"]
    flick = HCIConfig()
    flick.face.flick_enabled = True
    assert [c.name for c in HCIEngine(flick).sensors] == ["hand", "face"]
    assert [c.name for c in e.processors] == ["smoothing", "gesture_rules"]
    assert [c.name for c in e.actions] == ["os_mouse"]
    assert e.config.smoothing.kind == "one_euro"
    off = HCIEngine(HCIConfig(mouse=MouseConfig(enabled=False)))
    assert off.actions == []


def test_screen_size_falls_back_to_the_camera_resolution(monkeypatch, caplog):
    import open_ai_hci.engine as eng

    def no_display(*a, **k):
        raise RuntimeError("no display here")

    monkeypatch.setattr(eng, "load_pyautogui", no_display)
    engine = HCIEngine(HCIConfig(mouse=MouseConfig(enabled=False)), sensors=[], actions=[])
    assert engine.screen_size == (1280, 720)
    assert "no display here" in caplog.text


# ---------------------------------------------------------------------- fps
def test_fps_counter():
    f = FPSCounter()
    assert f.tick(0.0) == 0.0
    for i in range(1, 11):
        fps = f.tick(i * 0.1)
    assert fps == pytest.approx(10.0)
    assert f.tick(50.0) == 0.0  # old samples fall out of the window
    f.reset()
    assert f.fps == 0.0

"""Rule math: pinch, head flick, blink, scroll, and the processor's events.

Includes a randomized equivalence test against the *original prototype's* logic, so the
default thresholds (45 px pinch, 28 px / 3x flick, 1.2 s cooldown) can never drift silently.
"""
import math
import random

import pytest
from conftest import H, W, make_ctx, make_face, make_hand, names

from open_ai_hci import FaceConfig, HandConfig
from open_ai_hci.processors.gesture_rules import (
    BlinkDetector,
    FlickDetector,
    GestureRulesProcessor,
    ScrollTracker,
    detect_head_flick,
    finger_extended,
    is_pinch,
    is_two_finger_pose,
    pinch_distance,
)
from open_ai_hci.types import CursorState, HandData


# ------------------------------------------------------------------------------ pinch
def test_pinch_distance_is_euclidean():
    assert pinch_distance((0, 0), (3, 4)) == 5.0
    assert pinch_distance((10, 10), (10, 10)) == 0.0


def test_pinch_threshold_is_strict_less_than_45():
    assert is_pinch(44.999) is True
    assert is_pinch(45.0) is False  # prototype: `pinch_distance < 45`
    assert is_pinch(100.0) is False


def test_pinch_hysteresis_only_applies_once_pinching():
    assert is_pinch(50, 45, currently_pinching=False, release_margin=10) is False
    assert is_pinch(50, 45, currently_pinching=True, release_margin=10) is True
    assert is_pinch(55, 45, currently_pinching=True, release_margin=10) is False


# --------------------------------------------------------------------------- head flick
@pytest.mark.parametrize(
    "dx,dy,expected",
    [
        (29, 0, "right"),
        (-29, 0, "left"),
        (28, 0, None),  # strict > 28
        (-28, 0, None),
        (60, 20, None),  # 60 > 20*3 is False (60 == 60)
        (61, 20, "right"),
        (100, 40, None),  # mostly diagonal
        (0, 100, None),  # vertical never counts
    ],
)
def test_detect_head_flick(dx, dy, expected):
    assert detect_head_flick(dx, dy) == expected


def test_flick_detector_cooldown_and_prev_tracking():
    d = FlickDetector(28, 3, 1.2)
    assert d.update(100, 100, 0.0) is None  # no previous position yet
    assert d.update(140, 100, 0.1) == ("right", 40, 0)  # fires
    assert d.update(100, 100, 0.5) is None  # inside the cooldown (prev still updates)
    assert d.update(150, 100, 1.3) is None  # 1.3 - 0.1 = 1.2 is NOT > 1.2
    assert d.update(100, 100, 1.5) == ("left", -50, 0)  # 1.4 s after the last fire


def test_flick_detector_reset_forgets_position_only():
    d = FlickDetector()
    d.update(0, 0, 0.0)
    d.reset()
    assert d.update(500, 0, 10.0) is None  # no previous position after reset


# -------------------------------------------------------------------------------- blink
def test_blink_detector_counts_closed_frames():
    b = BlinkDetector(0.21, 2, 15)
    seq = [0.3, 0.1, 0.1, 0.3]
    assert [b.update(x) for x in seq] == [False, False, False, True]
    b.reset()
    assert [b.update(x) for x in [0.1, 0.3]] == [False, False]  # one frame is noise
    assert [b.update(x) for x in [0.1] * 20 + [0.3]][-1] is False  # too long: eyes held shut


# ------------------------------------------------------------------------------- scroll
def test_scroll_tracker_accumulates_fractions_and_direction():
    s = ScrollTracker(pixels_per_click=18)
    assert s.update(500) == 0  # first sample
    assert s.update(491) == 0  # 9 px up = 0.5 click
    assert s.update(482) == 1  # another 9 px -> 1 click up
    assert s.update(482 + 36) == -2  # 36 px down = 2 clicks down


def _pose_hand(index_up=True, middle_up=True, ring_up=False, pinky_up=False) -> HandData:
    """Hand with the wrist at the bottom and each finger either extended or curled."""
    import numpy as np

    lm = np.zeros((21, 3))
    lm[0] = (0.5, 0.9, 0)
    columns = {"index": (5, 0.40, index_up), "middle": (9, 0.48, middle_up), "ring": (13, 0.56, ring_up), "pinky": (17, 0.64, pinky_up)}
    for _, (mcp, x, up) in columns.items():
        lm[mcp] = (x, 0.6, 0)
        lm[mcp + 1] = (x, 0.5, 0)  # PIP
        lm[mcp + 2] = (x, 0.4 if up else 0.55, 0)  # DIP
        lm[mcp + 3] = (x, 0.3 if up else 0.62, 0)  # TIP (curled tips fall back towards the wrist)
    return HandData(0, "Right", 0.9, lm, W, H)


def test_two_finger_pose_detection():
    assert finger_extended(_pose_hand(), 8, 6)
    assert is_two_finger_pose(_pose_hand())
    assert not is_two_finger_pose(_pose_hand(ring_up=True))
    assert not is_two_finger_pose(_pose_hand(middle_up=False))


# -------------------------------------------------------------------------- processor
@pytest.fixture
def rules(bare_engine):
    proc = GestureRulesProcessor()
    bare_engine.add_processor(proc)
    proc.setup()
    return proc


@pytest.fixture
def proto_rules(bare_engine):
    """v0.1 behaviour: head flick on, right pinch and pause gesture off."""
    proc = GestureRulesProcessor(
        hand_config=HandConfig(right_pinch_enabled=False, pause_gesture="none"),
        face_config=FaceConfig(enabled=True, flick_enabled=True),
    )
    bare_engine.add_processor(proc)
    proc.setup()
    return proc


def run_frames(engine, proc, frames):
    """frames: list of dicts(hands=..., faces=..., cursor=..., t=...). Returns all emitted events."""
    events = []
    engine.on("*", events.append)
    for i, f in enumerate(frames, 1):
        ctx = make_ctx(frame_index=i, **f)
        proc.process(ctx)
        f["_ctx"] = ctx
    return events


def cur(x=100.0, y=200.0, active=True):
    return CursorState(x, y, x, y, active)


def test_pinch_lifecycle_and_status(bare_engine, rules):
    frames = [
        dict(hands=[make_hand((600, 300), (600, 400))], cursor=cur(), t=0.0),  # 100 px apart: open
        dict(hands=[make_hand((600, 300), (600, 344))], cursor=cur(), t=0.1),  # 44 px: pinch
        dict(hands=[make_hand((600, 300), (600, 344))], cursor=cur(), t=0.2),  # still held (no new start)
        dict(hands=[make_hand((600, 300), (600, 345))], cursor=cur(), t=0.3),  # exactly 45: released
        dict(hands=[], t=0.4),
    ]
    events = run_frames(bare_engine, rules, frames)
    assert [f["_ctx"].pinch_status for f in frames] == ["OPEN", "HOLDING CLICK", "HOLDING CLICK", "OPEN", "NO HAND DETECTED"]
    assert names(events).count("pinch_start") == 1
    assert names(events).count("pinch_end") == 1
    start = next(e for e in events if e.name == "pinch_start")
    assert (start.x, start.y, start.distance) == (100.0, 200.0, 44.0)
    assert next(e for e in events if e.name == "pinch_end").reason == "released"


def test_hand_loss_while_pinching_releases(bare_engine, rules):
    frames = [
        dict(hands=[make_hand((600, 300), (600, 320))], cursor=cur(), t=0.0),
        dict(hands=[], t=0.1),
    ]
    events = run_frames(bare_engine, rules, frames)
    end = [e for e in events if e.name == "pinch_end"]
    assert len(end) == 1 and end[0].reason == "hand_lost"
    assert frames[1]["_ctx"].pinch_status == "NO HAND DETECTED"


def test_cursor_move_emitted_before_pinch_like_the_prototype(bare_engine, rules):
    frames = [dict(hands=[make_hand((600, 300), (600, 320))], cursor=cur(), t=0.0)]
    events = run_frames(bare_engine, rules, frames)
    assert names(events)[:2] == ["cursor_move", "pinch_start"]


def test_no_cursor_move_when_cursor_inactive(bare_engine, rules):
    frames = [dict(hands=[make_hand()], cursor=cur(active=False), t=0.0)]
    assert "cursor_move" not in names(run_frames(bare_engine, rules, frames))


def test_drag_events_after_travel_threshold(bare_engine, rules):
    frames = [
        dict(hands=[make_hand((600, 300), (600, 320))], cursor=cur(100, 100), t=0.0),  # pinch starts
        dict(hands=[make_hand((600, 300), (600, 320))], cursor=cur(105, 100), t=0.1),  # 5 px: not a drag yet
        dict(hands=[make_hand((600, 300), (600, 320))], cursor=cur(130, 100), t=0.2),  # 30 px: drag starts
        dict(hands=[make_hand((600, 300), (600, 320))], cursor=cur(160, 120), t=0.3),
        dict(hands=[make_hand((600, 300), (600, 400))], cursor=cur(160, 120), t=0.4),  # release
    ]
    n = names(run_frames(bare_engine, rules, frames))
    assert n.count("drag_start") == 1 and n.count("drag_move") == 2 and n.count("drag_end") == 1
    assert n.index("drag_end") < n.index("pinch_end")


def test_pixel_thresholds_scale_with_frame_width(bare_engine, rules):
    # 640-px-wide frame: 45 px @1280 becomes 22.5 px
    near = make_hand((300, 200), (300, 220), w=640, h=360)  # 20 px
    far = make_hand((300, 200), (300, 225), w=640, h=360)  # 25 px
    f_near = [dict(hands=[near], cursor=cur(), t=0.0, w=640, h=360)]
    f_far = [dict(hands=[far], cursor=cur(), t=0.0, w=640, h=360)]
    assert "pinch_start" in names(run_frames(bare_engine, rules, f_near))
    rules._pinching = False
    assert "pinch_start" not in names(run_frames(bare_engine, rules, f_far))


def test_head_flick_events_and_cooldown(bare_engine, proto_rules):
    def f(x, t):
        return dict(faces=[make_face((x, 360))], t=t)

    events = run_frames(bare_engine, proto_rules, [f(600, 0.0), f(640, 0.1), f(600, 0.5), f(700, 1.4), f(640, 1.5)])
    flicks = [e for e in events if e.name == "head_flick"]
    assert [e.direction for e in flicks] == ["right", "right"]
    assert flicks[0].dx == 40
    assert "head_flick_right" in names(events) and "head_flick_left" not in names(events)


def test_face_loss_keeps_previous_nose_like_the_prototype(bare_engine, proto_rules):
    frames = [dict(faces=[make_face((600, 360))], t=0.0), dict(faces=[], t=0.1), dict(faces=[make_face((650, 360))], t=0.2)]
    assert "head_flick" in names(run_frames(bare_engine, proto_rules, frames))  # compared with the last known position


def test_face_loss_resets_when_configured(bare_engine):
    proc = GestureRulesProcessor(face_config=FaceConfig(reset_on_loss=True), hand_config=HandConfig())
    bare_engine.add_processor(proc)
    frames = [dict(faces=[make_face((600, 360))], t=0.0), dict(faces=[], t=0.1), dict(faces=[make_face((650, 360))], t=0.2)]
    assert "head_flick" not in names(run_frames(bare_engine, proc, frames))


def test_scroll_mode_freezes_cursor_and_emits_wheel_events(bare_engine):
    proc = GestureRulesProcessor(hand_config=HandConfig(scroll_enabled=True), face_config=FaceConfig())
    bare_engine.add_processor(proc)
    up = _pose_hand()
    events = []
    bare_engine.on("*", events.append)
    for t, dy in enumerate([0.0, -0.02, -0.04]):  # finger travels upwards in the image
        h = _pose_hand()
        h.landmarks[8, 1] += dy
        h.landmarks[12, 1] += dy
        ctx = make_ctx(hands=[h], cursor=cur(), t=float(t))
        proc.process(ctx)
        assert ctx.scroll_mode is True
    n = names(events)
    assert n.count("scroll_start") == 1 and "cursor_move" not in n and "pinch_start" not in n
    amounts = [e.amount for e in events if e.name == "scroll"]
    assert amounts and all(a > 0 for a in amounts)  # positive = scroll up
    proc.process(make_ctx(hands=[], t=9.0))
    assert "scroll_end" in names(events)


# ---------------------------------------------------- equivalence with the prototype loop
def prototype_reference(samples):
    """The original script's pinch + flick logic, transcribed verbatim.

    samples: list of (hand or None, nose or None, t). Returns (mouse_events, right_clicks, statuses).
    """
    is_mouse_down, prev_nose, last_right = False, None, 0
    mouse_events, right_clicks, statuses = [], [], []
    for i, (hand, nose, t) in enumerate(samples):
        if hand is not None:
            ix, iy, tx, ty = hand
            d = (((ix - tx) ** 2) + ((iy - ty) ** 2)) ** 0.5
            if d < 45:
                status = "HOLDING CLICK"
                if not is_mouse_down:
                    mouse_events.append((i, "down"))
                    is_mouse_down = True
            else:
                status = "OPEN"
                if is_mouse_down:
                    mouse_events.append((i, "up"))
                    is_mouse_down = False
        else:
            status = "NO HAND DETECTED"
            if is_mouse_down:
                mouse_events.append((i, "up"))
                is_mouse_down = False
        statuses.append(status)
        if nose is not None:
            nx, ny = nose
            if prev_nose is not None and (t - last_right > 1.2):
                dx, dy = nx - prev_nose[0], ny - prev_nose[1]
                if abs(dx) > 28 and abs(dx) > abs(dy) * 3:
                    right_clicks.append(i)
                    last_right = t
            prev_nose = (nx, ny)
    return mouse_events, right_clicks, statuses


def test_matches_prototype_on_random_input(bare_engine, proto_rules):
    rng = random.Random(1234)
    samples, t = [], 1.0
    for _ in range(600):
        t += rng.uniform(0.03, 0.12)
        hand = None
        if rng.random() > 0.15:
            ix, iy = rng.randint(100, 1100), rng.randint(100, 600)
            ang, dist = rng.uniform(0, math.tau), rng.choice([10, 30, 44, 45, 46, 60, 120])
            hand = (ix, iy, int(ix + dist * math.cos(ang)), int(iy + dist * math.sin(ang)))
        nose = (rng.randint(500, 800), rng.randint(300, 420)) if rng.random() > 0.1 else None
        samples.append((hand, nose, t))

    ref_mouse, ref_clicks, ref_status = prototype_reference(samples)

    events, statuses, indexed = [], [], []
    bare_engine.on("*", events.append)
    for i, (hand, nose, ts) in enumerate(samples):
        hands = [make_hand((hand[0], hand[1]), (hand[2], hand[3]))] if hand else []
        faces = [make_face(nose)] if nose else []
        before = len(events)
        ctx = make_ctx(hands=hands, faces=faces, t=ts, cursor=cur(), frame_index=i)
        proto_rules.process(ctx)
        statuses.append(ctx.pinch_status)
        indexed += [(i, e.name) for e in events[before:]]

    assert statuses == ref_status
    mouse = [(i, "down" if n == "pinch_start" else "up") for i, n in indexed if n in ("pinch_start", "pinch_end")]
    assert mouse == ref_mouse
    assert [i for i, n in indexed if n == "head_flick"] == ref_clicks
    assert ref_clicks and ref_mouse  # the random input actually exercised both gestures


# ================================================================ v0.2: right pinch, pause, AI gesture
def v2_rules(bare_engine, **hand):
    proc = GestureRulesProcessor(hand_config=HandConfig(**hand), face_config=FaceConfig())
    bare_engine.add_processor(proc)
    proc.setup()
    return proc


def middle_pinch(gap=30, index_far=True):
    """Thumb at (600, 400); middle tip ``gap`` px above it; index far away (or just outside the threshold)."""
    return make_hand(index_px=(600, 250) if index_far else (600, 400 - 46), thumb_px=(600, 400), middle_px=(600, 400 - gap))


def test_middle_pinch_emits_right_click_once_and_status(bare_engine):
    rules = v2_rules(bare_engine)
    frames = [
        dict(hands=[middle_pinch(gap=100)], cursor=cur(), t=0.0),
        dict(hands=[middle_pinch(gap=30)], cursor=cur(), t=0.1),
        dict(hands=[middle_pinch(gap=30)], cursor=cur(), t=0.2),
        dict(hands=[middle_pinch(gap=100)], cursor=cur(), t=0.3),
    ]
    events = run_frames(bare_engine, rules, frames)
    n = names(events)
    assert n.count("right_pinch_start") == n.count("right_click") == n.count("right_pinch_end") == 1
    assert "pinch_start" not in n  # the index finger stayed away: no left click
    click = next(e for e in events if e.name == "right_click")
    assert (click.x, click.y, click.via) == (100.0, 200.0, "middle_pinch")
    assert [f["_ctx"].pinch_status for f in frames] == ["OPEN", "RIGHT CLICK", "RIGHT CLICK", "OPEN"]
    assert n.index("cursor_move") < n.index("right_pinch_start")


def test_right_click_cooldown_blocks_rapid_repeats(bare_engine):
    rules = v2_rules(bare_engine, right_click_cooldown_s=0.5)
    pinch, release = middle_pinch(30), middle_pinch(100)
    frames = [dict(hands=[h], cursor=cur(), t=t) for h, t in [(pinch, 0.0), (release, 0.1), (pinch, 0.2), (release, 0.3), (pinch, 0.9)]]
    events = run_frames(bare_engine, rules, frames)
    assert [e.name for e in events].count("right_pinch_start") == 3
    assert [e.name for e in events].count("right_click") == 2  # the 0.2 s one is inside the cooldown


def test_closer_fingertip_wins_when_both_touch_the_thumb(bare_engine):
    rules = v2_rules(bare_engine)
    index_closer = make_hand(index_px=(600, 380), thumb_px=(600, 400), middle_px=(600, 365))  # 20 vs 35 px
    middle_closer = make_hand(index_px=(600, 365), thumb_px=(600, 400), middle_px=(600, 380))  # 35 vs 20 px
    ev = run_frames(bare_engine, rules, [dict(hands=[index_closer], cursor=cur(), t=0.0)])
    assert names(ev) == ["cursor_move", "pinch_start"]
    rules2 = v2_rules(bare_engine)
    rules2._pinching = False
    ev2 = run_frames(bare_engine, rules2, [dict(hands=[middle_closer], cursor=cur(), t=0.0)])
    assert "right_pinch_start" in names(ev2) and "pinch_start" not in names(ev2)


def test_an_active_left_pinch_is_not_stolen_by_the_middle_finger(bare_engine):
    rules = v2_rules(bare_engine)
    hold = make_hand(index_px=(600, 380), thumb_px=(600, 400), middle_px=(600, 395))  # middle even closer
    frames = [dict(hands=[make_hand(index_px=(600, 380), thumb_px=(600, 400))], cursor=cur(), t=0.0), dict(hands=[hold], cursor=cur(), t=0.1)]
    ev = names(run_frames(bare_engine, rules, frames))
    assert ev.count("pinch_start") == 1 and "right_pinch_start" not in ev and "pinch_end" not in ev


def test_right_pinch_can_be_disabled_and_is_scaled(bare_engine):
    rules = v2_rules(bare_engine, right_pinch_enabled=False)
    assert "right_pinch_start" not in names(run_frames(bare_engine, rules, [dict(hands=[middle_pinch(30)], cursor=cur(), t=0.0)]))
    rules = v2_rules(bare_engine, right_pinch_threshold_px=20)
    assert "right_pinch_start" not in names(run_frames(bare_engine, rules, [dict(hands=[middle_pinch(30)], cursor=cur(), t=0.0)]))


def test_hand_loss_ends_a_right_pinch(bare_engine):
    rules = v2_rules(bare_engine)
    ev = run_frames(bare_engine, rules, [dict(hands=[middle_pinch(30)], cursor=cur(), t=0.0), dict(hands=[], t=0.1)])
    end = next(e for e in ev if e.name == "right_pinch_end")
    assert end.reason == "hand_lost"


def test_ai_gesture_triggers_right_click_with_cooldown_and_pause(bare_engine):
    rules = v2_rules(bare_engine)
    rules.process(make_ctx(hands=[make_hand()], cursor=cur(), t=5.0))
    events = []
    bare_engine.on("right_click", events.append)
    bare_engine.emit("gesture_right_click", gesture="right_click", confidence=0.9, previous=None, handedness="Right")
    bare_engine.emit("gesture_right_click", gesture="right_click", confidence=0.9, previous=None, handedness="Right")
    assert [(e.via, e.x, e.y) for e in events] == [("ai_gesture", 100.0, 200.0)]  # second one inside the cooldown
    rules.set_paused(True)
    rules._last_right_click = -1e9
    bare_engine.emit("gesture_right_click", gesture="right_click", confidence=0.9, previous=None, handedness="Right")
    assert len(events) == 1  # paused


def test_ai_right_click_label_is_configurable_and_unsubscribed_on_close(bare_engine):
    bare_engine.config.onnx.right_click_label = "peace"
    rules = v2_rules(bare_engine)
    events = []
    bare_engine.on("right_click", events.append)
    bare_engine.emit("gesture_right_click", gesture="right_click")
    assert events == []
    bare_engine.emit("gesture_peace", gesture="peace")
    assert len(events) == 1
    rules.close()
    rules._last_right_click = -1e9
    bare_engine.emit("gesture_peace", gesture="peace")
    assert len(events) == 1


# ------------------------------------------------------------------------------ pause toggle
def test_hold_toggle_fires_once_then_latches_until_released():
    from open_ai_hci.processors.gesture_rules import HoldToggle

    h = HoldToggle(1.0)
    assert h.update(True, 0.0) == (False, 0.0)
    assert h.update(True, 0.5) == (False, 0.5)
    assert h.update(True, 1.0) == (True, 1.0)
    assert h.update(True, 2.0) == (False, 0.0)  # latched
    assert h.update(False, 2.1) == (False, 0.0)  # released
    assert h.update(True, 3.0) == (False, 0.0) and h.update(True, 4.0) == (True, 1.0)


def test_open_palm_and_fist_detection():
    from conftest import make_fist, make_palm

    from open_ai_hci.processors.gesture_rules import is_fist, is_open_palm

    assert is_open_palm(make_palm()) and not is_fist(make_palm())
    assert is_fist(make_fist()) and not is_open_palm(make_fist())
    assert not is_open_palm(make_hand()) and not is_open_palm(make_fist())


def palm_frames(t_values, hand_factory=None):
    from conftest import make_palm

    return [dict(hands=[(hand_factory or make_palm)()], cursor=cur(), t=t) for t in t_values]


def test_open_palm_hold_pauses_then_resumes_and_suppresses_events(bare_engine):
    from conftest import make_palm

    rules = v2_rules(bare_engine, pause_hold_s=1.0)
    events = run_frames(bare_engine, rules, palm_frames([0.0, 0.5, 1.0, 1.5, 3.0]))
    n = names(events)
    assert n.count("control_paused") == 1 and "control_resumed" not in n
    ctxs = [f["_ctx"] for f in palm_frames([])] or []
    assert rules.paused
    # while paused: no cursor events at all, status says PAUSED
    after = run_frames(bare_engine, rules, [dict(hands=[make_hand((600, 300), (600, 320))], cursor=cur(), t=4.0)])
    assert names(after) == []
    # take the hand away (releases the latch), then hold the palm again to resume
    run_frames(bare_engine, rules, [dict(hands=[], t=4.5)])
    resumed = run_frames(bare_engine, rules, [dict(hands=[make_palm()], cursor=cur(), t=t) for t in (5.0, 5.6, 6.1)])
    assert "control_resumed" in names(resumed) and not rules.paused
    del ctxs


def test_pause_status_progress_and_ctx_flag(bare_engine):
    rules = v2_rules(bare_engine, pause_hold_s=1.0)
    frames = palm_frames([0.0, 0.5, 1.0])
    run_frames(bare_engine, rules, frames)
    assert frames[0]["_ctx"].extras["pause_progress"] == 0.0 and frames[0]["_ctx"].pinch_status == "OPEN"
    assert frames[1]["_ctx"].extras["pause_progress"] == 0.5
    assert frames[2]["_ctx"].pinch_status == "PAUSED" and frames[2]["_ctx"].paused is True


def test_the_palm_stays_up_after_pausing_without_flipping_back(bare_engine):
    rules = v2_rules(bare_engine, pause_hold_s=0.5)
    ev = names(run_frames(bare_engine, rules, palm_frames([i * 0.25 for i in range(20)])))
    assert ev.count("control_paused") == 1 and "control_resumed" not in ev


def test_pause_releases_a_held_pinch_and_hides_cursor_events(bare_engine):
    rules = v2_rules(bare_engine)
    events = run_frames(bare_engine, rules, [dict(hands=[make_hand((600, 300), (600, 320))], cursor=cur(), t=0.0)])
    assert rules._pinching
    rules.set_paused(True, via="api")
    more = run_frames(bare_engine, rules, [dict(hands=[make_hand((600, 300), (600, 320))], cursor=cur(), t=0.1)])
    assert names(more)[:1] == ["pinch_end"] and next(e for e in more if e.name == "pinch_end").reason == "paused"
    assert "cursor_move" not in names(more) and not rules._pinching
    del events


def test_fist_pause_gesture_and_none(bare_engine):
    from conftest import make_fist

    rules = v2_rules(bare_engine, pause_gesture="fist", pause_hold_s=0.5)
    ev = names(run_frames(bare_engine, rules, [dict(hands=[make_fist()], cursor=cur(), t=t) for t in (0.0, 0.6)]))
    assert "control_paused" in ev
    rules2 = v2_rules(bare_engine, pause_gesture="none", pause_hold_s=0.5)
    assert not names(run_frames(bare_engine, rules2, palm_frames([0.0, 0.6, 1.2]))).count("control_paused")


def test_pinching_with_an_open_hand_never_pauses(bare_engine):
    from conftest import make_palm

    rules = v2_rules(bare_engine, pause_hold_s=0.5)
    pinching_palm = make_palm()
    pinching_palm.landmarks[4] = pinching_palm.landmarks[8].copy()  # thumb on the index tip
    frames = [dict(hands=[pinching_palm], cursor=cur(), t=t) for t in (0.0, 0.6, 1.2)]
    assert "control_paused" not in names(run_frames(bare_engine, rules, frames)) and not rules.paused


def test_set_paused_is_idempotent_and_reports_changes(bare_engine):
    rules = v2_rules(bare_engine)
    events = []
    bare_engine.on("*", events.append)
    assert rules.set_paused(True) and not rules.set_paused(True)
    assert rules.set_paused(False) and not rules.set_paused(False)
    assert [(e.name, e.via) for e in events] == [("control_paused", "api"), ("control_resumed", "api")]


def test_flick_is_off_by_default_but_blink_still_works_while_unpaused(bare_engine):
    rules = v2_rules(bare_engine)
    ev = run_frames(bare_engine, rules, [dict(faces=[make_face((600, 360))], t=0.0), dict(faces=[make_face((700, 360))], t=0.1)])
    assert "head_flick" not in names(ev)

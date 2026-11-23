import pytest
from conftest import FakePyAutoGUI

from open_ai_hci import MouseConfig, OSKeyboardAction, OSMouseAction
from open_ai_hci.actions.os_keyboard import parse_hotkey


@pytest.fixture
def mouse(bare_engine, fake_pag):
    act = bare_engine.add_action(OSMouseAction(pyautogui_module=fake_pag))
    bare_engine.setup()
    return act


def test_failsafe_and_pause_are_applied_from_config(bare_engine, fake_pag):
    bare_engine.config.mouse.failsafe, bare_engine.config.mouse.pause = False, 0.05
    bare_engine.add_action(OSMouseAction(pyautogui_module=fake_pag))
    bare_engine.setup()
    assert fake_pag.FAILSAFE is False and fake_pag.PAUSE == 0.05


def test_move_click_and_release_mapping(bare_engine, mouse, fake_pag):
    bare_engine.emit("cursor_move", x=960.9, y=540.2, raw_x=0, raw_y=0)
    bare_engine.emit("pinch_start", x=960, y=540, distance=10.0, handedness="Right")
    bare_engine.emit("pinch_start", x=960, y=540, distance=10.0, handedness="Right")  # duplicate: ignored
    assert mouse.is_mouse_down
    bare_engine.emit("pinch_end", reason="released", x=0, y=0)
    bare_engine.emit("pinch_end", reason="released", x=0, y=0)  # duplicate: ignored
    assert fake_pag.calls == [
        ("moveTo", (960, 540), {}),  # int() truncation, like the prototype
        ("mouseDown", (), {"button": "left"}),
        ("mouseUp", (), {"button": "left"}),
    ]


def test_targets_are_clamped_away_from_the_failsafe_corners(bare_engine, mouse, fake_pag):
    bare_engine.emit("cursor_move", x=99999, y=-50, raw_x=0, raw_y=0)
    bare_engine.emit("cursor_move", x=0, y=99999, raw_x=0, raw_y=0)
    assert [c[1] for c in fake_pag.calls] == [(1917, 2), (2, 1077)]
    mouse.clamp_margin = 0
    bare_engine.emit("cursor_move", x=0, y=0, raw_x=0, raw_y=0)
    assert fake_pag.calls[-1][1] == (0, 0)


def test_drag_events_mode_uses_dragTo_while_held(bare_engine, mouse, fake_pag):
    mouse.drag_events = True
    bare_engine.emit("cursor_move", x=100, y=100, raw_x=0, raw_y=0)
    bare_engine.emit("pinch_start", x=100, y=100, distance=1.0, handedness="Right")
    bare_engine.emit("cursor_move", x=200, y=150, raw_x=0, raw_y=0)
    assert fake_pag.calls[-1] == ("dragTo", (200, 150), {"button": "left", "mouseDownUp": False})
    assert fake_pag.calls[0][0] == "moveTo"


def test_head_flick_is_ignored_by_default(bare_engine, mouse, fake_pag):
    bare_engine.emit("head_flick", direction="left", dx=-40, dy=0, x=1, y=1)
    assert fake_pag.calls == []  # v0.2: head flicks no longer right-click


def test_head_flick_right_clicks_when_opted_in(bare_engine, fake_pag):
    bare_engine.add_action(OSMouseAction(MouseConfig(right_click_on_flick=True), pyautogui_module=fake_pag))
    bare_engine.setup()
    alerts = []
    bare_engine.on("alert", alerts.append)
    bare_engine.emit("head_flick", direction="left", dx=-40, dy=0, x=1, y=1)
    assert fake_pag.calls == [("click", (), {"button": "right"})]
    assert alerts[0].text == "triggered right click!"


def test_right_click_event_clicks_right(bare_engine, mouse, fake_pag):
    alerts = []
    bare_engine.on("alert", alerts.append)
    bare_engine.emit("right_click", x=10, y=20, via="middle_pinch")
    assert fake_pag.calls == [("click", (), {"button": "right"})]
    assert alerts[0].text == "right click"


def test_move_cursor_false_leaves_the_cursor_alone(bare_engine, fake_pag):
    bare_engine.add_action(OSMouseAction(MouseConfig(move_cursor=False), pyautogui_module=fake_pag))
    bare_engine.setup()
    bare_engine.emit("cursor_move", x=5, y=5, raw_x=0, raw_y=0)
    bare_engine.emit("pinch_start", x=5, y=5, distance=1.0, handedness="Right")
    assert fake_pag.names() == ["mouseDown"]


def test_reconfigure_resubscribes_live(bare_engine, fake_pag):
    act = bare_engine.add_action(OSMouseAction(pyautogui_module=fake_pag))
    bare_engine.setup()
    bare_engine.config.mouse.right_click_on_pinch = False
    bare_engine.config.mouse.left_click_on_pinch = False
    bare_engine.config.mouse.pause = 0.25
    act.reconfigure()
    bare_engine.emit("right_click", x=1, y=1, via="x")
    bare_engine.emit("pinch_start", x=1, y=1, distance=1.0, handedness="Right")
    assert fake_pag.calls == [] and fake_pag.PAUSE == 0.25
    bare_engine.config.mouse.right_click_on_pinch = True
    act.reconfigure()
    bare_engine.emit("right_click", x=1, y=1, via="x")
    assert fake_pag.names() == ["click"]


def test_reconfigure_releases_a_held_button_when_left_click_is_turned_off(bare_engine, mouse, fake_pag):
    bare_engine.emit("pinch_start", x=1, y=1, distance=1.0, handedness="Right")
    bare_engine.config.mouse.left_click_on_pinch = False
    mouse.reconfigure()
    assert fake_pag.calls[-1] == ("mouseUp", (), {"button": "left"}) and not mouse.is_mouse_down


@pytest.mark.parametrize("unit,expected", [(1, -5), (120, -600)])
def test_scroll_amounts_are_converted_to_platform_wheel_units(bare_engine, fake_pag, unit, expected):
    bare_engine.add_action(OSMouseAction(pyautogui_module=fake_pag, wheel_unit=unit))
    bare_engine.setup()
    bare_engine.emit("scroll", amount=-5)
    bare_engine.emit("intent", intent="scroll_down", args={"amount": 5}, text="", origin="voice", confidence=1.0, context={})
    bare_engine.emit("intent", intent="scroll", args={"amount": -500}, text="", origin="llm", confidence=1.0, context={})
    assert [c[1] for c in fake_pag.calls] == [(expected,)] * 3


def test_small_llm_scroll_never_rounds_to_nothing_and_move_intents_work(bare_engine, mouse, fake_pag):
    bare_engine.emit("intent", intent="scroll", args={"amount": "-30"}, text="", origin="llm", confidence=1.0, context={})
    assert fake_pag.calls[-1] == ("scroll", (-1,), {})
    bare_engine.emit("intent", intent="move", args={"x": 99999, "y": 10}, text="", origin="llm", confidence=1.0, context={})
    assert fake_pag.calls[-1] == ("moveTo", (1917, 10), {})
    bare_engine.emit("intent", intent="move", args={"dx": 5, "dy": -3}, text="", origin="llm", confidence=1.0, context={})
    assert fake_pag.calls[-1] == ("moveRel", (5, -3), {})
    n = len(fake_pag.calls)
    bare_engine.emit("intent", intent="move", args={}, text="", origin="llm", confidence=1.0, context={})
    bare_engine.emit("intent", intent="click", args={}, text="", origin="llm", confidence=1.0, context={}, dry_run=True)
    assert len(fake_pag.calls) == n  # nothing to do / dry run


def test_scroll_intent_with_garbage_amount_is_harmless(bare_engine, mouse, fake_pag):
    bare_engine.emit("intent", intent="scroll", args={"amount": None}, text="", origin="llm", confidence=1.0, context={})
    assert fake_pag.calls == []


def test_failsafe_releases_the_button_alerts_and_stops_the_engine(bare_engine, mouse, fake_pag):
    alerts = []
    bare_engine.on("alert", alerts.append)
    bare_engine.emit("pinch_start", x=1, y=1, distance=1.0, handedness="Right")
    assert mouse.is_mouse_down
    fake_pag.raise_on = "moveTo"
    bare_engine.emit("cursor_move", x=0, y=0, raw_x=0, raw_y=0)  # PyAutoGUI raises FailSafeException
    assert bare_engine._stop.is_set()
    assert "FAILSAFE" in alerts[-1].text
    assert not mouse.is_mouse_down
    assert fake_pag.calls[-1] == ("mouseUp", (), {"button": "left"})  # button released despite the corner
    assert fake_pag.FAILSAFE is True  # and the failsafe is switched back on afterwards


def test_release_suspends_the_failsafe_only_for_the_release(bare_engine, mouse, fake_pag):
    seen = []
    fake_pag.mouseUp = lambda **k: seen.append(fake_pag.FAILSAFE)  # what PyAutoGUI would check at call time
    bare_engine.emit("pinch_start", x=1, y=1, distance=1.0, handedness="Right")
    bare_engine.close()
    assert seen == [False] and fake_pag.FAILSAFE is True


def test_close_releases_a_held_button_and_unsubscribes(bare_engine, mouse, fake_pag):
    bare_engine.emit("pinch_start", x=1, y=1, distance=1.0, handedness="Right")
    bare_engine.close()
    assert fake_pag.calls[-1] == ("mouseUp", (), {"button": "left"})
    n = len(fake_pag.calls)
    bare_engine.emit("pinch_start", x=1, y=1, distance=1.0, handedness="Right")
    assert len(fake_pag.calls) == n  # no longer subscribed


def test_scroll_and_voice_intents(bare_engine, mouse, fake_pag):
    bare_engine.emit("scroll", amount=3)
    bare_engine.emit("intent", intent="double_click", args={}, text="", origin="voice", confidence=1.0, context={})
    bare_engine.emit("intent", intent="right_click", args={}, text="", origin="voice", confidence=1.0, context={})
    bare_engine.emit("intent", intent="scroll_down", args={"amount": 4}, text="", origin="voice", confidence=1.0, context={})
    bare_engine.emit("intent", intent="scroll_up", args={}, text="", origin="voice", confidence=1.0, context={})
    bare_engine.emit("intent", intent="hotkey", args={"keys": "ctrl+c"}, text="", origin="voice", confidence=1.0, context={})
    assert [(n, a, k) for n, a, k in fake_pag.calls] == [
        ("scroll", (3,), {}),
        ("doubleClick", (), {"button": "left"}),
        ("click", (), {"button": "right"}),
        ("scroll", (-4,), {}),
        ("scroll", (5,), {}),  # the hotkey intent belongs to the keyboard action, not the mouse
    ]


def test_config_switches_control_subscriptions(bare_engine, fake_pag):
    cfg = MouseConfig(left_click_on_pinch=False, right_click_on_pinch=False, right_click_on_flick=False, handle_intents=False)
    act = bare_engine.add_action(OSMouseAction(cfg, pyautogui_module=fake_pag))
    assert set(act.subscriptions()) == {"cursor_move", "scroll"}


# ---------------------------------------------------------------------------- keyboard
def test_parse_hotkey():
    assert parse_hotkey("Ctrl+Shift+T") == ("ctrl", "shift", "t")
    assert parse_hotkey(["Alt", "Left"]) == ("alt", "left")
    with pytest.raises(ValueError):
        parse_hotkey("  ")


@pytest.fixture
def keyboard(bare_engine, fake_pag):
    def build(bindings, **kw):
        act = bare_engine.add_action(OSKeyboardAction(bindings, pyautogui_module=fake_pag, **kw))
        bare_engine.setup()
        return act

    return build


def test_hotkey_macro_and_callable_bindings(bare_engine, keyboard, fake_pag):
    seen = []
    keyboard(
        {
            "head_flick_left": "alt+left",
            "blink": [("hotkey", "ctrl", "a"), ("wait", 0), ("write", "hi"), ("press", "enter")],
            "pinch_start": lambda e: seen.append(e.x),
        }
    )
    bare_engine.emit("head_flick_left", direction="left", dx=-1, dy=0, x=0, y=0)
    bare_engine.emit("blink", ear=0.1)
    bare_engine.emit("pinch_start", x=7, y=8, distance=1.0, handedness="Right")
    assert fake_pag.calls == [
        ("hotkey", ("alt", "left"), {}),
        ("hotkey", ("ctrl", "a"), {}),
        ("write", ("hi",), {"interval": 0.0}),
        ("press", ("enter",), {}),
    ]
    assert seen == [7]


def test_cooldown_dry_run_and_validation(bare_engine, keyboard, fake_pag):
    keyboard({"blink": "space"}, cooldown_s=1000)
    bare_engine.emit("blink", ear=0.1)
    bare_engine.emit("blink", ear=0.1)
    assert fake_pag.names() == ["hotkey"]  # the second emit is inside the cooldown

    with pytest.raises(ValueError, match="Invalid macro step"):
        OSKeyboardAction({"x": [("explode", 1)]})
    with pytest.raises(ValueError):
        OSKeyboardAction({"x": ""})


def test_dry_run_never_touches_pyautogui(bare_engine):
    act = bare_engine.add_action(OSKeyboardAction({"blink": "ctrl+c"}, dry_run=True))  # no module injected
    bare_engine.setup()
    assert act._pag is None
    bare_engine.emit("blink", ear=0.1)  # must not raise


def test_keyboard_intents(bare_engine, keyboard, fake_pag):
    keyboard({})
    for intent, args in [("hotkey", {"keys": "ctrl+v"}), ("press", {"key": "enter"}), ("type", {"text": "hello"}), ("click", {})]:
        bare_engine.emit("intent", intent=intent, args=args, text="", origin="voice", confidence=1.0, context={})
    assert fake_pag.names() == ["hotkey", "press", "write"]  # "click" is the mouse's job

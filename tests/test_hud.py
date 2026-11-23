import numpy as np
import pytest
from conftest import make_ctx, make_face, make_hand

from open_ai_hci import HUD, HUDConfig
from open_ai_hci.types import CursorState
from open_ai_hci.ui.hud import pinch_color

GRAY = 100


def gray_ctx(**kw):
    ctx = make_ctx(**kw)
    ctx.frame[:] = GRAY
    return ctx


def px(ctx, x, y):
    return tuple(int(v) for v in ctx.frame[y, x])


def test_pinch_colours_match_the_prototype():
    assert pinch_color("HOLDING CLICK") == (0, 255, 0)  # green
    assert pinch_color("NO HAND DETECTED") == (0, 0, 255)  # red
    assert pinch_color("OPEN") == (0, 165, 255)  # orange
    assert pinch_color("DISCONNECTED") == (0, 165, 255)


def test_panel_and_markers_use_the_prototype_geometry_and_colours():
    ctx = gray_ctx(hands=[make_hand((600, 300), (600, 400))], faces=[make_face((640, 200))])
    HUD(headless=True).draw(ctx)
    assert px(ctx, 15, 15) == (0, 0, 0)  # black panel (10,10)-(380,150)
    assert px(ctx, 379, 149) == (0, 0, 0)
    assert px(ctx, 395, 100) == (GRAY,) * 3  # outside the panel
    # circles are drawn first and the white line on top (prototype order), so sample off the line
    assert px(ctx, 606, 300) == (0, 255, 255)  # yellow index tip (radius 8)
    assert px(ctx, 606, 400) == (255, 0, 255)  # magenta thumb tip
    assert px(ctx, 640, 200) == (0, 0, 255)  # red nose
    assert px(ctx, 600, 350) == (255, 255, 255)  # white measuring line


def test_status_line_colour_follows_the_pinch_state():
    def line_pixels(status):
        ctx = gray_ctx()
        ctx.pinch_status = status
        HUD(headless=True).draw(ctx)
        region = ctx.frame[78:96, 20:300].reshape(-1, 3)  # where "Pinch Grip: ..." is drawn
        return {tuple(int(v) for v in p) for p in region}

    assert (0, 255, 0) in line_pixels("HOLDING CLICK")
    assert (0, 0, 255) in line_pixels("NO HAND DETECTED")
    assert (0, 165, 255) in line_pixels("OPEN")


def test_cursor_text_is_drawn_and_missing_cursor_shows_zero():
    a, b = gray_ctx(cursor=CursorState(960, 540, 0, 0)), gray_ctx()
    hud = HUD(headless=True)
    hud.draw(a)
    hud.draw(b)
    assert not np.array_equal(a.frame, b.frame)  # different cursor -> different text pixels


def test_alert_banner_appears_and_expires(bare_engine):
    hud = HUD(headless=True)
    hud.attach(bare_engine)
    hud.setup()
    ctx = gray_ctx()
    hud.draw(ctx)
    assert px(ctx, 15, 165) == (GRAY,) * 3  # no banner yet
    bare_engine.emit("alert", text="triggered right click!")  # components publish alerts on the bus
    ctx = gray_ctx()
    hud.draw(ctx)
    assert px(ctx, 15, 165) == (0, 128, 0)  # dark-green banner (10,160)-(380,200)
    hud.show_alert("gone", seconds=0)
    ctx = gray_ctx()
    hud.draw(ctx)
    assert px(ctx, 15, 165) == (GRAY,) * 3
    hud.close()
    assert bare_engine.bus.listener_count("alert") == 0  # unsubscribed


def test_extras_panel_only_when_there_is_something_to_show():
    plain, rich = gray_ctx(), gray_ctx()
    rich.gesture, rich.gesture_confidence = "fist", 0.93
    HUD(headless=True).draw(plain)
    HUD(headless=True).draw(rich)
    assert px(plain, 15, 215) == (GRAY,) * 3 and px(rich, 15, 215) == (0, 0, 0)
    off = gray_ctx()
    off.gesture = "fist"
    HUD(HUDConfig(show_extras=False), headless=True).draw(off)
    assert px(off, 15, 215) == (GRAY,) * 3


def test_skeleton_drawing_is_optional():
    hand = make_hand()
    on, off = gray_ctx(hands=[hand]), gray_ctx(hands=[hand])
    HUD(HUDConfig(show_skeleton=True), headless=True).draw(on)
    HUD(headless=True).draw(off)
    assert (on.frame != off.frame).any()


def test_headless_show_returns_no_key_and_quit_detection():
    hud = HUD(headless=True)
    assert hud.show(gray_ctx()) == -1
    hud._quit_codes = {ord("q"), ord("Q")}
    assert hud.is_quit_key(ord("q")) and hud.is_quit_key(-2) and not hud.is_quit_key(ord("x")) and not hud.is_quit_key(-1)

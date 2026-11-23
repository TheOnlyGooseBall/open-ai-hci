"""The extension snippets from CONTRIBUTING.md, verbatim, so the guide can never go stale."""
import numpy as np
import pytest
from conftest import FakeHandBackend

from open_ai_hci import (
    BaseAction,
    BaseProcessor,
    BaseSensor,
    FrameContext,
    HandSensor,
    HCIConfig,
    HCIEngine,
    MouseConfig,
)
from open_ai_hci.processors.gesture_rules import finger_extended

# ----------------------------------------------------------------- snippet: a sensor


class BrightnessSensor(BaseSensor):
    """Publishes average frame brightness and lights_on / lights_off events on transitions."""

    name = "brightness"

    def __init__(self, dark_below: float = 40.0) -> None:
        super().__init__()
        self.dark_below = dark_below
        self._dark = False

    def process(self, ctx: FrameContext) -> None:
        level = float(ctx.rgb.mean())
        ctx.extras["brightness"] = level
        dark = level < self.dark_below
        if dark != self._dark:
            self._dark = dark
            self.emit("lights_off" if dark else "lights_on", level=level)


# -------------------------------------------------------------- snippet: a processor


class OpenPalmProcessor(BaseProcessor):
    """Emits open_palm_start / open_palm_end when all four fingers are extended."""

    name = "open_palm"
    FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))  # (tip, pip) landmark indices

    def __init__(self) -> None:
        self._open = False

    def process(self, ctx: FrameContext) -> None:
        hand = ctx.hand
        is_open = hand is not None and all(finger_extended(hand, tip, pip) for tip, pip in self.FINGERS)
        if is_open and not self._open:
            self.emit("open_palm_start", handedness=hand.handedness)
        elif not is_open and self._open:
            self.emit("open_palm_end")
        self._open = is_open
        ctx.extras["open_palm"] = is_open


# ----------------------------------------------------------------- snippet: an action


class ConsoleAction(BaseAction):
    """Prints a line for every pinch and for lights going off."""

    name = "console"

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def subscriptions(self):
        return {"pinch_start": self.on_pinch, "lights_off": self.on_dark}

    def on_pinch(self, event) -> None:
        self.lines.append(f"pinch at {event.x:.0f},{event.y:.0f}")

    def on_dark(self, event) -> None:
        self.lines.append(f"dark ({event.level:.0f})")


# ------------------------------------------------------------------------------ tests
def engine_with_plugins(hand_script):
    cfg = HCIConfig(mouse=MouseConfig(enabled=False))
    cfg.screen.width, cfg.screen.height = 1920, 1080
    hci = HCIEngine(cfg, sensors=[HandSensor(backend=FakeHandBackend(hand_script))], raise_errors=True)
    hci.add_sensor(BrightnessSensor())
    hci.add_processor(OpenPalmProcessor())
    console = hci.add_action(ConsoleAction())
    return hci, console


def test_custom_sensor_and_action_work_together():
    hci, console = engine_with_plugins([None, None])
    events = []
    hci.on("*", events.append)
    bright, dark = np.full((72, 128, 3), 200, np.uint8), np.zeros((72, 128, 3), np.uint8)
    hci.process_frame(bright)
    ctx = hci.process_frame(dark)
    assert [e.name for e in events if e.name.startswith("lights")] == ["lights_off"]
    assert ctx.extras["brightness"] == 0 and console.lines == ["dark (0)"]
    hci.process_frame(bright)
    assert events[-1].name == "lights_on"


def test_custom_processor_sees_hand_data():
    # A fake hand has every landmark at the same distance from the wrist only by construction,
    # so drive the processor directly with a hand whose four fingers are visibly extended.
    from conftest import make_ctx, make_hand

    hci, _ = engine_with_plugins([])
    proc = hci.processors[-1]
    open_hand = make_hand()
    lm = open_hand.landmarks
    lm[0] = (0.5, 0.9, 0)
    for tip, pip in OpenPalmProcessor.FINGERS:
        lm[pip] = (0.5, 0.6, 0)
        lm[tip] = (0.5, 0.3, 0)  # tip farther from the wrist than the PIP joint
    events = []
    hci.on("*", events.append)
    ctx = make_ctx(hands=[open_hand])
    proc.process(ctx)
    assert ctx.extras["open_palm"] is True and [e.name for e in events] == ["open_palm_start"]
    proc.process(make_ctx(hands=[]))
    assert [e.name for e in events] == ["open_palm_start", "open_palm_end"]


def test_action_unsubscribes_on_close():
    hci, console = engine_with_plugins([])
    hci.setup()
    hci.emit("lights_off", level=1.0)
    hci.close()
    hci.emit("lights_off", level=2.0)
    assert console.lines == ["dark (1)"]


def test_plugin_without_engine_gives_a_clear_error():
    with pytest.raises(RuntimeError, match="not attached"):
        BrightnessSensor().emit("x")

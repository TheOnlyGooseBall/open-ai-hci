import threading
import warnings

import numpy as np
import pytest

from open_ai_hci import Event, EventBus, EventType


def test_on_off_and_payload_as_attributes():
    bus, got = EventBus(), []
    handler = got.append
    assert bus.on("pinch_start", handler) is handler  # plain-call form returns the callback
    bus.emit("pinch_start", x=1, y=2)
    assert got[0].x == 1 and got[0].get("y") == 2 and got[0].get("missing", 7) == 7
    with pytest.raises(AttributeError):
        got[0].nope
    assert bus.off("pinch_start", handler) == 1
    bus.emit("pinch_start", x=3, y=4)
    assert len(got) == 1


def test_decorator_form_and_enum_names():
    bus, got = EventBus(), []

    @bus.on(EventType.PINCH_END)
    def handler(event):
        got.append(event.name)

    bus.emit("pinch_end")
    bus.emit(EventType.PINCH_END)
    assert got == ["pinch_end", "pinch_end"]


def test_once_fires_a_single_time():
    bus, got = EventBus(), []
    bus.once("blink", got.append)
    bus.emit("blink")
    bus.emit("blink")
    assert len(got) == 1
    assert bus.listener_count("blink") == 0


def test_once_removes_wildcard_registration_too():
    bus, got = EventBus(), []
    bus.once("*", got.append)
    bus.emit("a")
    bus.emit("b")
    assert [e.name for e in got] == ["a"]


def test_wildcard_gets_everything_except_frame():
    bus, got = EventBus(), []
    bus.on("*", got.append)
    bus.emit("alpha")
    bus.emit("frame", ctx=None)
    bus.emit("beta")
    assert [e.name for e in got] == ["alpha", "beta"]
    framed = []
    bus.on("frame", framed.append)
    bus.emit("frame", ctx=None)
    assert len(framed) == 1 and bus.has_subscribers("frame") and not bus.has_subscribers("alpha")


def test_handler_errors_are_isolated_by_default(caplog):
    bus, got = EventBus(), []

    def bad(event):
        raise RuntimeError("boom")

    bus.on("x", bad)
    bus.on("x", got.append)
    bus.emit("x")  # must not raise
    assert len(got) == 1
    assert "boom" in caplog.text


def test_raise_errors_mode_propagates():
    bus = EventBus(raise_errors=True)
    bus.on("x", lambda e: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        bus.emit("x")


def test_typo_detection_warns_but_custom_events_do_not():
    bus = EventBus()
    with pytest.warns(UserWarning, match="pinch_start"):
        bus.on("pinch_strat", lambda e: None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bus.on("my_totally_custom_event", lambda e: None)
        bus.register_event("pinch_strat2")
        bus.on("pinch_strat2", lambda e: None)
        bus.on("pinch_start", lambda e: None)


def test_reserved_payload_keys_are_rejected():
    bus = EventBus()
    for key in ("name", "source", "data", "timestamp", "frame_index"):
        with pytest.raises(ValueError, match="clash"):
            bus.emit("x", **{key: 1})


def test_event_to_dict_is_json_safe():
    import json

    e = Event("t", {"arr": np.arange(3, dtype=np.float32), "n": np.int64(5), "f": float("nan"), "nested": {"v": (1, 2)}})
    out = e.to_dict()
    json.dumps(out)  # must not raise
    assert out["data"] == {"arr": [0.0, 1.0, 2.0], "n": 5, "f": None, "nested": {"v": [1, 2]}}


def test_emit_stamps_frame_index_and_is_thread_safe():
    bus, got, lock = EventBus(), [], threading.Lock()
    bus.frame_index = 42

    def collect(e):
        with lock:
            got.append(e)

    bus.on("t", collect)
    threads = [threading.Thread(target=lambda: [bus.emit("t") for _ in range(100)]) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(got) == 800 and got[0].frame_index == 42


def test_off_without_callback_clears_all_and_clear():
    bus = EventBus()
    bus.on("a", lambda e: None)
    bus.on("a", lambda e: None)
    assert bus.off("a") == 2
    bus.on("b", lambda e: None)
    bus.clear()
    assert bus.listener_count("b") == 0


def test_off_works_with_bound_methods():
    class Listener:
        def __init__(self):
            self.n = 0

        def handle(self, event):
            self.n += 1

    bus, obj = EventBus(), Listener()
    bus.on("x", obj.handle)
    bus.emit("x")
    assert bus.off("x", obj.handle) == 1  # a fresh bound-method object must still match
    bus.emit("x")
    assert obj.n == 1

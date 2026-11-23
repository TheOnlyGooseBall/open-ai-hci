import contextlib
import json

import pytest
from conftest import FakeHandBackend

from open_ai_hci import HandSensor, HCIConfig, HCIEngine, WebSocketBridge
from open_ai_hci._optional import MissingOptionalDependency, is_available

pytestmark = pytest.mark.skipif(not is_available("websockets"), reason="websockets not installed")


@contextlib.contextmanager
def client(bridge):
    """Connect, consume the hello message, yield ``(ws, hello)``."""
    from websockets.sync.client import connect

    with connect(bridge.url, open_timeout=5) as ws:
        yield ws, json.loads(ws.recv(timeout=5))


def recv_json(ws, timeout=3):
    return json.loads(ws.recv(timeout=timeout))


@pytest.fixture
def bridge_engine(bare_engine):
    bridge = bare_engine.add_action(WebSocketBridge(port=0, events=["pinch_start", "hand_detected"], max_stream_hz=None))
    bare_engine.setup()
    yield bare_engine, bridge
    bare_engine.close()


def test_hello_and_event_forwarding(bridge_engine):
    engine, bridge = bridge_engine
    assert bridge.port != 0 and bridge.url.startswith("ws://127.0.0.1:")
    with client(bridge) as (ws, hello):
        assert hello["type"] == "hello" and hello["events"] == ["pinch_start", "hand_detected"] and hello["landmarks"] is False
        engine.emit("pinch_start", x=10.123456, y=20.0, distance=33.3, handedness="Right")
        msg = recv_json(ws)
        assert msg["type"] == "event" and msg["name"] == "pinch_start"
        assert msg["data"]["x"] == 10.1235 and msg["data"]["handedness"] == "Right"  # rounded, JSON-safe
        assert msg["source"] == "" and "timestamp" in msg
        engine.emit("blink", ear=0.1)  # not in the forwarded list
        with pytest.raises(TimeoutError):
            ws.recv(timeout=0.3)


def test_per_client_subscription_ping_and_multiple_clients(bridge_engine):
    engine, bridge = bridge_engine
    with client(bridge) as (ws1, _), client(bridge) as (ws2, _):
        ws1.send(json.dumps({"type": "subscribe", "events": ["hand_detected"]}))
        ws1.send(json.dumps({"type": "ping"}))
        assert recv_json(ws1)["type"] == "pong"
        engine.emit("pinch_start", x=1, y=1, distance=1.0, handedness="Right")
        assert recv_json(ws2)["name"] == "pinch_start"
        with pytest.raises(TimeoutError):
            ws1.recv(timeout=0.3)  # ws1 narrowed itself to hand_detected
        engine.emit("hand_detected", count=1)
        assert recv_json(ws1)["name"] == "hand_detected"
        assert recv_json(ws2)["name"] == "hand_detected"
        ws1.send("not json")  # garbage must not kill the connection
        ws1.send(json.dumps({"type": "ping"}))
        assert recv_json(ws1)["type"] == "pong"
        assert bridge.client_count == 2


def test_landmark_streaming_from_a_processed_frame():
    import numpy as np

    cfg = HCIConfig()
    cfg.screen.width, cfg.screen.height = 1920, 1080
    engine = HCIEngine(
        cfg,
        sensors=[HandSensor(backend=FakeHandBackend([((640, 360), (640, 460))] * 3))],
        actions=[WebSocketBridge(port=0, stream_landmarks=True, max_stream_hz=None)],
        raise_errors=True,
    )
    engine.setup()
    try:
        bridge = engine.get(WebSocketBridge)
        with client(bridge) as (ws, hello):
            assert hello["landmarks"] is True
            engine.process_frame(np.zeros((720, 1280, 3), np.uint8))
            frame = None
            for _ in range(5):  # hand_detected / cursor events may arrive first
                m = recv_json(ws)
                if m["type"] == "frame":
                    frame = m
                    break
            assert frame is not None
            assert frame["frame"] == 1 and frame["width"] == 1280 and frame["pinch"] in ("DISCONNECTED", "NO HAND DETECTED", "OPEN", "HOLDING CLICK")
            hand = frame["hands"][0]
            assert hand["handedness"] == "Right" and len(hand["landmarks"]) == 21 and len(hand["landmarks"][0]) == 3
            assert hand["landmarks"][8][0] == pytest.approx(640.5 / 1280, abs=1e-3)
            assert frame["cursor"]["active"] is True
    finally:
        engine.close()


def test_stream_rate_is_throttled(bare_engine):
    import numpy as np

    bridge = bare_engine.add_action(WebSocketBridge(port=0, stream_landmarks=True, max_stream_hz=1.0))
    bare_engine.setup()
    try:
        with client(bridge) as (ws, _):
            for _ in range(5):
                bare_engine.process_frame(np.zeros((72, 128, 3), np.uint8))
            assert recv_json(ws)["type"] == "frame"
            with pytest.raises(TimeoutError):
                ws.recv(timeout=0.3)  # the other 4 frames were dropped (1 Hz cap)
    finally:
        bare_engine.close()


def test_wildcard_forwards_everything(bare_engine):
    bridge = bare_engine.add_action(WebSocketBridge(port=0, events=["*"]))
    bare_engine.setup()
    try:
        with client(bridge) as (ws, _):
            bare_engine.emit("some_custom_event", answer=42)
            m = recv_json(ws)
            assert m["name"] == "some_custom_event" and m["data"] == {"answer": 42}
    finally:
        bare_engine.close()


def test_port_in_use_is_reported_clearly(bridge_engine, bare_engine):
    _, bridge = bridge_engine
    from open_ai_hci import HCIEngine

    other = HCIEngine(HCIConfig(), sensors=[], processors=[], actions=[WebSocketBridge(port=bridge.port)])
    with pytest.raises(RuntimeError, match="Could not start the WebSocket server"):
        other.setup()


def test_close_stops_the_server_thread(bare_engine):
    bridge = bare_engine.add_action(WebSocketBridge(port=0))
    bare_engine.setup()
    thread = bridge._thread
    bare_engine.close()
    assert not thread.is_alive()


def test_missing_websockets_gives_an_install_hint(bare_engine, monkeypatch):
    import open_ai_hci.actions.websocket_bridge as wb

    def fake_require(module, feature):
        raise MissingOptionalDependency(f"{feature} requires 'websockets'. Install it with: pip install \"open-ai-hci[ai]\"")

    monkeypatch.setattr(wb, "require", fake_require)
    bare_engine.add_action(WebSocketBridge(port=0))
    with pytest.raises(MissingOptionalDependency, match=r"open-ai-hci\[ai\]"):
        bare_engine.setup()

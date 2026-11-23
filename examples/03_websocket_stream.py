"""Example 3: stream HCI events and landmarks to external apps (Unity, Unreal, Godot, web pages).

Server:  python examples/03_websocket_stream.py              (ws://127.0.0.1:8765)
         python examples/03_websocket_stream.py --hud        (with the preview window)
Client:  python examples/03_websocket_stream.py --client     (prints what any client would receive)

Requires:  pip install "open-ai-hci[ai]"   (installs `websockets`)

Message formats are documented in open_ai_hci/actions/websocket_bridge.py. In a browser:

    const ws = new WebSocket("ws://127.0.0.1:8765");
    ws.onmessage = e => { const m = JSON.parse(e.data);
                          if (m.type === "event") console.log(m.name, m.data);
                          if (m.type === "frame") draw(m.hands[0]?.landmarks); };

In Unity use any WebSocket client (e.g. NativeWebSocket) and parse the same JSON.
"""
import json
import sys

from open_ai_hci import HCIConfig, HCIEngine, MouseConfig, WebSocketBridge
from open_ai_hci.actions.websocket_bridge import DEFAULT_EVENTS


def run_server() -> None:
    config = HCIConfig(mouse=MouseConfig(enabled=False))        # don't move the OS cursor, just stream
    hci = HCIEngine(config)
    bridge = WebSocketBridge(
        host="127.0.0.1",          # local only; the bridge has no authentication
        port=8765,
        events=[*DEFAULT_EVENTS, "cursor_move"],   # add the smoothed cursor stream to the defaults
        stream_landmarks=True,     # per-frame 21-point hand landmarks, throttled to max_stream_hz
        max_stream_hz=30,
    )
    hci.add_action(bridge)
    hci.on("pinch_start", lambda e: print(f"pinch_start -> {bridge.client_count} client(s) notified"))
    print(f"Streaming on {bridge.url}. Press Ctrl+C{' or q' if '--hud' in sys.argv else ''} to stop.")
    hci.run(show_hud="--hud" in sys.argv)


def run_client(url: str = "ws://127.0.0.1:8765") -> None:
    from websockets.sync.client import connect

    with connect(url) as ws:
        print("connected;", json.loads(ws.recv()))
        ws.send(json.dumps({"type": "subscribe", "events": ["pinch_start", "pinch_end", "head_flick", "frame"]}))
        for raw in ws:
            msg = json.loads(raw)
            if msg["type"] == "frame":
                hand = msg["hands"][0] if msg["hands"] else None
                idx = hand["landmarks"][8][:2] if hand else None
                print(f"frame {msg['frame']:>5}  index tip (normalised): {idx}  pinch: {msg['pinch']}")
            else:
                print(msg["type"], msg.get("name"), msg.get("data"))


if __name__ == "__main__":
    run_client() if "--client" in sys.argv else run_server()

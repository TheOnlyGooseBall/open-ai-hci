"""Stream HCI events and landmarks as JSON over WebSocket (Unity, Unreal, Godot, browsers, ...).

Requires ``pip install "open-ai-hci[ai]"`` (the ``websockets`` package, >= 13). The core
framework never imports it unless a :class:`WebSocketBridge` is actually used.

Wire protocol (server -> client, one JSON object per text message)
-----------------------------------------------------------------
``{"type": "hello", "version": "0.2.0", "events": [...], "landmarks": false}``
    Sent once on connect.

``{"type": "event", "name": "pinch_start", "timestamp": 1730000000.12, "frame": 42,
"source": "gesture_rules", "data": {...}}``
    Every forwarded event. ``data`` is JSON-safe (arrays become lists, floats are rounded).

``{"type": "frame", "frame": 42, "width": 1280, "height": 720, "timestamp": ..., "fps": 29.8,
"pinch": "OPEN", "cursor": {"x": 100.0, "y": 200.0}, "hands": [{"handedness": "Right",
"score": 0.98, "landmarks": [[x, y, z] x 21]}], "faces": [{"nose": [640, 360], ...}]}``
    Per-frame landmark stream (only with ``stream_landmarks=True``). Landmarks are normalised (0..1).

Client -> server (optional)
---------------------------
``{"type": "subscribe", "events": ["pinch_start", "pinch_end"]}``  limit this client to those events
(``["*"]`` = all). ``{"type": "ping"}`` is answered with ``{"type": "pong"}``.

Minimal browser client::

    const ws = new WebSocket("ws://127.0.0.1:8765");
    ws.onmessage = (m) => { const msg = JSON.parse(m.data); if (msg.type === "event") console.log(msg.name, msg.data); };
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Iterable
from typing import Any

from .._optional import require
from .._version import __version__
from ..event_bus import Event, EventCallback
from ..types import FrameContext, jsonable
from .base import BaseAction

__all__ = ["WebSocketBridge", "DEFAULT_EVENTS"]

logger = logging.getLogger("open_ai_hci")

#: Events forwarded by default (everything meaningful except the high-rate ``cursor_move``/``frame``).
DEFAULT_EVENTS: tuple[str, ...] = (
    "hand_detected", "hand_lost", "face_detected", "face_lost", "pose_detected", "pose_lost",
    "pinch_start", "pinch_end", "drag_start", "drag_end",
    "head_flick", "head_flick_left", "head_flick_right", "blink",
    "scroll_start", "scroll", "scroll_end",
    "gesture", "gesture_end", "voice_command", "intent", "vlm_result", "alert",
)


class WebSocketBridge(BaseAction):
    """WebSocket server that broadcasts events (and optionally landmarks) to any number of clients.

    Args:
        host: Bind address. Default ``127.0.0.1`` (local only). Use ``0.0.0.0`` deliberately, since
            there is **no authentication**: anyone who can reach the port receives your tracking data.
        port: TCP port (``0`` = pick a free one; read it back from :attr:`port`).
        events: Event names to forward. Default :data:`DEFAULT_EVENTS`. Include ``"cursor_move"``
            for the smoothed cursor, or ``"*"`` for everything except ``frame``.
        stream_landmarks: Also send a ``frame`` message per processed frame.
        include_faces: Include face summaries in ``frame`` messages (the 468 face landmarks are never sent).
        max_stream_hz: Upper bound for ``frame`` messages per second (``None`` = every processed frame).
        precision: Decimal places for floats.
    """

    name = "websocket_bridge"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        *,
        events: Iterable[str] | None = None,
        stream_landmarks: bool = False,
        include_faces: bool = True,
        max_stream_hz: float | None = 30.0,
        precision: int = 4,
    ) -> None:
        super().__init__()
        self.host = host
        self._requested_port = port
        self.events = list(events) if events is not None else list(DEFAULT_EVENTS)
        self.stream_landmarks = stream_landmarks
        self.include_faces = include_faces
        self.max_stream_hz = max_stream_hz
        self.precision = precision
        self._port: int | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._stop_future: asyncio.Future[None] | None = None
        self._clients: dict[Any, set[str] | None] = {}  # loop-thread only
        self._last_stream = 0.0

    # ------------------------------------------------------------------ properties
    @property
    def port(self) -> int:
        """The actual listening port (valid after setup)."""
        if self._port is None:
            return self._requested_port
        return self._port

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # ------------------------------------------------------------------ lifecycle
    def setup(self) -> None:
        require("websockets", "WebSocketBridge")
        self._thread = threading.Thread(target=self._serve_forever, name="open-ai-hci-ws", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            raise RuntimeError("WebSocket server did not start within 10 s")
        if self._startup_error is not None:
            raise RuntimeError(f"Could not start the WebSocket server on {self.host}:{self._requested_port}: "
                               f"{self._startup_error}") from self._startup_error
        super().setup()
        logger.info("WebSocket bridge listening on %s", self.url)

    def close(self) -> None:
        super().close()
        loop = self._loop
        if loop is not None and loop.is_running() and self._stop_future is not None:
            loop.call_soon_threadsafe(self._set_stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _set_stop(self) -> None:
        if self._stop_future is not None and not self._stop_future.done():
            self._stop_future.set_result(None)

    def subscriptions(self) -> dict[str, EventCallback]:
        names = set(self.events)
        if "*" in names:
            subs: dict[str, EventCallback] = {"*": self._on_event}
        else:
            subs = {n: self._on_event for n in names}
        if self.stream_landmarks:
            subs["frame"] = self._on_frame
        return subs

    # --------------------------------------------------------------------- server
    def _serve_forever(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            loop.run_until_complete(self._main())
        except BaseException as exc:  # startup failure is reported to setup()
            if not self._ready.is_set():
                self._startup_error = exc
                self._ready.set()
            else:  # pragma: no cover - unexpected runtime failure
                logger.exception("WebSocket server crashed")
        finally:
            loop.close()

    async def _main(self) -> None:
        from websockets.asyncio.server import serve  # imported here: only when the bridge is used

        self._stop_future = asyncio.get_running_loop().create_future()
        try:
            server_cm = serve(self._handler, self.host, self._requested_port)
            async with server_cm as server:
                self._port = int(server.sockets[0].getsockname()[1])
                self._ready.set()
                await self._stop_future
        except BaseException as exc:
            if not self._ready.is_set():
                self._startup_error = exc
                self._ready.set()
                return
            raise

    async def _handler(self, connection: Any) -> None:
        self._clients[connection] = None  # None = receive every forwarded event
        try:
            await connection.send(
                json.dumps(
                    {"type": "hello", "version": __version__, "events": self.events, "landmarks": self.stream_landmarks}
                )
            )
            async for raw in connection:
                self._handle_client_message(connection, raw)
        except Exception:  # client vanished / protocol error: just drop it
            logger.debug("WebSocket client dropped", exc_info=True)
        finally:
            self._clients.pop(connection, None)

    def _handle_client_message(self, connection: Any, raw: Any) -> None:
        try:
            msg = json.loads(raw)
        except (TypeError, ValueError):
            return
        if not isinstance(msg, dict):
            return
        kind = msg.get("type")
        if kind == "subscribe":
            wanted = msg.get("events")
            if isinstance(wanted, list):
                names = {str(n) for n in wanted}
                self._clients[connection] = None if "*" in names else names
        elif kind == "ping":
            asyncio.ensure_future(connection.send(json.dumps({"type": "pong"})))

    # ------------------------------------------------------------------ broadcasting
    def _on_event(self, event: Event) -> None:
        payload = {"type": "event", **event.to_dict()}
        payload["data"] = jsonable(event.data, self.precision)
        self._enqueue(json.dumps(payload, separators=(",", ":")), event.name)

    def _on_frame(self, event: Event) -> None:
        if self.max_stream_hz:
            now = event.timestamp
            if now - self._last_stream < 1.0 / self.max_stream_hz:
                return
            self._last_stream = now
        ctx: FrameContext = event.ctx
        message: dict[str, Any] = {
            "type": "frame",
            "frame": ctx.frame_index,
            "width": ctx.width,
            "height": ctx.height,
            "timestamp": round(event.timestamp, 6),
            "fps": event.get("fps"),
            "pinch": ctx.pinch_status,
            "hands": [
                {"handedness": h.handedness, "score": h.score, "landmarks": h.landmarks} for h in ctx.hands
            ],
            "cursor": None
            if ctx.cursor is None
            else {"x": ctx.cursor.x, "y": ctx.cursor.y, "active": ctx.cursor.active},
        }
        if self.include_faces:
            message["faces"] = [
                {"nose": f.nose, "ear": f.ear, "head_pose": f.head_pose} for f in ctx.faces
            ]
        if ctx.poses:
            message["poses"] = [{"landmarks": p.landmarks, "visibility": p.visibility} for p in ctx.poses]
        if ctx.gesture:
            message["gesture"] = ctx.gesture
        self._enqueue(json.dumps(jsonable(message, self.precision), separators=(",", ":")), "frame")

    def _enqueue(self, text: str, event_name: str) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        try:
            loop.call_soon_threadsafe(self._broadcast, text, event_name)
        except RuntimeError:  # loop closed during shutdown
            pass

    def _broadcast(self, text: str, event_name: str) -> None:
        """Runs on the server loop thread."""
        from websockets.asyncio.server import broadcast

        targets = [c for c, wanted in self._clients.items() if wanted is None or event_name in wanted]
        if targets:
            broadcast(targets, text)

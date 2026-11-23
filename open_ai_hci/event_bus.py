"""Typed publish/subscribe event system.

Usage::

    from open_ai_hci import HCIEngine

    hci = HCIEngine()

    @hci.on("pinch_start")
    def handle(event):
        print(event.x, event.y)      # data keys are readable as attributes

    hci.on("head_flick", lambda e: print(e.direction))
    hci.on("*", print)               # wildcard: every event except the per-frame "frame"

Handlers run synchronously on the thread that emitted the event (normally the
engine loop), so keep them short. Exceptions inside handlers are logged and
never crash the engine (set ``raise_errors=True`` while testing to change that).
"""
from __future__ import annotations

import difflib
import logging
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union

from .types import jsonable

__all__ = ["Event", "EventBus", "EventCallback", "EventType", "KNOWN_EVENTS", "QUIET_EVENTS"]

logger = logging.getLogger("open_ai_hci")


class EventType(str, Enum):
    """Built-in event names. Plain strings work everywhere these do."""

    # sensors
    HAND_DETECTED = "hand_detected"
    HAND_LOST = "hand_lost"
    FACE_DETECTED = "face_detected"
    FACE_LOST = "face_lost"
    POSE_DETECTED = "pose_detected"
    POSE_LOST = "pose_lost"
    # gesture rules
    CURSOR_MOVE = "cursor_move"
    PINCH_START = "pinch_start"
    PINCH_END = "pinch_end"
    RIGHT_PINCH_START = "right_pinch_start"
    RIGHT_PINCH_END = "right_pinch_end"
    RIGHT_CLICK = "right_click"
    CONTROL_PAUSED = "control_paused"
    CONTROL_RESUMED = "control_resumed"
    DRAG_START = "drag_start"
    DRAG_MOVE = "drag_move"
    DRAG_END = "drag_end"
    HEAD_FLICK = "head_flick"
    HEAD_FLICK_LEFT = "head_flick_left"
    HEAD_FLICK_RIGHT = "head_flick_right"
    BLINK = "blink"
    SCROLL_START = "scroll_start"
    SCROLL = "scroll"
    SCROLL_END = "scroll_end"
    # AI plugins
    GESTURE = "gesture"
    GESTURE_END = "gesture_end"
    VOICE_COMMAND = "voice_command"
    INTENT = "intent"
    VLM_RESULT = "vlm_result"
    LLM_RESPONSE = "llm_response"
    LLM_ERROR = "llm_error"
    # engine
    ALERT = "alert"
    FRAME = "frame"
    KEY_PRESS = "key_press"
    ENGINE_START = "engine_start"
    ENGINE_STOP = "engine_stop"
    ERROR = "error"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


KNOWN_EVENTS: set[str] = {e.value for e in EventType}
WILDCARD = "*"
#: Per-frame events that wildcard (``"*"``) subscribers do *not* receive, because they
#: would flood a catch-all handler. Subscribe to them by name.
QUIET_EVENTS: frozenset[str] = frozenset({"frame"})


@dataclass
class Event:
    """A single occurrence published on the bus.

    Attributes:
        name: Event name, e.g. ``"pinch_start"``.
        data: Payload. Keys are also readable as attributes (``event.x``).
        timestamp: Wall-clock seconds (``time.time()``) when it was created.
        frame_index: Camera frame counter at emission time (``-1`` if unknown).
        source: Name of the component that emitted it.
    """

    name: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    frame_index: int = -1
    source: str = ""

    #: Payload keys that would be shadowed by the fields above; rejected at creation.
    RESERVED = frozenset({"name", "data", "timestamp", "frame_index", "source"})

    def __post_init__(self) -> None:
        clash = self.RESERVED.intersection(self.data)
        if clash:
            raise ValueError(
                f"Event payload key(s) {sorted(clash)} clash with Event attributes; "
                f"choose other names (e.g. 'gesture' instead of 'name', 'origin' instead of 'source')."
            )

    def __getattr__(self, item: str) -> Any:
        # Only called when normal lookup fails -> expose payload keys as attributes.
        data = self.__dict__.get("data")
        if data is not None and item in data:
            return data[item]
        raise AttributeError(f"{type(self).__name__!r} has no attribute or data key {item!r}")

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style payload access with a default."""
        return self.data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "name": self.name,
            "timestamp": round(self.timestamp, 6),
            "frame": self.frame_index,
            "source": self.source,
            "data": jsonable(self.data),
        }


EventCallback = Callable[[Event], Any]
EventName = Union[str, EventType]  # noqa: UP007 - runtime alias, keep 3.10-safe


def _norm(name: EventName) -> str:
    return name.value if isinstance(name, EventType) else str(name)


@dataclass
class _Listener:
    callback: EventCallback
    once: bool = False


class EventBus:
    """Thread-safe pub/sub hub.

    Args:
        raise_errors: Re-raise exceptions from handlers instead of logging them.
        warn_on_typos: Warn when subscribing to a name that is *almost* (but not
            exactly) a built-in event, e.g. ``"pinch_strat"``.
    """

    def __init__(self, *, raise_errors: bool = False, warn_on_typos: bool = True) -> None:
        self._listeners: dict[str, list[_Listener]] = {}
        self._lock = threading.RLock()
        self._known: set[str] = set(KNOWN_EVENTS)
        self.raise_errors = raise_errors
        self.warn_on_typos = warn_on_typos
        self.frame_index: int = -1  # stamped onto events by the engine

    # ------------------------------------------------------------------ subscribe
    def on(self, name: EventName, callback: EventCallback | None = None, *, once: bool = False):
        """Subscribe to ``name`` (``"*"`` = every event).

        Works as a plain call (returns the callback) or as a decorator::

            bus.on("pinch_start", handler)

            @bus.on("pinch_start")
            def handler(event): ...
        """
        key = _norm(name)
        self._check_typo(key)

        def register(cb: EventCallback) -> EventCallback:
            with self._lock:
                self._listeners.setdefault(key, []).append(_Listener(cb, once))
            return cb

        if callback is None:
            return register
        return register(callback)

    def once(self, name: EventName, callback: EventCallback | None = None):
        """Like :meth:`on`, but the handler is removed after its first call."""
        return self.on(name, callback, once=True)

    def off(self, name: EventName, callback: EventCallback | None = None) -> int:
        """Unsubscribe ``callback`` (or every handler if ``None``). Returns count removed."""
        key = _norm(name)
        with self._lock:
            current = self._listeners.get(key, [])
            if callback is None:
                removed = len(current)
                self._listeners.pop(key, None)
                return removed
            kept = [item for item in current if item.callback != callback]  # == so bound methods match
            removed = len(current) - len(kept)
            if kept:
                self._listeners[key] = kept
            else:
                self._listeners.pop(key, None)
            return removed

    def clear(self) -> None:
        """Remove every handler."""
        with self._lock:
            self._listeners.clear()

    def register_event(self, name: EventName) -> None:
        """Declare a custom event name (silences the typo detector for it)."""
        self._known.add(_norm(name))

    # ----------------------------------------------------------------------- emit
    def emit(self, event: Event | EventName, /, **data: Any) -> Event:
        """Publish an event. Pass an :class:`Event` or a name plus payload keywords."""
        if not isinstance(event, Event):
            event = Event(name=_norm(event), data=data, frame_index=self.frame_index)
        elif data:
            event.data.update(data)
            event.__post_init__()
        self._known.add(event.name)

        with self._lock:
            targets = list(self._listeners.get(event.name, ()))
            if event.name not in QUIET_EVENTS:
                targets += self._listeners.get(WILDCARD, ())
            for listener in targets:
                if listener.once:
                    self._remove_listener(event.name, listener)
                    self._remove_listener(WILDCARD, listener)

        for listener in targets:
            try:
                listener.callback(event)
            except Exception:
                if self.raise_errors:
                    raise
                logger.exception(
                    "Error in handler %r for event %r", getattr(listener.callback, "__name__", listener.callback), event.name
                )
        return event

    # ------------------------------------------------------------------ inspection
    def listener_count(self, name: EventName) -> int:
        """Number of handlers that would receive ``name`` (including wildcard)."""
        key = _norm(name)
        with self._lock:
            return len(self._listeners.get(key, ())) + (0 if key == WILDCARD else len(self._listeners.get(WILDCARD, ())))

    def has_listeners(self, name: EventName) -> bool:
        return self.listener_count(name) > 0

    def has_subscribers(self, name: EventName) -> bool:
        """``True`` if something subscribed to exactly ``name`` (wildcards are ignored)."""
        with self._lock:
            return bool(self._listeners.get(_norm(name)))

    # --------------------------------------------------------------------- helpers
    def _remove_listener(self, key: str, listener: _Listener) -> None:
        lst = self._listeners.get(key)
        if lst and listener in lst:
            lst.remove(listener)
            if not lst:
                self._listeners.pop(key, None)

    def _check_typo(self, key: str) -> None:
        if not self.warn_on_typos or key == WILDCARD or key in self._known:
            return
        close = difflib.get_close_matches(key, sorted(self._known), n=1, cutoff=0.8)
        if close:
            warnings.warn(
                f"Subscribed to unknown event {key!r}. Did you mean {close[0]!r}? "
                f"(Use bus.register_event({key!r}) if it is a custom event.)",
                stacklevel=3,
            )

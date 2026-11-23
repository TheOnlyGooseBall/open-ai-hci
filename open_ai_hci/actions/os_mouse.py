"""OS mouse control through PyAutoGUI."""
from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import Any

from ..config import MouseConfig
from ..event_bus import Event, EventCallback
from ._pyautogui import load_pyautogui
from .base import BaseAction

__all__ = ["OSMouseAction"]

logger = logging.getLogger("open_ai_hci")


class OSMouseAction(BaseAction):
    """Moves the real cursor, clicks, drags and scrolls in response to gesture events.

    Behaviour (identical to the prototype by default):

    ===================  ==============================================
    ``cursor_move``      ``moveTo(int(x), int(y))``
    ``pinch_start``      ``mouseDown(button="left")``
    ``pinch_end``        ``mouseUp(button="left")`` (also when the hand is lost)
    ``right_click``      ``click(button="right")`` (thumb + middle pinch, or the AI right-click gesture)
    ``head_flick``       ``click(button="right")`` only if ``MouseConfig.right_click_on_flick`` (v0.1 behaviour)
    ===================  ==============================================

    Additional: ``scroll`` (mouse-wheel notches) and ``intent`` events from the voice / VLM / LLM
    agents (``click``, ``double_click``, ``right_click``, ``scroll_up``, ``scroll_down`` in notches,
    ``scroll`` in LLM units where ~100 is one notch and positive is up, ``move`` to screen pixels).
    Wheel amounts are converted to the platform's units (Windows expects 120 per notch).

    Safety:
        * PyAutoGUI's **failsafe** is on by default: slam the *physical* mouse into a screen
          corner to abort. The action then releases any held button, shows an alert and
          stops the engine.
        * Targets are clamped to ``clamp_margin`` px inside the screen so a hand at the edge
          of the camera view cannot trip the failsafe by accident.
        * A held button is always released on shutdown.

    Args:
        config: Overrides ``engine.config.mouse``.
        clamp_margin: Pixels kept free at every screen edge (``0`` = no clamping).
        drag_events: Use drag motion events while the button is held. Needed on macOS
            for drags to register; default is on for macOS only.
        wheel_unit: PyAutoGUI scroll units per wheel notch. Default: 120 on Windows, 1 elsewhere.
        pyautogui_module: Inject a PyAutoGUI-compatible object (tests, other backends).
    """

    name = "os_mouse"

    def __init__(
        self,
        config: MouseConfig | None = None,
        *,
        clamp_margin: int = 2,
        drag_events: bool | None = None,
        wheel_unit: int | None = None,
        pyautogui_module: Any = None,
    ) -> None:
        super().__init__()
        self._cfg = config
        self.clamp_margin = clamp_margin
        self.drag_events = sys.platform == "darwin" if drag_events is None else drag_events
        self.wheel_unit = (120 if sys.platform == "win32" else 1) if wheel_unit is None else wheel_unit
        self._pag: Any = pyautogui_module
        self._mouse_down = False
        self._screen: tuple[int, int] = (0, 0)

    @property
    def cfg(self) -> MouseConfig:
        return self._cfg if self._cfg is not None else self.config.mouse

    @property
    def is_mouse_down(self) -> bool:
        return self._mouse_down

    # ------------------------------------------------------------------ lifecycle
    def setup(self) -> None:
        if self._pag is None:
            self._pag = load_pyautogui(self.cfg.failsafe, self.cfg.pause)
        else:
            self._pag.FAILSAFE = self.cfg.failsafe
            self._pag.PAUSE = self.cfg.pause
        size = self._pag.size()
        self._screen = (int(size[0]), int(size[1]))
        super().setup()

    def close(self) -> None:
        super().close()
        self._release()

    def subscriptions(self) -> dict[str, EventCallback]:
        subs: dict[str, EventCallback] = {"scroll": self._on_scroll}
        if self.cfg.move_cursor:
            subs["cursor_move"] = self._on_move
        if self.cfg.left_click_on_pinch:
            subs["pinch_start"] = self._on_pinch_start
            subs["pinch_end"] = self._on_pinch_end
        if self.cfg.right_click_on_pinch:
            subs["right_click"] = self._on_right_click
        if self.cfg.right_click_on_flick:
            subs["head_flick"] = self._on_flick
        if self.cfg.handle_intents:
            subs["intent"] = self._on_intent
        return subs

    def reconfigure(self) -> None:
        """Apply changed :class:`MouseConfig` values to a running action (re-subscribes, updates PyAutoGUI)."""
        if self._pag is None or self.engine is None:
            return
        self._pag.FAILSAFE = self.cfg.failsafe
        self._pag.PAUSE = self.cfg.pause
        for event_name, handler in self._subscribed:
            self.bus.off(event_name, handler)
        self._subscribed.clear()
        for event_name, handler in self.subscriptions().items():
            self.bus.on(event_name, handler)
            self._subscribed.append((event_name, handler))
        if not self.cfg.left_click_on_pinch:
            self._release()

    # -------------------------------------------------------------------- helpers
    def _safe(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
        """Run a PyAutoGUI call; on failsafe release the button and stop the engine."""
        try:
            fn(*args, **kwargs)
            return True
        except self._pag.FailSafeException:
            self._release()  # the cursor is parked in a corner; the button must not stay held
            self.emit("alert", text="FAILSAFE: mouse in corner, stopping")
            logger.warning("PyAutoGUI failsafe triggered; stopping the engine")
            if self.engine is not None:
                self.engine.stop()
            return False

    def _release(self) -> None:
        """Release the left button if we hold it.

        PyAutoGUI refuses *every* call while the cursor is in a failsafe corner, including this
        one, so the failsafe is suspended for the duration of the release only.
        """
        if not self._mouse_down or self._pag is None:
            return
        self._mouse_down = False
        previous = self._pag.FAILSAFE
        self._pag.FAILSAFE = False
        try:
            self._pag.mouseUp(button="left")
        except Exception:  # pragma: no cover - best effort during shutdown
            logger.debug("Could not release the mouse button", exc_info=True)
        finally:
            self._pag.FAILSAFE = previous

    def _clamp(self, x: float, y: float) -> tuple[int, int]:
        ix, iy = int(x), int(y)
        m, (w, h) = self.clamp_margin, self._screen
        if m > 0 and w > 2 * m and h > 2 * m:
            ix = min(max(ix, m), w - 1 - m)
            iy = min(max(iy, m), h - 1 - m)
        return ix, iy

    # ------------------------------------------------------------------- handlers
    def _on_move(self, event: Event) -> None:
        x, y = self._clamp(event.x, event.y)
        if self._mouse_down and self.drag_events:
            try:
                self._safe(self._pag.dragTo, x, y, button="left", mouseDownUp=False)
                return
            except TypeError:  # old PyAutoGUI without mouseDownUp
                self.drag_events = False
        self._safe(self._pag.moveTo, x, y)

    def _on_pinch_start(self, event: Event) -> None:
        if not self._mouse_down and self._safe(self._pag.mouseDown, button="left"):
            self._mouse_down = True

    def _on_pinch_end(self, event: Event) -> None:
        if self._mouse_down:
            self._mouse_down = False
            self._safe(self._pag.mouseUp, button="left")

    def _on_flick(self, event: Event) -> None:
        if self._safe(self._pag.click, button="right"):
            self.emit("alert", text="triggered right click!")

    def _on_right_click(self, event: Event) -> None:
        if self._safe(self._pag.click, button="right"):
            self.emit("alert", text="right click")

    def _wheel(self, notches: float) -> int:
        """Mouse-wheel notches -> PyAutoGUI units (never rounds a real request down to nothing)."""
        value = int(round(notches * self.wheel_unit))
        if value == 0 and notches != 0:
            value = 1 if notches > 0 else -1
        return value

    def _on_scroll(self, event: Event) -> None:
        self._safe(self._pag.scroll, self._wheel(float(event.amount)))

    def _on_intent(self, event: Event) -> None:
        if event.get("dry_run"):
            return
        intent, args = event.intent, event.args
        label = intent.replace("_", " ")
        if intent == "click":
            self._safe(self._pag.click, button="left")
        elif intent == "double_click":
            self._safe(self._pag.doubleClick, button="left")
        elif intent == "right_click":
            self._safe(self._pag.click, button="right")
        elif intent in ("scroll_up", "scroll_down"):
            amount = _number(args.get("amount"), 5.0)
            self._safe(self._pag.scroll, self._wheel(amount if intent == "scroll_up" else -amount))
        elif intent == "scroll":
            units = self._wheel(_number(args.get("amount"), 0.0) / 100.0)
            if units == 0:
                return
            self._safe(self._pag.scroll, units)
        elif intent == "move":
            if "x" in args and "y" in args:
                x, y = self._clamp(_number(args["x"], 0.0), _number(args["y"], 0.0))
                self._safe(self._pag.moveTo, x, y)
            elif "dx" in args or "dy" in args:
                self._safe(self._pag.moveRel, int(_number(args.get("dx"), 0.0)), int(_number(args.get("dy"), 0.0)))
            else:
                return
        else:
            return  # not a mouse intent (the keyboard action or your own handler may take it)
        self.emit("alert", text=f"{event.get('origin') or 'voice'}: {label}")


def _number(value: Any, default: float) -> float:
    """Best-effort float (LLMs sometimes answer ``"-500"`` or ``null``)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

"""Hotkey and macro execution bound to events."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Union

from ..event_bus import Event, EventCallback
from ._pyautogui import load_pyautogui
from .base import BaseAction

__all__ = ["OSKeyboardAction", "parse_hotkey", "Step", "Binding"]

logger = logging.getLogger("open_ai_hci")

#: One macro step: ``("hotkey", "ctrl", "c")``, ``("press", "enter")``, ``("write", "text")``, ``("wait", 0.2)``.
Step = tuple
#: What an event can be bound to.
Binding = Union[str, Sequence[Step], Callable[[Event], Any]]  # noqa: UP007 - runtime alias, keep 3.10-safe


def parse_hotkey(spec: str | Sequence[str]) -> tuple[str, ...]:
    """``"ctrl+shift+t"`` -> ``("ctrl", "shift", "t")``; sequences pass through (lower-cased)."""
    parts = spec.split("+") if isinstance(spec, str) else list(spec)
    keys = tuple(p.strip().lower() for p in parts if p.strip())
    if not keys:
        raise ValueError(f"Empty hotkey: {spec!r}")
    return keys


class OSKeyboardAction(BaseAction):
    """Runs hotkeys and macros when events fire.

    ``bindings`` maps an event name to one of:

    * a hotkey string: ``"ctrl+c"``;
    * a macro: a list of steps, e.g. ``[("hotkey", "ctrl", "a"), ("wait", 0.1), ("write", "hello"), ("press", "enter")]``;
    * a Python callable ``f(event)`` (runs your own code; no PyAutoGUI needed).

    Example::

        hci.add_action(OSKeyboardAction({
            "head_flick_left":  "alt+left",    # browser back
            "head_flick_right": "alt+right",   # browser forward
            "blink":            [("press", "space")],
        }, cooldown_s=0.5))

    It also executes ``hotkey`` / ``press`` / ``type`` intents from the multimodal agent
    (``handle_intents=True``).

    Args:
        bindings: Event name -> binding (see above).
        cooldown_s: Minimum seconds between two firings of the *same* binding.
        dry_run: Log what would be pressed without touching the keyboard.
        handle_intents: React to ``intent`` events named ``hotkey`` / ``press`` / ``type``.
        pyautogui_module: Inject a PyAutoGUI-compatible object (tests).
    """

    name = "os_keyboard"

    def __init__(
        self,
        bindings: Mapping[str, Binding] | None = None,
        *,
        cooldown_s: float = 0.0,
        dry_run: bool = False,
        handle_intents: bool = True,
        pyautogui_module: Any = None,
    ) -> None:
        super().__init__()
        self.bindings: dict[str, Binding] = dict(bindings or {})
        self.cooldown_s = cooldown_s
        self.dry_run = dry_run
        self.handle_intents = handle_intents
        self._pag: Any = pyautogui_module
        self._last_fired: dict[str, float] = {}
        for spec in self.bindings.values():  # fail early on malformed bindings
            self._validate(spec)

    # ------------------------------------------------------------------ lifecycle
    def setup(self) -> None:
        needs_pag = not self.dry_run and (
            self.handle_intents or any(not callable(b) for b in self.bindings.values())
        )
        if needs_pag and self._pag is None:
            self._pag = load_pyautogui(self.config.mouse.failsafe, self.config.mouse.pause)
        super().setup()

    def subscriptions(self) -> dict[str, EventCallback]:
        subs: dict[str, EventCallback] = {}
        for event_name in self.bindings:
            subs[event_name] = self._make_handler(event_name)
        if self.handle_intents:
            subs.setdefault("intent", self._on_intent)
        return subs

    def bind(self, event_name: str, binding: Binding) -> None:
        """Add or replace a binding (before the engine starts)."""
        self._validate(binding)
        self.bindings[event_name] = binding

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _validate(binding: Binding) -> None:
        if callable(binding):
            return
        if isinstance(binding, str):
            parse_hotkey(binding)
            return
        for step in binding:
            if not step or step[0] not in ("hotkey", "press", "write", "wait"):
                raise ValueError(f"Invalid macro step {step!r}; use hotkey/press/write/wait")

    def _make_handler(self, event_name: str) -> EventCallback:
        def handler(event: Event) -> None:
            now = time.monotonic()
            if now - self._last_fired.get(event_name, -1e9) < self.cooldown_s:
                return
            self._last_fired[event_name] = now
            self._run(self.bindings[event_name], event)

        return handler

    def _run(self, binding: Binding, event: Event) -> None:
        if callable(binding):
            binding(event)
        elif isinstance(binding, str):
            self._hotkey(parse_hotkey(binding))
        else:
            for step in binding:
                kind, *args = step
                if kind == "hotkey":
                    self._hotkey(parse_hotkey(args))
                elif kind == "press":
                    self._press(str(args[0]))
                elif kind == "write":
                    self._write(str(args[0]))
                elif kind == "wait":
                    time.sleep(float(args[0]))
        self.emit("alert", text=f"key action: {event.name}")

    def _hotkey(self, keys: tuple[str, ...]) -> None:
        if self.dry_run:
            logger.info("[dry-run] hotkey %s", "+".join(keys))
        else:
            self._pag.hotkey(*keys)

    def _press(self, key: str) -> None:
        if self.dry_run:
            logger.info("[dry-run] press %s", key)
        else:
            self._pag.press(key)

    def _write(self, text: str) -> None:
        if self.dry_run:
            logger.info("[dry-run] write %r", text)
        else:
            self._pag.write(text, interval=0.0)

    def _on_intent(self, event: Event) -> None:
        if event.get("dry_run"):
            return
        intent, args = event.intent, event.args
        try:
            if intent == "hotkey":
                self._hotkey(parse_hotkey(args["keys"]))
            elif intent == "press":
                self._press(str(args["key"]))
            elif intent == "type":
                self._write(str(args["text"]))
            else:
                return
        except (KeyError, ValueError) as exc:  # a malformed LLM / voice intent must never crash the loop
            logger.warning("Ignoring malformed %r intent %r: %s", intent, args, exc)
            return
        shown = args.get("keys") or args.get("key") or args.get("text")
        if isinstance(shown, (list, tuple)):
            shown = "+".join(str(k) for k in shown)
        self.emit("alert", text=f"{event.get('origin') or 'voice'}: {intent} {str(shown)[:40]}")

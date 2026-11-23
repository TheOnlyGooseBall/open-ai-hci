"""Abstract base class for output actions."""
from __future__ import annotations

from abc import ABC, abstractmethod

from .._component import Component
from ..event_bus import EventCallback

__all__ = ["BaseAction"]


class BaseAction(Component, ABC):
    """An action turns events into effects (mouse, keyboard, network, robot, ...).

    Implement :meth:`subscriptions`, a mapping of event name -> handler. The base
    class subscribes the handlers to the engine's bus in :meth:`setup` and removes
    them in :meth:`close`, so subclasses only describe *what* they react to::

        class Beep(BaseAction):
            name = "beep"

            def subscriptions(self):
                return {"pinch_start": self.on_pinch}

            def on_pinch(self, event):
                print("\\a")

    Override :meth:`setup` / :meth:`close` for resources, but call ``super()`` first/last.
    """

    name = "action"

    def __init__(self) -> None:
        self._subscribed: list[tuple[str, EventCallback]] = []

    @abstractmethod
    def subscriptions(self) -> dict[str, EventCallback]:
        """Return ``{event_name: handler}``."""

    def setup(self) -> None:
        for event_name, handler in self.subscriptions().items():
            self.bus.on(event_name, handler)
            self._subscribed.append((event_name, handler))

    def close(self) -> None:
        if self.engine is not None:
            for event_name, handler in self._subscribed:
                self.bus.off(event_name, handler)
        self._subscribed.clear()

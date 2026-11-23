"""Common lifecycle shared by sensors, processors, actions and the HUD."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .event_bus import Event, EventBus

if TYPE_CHECKING:  # pragma: no cover
    from .config import HCIConfig
    from .engine import HCIEngine

__all__ = ["Component"]


class Component:
    """Base for everything the :class:`~open_ai_hci.engine.HCIEngine` hosts.

    Lifecycle:

    1. ``attach(engine)``: called when the component is added. Gives access to
       ``self.engine``, ``self.bus`` and ``self.config``. Cheap; no hardware.
    2. ``setup()``: called once, right before the first frame. Load models,
       open devices, start threads here.
    3. ``close()``: called on shutdown (in reverse order). Release everything.
    """

    name: str = "component"
    engine: HCIEngine | None = None

    def attach(self, engine: HCIEngine) -> None:
        self.engine = engine

    def setup(self) -> None:  # noqa: B027 - intentionally optional
        """Acquire resources. Default: nothing to do."""

    def close(self) -> None:  # noqa: B027 - intentionally optional
        """Release resources. Default: nothing to do."""

    @property
    def bus(self) -> EventBus:
        if self.engine is None:
            raise RuntimeError(f"{type(self).__name__} is not attached to an engine yet")
        return self.engine.bus

    @property
    def config(self) -> HCIConfig:
        if self.engine is None:
            raise RuntimeError(f"{type(self).__name__} is not attached to an engine yet")
        return self.engine.config

    def emit(self, name: str, /, **data: Any) -> Event:
        """Emit an event on the engine bus, tagged with this component's name."""
        bus = self.bus
        return bus.emit(Event(name=str(name), data=data, frame_index=bus.frame_index, source=self.name))

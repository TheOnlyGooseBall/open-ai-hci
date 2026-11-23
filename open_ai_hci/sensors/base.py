"""Abstract base class for input sensors."""
from __future__ import annotations

from abc import ABC, abstractmethod

from .._component import Component
from ..types import FrameContext

__all__ = ["BaseSensor"]


class BaseSensor(Component, ABC):
    """An input sensor turns a camera frame into structured observations.

    Subclasses implement :meth:`process`, which reads ``ctx.rgb`` / ``ctx.frame``
    and writes its results onto the context (``ctx.hands``, ``ctx.faces``,
    ``ctx.poses`` or ``ctx.extras[...]`` for custom data). Heavy resources
    (models, devices) belong in :meth:`setup`, not ``__init__``.

    Sensors may call :meth:`emit_presence` to publish ``<kind>_detected`` /
    ``<kind>_lost`` events when the tracked thing appears or disappears.
    """

    name = "sensor"

    def __init__(self) -> None:
        self._present = False

    @abstractmethod
    def process(self, ctx: FrameContext) -> None:
        """Analyse the frame in ``ctx`` and store results on it."""

    def emit_presence(self, kind: str, present: bool, **data: object) -> None:
        """Emit ``<kind>_detected`` / ``<kind>_lost`` on state transitions only."""
        if present and not self._present:
            self.emit(f"{kind}_detected", **data)
        elif not present and self._present:
            self.emit(f"{kind}_lost", **data)
        self._present = present

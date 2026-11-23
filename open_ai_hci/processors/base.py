"""Abstract base class for processors (the decision / intelligence layer)."""
from __future__ import annotations

from abc import ABC, abstractmethod

from .._component import Component
from ..types import FrameContext

__all__ = ["BaseProcessor"]


class BaseProcessor(Component, ABC):
    """A processor reads what the sensors observed and decides what it *means*.

    Processors run in order, once per processed frame, after all sensors.
    They may enrich the :class:`~open_ai_hci.types.FrameContext` (smoothed
    cursor, gesture label, ...) and publish events with :meth:`emit`
    (``pinch_start``, ``gesture``, ...). Actions then react to those events.

    Keep :meth:`process` fast: it runs on the real-time loop. Do slow work
    (network calls, big models) in a background thread and emit the result when
    it is ready (see ``MultimodalAgent`` for the pattern).
    """

    name = "processor"

    @abstractmethod
    def process(self, ctx: FrameContext) -> None:
        """Handle one frame."""

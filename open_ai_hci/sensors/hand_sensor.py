"""MediaPipe hand tracking sensor."""
from __future__ import annotations

from typing import Any

from ..config import HandConfig
from ..types import FrameContext, HandData
from ._mediapipe import create_hand_backend
from .base import BaseSensor

__all__ = ["HandSensor"]


class HandSensor(BaseSensor):
    """Tracks up to ``HandConfig.max_hands`` hands and normalises their landmarks.

    Fills ``ctx.hands`` with :class:`~open_ai_hci.types.HandData` (21 normalised
    landmarks each). The hand selected by ``HandConfig.primary`` is placed first,
    so ``ctx.hand`` is the one that drives the cursor and pinch gestures.

    Emits ``hand_detected`` / ``hand_lost`` when the first hand appears / the
    last one disappears.

    Args:
        config: Overrides ``engine.config.hand``.
        backend: Inject a custom detector (anything with ``detect(rgb, ts_ms)``
            and ``close()``). Mainly for tests and alternative trackers.
    """

    name = "hand"

    def __init__(self, config: HandConfig | None = None, *, backend: Any = None) -> None:
        super().__init__()
        self._cfg = config
        self._backend = backend

    @property
    def cfg(self) -> HandConfig:
        return self._cfg if self._cfg is not None else self.config.hand

    def setup(self) -> None:
        if self._backend is None:
            self._backend = create_hand_backend(self.cfg, self.config.mediapipe)

    def process(self, ctx: FrameContext) -> None:
        raw = self._backend.detect(ctx.rgb, int(ctx.timestamp * 1000))
        hands = [
            HandData(
                index=i,
                handedness=r.label,
                score=r.score,
                landmarks=r.landmarks,
                frame_width=ctx.width,
                frame_height=ctx.height,
            )
            for i, r in enumerate(raw)
        ]
        primary = self.cfg.primary
        if primary in ("Left", "Right") and len(hands) > 1:
            hands.sort(key=lambda h: h.handedness != primary)  # stable: primary first
            for i, h in enumerate(hands):
                h.index = i
        ctx.hands = hands
        self.emit_presence("hand", bool(hands), count=len(hands))

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None

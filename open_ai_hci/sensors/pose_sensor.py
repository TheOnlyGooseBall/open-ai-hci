"""Full-body pose sensor (MediaPipe Pose). Disabled by default.

This module is also the smallest complete example of a sensor: copy it as a
starting point for your own (see CONTRIBUTING.md).
"""
from __future__ import annotations

from typing import Any

from ..config import PoseConfig
from ..types import FrameContext, PoseData
from ._mediapipe import create_pose_backend
from .base import BaseSensor

__all__ = ["PoseSensor"]


class PoseSensor(BaseSensor):
    """Tracks full-body pose landmarks (33 points) into ``ctx.poses``.

    Enable it with ``HCIConfig(pose=PoseConfig(enabled=True))`` or by passing
    ``sensors=[HandSensor(), PoseSensor()]`` to the engine. Emits
    ``pose_detected`` / ``pose_lost``.
    """

    name = "pose"

    def __init__(self, config: PoseConfig | None = None, *, backend: Any = None) -> None:
        super().__init__()
        self._cfg = config
        self._backend = backend

    @property
    def cfg(self) -> PoseConfig:
        return self._cfg if self._cfg is not None else self.config.pose

    def setup(self) -> None:
        if self._backend is None:
            self._backend = create_pose_backend(self.cfg, self.config.mediapipe)

    def process(self, ctx: FrameContext) -> None:
        raw = self._backend.detect(ctx.rgb, int(ctx.timestamp * 1000))
        ctx.poses = [
            PoseData(
                index=i,
                landmarks=r.landmarks,
                visibility=r.visibility,
                frame_width=ctx.width,
                frame_height=ctx.height,
            )
            for i, r in enumerate(raw)
        ]
        self.emit_presence("pose", bool(ctx.poses), count=len(ctx.poses))

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None

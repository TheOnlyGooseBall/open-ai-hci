"""Cursor smoothing: EMA, One-Euro and Kalman filters plus the screen-mapping processor.

The default ``EMAFilter(alpha=0.20)`` reproduces the prototype::

    actual = prev + (target - prev) * 0.20

One-Euro (Casiez et al., 2012) adapts its cutoff to hand speed: very steady when
the hand is slow, low-lag when it moves fast. The constant-velocity Kalman filter
trades a little lag for the best jitter rejection. All filters are pure Python
(no NumPy needed) so they cost microseconds per sample.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

from ..config import SmoothingConfig
from ..types import CursorState, FrameContext, HandLandmark
from .base import BaseProcessor

__all__ = [
    "ScalarFilter",
    "IdentityFilter",
    "EMAFilter",
    "OneEuroFilter",
    "KalmanFilter1D",
    "Filter2D",
    "make_filter_2d",
    "map_to_screen",
    "SmoothingProcessor",
]


class ScalarFilter(Protocol):
    """Anything that smooths one number at a time."""

    def reset(self) -> None: ...

    def __call__(self, value: float, t: float | None = None) -> float: ...


class IdentityFilter:
    """No smoothing (raw passthrough)."""

    def reset(self) -> None:  # noqa: D401 - trivial
        pass

    def __call__(self, value: float, t: float | None = None) -> float:
        return value


class EMAFilter:
    """Exponential moving average: ``y = y_prev + (x - y_prev) * alpha``.

    The first sample passes through unchanged. ``alpha`` in ``(0, 1]``; smaller
    means smoother but laggier.
    """

    def __init__(self, alpha: float = 0.20) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._y: float | None = None

    def reset(self) -> None:
        self._y = None

    def __call__(self, value: float, t: float | None = None) -> float:
        self._y = value if self._y is None else self._y + (value - self._y) * self.alpha
        return self._y


def _smoothing_factor(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One-Euro filter (adaptive low-pass) for jitter-free yet responsive pointing.

    Args:
        min_cutoff: Cutoff (Hz) at rest. Lower = steadier but laggier.
        beta: Speed coefficient. Higher = less lag during fast motion.
        d_cutoff: Cutoff (Hz) of the internal velocity estimate.
        default_dt: Time step (s) used when timestamps are missing or not increasing.
    """

    def __init__(
        self, min_cutoff: float = 1.0, beta: float = 0.01, d_cutoff: float = 1.0, default_dt: float = 1.0 / 30.0
    ) -> None:
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("cutoffs must be positive")
        self.min_cutoff, self.beta, self.d_cutoff, self.default_dt = min_cutoff, beta, d_cutoff, default_dt
        self.reset()

    def reset(self) -> None:
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    def __call__(self, value: float, t: float | None = None) -> float:
        if self._x is None:
            self._x, self._t, self._dx = value, t, 0.0
            return value
        dt = self.default_dt
        if t is not None and self._t is not None and t > self._t:
            dt = t - self._t
        self._t = t if t is not None else self._t
        dx = (value - self._x) / dt
        a_d = _smoothing_factor(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = _smoothing_factor(cutoff, dt)
        self._x = a * value + (1.0 - a) * self._x
        return self._x


class KalmanFilter1D:
    """Constant-velocity Kalman filter (position + velocity state).

    Args:
        process_noise: Acceleration variance ``q`` (higher = trusts motion less, follows faster).
        measurement_noise: Measurement variance ``r`` in squared input units (higher = smoother).
        default_dt: Time step (s) used when timestamps are missing or not increasing.
    """

    def __init__(self, process_noise: float = 2e5, measurement_noise: float = 50.0, default_dt: float = 1.0 / 30.0):
        if process_noise <= 0 or measurement_noise <= 0:
            raise ValueError("noise parameters must be positive")
        self.q, self.r, self.default_dt = process_noise, measurement_noise, default_dt
        self.reset()

    def reset(self) -> None:
        self._pos: float | None = None
        self._vel = 0.0
        self._p = [[self.r, 0.0], [0.0, 1e4]]  # state covariance
        self._t: float | None = None

    def __call__(self, value: float, t: float | None = None) -> float:
        if self._pos is None:
            self._pos, self._t = value, t
            return value
        dt = self.default_dt
        if t is not None and self._t is not None and t > self._t:
            dt = t - self._t
        self._t = t if t is not None else self._t

        # predict
        pos = self._pos + self._vel * dt
        vel = self._vel
        p00, p01, p10, p11 = self._p[0][0], self._p[0][1], self._p[1][0], self._p[1][1]
        n00 = p00 + dt * (p01 + p10) + dt * dt * p11 + self.q * dt**4 / 4.0
        n01 = p01 + dt * p11 + self.q * dt**3 / 2.0
        n10 = p10 + dt * p11 + self.q * dt**3 / 2.0
        n11 = p11 + self.q * dt**2

        # update (H = [1, 0])
        s = n00 + self.r
        k0, k1 = n00 / s, n10 / s
        y = value - pos
        self._pos = pos + k0 * y
        self._vel = vel + k1 * y
        self._p = [[(1 - k0) * n00, (1 - k0) * n01], [n10 - k1 * n00, n11 - k1 * n01]]
        return self._pos


class Filter2D:
    """Applies one independent scalar filter per axis."""

    def __init__(self, factory: Callable[[], ScalarFilter]) -> None:
        self.fx, self.fy = factory(), factory()

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()

    def __call__(self, x: float, y: float, t: float | None = None) -> tuple[float, float]:
        return self.fx(x, t), self.fy(y, t)


def make_filter_2d(cfg: SmoothingConfig) -> Filter2D:
    """Build the 2-D filter selected by ``cfg.kind``."""
    if cfg.kind == "ema":
        return Filter2D(lambda: EMAFilter(cfg.alpha))
    if cfg.kind == "one_euro":
        return Filter2D(lambda: OneEuroFilter(cfg.one_euro_min_cutoff, cfg.one_euro_beta, cfg.one_euro_d_cutoff))
    if cfg.kind == "kalman":
        return Filter2D(lambda: KalmanFilter1D(cfg.kalman_process_noise, cfg.kalman_measurement_noise))
    if cfg.kind == "none":
        return Filter2D(IdentityFilter)
    raise ValueError(f"Unknown smoothing kind {cfg.kind!r}")


def map_to_screen(
    ix: float,
    iy: float,
    frame_w: int,
    frame_h: int,
    screen_w: int,
    screen_h: int,
    margin: float = 0.0,
) -> tuple[float, float]:
    """Map a camera pixel to a screen pixel.

    With ``margin == 0`` this is exactly the prototype's ``screen / frame * pixel``.
    A positive ``margin`` (fraction of the frame ignored at each edge) lets the
    full screen, corners included, be reached without stretching your arm to the
    very edge of the camera view; the result is clamped to the screen.
    """
    if margin <= 0.0:
        return screen_w / frame_w * ix, screen_h / frame_h * iy
    span = 1.0 - 2.0 * margin
    nx = min(max((ix / frame_w - margin) / span, 0.0), 1.0)
    ny = min(max((iy / frame_h - margin) / span, 0.0), 1.0)
    return nx * (screen_w - 1), ny * (screen_h - 1)


class SmoothingProcessor(BaseProcessor):
    """Turns the primary hand's index fingertip into a smoothed screen cursor.

    Writes ``ctx.cursor`` (a :class:`~open_ai_hci.types.CursorState`). When the
    hand is lost the last known position is kept with ``active=False``, and the
    filter state is *not* reset, exactly like the prototype.

    This processor only computes; the :class:`GestureRulesProcessor` publishes
    ``cursor_move`` so it can freeze the cursor during gestures such as scrolling.
    """

    name = "smoothing"

    def __init__(self, config: SmoothingConfig | None = None, *, screen_size: tuple[int, int] | None = None):
        self._cfg = config
        self._screen_size = screen_size
        self._filter: Filter2D | None = None
        self._last: CursorState | None = None

    @property
    def cfg(self) -> SmoothingConfig:
        return self._cfg if self._cfg is not None else self.config.smoothing

    def setup(self) -> None:
        self._filter = make_filter_2d(self.cfg)
        if self._screen_size is None:
            self._screen_size = self.engine.screen_size  # type: ignore[union-attr]

    def reset(self) -> None:
        if self._filter is not None:
            self._filter.reset()
        self._last = None

    def reconfigure(self) -> None:
        """Rebuild the filter from the current :class:`SmoothingConfig` without a cursor jump.

        The new filter is seeded with the last smoothed position, so changing the slider while the
        engine runs only changes how fast the cursor follows from then on. Safe to call from another thread.
        """
        new = make_filter_2d(self.cfg)
        if self._last is not None:
            new(self._last.x, self._last.y, time.monotonic())
        self._filter = new

    def process(self, ctx: FrameContext) -> None:
        if self._filter is None:  # used standalone without engine.setup()
            self._filter = make_filter_2d(self.cfg)
        hand = ctx.hand
        if hand is None:
            ctx.cursor = replace(self._last, active=False) if self._last is not None else None
            return
        assert self._screen_size is not None, "screen_size unknown; pass screen_size= or attach to an engine"
        ix, iy = hand.px(HandLandmark.INDEX_TIP)
        target_x, target_y = map_to_screen(
            ix, iy, ctx.width, ctx.height, self._screen_size[0], self._screen_size[1], self.cfg.screen_margin
        )
        x, y = self._filter(target_x, target_y, ctx.timestamp)
        self._last = CursorState(x=x, y=y, raw_x=target_x, raw_y=target_y, active=True)
        ctx.cursor = self._last

"""Deterministic gesture rules: pinch, right pinch, pause toggle, drag, scroll, blink, head flick.

The math lives in small pure functions / tiny state machines that can be unit
tested without a camera; :class:`GestureRulesProcessor` wires them to the frame
pipeline and publishes events.

Thresholds reproduce the prototype exactly:

* pinch while ``distance < 45`` px (strict), measured on int-truncated pixel
  coordinates of landmarks 8 (index tip) and 4 (thumb tip);
* head flick (opt-in since v0.2.0, ``FaceConfig.flick_enabled``) when ``abs(dx) > 28``
  and ``abs(dx) > abs(dy) * 3`` between two processed frames, then a 1.2 s cooldown; the
  previous nose position keeps updating during the cooldown.

New in v0.2.0: a thumb + middle-finger pinch triggers a right click, and holding an open palm
(or a fist) for ``HandConfig.pause_hold_s`` seconds pauses / resumes cursor control.
"""
from __future__ import annotations

import math

from ..config import REFERENCE_WIDTH, FaceConfig, HandConfig
from ..event_bus import Event
from ..types import FACE_NOSE_TIP, FrameContext, HandData, HandLandmark
from .base import BaseProcessor

__all__ = [
    "pinch_distance",
    "is_pinch",
    "detect_head_flick",
    "FlickDetector",
    "BlinkDetector",
    "finger_extended",
    "is_two_finger_pose",
    "is_open_palm",
    "is_fist",
    "HoldToggle",
    "ScrollTracker",
    "GestureRulesProcessor",
]


# ----------------------------------------------------------------------- pure rules
def pinch_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two points (same expression as the prototype)."""
    return (((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2)) ** 0.5


def is_pinch(distance: float, threshold: float = 45.0, currently_pinching: bool = False, release_margin: float = 0.0) -> bool:
    """Pinch state with optional hysteresis.

    A pinch *starts* when ``distance < threshold`` (strict, like the prototype)
    and, if it was already active, *ends* once ``distance >= threshold + release_margin``.
    With the default margin of 0 this is exactly ``distance < threshold``.
    """
    if currently_pinching:
        return distance < threshold + release_margin
    return distance < threshold


def detect_head_flick(dx: float, dy: float, threshold: float = 28.0, axis_ratio: float = 3.0) -> str | None:
    """Classify a nose displacement as a horizontal flick.

    Returns ``"right"`` / ``"left"`` (in the mirrored image: ``dx > 0`` is right)
    when ``abs(dx) > threshold`` and ``abs(dx) > abs(dy) * axis_ratio``, else ``None``.
    """
    if abs(dx) > threshold and abs(dx) > abs(dy) * axis_ratio:
        return "right" if dx > 0 else "left"
    return None


class FlickDetector:
    """Stateful head-flick detector with cooldown.

    Mirrors the prototype loop: the previous nose position is updated on *every*
    call (including during the cooldown), and only a flick outside the cooldown
    window can fire.
    """

    def __init__(self, threshold: float = 28.0, axis_ratio: float = 3.0, cooldown_s: float = 1.2) -> None:
        self.threshold, self.axis_ratio, self.cooldown_s = threshold, axis_ratio, cooldown_s
        self._last_fire = -math.inf
        self._prev: tuple[float, float] | None = None

    def reset(self) -> None:
        """Forget the previous nose position (the cooldown timer is kept)."""
        self._prev = None

    def update(
        self,
        x: float,
        y: float,
        t: float,
        *,
        threshold: float | None = None,
        axis_ratio: float | None = None,
    ) -> tuple[str, float, float] | None:
        """Feed one nose position at time ``t`` (seconds).

        Returns ``(direction, dx, dy)`` when a flick fires, otherwise ``None``.
        """
        result = None
        if self._prev is not None and (t - self._last_fire > self.cooldown_s):
            dx, dy = x - self._prev[0], y - self._prev[1]
            direction = detect_head_flick(
                dx, dy, self.threshold if threshold is None else threshold, self.axis_ratio if axis_ratio is None else axis_ratio
            )
            if direction is not None:
                self._last_fire = t
                result = (direction, dx, dy)
        self._prev = (x, y)
        return result


class BlinkDetector:
    """Detects blinks from a stream of eye-aspect-ratio values.

    A blink is ``min_frames <= n <= max_frames`` consecutive frames below
    ``ear_threshold`` followed by an open eye. Longer closures are treated as
    deliberate eye-closing, not blinks.
    """

    def __init__(self, ear_threshold: float = 0.21, min_frames: int = 2, max_frames: int = 15) -> None:
        self.ear_threshold, self.min_frames, self.max_frames = ear_threshold, min_frames, max_frames
        self._closed = 0

    def reset(self) -> None:
        self._closed = 0

    def update(self, ear: float) -> bool:
        """Feed one EAR sample; returns ``True`` on the frame the eye re-opens after a blink."""
        if ear < self.ear_threshold:
            self._closed += 1
            return False
        blinked = self.min_frames <= self._closed <= self.max_frames
        self._closed = 0
        return blinked


def finger_extended(hand: HandData, tip: int, pip: int) -> bool:
    """A finger is extended when its tip is farther from the wrist than its PIP joint.

    Rotation-invariant, so it works for any hand orientation.
    """
    wx, wy = hand.px(HandLandmark.WRIST)
    tx, ty = hand.px(tip)
    px_, py_ = hand.px(pip)
    return math.hypot(tx - wx, ty - wy) > math.hypot(px_ - wx, py_ - wy)


def is_two_finger_pose(hand: HandData) -> bool:
    """Index + middle extended, ring + pinky curled (the "scroll" pose)."""
    L = HandLandmark
    return (
        finger_extended(hand, L.INDEX_TIP, L.INDEX_PIP)
        and finger_extended(hand, L.MIDDLE_TIP, L.MIDDLE_PIP)
        and not finger_extended(hand, L.RING_TIP, L.RING_PIP)
        and not finger_extended(hand, L.PINKY_TIP, L.PINKY_PIP)
    )


def is_open_palm(hand: HandData) -> bool:
    """All four fingers extended (thumb ignored). Rotation-invariant."""
    L = HandLandmark
    return all(
        finger_extended(hand, tip, pip)
        for tip, pip in ((L.INDEX_TIP, L.INDEX_PIP), (L.MIDDLE_TIP, L.MIDDLE_PIP), (L.RING_TIP, L.RING_PIP), (L.PINKY_TIP, L.PINKY_PIP))
    )


def is_fist(hand: HandData) -> bool:
    """All four fingers curled."""
    L = HandLandmark
    return not any(
        finger_extended(hand, tip, pip)
        for tip, pip in ((L.INDEX_TIP, L.INDEX_PIP), (L.MIDDLE_TIP, L.MIDDLE_PIP), (L.RING_TIP, L.RING_PIP), (L.PINKY_TIP, L.PINKY_PIP))
    )


class HoldToggle:
    """Fires once after a pose has been held for ``hold_s`` seconds, then latches.

    The latch is released only when the pose is *not* shown for at least one update, so keeping
    the palm up after pausing does not immediately resume again.
    """

    def __init__(self, hold_s: float = 1.0) -> None:
        self.hold_s = hold_s
        self.reset()

    def reset(self) -> None:
        self._since: float | None = None
        self._latched = False

    def update(self, active: bool, t: float) -> tuple[bool, float]:
        """Feed one observation. Returns ``(toggled, progress)``; ``progress`` runs 0..1 while holding."""
        if not active:
            self.reset()
            return False, 0.0
        if self._since is None:
            self._since = t
        if self._latched:
            return False, 0.0
        progress = min(1.0, (t - self._since) / self.hold_s)
        if progress >= 1.0:
            self._latched = True
            return True, 1.0
        return False, progress


class ScrollTracker:
    """Converts vertical finger travel into whole mouse-wheel clicks.

    Fractional travel accumulates, so slow motion still scrolls eventually.
    Positive output scrolls *up* (finger moving up the image).
    """

    def __init__(self, pixels_per_click: float = 18.0) -> None:
        self.pixels_per_click = pixels_per_click
        self.reset()

    def reset(self) -> None:
        self._prev_y: float | None = None
        self._acc = 0.0

    def update(self, y: float) -> int:
        if self._prev_y is None:
            self._prev_y = y
            return 0
        self._acc += (self._prev_y - y) / self.pixels_per_click  # image y grows downwards
        self._prev_y = y
        clicks = int(self._acc)  # truncates toward zero
        self._acc -= clicks
        return clicks


# -------------------------------------------------------------------------- processor
class GestureRulesProcessor(BaseProcessor):
    """Applies the deterministic rules and publishes gesture events.

    Per processed frame (in prototype order: move, then click):

    * ``cursor_move``: smoothed cursor position (skipped while scrolling or paused);
    * ``pinch_start`` / ``pinch_end``: thumb-index pinch (``HandConfig.pinch_threshold_px``);
    * ``right_pinch_start`` / ``right_pinch_end`` + ``right_click``: thumb-middle pinch
      (``HandConfig.right_pinch_enabled``); the closer of the two fingertips wins if both touch;
    * ``right_click`` (``via="ai_gesture"``) when the ONNX classifier reports ``OnnxConfig.right_click_label``;
    * ``control_paused`` / ``control_resumed``: open palm (or fist) held for ``HandConfig.pause_hold_s``;
      while paused nothing is emitted except the pause gesture itself;
    * ``drag_start`` / ``drag_move`` / ``drag_end``: pinch held while the cursor travels
      more than ``HandConfig.drag_threshold_px`` screen pixels;
    * ``scroll_start`` / ``scroll`` / ``scroll_end``: two-finger pose, if ``HandConfig.scroll_enabled``;
    * ``head_flick`` (+ ``_left`` / ``_right``): sharp nose flick, only if ``FaceConfig.flick_enabled``;
    * ``blink``: if ``FaceConfig.enable_blink``.

    It also sets ``ctx.pinch_status`` (``"HOLDING CLICK"``, ``"RIGHT CLICK"``, ``"OPEN"``, ``"PAUSED"`` or
    ``"NO HAND DETECTED"``), ``ctx.pinch_distance``, ``ctx.scroll_mode``, ``ctx.paused`` and
    ``ctx.extras["pause_progress"]``. Pixel thresholds scale with ``ctx.width / 1280``.
    """

    name = "gesture_rules"

    def __init__(self, hand_config: HandConfig | None = None, face_config: FaceConfig | None = None) -> None:
        self._hand_cfg = hand_config
        self._face_cfg = face_config
        self._pinching = False
        self._right_pinching = False
        self._dragging = False
        self._pinch_origin: tuple[float, float] | None = None
        self._scrolling = False
        self._paused = False
        self._now = 0.0
        self._last_right_click = -math.inf
        self._last_xy: tuple[float, float] | None = None
        self._ai_label: str | None = None
        self._build()

    @property
    def hand_cfg(self) -> HandConfig:
        return self._hand_cfg if self._hand_cfg is not None else self.config.hand

    @property
    def face_cfg(self) -> FaceConfig:
        return self._face_cfg if self._face_cfg is not None else self.config.face

    @property
    def paused(self) -> bool:
        """``True`` while cursor control is paused."""
        return self._paused

    def set_paused(self, paused: bool, via: str = "api") -> bool:
        """Pause / resume control (thread-safe: the held button is released on the next frame).

        Returns ``True`` if the state changed.
        """
        if paused == self._paused:
            return False
        self._paused = paused  # (the hold latch is kept: a palm still raised must not flip it straight back)
        if self.engine is not None:
            self.emit("control_paused" if paused else "control_resumed", via=via)
        return True

    def setup(self) -> None:
        self._build()
        self._unsubscribe_ai()
        label = self.config.onnx.right_click_label if self.engine is not None else ""
        if label:
            self._ai_label = f"gesture_{label}"
            self.bus.on(self._ai_label, self._on_ai_gesture)

    def close(self) -> None:
        self._unsubscribe_ai()

    def _unsubscribe_ai(self) -> None:
        if self._ai_label is not None and self.engine is not None:
            self.bus.off(self._ai_label, self._on_ai_gesture)
        self._ai_label = None

    def _build(self) -> None:
        """(Re)create the stateful detectors from the effective configs."""
        f = self._face_cfg if self._face_cfg is not None else (self.config.face if self.engine else FaceConfig())
        h = self._hand_cfg if self._hand_cfg is not None else (self.config.hand if self.engine else HandConfig())
        self._flick = FlickDetector(f.flick_threshold_px, f.flick_axis_ratio, f.flick_cooldown_s)
        self._blink = BlinkDetector(f.blink_ear_threshold, f.blink_min_frames, f.blink_max_frames)
        self._scroll = ScrollTracker(h.scroll_pixels_per_click)
        self._pause_toggle = HoldToggle(h.pause_hold_s)

    # ---------------------------------------------------------------------- frame
    def process(self, ctx: FrameContext) -> None:
        self._now = ctx.timestamp
        ctx.paused = self._paused
        self._process_hand(ctx)
        self._process_face(ctx)

    def _pause_pose(self, hand: HandData) -> bool:
        kind = self.hand_cfg.pause_gesture
        if kind == "open_palm":
            return is_open_palm(hand)
        if kind == "fist":
            return is_fist(hand)
        return False

    def _process_hand(self, ctx: FrameContext) -> None:
        cfg = self.hand_cfg
        scale = ctx.width / REFERENCE_WIDTH
        hand = ctx.hand
        ctx.extras["pause_progress"] = 0.0
        self._pause_toggle.hold_s = cfg.pause_hold_s

        if self._paused:  # (also covers pause() called from another thread)
            self._release_held(ctx, reason="paused")

        if hand is None:
            ctx.pinch_status = "PAUSED" if self._paused else "NO HAND DETECTED"
            ctx.pinch_distance = None
            self._pause_toggle.update(False, ctx.timestamp)
            self._release_held(ctx, reason="hand_lost")
            return

        index_tip, thumb_tip = hand.index_tip, hand.thumb_tip
        distance = pinch_distance(index_tip, thumb_tip)
        ctx.pinch_distance = distance
        left = is_pinch(distance, cfg.pinch_threshold_px * scale, self._pinching, cfg.pinch_release_margin_px * scale)

        # thumb + middle finger = right pinch; if both fingertips touch the thumb, whoever already
        # owns the gesture keeps it, otherwise the closer fingertip wins (ties -> left click)
        right = False
        if cfg.right_pinch_enabled:
            d_mid = pinch_distance(hand.px(HandLandmark.MIDDLE_TIP), thumb_tip)
            ctx.extras["right_pinch_distance"] = d_mid
            right = is_pinch(d_mid, cfg.right_pinch_threshold_px * scale, self._right_pinching, cfg.pinch_release_margin_px * scale)
            if left and right:
                if self._pinching:
                    right = False
                elif self._right_pinching:
                    left = False
                else:
                    left, right = d_mid >= distance, d_mid < distance
        pinching = left

        # hold an open palm / fist to pause or resume (never while a pinch is in progress)
        pose = self._pause_pose(hand) and not left and not right
        toggled, progress = self._pause_toggle.update(pose, ctx.timestamp)
        ctx.extras["pause_progress"] = progress
        if toggled:
            self.set_paused(not self._paused, via=cfg.pause_gesture)
            ctx.paused = self._paused
            if self._paused:
                self._release_held(ctx, reason="paused")
        if self._paused:
            ctx.pinch_status = "PAUSED"
            ctx.scroll_mode = False
            return

        # scroll mode (opt-in) freezes the cursor and turns vertical travel into wheel clicks
        scroll_mode = cfg.scroll_enabled and not pinching and not right and is_two_finger_pose(hand)
        ctx.scroll_mode = scroll_mode
        if scroll_mode:
            self._handle_scroll(ctx, hand)
        elif self._scrolling:
            self._end_scroll(ctx)

        cursor = ctx.cursor
        if cursor is not None:
            self._last_xy = (cursor.x, cursor.y)
        if cursor is not None and cursor.active and not scroll_mode:
            self.emit("cursor_move", x=cursor.x, y=cursor.y, raw_x=cursor.raw_x, raw_y=cursor.raw_y)

        if pinching and not self._pinching:
            self._start_pinch(ctx, hand, distance)
        elif not pinching and self._pinching:
            self._end_pinch(ctx, reason="released")
        elif pinching:
            self._update_drag(ctx)

        if right and not self._right_pinching:
            self._start_right_pinch(ctx, ctx.extras.get("right_pinch_distance", 0.0))
        elif not right and self._right_pinching:
            self._end_right_pinch(ctx, reason="released")

        ctx.pinch_status = "HOLDING CLICK" if self._pinching else ("RIGHT CLICK" if self._right_pinching else "OPEN")

    def _release_held(self, ctx: FrameContext, reason: str) -> None:
        """End scroll / pinch / right pinch (hand lost, paused)."""
        if self._scrolling:
            self._end_scroll(ctx)
        if self._pinching:
            self._end_pinch(ctx, reason=reason)
        if self._right_pinching:
            self._end_right_pinch(ctx, reason=reason)

    # ----------------------------------------------------------------- pinch / drag
    def _start_pinch(self, ctx: FrameContext, hand: HandData, distance: float) -> None:
        self._pinching = True
        self._dragging = False
        c = ctx.cursor
        self._pinch_origin = (c.x, c.y) if c is not None else None
        self.emit(
            "pinch_start",
            x=c.x if c else None,
            y=c.y if c else None,
            distance=distance,
            handedness=hand.handedness,
        )

    def _update_drag(self, ctx: FrameContext) -> None:
        c, origin = ctx.cursor, self._pinch_origin
        if c is None or origin is None or not c.active:
            return
        dx, dy = c.x - origin[0], c.y - origin[1]
        if not self._dragging and math.hypot(dx, dy) > self.hand_cfg.drag_threshold_px:
            self._dragging = True
            self.emit("drag_start", x=c.x, y=c.y, start_x=origin[0], start_y=origin[1])
        if self._dragging:
            self.emit("drag_move", x=c.x, y=c.y, dx=dx, dy=dy)

    def _end_pinch(self, ctx: FrameContext, reason: str) -> None:
        c = ctx.cursor
        pos = {"x": c.x if c else None, "y": c.y if c else None}
        if self._dragging:
            self.emit("drag_end", reason=reason, **pos)
        self.emit("pinch_end", reason=reason, **pos)
        self._pinching = self._dragging = False
        self._pinch_origin = None

    # ------------------------------------------------------------------ right pinch
    def _start_right_pinch(self, ctx: FrameContext, distance: float) -> None:
        self._right_pinching = True
        c = ctx.cursor
        x, y = (c.x, c.y) if c is not None else (None, None)
        self.emit("right_pinch_start", x=x, y=y, distance=distance)
        self._fire_right_click("middle_pinch", x, y)

    def _end_right_pinch(self, ctx: FrameContext, reason: str) -> None:
        c = ctx.cursor
        self._right_pinching = False
        self.emit("right_pinch_end", reason=reason, x=c.x if c else None, y=c.y if c else None)

    def _fire_right_click(self, via: str, x: float | None = None, y: float | None = None) -> bool:
        """Emit ``right_click`` unless paused or inside the cooldown. Returns ``True`` if it fired."""
        if self._paused or self._now - self._last_right_click < self.hand_cfg.right_click_cooldown_s:
            return False
        if x is None and self._last_xy is not None:
            x, y = self._last_xy
        self._last_right_click = self._now
        self.emit("right_click", x=x, y=y, via=via)
        return True

    def _on_ai_gesture(self, event: Event) -> None:
        self._fire_right_click("ai_gesture")

    # -------------------------------------------------------------------- scrolling
    def _handle_scroll(self, ctx: FrameContext, hand: HandData) -> None:
        if not self._scrolling:
            self._scrolling = True
            self._scroll.reset()
            self.emit("scroll_start")
        y = (hand.px(HandLandmark.INDEX_TIP)[1] + hand.px(HandLandmark.MIDDLE_TIP)[1]) / 2.0
        scale = ctx.width / REFERENCE_WIDTH
        self._scroll.pixels_per_click = self.hand_cfg.scroll_pixels_per_click * scale
        clicks = self._scroll.update(y)
        if clicks:
            self.emit("scroll", amount=clicks)

    def _end_scroll(self, ctx: FrameContext) -> None:
        self._scrolling = False
        self._scroll.reset()
        self.emit("scroll_end")

    # ------------------------------------------------------------------------- face
    def _process_face(self, ctx: FrameContext) -> None:
        cfg = self.face_cfg
        face = ctx.face
        if face is None:
            if cfg.reset_on_loss:
                self._flick.reset()
            self._blink.reset()
            return

        if cfg.flick_enabled:
            nose_x, nose_y = face.px(FACE_NOSE_TIP)
            scale = ctx.width / REFERENCE_WIDTH
            fired = self._flick.update(nose_x, nose_y, ctx.timestamp, threshold=cfg.flick_threshold_px * scale)
            if fired is not None and not self._paused:
                direction, dx, dy = fired
                payload = dict(direction=direction, dx=dx, dy=dy, x=nose_x, y=nose_y)
                self.emit("head_flick", **payload)
                self.emit(f"head_flick_{direction}", **payload)

        if cfg.enable_blink and not self._paused:
            ear = face.ear
            if ear is not None and self._blink.update(ear):
                self.emit("blink", ear=ear)

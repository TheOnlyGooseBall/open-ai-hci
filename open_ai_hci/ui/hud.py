"""Modular OpenCV diagnostic HUD.

With default settings the overlay is pixel-for-pixel the prototype's: the same
title box, text positions, fonts and colours (BGR), fingertip / thumb / nose
markers and the temporary "ACTION" banner. Extras (FPS, AI gesture label, head
pose, blink EAR, full skeleton) are additive and can be switched off.

Colour code for the pinch line: green = ``HOLDING CLICK``, yellow = ``RIGHT CLICK``,
magenta = ``PAUSED``, red = ``NO HAND DETECTED``, orange = everything else (``OPEN`` / ``DISCONNECTED``).
While control is paused a banner is drawn along the bottom edge; while the pause pose is being
held a progress bar fills up.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .._component import Component
from ..config import HUDConfig
from ..event_bus import Event
from ..types import HAND_CONNECTIONS, POSE_CONNECTIONS, FrameContext, HandLandmark

__all__ = ["HUD", "pinch_color"]

# BGR colours
WHITE, GREY, DIM, RED, GREEN, ORANGE = (255, 255, 255), (200, 200, 200), (150, 150, 150), (0, 0, 255), (0, 255, 0), (0, 165, 255)
YELLOW, MAGENTA, DARK_GREEN = (0, 255, 255), (255, 0, 255), (0, 128, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def pinch_color(status: str) -> tuple[int, int, int]:
    """Colour for a pinch status string (see module docs)."""
    if status == "HOLDING CLICK":
        return GREEN
    if status == "RIGHT CLICK":
        return YELLOW
    if status == "PAUSED":
        return MAGENTA
    if status == "NO HAND DETECTED":
        return RED
    return ORANGE


class HUD(Component):
    """Draws diagnostics onto ``ctx.frame`` and shows it in an OpenCV window.

    The HUD listens for ``alert`` events (``emit("alert", text="...")``) and shows the most
    recent one in a green banner for ``HUDConfig.alert_seconds``. Other components (mouse
    action, voice agent, your own code) can therefore surface messages without touching the HUD.

    Args:
        config: Overrides ``engine.config.hud``.
        headless: Draw only; never open a window (used by tests and video export).
    """

    name = "hud"

    def __init__(self, config: HUDConfig | None = None, *, headless: bool = False) -> None:
        self._cfg = config
        self.headless = headless
        self._alert_text = ""
        self._alert_expiry = 0.0
        self._window_open = False
        self._quit_codes: set[int] = set()

    @property
    def cfg(self) -> HUDConfig:
        """Effective settings: the constructor's, else the engine's, else defaults (standalone use)."""
        if self._cfg is not None:
            return self._cfg
        return self.engine.config.hud if self.engine is not None else HUDConfig()

    # ------------------------------------------------------------------ lifecycle
    def setup(self) -> None:
        self._quit_codes = {ord(k) for k in self.cfg.quit_keys} | {ord(k.upper()) for k in self.cfg.quit_keys}
        self.bus.on("alert", self._on_alert)

    def close(self) -> None:
        if self.engine is not None:
            self.bus.off("alert", self._on_alert)
        if self._window_open:
            try:
                cv2.destroyWindow(self.cfg.window_name)
            except cv2.error:  # pragma: no cover - window already gone
                pass
            self._window_open = False

    def _on_alert(self, event: Event) -> None:
        self.show_alert(str(event.get("text", "")))

    def show_alert(self, text: str, seconds: float | None = None) -> None:
        """Show ``ACTION: <text>`` in the banner (default duration ``HUDConfig.alert_seconds``)."""
        self._alert_text = text
        self._alert_expiry = time.monotonic() + (self.cfg.alert_seconds if seconds is None else seconds)

    # -------------------------------------------------------------------- drawing
    def draw(self, ctx: FrameContext) -> np.ndarray:
        """Render everything onto ``ctx.frame`` (in place) and return it."""
        frame, cfg = ctx.frame, self.cfg
        if cfg.show_skeleton:
            self._draw_skeleton(frame, ctx)
        if cfg.show_landmarks:
            self._draw_markers(frame, ctx)
        self._draw_panel(frame, ctx)
        if time.monotonic() < self._alert_expiry:
            cv2.rectangle(frame, (10, 160), (380, 200), DARK_GREEN, -1)
            cv2.putText(frame, f"ACTION: {self._alert_text}", (20, 185), FONT, 0.5, WHITE, 2)
        if cfg.show_extras:
            self._draw_extras(frame, ctx)
        self._draw_pause(frame, ctx)
        return frame

    @staticmethod
    def _draw_pause(frame: np.ndarray, ctx: FrameContext) -> None:
        """Bottom banner while paused; a progress bar while the pause pose is being held."""
        h, w = frame.shape[:2]
        progress = float(ctx.extras.get("pause_progress") or 0.0)
        if ctx.paused:
            cv2.rectangle(frame, (0, h - 44), (w, h), (90, 0, 90), -1)
            text = "PAUSED - hold the pause pose to resume"
            cv2.putText(frame, text, (20, h - 15), FONT, 0.7, WHITE, 2)
        if 0.0 < progress < 1.0:
            cv2.rectangle(frame, (10, 152), (380, 160), (60, 60, 60), -1)
            cv2.rectangle(frame, (10, 152), (10 + int(370 * progress), 160), MAGENTA, -1)

    @staticmethod
    def _draw_markers(frame: np.ndarray, ctx: FrameContext) -> None:
        for hand in ctx.hands:
            ix, iy = hand.px(HandLandmark.INDEX_TIP)
            tx, ty = hand.px(HandLandmark.THUMB_TIP)
            cv2.circle(frame, (ix, iy), 8, YELLOW, -1)  # index tip
            cv2.circle(frame, (tx, ty), 8, MAGENTA, -1)  # thumb tip
            cv2.line(frame, (ix, iy), (tx, ty), WHITE, 2)  # measuring line
        for face in ctx.faces:
            cv2.circle(frame, face.nose, 5, RED, -1)  # nose node

    @staticmethod
    def _draw_skeleton(frame: np.ndarray, ctx: FrameContext) -> None:
        for hand in ctx.hands:
            pts = [hand.px(i) for i in range(21)]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (80, 200, 80), 1)
            for p in pts:
                cv2.circle(frame, p, 3, (60, 160, 255), -1)
        for pose in ctx.poses:
            pts = [pose.px(i) for i in range(len(pose.landmarks))]
            for a, b in POSE_CONNECTIONS:
                if a < len(pts) and b < len(pts) and pose.visibility[a] > 0.4 and pose.visibility[b] > 0.4:
                    cv2.line(frame, pts[a], pts[b], (255, 180, 60), 2)

    def _draw_panel(self, frame: np.ndarray, ctx: FrameContext) -> None:
        cfg = self.cfg
        cx, cy = (int(ctx.cursor.x), int(ctx.cursor.y)) if ctx.cursor is not None else (0, 0)
        cv2.rectangle(frame, (10, 10), (380, 150), (0, 0, 0), -1)
        cv2.putText(frame, cfg.title, (20, 35), FONT, 0.5, WHITE, 1)
        cv2.putText(frame, f"Cursor Position: X={cx} Y={cy}", (20, 60), FONT, 0.5, GREY, 1)
        cv2.putText(frame, f"Pinch Grip: {ctx.pinch_status}", (20, 90), FONT, 0.5, pinch_color(ctx.pinch_status), 2)
        count = f"Camera Frame Count: {ctx.frame_index}"
        fps = ctx.extras.get("fps")
        if cfg.show_fps and fps:
            count += f"  |  {fps:.0f} FPS"
        cv2.putText(frame, count, (20, 115), FONT, 0.4, DIM, 1)
        keys = "/".join(k.upper() for k in cfg.quit_keys)
        cv2.putText(frame, f"Press '{keys}' on keyboard to Exit", (20, 135), FONT, 0.4, RED, 1)

    def _draw_extras(self, frame: np.ndarray, ctx: FrameContext) -> None:
        lines: list[tuple[str, tuple[int, int, int]]] = []
        if ctx.gesture:
            lines.append((f"AI gesture: {ctx.gesture} ({ctx.gesture_confidence:.0%})", YELLOW))
        if ctx.scroll_mode:
            lines.append(("Scroll mode", ORANGE))
        face = ctx.face
        if face is not None:
            if face.head_pose is not None:
                hp = face.head_pose
                lines.append((f"Head yaw {hp.yaw:+.0f}  pitch {hp.pitch:+.0f}  roll {hp.roll:+.0f}", GREY))
            if face.ear is not None:
                lines.append((f"Eye openness (EAR): {face.ear:.2f}", GREY))
        if not lines:
            return
        top = 210
        cv2.rectangle(frame, (10, top), (380, top + 8 + 22 * len(lines)), (0, 0, 0), -1)
        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (20, top + 22 + 22 * i), FONT, 0.5, color, 1)

    # ------------------------------------------------------------------- display
    def show(self, ctx: FrameContext) -> int:
        """Draw, display and poll the keyboard for 1 ms. Returns the key code (``-1`` if none).

        Returns ``-2`` when the window was closed with the mouse (treated as quit by the engine).
        """
        self.draw(ctx)
        if self.headless:
            return -1
        name = self.cfg.window_name
        cv2.imshow(name, ctx.frame)
        self._window_open = True
        key = cv2.waitKey(1) & 0xFF
        if key == 0xFF:
            key = -1
        try:
            if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                return -2
        except cv2.error:  # some backends cannot report visibility
            pass
        return key

    def is_quit_key(self, key: int) -> bool:
        return key == -2 or key in self._quit_codes

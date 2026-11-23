"""The orchestration loop: camera -> sensors -> processors -> actions (+ HUD).

Five-line integration::

    from open_ai_hci import HCIEngine
    hci = HCIEngine()
    hci.on("pinch_start", lambda event: print("Pinched!", event))
    hci.run(show_hud=True)

The default pipeline reproduces the original prototype: hand + face sensors,
EMA smoothing, deterministic gesture rules and the OS mouse action. Replace or
extend any layer::

    hci = HCIEngine(config, sensors=[HandSensor()], actions=[WebSocketBridge()])
    hci.add_processor(ONNXGestureClassifier("my_model.onnx"))
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from collections.abc import Sequence
from types import TracebackType
from typing import Any, TypeVar

import cv2
import numpy as np

from ._component import Component
from .actions._pyautogui import load_pyautogui
from .actions.base import BaseAction
from .actions.os_keyboard import OSKeyboardAction
from .actions.os_mouse import OSMouseAction
from .config import HCIConfig
from .event_bus import Event, EventBus, EventCallback, EventName
from .processors.base import BaseProcessor
from .processors.gesture_rules import GestureRulesProcessor
from .processors.smoothing import SmoothingProcessor
from .sensors.base import BaseSensor
from .sensors.face_sensor import FaceSensor
from .sensors.hand_sensor import HandSensor
from .sensors.pose_sensor import PoseSensor
from .types import FrameContext
from .ui.hud import HUD

__all__ = ["HCIEngine", "FPSCounter"]

logger = logging.getLogger("open_ai_hci")

C = TypeVar("C", bound=Component)


class FPSCounter:
    """Sliding-window frames-per-second estimator (default window: 1 second)."""

    def __init__(self, window_s: float = 1.0) -> None:
        self.window_s = window_s
        self._stamps: collections.deque[float] = collections.deque()

    def tick(self, now: float | None = None) -> float:
        """Record one frame and return the current FPS."""
        now = time.monotonic() if now is None else now
        self._stamps.append(now)
        while self._stamps and now - self._stamps[0] > self.window_s:
            self._stamps.popleft()
        return self.fps

    @property
    def fps(self) -> float:
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0

    def reset(self) -> None:
        self._stamps.clear()


class HCIEngine:
    """Owns the pipeline, the event bus and the main loop.

    Args:
        config: Settings (defaults reproduce the prototype). Pass a profile name such as
            ``"precise"`` as a string for convenience.
        sensors / processors / actions: Replace the default layers entirely. Leave ``None``
            for the default pipeline; add more later with :meth:`add_sensor` & co.
        hud: A custom :class:`~open_ai_hci.ui.hud.HUD` instance (default: built from ``config.hud``).
        screen_size: ``(width, height)`` of the target screen; default auto-detected.
        raise_errors: Re-raise exceptions from event handlers and components instead of logging them.
    """

    def __init__(
        self,
        config: HCIConfig | str | None = None,
        *,
        sensors: Sequence[BaseSensor] | None = None,
        processors: Sequence[BaseProcessor] | None = None,
        actions: Sequence[BaseAction] | None = None,
        hud: HUD | None = None,
        screen_size: tuple[int, int] | None = None,
        raise_errors: bool = False,
    ) -> None:
        if isinstance(config, str):
            config = HCIConfig.from_profile(config)
        self.config: HCIConfig = (config or HCIConfig()).validate()
        logging.getLogger("open_ai_hci").setLevel(self.config.log_level.upper())
        self.bus = EventBus(raise_errors=raise_errors)
        self.raise_errors = raise_errors
        self._screen_size = screen_size
        self._hud = hud
        self._hud_active = False
        if hud is not None:
            hud.attach(self)

        self.sensors: list[BaseSensor] = []
        self.processors: list[BaseProcessor] = []
        self.actions: list[BaseAction] = []
        for s in sensors if sensors is not None else self._default_sensors():
            self.add_sensor(s)
        for p in processors if processors is not None else self._default_processors():
            self.add_processor(p)
        for a in actions if actions is not None else self._default_actions():
            self.add_action(a)

        self.frame_index = 0  # counts every captured frame, like the prototype's frame_count
        self.processed_frames = 0
        self._fps = FPSCounter()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._setup_done = False
        self._running = False
        self._error_counts: dict[str, int] = {}

    # ------------------------------------------------------------------ defaults
    def _default_sensors(self) -> list[BaseSensor]:
        cfg = self.config
        out: list[BaseSensor] = []
        if cfg.hand.enabled:
            out.append(HandSensor())
        if cfg.face.active:
            out.append(FaceSensor())
        if cfg.pose.enabled:
            out.append(PoseSensor())
        return out

    def _default_processors(self) -> list[BaseProcessor]:
        cfg = self.config
        procs: list[BaseProcessor] = [SmoothingProcessor(), GestureRulesProcessor()]
        # optional AI layers are imported lazily and switched on from the configuration
        if cfg.onnx.enabled and cfg.onnx.model_path:
            from .processors.ai.onnx_classifier import ONNXGestureClassifier

            procs.append(ONNXGestureClassifier.from_config(cfg.onnx))
        if cfg.llm.enabled:
            from .processors.ai.llm_router import LLMRouter

            procs.append(LLMRouter())
        if cfg.voice.enabled:
            from .processors.ai.voice_intent import VoiceIntentProcessor

            procs.append(VoiceIntentProcessor())
        return procs

    def _default_actions(self) -> list[BaseAction]:
        actions: list[BaseAction] = [OSMouseAction()] if self.config.mouse.enabled else []
        if self.config.llm.enabled or self.config.voice.enabled:  # hotkey / press / type intents
            actions.append(OSKeyboardAction())
        return actions

    # ------------------------------------------------------------------ composition
    def add_sensor(self, sensor: BaseSensor) -> BaseSensor:
        """Append a sensor (runs before processors). Returns it for chaining."""
        return self._add(self.sensors, sensor)

    def add_processor(self, processor: BaseProcessor) -> BaseProcessor:
        """Append a processor (runs after all sensors, in order)."""
        return self._add(self.processors, processor)

    def add_action(self, action: BaseAction) -> BaseAction:
        """Append an action. If the engine is already set up the action is set up immediately."""
        self._add(self.actions, action)
        if getattr(self, "_setup_done", False):
            action.setup()
        return action

    def _add(self, bucket: list[Any], component: C) -> C:
        component.attach(self)
        bucket.append(component)
        return component

    def remove(self, component_or_type: Component | type) -> int:
        """Remove a component instance, or every component of a type (e.g. ``OSMouseAction``). Returns the count."""
        removed = 0
        for bucket in (self.sensors, self.processors, self.actions):
            for comp in list(bucket):
                match = comp is component_or_type if not isinstance(component_or_type, type) else isinstance(comp, component_or_type)
                if match:
                    if self._setup_done:
                        self._close_one(comp)
                    bucket.remove(comp)
                    removed += 1
        return removed

    def get(self, component_type: type[C]) -> C | None:
        """First component of the given type, or ``None``."""
        for comp in (*self.sensors, *self.processors, *self.actions):
            if isinstance(comp, component_type):
                return comp
        return None

    # ------------------------------------------------------------------ events
    def on(self, name: EventName, callback: EventCallback | None = None, *, once: bool = False):
        """Subscribe to an event; usable as ``hci.on("pinch_start", fn)`` or as a decorator."""
        return self.bus.on(name, callback, once=once)

    def once(self, name: EventName, callback: EventCallback | None = None):
        """Subscribe for a single delivery."""
        return self.bus.once(name, callback)

    def off(self, name: EventName, callback: EventCallback | None = None) -> int:
        """Unsubscribe."""
        return self.bus.off(name, callback)

    def emit(self, name: str | Event, /, **data: Any) -> Event:
        """Publish an event from your own code."""
        return self.bus.emit(name, **data)

    # ------------------------------------------------------------------ properties
    @property
    def fps(self) -> float:
        """Processed frames per second over the last second."""
        return self._fps.fps

    @property
    def running(self) -> bool:
        return self._running

    @property
    def paused(self) -> bool:
        """``True`` while cursor control is paused (open-palm hold or :meth:`pause`)."""
        rules = self.get(GestureRulesProcessor)
        return rules.paused if rules is not None else False

    def pause(self) -> None:
        """Pause cursor control: tracking continues, but no cursor, pinch or scroll events are emitted.

        The pause gesture still works, so holding an open palm resumes. Thread-safe.
        """
        rules = self.get(GestureRulesProcessor)
        if rules is not None:
            rules.set_paused(True, via="api")

    def resume(self) -> None:
        """Resume cursor control after :meth:`pause` or the pause gesture. Thread-safe."""
        rules = self.get(GestureRulesProcessor)
        if rules is not None:
            rules.set_paused(False, via="api")

    def toggle_pause(self) -> bool:
        """Flip the pause state and return the new value."""
        self.resume() if self.paused else self.pause()
        return self.paused

    def apply_config(self, config: HCIConfig) -> None:
        """Swap in new settings while the engine is running (what the desktop GUI uses).

        Thresholds, smoothing, mouse behaviour and the LLM / HUD settings take effect immediately.
        Camera, MediaPipe and sensor choices need a restart. Components that define ``reconfigure()``
        are told to re-read the configuration.
        """
        self.config = config.validate()
        logging.getLogger("open_ai_hci").setLevel(self.config.log_level.upper())
        for comp in (*self.sensors, *self.processors, *self.actions):
            hook = getattr(comp, "reconfigure", None)
            if callable(hook):
                try:
                    hook()
                except Exception:
                    logger.exception("reconfigure() failed in %s", getattr(comp, "name", comp))

    @property
    def screen_size(self) -> tuple[int, int]:
        """Target screen size: config, then PyAutoGUI, then the camera resolution as a last resort."""
        if self._screen_size is None:
            scr = self.config.screen
            if scr.width and scr.height:
                self._screen_size = (scr.width, scr.height)
            else:
                try:
                    size = load_pyautogui(self.config.mouse.failsafe, self.config.mouse.pause).size()
                    self._screen_size = (int(size[0]), int(size[1]))
                except (ImportError, RuntimeError) as exc:
                    self._screen_size = (self.config.camera.width, self.config.camera.height)
                    logger.warning(
                        "Could not detect the screen size (%s); using %dx%d. Set HCIConfig.screen to override.",
                        str(exc).splitlines()[0], *self._screen_size,
                    )
        return self._screen_size

    @property
    def hud(self) -> HUD:
        if self._hud is None:
            self._hud = HUD()
            self._hud.attach(self)
        return self._hud

    # ------------------------------------------------------------------ lifecycle
    def setup(self) -> None:
        """Acquire resources for every component (idempotent; ``run`` calls it for you)."""
        if self._setup_done:
            return
        started: list[Component] = []
        try:
            for comp in (*self.sensors, *self.processors, *self.actions):
                comp.setup()
                started.append(comp)
        except BaseException:
            for comp in reversed(started):
                self._close_one(comp)
            raise
        self._setup_done = True

    def close(self) -> None:
        """Release everything (idempotent)."""
        if not self._setup_done and not self._hud_active:
            return
        if self._hud is not None and self._hud_active:
            self._close_one(self._hud)
            self._hud_active = False
        for comp in reversed((*self.sensors, *self.processors, *self.actions)):
            self._close_one(comp)
        self._setup_done = False

    @staticmethod
    def _close_one(comp: Component) -> None:
        try:
            comp.close()
        except Exception:
            logger.exception("Error while closing %s", getattr(comp, "name", comp))

    def __enter__(self) -> HCIEngine:
        self.setup()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        self.close()

    # ------------------------------------------------------------------ frame pipeline
    def process_frame(self, frame: np.ndarray, *, mirror: bool | None = None, timestamp: float | None = None) -> FrameContext:
        """Run one BGR frame through sensors -> processors -> actions and return its context.

        Use this to drive the engine from your own capture source (a game, a screen grabber,
        a test video) instead of :meth:`run`. ``mirror`` defaults to ``config.camera.mirror``.
        Every call counts as a processed frame (no frame skipping here).
        """
        self.setup()
        self.frame_index += 1
        return self._process(frame, mirror, timestamp)

    def _process(self, frame: np.ndarray, mirror: bool | None = None, timestamp: float | None = None) -> FrameContext:
        mirror = self.config.camera.mirror if mirror is None else mirror
        frame = cv2.flip(frame, 1) if mirror else frame.copy()
        height, width = frame.shape[:2]
        ts = time.monotonic() if timestamp is None else timestamp
        self.bus.frame_index = self.frame_index
        ctx = FrameContext(
            frame=frame,
            rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            width=width,
            height=height,
            frame_index=self.frame_index,
            timestamp=ts,
        )

        t0 = time.perf_counter()
        for sensor in self.sensors:
            self._guard(sensor, ctx)
        for processor in self.processors:
            self._guard(processor, ctx)
        ctx.extras["latency_ms"] = (time.perf_counter() - t0) * 1000.0

        self.processed_frames += 1
        ctx.extras["fps"] = self._fps.tick()
        if self.bus.has_subscribers("frame"):
            self.bus.emit("frame", ctx=ctx, fps=ctx.extras["fps"])
        return ctx

    def _guard(self, comp: BaseSensor | BaseProcessor, ctx: FrameContext) -> None:
        """Run a component; one failing plugin must not take the whole loop down."""
        try:
            comp.process(ctx)  # type: ignore[union-attr]
        except Exception as exc:
            if self.raise_errors:
                raise
            n = self._error_counts[comp.name] = self._error_counts.get(comp.name, 0) + 1
            if n == 1 or n % 100 == 0:  # don't flood the log at 30 fps
                logger.exception("%s failed on frame %d (occurrence %d)", comp.name, ctx.frame_index, n)
            self.bus.emit("error", component=comp.name, message=f"{type(exc).__name__}: {exc}", count=n)

    # ------------------------------------------------------------------ camera loop
    def _open_camera(self) -> cv2.VideoCapture:
        cam = self.config.camera
        cap = cv2.VideoCapture(cam.source, cam.backend) if cam.backend is not None else cv2.VideoCapture(cam.source)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not open camera/video source {cam.source!r}. Check the index, that no other app is "
                "using the camera, and your OS camera permissions."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.height)
        return cap

    def run(self, show_hud: bool | None = None, *, max_frames: int | None = None) -> None:
        """Run the capture loop until ``q`` (HUD), Ctrl+C, :meth:`stop`, end of video or the failsafe.

        Args:
            show_hud: Show the diagnostic window. ``None`` = ``config.hud.enabled``.
            max_frames: Stop after this many *captured* frames (handy for demos/tests).
        """
        show = self.config.hud.enabled if show_hud is None else show_hud
        cam = self.config.camera
        self._stop.clear()
        cap: cv2.VideoCapture | None = None
        started = False
        try:
            self.setup()
            hud: HUD | None = None
            if show:
                hud = self.hud
                hud.setup()
                self._hud_active = True
            cap = self._open_camera()
            self._running = started = True
            self._fps.reset()
            self.bus.emit("engine_start")
            frame_budget = 1.0 / cam.max_fps if cam.max_fps else 0.0

            while not self._stop.is_set():
                loop_start = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break
                self.frame_index += 1
                if self.frame_index % cam.frame_skip != 0:
                    if max_frames is not None and self.frame_index >= max_frames:
                        break
                    continue

                ctx = self._process(frame)
                if hud is not None:
                    key = hud.show(ctx)
                    ctx.key = key
                    if key >= 0:
                        self.bus.emit("key_press", key=key, char=chr(key) if 32 <= key < 127 else None)
                    if hud.is_quit_key(key):
                        break
                if max_frames is not None and self.frame_index >= max_frames:
                    break
                if frame_budget:
                    spare = frame_budget - (time.perf_counter() - loop_start)
                    if spare > 0:
                        time.sleep(spare)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            if cap is not None:
                cap.release()
            self._running = False
            if started:
                self.bus.emit("engine_stop")
            self.close()

    def start(self, show_hud: bool = False, *, max_frames: int | None = None) -> threading.Thread:
        """Run the loop on a background thread (non-blocking).

        Note: on macOS, OpenCV windows must live on the main thread, so keep ``show_hud=False``
        there or use :meth:`run`.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Engine is already running")
        self._thread = threading.Thread(
            target=self.run, kwargs={"show_hud": show_hud, "max_frames": max_frames}, name="open-ai-hci-engine", daemon=True
        )
        self._thread.start()
        return self._thread

    def join(self, timeout: float | None = None) -> None:
        """Wait for a thread started with :meth:`start`."""
        if self._thread is not None:
            self._thread.join(timeout)

    def stop(self) -> None:
        """Ask the loop to finish after the current frame (thread-safe)."""
        self._stop.set()

"""The desktop control panel's logic, with no GUI toolkit in sight.

:class:`ControlPanelModel` owns the :class:`~open_ai_hci.config.HCIConfig` being edited, maps the panel's
checkboxes / sliders / text fields onto it, applies changes to a running engine, saves ``config.yaml`` and tests the LLM
connection. :class:`EngineRunner` runs the engine on a background thread and reports ``Running`` / ``Paused`` /
``Stopped``. ``gui.py`` is only the thin tkinter skin on top, so everything here is unit-tested without a display.
"""
from __future__ import annotations

import copy
import logging
import re
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import LLM_PROVIDER_DEFAULTS, HCIConfig
from ..processors.ai.llm_router import ConnectionResult, LLMRouter, check_connection

__all__ = [
    "ProviderChoice",
    "PROVIDERS",
    "FLAGS",
    "SLIDERS",
    "EngineRunner",
    "ControlPanelModel",
    "provider_by_label",
    "provider_by_key",
]

logger = logging.getLogger("open_ai_hci")


# ------------------------------------------------------------------------------- providers
@dataclass(frozen=True)
class ProviderChoice:
    """One entry of the LLM provider dropdown."""

    label: str
    key: str  # LLMConfig.provider
    base_url: str  # auto-filled into the Base URL field ("" = the user types one)
    default_model: str
    cloud: bool


def _choice(label: str, key: str, *, cloud: bool, fill_url: bool = True) -> ProviderChoice:
    d = LLM_PROVIDER_DEFAULTS[key]
    return ProviderChoice(label, key, d["base_url"] if fill_url else "", d["model"], cloud)


PROVIDERS: tuple[ProviderChoice, ...] = (
    _choice("Ollama (Local)", "ollama", cloud=False),
    _choice("LM Studio (Local)", "lmstudio", cloud=False),
    _choice("Custom OpenAI-Compatible URL", "openai_compatible", cloud=False, fill_url=False),
    _choice("OpenAI (Cloud)", "openai", cloud=True),
    _choice("Anthropic (Cloud)", "anthropic", cloud=True),
    _choice("Google Gemini (Cloud)", "gemini", cloud=True),
)


def provider_by_label(label: str) -> ProviderChoice:
    for p in PROVIDERS:
        if p.label == label:
            return p
    raise KeyError(f"Unknown provider label {label!r}")


def provider_by_key(key: str) -> ProviderChoice:
    for p in PROVIDERS:
        if p.key == key:
            return p
    raise KeyError(f"Unknown provider {key!r}")


# ------------------------------------------------------------------------ checkboxes / sliders
#: Check-box name -> human label. Their getters / setters are in :class:`ControlPanelModel`.
FLAGS: dict[str, str] = {
    "hand_cursor": "Hand Cursor",
    "left_click": "Pinch Left-Click",
    "right_pinch": "Middle-Pinch Right-Click",
    "palm_pause": "Open-Palm Pause Toggle",
    "hud": "Diagnostic HUD overlay",
    "onnx_enabled": "Use custom ONNX gesture model",
    "llm_enabled": "Enable LLM intent router",
}

#: Slider name -> (label, minimum, maximum, resolution)
SLIDERS: dict[str, tuple[str, float, float, float]] = {
    "smoothing": ("Cursor Smoothing (EMA factor; lower = smoother but slower)", 0.05, 0.80, 0.01),
    "pinch": ("Pinch Threshold (px at 1280 wide)", 20.0, 80.0, 1.0),
    "onnx_confidence": ("Model confidence threshold", 0.50, 0.95, 0.01),
}


# --------------------------------------------------------------------------------- runner
class EngineRunner:
    """Runs an :class:`~open_ai_hci.engine.HCIEngine` on a background thread.

    Args:
        factory: ``callable(config) -> engine`` (default: build a real :class:`HCIEngine`). Tests inject fakes.
    """

    def __init__(self, factory: Callable[[HCIConfig], Any] | None = None) -> None:
        self._factory = factory or _default_factory
        self.engine: Any = None
        self._thread: threading.Thread | None = None
        self.error = ""
        self._lock = threading.Lock()

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def status(self) -> str:
        """``"Running"``, ``"Paused"``, ``"Stopped"`` (or ``"Starting"`` while the camera opens)."""
        if not self.alive or self.engine is None:
            return "Stopped"
        if not getattr(self.engine, "running", False):
            return "Starting"
        return "Paused" if self.engine.paused else "Running"

    @property
    def fps(self) -> float:
        return float(self.engine.fps) if self.alive and self.engine is not None else 0.0

    def start(self, config: HCIConfig, show_hud: bool = False) -> None:
        """Build an engine from ``config`` and run it. Raises ``RuntimeError`` if already running."""
        with self._lock:
            if self.alive:
                raise RuntimeError("The engine is already running")
            self.error = ""
            self.engine = self._factory(config)
            self._thread = threading.Thread(target=self._run, args=(show_hud,), name="open-ai-hci-gui-engine", daemon=True)
            self._thread.start()

    def _run(self, show_hud: bool) -> None:
        engine = self.engine
        try:
            engine.run(show_hud=show_hud)
        except Exception as exc:  # camera missing, model download blocked, no display ...
            self.error = f"{type(exc).__name__}: {exc}".splitlines()[0]
            logger.warning("Engine stopped with an error: %s", exc)
            try:
                engine.close()
            except Exception:  # pragma: no cover - best effort
                pass

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the loop to finish and wait for it (the engine releases the camera and any held mouse button)."""
        engine, thread = self.engine, self._thread
        if engine is not None:
            engine.stop()
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    def apply(self, config: HCIConfig) -> None:
        if self.alive and self.engine is not None:
            self.engine.apply_config(config)

    def toggle_pause(self) -> None:
        if self.alive and self.engine is not None:
            self.engine.toggle_pause()


def _default_factory(config: HCIConfig) -> Any:
    from ..engine import HCIEngine

    return HCIEngine(config)


# ---------------------------------------------------------------------------------- model
_LABEL_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]{0,39}$")


class ControlPanelModel:
    """What the control panel edits and does.

    Args:
        config: Starting configuration (default: loaded from ``config_path`` if it exists, else the defaults).
        config_path: Where :meth:`save_profile` writes (``config.yaml``).
        runner: Engine runner (tests inject a fake one).
        http / env: Injection points for :func:`check_connection` (mocked network in tests).
    """

    def __init__(
        self,
        config: HCIConfig | None = None,
        config_path: str | Path = "config.yaml",
        *,
        runner: EngineRunner | None = None,
        http: Any = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.load_error = ""
        if config is None:
            config = self._load_existing()
        self.cfg: HCIConfig = config
        self.runner = runner or EngineRunner()
        self._http, self._env = http, env
        self.api_key = self.cfg.llm.api_key  # lives in memory; only written to disk if the user opts in
        self.save_api_key = bool(self.cfg.llm.api_key)
        self._started_with: tuple[Any, ...] | None = None
        self.last_message = ""

    def _load_existing(self) -> HCIConfig:
        if self.config_path.is_file():
            try:
                return HCIConfig.load(self.config_path)
            except Exception as exc:  # a corrupt file must not stop the panel from opening
                self.load_error = f"Could not read {self.config_path.name}: {exc}"
                logger.warning(self.load_error)
        return HCIConfig()

    # ------------------------------------------------------------------ checkboxes
    def get_flag(self, name: str) -> bool:
        c = self.cfg
        if name == "hand_cursor":
            return c.mouse.move_cursor
        if name == "left_click":
            return c.mouse.left_click_on_pinch
        if name == "right_pinch":
            return c.hand.right_pinch_enabled and c.mouse.right_click_on_pinch
        if name == "palm_pause":
            return c.hand.pause_gesture != "none"
        if name == "hud":
            return c.hud.enabled
        if name == "onnx_enabled":
            return c.onnx.enabled
        if name == "llm_enabled":
            return c.llm.enabled
        raise KeyError(name)

    def set_flag(self, name: str, value: bool) -> str:
        """Change a check box. Returns an error message (empty when fine) and applies it to a running engine."""
        c, value = self.cfg, bool(value)
        if name == "hand_cursor":
            c.mouse.move_cursor = value
        elif name == "left_click":
            c.mouse.left_click_on_pinch = value
        elif name == "right_pinch":
            c.hand.right_pinch_enabled = c.mouse.right_click_on_pinch = value
        elif name == "palm_pause":
            c.hand.pause_gesture = ("open_palm" if c.hand.pause_gesture == "none" else c.hand.pause_gesture) if value else "none"
        elif name == "hud":
            c.hud.enabled = value
        elif name == "onnx_enabled":
            c.onnx.enabled = value
        elif name == "llm_enabled":
            c.llm.enabled = value
        else:
            raise KeyError(name)
        return self.apply_live()

    # ---------------------------------------------------------------------- sliders
    def get_value(self, name: str) -> float:
        c = self.cfg
        if name == "smoothing":
            return c.smoothing.alpha
        if name == "pinch":
            return c.hand.pinch_threshold_px
        if name == "onnx_confidence":
            return c.onnx.min_confidence
        raise KeyError(name)

    def set_value(self, name: str, value: float) -> str:
        """Set a slider value (clamped to its range) and apply it live."""
        _, lo, hi, step = SLIDERS[name]
        value = round(min(max(float(value), lo), hi) / step) * step
        value = round(value, 4)
        c = self.cfg
        if name == "smoothing":
            c.smoothing.kind, c.smoothing.alpha = "ema", value  # the slider is the EMA factor
        elif name == "pinch":
            c.hand.pinch_threshold_px = c.hand.right_pinch_threshold_px = value
        elif name == "onnx_confidence":
            c.onnx.min_confidence = value
        return self.apply_live()

    # ------------------------------------------------------------------ text fields
    @property
    def provider(self) -> ProviderChoice:
        return provider_by_key(self.cfg.llm.provider)

    def set_provider(self, label: str) -> tuple[str, str]:
        """Pick a provider from the dropdown. Auto-fills the base URL and model unless the user typed their own.

        Returns the ``(base_url, model)`` now in the config so the panel can refresh its fields.
        """
        new = provider_by_label(label)
        llm = self.cfg.llm
        known_urls = {p.base_url for p in PROVIDERS} | {""}
        known_models = {p.default_model for p in PROVIDERS} | {""}
        if llm.base_url.strip() in known_urls:
            llm.base_url = new.base_url
        if llm.model.strip() in known_models:
            llm.model = new.default_model
        llm.provider = new.key
        self.apply_live()
        return llm.base_url, llm.model

    def set_text(self, name: str, value: str) -> str:
        """``base_url`` / ``model`` / ``api_key`` / ``onnx_path`` text fields."""
        value = value.strip() if name != "api_key" else value
        if name == "base_url":
            self.cfg.llm.base_url = value
        elif name == "model":
            self.cfg.llm.model = value
        elif name == "api_key":
            self.api_key = value
        elif name == "onnx_path":
            self.cfg.onnx.model_path = value
        else:
            raise KeyError(name)
        return self.apply_live()

    def set_camera(self, source: str) -> None:
        """Camera index or video file / stream URL (applies on the next Start)."""
        source = source.strip()
        self.cfg.camera.source = int(source) if source.isdigit() else (source or 0)

    # --------------------------------------------------------------- runtime config
    def runtime_config(self) -> HCIConfig:
        """A validated copy for the engine: includes the API key typed into the (masked) field."""
        cfg = copy.deepcopy(self.cfg)
        cfg.llm.api_key = self.api_key
        return cfg.validate()

    def apply_live(self) -> str:
        """Push the current settings to a running engine. Returns an error message or ``""``."""
        try:
            self.runtime_config()
        except ValueError as exc:
            return str(exc).replace("Invalid HCIConfig:\n  - ", "")
        self.runner.apply(self.runtime_config())
        return ""

    def _structure(self) -> tuple[Any, ...]:
        c = self.cfg
        return (c.onnx.enabled, c.onnx.model_path, c.llm.enabled, c.voice.enabled, c.hud.enabled, c.camera.source,
                c.face.active, c.hand.enabled)

    @property
    def restart_required(self) -> bool:
        """``True`` when a setting that is only read at start-up changed while the engine runs."""
        return self.runner.alive and self._started_with is not None and self._structure() != self._started_with

    # ---------------------------------------------------------------------- engine
    @property
    def hud_supported(self) -> bool:
        """OpenCV windows must live on the main thread on macOS, which the GUI's engine thread is not."""
        return sys.platform != "darwin"

    def start(self) -> str:
        """Start the engine from the current settings. Returns an error message or ``""``."""
        if self.runner.alive:
            return "The engine is already running."
        try:
            cfg = self.runtime_config()
        except ValueError as exc:
            return str(exc)
        self._started_with = self._structure()
        self.runner.start(cfg, show_hud=cfg.hud.enabled and self.hud_supported)
        return ""

    def stop(self) -> None:
        self.runner.stop()
        self._started_with = None

    @property
    def status(self) -> str:
        return self.runner.status

    # ------------------------------------------------------------------ profiles
    def save_profile(self, path: str | Path | None = None) -> Path:
        """Write every vision, ONNX, LLM and mouse setting to ``config.yaml``.

        The API key is **not** written unless ``save_api_key`` is set (prefer the ``OPENAI_API_KEY`` /
        ``ANTHROPIC_API_KEY`` / ``GEMINI_API_KEY`` environment variables).
        """
        cfg = copy.deepcopy(self.cfg)
        cfg.llm.api_key = self.api_key if self.save_api_key else ""
        cfg.validate()
        target = Path(path) if path is not None else self.config_path
        saved = cfg.save(target)
        self.config_path = saved
        return saved

    # ------------------------------------------------------------------------- LLM
    def test_connection(self) -> ConnectionResult:
        """Contact the configured provider (blocking; the GUI calls it from a worker thread)."""
        try:
            cfg = self.runtime_config()
        except ValueError as exc:
            return ConnectionResult(False, str(exc), self.cfg.llm.provider, self.cfg.llm.model)
        return check_connection(cfg.llm, http=self._http, env=self._env)

    def send_command(self, text: str) -> str:
        """Hand a typed command to the running engine's LLM router. Returns ``""`` on success, else a reason."""
        text = text.strip()
        if not text:
            return "Type a command first."
        engine = self.runner.engine if self.runner.alive else None
        router = engine.get(LLMRouter) if engine is not None else None
        if router is None:
            return "Start the engine with the LLM router enabled first."
        if not router.available:
            return router.unavailable_reason or "The LLM router is not available."
        return "" if router.submit(text, source="text") else "The model is busy; try again in a moment."

    # ------------------------------------------------------ record / train launchers
    def camera_busy_reason(self) -> str:
        return "Stop the engine first: the camera can only be used by one program at a time." if self.runner.alive else ""

    @staticmethod
    def record_command(label: str, samples: int = 150, out: str = "data/gestures.csv") -> list[str]:
        """argv for ``open-ai-hci record`` run with this Python. Raises ``ValueError`` for an unusable label / count."""
        if not _LABEL_RE.match(label or ""):
            raise ValueError("The label must be 1-40 letters, digits, '_' or '-' and not start with '-' (e.g. right_click).")
        if not 1 <= int(samples) <= 5000:
            raise ValueError("Samples must be between 1 and 5000.")
        return [sys.executable, "-m", "open_ai_hci", "record", "--label", label, "--samples", str(int(samples)), "--out", out]

    @staticmethod
    def train_command(data: str = "data/gestures.csv", out: str = "models/gestures.onnx") -> list[str]:
        if not data.strip() or not out.strip():
            raise ValueError("Choose the recording and the output file.")
        return [sys.executable, "-m", "open_ai_hci", "train", "--data", data, "--out", out]

    @staticmethod
    def launch(argv: list[str]) -> subprocess.Popen[str]:
        """Start a record / train subprocess with its output merged into one pipe (no shell involved)."""
        return subprocess.Popen(  # noqa: S603 - argv is built from validated parts, never through a shell
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )

    def adopt_trained_model(self, path: str | Path) -> None:
        """Point the ONNX settings at a freshly trained model and switch it on."""
        self.cfg.onnx.model_path = str(path)
        self.cfg.onnx.enabled = True
        self.apply_live()

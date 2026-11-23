"""Configuration dataclasses, validation, built-in calibration profiles and (de)serialisation.

The numeric defaults of :class:`HCIConfig` still match the original single-file
prototype (1280x720 capture, every 2nd frame processed, 0.20 EMA smoothing,
pinch < 45 px). Since v0.2.0 the *head flick* right-click is **off** by default (it
fired by accident); right-click is a thumb + middle-finger pinch or an AI gesture,
and holding an open palm pauses/resumes cursor control. The ``"prototype"`` profile
restores the v0.1 behaviour exactly::

    from open_ai_hci import HCIConfig

    cfg = HCIConfig()                          # desktop profile (v0.2)
    cfg = HCIConfig.from_profile("prototype")  # head-flick right-click, no pause gesture
    cfg = HCIConfig.from_profile("precise")    # One-Euro smoothing, steadier cursor
    cfg.hand.pinch_threshold_px = 55           # tweak anything
    cfg.save("my_profile.yaml")                # .json or .yaml (PyYAML optional)
    cfg = HCIConfig.load("my_profile.yaml")
"""
from __future__ import annotations

import copy
import dataclasses
import json
import logging
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._yaml import dump_yaml, load_yaml

__all__ = [
    "CameraConfig",
    "HandConfig",
    "FaceConfig",
    "PoseConfig",
    "SmoothingConfig",
    "MouseConfig",
    "ScreenConfig",
    "HUDConfig",
    "MediaPipeConfig",
    "OnnxConfig",
    "VoiceConfig",
    "LLMConfig",
    "LLM_ACTIONS",
    "LLM_PROVIDER_DEFAULTS",
    "HCIConfig",
    "PROFILES",
    "REFERENCE_WIDTH",
]

logger = logging.getLogger("open_ai_hci")

#: Pixel thresholds (pinch distance, flick travel) are defined for a 1280-px-wide
#: frame and scaled linearly for other capture widths. At 1280 the factor is 1.0.
REFERENCE_WIDTH = 1280


@dataclass
class CameraConfig:
    """Video source and capture settings."""

    source: int | str = 0  # camera index, or a video file / stream URL
    width: int = 1280
    height: int = 720
    mirror: bool = True  # selfie view; also what MediaPipe's handedness expects
    frame_skip: int = 2  # process 1 of every N frames (prototype: 2). 1 = every frame
    backend: int | None = None  # e.g. cv2.CAP_DSHOW on Windows; None = OpenCV default
    max_fps: float | None = None  # optional loop limiter; None = as fast as the camera


@dataclass
class HandConfig:
    """Hand tracking and hand-gesture thresholds."""

    enabled: bool = True
    max_hands: int = 1
    detection_confidence: float = 0.70
    tracking_confidence: float = 0.70
    model_complexity: int = 1  # legacy MediaPipe backend only
    primary: str = "first"  # "first" | "Left" | "Right": which hand drives the cursor
    pinch_threshold_px: float = 45.0  # pinch when distance < threshold (strict)
    pinch_release_margin_px: float = 0.0  # >0 adds hysteresis: release at threshold+margin
    drag_threshold_px: float = 12.0  # screen px of travel while pinched before "drag_start"
    scroll_enabled: bool = False  # two-finger scroll gesture (off to match the prototype)
    scroll_pixels_per_click: float = 18.0  # frame px of vertical finger travel per wheel click
    # --- v0.2: intentional right-click and a pause toggle --------------------------
    right_pinch_enabled: bool = True  # thumb + middle finger pinch = right click
    right_pinch_threshold_px: float = 45.0
    right_click_cooldown_s: float = 0.5  # minimum time between two right clicks (any source)
    pause_gesture: str = "open_palm"  # "open_palm" | "fist" | "none": hold to pause/resume cursor control
    pause_hold_s: float = 1.0  # how long the pose must be held


@dataclass
class FaceConfig:
    """Face tracking, head flick, blink and head-pose settings.

    Face tracking is **off** by default since v0.2.0 (the head-flick right-click caused accidental
    clicks). It switches itself on when any face feature is enabled (``flick_enabled``,
    ``enable_blink``, ``enable_head_pose``) or when ``enabled`` is set.
    """

    enabled: bool = False
    flick_enabled: bool = False  # emit head_flick events (the v0.1 right-click trigger)
    max_faces: int = 1
    detection_confidence: float = 0.70
    tracking_confidence: float = 0.50
    flick_threshold_px: float = 28.0  # |dx| must exceed this between processed frames
    flick_axis_ratio: float = 3.0  # ... and exceed |dy| * ratio
    flick_cooldown_s: float = 1.2
    reset_on_loss: bool = False  # True: forget previous nose position when the face is lost
    enable_blink: bool = False
    blink_ear_threshold: float = 0.21
    blink_min_frames: int = 2
    blink_max_frames: int = 15
    enable_head_pose: bool = False

    @property
    def active(self) -> bool:
        """``True`` when a face sensor is needed for any enabled feature."""
        return self.enabled or self.flick_enabled or self.enable_blink or self.enable_head_pose


@dataclass
class PoseConfig:
    """Full-body pose tracking (off by default)."""

    enabled: bool = False
    max_poses: int = 1
    model_complexity: int = 1  # legacy backend: 0/1/2; tasks backend: lite/full/heavy model
    detection_confidence: float = 0.50
    tracking_confidence: float = 0.50


@dataclass
class SmoothingConfig:
    """Cursor smoothing. ``kind``: ``"ema"`` | ``"one_euro"`` | ``"kalman"`` | ``"none"``."""

    kind: str = "ema"
    alpha: float = 0.20  # EMA factor (prototype: 0.20)
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.01
    one_euro_d_cutoff: float = 1.0
    kalman_process_noise: float = 2e5
    kalman_measurement_noise: float = 50.0
    screen_margin: float = 0.0  # fraction (0..0.45) of the frame ignored at each edge


@dataclass
class MouseConfig:
    """OS mouse output via PyAutoGUI."""

    enabled: bool = True
    failsafe: bool = True  # PyAutoGUI corner failsafe stops the engine and releases the button
    pause: float = 0.0  # pyautogui.PAUSE
    move_cursor: bool = True  # False: clicks/scroll/keys still work, the cursor is left alone
    left_click_on_pinch: bool = True
    right_click_on_pinch: bool = True  # honour "right_click" events (middle pinch / AI gesture)
    right_click_on_flick: bool = False  # v0.1 behaviour: click on every "head_flick" event
    handle_intents: bool = True  # react to "intent" events (voice / VLM / LLM agent)


@dataclass
class ScreenConfig:
    """Target screen size. ``None`` = auto-detect through PyAutoGUI."""

    width: int | None = None
    height: int | None = None


@dataclass
class HUDConfig:
    """Diagnostic overlay."""

    enabled: bool = False  # run(show_hud=...) overrides this
    window_name: str = "Natural Camera Frame Active"
    title: str = "System: Hand Tracker & Clicks Active"
    show_landmarks: bool = True  # fingertip / thumb / nose markers like the prototype
    show_skeleton: bool = False  # all 21 hand landmarks + bones, pose skeleton
    show_fps: bool = True
    show_extras: bool = True  # AI gesture label, head pose, blink EAR when available
    alert_seconds: float = 1.0
    quit_keys: list[str] = field(default_factory=lambda: ["q"])


@dataclass
class MediaPipeConfig:
    """How MediaPipe is loaded.

    ``backend``:
        * ``"auto"`` (default): the legacy ``mp.solutions`` API when the installed
          MediaPipe still ships it (identical to the prototype), otherwise the
          MediaPipe *Tasks* API.
        * ``"legacy"`` / ``"tasks"``: force one.

    The Tasks API needs ``.task`` model files. They are downloaded once into
    ``model_dir`` (default ``~/.cache/open_ai_hci/models``) unless you point
    ``hand_model_path`` / ``face_model_path`` / ``pose_model_path`` at local files.
    """

    backend: str = "auto"
    model_dir: str | None = None
    auto_download: bool = True
    hand_model_path: str | None = None
    face_model_path: str | None = None
    pose_model_path: str | None = None


@dataclass
class OnnxConfig:
    """A custom ONNX gesture model (``pip install "open-ai-hci[ai]"``)."""

    enabled: bool = False
    model_path: str = ""  # path to the .onnx file (a sibling <model>.labels.json is picked up)
    min_confidence: float = 0.70
    stable_frames: int = 3  # consecutive frames a label must win before it fires
    background_label: str = "none"
    right_click_label: str = "right_click"  # this gesture triggers a right click ("" = off)


@dataclass
class VoiceConfig:
    """Optional speech input (``pip install "open-ai-hci[voice]"``)."""

    enabled: bool = False
    mode: str = "rules_then_llm"  # "rules" | "llm" | "rules_then_llm"
    model_size: str = "tiny.en"  # faster-whisper model
    language: str | None = "en"
    energy_threshold: float = 0.015
    device: int | str | None = None


#: Default provider endpoints / models / key environment variables (see ``processors/ai/llm_router.py``).
LLM_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "llama3.2", "key_env": ""},
    "lmstudio": {"base_url": "http://localhost:1234/v1", "model": "local-model", "key_env": ""},
    "openai_compatible": {"base_url": "http://localhost:8000/v1", "model": "default", "key_env": "OPENAI_API_KEY"},
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "key_env": "OPENAI_API_KEY"},
    "anthropic": {"base_url": "https://api.anthropic.com", "model": "claude-haiku-5-5", "key_env": "ANTHROPIC_API_KEY"},
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-2.5-flash",
        "key_env": "GEMINI_API_KEY",
    },
}

#: Actions the router may dispatch unless ``LLMConfig.allowed_actions`` narrows them.
LLM_ACTIONS: tuple[str, ...] = ("click", "double_click", "right_click", "move", "scroll", "hotkey", "press", "type", "say")


@dataclass
class LLMConfig:
    """Optional LLM intent & action router (no extra dependencies: plain ``urllib``).

    ``provider``: ``"ollama"`` | ``"lmstudio"`` | ``"openai_compatible"`` | ``"openai"`` |
    ``"anthropic"`` | ``"gemini"``. Leave ``base_url`` / ``model`` empty to use the provider
    defaults (:data:`LLM_PROVIDER_DEFAULTS`). The API key is read from ``api_key`` or, better,
    from the environment variable named by ``api_key_env`` (default: ``OPENAI_API_KEY``,
    ``ANTHROPIC_API_KEY`` or ``GEMINI_API_KEY``). A cloud provider without a key is disabled
    gracefully with a log message.
    """

    enabled: bool = False
    provider: str = "ollama"
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    api_key_env: str = ""
    timeout_s: float = 20.0
    temperature: float = 0.0
    max_tokens: int = 256
    json_mode: bool = True  # ask OpenAI-compatible servers for a JSON object (retried without on HTTP 400)
    system_prompt: str = ""  # extra instructions appended to the built-in prompt
    user_context: dict[str, str] = field(default_factory=dict)  # e.g. {"email_signature": "Best, Sam"}
    allowed_actions: list[str] = field(default_factory=lambda: list(LLM_ACTIONS))
    blocked_hotkeys: list[str] = field(
        default_factory=lambda: ["ctrl+alt+delete", "win+r", "alt+f4", "cmd+q", "ctrl+shift+esc"]
    )
    max_actions: int = 5  # actions executed per response
    max_pending: int = 4  # queued requests before new ones are dropped
    dry_run: bool = False  # parse and emit, but mark intents so actions do not execute (see router docs)


# --------------------------------------------------------------------------- profiles
#: Built-in calibration profiles: partial overrides applied on top of the defaults.
PROFILES: dict[str, dict[str, Any]] = {
    "default": {},
    # exact v0.1.0 behaviour: head-flick right click, no pause gesture, no middle-finger pinch
    "prototype": {
        "face": {"enabled": True, "flick_enabled": True},
        "hand": {"right_pinch_enabled": False, "pause_gesture": "none"},
        "mouse": {"right_click_on_flick": True, "right_click_on_pinch": False},
    },
    "precise": {
        "camera": {"frame_skip": 1},
        "smoothing": {"kind": "one_euro", "one_euro_min_cutoff": 1.0, "one_euro_beta": 0.01},
        "hand": {"pinch_release_margin_px": 10.0},
    },
    "fast": {
        "camera": {"width": 640, "height": 480, "frame_skip": 1},
        "smoothing": {"kind": "ema", "alpha": 0.35},
        "hand": {"pinch_release_margin_px": 6.0},
    },
    "accessibility": {
        # steadier cursor, easier pinches and a longer pause hold for limited motor control
        "smoothing": {"kind": "one_euro", "one_euro_min_cutoff": 0.6, "one_euro_beta": 0.004},
        "hand": {
            "pinch_threshold_px": 60.0,
            "pinch_release_margin_px": 15.0,
            "right_pinch_threshold_px": 60.0,
            "pause_hold_s": 1.5,
        },
    },
}


@dataclass
class HCIConfig:
    """Top-level configuration. Every field has a sensible default."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    hud: HUDConfig = field(default_factory=HUDConfig)
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    onnx: OnnxConfig = field(default_factory=OnnxConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    log_level: str = "WARNING"

    # ------------------------------------------------------------- construction
    @classmethod
    def from_profile(cls, name: str) -> HCIConfig:
        """Build a config from a built-in profile (see :data:`PROFILES`)."""
        if name not in PROFILES:
            raise ValueError(f"Unknown profile {name!r}. Available: {', '.join(sorted(PROFILES))}")
        return cls.from_dict(copy.deepcopy(PROFILES[name]))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HCIConfig:
        """Build from a (possibly partial) nested dict. Unknown keys are ignored with a warning."""
        return _from_dict(cls, data)

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict (JSON/YAML friendly)."""
        return dataclasses.asdict(self)

    def with_overrides(self, **dotted: Any) -> HCIConfig:
        """Return a copy with dotted-path overrides, e.g. ``cfg.with_overrides(**{"hand.pinch_threshold_px": 50})``."""
        new = copy.deepcopy(self)
        for path, value in dotted.items():
            target: Any = new
            *parents, leaf = path.split(".")
            for p in parents:
                target = getattr(target, p)
            if not hasattr(target, leaf):
                raise AttributeError(f"Unknown config field {path!r}")
            setattr(target, leaf, value)
        new.validate()
        return new

    # -------------------------------------------------------------------- files
    def save(self, path: str | Path) -> Path:
        """Write to ``.json`` or ``.yaml``/``.yml`` (PyYAML is used if installed, else a built-in writer)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() in (".yaml", ".yml"):
            p.write_text(dump_yaml(self.to_dict()), encoding="utf-8")
        else:
            p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> HCIConfig:
        """Load a ``.json`` or ``.yaml``/``.yml`` file written by :meth:`save` (partial files are fine)."""
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            data = load_yaml(text)
        else:
            data = json.loads(text)
        cfg = cls.from_dict(data)
        cfg.validate()
        return cfg

    # --------------------------------------------------------------- validation
    def validate(self) -> HCIConfig:
        """Raise ``ValueError`` on impossible settings. Returns ``self`` for chaining."""
        errors: list[str] = []

        def check(cond: bool, msg: str) -> None:
            if not cond:
                errors.append(msg)

        c, h, f, s = self.camera, self.hand, self.face, self.smoothing
        check(c.width > 0 and c.height > 0, "camera.width/height must be positive")
        check(c.frame_skip >= 1, "camera.frame_skip must be >= 1")
        check(c.max_fps is None or c.max_fps > 0, "camera.max_fps must be positive or None")
        check(h.max_hands >= 1, "hand.max_hands must be >= 1")
        check(h.primary in ("first", "Left", "Right"), "hand.primary must be 'first', 'Left' or 'Right'")
        check(h.pinch_threshold_px > 0, "hand.pinch_threshold_px must be positive")
        check(h.pinch_release_margin_px >= 0, "hand.pinch_release_margin_px must be >= 0")
        check(h.scroll_pixels_per_click > 0, "hand.scroll_pixels_per_click must be positive")
        check(h.right_pinch_threshold_px > 0, "hand.right_pinch_threshold_px must be positive")
        check(h.right_click_cooldown_s >= 0, "hand.right_click_cooldown_s must be >= 0")
        check(h.pause_gesture in ("open_palm", "fist", "none"), "hand.pause_gesture must be open_palm|fist|none")
        check(h.pause_hold_s > 0, "hand.pause_hold_s must be positive")
        for name, v in (
            ("hand.detection_confidence", h.detection_confidence),
            ("hand.tracking_confidence", h.tracking_confidence),
            ("face.detection_confidence", f.detection_confidence),
            ("face.tracking_confidence", f.tracking_confidence),
        ):
            check(0.0 <= v <= 1.0, f"{name} must be within [0, 1]")
        check(f.max_faces >= 1, "face.max_faces must be >= 1")
        check(f.flick_threshold_px > 0, "face.flick_threshold_px must be positive")
        check(f.flick_axis_ratio > 0, "face.flick_axis_ratio must be positive")
        check(f.flick_cooldown_s >= 0, "face.flick_cooldown_s must be >= 0")
        check(f.blink_min_frames >= 1 and f.blink_max_frames >= f.blink_min_frames, "invalid face.blink_*_frames")
        check(s.kind in ("ema", "one_euro", "kalman", "none"), "smoothing.kind must be ema|one_euro|kalman|none")
        check(0.0 < s.alpha <= 1.0, "smoothing.alpha must be in (0, 1]")
        check(0.0 <= s.screen_margin < 0.5, "smoothing.screen_margin must be in [0, 0.5)")
        check(self.mediapipe.backend in ("auto", "legacy", "tasks"), "mediapipe.backend must be auto|legacy|tasks")
        o, v, m = self.onnx, self.voice, self.llm
        check(0.0 <= o.min_confidence <= 1.0, "onnx.min_confidence must be within [0, 1]")
        check(o.stable_frames >= 1, "onnx.stable_frames must be >= 1")
        check(v.mode in ("rules", "llm", "rules_then_llm"), "voice.mode must be rules|llm|rules_then_llm")
        check(m.provider in LLM_PROVIDER_DEFAULTS, f"llm.provider must be one of {', '.join(sorted(LLM_PROVIDER_DEFAULTS))}")
        check(m.timeout_s > 0, "llm.timeout_s must be positive")
        check(0.0 <= m.temperature <= 2.0, "llm.temperature must be within [0, 2]")
        check(m.max_tokens >= 1 and m.max_actions >= 1 and m.max_pending >= 1, "llm.max_tokens/max_actions/max_pending must be >= 1")
        bad = [a for a in m.allowed_actions if a not in LLM_ACTIONS]
        check(not bad, f"llm.allowed_actions has unknown action(s) {bad}; valid: {', '.join(LLM_ACTIONS)}")
        check(
            (self.screen.width is None) == (self.screen.height is None),
            "screen.width and screen.height must be set together",
        )
        if errors:
            raise ValueError("Invalid HCIConfig:\n  - " + "\n  - ".join(errors))
        return self


# ------------------------------------------------------------------ dict -> dataclass
def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping for {cls.__name__}, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in known:
            logger.warning("Ignoring unknown config key %r in %s", key, cls.__name__)
            continue
        hint = hints[key]
        if dataclasses.is_dataclass(hint):
            kwargs[key] = _from_dict(hint, value)  # raises TypeError unless `value` is a mapping
        else:
            kwargs[key] = value
    return cls(**kwargs)

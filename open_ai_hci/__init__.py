"""open-ai-hci: modular, real-time Human-Computer Interaction for Python.

Fast by default (OpenCV + MediaPipe + geometry), AI by choice (optional plugins).

Five lines to a working touchless mouse::

    from open_ai_hci import HCIEngine

    hci = HCIEngine()
    hci.on("pinch_start", lambda event: print("Pinched!", event))
    hci.run(show_hud=True)

Architecture: ``Sensors -> Processors -> Actions`` connected by a typed event bus.
"""
from ._version import __version__
from .actions import BaseAction, OSKeyboardAction, OSMouseAction, WebSocketBridge
from .config import (
    PROFILES,
    CameraConfig,
    FaceConfig,
    HandConfig,
    HCIConfig,
    HUDConfig,
    LLMConfig,
    MediaPipeConfig,
    MouseConfig,
    OnnxConfig,
    PoseConfig,
    ScreenConfig,
    SmoothingConfig,
    VoiceConfig,
)
from .engine import FPSCounter, HCIEngine
from .event_bus import Event, EventBus, EventType
from .processors import (
    BaseProcessor,
    EMAFilter,
    GestureRulesProcessor,
    KalmanFilter1D,
    OneEuroFilter,
    SmoothingProcessor,
)
from .processors.ai import Intent, LLMRouter, MultimodalAgent, ONNXGestureClassifier, VoiceListener
from .sensors import BaseSensor, FaceSensor, HandSensor, PoseSensor
from .types import CursorState, FaceData, FrameContext, HandData, HandLandmark, HeadPose, PoseData
from .ui import HUD

__all__ = [
    "__version__",
    # engine & events
    "HCIEngine", "FPSCounter", "Event", "EventBus", "EventType",
    # config
    "HCIConfig", "CameraConfig", "HandConfig", "FaceConfig", "PoseConfig", "SmoothingConfig",
    "MouseConfig", "ScreenConfig", "HUDConfig", "MediaPipeConfig", "OnnxConfig", "VoiceConfig", "LLMConfig", "PROFILES",
    # sensors
    "BaseSensor", "HandSensor", "FaceSensor", "PoseSensor",
    # processors
    "BaseProcessor", "SmoothingProcessor", "GestureRulesProcessor",
    "EMAFilter", "OneEuroFilter", "KalmanFilter1D",
    # optional AI plugins (import safely without their dependencies)
    "ONNXGestureClassifier", "MultimodalAgent", "VoiceListener", "Intent", "LLMRouter",
    # actions
    "BaseAction", "OSMouseAction", "OSKeyboardAction", "WebSocketBridge",
    # ui
    "HUD",
    # data types
    "FrameContext", "HandData", "FaceData", "PoseData", "CursorState", "HeadPose", "HandLandmark",
]

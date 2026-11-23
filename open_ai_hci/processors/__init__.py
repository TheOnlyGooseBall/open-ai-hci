"""Decision & intelligence layer."""
from .base import BaseProcessor
from .gesture_rules import (
    BlinkDetector,
    FlickDetector,
    GestureRulesProcessor,
    ScrollTracker,
    detect_head_flick,
    is_pinch,
    pinch_distance,
)
from .smoothing import (
    EMAFilter,
    Filter2D,
    IdentityFilter,
    KalmanFilter1D,
    OneEuroFilter,
    SmoothingProcessor,
    make_filter_2d,
    map_to_screen,
)

__all__ = [
    "BaseProcessor",
    "SmoothingProcessor",
    "GestureRulesProcessor",
    "EMAFilter",
    "OneEuroFilter",
    "KalmanFilter1D",
    "IdentityFilter",
    "Filter2D",
    "make_filter_2d",
    "map_to_screen",
    "pinch_distance",
    "is_pinch",
    "detect_head_flick",
    "FlickDetector",
    "BlinkDetector",
    "ScrollTracker",
]

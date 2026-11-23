"""OPTIONAL AI plugins.

Nothing here is imported by the core pipeline, and importing this package never
requires ``onnxruntime`` / ``torch`` / Whisper: each plugin imports its heavy
dependency lazily and degrades gracefully with an install hint::

    pip install "open-ai-hci[ai]"       # onnxruntime + websockets
    pip install "open-ai-hci[voice]"    # faster-whisper + sounddevice
    pip install "open-ai-hci[train]"    # scikit-learn + skl2onnx (custom gesture training)

The LLM router needs no package at all (plain HTTP via the standard library).
"""
from .llm_router import LLMError, LLMProvider, LLMRouter, check_connection, make_provider
from .multimodal_agent import DEFAULT_PHRASES, Intent, MultimodalAgent, VoiceListener, parse_intent
from .onnx_classifier import ONNXGestureClassifier
from .voice_intent import VoiceIntentProcessor

__all__ = [
    "ONNXGestureClassifier",
    "LLMRouter",
    "VoiceIntentProcessor",
    "LLMProvider",
    "LLMError",
    "make_provider",
    "check_connection",
    "MultimodalAgent",
    "VoiceListener",
    "Intent",
    "parse_intent",
    "DEFAULT_PHRASES",
]

"""Helpers for optional dependencies.

The core framework only needs OpenCV, MediaPipe, NumPy and PyAutoGUI. Everything
else (ONNX Runtime, PyTorch, websockets, Whisper, PyYAML) is optional and is only
imported when the feature that needs it is actually *used*. If the package is
missing, a :class:`MissingOptionalDependency` with a copy-pasteable install
command is raised instead of a bare ``ModuleNotFoundError``.
"""
from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

__all__ = ["MissingOptionalDependency", "install_hint", "is_available", "require"]

# import name -> (pip distribution name, extra of open-ai-hci that provides it)
_OPTIONAL: dict[str, tuple[str, str]] = {
    "onnxruntime": ("onnxruntime", "ai"),
    "websockets": ("websockets", "ai"),
    "torch": ("torch", "torch"),
    "faster_whisper": ("faster-whisper", "voice"),
    "sounddevice": ("sounddevice", "voice"),
    "yaml": ("pyyaml", "yaml"),
    "sklearn": ("scikit-learn", "train"),
    "skl2onnx": ("skl2onnx", "train"),
}


class MissingOptionalDependency(ImportError):
    """Raised when a feature needs an optional package that is not installed."""


def install_hint(module: str) -> str:
    """Return a human-readable install command for an optional module."""
    root = module.split(".")[0]
    dist, extra = _OPTIONAL.get(root, (root, "ai"))
    return f'pip install "open-ai-hci[{extra}]"   (or just: pip install {dist})'


def is_available(module: str) -> bool:
    """Cheaply check whether ``module`` can be imported, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def require(module: str, feature: str) -> ModuleType:
    """Import ``module`` or raise :class:`MissingOptionalDependency`.

    Args:
        module: Dotted import path, e.g. ``"onnxruntime"``.
        feature: Human-readable name of the feature that needs it, used in the
            error message.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        root = module.split(".")[0]
        raise MissingOptionalDependency(
            f"{feature} requires the optional package '{root}', which is not installed "
            f"or failed to import ({exc}).\nInstall it with:  {install_hint(module)}"
        ) from exc

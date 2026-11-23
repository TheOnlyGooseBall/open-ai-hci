"""Lazy, friendly PyAutoGUI loader.

``import pyautogui`` fails with confusing errors (``KeyError: 'DISPLAY'``, Xlib
errors, ...) on headless machines, so it is imported only when an action that
needs it is set up, and failures are turned into one clear message.
"""
from __future__ import annotations

from types import ModuleType

__all__ = ["load_pyautogui"]


def load_pyautogui(failsafe: bool = True, pause: float = 0.0) -> ModuleType:
    """Import PyAutoGUI and apply ``FAILSAFE`` / ``PAUSE`` (the prototype used ``True`` / ``0.0``)."""
    try:
        import pyautogui  # noqa: WPS433 - deliberate lazy import
    except ImportError as exc:
        raise ImportError("OS mouse/keyboard control needs PyAutoGUI. Install it with: pip install pyautogui") from exc
    except Exception as exc:  # no display, missing X11 libs, macOS permission problems, ...
        raise RuntimeError(
            "PyAutoGUI could not initialise (is there a desktop session / DISPLAY available?). "
            f"Original error: {type(exc).__name__}: {exc}\n"
            "Run on a machine with a display, or disable OS output with HCIConfig(mouse=MouseConfig(enabled=False))."
        ) from exc
    pyautogui.FAILSAFE = failsafe
    pyautogui.PAUSE = pause
    return pyautogui

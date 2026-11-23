"""Pluggable output bindings."""
from .base import BaseAction
from .os_keyboard import OSKeyboardAction
from .os_mouse import OSMouseAction
from .websocket_bridge import WebSocketBridge

__all__ = ["BaseAction", "OSMouseAction", "OSKeyboardAction", "WebSocketBridge"]

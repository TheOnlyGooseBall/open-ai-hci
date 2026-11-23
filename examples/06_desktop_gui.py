"""Example 6: the desktop control panel (tkinter, no extra dependencies).

    python examples/06_desktop_gui.py                 # same as:  open-ai-hci gui
    python examples/06_desktop_gui.py --autostart     # start the engine straight away
    python examples/06_desktop_gui.py --config my.yaml

Tabs: Engine Controls (Start / Stop, live Running / Paused / Stopped), Core Vision & Gestures (cursor, left click,
middle-pinch right click, open-palm pause, HUD, smoothing and pinch sliders), Custom ONNX AI Model (model picker,
confidence, Record / Train buttons) and LLM Integration (Ollama, LM Studio, OpenAI-compatible, OpenAI, Anthropic,
Gemini + Test Connection). "Save Profile" writes config.yaml, which `open-ai-hci --config config.yaml` can run.

No tkinter? Windows / macOS: re-run the Python installer and tick "tcl/tk". Debian / Ubuntu: sudo apt install python3-tk
"""
from __future__ import annotations

import sys

from open_ai_hci.ui.gui import main

if __name__ == "__main__":
    sys.exit(main())

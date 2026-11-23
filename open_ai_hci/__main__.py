"""Command line entry point: ``python -m open_ai_hci`` (or the ``open-ai-hci`` script).

Examples::

    open-ai-hci                                 # touchless mouse with the HUD
    open-ai-hci --profile precise               # One-Euro smoothing, every frame
    open-ai-hci --profile prototype             # v0.1 behaviour (head-flick right click)
    open-ai-hci --no-mouse                      # tracking + HUD only, cursor untouched
    open-ai-hci --config config.yaml            # settings saved by the GUI
    open-ai-hci doctor                          # check the installation
    open-ai-hci gui                             # desktop control panel (tkinter)
    open-ai-hci record --label right_click --samples 150    # record a custom gesture
    open-ai-hci train --data data/gestures.csv --out models/gestures.onnx
"""
from __future__ import annotations

import argparse
import importlib
import platform
import sys
from collections.abc import Sequence

from ._optional import is_available
from ._version import __version__
from .config import PROFILES, HCIConfig


def _doctor() -> int:
    print(f"open-ai-hci {__version__}  |  Python {platform.python_version()}  |  {platform.platform()}")
    ok = True

    def line(label: str, status: str, good: bool = True) -> None:
        nonlocal ok
        ok = ok and good
        print(f"  [{'ok' if good else '!!'}] {label:<22} {status}")

    for mod, label in (("cv2", "opencv-python"), ("numpy", "numpy"), ("mediapipe", "mediapipe")):
        try:
            m = importlib.import_module(mod)
            line(label, getattr(m, "__version__", "installed"))
        except Exception as exc:
            line(label, f"MISSING ({exc})", False)
    try:
        import mediapipe as mp

        if hasattr(mp, "solutions"):
            line("mediapipe backend", "legacy (mp.solutions), models are bundled")
        else:
            from ._models import MODEL_URLS, cache_dir

            line("mediapipe backend", "tasks (needs small .task model files)")
            for kind in ("hand", "face"):
                cached = (cache_dir() / MODEL_URLS[kind].rsplit("/", 1)[-1]).is_file()
                print(f"       {kind + ' model':<20} {'cached' if cached else 'not cached yet: downloaded on first run (or set MediaPipeConfig.' + kind + '_model_path)'}")
    except Exception:
        pass
    try:
        from .actions._pyautogui import load_pyautogui

        size = load_pyautogui().size()
        line("pyautogui / display", f"{size[0]}x{size[1]}")
    except Exception as exc:
        line("pyautogui / display", str(exc).splitlines()[0], False)

    print("Optional extras (not required for the core):")
    for mod, extra in (
        ("onnxruntime", "ai"),
        ("websockets", "ai"),
        ("faster_whisper", "voice"),
        ("sounddevice", "voice"),
        ("sklearn", "train"),
        ("skl2onnx", "train"),
        ("yaml", "yaml"),
    ):
        have = is_available(mod)
        print(f"  [{'ok' if have else '--'}] {mod:<22} {'installed' if have else f'pip install open-ai-hci[{extra}]'}")
    gui = is_available("tkinter")
    print(f"  [{'ok' if gui else '--'}] {'tkinter (desktop GUI)':<22} {'installed' if gui else 'not available: install your OS package (Linux: sudo apt install python3-tk)'}")
    print("LLM router: no package needed (plain HTTP); set OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY for cloud models,")
    print("            or run Ollama (http://localhost:11434) / LM Studio (http://localhost:1234) locally.")
    return 0 if ok else 1


SUBCOMMANDS = ("record", "train", "gui")


def _subcommand(name: str, argv: list[str]) -> int:
    """``record`` / ``train`` / ``gui`` own their arguments, so hand them over untouched."""
    if name == "record":
        from .record import main as sub_main
    elif name == "train":
        from .training import main as sub_main
    else:
        from .ui.gui import main as sub_main
    return sub_main(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] in SUBCOMMANDS:
        return _subcommand(args_list[0], args_list[1:])
    parser = argparse.ArgumentParser(
        prog="open-ai-hci",
        description="Touchless mouse demo and diagnostics.",
        epilog="Other commands: record (capture a custom gesture), train (train an ONNX model from it), gui (desktop control panel). "
        "Try `open-ai-hci record --help`.",
    )
    parser.add_argument("command", nargs="?", choices=["run", "doctor"], default="run")
    parser.add_argument("--version", action="version", version=f"open-ai-hci {__version__}")
    parser.add_argument("--profile", choices=sorted(PROFILES), help="built-in calibration profile")
    parser.add_argument("--config", help="path to a .json/.yaml config written by HCIConfig.save()")
    parser.add_argument("--camera", help="camera index or video file/URL")
    parser.add_argument("--no-hud", action="store_true", help="run without the preview window")
    parser.add_argument("--no-mouse", action="store_true", help="do not control the OS cursor")
    args = parser.parse_args(args_list)

    if args.command == "doctor":
        return _doctor()

    from .engine import HCIEngine

    if args.config:
        cfg = HCIConfig.load(args.config)
    else:
        cfg = HCIConfig.from_profile(args.profile or "default")
    if args.camera is not None:
        cfg.camera.source = int(args.camera) if args.camera.isdigit() else args.camera
    if args.no_mouse:
        cfg.mouse.enabled = False

    try:
        hci = HCIEngine(cfg)
        print(
            "open-ai-hci running. Pinch (thumb + index) = left click/drag, thumb + middle finger = right click, "
            "hold an open palm to pause/resume. Press 'q' to quit."
        )
        hci.run(show_hud=not args.no_hud)
    except (RuntimeError, ImportError, OSError) as exc:  # missing camera / display / model / dependency
        print(f"error: {exc}", file=sys.stderr)
        print("Run `python -m open_ai_hci doctor` to diagnose your setup.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

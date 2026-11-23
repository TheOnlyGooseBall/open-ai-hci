"""Example 4: your own gestures with an ONNX model (e.g. a peace sign = right click).

Step 1 - record samples (normalised, scale-invariant landmarks, ~15 s per label):

    open-ai-hci record --label right_click --samples 150
    open-ai-hci record --label none --samples 300          # relaxed / in-between poses: what NOT to fire on

Step 2 - train and export (scikit-learn + skl2onnx, runs at 60+ FPS with onnxruntime):

    pip install "open-ai-hci[ai,train]"
    open-ai-hci train --data data/gestures.csv --out models/gestures.onnx

Step 3 - run this script:

    python examples/04_ai_custom_gestures.py                          # prints recognised gestures only
    python examples/04_ai_custom_gestures.py --live                   # also lets the right_click gesture click
    python examples/04_ai_custom_gestures.py --model path/to/other.onnx --confidence 0.85

The gesture named by OnnxConfig.right_click_label ("right_click") becomes a normal ``right_click`` event, handled
exactly like the thumb + middle-finger pinch. Every other label is available as ``gesture_<label>`` events, so you
can bind it to anything, here: a hotkey for each label that is not "none"/"right_click" (dry-run unless --live).
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

from open_ai_hci import HCIConfig, HCIEngine, MouseConfig, OnnxConfig, ONNXGestureClassifier, OSKeyboardAction

# gesture label -> hotkey (edit to taste). Labels you did not record are simply never used.
HOTKEYS = {"peace": "alt+left", "thumbs_up": "ctrl+s", "fist": "esc"}


def build_engine(model: str = "models/gestures.onnx", *, live: bool = False, confidence: float = 0.75, classifier: Any = None) -> HCIEngine:
    """Create the engine. ``classifier`` lets tests / custom runtimes inject a ready-made ONNXGestureClassifier."""
    config = HCIConfig(mouse=MouseConfig(enabled=live))  # without --live the cursor and clicks stay untouched
    config.onnx = OnnxConfig(enabled=True, model_path=model, min_confidence=confidence, stable_frames=3, right_click_label="right_click")
    hci = HCIEngine(config)
    if classifier is not None:
        hci.remove(ONNXGestureClassifier)
        hci.add_processor(classifier)

    hci.on("gesture", lambda e: print(f"gesture: {e.gesture} ({e.confidence:.0%})"))
    hci.on("gesture_end", lambda e: print(f"   ({e.get('previous') or 'gesture'} ended)"))
    hci.on("right_click", lambda e: print(f"right click via {e.via}" + ("" if live else "  [dry run: pass --live to really click]")))

    bindings = {f"gesture_{label}": keys for label, keys in HOTKEYS.items()}
    hci.add_action(OSKeyboardAction(bindings, cooldown_s=1.0, dry_run=not live, handle_intents=False))
    return hci


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/gestures.onnx")
    p.add_argument("--confidence", type=float, default=0.75, help="minimum probability for a gesture to count (0.5-0.95)")
    p.add_argument("--live", action="store_true", help="really move the mouse, click and press the hotkeys")
    args = p.parse_args(argv)
    hci = build_engine(args.model, live=args.live, confidence=args.confidence)
    print(f"Model: {args.model}  | recognised gestures are printed. 'q' quits. {'[LIVE]' if args.live else '[dry run]'}")
    hci.run(show_hud=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

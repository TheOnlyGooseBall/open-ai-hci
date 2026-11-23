#!/usr/bin/env python3
"""Train a custom gesture model from a recording (thin wrapper around ``open-ai-hci train``).

    python tools/train_gesture_model.py --data data/gestures.csv --out models/gestures.onnx
    open-ai-hci train --data data/gestures.csv --out models/gestures.onnx    # same thing once installed

See ``open_ai_hci/training.py`` for all options.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from open_ai_hci.training import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

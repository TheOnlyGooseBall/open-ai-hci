#!/usr/bin/env python3
"""Record labelled hand landmarks (thin wrapper around ``open-ai-hci record``).

    python tools/record_gestures.py --label right_click --samples 150
    open-ai-hci record --label right_click --samples 150        # same thing once the package is installed

See ``open_ai_hci/record.py`` for all options.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from open_ai_hci.record import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

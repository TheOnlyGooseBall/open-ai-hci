"""``open-ai-hci record``: capture labelled hand landmarks from the webcam (or a video) for AI training.

Interactive (default)::

    open-ai-hci record --label right_click --samples 150
    open-ai-hci record --labels fist,open,point,peace,none --samples 200 --out data/gestures.csv

    keys:  1-9  choose the label        SPACE  start / stop recording (a countdown runs first)
           q    quit and save           recording stops by itself at --samples per label

Hands-free (e.g. from a video file of one gesture)::

    open-ai-hci record --label fist --source fist.mp4 --auto --samples 300

Rows are saved as **normalised, scale-invariant** 21-point landmarks (wrist-relative, divided by hand
size, left hands mirrored) so one model serves both hands at any distance from the camera; ``--raw``
keeps the raw MediaPipe coordinates instead. Train with ``open-ai-hci train`` (see ``training.py``).

Tips for a good dataset: record each label from several distances and angles, with both hands, in
different lighting, and add a ``none`` class with relaxed / in-between poses so the classifier learns
what *not* to fire on.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import cv2

from .config import HandConfig, HCIConfig
from .engine import HCIEngine
from .recording import GestureRecorder
from .sensors.hand_sensor import HandSensor
from .ui.hud import HUD

__all__ = ["build_parser", "main"]

WINDOW = "open-ai-hci gesture recorder"


def build_parser(prog: str = "open-ai-hci record") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--label", help="single label for this session, e.g. right_click")
    g.add_argument("--labels", help="comma-separated labels; switch with keys 1-9")
    p.add_argument("--out", default="data/gestures.csv", help="output file (.csv, .jsonl or .json); appended to if it exists")
    p.add_argument("--source", default="0", help="camera index or video file (default 0)")
    p.add_argument("--samples", type=int, default=150, help="samples per label; recording stops there (default 150)")
    p.add_argument("--countdown", type=float, default=3.0, help="seconds between SPACE and recording start")
    p.add_argument("--hand", choices=["Left", "Right", "any"], default="any", help="only record this hand")
    p.add_argument("--auto", action="store_true", help="record immediately, no window, stop after --samples (or end of video)")
    p.add_argument("--min-score", type=float, default=0.8, help="skip samples with lower handedness confidence")
    p.add_argument("--raw", action="store_true", help="store raw MediaPipe landmarks instead of normalised features")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    labels = [args.label] if args.label else [s.strip() for s in args.labels.split(",") if s.strip()]
    if not labels or len(labels) > 9 or len(set(labels)) != len(labels):
        print("Provide between 1 and 9 distinct labels.", file=sys.stderr)
        return 2
    if args.samples < 1:
        print("--samples must be at least 1.", file=sys.stderr)
        return 2

    cfg = HCIConfig()
    cfg.camera.source = int(args.source) if args.source.isdigit() else args.source
    cfg.camera.frame_skip = 1
    cfg.hand = HandConfig(max_hands=1, detection_confidence=0.7, tracking_confidence=0.7)
    engine = HCIEngine(cfg, sensors=[HandSensor()], processors=[], actions=[])
    hud = HUD(cfg.hud)
    hud.attach(engine)
    hud.cfg.show_skeleton = True
    hud.cfg.show_landmarks = False
    hud.cfg.title = "Gesture recorder"

    try:
        recorder = GestureRecorder(args.out, normalize=not args.raw)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    existing = dict(recorder.counts)
    current, recording, record_from = 0, args.auto, 0.0

    try:
        engine.setup()
        cap = engine._open_camera()
    except (RuntimeError, ImportError, OSError) as exc:
        recorder.close()
        print(f"error: {exc}", file=sys.stderr)
        print("Run `open-ai-hci doctor` to diagnose your setup.", file=sys.stderr)
        return 1

    kind = "raw landmarks" if args.raw else "normalised features"
    print(f"Recording {kind} to {Path(args.out).resolve()}  | labels: {', '.join(labels)}  | {args.samples} samples each")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            engine.frame_index += 1
            ctx = engine._process(frame)
            label = labels[current]
            hand = ctx.hand
            usable = hand is not None and hand.score >= args.min_score and (args.hand == "any" or hand.handedness == args.hand)
            done = recorder.counts[label] >= args.samples

            if recording and time.monotonic() >= record_from and usable and not done:
                recorder.add(label, hand)
                if recorder.counts[label] >= args.samples:
                    recording = False
                    recorder.flush()
                    print(f"  {label}: {args.samples} samples")

            all_done = all(recorder.counts[name] >= args.samples for name in labels)
            if args.auto:
                if all_done:
                    break
                if recorder.counts[labels[current]] >= args.samples:  # next unfinished label
                    current = next(i for i, name in enumerate(labels) if recorder.counts[name] < args.samples)
                    recording = True
                continue
            if all_done:
                break

            hud.draw(ctx)
            _overlay(ctx.frame, labels, current, recorder, args.samples, recording, record_from, usable)
            cv2.imshow(WINDOW, ctx.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                recording = not recording and not done
                if recording:
                    record_from = time.monotonic() + args.countdown
                recorder.flush()
            elif ord("1") <= key <= ord("9") and key - ord("1") < len(labels):
                current = key - ord("1")
                recording = False
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.auto:  # --auto never opens a window (and headless OpenCV builds cannot close one)
            try:
                cv2.destroyAllWindows()
            except cv2.error:  # pragma: no cover - build without GUI support
                pass
        recorder.close()
        engine.close()
    added = {k: v - existing.get(k, 0) for k, v in recorder.counts.items()}
    summary = ", ".join(f"{k}: {v}" for k, v in added.items() if v) or "nothing recorded"
    print(f"Saved {sum(added.values())} new samples to {args.out} ({summary}).")
    print(f"Next: open-ai-hci train --data {args.out} --out models/gestures.onnx")
    return 0


def _overlay(frame, labels, current, recorder, target, recording, record_from, usable) -> None:
    h, _ = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 34 - 22 * len(labels)), (400, h), (0, 0, 0), -1)
    y = h - 14 - 22 * len(labels)
    now = time.monotonic()
    if recording and now >= record_from:
        state = "REC"
    elif recording:
        state = f"GET READY {max(0.0, record_from - now):.0f}"
    else:
        state = "paused"
    color = (0, 0, 255) if state == "REC" else (0, 200, 255) if state.startswith("GET") else (200, 200, 200)
    cv2.putText(frame, f"{state}  hand: {'ok' if usable else 'not usable'}   [SPACE] toggle  [q] quit", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    for i, name in enumerate(labels):
        marker = ">" if i == current else " "
        count = recorder.counts[name]
        txt = f"{marker}[{i + 1}] {name}: {count}/{target}" + ("  done" if count >= target else "")
        cv2.putText(frame, txt, (10, y + 22 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if i == current else (255, 255, 255), 1)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

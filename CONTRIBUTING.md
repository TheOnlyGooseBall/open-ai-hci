# Contributing to open-ai-hci

Thanks for helping! The project is deliberately small and plugin-shaped, so most contributions are one new file: a **sensor**, a **processor**, an **action**, or a **trained model**.

## Ground rules

1. **Fast by default, AI by choice.** The core install must stay light and deterministic. Anything that needs a neural runtime, a network service or a big download is an *optional plugin*:
   - import the heavy dependency lazily, inside `setup()` or the method that needs it (never at module top level);
   - use `open_ai_hci._optional.require("pkg", "My feature")`, which raises an error carrying the `pip install "open-ai-hci[...]"` hint;
   - degrade gracefully (log the hint, disable the plugin) unless the user passes `strict=True`;
   - add the dependency to an extra in `pyproject.toml`, never to the core `dependencies`.
2. **Real-time first.** `process()` runs on the camera loop. Keep it in the low-millisecond range; push slow work (network, big models) to a thread and publish the result as an event when it is ready (see `MultimodalAgent`).
3. **Behaviour changes need tests.** The default thresholds (45 px left and right pinch, 1 s open-palm pause hold, 0.20 EMA, and for the opt-in head flick 28 px / 3× ratio / 1.2 s cooldown) are pinned by tests, including a replay of the `prototype` profile against the original prototype logic. Changing them needs a very good reason.
4. **No hardware in tests.** CI has no camera, display or microphone. Use the fakes in `tests/conftest.py` (`FakeHandBackend`, `FakeFaceBackend`, `FakePyAutoGUI`) or inject backends/modules via constructor arguments.

## Development setup

```bash
git clone https://github.com/TheOnlyGooseBall/open-ai-hci
cd open-ai-hci
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"      # includes the [ai] and [train] extras
pytest                    # whole suite, no camera needed (tkinter tests need a display: xvfb-run pytest on headless Linux)
ruff check open_ai_hci tests examples tools
python -m open_ai_hci doctor
```

Style: Python 3.10+, type hints on every public function, `from __future__ import annotations`, Google-style docstrings (`Args:` / `Returns:` / `Raises:`), dataclasses for data and configuration. `ruff` is the linter.

## How the pieces fit

Every component derives from `Component` and has the same lifecycle:

| Step | When | What to do there |
|---|---|---|
| `__init__` | construction | store parameters only, no I/O |
| `attach(engine)` | added to an engine | gives `self.engine`, `self.bus`, `self.config`, `self.emit(...)` |
| `setup()` | once, before the first frame | load models, open devices, start threads, subscribe to events |
| `process(ctx)` | every processed frame (sensors, processors) | read/write the `FrameContext`, emit events |
| `close()` | on shutdown, reverse order | release everything |

The `FrameContext` carries the frame (`ctx.frame` BGR, `ctx.rgb`), `ctx.hands` / `ctx.faces` / `ctx.poses`, the smoothed `ctx.cursor`, gesture state, and a free-form `ctx.extras` dict for your own data.

**Event payload keys must not be `name`, `data`, `timestamp`, `frame_index` or `source`** (they would be shadowed by the `Event` object, and the bus raises `ValueError`). Use e.g. `gesture=`, `origin=`.

## Adding a sensor

Sensors turn a frame into observations. Subclass `BaseSensor`, implement `process`:

```python
from open_ai_hci import BaseSensor, FrameContext


class BrightnessSensor(BaseSensor):
    """Publishes average frame brightness and lights_on / lights_off events on transitions."""

    name = "brightness"

    def __init__(self, dark_below: float = 40.0) -> None:
        super().__init__()
        self.dark_below = dark_below
        self._dark = False

    def process(self, ctx: FrameContext) -> None:
        level = float(ctx.rgb.mean())
        ctx.extras["brightness"] = level
        dark = level < self.dark_below
        if dark != self._dark:
            self._dark = dark
            self.emit("lights_off" if dark else "lights_on", level=level)
```

```python
hci = HCIEngine()
hci.add_sensor(BrightnessSensor())
```

For a model-backed sensor, follow `sensors/hand_sensor.py`: take an optional `backend=` argument (a duck-typed `detect(rgb, timestamp_ms)` / `close()`) so tests can inject a scripted fake, create the real detector in `setup()`, and use `emit_presence("thing", bool)` to get `thing_detected` / `thing_lost` for free. `sensors/pose_sensor.py` is the smallest complete example. If you add a new data type, put its dataclass in `types.py` and a field on `FrameContext`.

## Adding a processor (rules or AI)

Processors decide what sensor data *means*. Subclass `BaseProcessor`:

```python
from open_ai_hci import BaseProcessor, FrameContext
from open_ai_hci.processors.gesture_rules import finger_extended


class OpenPalmProcessor(BaseProcessor):
    """Emits open_palm_start / open_palm_end when all four fingers are extended."""

    name = "open_palm"
    FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))  # (tip, pip) landmark indices

    def __init__(self) -> None:
        self._open = False

    def process(self, ctx: FrameContext) -> None:
        hand = ctx.hand
        is_open = hand is not None and all(finger_extended(hand, tip, pip) for tip, pip in self.FINGERS)
        if is_open and not self._open:
            self.emit("open_palm_start", handedness=hand.handedness)
        elif not is_open and self._open:
            self.emit("open_palm_end")
        self._open = is_open
        ctx.extras["open_palm"] = is_open
```

Tips for **deterministic rules**: put the math in small pure functions (see `pinch_distance`, `detect_head_flick`, `is_open_palm`) and the state in tiny classes (`FlickDetector`, `HoldToggle`) so they can be unit tested without frames; scale pixel thresholds with `ctx.width / REFERENCE_WIDTH`; add a config field instead of a magic number.

### Adding an AI model

Neural classifiers are processors that live in `processors/ai/`:

- **A trained gesture model** needs no code: record data with `tools/record_gestures.py`, train, export to ONNX, load it with `ONNXGestureClassifier` (see the README walk-through). Share useful models with their `.labels.json` and a note on the training data.
- **A new model type** (TorchScript, TFLite, a cloud API...) should mirror `ONNXGestureClassifier`: lazy `require(...)` in `setup()`, a `session=`-style injection point for tests, debouncing (`stable_frames`), a `min_confidence`, graceful degradation with an install hint, and events named `gesture` / `gesture_<label>` / `gesture_end` so existing bindings keep working.
- Use `open_ai_hci.features.landmarks_to_features` for inputs so training and inference stay identical.
- Add a new optional extra to `pyproject.toml` if it needs a new dependency.

## Adding an action

Actions turn events into effects. Subclass `BaseAction` and return the event → handler map; subscription and cleanup are automatic:

```python
from open_ai_hci import BaseAction


class ConsoleAction(BaseAction):
    """Prints a line for every pinch and for lights going off."""

    name = "console"

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def subscriptions(self):
        return {"pinch_start": self.on_pinch, "lights_off": self.on_dark}

    def on_pinch(self, event) -> None:
        self.lines.append(f"pinch at {event.x:.0f},{event.y:.0f}")

    def on_dark(self, event) -> None:
        self.lines.append(f"dark ({event.level:.0f})")
```

```python
hci.add_action(ConsoleAction())
```

Guidelines for actions that touch the outside world: make external modules injectable (`pyautogui_module=` in `OSMouseAction`), release held state in `close()`, honour the failsafe, surface user-visible results with `self.emit("alert", text="...")`, and never block the loop (hand slow I/O to a thread, as `WebSocketBridge` does).

These three snippets are executed by `tests/test_extending.py`, so they stay correct.

## Tests

- One `tests/test_<area>.py` per area; use the helpers in `conftest.py` (`make_hand`, `make_face`, `make_ctx`, fakes, `bare_engine`).
- Pixel positions in helpers are exact: `make_hand(index_px=(600, 300), thumb_px=(600, 344))` yields a 44 px pinch at 1280×720.
- Tests for optional dependencies use `pytest.mark.skipif(not is_available("pkg"), ...)` and a second test that fakes the import failure to cover the graceful-degradation path.
- Network code is tested with an injected `http(method, url, headers, body, timeout)` function (see `FakeHTTP` in `tests/test_llm.py`) and, for the real `urllib` path, a throwaway local HTTP server. Never call a real LLM from a test.
- GUI logic lives in `ui/gui_model.py` and is tested without tkinter; `tests/test_gui.py` drives the real window and skips itself when `tkinter` or a display is missing.
- Tests that build an engine with MediaPipe sensors must clear `hci.sensors` (or inject fakes) so no model download is attempted.
- Run `pytest -W error::DeprecationWarning` before opening a PR.

## Pull requests

- Small and focused, with tests and docs (README table, docstrings) for anything user-visible.
- Describe how you tested on real hardware if the change touches tracking, the mouse or the HUD (OS, camera, MediaPipe version).
- Keep the diff free of unrelated reformatting; run `ruff check` first.

## Reporting issues

Include the output of `python -m open_ai_hci doctor`, your OS, camera, what you expected and what happened. For tracking problems, a short screen recording of the HUD helps a lot.

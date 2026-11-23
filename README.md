# open-ai-hci

**Turn any webcam into a programmable input device.** `open-ai-hci` is a modular Python framework for real-time Human-Computer Interaction: hand and face gestures in, mouse / keyboard / WebSocket events out, with a typed event bus in the middle.

> **Fast by default, AI by choice.** The core runs on deterministic computer vision only (OpenCV + MediaPipe + geometry), with no neural runtime to install and nothing that adds latency. Custom ONNX gesture classifiers, voice commands and VLM agents are optional plugins (`open-ai-hci[ai]`) that never load unless you ask for them.

```python
from open_ai_hci import HCIEngine

hci = HCIEngine()
hci.on("pinch_start", lambda event: print("Pinched!", event))
hci.run(show_hud=True)
```

That is a working touchless mouse: your index finger moves the cursor, a thumb + index pinch left-clicks and drags, a thumb + middle-finger pinch right-clicks, and holding an open palm for a second pauses or resumes control.

Prefer buttons to code? Run **`open-ai-hci gui`** for a desktop control panel (built-in `tkinter`, no extra dependencies).

## Use cases

Desktop apps and games (camera input without writing vision code), accessibility tools (hands-free pointing with tunable thresholds), robotics and installations (stream landmarks over WebSocket), spatial UIs (Unity / Unreal / Godot / web clients), research (record labelled landmarks, train your own classifier, drop it back in), and assistants that combine a gesture, the cursor position and a spoken or typed command through a local or cloud LLM.

---

## Install

```bash
git clone https://github.com/TheOnlyGooseBall/open-ai-hci
cd open-ai-hci
pip install -e .              # lightweight core: numpy, opencv-python, mediapipe, pyautogui
pip install -e ".[ai]"        # + onnxruntime, websockets  (custom gesture models, WebSocket bridge)
pip install -e ".[train]"     # + scikit-learn, skl2onnx, onnxruntime  (open-ai-hci train)
pip install -e ".[voice]"     # + faster-whisper, sounddevice  (voice commands)
pip install -e ".[all]"       # everything optional
```

Check your setup at any time:

```bash
python -m open_ai_hci doctor
```

Requires Python 3.10+, a webcam, and (for OS mouse control) a desktop session. The desktop GUI needs `tkinter`, which ships with Python on Windows and macOS (Debian / Ubuntu: `sudo apt install python3-tk`). LLM routing needs **no** extra package: it uses the standard library only. See [Troubleshooting](#troubleshooting) for macOS permissions and Linux/Wayland notes.

## Try it

```bash
python -m open_ai_hci                    # the prototype behaviour with the diagnostic HUD
python -m open_ai_hci --profile precise  # One-Euro smoothing, every frame processed
python -m open_ai_hci --no-mouse         # tracking + HUD only; your cursor is untouched
python -m open_ai_hci gui                # desktop control panel (same as examples/06_desktop_gui.py)
python examples/01_mouse_hud_demo.py
```

Press **q** in the preview window to quit. **Safety:** PyAutoGUI's failsafe is on: throw the *physical* mouse into any screen corner to stop immediately (any held button is released).

---

## How it works

```
 Camera ──► frame (mirrored, BGR + RGB)
              │
   ┌──────────▼───────────┐   HandSensor · FaceSensor · PoseSensor · your sensor
   │       SENSORS        │   → fill ctx.hands / ctx.faces / ctx.poses / ctx.extras
   └──────────┬───────────┘
   ┌──────────▼───────────┐   SmoothingProcessor        (EMA / One-Euro / Kalman cursor)
   │      PROCESSORS      │   GestureRulesProcessor     (pinch, right pinch, pause, drag, scroll, blink)
   │  decide what it MEANS│   ONNXGestureClassifier  ┐  optional AI plugins,
   └──────────┬───────────┘   LLMRouter · VoiceIntent ┘  loaded only if you add them
              │  emits typed events
   ┌──────────▼───────────┐
   │      EVENT  BUS      │◄── hci.on("pinch_start", fn)   hci.on("*", fn)   hci.emit(...)
   └──────────┬───────────┘
   ┌──────────▼───────────┐   OSMouseAction · OSKeyboardAction · WebSocketBridge
   │       ACTIONS        │   your action (print, robot, game, MQTT, ...)
   └──────────────────────┘
              HUD (optional) draws diagnostics on the frame and shows the window
```

Everything is a small class with a three-method lifecycle (`attach` → `setup` → `close`) and one job. The engine owns the loop, the FPS counter and error isolation: a plugin that raises is logged and reported as an `error` event, and the loop keeps running.

### Default behaviour (v0.2)

| Behaviour | Default | Config key |
|---|---|---|
| Capture | 1280×720, mirrored | `camera.width/height/mirror` |
| Frame skipping | process every 2nd frame | `camera.frame_skip` |
| Hands | 1 hand, detection/tracking confidence 0.70 | `hand.*` |
| Cursor | index fingertip (landmark 8), EMA smoothing **0.20** | `smoothing.kind/alpha` |
| Left click / drag | thumb tip (4) to **index** tip (8) distance **< 45 px** = button held | `hand.pinch_threshold_px` |
| **Right click** | thumb tip (4) to **middle** tip (12) distance **< 45 px** (a quick tap; 0.5 s cooldown) | `hand.right_pinch_*`, `mouse.right_click_on_pinch` |
| **Pause / resume** | hold an **open palm** for **1 s** (or set `hand.pause_gesture = "fist"`; `"none"` turns it off) | `hand.pause_gesture`, `hand.pause_hold_s` |
| AI right click | a model label called `right_click` (e.g. a peace sign) | `onnx.right_click_label` |
| Hand lost / paused | held button is released, cursor stays put | |
| HUD | dark info box, colour-coded pinch state (green = holding, yellow = right click, magenta = paused, orange = open, red = no hand), pause progress bar and banner, 1 s action banner | `hud.*` |
| Failsafe | PyAutoGUI corner failsafe on | `mouse.failsafe` |

When two fingers touch the thumb at once, the closer one wins and keeps ownership until released, so a left drag never turns into a right click halfway through.

**Head-flick right click was removed from the default profile.** A nodding or head-turning user kept clicking by accident, and it needed a face model running on every frame. The face sensor is now **off** unless you ask for it (`face.enabled = True`; blink and head pose still work). To bring the flick back: `HCIConfig(face=FaceConfig(enabled=True, flick_enabled=True), mouse=MouseConfig(right_click_on_flick=True))`, or use `HCIConfig.from_profile("prototype")` for the exact v0.1.0 behaviour (flick on, no middle pinch, no pause gesture).

**Deliberate differences from the single-file prototype** (all configurable):

- The first cursor sample is used as-is instead of gliding in from the top-left corner (the prototype's EMA started at `(0, 0)`).
- Cursor targets are kept 2 px inside the screen (`OSMouseAction(clamp_margin=...)`) so a hand at the edge of the camera view cannot trip the failsafe by accident.
- Pixel thresholds scale with the capture width relative to 1280 (factor 1.0 at 1280 px), so a camera that refuses 1280×720 behaves consistently.
- A clear error replaces the silent failure when the camera cannot be opened.
- Mouse-wheel amounts are converted to the platform's unit (Windows counts 120 per notch), so "scroll down" scrolls a sensible amount everywhere.
- Additions: `pinch_end` / `drag_*` / `right_pinch_*` / `right_click` / `control_paused` / `control_resumed` events and everything under [AI plugins](#ai-plugins-optional).

The `prototype` profile is pinned by a test that replays 600 random frames through both the framework and a verbatim copy of the prototype's logic and asserts identical mouse and click decisions.

---

## Events

Subscribe with `hci.on(name, handler)`, as a decorator (`@hci.on("pinch_start")`), with `hci.once(...)`, or to everything with `hci.on("*", handler)` (the per-frame `frame` event is excluded from `"*"`). Payload keys are readable as attributes: `event.x`, `event.get("x")` or `event.data["x"]`. Misspelled built-in names (`"pinch_strat"`) trigger a warning that suggests the right one.

| Event | Payload | Fired when |
|---|---|---|
| `hand_detected` / `hand_lost` | `count` | first hand appears / last disappears |
| `face_detected` / `face_lost` | `count` | same for faces |
| `pose_detected` / `pose_lost` | `count` | same for full-body pose (opt-in sensor) |
| `cursor_move` | `x, y, raw_x, raw_y` | every processed frame with a hand (smoothed + raw screen px) |
| `pinch_start` | `x, y, distance, handedness` | thumb-index distance drops below the threshold |
| `pinch_end` | `reason ("released"/"hand_lost"), x, y` | pinch released |
| `drag_start` / `drag_move` / `drag_end` | `x, y, start_x, start_y` / `x, y, dx, dy` / `reason, x, y` | cursor travels `hand.drag_threshold_px` while pinched |
| `right_pinch_start` / `right_pinch_end` | `x, y, distance` / `reason, x, y` | thumb + middle finger touch / release |
| `right_click` | `x, y, via ("pinch" / "flick" / "ai_gesture")` | any right-click source fired (the mouse action listens to this) |
| `control_paused` / `control_resumed` | `via` | the pause pose was held, or `hci.pause()` / `hci.resume()` was called |
| `head_flick`, `head_flick_left`, `head_flick_right` | `direction, dx, dy, x, y` | sharp horizontal nose flick (opt-in: `face.flick_enabled`) |
| `blink` | `ear` | a blink (`face.enable_blink = True`) |
| `scroll_start` / `scroll` / `scroll_end` | `amount` (wheel clicks, + is up) | two-finger scroll pose (`hand.scroll_enabled = True`) |
| `gesture`, `gesture_<label>`, `gesture_end` | `gesture, confidence, previous, handedness` | ONNX classifier |
| `voice_command` / `intent` / `vlm_result` | see [voice and intent fusion](#voice-vlm-and-intent-fusion) | voice / VLM / LLM plugins |
| `llm_response` / `llm_error` | `text, actions, rejected, latency_ms` / `message` | the [LLM router](#llm-intent--action-router) answered or failed |
| `alert` | `text` | anything worth showing in the HUD banner |
| `frame` | `ctx, fps` | every processed frame (subscribe by name) |
| `key_press` | `key, char` | key pressed in the HUD window |
| `engine_start` / `engine_stop` / `error` | – / – / `component, message, count` | lifecycle and plugin failures |

Event objects also carry `name`, `timestamp`, `frame_index` and `source`. Payload keys with those names are rejected at emit time, because they would be shadowed.

```python
@hci.on("drag_move")
def on_drag(event):
    canvas.move_to(event.x, event.y)

hci.on("head_flick_left", lambda e: print("back"))
hci.once("hand_detected", lambda e: print("hello"))

hci.bus.register_event("my_event")          # declare custom events (silences the typo check)
hci.emit("my_event", answer=42)
```

## Configuration

```python
from open_ai_hci import HCIConfig, HCIEngine

cfg = HCIConfig()                          # prototype defaults
cfg = HCIConfig.from_profile("precise")    # "default" | "precise" | "fast" | "accessibility"
cfg.hand.pinch_threshold_px = 55
cfg.smoothing.kind = "one_euro"            # "ema" | "one_euro" | "kalman" | "none"
cfg.smoothing.screen_margin = 0.1          # reach the screen corners without stretching your arm
cfg.camera.source = "recording.mp4"        # camera index, video file, or stream URL

cfg.save("my_profile.json")                # .yaml / .yml works too (pip install pyyaml)
hci = HCIEngine(HCIConfig.load("my_profile.json"))     # or: HCIEngine("accessibility")
```

The desktop GUI writes the same format (`config.yaml`; run it with `open-ai-hci --config config.yaml`). Pause and resume from code with `hci.pause()`, `hci.resume()`, `hci.toggle_pause()` and `hci.paused`; push new settings into a running engine with `hci.apply_config(cfg)`.

Invalid values are rejected with a readable list of every problem (`cfg.validate()` runs automatically on load). All options are plain dataclasses under `open_ai_hci/config.py`.

**Profiles:** `precise` = One-Euro smoothing + every frame + pinch hysteresis; `fast` = 640×480, lighter smoothing; `accessibility` = steadier cursor, easier pinch (60 px), longer pause hold; `prototype` = exact v0.1.0 behaviour (head-flick right click).

### Smoothing

| `smoothing.kind` | Character |
|---|---|
| `ema` (default) | the prototype's exponential average, `y += (x − y) · alpha` |
| `one_euro` | adaptive: steady when you hold still, low-lag when you move fast. Tune `one_euro_min_cutoff` (lower = steadier) and `one_euro_beta` (higher = less lag in fast motion) |
| `kalman` | constant-velocity Kalman filter: best jitter rejection, a little lag |
| `none` | raw |

## Compose your own pipeline

```python
from open_ai_hci import (HCIEngine, HCIConfig, MouseConfig, HandSensor, OSKeyboardAction,
                         WebSocketBridge, ONNXGestureClassifier)

hci = HCIEngine(HCIConfig(mouse=MouseConfig(enabled=False)))    # events only, cursor untouched

hci.add_action(OSKeyboardAction({
    "head_flick_left":  "alt+left",                              # hotkey
    "head_flick_right": "alt+right",
    "blink":            [("hotkey", "ctrl", "a"), ("press", "delete")],   # macro
    "pinch_start":      lambda e: my_app.select(e.x, e.y),       # or any Python callable
}, cooldown_s=1.0))
hci.add_action(WebSocketBridge(port=8765, stream_landmarks=True))
hci.add_processor(ONNXGestureClassifier("fist_open_peace.onnx"))

hci.run()
```

Pass `sensors=`, `processors=` or `actions=` to the constructor to replace a whole layer, and use `hci.remove(OSMouseAction)` / `hci.get(HandSensor)` to adjust the defaults. To feed your own frames (a game, a screen grabber) skip the camera loop:

```python
with HCIEngine() as hci:
    ctx = hci.process_frame(bgr_frame)     # runs sensors → processors → actions, returns the FrameContext
    print(ctx.hand.index_tip, ctx.pinch_status)
```

`hci.start()` runs the loop on a background thread (keep the HUD off on macOS, where OpenCV windows need the main thread).

---

## AI plugins (optional)

Nothing in this section is imported by the core. Importing `open_ai_hci` never needs `onnxruntime`, `websockets`, Whisper or PyTorch, and a missing dependency produces a message like

```
ONNXGestureClassifier requires the optional package 'onnxruntime' ...
Install it with:  pip install "open-ai-hci[ai]"   (or just: pip install onnxruntime)
```

The classifier and agent *degrade gracefully* (log the hint and switch themselves off) unless you pass `strict=True`; the WebSocket bridge raises the hint when you add it.

### Train your own gesture classifier

1. **Record** labelled landmarks. Samples are saved as *normalised, scale-invariant* 21-point features (wrist-relative, divided by hand size, left hands mirrored), so one model serves both hands at any distance from the camera:

   ```bash
   open-ai-hci record --label right_click --samples 150          # e.g. a peace sign
   open-ai-hci record --label none --samples 300                 # relaxed poses: what NOT to fire on
   open-ai-hci record --labels fist,open,peace,none --samples 200 --out data/gestures.csv
   # keys: 1-9 choose label · SPACE start/stop (3 s countdown) · q save and quit; stops by itself at --samples
   # also: --label fist --source clip.mp4 --auto   (record from a video, no window) · --raw keeps raw coordinates
   ```

   Record each gesture from several distances and angles with both hands, and always add a `none` class. Recordings are appended, so you can add labels in separate sessions.

2. **Train and export** (`pip install "open-ai-hci[train]"`):

   ```bash
   open-ai-hci train --data data/gestures.csv --out models/gestures.onnx
   # writes models/gestures.onnx and models/gestures.labels.json, prints accuracy per label,
   # and benchmarks inference (a single sample takes well under a millisecond with onnxruntime)
   ```

   `--model mlp|logreg|forest` picks the classifier. From Python: `from open_ai_hci.training import train_model`. `python tools/train_gesture_model.py` is the same thing. Using PyTorch instead? Export with `torch.onnx.export` to a model with input `(1, 63)` and probabilities or logits out; `open_ai_hci.features` is shared by recorder and classifier so training and inference can never drift.

3. **Use it:**

   ```python
   hci.add_processor(ONNXGestureClassifier("models/gestures.onnx", min_confidence=0.7, stable_frames=3))
   hci.on("gesture_peace", lambda e: print("peace!", e.confidence))
   ```

   or just set `cfg.onnx = OnnxConfig(enabled=True, model_path="models/gestures.onnx")`. Predictions are debounced (`stable_frames`), thresholded (`min_confidence`), and the `none` label is treated as "no gesture". **A label called `right_click` (`onnx.right_click_label`) becomes a normal `right_click` event**, handled exactly like the middle-finger pinch; every other label is available as `gesture_<label>`. `python examples/04_ai_custom_gestures.py` wires the whole thing up.

### LLM intent / action router

`LLMRouter` turns **a command (typed or spoken) + your current hand gesture + the cursor position** into structured actions, and the existing mouse and keyboard actions execute them. It works with local and cloud models through plain `urllib`; there is nothing to install.

```python
from open_ai_hci import HCIEngine, HCIConfig, LLMConfig, LLMRouter

cfg = HCIConfig(llm=LLMConfig(enabled=True, provider="ollama", model="llama3.2"))
hci = HCIEngine(cfg)                      # the router joins the default pipeline when llm.enabled
hci.get(LLMRouter).submit("open this in a new tab", source="text")
# model says {"actions":[{"action":"hotkey","keys":["ctrl","t"]}]}  ->  intent events  ->  OSKeyboardAction
```

| `llm.provider` | Default endpoint | Key |
|---|---|---|
| `ollama` | `http://localhost:11434/v1` (OpenAI-compatible; a URL without `/v1` uses the native `/api/chat`) | none |
| `lmstudio` | `http://localhost:1234/v1` | none |
| `openai_compatible` | your `base_url` (vLLM, llama.cpp, LocalAI, ...) | optional |
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `anthropic` | `https://api.anthropic.com` (`/v1/messages`) | `ANTHROPIC_API_KEY` |
| `gemini` | `https://generativelanguage.googleapis.com/v1beta` | `GEMINI_API_KEY` |

Actions the model may return: `click`, `double_click`, `right_click`, `move`, `scroll` (`{"action":"scroll","amount":-500}`: 100 is about one wheel notch, negative scrolls down), `hotkey` (`{"action":"hotkey","keys":["ctrl","t"]}`), `press`, `type` and `say`. Examples: *"open this in a new tab"* (the model sees the pointer position and gesture), *"scroll down slowly"*, *"type my email signature"* (set `llm.user_context = {"email_signature": "Best, Sam"}`).

- **Never blocks the 60 FPS loop.** Requests run on one background worker thread; results come back as events (`llm_response`, `llm_error`, `intent`). If the model is slow, extra requests are dropped (`llm.max_pending`), not queued forever.
- **Disabled gracefully.** No key for a cloud provider, or no server on the port: the router logs a one-line reason, switches itself off and the rest of the engine carries on.
- **The model is not trusted.** Answers are validated: unknown actions are rejected, hotkeys in `llm.blocked_hotkeys` (`ctrl+alt+delete`, `win+r`, `alt+f4`, `cmd+q`, `ctrl+shift+esc`) are refused, `type` text is capped at 500 characters, scrolls are clamped, at most `llm.max_actions` run per answer, and `llm.allowed_actions` can narrow the list further. Use `llm.dry_run = True` to see what it *would* do.
- **Keys stay out of files and logs.** Keys come from the environment variables above (or `llm.api_key`, or the GUI field). A key is never sent over plain `http://` to a non-local host, and is redacted from error messages.
- `check_connection(cfg.llm)` (used by the GUI's *Test Connection*) tells you whether the server answers and the key is accepted. `python examples/05_llm_multimodal_assistant.py` is a ready-made assistant (dry run by default).

### Voice, VLM and intent fusion

```python
from open_ai_hci import MultimodalAgent

agent = MultimodalAgent(
    voice=True,                                   # microphone + faster-whisper (pip install open-ai-hci[voice])
    vlm=lambda frame_bgr, prompt: my_model(frame_bgr, prompt),     # optional: any callable returning text
    vlm_intent_parser=lambda text: ...,           # optional: turn VLM answers into Intents
)
hci.add_processor(agent)
hci.on("intent", lambda e: print(e.intent, e.args, e.context))    # context: cursor x/y, pinching, gesture, hand
agent.ask("What is on screen?")                   # runs on a background thread; result → "vlm_result" event
agent.submit_text("scroll down")                  # or feed any speech-to-text / chat box
```

**Voice with `faster-whisper`:** `VoiceIntentProcessor` listens on the microphone (`pip install "open-ai-hci[voice]"`) and routes each phrase by `voice.mode`: `rules` (deterministic keyword rules only), `llm` (always ask the configured LLM) or `rules_then_llm` (the default: rules first, the LLM for anything they do not understand). Enable it with `cfg.voice.enabled = True`.

Built-in phrases map to intents that the mouse and keyboard actions execute out of the box: *click, double click, right click, scroll up/down, copy, paste, undo, select all, new tab, close tab, go back, press enter, type …, stop listening*. Unknown speech is **never** guessed at; it only produces a HUD note. Swap `intent_parser=` for your own (including an LLM call), add `fusers=[...]` to veto or edit intents using live gesture state, and `require_hand_for=[...]` to gate intents on a visible hand.

---

## Desktop GUI

```bash
open-ai-hci gui                         # or: python examples/06_desktop_gui.py
open-ai-hci gui --config my.yaml --autostart
```

A tabbed control panel built on the standard library's `tkinter` (no extra dependencies):

| Tab | What it does |
|---|---|
| **Engine Controls** | Start / Stop / Pause, and a live `Running` / `Paused` / `Stopped` status with FPS |
| **Core Vision & Gestures** | Hand Cursor, Pinch Left-Click, Middle-Pinch Right-Click, Open-Palm Pause Toggle, Diagnostic HUD; sliders for Cursor Smoothing (0.05 to 0.80) and Pinch Threshold (20 to 80 px) that apply **live** |
| **Custom ONNX AI Model** | Enable a model, pick the `.onnx` file, confidence slider (0.50 to 0.95), **Record Gesture** and **Train Model** buttons (they run `open-ai-hci record` / `train`; a trained model is selected automatically) |
| **LLM Integration** | Enable the router, provider dropdown (Ollama, LM Studio, Custom OpenAI-Compatible URL, OpenAI, Anthropic, Google Gemini) that auto-fills the Base URL and model, a masked API key field, **Test Connection** with a green / red badge, and a box to try a command |

**Save Profile** writes everything to `config.yaml`. The API key is only written if you tick *Remember the key* (it is then plain text); otherwise set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`. Camera, HUD, ONNX and LLM changes take effect after Stop, then Start; sliders and gesture check boxes apply instantly. The camera can only be used by one program, so Record Gesture asks you to stop the engine first. On macOS the HUD window cannot run next to the GUI (OpenCV needs the main thread), so that check box is disabled there.

---

## WebSocket bridge

```python
hci.add_action(WebSocketBridge(host="127.0.0.1", port=8765, stream_landmarks=True))
```

Messages are JSON text frames. **There is no authentication**, so the default bind is localhost only; think before binding `0.0.0.0`.

```jsonc
{"type":"hello","version":"0.2.0","events":["pinch_start", ...],"landmarks":true}
{"type":"event","name":"pinch_start","timestamp":1730000000.12,"frame":42,"source":"gesture_rules",
 "data":{"x":960.0,"y":540.0,"distance":38.2,"handedness":"Right"}}
{"type":"frame","frame":42,"width":1280,"height":720,"fps":29.8,"pinch":"OPEN",
 "cursor":{"x":960.0,"y":540.0,"active":true},
 "hands":[{"handedness":"Right","score":0.98,"landmarks":[[0.51,0.62,-0.01], ...21 points...]}],
 "faces":[{"nose":[640,360],"ear":null,"head_pose":null}]}
```

Client → server: `{"type":"subscribe","events":["pinch_start","frame"]}` narrows that client (`["*"]` = all) and `{"type":"ping"}` answers `{"type":"pong"}`. Landmarks are normalised 0–1; `frame` messages are throttled by `max_stream_hz`. Run `python examples/03_websocket_stream.py` and `... --client` to see it live.

---

## Examples and tools

| File | What it shows |
|---|---|
| `examples/01_mouse_hud_demo.py` | The prototype, rebuilt on the SDK, every threshold spelled out |
| `examples/02_custom_callbacks.py` | Decorators, one-shot handlers, custom events, hotkey bindings |
| `examples/03_websocket_stream.py` | Stream events and landmarks to Unity / web / Godot; includes a test client |
| `examples/04_ai_custom_gestures.py` | Use your own ONNX gestures (peace sign = right click, hotkey per label) |
| `examples/05_llm_multimodal_assistant.py` | Type or speak commands; Ollama / LM Studio / OpenAI / Anthropic / Gemini turn them into actions |
| `examples/06_desktop_gui.py` | The tkinter control panel |
| `tools/record_gestures.py`, `tools/train_gesture_model.py` | Thin wrappers around `open-ai-hci record` / `train` |

## Project layout

```
open_ai_hci/
  engine.py        orchestration loop, FPS manager, error isolation
  event_bus.py     typed pub/sub
  config.py        dataclass config, YAML/JSON, calibration profiles
  sensors/         hand_sensor · face_sensor (blink, head pose) · pose_sensor · MediaPipe backends
  processors/      smoothing · gesture_rules · ai/ (onnx_classifier, llm_router, voice_intent, multimodal_agent)
  actions/         os_mouse · os_keyboard · websocket_bridge
  ui/              hud (diagnostic overlay) · gui + gui_model (tkinter control panel)
  features.py      landmark features shared by training and inference
  recording.py     dataset writer/loader (`open-ai-hci record`)
  training.py      sklearn -> ONNX trainer (`open-ai-hci train`)
tools/  examples/  tests/
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The suite needs no camera, display or model downloads: detectors and PyAutoGUI are replaced by scripted fakes, and the MediaPipe Tasks adapters run against the real installed MediaPipe classes. It covers the rule math (including the prototype-equivalence replay), event bus, smoothing filters, config, the engine loop on synthetic video, actions, the WebSocket server with a real client, the AI plugins (including a real ONNX round trip and a training run), the LLM router against mocked HTTP and a real local test server, the voice processor, and the HUD. The tkinter tests drive the real window and skip themselves when there is no display (on Linux CI: `xvfb-run pytest`).

## Troubleshooting

- **`python -m open_ai_hci doctor`** reports what is missing.
- **Camera does not open**: try `camera.source = 1`; close other apps using it; on Windows try `camera.backend = cv2.CAP_DSHOW`.
- **macOS**: grant your terminal *Camera* and *Accessibility* permission (System Settings → Privacy & Security), or the cursor will not move. Drags use drag events automatically.
- **Linux**: PyAutoGUI needs X11 (on Wayland, run the app under XWayland or use `--no-mouse` and the WebSocket / keyboard actions).
- **`cv2.imshow` errors**: you probably have `opencv-python-headless` installed alongside `opencv-python`; keep only `opencv-python` for the HUD.
- **MediaPipe versions**: releases that still ship `mp.solutions` use it (matching the prototype). Newer releases dropped it, and the framework switches to the MediaPipe Tasks API, which downloads up to three small `.task` model files (hand, face, and pose if enabled) once into `~/.cache/open_ai_hci/models` (override with `MediaPipeConfig.model_dir`, or point `hand_model_path` / `face_model_path` / `pose_model_path` at local files for offline use). Landmark topology is identical; tracking can differ slightly between the two.
- **Jittery cursor**: use `HCIConfig.from_profile("precise")`, or lower `smoothing.alpha` (EMA) / `smoothing.one_euro_min_cutoff` (One-Euro).
- **Accidental clicks**: lower `hand.pinch_threshold_px`, or add `hand.pinch_release_margin_px` hysteresis.
- **Right click fires when you pinch left (or the other way round)**: lower `hand.right_pinch_threshold_px`; the closer finger already wins, but a loose hand can bring both close to the thumb.
- **Control pauses by itself**: raise `hand.pause_hold_s`, switch to `hand.pause_gesture = "fist"`, or turn it off with `"none"`.
- **LLM Test Connection fails**: Ollama and LM Studio must be running (`ollama serve`; LM Studio's *Local Server* tab) and the model name must be one they have loaded. Cloud providers need their key; a key typed into the GUI is used for that session only.
- **`open-ai-hci gui` says tkinter is missing**: reinstall Python with *tcl/tk* (python.org installer) or `sudo apt install python3-tk`.

## Contributing

New sensors, models and actions are very welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE).

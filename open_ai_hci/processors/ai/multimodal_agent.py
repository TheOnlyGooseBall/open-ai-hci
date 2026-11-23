"""Optional plugin: fuse voice commands, a vision-language model and live gesture context into intents.

Three independent, pluggable parts, all optional:

* **Voice**: :class:`VoiceListener` (``pip install "open-ai-hci[voice]"``) records the
  microphone, detects speech with a simple energy gate and transcribes it with
  faster-whisper. Or bring your own speech-to-text and call :meth:`MultimodalAgent.submit_text`.
* **Intent parsing**: :func:`parse_intent` maps a transcript to an :class:`Intent`
  with a small built-in phrase table. Swap in your own parser (e.g. an LLM call).
* **VLM / screen context**: give the agent any ``callable(frame_bgr, prompt) -> str``
  (local model, cloud API, ...). :meth:`MultimodalAgent.ask` runs it on a background
  thread against the latest camera frame; the answer arrives as a ``vlm_result`` event.

Everything the agent produces is an ordinary event, so any action can react::

    agent = MultimodalAgent(voice=True)
    hci.add_processor(agent)
    hci.on("intent", lambda e: print(e.intent, e.args, e.context))

Intents named ``click``, ``double_click``, ``right_click``, ``scroll_up``, ``scroll_down``
are executed by :class:`~open_ai_hci.actions.OSMouseAction`; ``hotkey``, ``press`` and ``type``
by :class:`~open_ai_hci.actions.OSKeyboardAction`.
"""
from __future__ import annotations

import logging
import queue
import re
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..._optional import MissingOptionalDependency, install_hint, require
from ...types import FrameContext
from ..base import BaseProcessor

__all__ = ["Intent", "DEFAULT_PHRASES", "parse_intent", "VoiceListener", "MultimodalAgent"]

logger = logging.getLogger("open_ai_hci")


@dataclass
class Intent:
    """A machine-actionable request.

    Attributes:
        name: e.g. ``"click"``, ``"scroll_down"``, ``"hotkey"``, ``"type"`` or anything custom.
        args: Parameters (``{"keys": "ctrl+c"}``, ``{"text": "hello"}``, ``{"amount": 5}``).
        text: The utterance / model output it came from.
        source: ``"voice"``, ``"text"`` or ``"vlm"``.
        confidence: Parser confidence in ``[0, 1]`` (1.0 for exact phrase matches).
        context: Gesture/cursor context captured when it was fused (see :meth:`MultimodalAgent`).
    """

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    source: str = "voice"
    confidence: float = 1.0
    context: dict[str, Any] = field(default_factory=dict)


# (regex over the lower-cased, punctuation-stripped utterance, intent name, args)
DEFAULT_PHRASES: list[tuple[str, str, dict[str, Any]]] = [
    (r"^(double click|double tap)( here| this)?$", "double_click", {}),
    (r"^(right click|context menu)( here| this)?$", "right_click", {}),
    (r"^(left click|click|tap|select)( here| this)?$", "click", {}),
    (r"^scroll (down|page down)$", "scroll_down", {"amount": 5}),
    (r"^scroll up$", "scroll_up", {"amount": 5}),
    (r"^(go back|back)$", "hotkey", {"keys": "alt+left"}),
    (r"^(go forward|forward)$", "hotkey", {"keys": "alt+right"}),
    (r"^copy( that| this)?$", "hotkey", {"keys": "ctrl+c"}),
    (r"^paste( here| that)?$", "hotkey", {"keys": "ctrl+v"}),
    (r"^cut( that| this)?$", "hotkey", {"keys": "ctrl+x"}),
    (r"^undo$", "hotkey", {"keys": "ctrl+z"}),
    (r"^redo$", "hotkey", {"keys": "ctrl+y"}),
    (r"^select all$", "hotkey", {"keys": "ctrl+a"}),
    (r"^save( file)?$", "hotkey", {"keys": "ctrl+s"}),
    (r"^new tab$", "hotkey", {"keys": "ctrl+t"}),
    (r"^close tab$", "hotkey", {"keys": "ctrl+w"}),
    (r"^(press )?enter$", "press", {"key": "enter"}),
    (r"^(press )?escape$", "press", {"key": "esc"}),
    (r"^(press )?(tab)$", "press", {"key": "tab"}),
    (r"^(press )?(space|spacebar)$", "press", {"key": "space"}),
    (r"^type (?P<text>.+)$", "type", {}),
    (r"^(stop listening|pause listening|mute)$", "pause_listening", {}),
    (r"^(start listening|resume listening|unmute)$", "resume_listening", {}),
]
_COMPILED = [(re.compile(p), n, a) for p, n, a in DEFAULT_PHRASES]
_PUNCT = re.compile(r"[^\w\s']")


def parse_intent(text: str, phrases: Sequence[tuple[str, str, dict[str, Any]]] | None = None) -> Intent | None:
    """Map an utterance to an :class:`Intent` using exact phrase patterns, else ``None``.

    ``phrases`` replaces the built-in table; each entry is ``(regex, intent_name, args)``.
    Named regex groups are merged into ``args`` (``type (?P<text>.+)`` -> ``{"text": ...}``).
    Intentionally conservative: unknown speech is *not* guessed at, so stray
    conversation never moves your mouse.
    """
    cleaned = " ".join(_PUNCT.sub(" ", text.lower()).split())
    if not cleaned:
        return None
    table = _COMPILED if phrases is None else [(re.compile(p), n, a) for p, n, a in phrases]
    for rx, name, args in table:
        m = rx.match(cleaned)
        if m:
            merged = dict(args)
            merged.update({k: v for k, v in m.groupdict().items() if v is not None})
            return Intent(name=name, args=merged, text=text.strip())
    return None


class VoiceListener:
    """Microphone -> text, in a background thread (energy VAD + faster-whisper).

    Requires ``pip install "open-ai-hci[voice]"`` (faster-whisper + sounddevice).
    The first start downloads the Whisper model. Use ``transcriber=`` to plug in a
    different engine: any ``callable(np.ndarray[float32, 16 kHz mono]) -> str``.

    Args:
        on_text: Called from the listener thread with each transcribed utterance.
        model_size: faster-whisper model name (``"tiny.en"`` is fastest).
        language: Language code, or ``None`` to auto-detect.
        energy_threshold: RMS level (0..1) that counts as speech.
        silence_s: Trailing silence that ends an utterance.
        max_utterance_s: Hard cap on one utterance.
        device: sounddevice input device (index or name), default = system default.
        transcriber: Custom speech-to-text callable (skips faster-whisper).
    """

    SAMPLE_RATE = 16_000

    def __init__(
        self,
        on_text: Callable[[str], None],
        *,
        model_size: str = "tiny.en",
        language: str | None = "en",
        energy_threshold: float = 0.015,
        silence_s: float = 0.8,
        max_utterance_s: float = 10.0,
        min_utterance_s: float = 0.3,
        device: int | str | None = None,
        transcriber: Callable[[np.ndarray], str] | None = None,
    ) -> None:
        self.on_text = on_text
        self.model_size, self.language = model_size, language
        self.energy_threshold, self.silence_s = energy_threshold, silence_s
        self.max_utterance_s, self.min_utterance_s = max_utterance_s, min_utterance_s
        self.device = device
        self._transcriber = transcriber
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.paused = False

    def start(self) -> None:
        """Load the model and start listening. Raises :class:`MissingOptionalDependency` if unavailable."""
        sd = require("sounddevice", "VoiceListener")
        if self._transcriber is None:
            fw = require("faster_whisper", "VoiceListener")
            model = fw.WhisperModel(self.model_size, device="cpu", compute_type="int8")
            lang = self.language

            def transcribe(audio: np.ndarray) -> str:
                segments, _ = model.transcribe(audio, language=lang, beam_size=1, vad_filter=False)
                return " ".join(seg.text.strip() for seg in segments).strip()

            self._transcriber = transcribe
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(sd,), name="open-ai-hci-voice", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # ------------------------------------------------------------------ internals
    def _run(self, sd: Any) -> None:
        sr = self.SAMPLE_RATE
        block = int(sr * 0.05)  # 50 ms
        blocks: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            blocks.put(indata[:, 0].copy())

        try:
            stream = sd.InputStream(
                samplerate=sr, channels=1, dtype="float32", blocksize=block, device=self.device, callback=callback
            )
        except Exception as exc:
            logger.error("Could not open the microphone: %s", exc)
            return

        speech: list[np.ndarray] = []
        silent_blocks = 0
        silence_limit = int(self.silence_s / 0.05)
        max_blocks = int(self.max_utterance_s / 0.05)
        with stream:
            while not self._stop.is_set():
                try:
                    chunk = blocks.get(timeout=0.2)
                except queue.Empty:
                    continue
                if self.paused:
                    speech.clear()
                    silent_blocks = 0
                    continue
                loud = float(np.sqrt(np.mean(chunk**2))) >= self.energy_threshold
                if loud:
                    speech.append(chunk)
                    silent_blocks = 0
                elif speech:
                    speech.append(chunk)
                    silent_blocks += 1
                if speech and (silent_blocks >= silence_limit or len(speech) >= max_blocks):
                    audio = np.concatenate(speech)
                    speech, silent_blocks = [], 0
                    if len(audio) / sr >= self.min_utterance_s:
                        self._transcribe_and_emit(audio)

    def _transcribe_and_emit(self, audio: np.ndarray) -> None:
        try:
            assert self._transcriber is not None
            text = self._transcriber(audio.astype(np.float32))
        except Exception:
            logger.exception("Speech transcription failed")
            return
        if text:
            self.on_text(text)


FuserFn = Callable[[Intent, FrameContext], Intent | None]
VLMFn = Callable[[np.ndarray, str], str]


class MultimodalAgent(BaseProcessor):
    """Voice + VLM + gesture intent fusion.

    Args:
        voice: ``False`` (default) = no microphone; ``True`` = start a :class:`VoiceListener`
            with defaults; or pass a configured :class:`VoiceListener`. If the voice
            dependencies are missing the agent logs the install hint and carries on
            without voice (``strict=True`` raises instead).
        intent_parser: ``callable(text) -> Optional[Intent]``. Default :func:`parse_intent`.
        vlm: ``callable(frame_bgr, prompt) -> str`` for screen / scene understanding.
        vlm_intent_parser: ``callable(vlm_text) -> Optional[Intent]``; when given, every VLM
            answer is also turned into an intent (source ``"vlm"``).
        fusers: Extra ``callable(intent, ctx) -> Optional[Intent]`` hooks that may edit or
            veto (return ``None``) an intent using the live frame context.
        require_hand_for: Intent names that are only honoured while a hand is visible.
            Default: none (voice works without a hand).
        strict: Raise instead of degrading when optional dependencies are missing.

    Events emitted: ``voice_command`` (``text``, ``origin``), ``intent`` (``intent``, ``args``,
    ``text``, ``origin``, ``confidence``, ``context``), ``vlm_result`` (``prompt``, ``text``),
    ``alert`` (when speech was heard but not understood, or on VLM errors).
    The intent ``context`` holds ``x``/``y`` (cursor), ``pinching``, ``gesture`` and ``handedness``.
    """

    name = "multimodal_agent"

    def __init__(
        self,
        voice: bool | VoiceListener = False,
        *,
        intent_parser: Callable[[str], Intent | None] | None = None,
        vlm: VLMFn | None = None,
        vlm_intent_parser: Callable[[str], Intent | None] | None = None,
        fusers: Sequence[FuserFn] | None = None,
        require_hand_for: Sequence[str] = (),
        strict: bool = False,
    ) -> None:
        self._voice_arg = voice
        self._voice: VoiceListener | None = None
        self.intent_parser = intent_parser or parse_intent
        self.vlm = vlm
        self.vlm_intent_parser = vlm_intent_parser
        self.fusers = list(fusers or [])
        self.require_hand_for = set(require_hand_for)
        self.strict = strict
        self._inbox: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._pending_vlm: str | None = None
        self._lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None
        self._listening = True

    # ------------------------------------------------------------------ lifecycle
    def setup(self) -> None:
        if self._voice_arg:
            listener = self._voice_arg if isinstance(self._voice_arg, VoiceListener) else VoiceListener(self.submit_text)
            listener.on_text = self.submit_text
            try:
                listener.start()
                self._voice = listener
            except MissingOptionalDependency as exc:
                if self.strict:
                    raise
                logger.warning("Voice commands disabled: %s", exc)
                logger.warning("To enable them: %s", install_hint("faster_whisper"))
        if self.vlm is not None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="open-ai-hci-vlm")

    def close(self) -> None:
        if self._voice is not None:
            self._voice.stop()
            self._voice = None
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    # --------------------------------------------------------------- public inputs
    def submit_text(self, text: str, source: str = "voice") -> None:
        """Feed an utterance (from any speech-to-text, a chat box, a test...). Thread-safe."""
        if text and text.strip():
            self._inbox.put(("text", (text.strip(), source)))

    def ask(self, prompt: str = "Describe what is on screen and what the user is trying to do.") -> None:
        """Run the VLM on the next processed frame (background thread). Thread-safe."""
        if self.vlm is None:
            raise RuntimeError("MultimodalAgent was created without a vlm= callable")
        with self._lock:
            self._pending_vlm = prompt

    # --------------------------------------------------------------------- frames
    def process(self, ctx: FrameContext) -> None:
        with self._lock:
            prompt, self._pending_vlm = self._pending_vlm, None
        if prompt is not None and self._pool is not None and self.vlm is not None:
            frame = ctx.frame.copy()
            self._pool.submit(self._run_vlm, self.vlm, frame, prompt)

        while True:
            try:
                kind, payload = self._inbox.get_nowait()
            except queue.Empty:
                break
            if kind == "text":
                self._handle_text(payload[0], payload[1], ctx)
            elif kind == "vlm":
                self._handle_vlm(payload[0], payload[1], ctx)

    def _run_vlm(self, fn: VLMFn, frame: np.ndarray, prompt: str) -> None:
        try:
            answer = str(fn(frame, prompt))
        except Exception as exc:
            logger.exception("VLM call failed")
            self._inbox.put(("vlm", (prompt, f"[error] {exc}")))
            return
        self._inbox.put(("vlm", (prompt, answer)))

    # -------------------------------------------------------------------- handling
    def _handle_text(self, text: str, source: str, ctx: FrameContext) -> None:
        self.emit("voice_command", text=text, origin=source)
        intent = self.intent_parser(text)
        if intent is None:
            self.emit("alert", text=f"heard: {text!r} (no command)")
            return
        intent.source = source if source != "voice" else intent.source
        if intent.name == "pause_listening":
            self._set_listening(False)
            return
        if intent.name == "resume_listening":
            self._set_listening(True)
            return
        if not self._listening:
            return
        self._dispatch(intent, ctx)

    def _handle_vlm(self, prompt: str, answer: str, ctx: FrameContext) -> None:
        self.emit("vlm_result", prompt=prompt, text=answer)
        if self.vlm_intent_parser is not None and not answer.startswith("[error]"):
            intent = self.vlm_intent_parser(answer)
            if intent is not None:
                intent.source = "vlm"
                self._dispatch(intent, ctx)

    def _set_listening(self, on: bool) -> None:
        self._listening = on
        if self._voice is not None:
            self._voice.paused = not on
        self.emit("alert", text="voice commands on" if on else "voice commands paused")

    def _dispatch(self, intent: Intent | None, ctx: FrameContext) -> None:
        if intent is None:
            return
        if intent.name in self.require_hand_for and ctx.hand is None:
            self.emit("alert", text=f"'{intent.name}' needs a visible hand")
            return
        c = ctx.cursor
        intent.context = {
            "x": c.x if c is not None else None,
            "y": c.y if c is not None else None,
            "pinching": ctx.pinch_status == "HOLDING CLICK",
            "gesture": ctx.gesture,
            "handedness": ctx.hand.handedness if ctx.hand is not None else None,
        }
        for fuser in self.fusers:
            intent = fuser(intent, ctx)
            if intent is None:
                return
        self.emit("intent", intent=intent.name, args=intent.args, text=intent.text,
                  origin=intent.source, confidence=intent.confidence, context=intent.context)

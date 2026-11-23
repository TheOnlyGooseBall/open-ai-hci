"""Optional plugin: speech -> action, through deterministic keyword rules, the LLM router, or both.

::

    hci.add_processor(LLMRouter())                              # optional, see llm_router.py
    hci.add_processor(VoiceIntentProcessor(mode="rules_then_llm"))

Modes (``VoiceConfig.mode``):

* ``"rules"``: only the built-in phrase table (:data:`~open_ai_hci.processors.ai.DEFAULT_PHRASES`,
  or your own ``phrases=``): "click", "scroll down", "new tab", "type hello" ... Instant, offline,
  fully predictable; unknown speech is ignored.
* ``"llm"``: every utterance goes to the :class:`~open_ai_hci.processors.ai.LLMRouter`, which can
  also understand free-form requests ("open this in a new tab", "type my email signature").
* ``"rules_then_llm"`` (default): the phrase table first (no network, no latency for the common
  commands), the LLM only for what the rules do not understand.

The microphone part needs ``pip install "open-ai-hci[voice]"`` (faster-whisper + sounddevice) and
degrades gracefully with an install hint when missing; the LLM part needs nothing but a reachable
model. You can also skip the microphone completely and call :meth:`submit_text` from a chat box or
any other speech-to-text. ``voice_command`` events are emitted for every utterance heard.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from ...config import VoiceConfig
from ...types import FrameContext
from .llm_router import LLMRouter
from .multimodal_agent import Intent, MultimodalAgent, VoiceListener

__all__ = ["VoiceIntentProcessor"]

logger = logging.getLogger("open_ai_hci")

MODES = ("rules", "llm", "rules_then_llm")


class VoiceIntentProcessor(MultimodalAgent):
    """Transcribed speech -> intents, using keyword rules, the configured LLM, or both.

    Args:
        mode: ``"rules"`` | ``"llm"`` | ``"rules_then_llm"`` (default: ``VoiceConfig.mode``).
        router: The :class:`LLMRouter` to use; default: the one registered on the engine.
        config: Overrides ``engine.config.voice``.
        listener: A configured :class:`VoiceListener`; default: built from ``VoiceConfig``.
            Pass ``voice=False`` to run without a microphone (text input via :meth:`submit_text` only).
        phrases: Custom phrase table (see :func:`parse_intent`) replacing the built-in one.
        voice: ``False`` disables the microphone.
        intent_parser, fusers, require_hand_for, strict: as in :class:`MultimodalAgent`.
    """

    name = "voice_intent"

    def __init__(
        self,
        mode: str | None = None,
        *,
        router: LLMRouter | None = None,
        config: VoiceConfig | None = None,
        listener: VoiceListener | None = None,
        voice: bool = True,
        phrases: Sequence[tuple[str, str, dict[str, Any]]] | None = None,
        intent_parser: Callable[[str], Intent | None] | None = None,
        fusers: Sequence[Callable[[Intent, FrameContext], Intent | None]] | None = None,
        require_hand_for: Sequence[str] = (),
        strict: bool = False,
    ) -> None:
        if mode is not None and mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        if phrases is not None and intent_parser is None:
            from .multimodal_agent import parse_intent

            def intent_parser(text: str, _phrases: Any = phrases) -> Intent | None:
                return parse_intent(text, _phrases)

        super().__init__(
            voice=listener if listener is not None else voice,
            intent_parser=intent_parser,
            fusers=fusers,
            require_hand_for=require_hand_for,
            strict=strict,
        )
        self._mode_arg = mode
        self._vcfg = config
        self._router_arg = router
        self._use_default_listener = listener is None and voice

    @property
    def cfg(self) -> VoiceConfig:
        if self._vcfg is not None:
            return self._vcfg
        return self.engine.config.voice if self.engine is not None else VoiceConfig()

    @property
    def mode(self) -> str:
        return self._mode_arg or self.cfg.mode

    @property
    def router(self) -> LLMRouter | None:
        if self._router_arg is not None:
            return self._router_arg
        return self.engine.get(LLMRouter) if self.engine is not None else None

    def setup(self) -> None:
        if self._use_default_listener:
            c = self.cfg
            self._voice_arg = VoiceListener(
                self.submit_text,
                model_size=c.model_size,
                language=c.language,
                energy_threshold=c.energy_threshold,
                device=c.device,
            )
        if self.mode != "rules" and self.router is None:
            logger.warning("VoiceIntentProcessor mode %r needs an LLMRouter, but none is registered; using keyword rules only.", self.mode)
        super().setup()

    # ------------------------------------------------------------------ handling
    def _handle_text(self, text: str, source: str, ctx: FrameContext) -> None:
        self.emit("voice_command", text=text, origin=source)
        mode, router = self.mode, self.router
        intent = self.intent_parser(text) if mode in ("rules", "rules_then_llm") or router is None else None

        if intent is not None:
            intent.source = source if source != "voice" else intent.source
            if intent.name == "pause_listening":
                self._set_listening(False)
            elif intent.name == "resume_listening":
                self._set_listening(True)
            elif self._listening:
                self._dispatch(intent, ctx)
            return
        if not self._listening:
            return
        if mode != "rules" and router is not None:
            if router.submit(text, source=source):
                self.emit("alert", text=f"thinking: {text[:40]}")
            return  # LLMRouter reports failures as llm_error events
        self.emit("alert", text=f"heard: {text!r} (no command)")

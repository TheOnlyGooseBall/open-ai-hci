"""Example 5: talk (or type) to your computer. Hand gesture + cursor position + a command -> an LLM -> mouse / keyboard actions.

Local models need no key and no extra packages:

    ollama pull llama3.2 && python examples/05_llm_multimodal_assistant.py                  # Ollama on :11434
    python examples/05_llm_multimodal_assistant.py --provider lmstudio --model qwen2.5-7b    # LM Studio on :1234

Cloud models read the key from the environment (it is never printed or saved):

    export OPENAI_API_KEY=...     && python examples/05_llm_multimodal_assistant.py --provider openai --model gpt-4o-mini
    export ANTHROPIC_API_KEY=...  && python examples/05_llm_multimodal_assistant.py --provider anthropic
    export GEMINI_API_KEY=...     && python examples/05_llm_multimodal_assistant.py --provider gemini

Then type commands in this terminal while the hand-tracking window is running, for example:

    open this in a new tab        scroll down slowly        type my email signature        go back

Options: --voice (microphone, needs pip install "open-ai-hci[voice]"; rules first, the LLM for everything else),
--signature "Best, Sam" (what "type my email signature" types), --live (really execute; the default is a DRY RUN that only
prints what the model decided), --no-hand (skip the camera: a pure text assistant).

The LLM runs on a background thread, so hand tracking never stalls while the model thinks. Dangerous hotkeys
(Ctrl+Alt+Del, Alt+F4, Win+R, ...) are refused no matter what the model says.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Any

import numpy as np

from open_ai_hci import HCIConfig, HCIEngine, LLMConfig, LLMRouter, MouseConfig, OSKeyboardAction, OSMouseAction, VoiceConfig


def build_engine(args: argparse.Namespace, *, http: Any = None, env: Any = None) -> HCIEngine:
    """Assemble the engine from parsed arguments (``http`` / ``env`` are injection points for tests)."""
    config = HCIConfig(mouse=MouseConfig(enabled=args.live))
    config.llm = LLMConfig(
        enabled=True,
        provider=args.provider,
        base_url=args.base_url or "",
        model=args.model or "",
        dry_run=not args.live,
        user_context={"email_signature": args.signature} if args.signature else {},
    )
    config.voice = VoiceConfig(enabled=args.voice, mode="rules_then_llm")
    # llm.enabled / voice.enabled make the default pipeline add the LLMRouter / VoiceIntentProcessor for us
    hci = HCIEngine(config, sensors=[] if args.no_hand else None, actions=[])
    if http is not None or env is not None:  # tests: swap in a router with a fake network / environment
        hci.remove(LLMRouter)
        hci.add_processor(LLMRouter(http=http, env=env))
    if args.live:
        hci.add_action(OSMouseAction())
        hci.add_action(OSKeyboardAction())

    hci.on("llm_response", lambda e: print(f"  LLM ({e.latency_ms:.0f} ms): {e.actions or 'no action'}" + (f"  {e.reply}" if e.reply else "")))
    hci.on("llm_error", lambda e: print(f"  LLM error: {e.message}"))
    hci.on("control_paused", lambda e: print("  paused"))
    hci.on("control_resumed", lambda e: print("  resumed"))
    return hci


def type_loop(hci: HCIEngine) -> None:
    """Read commands from the terminal and hand them to the router (runs on its own thread)."""
    router = hci.get(LLMRouter)
    for line in sys.stdin:
        text = line.strip()
        if text:
            router.submit(text, source="text")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", default="ollama", choices=["ollama", "lmstudio", "openai_compatible", "openai", "anthropic", "gemini"])
    p.add_argument("--model", help="model name (defaults per provider, e.g. llama3.2 / gpt-4o-mini / claude-haiku-5-5)")
    p.add_argument("--base-url", help="override the endpoint, e.g. http://192.168.1.20:11434/v1")
    p.add_argument("--signature", default="", help='text typed for "type my email signature"')
    p.add_argument("--voice", action="store_true", help="listen to the microphone")
    p.add_argument("--live", action="store_true", help="really execute actions (default: dry run)")
    p.add_argument("--no-hand", action="store_true", help="no camera / hand tracking, text assistant only")
    args = p.parse_args(argv)

    hci = build_engine(args)
    router = hci.get(LLMRouter)
    hci.setup()
    if not router.available:
        print(f"LLM unavailable: {router.unavailable_reason}")
        hci.close()
        return 1
    print(f"Provider: {router.provider!r}")
    print(("LIVE: actions will be executed." if args.live else "DRY RUN: actions are only printed. Use --live to execute.") + " Type a command and press Enter; Ctrl+C quits.")
    threading.Thread(target=type_loop, args=(hci,), daemon=True).start()
    if args.no_hand:
        try:
            while True:
                hci.process_frame(np.zeros((72, 128, 3), np.uint8))
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            hci.close()
    else:
        hci.run(show_hud=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

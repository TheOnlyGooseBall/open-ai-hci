"""Optional plugin: an LLM *intent & action router* for multimodal control.

It turns "what the user said or typed" plus "what the hand is doing right now" into structured
HCI actions and publishes them as ordinary ``intent`` events, which the OS mouse / keyboard
actions already execute::

    "open this in a new tab"  ->  {"actions": [{"action": "hotkey", "keys": ["ctrl", "t"]}]}
    "scroll down slowly"      ->  {"actions": [{"action": "scroll", "amount": -150}]}
    "type my email signature" ->  {"actions": [{"action": "type", "text": "Best regards,\\nSam"}]}

Zero extra dependencies: every provider talks plain HTTPS/HTTP through :mod:`urllib` (or an
``http=`` callable you inject, which is also how the tests mock the network).

Providers (``LLMConfig.provider``):

====================  ===================================  ============================
``ollama``            ``http://localhost:11434/v1``        no key (native ``/api/chat`` if the
                                                           base URL has no ``/v1`` suffix)
``lmstudio``          ``http://localhost:1234/v1``         no key
``openai_compatible`` any ``.../v1/chat/completions``      optional key (vLLM, llama.cpp, ...)
``openai``            ``https://api.openai.com/v1``        ``OPENAI_API_KEY``
``anthropic``         ``https://api.anthropic.com``        ``ANTHROPIC_API_KEY``
``gemini``            ``generativelanguage.googleapis``    ``GEMINI_API_KEY``
====================  ===================================  ============================

A cloud provider without an API key (or a local server that is not running) never raises inside
the camera loop: the router disables itself with a log message and an ``llm_error`` event.

Real-time safety:
    * Network calls run on one background worker thread; ``process()`` only snapshots the live
      context and drains finished answers, so tracking never waits for a model.
    * Answers are emitted from ``process()`` (the engine thread), so mouse / keyboard side effects
      stay single-threaded.

Safety:
    * Only actions in ``LLMConfig.allowed_actions`` run; unknown ones are dropped.
    * Dangerous hotkeys (``LLMConfig.blocked_hotkeys``: Ctrl+Alt+Del, Win+R, Alt+F4, ...) are refused.
    * ``type`` text is capped at 500 characters, scrolls at +-3000 units, and at most
      ``LLMConfig.max_actions`` actions run per answer.
    * API keys are sent only in headers, never logged, redacted from error messages, and refused
      over plain ``http://`` to non-local hosts. Redirects are not followed.
    * ``LLMConfig.dry_run`` parses and reports without executing anything.

Events emitted: ``llm_response`` (``request``, ``actions``, ``rejected``, ``reply``, ``latency_ms``,
``provider``, ``model``), ``intent`` (``origin="llm"``), ``llm_error`` (``request``, ``message``),
``alert``.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ...actions.os_keyboard import parse_hotkey
from ...config import LLM_ACTIONS, LLM_PROVIDER_DEFAULTS, LLMConfig
from ...types import FrameContext
from ..base import BaseProcessor

__all__ = [
    "LLMError",
    "LLMUnavailable",
    "HTTPFn",
    "urllib_http",
    "LLMProvider",
    "OpenAICompatibleProvider",
    "OllamaProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "make_provider",
    "resolve_api_key",
    "LLMAction",
    "build_system_prompt",
    "build_user_prompt",
    "extract_json",
    "parse_actions",
    "ConnectionResult",
    "check_connection",
    "LLMResult",
    "LLMRouter",
]

logger = logging.getLogger("open_ai_hci")

MAX_RESPONSE_BYTES = 1_000_000
MAX_TYPE_CHARS = 500
MAX_SCROLL = 3000.0

#: ``http(method, url, headers, body, timeout) -> (status_code, response_text)``
HTTPFn = Callable[[str, str, Mapping[str, str], "bytes | None", float], "tuple[int, str]"]


class LLMError(RuntimeError):
    """A provider call failed (network, HTTP status, malformed reply). Never contains an API key."""


class LLMUnavailable(LLMError):
    """The provider cannot be used at all right now (no key, bad URL, ...)."""


# ------------------------------------------------------------------------------- HTTP
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None  # surface the 3xx as an HTTPError instead of forwarding headers (API keys) elsewhere


def urllib_http(method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float) -> tuple[int, str]:
    """Default HTTP client: stdlib only, no redirects, response capped at 1 MB.

    Returns ``(status, text)``; HTTP error statuses are returned, not raised. Network problems
    raise :class:`LLMError`.
    """
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in ("http", "https"):
        raise LLMUnavailable(f"Unsupported URL scheme {scheme!r}; use http:// or https://")
    req = urllib.request.Request(url, data=body, headers=dict(headers), method=method)  # noqa: S310 - scheme checked above
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(req, timeout=timeout) as resp:
            return resp.status, resp.read(MAX_RESPONSE_BYTES).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        try:
            text = exc.read(MAX_RESPONSE_BYTES).decode("utf-8", "replace")
        except Exception:  # pragma: no cover - body unreadable
            text = ""
        return exc.code, text
    except urllib.error.URLError as exc:
        raise LLMError(f"Cannot reach {_origin(url)}: {exc.reason}") from None
    except TimeoutError:
        raise LLMError(f"Timed out after {timeout:g}s talking to {_origin(url)}") from None
    except OSError as exc:
        raise LLMError(f"Network error talking to {_origin(url)}: {exc}") from None


def _origin(url: str) -> str:
    p = urllib.parse.urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def _is_local_host(host: str) -> bool:
    if host in ("localhost", ""):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host.endswith(".local") or host.endswith(".localhost")
    return ip.is_loopback or ip.is_private


# --------------------------------------------------------------------------- providers
def resolve_api_key(cfg: LLMConfig, env: Mapping[str, str] | None = None) -> str:
    """The key to use: ``cfg.api_key``, else the environment variable ``cfg.api_key_env`` (or the provider default)."""
    if cfg.api_key.strip():
        return cfg.api_key.strip()
    env = os.environ if env is None else env
    name = cfg.api_key_env.strip() or LLM_PROVIDER_DEFAULTS[cfg.provider]["key_env"]
    return env.get(name, "").strip() if name else ""


class LLMProvider:
    """One LLM backend. Subclasses implement :meth:`chat` and :meth:`ping`.

    Args:
        base_url: Endpoint root (see the table in the module docs).
        model: Model name sent with every request.
        api_key: Optional secret; sent only in request headers.
        http: Injectable HTTP function (see :data:`HTTPFn`); default :func:`urllib_http`.
        timeout_s / temperature / max_tokens: Generation settings.
    """

    name = "llm"
    needs_key = False

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        *,
        http: HTTPFn | None = None,
        timeout_s: float = 20.0,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._http: HTTPFn = http or urllib_http
        self.timeout_s, self.temperature, self.max_tokens = timeout_s, temperature, max_tokens

    # ---- to implement
    def chat(self, system: str, user: str, *, json_mode: bool = True) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def ping(self) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    # ---- helpers
    def __repr__(self) -> str:
        return f"{type(self).__name__}(base_url={self.base_url!r}, model={self.model!r}, key={'set' if self._api_key else 'none'})"

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "open-ai-hci"}

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "***") if self._api_key else text

    def _request(self, method: str, url: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        status, text = self._http(method, url, self._headers(), body, self.timeout_s)
        try:
            data: Any = json.loads(text) if text.strip() else {}
        except ValueError:
            data = {"_raw": text[:300]}
        return status, data

    def _check(self, status: int, data: Any) -> Any:
        if 200 <= status < 300:
            return data
        detail = _error_text(data)
        if status in (401, 403):
            raise LLMError(f"{self.name}: authentication failed (HTTP {status}); check the API key. {self._redact(detail)}".strip())
        if status == 404:
            raise LLMError(f"{self.name}: endpoint or model not found (HTTP 404). Check base URL and model name. {self._redact(detail)}".strip())
        if status == 429:
            raise LLMError(f"{self.name}: rate limited (HTTP 429). {self._redact(detail)}".strip())
        raise LLMError(f"{self.name}: HTTP {status}. {self._redact(detail)}".strip())


def _error_text(data: Any) -> str:
    """Pull a short human message out of the usual ``{"error": {"message": ...}}`` shapes."""
    if isinstance(data, dict):
        err = data.get("error", data.get("message", data.get("_raw")))
        if isinstance(err, dict):
            err = err.get("message") or err.get("status") or json.dumps(err)[:200]
        if err:
            return str(err)[:240]
    return ""


class OpenAICompatibleProvider(LLMProvider):
    """``POST {base}/chat/completions``: OpenAI, LM Studio, Ollama's ``/v1``, vLLM, llama.cpp server, ..."""

    name = "openai-compatible"

    def _headers(self) -> dict[str, str]:
        h = super()._headers()
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def chat(self, system: str, user: str, *, json_mode: bool = True) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        url = f"{self.base_url}/chat/completions"
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        status, data = self._request("POST", url, payload)
        if status == 400 and json_mode:  # some local servers reject response_format: retry plain
            payload.pop("response_format", None)
            status, data = self._request("POST", url, payload)
        data = self._check(status, data)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"{self.name}: unexpected response shape (no choices[0].message.content)") from None
        return content if isinstance(content, str) else json.dumps(content)

    def ping(self) -> str:
        status, data = self._request("GET", f"{self.base_url}/models")
        data = self._check(status, data)
        models = [m.get("id", "?") for m in data.get("data", []) if isinstance(m, dict)] if isinstance(data, dict) else []
        found = f"{len(models)} model(s) available" if models else "server reachable"
        if models and self.model and self.model not in models:
            return f"{found}; note: model {self.model!r} is not listed ({', '.join(models[:4])}...)"
        return found


class OllamaProvider(OpenAICompatibleProvider):
    """Ollama. A base URL ending in ``/v1`` uses the OpenAI-compatible endpoint; otherwise the native ``/api/chat``."""

    name = "ollama"

    @property
    def _native(self) -> bool:
        return not self.base_url.endswith("/v1")

    @property
    def _root(self) -> str:
        return self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url

    def chat(self, system: str, user: str, *, json_mode: bool = True) -> str:
        if not self._native:
            return super().chat(system, user, json_mode=json_mode)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        status, data = self._request("POST", f"{self.base_url}/api/chat", payload)
        data = self._check(status, data)
        try:
            return str(data["message"]["content"])
        except (KeyError, TypeError):
            raise LLMError("ollama: unexpected response shape (no message.content)") from None

    def ping(self) -> str:
        status, data = self._request("GET", f"{self._root}/api/tags")
        data = self._check(status, data)
        models = [m.get("name", "?") for m in data.get("models", []) if isinstance(m, dict)] if isinstance(data, dict) else []
        if not models:
            return "Ollama reachable, but no models are pulled yet (try: ollama pull llama3.2)"
        wanted = self.model.split(":")[0]
        if not any(m == self.model or m.split(":")[0] == wanted for m in models):
            return f"Ollama reachable; note: model {self.model!r} is not pulled ({', '.join(models[:4])}...)"
        return f"Ollama reachable, {len(models)} model(s)"


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API: ``POST {base}/v1/messages``."""

    name = "anthropic"
    needs_key = True
    API_VERSION = "2023-06-01"

    def _headers(self) -> dict[str, str]:
        h = super()._headers()
        h["x-api-key"] = self._api_key
        h["anthropic-version"] = self.API_VERSION
        return h

    def chat(self, system: str, user: str, *, json_mode: bool = True) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        status, data = self._request("POST", f"{self.base_url}/v1/messages", payload)
        data = self._check(status, data)
        try:
            text = "".join(b.get("text", "") for b in data["content"] if isinstance(b, dict) and b.get("type") == "text")
        except (KeyError, TypeError):
            raise LLMError("anthropic: unexpected response shape (no content blocks)") from None
        return text

    def ping(self) -> str:
        status, data = self._request("GET", f"{self.base_url}/v1/models")
        self._check(status, data)
        return "Anthropic API reachable, key accepted"


class GeminiProvider(LLMProvider):
    """Google Gemini: ``POST {base}/models/{model}:generateContent`` (key in the ``x-goog-api-key`` header)."""

    name = "gemini"
    needs_key = True

    def _headers(self) -> dict[str, str]:
        h = super()._headers()
        h["x-goog-api-key"] = self._api_key
        return h

    def chat(self, system: str, user: str, *, json_mode: bool = True) -> str:
        gen: dict[str, Any] = {"temperature": self.temperature, "maxOutputTokens": self.max_tokens}
        if json_mode:
            gen["responseMimeType"] = "application/json"
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen,
        }
        model = self.model if self.model.startswith("models/") else f"models/{self.model}"
        status, data = self._request("POST", f"{self.base_url}/{model}:generateContent", payload)
        data = self._check(status, data)
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        except (KeyError, IndexError, TypeError):
            reason = ""
            if isinstance(data, dict):
                reason = str(data.get("promptFeedback", {}).get("blockReason", ""))
            raise LLMError(f"gemini: no candidates in the response{' (blocked: ' + reason + ')' if reason else ''}") from None

    def ping(self) -> str:
        status, data = self._request("GET", f"{self.base_url}/models?pageSize=1")
        self._check(status, data)
        return "Gemini API reachable, key accepted"


_PROVIDER_CLASSES: dict[str, type[LLMProvider]] = {
    "ollama": OllamaProvider,
    "lmstudio": OpenAICompatibleProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "openai": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}
_CLOUD = {"openai", "anthropic", "gemini"}


def make_provider(cfg: LLMConfig, *, http: HTTPFn | None = None, env: Mapping[str, str] | None = None) -> LLMProvider:
    """Build the provider described by ``cfg``.

    Raises:
        LLMUnavailable: unknown provider, malformed URL, a cloud provider without an API key, or an
            API key that would travel over plain ``http://`` to a non-local host.
    """
    if cfg.provider not in _PROVIDER_CLASSES:
        raise LLMUnavailable(f"Unknown LLM provider {cfg.provider!r}")
    defaults = LLM_PROVIDER_DEFAULTS[cfg.provider]
    base = (cfg.base_url.strip() or defaults["base_url"]).rstrip("/")
    model = cfg.model.strip() or defaults["model"]
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise LLMUnavailable(f"Invalid base URL {base!r}; expected something like http://localhost:11434/v1")
    key = resolve_api_key(cfg, env)
    cls = _PROVIDER_CLASSES[cfg.provider]
    if (cfg.provider in _CLOUD) and not key:
        var = cfg.api_key_env.strip() or defaults["key_env"]
        raise LLMUnavailable(f"{cfg.provider} needs an API key: set the {var} environment variable or LLMConfig.api_key")
    if key and parsed.scheme == "http" and not _is_local_host(parsed.hostname or ""):
        raise LLMUnavailable("Refusing to send an API key over plain http:// to a non-local host; use https://")
    return cls(base, model, key, http=http, timeout_s=cfg.timeout_s, temperature=cfg.temperature, max_tokens=cfg.max_tokens)


# ----------------------------------------------------------------------------- prompts
_ALIASES = {"control": "ctrl", "command": "cmd", "windows": "win", "super": "win", "escape": "esc", "return": "enter", "option": "alt"}


def _norm_keys(keys: Any) -> tuple[str, ...]:
    return tuple(_ALIASES.get(k, k) for k in parse_hotkey(keys))


def build_system_prompt(cfg: LLMConfig, screen: tuple[int, int] | None = None) -> str:
    """The instructions sent as the system prompt (JSON-only action protocol + the user's saved snippets)."""
    mod = "command" if sys.platform == "darwin" else "ctrl"
    allowed = [a for a in LLM_ACTIONS if a in cfg.allowed_actions] + ["none"]
    forms = {
        "click": '{"action": "click"}',
        "double_click": '{"action": "double_click"}',
        "right_click": '{"action": "right_click"}',
        "move": '{"action": "move", "x": 960, "y": 540}   (screen pixels; or "dx"/"dy" for relative)',
        "scroll": '{"action": "scroll", "amount": -300}   (positive = up, negative = down, about 100 per mouse-wheel notch)',
        "hotkey": f'{{"action": "hotkey", "keys": ["{mod}", "t"]}}',
        "press": '{"action": "press", "key": "enter"}',
        "type": '{"action": "type", "text": "text to type"}',
        "say": '{"action": "say", "text": "short reply for the user"}',
        "none": '{"action": "none"}   (when the request is not something you can do with the mouse and keyboard)',
    }
    lines = [
        "You translate a user's spoken or typed command into mouse and keyboard actions for a hand-tracking desktop interface.",
        "Reply with ONLY a JSON object, no prose and no code fences: {\"actions\": [ ... ]} with at most "
        f"{cfg.max_actions} actions, executed in order.",
        "Allowed actions:",
        *[f"  - {forms[a]}" for a in allowed],
        f"The platform's main modifier key is '{mod}'. Use lower-case key names (ctrl, shift, alt, win, command, enter, esc, tab, left, right, ...).",
        "The context JSON gives the pointer position, the screen size and the current hand gesture; use it to resolve words like 'this' or 'here'.",
        "The pointer is already where the user's hand is; prefer click / hotkey / scroll over move.",
        "Examples:",
    ]
    if "hotkey" in allowed:
        lines += [
            '  "open this in a new tab" -> {"actions": [{"action": "hotkey", "keys": ["' + mod + '", "t"]}]}',
            '  "go back" -> {"actions": [{"action": "hotkey", "keys": ["alt", "left"]}]}',
        ]
    if "scroll" in allowed:
        lines.append('  "scroll down slowly" -> {"actions": [{"action": "scroll", "amount": -150}]}')
    if "click" in allowed:
        lines.append('  "click that" -> {"actions": [{"action": "click"}]}')
    if "type" in allowed and cfg.user_context:
        lines.append("The user's saved snippets (use them verbatim for requests like 'type my email signature'):")
        lines += [f"  {k} = {json.dumps(v)}" for k, v in cfg.user_context.items()]
    if screen:
        lines.append(f"Screen size: {screen[0]}x{screen[1]} pixels.")
    if cfg.system_prompt.strip():
        lines.append(cfg.system_prompt.strip())
    return "\n".join(lines)


def build_user_prompt(text: str, context: Mapping[str, Any] | None = None) -> str:
    """The user message: the command plus the live multimodal context as JSON."""
    ctx = json.dumps(dict(context or {}), default=str)
    return f"Command: {json.dumps(text)}\nContext: {ctx}"


# ----------------------------------------------------------------------------- parsing
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def extract_json(text: str) -> Any:
    """Find the first JSON object / array in model output (tolerates code fences and surrounding prose).

    Raises:
        ValueError: no JSON value found.
    """
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    decoder = json.JSONDecoder()
    for chunk in candidates:
        for i, ch in enumerate(chunk):
            if ch in "{[":
                try:
                    value, _ = decoder.raw_decode(chunk[i:])
                except ValueError:
                    continue
                return value
    raise ValueError("no JSON found in the model output")


@dataclass
class LLMAction:
    """One validated action: ``intent`` is the event name (``hotkey``, ``scroll``, ...), ``args`` its parameters."""

    intent: str
    args: dict[str, Any] = field(default_factory=dict)


def _num(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def parse_actions(text: str, cfg: LLMConfig | None = None) -> tuple[list[LLMAction], list[str], str]:
    """Validate model output.

    Returns:
        ``(actions, rejected, reply)``: runnable actions (already limited to ``max_actions``), human readable
        reasons for anything dropped, and the text of ``say`` actions (the assistant's spoken reply).

    Raises:
        ValueError: the output contained no usable JSON.
    """
    cfg = cfg or LLMConfig()
    data = extract_json(text)
    if isinstance(data, dict) and isinstance(data.get("actions"), list):
        raw = data["actions"]
    elif isinstance(data, dict):
        raw = [data]
    elif isinstance(data, list):
        raw = data
    else:
        raise ValueError("expected a JSON object or list")

    allowed = set(cfg.allowed_actions)
    blocked = set()
    for spec in cfg.blocked_hotkeys:
        try:
            blocked.add(frozenset(_norm_keys(spec)))
        except ValueError:
            continue
    actions: list[LLMAction] = []
    rejected: list[str] = []
    replies: list[str] = []

    for item in raw:
        if not isinstance(item, dict):
            rejected.append(f"not an object: {str(item)[:40]}")
            continue
        kind = str(item.get("action", item.get("name", ""))).strip().lower()
        if kind in ("", "none", "noop"):
            continue
        if kind not in LLM_ACTIONS or kind not in allowed:
            rejected.append(f"action not allowed: {kind or '?'}")
            continue
        if kind in ("click", "double_click", "right_click"):
            actions.append(LLMAction(kind))
        elif kind == "say":
            reply = str(item.get("text", "")).strip()[:300]
            if reply:
                replies.append(reply)
        elif kind == "scroll":
            amount = _num(item.get("amount"))
            if amount is None:
                rejected.append("scroll without a numeric amount")
                continue
            actions.append(LLMAction("scroll", {"amount": max(-MAX_SCROLL, min(MAX_SCROLL, amount))}))
        elif kind == "move":
            x, y, dx, dy = (_num(item.get(k)) for k in ("x", "y", "dx", "dy"))
            if x is not None and y is not None:
                actions.append(LLMAction("move", {"x": x, "y": y}))
            elif dx is not None or dy is not None:
                actions.append(LLMAction("move", {"dx": dx or 0.0, "dy": dy or 0.0}))
            else:
                rejected.append("move without coordinates")
        elif kind == "hotkey":
            try:
                keys = _norm_keys(item.get("keys", item.get("key", "")))
            except (ValueError, AttributeError, TypeError):
                rejected.append("hotkey without keys")
                continue
            if frozenset(keys) in blocked:
                rejected.append(f"blocked hotkey: {'+'.join(keys)}")
                continue
            actions.append(LLMAction("hotkey", {"keys": list(keys)}))
        elif kind == "press":
            key = str(item.get("key", "")).strip().lower()
            key = _ALIASES.get(key, key)
            if not key or "+" in key:
                rejected.append("press needs a single key")
                continue
            actions.append(LLMAction("press", {"key": key}))
        elif kind == "type":
            typed = item.get("text")
            if not isinstance(typed, str) or not typed:
                rejected.append("type without text")
                continue
            if len(typed) > MAX_TYPE_CHARS:
                rejected.append(f"type text longer than {MAX_TYPE_CHARS} characters")
                continue
            actions.append(LLMAction("type", {"text": typed}))
    if len(actions) > cfg.max_actions:
        rejected.append(f"{len(actions) - cfg.max_actions} action(s) beyond max_actions={cfg.max_actions}")
        actions = actions[: cfg.max_actions]
    return actions, rejected, " ".join(replies)


# -------------------------------------------------------------------- connection test
@dataclass
class ConnectionResult:
    """Outcome of :func:`check_connection` (what the GUI's green / red badge shows)."""

    ok: bool
    message: str
    provider: str = ""
    model: str = ""


def check_connection(cfg: LLMConfig, *, http: HTTPFn | None = None, env: Mapping[str, str] | None = None) -> ConnectionResult:
    """Check that the configured server answers (and, for cloud providers, accepts the key). Never raises."""
    try:
        provider = make_provider(cfg, http=http, env=env)
    except LLMError as exc:
        return ConnectionResult(False, str(exc), cfg.provider, cfg.model)
    try:
        return ConnectionResult(True, provider.ping(), cfg.provider, provider.model)
    except LLMError as exc:
        return ConnectionResult(False, str(exc), cfg.provider, provider.model)
    except Exception as exc:  # a broken injected client must not crash the GUI
        return ConnectionResult(False, f"{type(exc).__name__}: {exc}", cfg.provider, provider.model)


# ----------------------------------------------------------------------------- router
@dataclass
class _Job:
    text: str
    source: str
    context: dict[str, Any]


@dataclass
class LLMResult:
    """A validated model answer (see :meth:`LLMRouter.route`)."""

    job: _Job
    actions: list[LLMAction] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    reply: str = ""
    error: str = ""
    latency_ms: float = 0.0


class LLMRouter(BaseProcessor):
    """Background LLM client that turns commands + live context into ``intent`` events.

    Args:
        config: Overrides ``engine.config.llm``.
        provider: A ready :class:`LLMProvider` (skips :func:`make_provider`; handy for tests / custom backends).
        http: Injectable HTTP function for the built-in providers.
        env: Environment mapping used to find API keys (default ``os.environ``).
        strict: Raise from ``setup()`` when the provider is unusable instead of degrading.

    Use :meth:`submit` from anywhere (a voice processor, a text box, your own code); the answer arrives
    as events on the next processed frame. :meth:`route` is the blocking variant for scripts.
    """

    name = "llm_router"

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        provider: LLMProvider | None = None,
        http: HTTPFn | None = None,
        env: Mapping[str, str] | None = None,
        strict: bool = False,
    ) -> None:
        self._cfg = config
        self._provider_arg = provider
        self._http = http
        self._env = env
        self.strict = strict
        self.provider: LLMProvider | None = None
        self.available = False
        self.unavailable_reason = ""
        self._context: dict[str, Any] = {}
        self._screen: tuple[int, int] | None = None
        self._lock = threading.Lock()
        self._jobs: queue.Queue[_Job | None] = queue.Queue()
        self._results: queue.Queue[LLMResult] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._inflight = 0
        self._idle = threading.Event()
        self._idle.set()

    @property
    def cfg(self) -> LLMConfig:
        """Effective settings: the constructor's, else the engine's, else defaults (standalone use)."""
        if self._cfg is not None:
            return self._cfg
        return self.engine.config.llm if self.engine is not None else LLMConfig()

    # ------------------------------------------------------------------ lifecycle
    def setup(self) -> None:
        self._build_provider()
        if not self.available and self.strict:
            raise LLMUnavailable(self.unavailable_reason)
        if self.engine is not None:
            try:
                self._screen = self.engine.screen_size
            except Exception:  # pragma: no cover - screen size is only a hint for the prompt
                self._screen = None
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._work, name="open-ai-hci-llm", daemon=True)
            self._worker.start()

    def close(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            self._jobs.put(None)
            worker.join(timeout=1.0)

    def reconfigure(self) -> None:
        """Rebuild the provider after the configuration changed (called by ``engine.apply_config``)."""
        self._build_provider()

    def _build_provider(self) -> None:
        try:
            self.provider = self._provider_arg or make_provider(self.cfg, http=self._http, env=self._env)
            self.available, self.unavailable_reason = True, ""
            logger.info("LLM router ready: %r", self.provider)
        except LLMError as exc:
            self.provider, self.available, self.unavailable_reason = None, False, str(exc)
            logger.warning("LLM router disabled: %s", exc)

    # ------------------------------------------------------------------ inputs
    def process(self, ctx: FrameContext) -> None:
        c = ctx.cursor
        self._context = {
            "cursor": {"x": round(c.x), "y": round(c.y)} if c is not None else None,
            "hand_visible": ctx.hand is not None,
            "gesture": ctx.gesture or ctx.extras.get("gesture"),
            "pinching": ctx.pinch_status == "HOLDING CLICK",
            "paused": ctx.paused,
        }
        self.drain()

    def submit(self, text: str, source: str = "text", context: Mapping[str, Any] | None = None) -> bool:
        """Queue a command for the model. Returns ``False`` if it was dropped (router unavailable or too busy). Thread-safe."""
        text = (text or "").strip()
        if not text:
            return False
        if not self.available:
            self._results.put(LLMResult(_Job(text, source, {}), error=self.unavailable_reason or "LLM router is not available"))
            return False
        with self._lock:
            if self._inflight >= self.cfg.max_pending:
                self._results.put(LLMResult(_Job(text, source, {}), error="LLM is busy; command dropped"))
                return False
            self._inflight += 1
            self._idle.clear()
        snapshot = dict(self._context)
        if context:
            snapshot.update(context)
        if self._screen:
            snapshot.setdefault("screen", {"width": self._screen[0], "height": self._screen[1]})
        self._jobs.put(_Job(text, source, snapshot))
        return True

    def route(self, text: str, context: Mapping[str, Any] | None = None) -> LLMResult:
        """Blocking call (no thread): ask the model and return the validated result. Raises :class:`LLMError` on failure."""
        if self.provider is None:
            self._build_provider()
        if self.provider is None:
            raise LLMUnavailable(self.unavailable_reason)
        result = self._ask(_Job(text, "text", dict(context or {})))
        if result.error:
            raise LLMError(result.error)
        return result

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Block until every submitted command has an answer waiting (for tests and scripts)."""
        return self._idle.wait(timeout)

    # ------------------------------------------------------------------ worker
    def _work(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                self._results.put(self._ask(job))
            except Exception as exc:  # pragma: no cover - _ask already converts errors
                self._results.put(LLMResult(job, error=f"{type(exc).__name__}: {exc}"))
            finally:
                with self._lock:
                    self._inflight = max(0, self._inflight - 1)
                    if self._inflight == 0:
                        self._idle.set()

    def _ask(self, job: _Job) -> LLMResult:
        cfg, provider = self.cfg, self.provider
        t0 = time.perf_counter()
        if provider is None:
            return LLMResult(job, error=self.unavailable_reason or "LLM router is not available")
        try:
            screen = self._screen
            reply = provider.chat(build_system_prompt(cfg, screen), build_user_prompt(job.text, job.context), json_mode=cfg.json_mode)
            actions, rejected, say = parse_actions(reply, cfg)
        except LLMError as exc:
            return LLMResult(job, error=str(exc), latency_ms=(time.perf_counter() - t0) * 1000)
        except ValueError as exc:
            return LLMResult(job, error=f"Could not understand the model's answer: {exc}", latency_ms=(time.perf_counter() - t0) * 1000)
        return LLMResult(job, actions, rejected, say, latency_ms=(time.perf_counter() - t0) * 1000)

    # ------------------------------------------------------------------ outputs
    def drain(self) -> int:
        """Emit the events for every finished answer. Called every frame; returns how many were handled."""
        handled = 0
        while True:
            try:
                res = self._results.get_nowait()
            except queue.Empty:
                return handled
            handled += 1
            self._emit_result(res)

    def _emit_result(self, res: LLMResult) -> None:
        cfg, job = self.cfg, res.job
        provider = self.provider
        if res.error:
            logger.warning("LLM request failed: %s", res.error)
            self.emit("llm_error", request=job.text, message=res.error)
            self.emit("alert", text=f"LLM: {res.error[:60]}")
            return
        self.emit(
            "llm_response",
            request=job.text,
            actions=[{"action": a.intent, **a.args} for a in res.actions],
            rejected=res.rejected,
            reply=res.reply,
            latency_ms=round(res.latency_ms, 1),
            provider=provider.name if provider else cfg.provider,
            model=provider.model if provider else cfg.model,
            dry_run=cfg.dry_run,
        )
        for reason in res.rejected:
            logger.info("LLM action rejected: %s", reason)
        if res.reply:
            self.emit("alert", text=res.reply[:80])
        elif res.rejected and not res.actions:
            self.emit("alert", text=f"LLM: {res.rejected[0][:60]}")
        for action in res.actions:
            self.emit(
                "intent",
                intent=action.intent,
                args=dict(action.args),
                text=job.text,
                origin="llm",
                confidence=1.0,
                context=dict(job.context),
                dry_run=cfg.dry_run,
            )

"""LLM router: providers (HTTP is mocked), prompt / JSON handling, safety rules and the background worker."""
import json
import threading

import pytest
from conftest import FakePyAutoGUI, make_ctx, names

from open_ai_hci import HCIConfig, HCIEngine, LLMConfig, MouseConfig, OSKeyboardAction, OSMouseAction
from open_ai_hci.processors.ai.llm_router import (
    AnthropicProvider,
    GeminiProvider,
    LLMError,
    LLMRouter,
    LLMUnavailable,
    OllamaProvider,
    OpenAICompatibleProvider,
    build_system_prompt,
    build_user_prompt,
    check_connection,
    extract_json,
    make_provider,
    parse_actions,
    resolve_api_key,
    urllib_http,
)
from open_ai_hci.types import CursorState


class FakeHTTP:
    """Scripted ``http(method, url, headers, body, timeout)``. Each response is ``(status, dict | str)`` or an Exception."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, method, url, headers, body, timeout):
        self.requests.append(
            {"method": method, "url": url, "headers": dict(headers), "json": json.loads(body) if body else None, "timeout": timeout}
        )
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        status, data = item
        return status, data if isinstance(data, str) else json.dumps(data)


def openai_reply(text):
    return 200, {"choices": [{"message": {"role": "assistant", "content": text}}]}


def actions_json(*actions):
    return json.dumps({"actions": list(actions)})


# ------------------------------------------------------------------------ providers
def test_openai_compatible_request_shape_and_response():
    http = FakeHTTP(openai_reply("hello"))
    cfg = LLMConfig(provider="openai", api_key="sk-secret", model="gpt-4o-mini", max_tokens=99, temperature=0.3, timeout_s=7)
    out = make_provider(cfg, http=http).chat("SYS", "USER")
    req = http.requests[0]
    assert out == "hello" and req["method"] == "POST" and req["timeout"] == 7
    assert req["url"] == "https://api.openai.com/v1/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer sk-secret"
    assert req["json"]["model"] == "gpt-4o-mini" and req["json"]["max_tokens"] == 99 and req["json"]["temperature"] == 0.3
    assert req["json"]["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    assert req["json"]["response_format"] == {"type": "json_object"}


def test_json_mode_is_retried_without_response_format_on_http_400():
    http = FakeHTTP((400, {"error": {"message": "response_format unsupported"}}), openai_reply("ok"))
    out = make_provider(LLMConfig(provider="lmstudio"), http=http).chat("s", "u")
    assert out == "ok" and len(http.requests) == 2
    assert "response_format" in http.requests[0]["json"] and "response_format" not in http.requests[1]["json"]


def test_lmstudio_defaults_and_no_auth_header():
    http = FakeHTTP(openai_reply("{}"))
    make_provider(LLMConfig(provider="lmstudio"), http=http).chat("s", "u")
    assert http.requests[0]["url"] == "http://localhost:1234/v1/chat/completions"
    assert "Authorization" not in http.requests[0]["headers"]


def test_custom_openai_compatible_url_and_optional_key():
    http = FakeHTTP(openai_reply("{}"))
    cfg = LLMConfig(provider="openai_compatible", base_url="http://192.168.1.20:8000/v1/", model="qwen2.5", api_key="k")
    make_provider(cfg, http=http, env={}).chat("s", "u")
    assert http.requests[0]["url"] == "http://192.168.1.20:8000/v1/chat/completions"  # private LAN host: key allowed over http
    assert http.requests[0]["headers"]["Authorization"] == "Bearer k"


def test_ollama_v1_uses_the_openai_endpoint_and_native_uses_api_chat():
    http = FakeHTTP(openai_reply("v1 answer"))
    assert make_provider(LLMConfig(provider="ollama"), http=http).chat("s", "u") == "v1 answer"
    assert http.requests[0]["url"] == "http://localhost:11434/v1/chat/completions"
    assert http.requests[0]["json"]["model"] == "llama3.2"

    native = FakeHTTP((200, {"message": {"role": "assistant", "content": "native answer"}}))
    prov = make_provider(LLMConfig(provider="ollama", base_url="http://localhost:11434", model="qwen2.5", temperature=0.1), http=native)
    assert isinstance(prov, OllamaProvider) and prov.chat("s", "u") == "native answer"
    req = native.requests[0]
    assert req["url"] == "http://localhost:11434/api/chat" and req["json"]["format"] == "json" and req["json"]["stream"] is False
    assert req["json"]["options"] == {"temperature": 0.1, "num_predict": 256}


def test_ollama_ping_lists_models_and_warns_when_the_model_is_missing():
    tags = {"models": [{"name": "llama3.2:latest"}, {"name": "qwen2.5:7b"}]}
    http = FakeHTTP((200, tags))
    msg = make_provider(LLMConfig(provider="ollama", model="llama3.2"), http=http).ping()
    assert http.requests[0]["url"] == "http://localhost:11434/api/tags" and "2 model(s)" in msg
    assert "not pulled" in make_provider(LLMConfig(provider="ollama", model="mistral"), http=FakeHTTP((200, tags))).ping()
    assert "no models" in make_provider(LLMConfig(provider="ollama"), http=FakeHTTP((200, {"models": []}))).ping()


def test_openai_ping_notes_unlisted_model():
    http = FakeHTTP((200, {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]}))
    prov = make_provider(LLMConfig(provider="openai", api_key="k", model="gpt-5"), http=http)
    assert http.requests == [] and "not listed" in prov.ping() and http.requests[0]["url"].endswith("/v1/models")


def test_anthropic_request_shape():
    http = FakeHTTP((200, {"content": [{"type": "text", "text": '{"actions": '}, {"type": "text", "text": "[]}"}]}))
    prov = make_provider(LLMConfig(provider="anthropic", api_key="sk-ant-xyz", max_tokens=64), http=http)
    assert isinstance(prov, AnthropicProvider) and prov.chat("SYS", "USER") == '{"actions": []}'
    req = http.requests[0]
    assert req["url"] == "https://api.anthropic.com/v1/messages"
    assert req["headers"]["x-api-key"] == "sk-ant-xyz" and req["headers"]["anthropic-version"] == "2023-06-01"
    assert req["json"] == {
        "model": "claude-haiku-5-5", "max_tokens": 64, "temperature": 0.0, "system": "SYS",
        "messages": [{"role": "user", "content": "USER"}],
    }


def test_gemini_request_shape_keeps_the_key_out_of_the_url():
    http = FakeHTTP((200, {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}))
    prov = make_provider(LLMConfig(provider="gemini", api_key="AIza-key"), http=http)
    assert isinstance(prov, GeminiProvider) and prov.chat("SYS", "USER") == "{}"
    req = http.requests[0]
    assert req["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    assert "AIza-key" not in req["url"] and req["headers"]["x-goog-api-key"] == "AIza-key"
    assert req["json"]["systemInstruction"] == {"parts": [{"text": "SYS"}]}
    assert req["json"]["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_blocked_prompt_gives_a_clear_error():
    http = FakeHTTP((200, {"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(LLMError, match="SAFETY"):
        make_provider(LLMConfig(provider="gemini", api_key="k"), http=http).chat("s", "u")


# ------------------------------------------------------------------------ keys / safety
def test_cloud_providers_are_unavailable_without_a_key_and_name_the_variable():
    for provider, var in (("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY"), ("gemini", "GEMINI_API_KEY")):
        with pytest.raises(LLMUnavailable, match=var):
            make_provider(LLMConfig(provider=provider), env={})


def test_keys_come_from_the_environment_or_a_custom_variable():
    assert resolve_api_key(LLMConfig(provider="openai"), {"OPENAI_API_KEY": " sk-env "}) == "sk-env"
    assert resolve_api_key(LLMConfig(provider="openai", api_key_env="MY_KEY"), {"MY_KEY": "k2", "OPENAI_API_KEY": "no"}) == "k2"
    assert resolve_api_key(LLMConfig(provider="openai", api_key="direct"), {"OPENAI_API_KEY": "env"}) == "direct"
    assert resolve_api_key(LLMConfig(provider="ollama"), {"OPENAI_API_KEY": "x"}) == ""  # local: no key needed
    http = FakeHTTP(openai_reply("{}"))
    make_provider(LLMConfig(provider="openai"), http=http, env={"OPENAI_API_KEY": "sk-env"}).chat("s", "u")
    assert http.requests[0]["headers"]["Authorization"] == "Bearer sk-env"


def test_api_key_never_appears_in_repr_or_errors():
    secret = "sk-very-secret-1234"
    prov = make_provider(LLMConfig(provider="openai", api_key=secret), http=FakeHTTP((401, {"error": {"message": f"Incorrect API key: {secret}"}})))
    assert secret not in repr(prov)
    with pytest.raises(LLMError) as exc:
        prov.chat("s", "u")
    assert "authentication failed" in str(exc.value) and secret not in str(exc.value) and "***" in str(exc.value)


def test_api_key_is_refused_over_plain_http_to_a_public_host_but_not_localhost():
    with pytest.raises(LLMUnavailable, match="plain http"):
        make_provider(LLMConfig(provider="openai_compatible", base_url="http://api.example.com/v1", api_key="k"))
    make_provider(LLMConfig(provider="openai_compatible", base_url="http://localhost:8000/v1", api_key="k"))
    make_provider(LLMConfig(provider="openai_compatible", base_url="http://api.example.com/v1"))  # no key: nothing to leak


@pytest.mark.parametrize("url", ["ftp://x/v1", "localhost:11434", "file:///etc/passwd", "http://"])
def test_bad_base_urls_are_rejected(url):
    with pytest.raises(LLMUnavailable, match="Invalid base URL"):
        make_provider(LLMConfig(provider="ollama", base_url=url))


def test_unknown_provider_is_rejected():
    with pytest.raises(LLMUnavailable, match="Unknown LLM provider"):
        make_provider(LLMConfig(provider="skynet"))


@pytest.mark.parametrize(
    "status,needle", [(401, "authentication"), (403, "authentication"), (404, "not found"), (429, "rate limited"), (500, "HTTP 500")]
)
def test_http_errors_are_translated(status, needle):
    prov = make_provider(LLMConfig(provider="lmstudio"), http=FakeHTTP((status, {"error": "boom"})))
    with pytest.raises(LLMError, match=needle):
        prov.chat("s", "u")


def test_malformed_provider_replies_raise_llm_error():
    for provider, body in (("lmstudio", {"nope": 1}), ("anthropic", {"content": 5}), ("gemini", {"candidates": []})):
        prov = make_provider(LLMConfig(provider=provider, api_key="k"), http=FakeHTTP((200, body)))
        with pytest.raises(LLMError, match="unexpected|no candidates"):
            prov.chat("s", "u")
    with pytest.raises(LLMError):  # non-JSON body
        make_provider(LLMConfig(provider="lmstudio"), http=FakeHTTP((200, "<html>proxy login</html>"))).chat("s", "u")


# ------------------------------------------------------------------------ connection test
def test_check_connection_reports_ok_and_failures_without_raising():
    ok = check_connection(LLMConfig(provider="ollama"), http=FakeHTTP((200, {"models": [{"name": "llama3.2:latest"}]})))
    assert ok.ok and "Ollama reachable" in ok.message and ok.model == "llama3.2"
    bad_key = check_connection(LLMConfig(provider="openai", api_key="k"), http=FakeHTTP((401, {"error": "no"})))
    assert not bad_key.ok and "authentication" in bad_key.message
    down = check_connection(LLMConfig(provider="lmstudio"), http=FakeHTTP(LLMError("Cannot reach http://localhost:1234: refused")))
    assert not down.ok and "Cannot reach" in down.message
    nokey = check_connection(LLMConfig(provider="gemini"), env={})
    assert not nokey.ok and "GEMINI_API_KEY" in nokey.message
    crash = check_connection(LLMConfig(provider="lmstudio"), http=FakeHTTP(ValueError("broken client")))
    assert not crash.ok and "ValueError" in crash.message


# ------------------------------------------------------------------------ real urllib client
@pytest.fixture
def local_server():
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def _send(self, code, payload, extra=None):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            seen["body"] = json.loads(self.rfile.read(n))
            seen["auth"] = self.headers.get("Authorization")
            if self.path == "/v1/chat/completions":
                self._send(200, {"choices": [{"message": {"content": actions_json({"action": "click"})}}]})
            elif self.path == "/redir/chat/completions":
                self._send(302, {}, {"Location": "http://127.0.0.1:1/steal"})
            else:
                self._send(404, {"error": "nope"})

        def do_GET(self):
            self._send(200, {"data": [{"id": "local-model"}]})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", seen
    server.shutdown()
    server.server_close()


def test_default_urllib_client_talks_to_a_real_local_server(local_server):
    base, seen = local_server
    cfg = LLMConfig(provider="lmstudio", base_url=f"{base}/v1", model="local-model", api_key="local-key")
    router = LLMRouter(cfg)
    result = router.route("click please")
    assert [a.intent for a in result.actions] == ["click"]
    assert seen["auth"] == "Bearer local-key" and seen["body"]["model"] == "local-model"
    assert check_connection(cfg).ok


def test_urllib_client_does_not_follow_redirects_and_rejects_odd_schemes(local_server):
    base, _ = local_server
    status, _text = urllib_http("POST", f"{base}/redir/chat/completions", {}, b"{}", 3)
    assert status == 302
    with pytest.raises(LLMUnavailable, match="scheme"):
        urllib_http("GET", "file:///etc/passwd", {}, None, 3)


def test_urllib_client_reports_unreachable_servers_clearly():
    with pytest.raises(LLMError, match="Cannot reach http://127.0.0.1:9"):
        urllib_http("GET", "http://127.0.0.1:9/v1/models", {}, None, 2)


# ------------------------------------------------------------------------ prompts + parsing
def test_system_prompt_lists_only_allowed_actions_and_user_snippets():
    cfg = LLMConfig(allowed_actions=["click", "hotkey"], user_context={"email_signature": "Best,\nSam"}, system_prompt="Be terse.")
    prompt = build_system_prompt(cfg, (1920, 1080))
    assert '"action": "click"' in prompt and '"action": "hotkey"' in prompt
    assert '"action": "type"' not in prompt and '"action": "scroll"' not in prompt
    assert "email_signature" not in prompt  # snippets are only offered when typing is allowed
    assert "1920x1080" in prompt and prompt.rstrip().endswith("Be terse.")
    typing = build_system_prompt(LLMConfig(user_context={"email_signature": "Best,\nSam"}))
    assert 'email_signature = "Best,\\nSam"' in typing and '"action": "type"' in typing


def test_user_prompt_contains_command_and_context():
    text = build_user_prompt('say "hi"', {"cursor": {"x": 1, "y": 2}})
    assert 'Command: "say \\"hi\\""' in text and '"cursor": {"x": 1, "y": 2}' in text


@pytest.mark.parametrize(
    "raw",
    [
        '{"actions": [{"action": "click"}]}',
        '```json\n{"actions": [{"action": "click"}]}\n```',
        'Sure! Here you go: {"actions": [{"action": "click"}]} Hope that helps.',
        '[{"action": "click"}]',
        '{"action": "click"}',
        'noise {broken ```\n[{"action": "click"}]\n```',
    ],
)
def test_json_extraction_variants(raw):
    actions, rejected, _ = parse_actions(raw)
    assert [a.intent for a in actions] == ["click"] and rejected == []


def test_extract_json_failure():
    with pytest.raises(ValueError):
        extract_json("I cannot help with that.")
    with pytest.raises(ValueError):
        parse_actions("42")


def test_parse_every_action_kind():
    raw = actions_json(
        {"action": "click"}, {"action": "double_click"}, {"action": "right_click"},
        {"action": "move", "x": "100", "y": 200}, {"action": "move", "dx": 5},
        {"action": "scroll", "amount": -500}, {"action": "hotkey", "keys": ["Control", "T"]},
        {"action": "hotkey", "keys": "cmd+shift+t"}, {"action": "press", "key": "Return"},
        {"action": "type", "text": "hello"}, {"action": "say", "text": "Done."}, {"action": "none"},
    )
    cfg = LLMConfig(max_actions=20)
    actions, rejected, reply = parse_actions(raw, cfg)
    assert rejected == [] and reply == "Done."
    assert [(a.intent, a.args) for a in actions] == [
        ("click", {}), ("double_click", {}), ("right_click", {}),
        ("move", {"x": 100.0, "y": 200.0}), ("move", {"dx": 5.0, "dy": 0.0}),
        ("scroll", {"amount": -500.0}), ("hotkey", {"keys": ["ctrl", "t"]}),
        ("hotkey", {"keys": ["cmd", "shift", "t"]}), ("press", {"key": "enter"}), ("type", {"text": "hello"}),
    ]


def test_dangerous_hotkeys_are_blocked_in_any_order_or_spelling():
    raw = actions_json(
        {"action": "hotkey", "keys": ["ctrl", "alt", "delete"]}, {"action": "hotkey", "keys": "Delete+Control+Alt"},
        {"action": "hotkey", "keys": ["alt", "f4"]}, {"action": "hotkey", "keys": ["win", "r"]},
        {"action": "hotkey", "keys": ["command", "q"]}, {"action": "hotkey", "keys": ["ctrl", "c"]},
    )
    actions, rejected, _ = parse_actions(raw)
    assert [a.args["keys"] for a in actions] == [["ctrl", "c"]] and len(rejected) == 5
    assert all(r.startswith("blocked hotkey") for r in rejected)
    # the block list is configurable
    permissive = LLMConfig(blocked_hotkeys=[], max_actions=10)
    assert len(parse_actions(raw, permissive)[0]) == 6


def test_allowed_actions_limit_what_runs():
    cfg = LLMConfig(allowed_actions=["click"])
    actions, rejected, _ = parse_actions(actions_json({"action": "type", "text": "x"}, {"action": "click"}, {"action": "format_c"}), cfg)
    assert [a.intent for a in actions] == ["click"] and len(rejected) == 2


def test_malformed_actions_are_rejected_individually():
    raw = actions_json(
        "click", {"action": "scroll"}, {"action": "scroll", "amount": "fast"}, {"action": "move"}, {"action": "hotkey"},
        {"action": "hotkey", "keys": []}, {"action": "press", "key": "ctrl+c"}, {"action": "type", "text": ""},
        {"action": "type", "text": "x" * 501}, {"action": "click"},
    )
    actions, rejected, _ = parse_actions(raw)
    assert [a.intent for a in actions] == ["click"] and len(rejected) == 9


def test_scroll_is_clamped_and_actions_are_capped():
    actions, _, _ = parse_actions(actions_json({"action": "scroll", "amount": 10**9}, {"action": "scroll", "amount": -10**9}))
    assert [a.args["amount"] for a in actions] == [3000.0, -3000.0]
    many = actions_json(*[{"action": "click"}] * 9)
    capped, rejected, _ = parse_actions(many, LLMConfig(max_actions=3))
    assert len(capped) == 3 and "beyond max_actions=3" in rejected[0]
    nan_actions, nan_rejected, _ = parse_actions('{"actions": [{"action": "scroll", "amount": NaN}]}')
    assert nan_actions == [] and nan_rejected  # NaN / Infinity never reach the mouse


# ------------------------------------------------------------------------ router
def router_engine(responses, *, cfg=None, actions=(), **router_kw):
    config = HCIConfig(mouse=MouseConfig(enabled=False))
    config.screen.width, config.screen.height = 1920, 1080
    if cfg is not None:
        config.llm = cfg
    http = responses if callable(responses) else FakeHTTP(*responses)
    router = LLMRouter(http=http, env={}, **router_kw)
    engine = HCIEngine(config, sensors=[], processors=[router], actions=list(actions), raise_errors=True)
    engine.setup()
    return engine, router, http


def pump(engine, router, n=1):
    assert router.wait_idle(5)
    for _ in range(n):
        engine.process_frame(make_ctx().frame)


def test_router_emits_llm_response_and_intents_from_the_engine_thread():
    reply = actions_json({"action": "hotkey", "keys": ["ctrl", "t"]}, {"action": "say", "text": "Opening a new tab"})
    engine, router, http = router_engine([openai_reply(reply)], cfg=LLMConfig(provider="lmstudio"))
    events = []
    engine.on("*", events.append)
    assert router.submit("open this in a new tab", source="voice") is True
    pump(engine, router)
    resp = next(e for e in events if e.name == "llm_response")
    assert resp.request == "open this in a new tab" and resp.actions == [{"action": "hotkey", "keys": ["ctrl", "t"]}]
    assert resp.reply == "Opening a new tab" and resp.provider == "openai-compatible" and resp.latency_ms >= 0
    intent = next(e for e in events if e.name == "intent")
    assert (intent.intent, intent.args, intent.origin, intent.dry_run) == ("hotkey", {"keys": ["ctrl", "t"]}, "llm", False)
    assert any(e.name == "alert" and e.text == "Opening a new tab" for e in events)
    engine.close()


def test_the_live_context_reaches_the_prompt():
    engine, router, http = router_engine([openai_reply(actions_json())], cfg=LLMConfig(provider="lmstudio"))
    ctx = make_ctx(cursor=CursorState(640.4, 360.6, 0, 0))
    ctx.gesture = "pinch_hold"
    router.process(ctx)
    router.submit("what now")
    assert router.wait_idle(5)
    user_msg = http.requests[0]["json"]["messages"][1]["content"]
    assert '"cursor": {"x": 640, "y": 361}' in user_msg and '"gesture": "pinch_hold"' in user_msg
    assert '"screen": {"width": 1920, "height": 1080}' in user_msg
    assert "1920x1080" in http.requests[0]["json"]["messages"][0]["content"]
    engine.close()


def test_slow_models_never_block_the_frame_loop():
    gate, started = threading.Event(), threading.Event()

    def slow_http(method, url, headers, body, timeout):
        started.set()
        gate.wait(5)
        return openai_reply(actions_json({"action": "click"}))[0], json.dumps(openai_reply(actions_json({"action": "click"}))[1])

    engine, router, _ = router_engine(slow_http, cfg=LLMConfig(provider="lmstudio"))
    events = []
    engine.on("intent", events.append)
    router.submit("click")
    assert started.wait(2)
    for _ in range(20):  # 20 frames while the "model" is still thinking: none of them waits for it
        engine.process_frame(make_ctx().frame)
    assert events == []
    gate.set()
    pump(engine, router)
    assert [e.intent for e in events] == ["click"]
    engine.close()


def test_errors_become_events_not_exceptions():
    engine, router, _ = router_engine([(500, {"error": "server exploded"})], cfg=LLMConfig(provider="lmstudio"))
    events = []
    engine.on("*", events.append)
    router.submit("scroll down")
    pump(engine, router)
    err = next(e for e in events if e.name == "llm_error")
    assert err.request == "scroll down" and "HTTP 500" in err.message and "intent" not in names(events)
    engine.close()


def test_garbage_answers_become_llm_error():
    engine, router, _ = router_engine([openai_reply("I'm sorry, I can't do that.")], cfg=LLMConfig(provider="lmstudio"))
    events = []
    engine.on("llm_error", events.append)
    router.submit("do something")
    pump(engine, router)
    assert "Could not understand" in events[0].message
    engine.close()


def test_missing_key_degrades_gracefully(caplog):
    config = HCIConfig(mouse=MouseConfig(enabled=False), llm=LLMConfig(provider="openai"))
    router = LLMRouter(env={})
    engine = HCIEngine(config, sensors=[], processors=[router], actions=[], raise_errors=True)
    with caplog.at_level("WARNING"):
        engine.setup()
    assert not router.available and "OPENAI_API_KEY" in router.unavailable_reason and "OPENAI_API_KEY" in caplog.text
    events = []
    engine.on("llm_error", events.append)
    assert router.submit("click") is False
    engine.process_frame(make_ctx().frame)
    assert "OPENAI_API_KEY" in events[0].message
    engine.close()
    with pytest.raises(LLMUnavailable):
        HCIEngine(config, sensors=[], processors=[LLMRouter(env={}, strict=True)], actions=[]).setup()


def test_unreachable_local_server_is_an_event_not_a_crash():
    engine, router, _ = router_engine([LLMError("Cannot reach http://localhost:11434: refused")], cfg=LLMConfig(provider="ollama"))
    events = []
    engine.on("llm_error", events.append)
    router.submit("click")
    pump(engine, router)
    assert "Cannot reach" in events[0].message
    engine.close()


def test_busy_router_drops_new_commands():
    gate = threading.Event()

    def blocked(method, url, headers, body, timeout):
        gate.wait(5)
        return 200, json.dumps(openai_reply(actions_json())[1])

    engine, router, _ = router_engine(blocked, cfg=LLMConfig(provider="lmstudio", max_pending=2))
    events = []
    engine.on("llm_error", events.append)
    assert [router.submit(f"cmd {i}") for i in range(4)] == [True, True, False, False]
    engine.process_frame(make_ctx().frame)
    assert len(events) == 2 and "busy" in events[0].message
    gate.set()
    assert router.wait_idle(5)
    engine.close()


def test_empty_text_is_ignored(bare_engine):
    router = LLMRouter(provider=OpenAICompatibleProvider("http://x/v1", "m", http=FakeHTTP(openai_reply("{}"))))
    bare_engine.add_processor(router)
    router.setup()
    assert router.submit("   ") is False
    router.close()


def test_dry_run_marks_intents_and_actions_do_not_execute():
    pag = FakePyAutoGUI()
    reply = actions_json({"action": "click"}, {"action": "hotkey", "keys": ["ctrl", "t"]}, {"action": "type", "text": "x"})
    engine, router, _ = router_engine(
        [openai_reply(reply)],
        cfg=LLMConfig(provider="lmstudio", dry_run=True),
        actions=[OSMouseAction(pyautogui_module=pag), OSKeyboardAction(pyautogui_module=pag)],
    )
    router.submit("do it")
    pump(engine, router)
    assert pag.calls == []
    engine.close()


def test_end_to_end_voice_style_commands_drive_the_real_actions():
    pag = FakePyAutoGUI()
    replies = [
        openai_reply(actions_json({"action": "hotkey", "keys": ["ctrl", "t"]})),
        openai_reply(actions_json({"action": "scroll", "amount": -500})),
        openai_reply(actions_json({"action": "type", "text": "Best regards,\nSam"})),
        openai_reply(actions_json({"action": "right_click"}, {"action": "hotkey", "keys": ["alt", "f4"]})),
    ]
    engine, router, _ = router_engine(
        replies,
        cfg=LLMConfig(provider="lmstudio"),
        actions=[OSMouseAction(pyautogui_module=pag, wheel_unit=120), OSKeyboardAction(pyautogui_module=pag)],
    )
    for text in ("open this in a new tab", "scroll down slowly", "type my email signature", "right click and close the window"):
        router.submit(text)
        pump(engine, router)
    assert pag.calls == [
        ("hotkey", ("ctrl", "t"), {}),
        ("scroll", (-600,), {}),  # -500 units = 5 notches = 600 Windows wheel units
        ("write", ("Best regards,\nSam",), {"interval": 0.0}),
        ("click", (), {"button": "right"}),  # the Alt+F4 half of the answer was refused
    ]
    engine.close()


def test_route_is_a_blocking_helper_and_raises_on_failure():
    router = LLMRouter(provider=OpenAICompatibleProvider("http://x/v1", "m", http=FakeHTTP(openai_reply(actions_json({"action": "click"})))))
    assert [a.intent for a in router.route("click").actions] == ["click"]
    bad = LLMRouter(provider=OpenAICompatibleProvider("http://x/v1", "m", http=FakeHTTP((500, {"error": "x"}))))
    with pytest.raises(LLMError):
        bad.route("click")
    unavailable = LLMRouter(config=LLMConfig(provider="gemini"), env={})
    with pytest.raises(LLMUnavailable):
        unavailable.route("click")


def test_reconfigure_rebuilds_the_provider(bare_engine):
    router = LLMRouter(http=FakeHTTP(openai_reply("{}")), env={})
    bare_engine.add_processor(router)
    router.setup()
    assert router.provider.model == "llama3.2" and router.provider.name == "ollama"
    bare_engine.config.llm.provider, bare_engine.config.llm.model = "lmstudio", "qwen2.5"
    router.reconfigure()
    assert router.provider.model == "qwen2.5" and router.available
    bare_engine.config.llm.provider = "openai"  # no key in env -> graceful
    router.reconfigure()
    assert not router.available and router.provider is None
    router.close()

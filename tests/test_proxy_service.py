import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openaiproxy.api.app import create_app
from openaiproxy.services.proxy_service import filter_response_headers
from openaiproxy.services.slot_cache_service import server_root, slot_filename
from openaiproxy.yaml_repository.yaml_repository import YAMLLLMProxyConfigRepository


def test_filter_response_headers_strips_framing_headers():
    upstream_headers = {
        "Content-Type": "application/json",
        "Content-Length": "9999",
        "Transfer-Encoding": "chunked",
        "Content-Encoding": "gzip",
        "Connection": "keep-alive",
        "X-Request-Id": "abc-123",
    }
    filtered = filter_response_headers(upstream_headers)
    assert filtered == {
        "Content-Type": "application/json",
        "X-Request-Id": "abc-123",
    }


def _make_client(
    tmp_path: Path,
    monkeypatch,
    log: bool,
    cache_prompt: bool = False,
    slot_cache: bool = False,
    slot_count: int = 0,
    backend: str | None = None,
    props_total_slots: int | None = None,
    router_mode: bool = False,
    captured_requests: list | None = None,
) -> TestClient:
    import httpx

    config_path = tmp_path / "llmproxy.yaml"
    config_path.write_text(
        "llmproxy:\n"
        "  endpoints:\n"
        "    default:\n"
        "      base_url: http://upstream.test/v1\n"
        "      models: ['*']\n"
        f"      log: {str(log).lower()}\n"
        f"      cache_prompt: {str(cache_prompt).lower()}\n"
        f"      slot_cache: {str(slot_cache).lower()}\n"
        + (f"      slot_count: {slot_count}\n" if slot_count else "")
        + (f"      backend: {backend}\n" if backend else "")
    )

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        if captured_requests is not None:
            captured_requests.append(request)
        if request.url.path == "/props":
            model = request.url.params.get("model")
            if router_mode and not model:
                # a router with no model answers 200 with a stub carrying no total_slots
                return httpx.Response(200, json={"role": "router", "build_info": "b10398"})
            if props_total_slots is None:
                return httpx.Response(404, content=b"not found")
            return httpx.Response(200, json={"total_slots": props_total_slots})
        body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "hi"}}]})
        return httpx.Response(
            200,
            content=body,
            headers={
                "Content-Type": "application/json",
                "Transfer-Encoding": "chunked",
                "Content-Length": str(len(body) + 42),
                "X-Upstream": "llama.cpp",
            },
        )

    transport = httpx.MockTransport(upstream_handler)
    original_client = httpx.AsyncClient

    def patched_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)

    repository = YAMLLLMProxyConfigRepository(config_path)
    app = create_app(
        config_repository=repository,
        logs_dir=tmp_path / "logs",
        trace_dir=tmp_path / "traces",
    )
    return TestClient(app)


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    return _make_client(tmp_path, monkeypatch, log=False)


def test_non_streaming_response_has_consistent_framing(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.1", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hi"

    assert "transfer-encoding" not in response.headers
    assert "content-encoding" not in response.headers
    assert response.headers["x-upstream"] == "llama.cpp"
    assert int(response.headers["content-length"]) == len(response.content)


def test_models_endpoint_not_traced_or_ledgered(tmp_path: Path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, log=True)
    trace_dir = tmp_path / "traces"

    response = client.get("/v1/models")
    assert response.status_code == 200

    assert not list(trace_dir.rglob("*_request.json"))
    assert not (trace_dir / "validation_ledger.ndjson").exists()

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert len(list((trace_dir / "completion").glob("*_request.json"))) == 1


def test_cache_prompt_added_when_enabled(tmp_path: Path, monkeypatch):
    captured: list = []
    client = _make_client(tmp_path, monkeypatch, log=False, cache_prompt=True, captured_requests=captured)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200

    forwarded = json.loads(captured[0].content)
    assert forwarded["cache_prompt"] is True
    assert forwarded["model"] == "gpt-5.1"


def test_cache_prompt_absent_when_disabled(tmp_path: Path, monkeypatch):
    captured: list = []
    client = _make_client(tmp_path, monkeypatch, log=False, cache_prompt=False, captured_requests=captured)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200

    forwarded = json.loads(captured[0].content)
    assert "cache_prompt" not in forwarded


def test_slot_filename_and_server_root():
    assert slot_filename(None) == "slot_default.bin"
    assert slot_filename("") == "slot_default.bin"
    assert slot_filename("thread-42") == "slot_thread-42.bin"
    assert slot_filename("agent/run:1") == "slot_agent_run_1.bin"
    # the model is part of the name: one session may span models, and a slot
    # snapshot is only valid for the instance that wrote it
    assert slot_filename("t1", "unsloth/DeepSeek-V4:Q2_K_XL") == "slot_t1__unsloth_DeepSeek-V4_Q2_K_XL.bin"
    assert server_root("http://gpu.lan:8080/v1") == "http://gpu.lan:8080"
    assert server_root("http://gpu.lan:8080") == "http://gpu.lan:8080"


def test_slot_cache_restore_and_save_with_correlation_id(tmp_path: Path, monkeypatch):
    captured: list = []
    client = _make_client(
        tmp_path, monkeypatch, log=False, slot_cache=True, props_total_slots=1, captured_requests=captured
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.1", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Correlation-Id": "thread-42"},
    )
    assert response.status_code == 200

    urls = [str(r.url) for r in captured]
    assert urls == [
        "http://upstream.test/props?model=gpt-5.1&autoload=true",
        "http://upstream.test/slots/0?action=restore",
        "http://upstream.test/v1/chat/completions",
        "http://upstream.test/slots/0?action=save",
    ]
    # the model rides in the body: a router routes POST /slots by it
    expected_body = {"filename": "slot_thread-42__gpt-5.1.bin", "model": "gpt-5.1"}
    assert json.loads(captured[1].content) == expected_body
    assert json.loads(captured[2].content)["id_slot"] == 0
    assert json.loads(captured[3].content) == expected_body


def test_slot_cache_falls_back_to_default_session(tmp_path: Path, monkeypatch):
    captured: list = []
    client = _make_client(
        tmp_path, monkeypatch, log=False, slot_cache=True, props_total_slots=1, captured_requests=captured
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200

    slot_calls = [r for r in captured if "/slots/" in str(r.url)]
    assert len(slot_calls) == 2
    assert all(
        json.loads(r.content) == {"filename": "slot_default__gpt-5.1.bin", "model": "gpt-5.1"}
        for r in slot_calls
    )


def test_slot_allocator_assigns_distinct_slots_per_session(tmp_path: Path, monkeypatch):
    captured: list = []
    client = _make_client(
        tmp_path, monkeypatch, log=False, slot_cache=True, props_total_slots=2, captured_requests=captured
    )
    body = {"model": "gpt-5.1", "messages": [{"role": "user", "content": "hi"}]}

    assert client.post("/v1/chat/completions", json=body, headers={"X-Correlation-Id": "a"}).status_code == 200
    assert client.post("/v1/chat/completions", json=body, headers={"X-Correlation-Id": "b"}).status_code == 200

    chat_calls = [r for r in captured if str(r.url).endswith("/chat/completions")]
    assert json.loads(chat_calls[0].content)["id_slot"] == 0
    assert json.loads(chat_calls[1].content)["id_slot"] == 1
    slot_urls = [str(r.url) for r in captured if "/slots/" in str(r.url)]
    assert slot_urls == [
        "http://upstream.test/slots/0?action=restore",
        "http://upstream.test/slots/0?action=save",
        "http://upstream.test/slots/1?action=restore",
        "http://upstream.test/slots/1?action=save",
    ]


def test_slot_allocator_skips_restore_when_session_resident(tmp_path: Path, monkeypatch):
    captured: list = []
    client = _make_client(
        tmp_path, monkeypatch, log=False, slot_cache=True, props_total_slots=1, captured_requests=captured
    )
    body = {"model": "gpt-5.1", "messages": [{"role": "user", "content": "hi"}]}

    assert client.post("/v1/chat/completions", json=body, headers={"X-Correlation-Id": "a"}).status_code == 200
    assert client.post("/v1/chat/completions", json=body, headers={"X-Correlation-Id": "a"}).status_code == 200

    slot_urls = [str(r.url).split("?")[1] for r in captured if "/slots/" in str(r.url)]
    # turn 1: restore + save; turn 2: the session's KV is still resident — save only
    assert slot_urls == ["action=restore", "action=save", "action=save"]


def test_slot_allocator_lru_eviction_restores_returning_session(tmp_path: Path, monkeypatch):
    captured: list = []
    client = _make_client(
        tmp_path, monkeypatch, log=False, slot_cache=True, props_total_slots=1, captured_requests=captured
    )
    body = {"model": "gpt-5.1", "messages": [{"role": "user", "content": "hi"}]}

    for cid in ("a", "b", "a"):
        assert client.post("/v1/chat/completions", json=body, headers={"X-Correlation-Id": cid}).status_code == 200

    restores = [json.loads(r.content)["filename"] for r in captured if "action=restore" in str(r.url)]
    # every turn changes the slot's occupant, so every turn restores
    assert restores == ["slot_a__gpt-5.1.bin", "slot_b__gpt-5.1.bin", "slot_a__gpt-5.1.bin"]


def test_responses_path_uses_slot_cache_and_cache_prompt(tmp_path: Path, monkeypatch):
    captured: list = []
    client = _make_client(
        tmp_path, monkeypatch, log=False,
        cache_prompt=True, slot_cache=True, props_total_slots=1,
        captured_requests=captured,
    )

    response = client.post(
        "/v1/responses",
        json={"model": "gpt-5.1", "input": "hi"},
        headers={"X-Correlation-Id": "thread-9"},
    )
    assert response.status_code == 200

    urls = [str(r.url) for r in captured]
    assert urls == [
        "http://upstream.test/props?model=gpt-5.1&autoload=true",
        "http://upstream.test/slots/0?action=restore",
        "http://upstream.test/v1/chat/completions",
        "http://upstream.test/slots/0?action=save",
    ]
    chat_body = json.loads(captured[2].content)
    assert chat_body["cache_prompt"] is True
    assert chat_body["id_slot"] == 0
    assert json.loads(captured[1].content) == {"filename": "slot_thread-9__gpt-5.1.bin", "model": "gpt-5.1"}


def test_responses_stream_path_uses_slot_cache(tmp_path: Path, monkeypatch):
    captured: list = []
    client = _make_client(
        tmp_path, monkeypatch, log=False, slot_cache=True, props_total_slots=1,
        captured_requests=captured,
    )

    response = client.post(
        "/v1/responses",
        json={"model": "gpt-5.1", "input": "hi", "stream": True},
        headers={"X-Correlation-Id": "thread-9"},
    )
    assert response.status_code == 200
    response.read()

    urls = [str(r.url) for r in captured]
    assert urls[:2] == [
        "http://upstream.test/props?model=gpt-5.1&autoload=true",
        "http://upstream.test/slots/0?action=restore",
    ]
    assert urls[-1] == "http://upstream.test/slots/0?action=save"
    chat_calls = [r for r in captured if str(r.url).endswith("/chat/completions")]
    assert all(json.loads(r.content)["id_slot"] == 0 for r in chat_calls)


def test_slot_count_override_skips_probe(tmp_path: Path, monkeypatch):
    captured: list = []
    client = _make_client(
        tmp_path, monkeypatch, log=False, slot_cache=True, slot_count=2, captured_requests=captured
    )
    body = {"model": "gpt-5.1", "messages": [{"role": "user", "content": "hi"}]}

    assert client.post("/v1/chat/completions", json=body, headers={"X-Correlation-Id": "a"}).status_code == 200
    assert client.post("/v1/chat/completions", json=body, headers={"X-Correlation-Id": "b"}).status_code == 200

    assert not any(str(r.url).endswith("/props") for r in captured)
    chat_calls = [r for r in captured if str(r.url).endswith("/chat/completions")]
    assert [json.loads(r.content)["id_slot"] for r in chat_calls] == [0, 1]


def test_slot_cache_disabled_makes_no_slot_calls(tmp_path: Path, monkeypatch):
    captured: list = []
    client = _make_client(tmp_path, monkeypatch, log=False, slot_cache=False, captured_requests=captured)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.1", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Correlation-Id": "thread-42"},
    )
    assert response.status_code == 200
    assert [str(r.url) for r in captured] == ["http://upstream.test/v1/chat/completions"]


def test_llamacpp_flags_ignored_for_generic_backend(tmp_path: Path, monkeypatch):
    captured: list = []
    client = _make_client(
        tmp_path, monkeypatch, log=False,
        cache_prompt=True, slot_cache=True, backend="generic",
        captured_requests=captured,
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.1", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Correlation-Id": "thread-42"},
    )
    assert response.status_code == 200

    assert [str(r.url) for r in captured] == ["http://upstream.test/v1/chat/completions"]
    assert "cache_prompt" not in json.loads(captured[0].content)


def test_backend_inferred_from_llamacpp_flags(tmp_path: Path):
    config_path = tmp_path / "llmproxy.yaml"
    config_path.write_text(
        "llmproxy:\n"
        "  endpoints:\n"
        "    cached:\n"
        "      base_url: http://a.test/v1\n"
        "      models: ['*']\n"
        "      cache_prompt: true\n"
        "    plain:\n"
        "      base_url: http://b.test/v1\n"
        "      models: ['m']\n"
        "    explicit:\n"
        "      base_url: http://c.test/v1\n"
        "      models: ['n']\n"
        "      slot_cache: true\n"
        "      backend: generic\n"
    )
    endpoints = {e.name: e for e in YAMLLLMProxyConfigRepository(config_path).load().endpoints}
    assert endpoints["cached"].backend == "llamacpp"
    assert endpoints["plain"].backend == "generic"
    assert endpoints["explicit"].backend == "generic"


def test_detect_backend_llamacpp(tmp_path: Path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, log=False, props_total_slots=4)

    response = client.post("/ui/api/config/detect_backend", json={"base_url": "http://upstream.test/v1"})
    assert response.status_code == 200
    assert response.json() == {"reachable": True, "backend": "llamacpp", "total_slots": 4, "router": False}


def test_detect_backend_generic(tmp_path: Path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch, log=False)

    response = client.post("/ui/api/config/detect_backend", json={"base_url": "http://upstream.test/v1"})
    assert response.status_code == 200
    assert response.json() == {
        "reachable": True, "backend": "generic", "total_slots": None, "router": False
    }


def test_detect_backend_router_without_model(tmp_path: Path, monkeypatch):
    # a router with no model answers 200 but omits total_slots — the old check
    # read that as "not llama.cpp"
    captured: list = []
    client = _make_client(
        tmp_path, monkeypatch, log=False, props_total_slots=2, router_mode=True, captured_requests=captured
    )

    response = client.post("/ui/api/config/detect_backend", json={"base_url": "http://upstream.test/v1"})
    assert response.status_code == 200
    assert response.json() == {"reachable": True, "backend": "llamacpp", "total_slots": None, "router": True}


def test_detect_backend_router_with_model_reports_slots(tmp_path: Path, monkeypatch):
    captured: list = []
    client = _make_client(
        tmp_path, monkeypatch, log=False, props_total_slots=2, router_mode=True, captured_requests=captured
    )

    response = client.post(
        "/ui/api/config/detect_backend",
        json={"base_url": "http://upstream.test/v1", "model": "DeepSeek-V4-Flash"},
    )
    assert response.status_code == 200
    assert response.json() == {"reachable": True, "backend": "llamacpp", "total_slots": 2, "router": False}
    # detection must not boot an idle model
    assert "autoload=false" in str(captured[0].url)

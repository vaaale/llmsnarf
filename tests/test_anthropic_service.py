import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from openaiproxy.api.app import create_app
from openaiproxy.yaml_repository.yaml_repository import YAMLLLMProxyConfigRepository


CONFIG = """
llmproxy:
  endpoints:
    anthropic-backend:
      base_url: http://anthropic.test/v1
      protocol: anthropic
      api_key: sk-ant-endpoint
      models: ['claude-x']
      aliases:
        claude-alias: claude-x
      log: false
    openai-backend:
      base_url: http://openai.test/v1
      api_key: sk-oai-endpoint
      models: ['gpt-x']
      log: false
"""


ANTHROPIC_MESSAGE_RESPONSE = {
    "id": "msg_upstream",
    "type": "message",
    "role": "assistant",
    "model": "claude-x",
    "content": [{"type": "text", "text": "Hello from Claude"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 11, "output_tokens": 7},
}

CHAT_COMPLETION_RESPONSE = {
    "id": "chatcmpl-upstream",
    "object": "chat.completion",
    "model": "gpt-x",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from GPT"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}

ANTHROPIC_STREAM = (
    'event: message_start\ndata: {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 4}}}\n\n'
    'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}\n\n'
    'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}\n\n'
    'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 0}\n\n'
    'event: message_delta\ndata: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}}\n\n'
    'event: message_stop\ndata: {"type": "message_stop"}\n\n'
)

CHAT_STREAM = (
    'data: {"id": "c1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]}\n\n'
    'data: {"id": "c1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": null}]}\n\n'
    'data: {"id": "c1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}\n\n'
    'data: {"id": "c1", "object": "chat.completion.chunk", "choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}}\n\n'
    "data: [DONE]\n\n"
)


@pytest.fixture
def captured_requests() -> list[httpx.Request]:
    return []


@pytest.fixture
def client(tmp_path: Path, monkeypatch, captured_requests) -> TestClient:
    config_path = tmp_path / "llmproxy.yaml"
    config_path.write_text(CONFIG)

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        payload = json.loads(request.content) if request.content else {}
        streaming = payload.get("stream")
        if request.url.host == "anthropic.test":
            assert request.url.path == "/v1/messages"
            if streaming:
                return httpx.Response(
                    200, content=ANTHROPIC_STREAM, headers={"Content-Type": "text/event-stream"}
                )
            return httpx.Response(200, json=ANTHROPIC_MESSAGE_RESPONSE)
        if request.url.host == "openai.test":
            assert request.url.path == "/v1/chat/completions"
            if streaming:
                return httpx.Response(
                    200, content=CHAT_STREAM, headers={"Content-Type": "text/event-stream"}
                )
            return httpx.Response(200, json=CHAT_COMPLETION_RESPONSE)
        return httpx.Response(500, json={"error": "unexpected host"})

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


def parse_sse_data(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data:"):
            data = line[5:].strip()
            if data == "[DONE]":
                events.append({"type": "[DONE]"})
            else:
                events.append(json.loads(data))
    return events


# ---------------------------------------------------------------------------
# OpenAI client -> Anthropic backend
# ---------------------------------------------------------------------------


def test_openai_client_to_anthropic_backend(client: TestClient, captured_requests):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-x",
            "messages": [
                {"role": "system", "content": "Be nice"},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 100,
        },
        headers={"Authorization": "Bearer client-token"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Hello from Claude"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "prompt_tokens_details": {"cached_tokens": 0},
    }

    upstream = captured_requests[0]
    assert upstream.headers["x-api-key"] == "sk-ant-endpoint"
    assert upstream.headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in upstream.headers
    sent = json.loads(upstream.content)
    assert sent["system"] == "Be nice"
    assert sent["max_tokens"] == 100
    assert sent["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]


def test_openai_client_to_anthropic_backend_streaming(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-x",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    chunks = parse_sse_data(response.text)
    role_chunk = chunks[0]
    assert role_chunk["object"] == "chat.completion.chunk"
    assert role_chunk["choices"][0]["delta"]["role"] == "assistant"
    content = "".join(
        c["choices"][0]["delta"].get("content", "")
        for c in chunks
        if c.get("choices")
    )
    assert content == "Hi"
    finish = next(c for c in chunks if c.get("choices") and c["choices"][0].get("finish_reason"))
    assert finish["choices"][0]["finish_reason"] == "stop"
    usage_chunk = next(c for c in chunks if c.get("usage"))
    assert usage_chunk["usage"]["prompt_tokens"] == 4
    assert usage_chunk["usage"]["completion_tokens"] == 2
    assert chunks[-1]["type"] == "[DONE]"


# ---------------------------------------------------------------------------
# Anthropic client -> OpenAI backend
# ---------------------------------------------------------------------------


def test_anthropic_client_to_openai_backend(client: TestClient, captured_requests):
    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-x",
            "max_tokens": 64,
            "system": "Be nice",
            "messages": [{"role": "user", "content": "Hello"}],
        },
        headers={"x-api-key": "client-anthropic-key"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["model"] == "gpt-x"
    assert data["content"] == [{"type": "text", "text": "Hello from GPT"}]
    assert data["stop_reason"] == "end_turn"
    assert data["usage"]["input_tokens"] == 5
    assert data["usage"]["output_tokens"] == 3

    upstream = captured_requests[0]
    assert upstream.headers["authorization"] == "Bearer sk-oai-endpoint"
    sent = json.loads(upstream.content)
    assert sent["messages"] == [
        {"role": "system", "content": "Be nice"},
        {"role": "user", "content": "Hello"},
    ]
    assert sent["max_tokens"] == 64


def test_anthropic_client_to_openai_backend_streaming(client: TestClient, captured_requests):
    response = client.post(
        "/v1/messages",
        json={
            "model": "gpt-x",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    lines = response.text
    assert "event: message_start\n" in lines
    assert "event: message_stop\n" in lines

    events = parse_sse_data(lines)
    types = [e["type"] for e in events]
    assert types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[0]["message"]["model"] == "gpt-x"
    assert events[2]["delta"] == {"type": "text_delta", "text": "Hi"}
    assert events[4]["delta"]["stop_reason"] == "end_turn"
    assert events[4]["usage"] == {"input_tokens": 4, "output_tokens": 2}

    sent = json.loads(captured_requests[0].content)
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}


# ---------------------------------------------------------------------------
# Anthropic client -> Anthropic backend (native passthrough)
# ---------------------------------------------------------------------------


def test_anthropic_client_to_anthropic_backend_passthrough(client: TestClient, captured_requests):
    payload = {
        "model": "claude-x",
        "max_tokens": 100,
        "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "Hello"}],
        "thinking": {"type": "enabled", "budget_tokens": 500},
    }
    response = client.post("/v1/messages", json=payload, headers={"x-api-key": "client-key"})
    assert response.status_code == 200
    assert response.json() == ANTHROPIC_MESSAGE_RESPONSE

    upstream = captured_requests[0]
    assert upstream.headers["x-api-key"] == "client-key"
    assert upstream.headers["anthropic-version"] == "2023-06-01"
    # payload must pass through untouched, including provider-specific fields
    assert json.loads(upstream.content) == payload


def test_anthropic_passthrough_applies_alias(client: TestClient, captured_requests):
    response = client.post(
        "/v1/messages",
        json={"model": "claude-alias", "max_tokens": 5, "messages": [{"role": "user", "content": "x"}]},
    )
    assert response.status_code == 200
    sent = json.loads(captured_requests[0].content)
    assert sent["model"] == "claude-x"


def test_anthropic_passthrough_uses_endpoint_key_when_client_has_none(
    client: TestClient, captured_requests
):
    response = client.post(
        "/v1/messages",
        json={"model": "claude-x", "max_tokens": 5, "messages": [{"role": "user", "content": "x"}]},
    )
    assert response.status_code == 200
    assert captured_requests[0].headers["x-api-key"] == "sk-ant-endpoint"


def test_anthropic_client_to_anthropic_backend_streaming(client: TestClient):
    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-x",
            "max_tokens": 5,
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    # passthrough: upstream SSE bytes arrive verbatim
    assert response.text == ANTHROPIC_STREAM


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_messages_routing_error_is_anthropic_format(client: TestClient):
    response = client.post(
        "/v1/messages",
        json={"model": "unknown-model", "max_tokens": 5, "messages": [{"role": "user", "content": "x"}]},
    )
    assert response.status_code == 404
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "not_found_error"


def test_messages_invalid_body_is_anthropic_format(client: TestClient):
    response = client.post(
        "/v1/messages", content=b"not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_chat_completions_error_from_anthropic_backend_is_openai_format(
    tmp_path: Path, monkeypatch
):
    config_path = tmp_path / "llmproxy.yaml"
    config_path.write_text(CONFIG)

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}},
        )

    transport = httpx.MockTransport(upstream_handler)
    original_client = httpx.AsyncClient

    def patched_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)

    repository = YAMLLLMProxyConfigRepository(config_path)
    app = create_app(
        config_repository=repository, logs_dir=tmp_path / "logs", trace_dir=tmp_path / "traces"
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "claude-x", "messages": [{"role": "user", "content": "x"}]},
    )
    assert response.status_code == 429
    body = response.json()
    assert body["error"]["message"] == "slow down"


def test_messages_error_from_openai_backend_is_anthropic_format(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "llmproxy.yaml"
    config_path.write_text(CONFIG)

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401, json={"error": {"message": "bad key", "type": "invalid_request_error"}}
        )

    transport = httpx.MockTransport(upstream_handler)
    original_client = httpx.AsyncClient

    def patched_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)

    repository = YAMLLLMProxyConfigRepository(config_path)
    app = create_app(
        config_repository=repository, logs_dir=tmp_path / "logs", trace_dir=tmp_path / "traces"
    )
    client = TestClient(app)

    response = client.post(
        "/v1/messages",
        json={"model": "gpt-x", "max_tokens": 5, "messages": [{"role": "user", "content": "x"}]},
    )
    assert response.status_code == 401
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "authentication_error"
    assert body["error"]["message"] == "bad key"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_protocol_field_round_trips_through_yaml(tmp_path: Path):
    config_path = tmp_path / "llmproxy.yaml"
    config_path.write_text(CONFIG)
    repository = YAMLLLMProxyConfigRepository(config_path)
    config = repository.load()

    by_name = {e.name: e for e in config.endpoints}
    assert by_name["anthropic-backend"].protocol == "anthropic"
    assert by_name["openai-backend"].protocol == "openai"

    repository.save(config)
    reloaded = YAMLLLMProxyConfigRepository(config_path).load()
    by_name = {e.name: e for e in reloaded.endpoints}
    assert by_name["anthropic-backend"].protocol == "anthropic"
    assert by_name["openai-backend"].protocol == "openai"

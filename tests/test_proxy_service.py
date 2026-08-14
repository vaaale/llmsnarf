import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openaiproxy.api.app import create_app
from openaiproxy.services.proxy_service import filter_response_headers
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
    )

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        if captured_requests is not None:
            captured_requests.append(request)
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

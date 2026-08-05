import asyncio
import base64
import json
import struct
from pathlib import Path

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from openaiproxy.api.app import create_app
from openaiproxy.models.trace_models import trace_category
from openaiproxy.yaml_repository.yaml_repository import YAMLLLMProxyConfigRepository


class FakeLocalBackend:
    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []

    async def embed(self, model_id: str, texts: list[str]):
        self.calls.append((model_id, list(texts)))
        return [[0.1 * (i + 1), 0.2, 0.3] for i in range(len(texts))], 7


def _make_app(tmp_path: Path, config_text: str):
    config_path = tmp_path / "llmproxy.yaml"
    config_path.write_text(config_text)
    repository = YAMLLLMProxyConfigRepository(config_path)
    return create_app(
        config_repository=repository,
        logs_dir=tmp_path / "logs",
        trace_dir=tmp_path / "traces",
    )


LOCAL_CONFIG = (
    "llmproxy:\n"
    "  endpoints:\n"
    "    local-embed:\n"
    "      mode: local\n"
    "      models: ['my-embed']\n"
    "      aliases:\n"
    "        my-embed: hf/mini-model\n"
)


@pytest.fixture
def local_client(tmp_path: Path):
    app = _make_app(tmp_path, LOCAL_CONFIG)
    backend = FakeLocalBackend()
    app.state.embeddings_service._local_backend = backend
    return TestClient(app), backend


def test_local_embeddings_basic(local_client):
    client, backend = local_client
    response = client.post(
        "/v1/embeddings",
        json={"model": "my-embed", "input": ["hello", "world"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["model"] == "my-embed"
    assert len(body["data"]) == 2
    assert body["data"][0]["object"] == "embedding"
    assert body["data"][0]["index"] == 0
    assert body["data"][1]["embedding"] == [pytest.approx(0.2), pytest.approx(0.2), pytest.approx(0.3)]
    assert body["usage"] == {"prompt_tokens": 7, "total_tokens": 7}
    # alias resolved to the transformer model id
    assert backend.calls == [("hf/mini-model", ["hello", "world"])]


def test_local_embeddings_string_input_and_base64(local_client):
    client, _ = local_client
    response = client.post(
        "/v1/embeddings",
        json={"model": "my-embed", "input": "hello", "encoding_format": "base64"},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["data"]) == 1
    raw = base64.b64decode(body["data"][0]["embedding"])
    floats = struct.unpack(f"<{len(raw) // 4}f", raw)
    assert list(floats) == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]


def test_local_embeddings_rejects_token_arrays(local_client):
    client, backend = local_client
    response = client.post(
        "/v1/embeddings",
        json={"model": "my-embed", "input": [[1, 2, 3]]},
    )
    assert response.status_code == 400
    assert backend.calls == []


def test_local_embeddings_traced_under_embeddings_folder(local_client, tmp_path: Path):
    client, _ = local_client
    response = client.post("/v1/embeddings", json={"model": "my-embed", "input": "hi"})
    assert response.status_code == 200
    trace_dir = tmp_path / "traces" / "embeddings"
    requests = list(trace_dir.glob("*_request.json"))
    responses = list(trace_dir.glob("*_response.json"))
    assert len(requests) == 1
    assert len(responses) == 1
    request_data = json.loads(requests[0].read_text())
    assert request_data["endpoint"] == "/embeddings"


def test_embeddings_no_route_returns_404(tmp_path: Path):
    app = _make_app(tmp_path, LOCAL_CONFIG)
    client = TestClient(app)
    response = client.post("/v1/embeddings", json={"model": "unknown", "input": "hi"})
    assert response.status_code == 404


def test_remote_embeddings_forwarded(tmp_path: Path, monkeypatch):
    import httpx

    seen: dict = {}

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 2.0]}],
                "model": "real-embed",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    transport = httpx.MockTransport(upstream_handler)
    original_client = httpx.AsyncClient

    def patched_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)

    app = _make_app(
        tmp_path,
        "llmproxy:\n"
        "  endpoints:\n"
        "    remote-embed:\n"
        "      base_url: http://upstream.test/v1\n"
        "      api_key: sk-upstream\n"
        "      models: ['text-embedding-3-small']\n"
        "      aliases:\n"
        "        text-embedding-3-small: real-embed\n",
    )
    client = TestClient(app)

    response = client.post(
        "/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "hi"},
    )
    assert response.status_code == 200
    assert response.json()["data"][0]["embedding"] == [1.0, 2.0]
    assert seen["url"] == "http://upstream.test/v1/embeddings"
    assert seen["payload"]["model"] == "real-embed"
    assert seen["auth"] == "Bearer sk-upstream"

    trace_dir = tmp_path / "traces" / "embeddings"
    assert len(list(trace_dir.glob("*_request.json"))) == 1
    assert len(list(trace_dir.glob("*_response.json"))) == 1


def test_local_endpoint_not_used_for_chat_completions(tmp_path: Path):
    app = _make_app(
        tmp_path,
        "llmproxy:\n"
        "  endpoints:\n"
        "    local-embed:\n"
        "      mode: local\n"
        "      models: ['*']\n",
    )
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404


def test_trace_category_mapping():
    assert trace_category("/chat/completions") == "completion"
    assert trace_category("/completions") == "completion"
    assert trace_category("/responses") == "responses"
    assert trace_category("/responses/map-reduce/1/3") == "responses"
    assert trace_category("/messages") == "messages"
    assert trace_category("/embeddings") == "embeddings"
    assert trace_category("/rerank") == "rerank"
    assert trace_category("") == "other"


def test_openai_sdk_parses_local_embeddings(tmp_path: Path):
    """The official SDK strictly parses the response (and asks for base64 by
    default), so this verifies wire-format correctness end to end."""
    app = _make_app(tmp_path, LOCAL_CONFIG)
    app.state.embeddings_service._local_backend = FakeLocalBackend()

    async def run():
        client = openai.AsyncOpenAI(
            api_key="sk-test",
            base_url="http://proxy.test/v1",
            http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app)),
        )
        return await client.embeddings.create(model="my-embed", input=["hello", "world"])

    result = asyncio.run(run())
    assert result.model == "my-embed"
    assert len(result.data) == 2
    assert result.data[0].embedding == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]
    assert result.usage.prompt_tokens == 7


def test_yaml_roundtrip_local_mode(tmp_path: Path):
    config_path = tmp_path / "llmproxy.yaml"
    config_path.write_text(LOCAL_CONFIG)
    repository = YAMLLLMProxyConfigRepository(config_path)
    config = repository.load()
    assert len(config.endpoints) == 1
    endpoint = config.endpoints[0]
    assert endpoint.mode == "local"
    assert endpoint.base_url == ""

    repository.save(config)
    reloaded = repository.load()
    assert reloaded.endpoints[0].mode == "local"
    assert reloaded.endpoints[0].models == ["my-embed"]

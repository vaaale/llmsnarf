import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openaiproxy.api.app import create_app
from openaiproxy.yaml_repository.yaml_repository import YAMLLLMProxyConfigRepository


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    config_path = tmp_path / "llmproxy.yaml"
    config_path.write_text(
        "llmproxy:\n"
        "  host: '0.0.0.0'\n"
        "  port: 8000\n"
        "  endpoints:\n"
        "    default:\n"
        "      base_url: http://localhost:9000/v1\n"
        "      api_key: sk-secret\n"
        "      models: ['*']\n"
        "      log: true\n"
    )
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "sk-test_20260706_111832_127166_request.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-07-06T11:18:32.127209",
                "endpoint": "/chat/completions",
                "headers": {},
                "payload": {"model": "gpt-5.1", "messages": [{"role": "user", "content": "hi"}]},
            }
        )
    )
    (trace_dir / "sk-test_20260706_111832_127166_response.json").write_text(
        json.dumps({"timestamp": "2026-07-06T11:18:33.000000", "status_code": 200, "headers": {}, "body": {}})
    )

    repository = YAMLLLMProxyConfigRepository(config_path)
    app = create_app(
        config_repository=repository,
        logs_dir=tmp_path / "logs",
        trace_dir=trace_dir,
    )
    return TestClient(app)


def test_get_config_masks_api_key(client: TestClient):
    response = client.get("/ui/api/config")
    assert response.status_code == 200
    endpoint = response.json()["endpoints"][0]
    assert endpoint["api_key"] is None
    assert endpoint["api_key_masked"] == "sk-s••••"


def test_put_config_updates_and_keeps_key(client: TestClient):
    payload = {
        "endpoints": [
            {
                "name": "default",
                "base_url": "http://localhost:9999/v1",
                "api_key": None,
                "models": ["*"],
                "aliases": {"a": "b"},
                "substitute_role": {},
                "log": False,
                "enabled": True,
            }
        ]
    }
    response = client.put("/ui/api/config", json=payload)
    assert response.status_code == 200
    endpoint = response.json()["endpoints"][0]
    assert endpoint["base_url"] == "http://localhost:9999/v1"
    assert endpoint["log"] is False
    assert endpoint["api_key_masked"] == "sk-s••••"


def test_put_config_validation_error(client: TestClient):
    payload = {
        "endpoints": [
            {"name": "", "base_url": "http://x/v1", "models": ["*"]},
        ]
    }
    response = client.put("/ui/api/config", json=payload)
    assert response.status_code == 422


def test_resolve_route(client: TestClient):
    response = client.get("/ui/api/config/resolve", params={"model": "anything"})
    assert response.status_code == 200
    body = response.json()
    assert body["endpoint"] == "default"
    assert body["wildcard"] is True


def test_list_and_get_traces(client: TestClient):
    response = client.get("/ui/api/traces")
    assert response.status_code == 200
    traces = response.json()
    assert len(traces) == 1
    trace_id = traces[0]["id"]

    detail = client.get(f"/ui/api/traces/{trace_id}")
    assert detail.status_code == 200
    assert detail.json()["summary"]["model"] == "gpt-5.1"

    missing = client.get("/ui/api/traces/nope_20260101_000000_000000")
    assert missing.status_code == 404


def test_stats(client: TestClient):
    response = client.get("/ui/api/stats")
    assert response.status_code == 200
    body = response.json()
    assert "total" in body
    assert len(body["requests_per_hour"]) == 24


@pytest.fixture
def indexed_client(client: TestClient) -> TestClient:
    """A client whose background indexer has run — exercises the SQLite path."""
    with client as started:
        service = started.app.state.trace_index_service
        assert service.ready.wait(timeout=10)
        yield started


def test_revision_tokens_are_stable_until_something_changes(indexed_client: TestClient):
    first = indexed_client.get("/ui/api/revision")
    assert first.status_code == 200
    body = first.json()
    assert set(body) == {"traces", "ledger", "indexing"}
    assert body["indexing"] is False

    # Reading data must not move any token; that is what makes polling cheap.
    indexed_client.get("/ui/api/traces")
    indexed_client.get("/ui/api/stats")

    assert indexed_client.get("/ui/api/revision").json() == body


def test_revision_traces_token_moves_when_a_trace_is_indexed(indexed_client: TestClient, write_trace, tmp_path: Path):
    before = indexed_client.get("/ui/api/revision").json()["traces"]

    write_trace(tmp_path / "traces", "sk-test_20260706_120000_000001")
    service = indexed_client.app.state.trace_index_service
    service._index_paths(service._scan(hot_only=False))

    assert indexed_client.get("/ui/api/revision").json()["traces"] != before


def test_index_status_reports_a_completed_build(indexed_client: TestClient):
    body = indexed_client.get("/ui/api/index/status").json()

    assert body["ready"] is True
    assert body["building"] is False
    assert body["indexed"] == 1
    assert body["last_error"] is None


def test_traces_are_served_from_the_index(indexed_client: TestClient):
    traces = indexed_client.get("/ui/api/traces").json()

    assert [t["id"] for t in traces] == ["sk-test_20260706_111832_127166"]
    assert traces[0]["model"] == "gpt-5.1"


def test_rebuild_reindexes_and_keeps_traces_available(indexed_client: TestClient):
    response = indexed_client.post("/ui/api/index/rebuild")
    assert response.status_code == 200
    # The rebuild is asynchronous, so the index reports itself as not ready.
    assert response.json()["ready"] is False

    # Listing keeps working off disk while the rebuild runs.
    assert len(indexed_client.get("/ui/api/traces").json()) == 1

    service = indexed_client.app.state.trace_index_service
    assert service.ready.wait(timeout=10)
    assert indexed_client.get("/ui/api/index/status").json()["indexed"] == 1
    assert len(indexed_client.get("/ui/api/traces").json()) == 1

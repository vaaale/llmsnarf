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

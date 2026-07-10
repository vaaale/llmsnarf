import asyncio
import json
import logging

import httpx

from openaiproxy.models.models import EndpointConfig
from openaiproxy.services.model_tracker_service import ModelTrackerService


def _endpoint(max_models: int) -> EndpointConfig:
    return EndpointConfig(
        name="default",
        base_url="http://upstream.test/v1",
        api_key="sk-test",
        models=["*"],
        aliases={},
        log=False,
        substitute_role={},
        max_models=max_models,
    )


def _make_tracker(requests: list, loaded_models: list[str], monkeypatch) -> ModelTrackerService:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/models":
            data = {
                "data": [
                    {"id": "model-a", "status": {"value": "loaded" if "model-a" in loaded_models else "unloaded"}},
                    {"id": "model-b", "status": {"value": "loaded" if "model-b" in loaded_models else "unloaded"}},
                    {"id": "model-c", "status": {"value": "unloaded"}},
                ]
            }
            return httpx.Response(200, json=data)
        if request.url.path == "/models/unload":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    def patched_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)
    return ModelTrackerService(logger=logging.getLogger("test"))


def test_disabled_when_max_models_below_one(monkeypatch):
    requests: list = []
    tracker = _make_tracker(requests, ["model-a"], monkeypatch)

    asyncio.run(tracker.ensure_capacity(_endpoint(max_models=0), "model-b"))

    assert requests == []


def test_initializes_loaded_models_and_skips_unload_when_under_limit(monkeypatch):
    requests: list = []
    tracker = _make_tracker(requests, ["model-a"], monkeypatch)

    asyncio.run(tracker.ensure_capacity(_endpoint(max_models=2), "model-b"))

    paths = [r.url.path for r in requests]
    assert paths == ["/v1/models"]
    assert requests[0].headers["authorization"] == "Bearer sk-test"


def test_already_loaded_model_triggers_no_unload(monkeypatch):
    requests: list = []
    tracker = _make_tracker(requests, ["model-a"], monkeypatch)

    asyncio.run(tracker.ensure_capacity(_endpoint(max_models=1), "model-a"))

    assert [r.url.path for r in requests] == ["/v1/models"]


def test_unloads_least_recently_used_model_when_over_limit(monkeypatch):
    requests: list = []
    tracker = _make_tracker(requests, ["model-a", "model-b"], monkeypatch)
    endpoint = _endpoint(max_models=2)

    async def scenario():
        # touch model-b so model-a becomes least recently used
        await tracker.ensure_capacity(endpoint, "model-b")
        await tracker.ensure_capacity(endpoint, "model-new")

    asyncio.run(scenario())

    unloads = [r for r in requests if r.url.path == "/models/unload"]
    assert len(unloads) == 1
    assert json.loads(unloads[0].content) == {"model": "model-a"}


def test_unloads_multiple_models_when_limit_reduced(monkeypatch):
    requests: list = []
    tracker = _make_tracker(requests, ["model-a", "model-b"], monkeypatch)

    asyncio.run(tracker.ensure_capacity(_endpoint(max_models=1), "model-new"))

    unloaded = [json.loads(r.content)["model"] for r in requests if r.url.path == "/models/unload"]
    assert sorted(unloaded) == ["model-a", "model-b"]

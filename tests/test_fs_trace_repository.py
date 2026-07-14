import json
from pathlib import Path

from openaiproxy.filesystem.fs_trace_repository import FSTraceRepository


def _write_trace(
    trace_dir: Path,
    trace_id: str,
    model: str = "gpt-5.1",
    status_code: int = 200,
    stream: bool = False,
    error: str | None = None,
    endpoint: str = "/chat/completions",
) -> None:
    request = {
        "timestamp": "2026-07-06T11:18:32.127209",
        "endpoint": endpoint,
        "headers": {"authorization": "Bearer sk-test"},
        "payload": {
            "model": model,
            "stream": stream,
            "messages": [{"role": "user", "content": "hi"}],
        },
    }
    response = {
        "timestamp": "2026-07-06T11:18:33.127209",
        "status_code": status_code,
        "headers": {},
    }
    if error:
        response = {"timestamp": "2026-07-06T11:18:33.127209", "error": error}
    if stream:
        response["chunks"] = ['data: {"choices":[{"delta":{"content":"hello"}}]}\n\n']
    else:
        response["body"] = {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}

    (trace_dir / f"{trace_id}_request.json").write_text(json.dumps(request))
    (trace_dir / f"{trace_id}_response.json").write_text(json.dumps(response))


def test_list_traces_sorted_and_parsed(tmp_path: Path):
    _write_trace(tmp_path, "sk-test_20260706_111832_127166")
    _write_trace(tmp_path, "sk-test_20260706_111933_000001", model="qwen3")

    repository = FSTraceRepository(tmp_path)
    traces = repository.list_traces()

    assert len(traces) == 2
    assert traces[0].id == "sk-test_20260706_111933_000001"
    assert traces[0].model == "qwen3"
    assert traces[1].api_key == "sk-test"
    assert traces[1].status_code == 200
    assert traces[1].message_count == 1
    assert traces[1].duration_ms == 1000.0


def test_list_traces_filters(tmp_path: Path):
    _write_trace(tmp_path, "sk-a_20260706_111832_000001", model="gpt-5.1")
    _write_trace(tmp_path, "sk-b_20260706_111832_000002", model="qwen3", status_code=500)
    _write_trace(tmp_path, "sk-b_20260706_111832_000003", model="qwen3", error="boom")

    repository = FSTraceRepository(tmp_path)

    assert len(repository.list_traces(model="qwen3")) == 2
    assert len(repository.list_traces(api_key="sk-a")) == 1
    assert len(repository.list_traces(status="5")) == 1
    assert len(repository.list_traces(status="error")) == 2
    assert len(repository.list_traces(query="gpt")) == 1


def test_list_traces_excludes_models_endpoint(tmp_path: Path):
    _write_trace(tmp_path, "sk-a_20260706_111832_000001")
    _write_trace(tmp_path, "sk-a_20260706_111832_000002", endpoint="/models")

    repository = FSTraceRepository(tmp_path)
    traces = repository.list_traces()

    assert len(traces) == 1
    assert traces[0].id == "sk-a_20260706_111832_000001"


def test_get_trace_detail(tmp_path: Path):
    _write_trace(tmp_path, "sk-test_20260706_111832_127166", stream=True)

    repository = FSTraceRepository(tmp_path)
    detail = repository.get_trace("sk-test_20260706_111832_127166")

    assert detail is not None
    assert detail.summary.stream is True
    assert len(detail.response_chunks) == 1
    assert detail.request_payload["model"] == "gpt-5.1"


def test_get_trace_rejects_path_traversal(tmp_path: Path):
    repository = FSTraceRepository(tmp_path)
    assert repository.get_trace("../etc/passwd") is None


def test_get_missing_trace_returns_none(tmp_path: Path):
    repository = FSTraceRepository(tmp_path)
    assert repository.get_trace("nope_20260101_000000_000000") is None

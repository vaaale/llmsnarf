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


def test_since_prefilters_on_the_filename(tmp_path: Path):
    """The since cut must come off the filename, so old traces are never opened."""
    _write_trace(tmp_path, "sk-a_20260706_111832_000001")
    _write_trace(tmp_path, "sk-a_20260709_111832_000002")

    repository = FSTraceRepository(tmp_path)
    paths = repository.iter_request_files(since="2026-07-08T00:00:00")

    assert [p.name for p in paths] == ["sk-a_20260709_111832_000002_request.json"]


def test_since_keeps_traces_written_in_the_same_second_as_the_cutoff(tmp_path: Path):
    # The filename is stamped microseconds before the body timestamp, so the
    # prefilter has to be slightly generous or it would drop a matching trace.
    _write_trace(tmp_path, "sk-a_20260706_111832_000001")

    repository = FSTraceRepository(tmp_path)

    assert len(repository.iter_request_files(since="2026-07-06T11:18:32.127209")) == 1


def test_traces_with_unparseable_ids_are_still_listed(tmp_path: Path):
    _write_trace(tmp_path, "not-a-trace-id")

    repository = FSTraceRepository(tmp_path)
    traces = repository.list_traces()

    assert len(traces) == 1
    assert traces[0].api_key == "unknown"


def test_day_shard_is_not_mistaken_for_a_correlation_id(tmp_path: Path):
    shard = tmp_path / "completion" / "2026-07-06"
    shard.mkdir(parents=True)
    # A legacy trace (no correlation_id key) sitting inside a day shard.
    (shard / "sk-a_20260706_111832_000001_request.json").write_text(
        json.dumps({
            "timestamp": "2026-07-06T11:18:32.127209",
            "endpoint": "/chat/completions",
            "headers": {},
            "payload": {"model": "gpt-5.1", "messages": []},
        })
    )

    repository = FSTraceRepository(tmp_path)

    assert repository.list_traces()[0].correlation_id is None


def test_correlation_id_still_inferred_from_a_thread_directory(tmp_path: Path):
    thread = tmp_path / "completion" / "2026-07-06" / "thread-1"
    thread.mkdir(parents=True)
    (thread / "sk-a_20260706_111832_000001_request.json").write_text(
        json.dumps({
            "timestamp": "2026-07-06T11:18:32.127209",
            "endpoint": "/chat/completions",
            "headers": {},
            "payload": {"model": "gpt-5.1", "messages": []},
        })
    )

    repository = FSTraceRepository(tmp_path)

    assert repository.list_traces()[0].correlation_id == "thread-1"


def test_get_trace_rejects_path_traversal(tmp_path: Path):
    repository = FSTraceRepository(tmp_path)
    assert repository.get_trace("../etc/passwd") is None


def test_get_missing_trace_returns_none(tmp_path: Path):
    repository = FSTraceRepository(tmp_path)
    assert repository.get_trace("nope_20260101_000000_000000") is None

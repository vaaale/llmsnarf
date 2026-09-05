import logging
import threading
from pathlib import Path

import pytest

from openaiproxy.filesystem.fs_trace_repository import FSTraceRepository
from openaiproxy.filesystem.indexed_trace_repository import IndexedTraceRepository
from openaiproxy.filesystem.sqlite_trace_index import SqliteTraceIndex
from openaiproxy.services.trace_index_service import TraceIndexService


@pytest.fixture
def parts(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    files = FSTraceRepository(trace_dir)
    index = SqliteTraceIndex(trace_dir / "trace_index.db")
    indexer = TraceIndexService(index, files, logging.getLogger("test"))
    ready = threading.Event()
    return IndexedTraceRepository(index, files, ready), indexer, index, files, trace_dir, ready


def test_falls_back_to_disk_until_the_index_is_ready(parts, write_trace):
    repository, indexer, index, _, trace_dir, ready = parts
    write_trace(trace_dir / "completion" / "2026-07-06", "sk-a_20260706_111832_000001")

    # Nothing indexed yet, but the trace must still be listed.
    assert index.count() == 0
    assert [t.id for t in repository.list_traces()] == ["sk-a_20260706_111832_000001"]

    indexer._catch_up()
    ready.set()

    assert [t.id for t in repository.list_traces()] == ["sk-a_20260706_111832_000001"]


def test_list_uses_the_index_once_ready(parts, write_trace):
    repository, indexer, index, _, trace_dir, ready = parts
    write_trace(trace_dir / "completion" / "2026-07-06", "sk-a_20260706_111832_000001")
    indexer._catch_up()
    ready.set()

    # Removing the files proves the listing no longer touches disk.
    for path in trace_dir.rglob("*.json"):
        path.unlink()

    assert [t.id for t in repository.list_traces()] == ["sk-a_20260706_111832_000001"]


def test_get_trace_reads_the_body_from_disk(parts, write_trace):
    repository, indexer, _, _, trace_dir, ready = parts
    write_trace(trace_dir / "completion" / "2026-07-06", "sk-a_20260706_111832_000001", model="qwen3")
    indexer._catch_up()
    ready.set()

    detail = repository.get_trace("sk-a_20260706_111832_000001")

    assert detail is not None
    assert detail.request_payload["model"] == "qwen3"
    assert detail.response_body["usage"]["prompt_tokens"] == 7


def test_get_trace_attaches_children_via_the_index(parts, write_trace):
    repository, indexer, _, _, trace_dir, ready = parts
    directory = trace_dir / "responses" / "2026-07-06"
    write_trace(directory, "sk-a_20260706_111832_000001", endpoint="/responses")
    write_trace(
        directory,
        "sk-a_20260706_111833_000002",
        endpoint="/responses/map-reduce/chunk",
        parent_trace_id="sk-a_20260706_111832_000001",
    )
    write_trace(directory, "sk-a_20260706_111834_000003", endpoint="/responses")
    indexer._catch_up()
    ready.set()

    detail = repository.get_trace("sk-a_20260706_111832_000001")

    assert detail is not None
    assert [c.summary.id for c in detail.children] == ["sk-a_20260706_111833_000002"]
    # Children are leaves; they must not recurse.
    assert detail.children[0].children == []


def test_get_trace_falls_back_when_the_index_is_stale(parts, write_trace):
    repository, indexer, index, _, trace_dir, ready = parts
    write_trace(trace_dir / "completion" / "2026-07-06", "sk-a_20260706_111832_000001")
    indexer._catch_up()
    ready.set()

    # The trace moves on disk, so the indexed path no longer resolves.
    shard = trace_dir / "completion" / "2026-07-06"
    for path in list(shard.glob("*.json")):
        path.rename(trace_dir / path.name)
    assert not index.get("sk-a_20260706_111832_000001").path.exists()

    detail = repository.get_trace("sk-a_20260706_111832_000001")

    assert detail is not None
    assert detail.summary.id == "sk-a_20260706_111832_000001"


def test_get_trace_rejects_path_traversal(parts, write_trace):
    repository = parts[0]

    assert repository.get_trace("../etc/passwd") is None
    assert repository.get_trace("a/b") is None


def test_get_missing_trace_returns_none(parts, write_trace):
    repository = parts[0]

    assert repository.get_trace("nope_20260101_000000_000000") is None

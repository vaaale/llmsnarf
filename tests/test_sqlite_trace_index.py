from pathlib import Path

import pytest

from openaiproxy.filesystem.sqlite_trace_index import SqliteTraceIndex
from openaiproxy.interface.trace_index import IndexedTrace
from openaiproxy.models.trace_models import TraceSummary, parse_trace_id


def _entry(
    trace_id: str,
    *,
    model: str | None = "gpt-5.1",
    endpoint: str = "/chat/completions",
    status_code: int | None = 200,
    error: str | None = None,
    correlation_id: str | None = None,
    parent_trace_id: str | None = None,
    timestamp: str | None = None,
    complete: bool = True,
    path: str = "/traces/x_request.json",
) -> IndexedTrace:
    api_key, sort_key = parse_trace_id(trace_id)
    summary = TraceSummary(
        id=trace_id,
        timestamp=timestamp if timestamp is not None else (sort_key or ""),
        api_key=api_key,
        endpoint=endpoint,
        model=model,
        stream=False,
        status_code=status_code,
        duration_ms=12.5,
        message_count=1,
        error=error,
        correlation_id=correlation_id,
        parent_trace_id=parent_trace_id,
        provider="local",
        input_tokens=10,
        output_tokens=20,
    )
    return IndexedTrace(summary=summary, path=Path(path), complete=complete, sort_key=sort_key or "")


@pytest.fixture
def index(tmp_path: Path) -> SqliteTraceIndex:
    return SqliteTraceIndex(tmp_path / "trace_index.db")


def test_query_returns_newest_first(index: SqliteTraceIndex):
    index.upsert_many([
        _entry("sk-a_20260706_111832_000001"),
        _entry("sk-a_20260706_111933_000002"),
    ])

    traces = index.query()

    assert [t.id for t in traces] == [
        "sk-a_20260706_111933_000002",
        "sk-a_20260706_111832_000001",
    ]


def test_round_trip_preserves_summary_fields(index: SqliteTraceIndex):
    index.upsert_many([_entry("sk-a_20260706_111832_000001", correlation_id="thread-1")])

    trace = index.query()[0]

    assert trace.api_key == "sk-a"
    assert trace.model == "gpt-5.1"
    assert trace.stream is False
    assert trace.status_code == 200
    assert trace.duration_ms == 12.5
    assert trace.correlation_id == "thread-1"
    assert trace.provider == "local"
    assert (trace.input_tokens, trace.output_tokens) == (10, 20)


def test_filters(index: SqliteTraceIndex):
    index.upsert_many([
        _entry("sk-a_20260706_111832_000001", model="gpt-5.1"),
        _entry("sk-b_20260706_111832_000002", model="qwen3", status_code=500),
        _entry("sk-b_20260706_111832_000003", model="qwen3", status_code=None, error="boom"),
        _entry("sk-b_20260706_111832_000004", model="qwen3", correlation_id="thread-1"),
    ])

    assert len(index.query(model="qwen3")) == 3
    assert len(index.query(api_key="sk-a")) == 1
    assert len(index.query(status="5")) == 1
    assert len(index.query(status="2")) == 2
    # "error" covers both an explicit error and any 4xx/5xx status.
    assert len(index.query(status="error")) == 2
    assert len(index.query(correlation_id="thread-1")) == 1
    assert len(index.query(query="gpt")) == 1
    assert len(index.query(query="CHAT")) == 4


def test_query_wildcards_are_literal(index: SqliteTraceIndex):
    index.upsert_many([
        _entry("sk-a_20260706_111832_000001", model="gpt-5.1"),
        _entry("sk-a_20260706_111832_000002", model="gpt%5"),
    ])

    assert [t.model for t in index.query(query="gpt%")] == ["gpt%5"]


def test_since_filter_keeps_traces_without_a_timestamp(index: SqliteTraceIndex):
    index.upsert_many([
        _entry("sk-a_20260706_111832_000001", timestamp="2026-07-06T11:18:32.000001"),
        _entry("sk-a_20260707_111832_000002", timestamp="2026-07-07T11:18:32.000002"),
        _entry("sk-a_20260708_111832_000003", timestamp=""),
    ])

    ids = {t.id for t in index.query(since="2026-07-07T00:00:00")}

    assert ids == {"sk-a_20260707_111832_000002", "sk-a_20260708_111832_000003"}


def test_models_endpoint_is_indexed_but_never_listed(index: SqliteTraceIndex):
    index.upsert_many([
        _entry("sk-a_20260706_111832_000001"),
        _entry("sk-a_20260706_111832_000002", endpoint="/models"),
    ])

    assert index.count() == 2
    assert [t.id for t in index.query()] == ["sk-a_20260706_111832_000001"]
    assert "sk-a_20260706_111832_000002" in index.known_trace_ids()


def test_pagination(index: SqliteTraceIndex):
    index.upsert_many([_entry(f"sk-a_20260706_111832_00000{i}") for i in range(1, 6)])

    first = index.query(limit=2)
    second = index.query(limit=2, offset=2)

    assert len(first) == len(second) == 2
    assert {t.id for t in first}.isdisjoint({t.id for t in second})


def test_upsert_replaces_existing_row(index: SqliteTraceIndex):
    index.upsert_many([_entry("sk-a_20260706_111832_000001", status_code=None, complete=False)])
    index.upsert_many([_entry("sk-a_20260706_111832_000001", status_code=200, complete=True)])

    assert index.count() == 1
    assert index.query()[0].status_code == 200
    assert index.incomplete("2000-01-01T00:00:00") == []


def test_incomplete_is_bounded_by_sort_key(index: SqliteTraceIndex):
    index.upsert_many([
        _entry("sk-a_20260706_111832_000001", complete=False, path="/traces/old_request.json"),
        _entry("sk-a_20260709_111832_000002", complete=False, path="/traces/new_request.json"),
    ])

    pending = index.incomplete("2026-07-08T00:00:00")

    assert [str(path) for _, path in pending] == ["/traces/new_request.json"]


def test_children_of_returns_sub_calls_oldest_first(index: SqliteTraceIndex):
    index.upsert_many([
        _entry("sk-a_20260706_111832_000001"),
        _entry("sk-a_20260706_111833_000002", parent_trace_id="sk-a_20260706_111832_000001"),
        _entry("sk-a_20260706_111834_000003", parent_trace_id="sk-a_20260706_111832_000001"),
    ])

    children = index.children_of("sk-a_20260706_111832_000001")

    assert [c.summary.id for c in children] == [
        "sk-a_20260706_111833_000002",
        "sk-a_20260706_111834_000003",
    ]
    assert index.children_of("sk-a_20260706_111833_000002") == []


def test_get_returns_path_and_completeness(index: SqliteTraceIndex):
    index.upsert_many([
        _entry("sk-a_20260706_111832_000001", path="/traces/completion/a_request.json", complete=False)
    ])

    entry = index.get("sk-a_20260706_111832_000001")

    assert entry is not None
    assert entry.path == Path("/traces/completion/a_request.json")
    assert entry.complete is False
    assert index.get("missing") is None


def test_revision_advances_on_write_only(index: SqliteTraceIndex):
    start = index.revision()

    index.upsert_many([_entry("sk-a_20260706_111832_000001")])
    after_write = index.revision()

    index.upsert_many([])
    index.query()

    assert after_write > start
    assert index.revision() == after_write

    index.clear()
    assert index.revision() > after_write
    assert index.count() == 0


def test_schema_version_mismatch_discards_the_index(tmp_path: Path):
    db_path = tmp_path / "trace_index.db"
    index = SqliteTraceIndex(db_path)
    index.upsert_many([_entry("sk-a_20260706_111832_000001")])
    index.close()

    import sqlite3

    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'")
        connection.commit()

    assert SqliteTraceIndex(db_path).count() == 0

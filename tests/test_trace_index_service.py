import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from openaiproxy.filesystem.fs_trace_repository import FSTraceRepository
from openaiproxy.filesystem.indexed_trace_repository import IndexedTraceRepository
from openaiproxy.filesystem.sqlite_trace_index import SqliteTraceIndex
from openaiproxy.services.trace_index_service import TraceIndexService


@pytest.fixture
def service(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    files = FSTraceRepository(trace_dir)
    index = SqliteTraceIndex(trace_dir / "trace_index.db")
    indexer = TraceIndexService(index, files, logging.getLogger("test"))
    return indexer, index, files, trace_dir


def test_catch_up_indexes_every_trace_on_disk(service, write_trace):
    indexer, index, _, trace_dir = service
    write_trace(trace_dir / "completion" / "2026-07-06", "sk-a_20260706_111832_000001")
    write_trace(trace_dir / "responses" / "2026-07-06" / "thread-1", "sk-a_20260706_111833_000002")

    indexer._catch_up()

    assert index.count() == 2


def test_index_matches_the_filesystem_repository(service, write_trace):
    """The index is a cache of what a disk scan would produce; drift is a bug."""
    indexer, index, files, trace_dir = service
    write_trace(trace_dir / "completion" / "2026-07-06", "sk-a_20260706_111832_000001", model="qwen3")
    write_trace(
        trace_dir / "completion" / "2026-07-06" / "thread-1",
        "sk-b_20260706_111833_000002",
        correlation_id="thread-1",
        status_code=500,
    )
    write_trace(trace_dir / "completion" / "2026-07-06", "sk-a_20260706_111834_000003", endpoint="/models")

    indexer._catch_up()

    for filters in [
        {},
        {"model": "qwen3"},
        {"api_key": "sk-b"},
        {"status": "error"},
        {"status": "2"},
        {"correlation_id": "thread-1"},
        {"query": "qwen"},
        {"limit": 1},
    ]:
        assert index.query(**filters) == files.list_traces(**filters), filters


def test_new_traces_are_picked_up_incrementally(service, write_trace):
    indexer, index, _, trace_dir = service
    today = date.today().isoformat()
    write_trace(trace_dir / "completion" / today, "sk-a_20260706_111832_000001")
    indexer._catch_up()
    assert index.count() == 1

    write_trace(trace_dir / "completion" / today, "sk-a_20260706_111833_000002")
    indexer._index_paths(indexer._scan(hot_only=True))

    assert index.count() == 2


def test_hot_scan_ignores_shards_outside_the_recent_window(service, write_trace):
    indexer, index, _, trace_dir = service
    write_trace(trace_dir / "completion" / "2020-01-01", "sk-a_20200101_111832_000001")

    assert indexer._scan(hot_only=True) == []
    assert len(indexer._scan(hot_only=False)) == 1


def test_trace_without_a_response_is_completed_later(service, write_trace):
    indexer, index, _, trace_dir = service
    today = date.today().isoformat()
    directory = trace_dir / "completion" / today
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    trace_id = f"sk-a_{stamp}"
    write_trace(directory, trace_id, with_response=False)

    indexer._catch_up()

    assert index.query()[0].status_code is None
    assert len(index.incomplete("2000-01-01T00:00:00")) == 1

    (directory / f"{trace_id}_response.json").write_text(json.dumps({
        "timestamp": "2026-07-06T11:18:33.127209", "status_code": 201, "headers": {}, "body": {},
    }))
    indexer._refresh_incomplete()

    assert index.query()[0].status_code == 201
    assert index.incomplete("2000-01-01T00:00:00") == []


def test_response_stragglers_are_abandoned_after_the_grace_period(service, write_trace):
    indexer, index, _, trace_dir = service
    # An old trace whose response never arrived must not be re-checked forever.
    write_trace(trace_dir / "completion" / "2020-01-01", "sk-a_20200101_111832_000001", with_response=False)
    indexer._catch_up()

    cutoff = (datetime.now() - timedelta(minutes=30)).isoformat()

    assert index.incomplete("2000-01-01T00:00:00") != []
    assert index.incomplete(cutoff) == []


def test_rebuild_drops_stale_rows_and_reindexes(service, write_trace):
    indexer, index, _, trace_dir = service
    directory = trace_dir / "completion" / "2026-07-06"
    write_trace(directory, "sk-a_20260706_111832_000001")
    write_trace(directory, "sk-a_20260706_111833_000002")
    indexer._catch_up()
    assert index.count() == 2

    # A trace deleted behind the proxy's back leaves the index stale; only a
    # rebuild can notice, which is why the button exists.
    (directory / "sk-a_20260706_111833_000002_request.json").unlink()
    (directory / "sk-a_20260706_111833_000002_response.json").unlink()

    indexer._rebuild()

    assert index.count() == 1
    assert indexer.ready.is_set()


def test_rebuild_leaves_listings_on_disk_until_it_finishes(service, write_trace):
    indexer, index, files, trace_dir = service
    write_trace(trace_dir / "completion" / "2026-07-06", "sk-a_20260706_111832_000001")
    indexer._catch_up()
    indexer.ready.set()
    repository = IndexedTraceRepository(index, files, indexer.ready)

    indexer.request_rebuild()

    # ready drops immediately so a half-rebuilt index is never served.
    assert not indexer.ready.is_set()
    assert len(repository.list_traces()) == 1


def test_failed_rebuild_does_not_mark_the_index_ready(service, write_trace):
    indexer, index, _, _ = service

    def explode(*args, **kwargs):
        raise OSError("disk gone")

    indexer._catch_up = explode
    indexer.ready.set()

    with pytest.raises(OSError):
        indexer._rebuild()

    assert not indexer.ready.is_set()


def test_start_and_stop_are_idempotent(service, write_trace):
    indexer, _, _, trace_dir = service
    write_trace(trace_dir / "completion" / "2026-07-06", "sk-a_20260706_111832_000001")

    indexer.start()
    indexer.start()
    assert indexer.ready.wait(timeout=5)

    indexer.stop()
    indexer.stop()

    assert indexer.status().indexed == 1

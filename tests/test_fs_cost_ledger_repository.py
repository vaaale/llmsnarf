from pathlib import Path

from openaiproxy.filesystem.fs_cost_ledger_repository import FSCostLedgerRepository
from openaiproxy.models.cost_models import CostUsageEntry


def _entry(entry_id: str, timestamp: str, model: str = "gpt-5.1", provider: str = "openai") -> CostUsageEntry:
    return CostUsageEntry(
        id=entry_id,
        trace_id=f"trace-{entry_id}",
        timestamp=timestamp,
        model=model,
        provider=provider,
        endpoint="/chat/completions",
        input_tokens=10,
        output_tokens=5,
    )


def test_append_and_list_entries(tmp_path: Path):
    repo = FSCostLedgerRepository(trace_dir=tmp_path)
    repo.append(_entry("1", "2026-01-01T00:00:00"))
    repo.append(_entry("2", "2026-01-02T00:00:00"))

    entries = repo.list_entries()
    assert len(entries) == 2
    assert {e.id for e in entries} == {"1", "2"}
    assert entries[0].input_tokens == 10
    assert entries[0].output_tokens == 5


def test_list_entries_filters_by_since(tmp_path: Path):
    repo = FSCostLedgerRepository(trace_dir=tmp_path)
    repo.append(_entry("1", "2026-01-01T00:00:00"))
    repo.append(_entry("2", "2026-01-05T00:00:00"))

    entries = repo.list_entries(since="2026-01-03T00:00:00")
    assert len(entries) == 1
    assert entries[0].id == "2"


def test_list_entries_handles_missing_file(tmp_path: Path):
    repo = FSCostLedgerRepository(trace_dir=tmp_path)
    assert repo.list_entries() == []


def test_append_persists_none_tokens(tmp_path: Path):
    repo = FSCostLedgerRepository(trace_dir=tmp_path)
    repo.append(
        CostUsageEntry(
            id="err",
            trace_id="trace-err",
            timestamp="2026-01-01T00:00:00",
            model="gpt-5.1",
            provider="openai",
            endpoint="/chat/completions",
            input_tokens=None,
            output_tokens=None,
        )
    )
    entries = repo.list_entries()
    assert len(entries) == 1
    assert entries[0].input_tokens is None
    assert entries[0].output_tokens is None

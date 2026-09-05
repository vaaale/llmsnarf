import logging

import pytest

from openaiproxy.interface.cost_ledger_repository import CostLedgerRepository
from openaiproxy.models.cost_models import CostUsageEntry
from openaiproxy.services.cost_recorder_service import CostRecorderService


class FakeCostLedgerRepository(CostLedgerRepository):
    def __init__(self):
        self.entries: list[CostUsageEntry] = []

    def append(self, entry: CostUsageEntry) -> None:
        self.entries.append(entry)

    def list_entries(self, since: str | None = None) -> list[CostUsageEntry]:
        return list(self.entries)


@pytest.mark.anyio
async def test_record_usage_extracts_tokens_and_appends():
    repo = FakeCostLedgerRepository()
    recorder = CostRecorderService(cost_ledger_repository=repo, logger=logging.getLogger("test"))

    await recorder.record_usage(
        trace_id="trace-1",
        model="gpt-5.1",
        provider="openai",
        endpoint="/chat/completions",
        response_body={"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        response_chunks=[],
    )

    assert len(repo.entries) == 1
    entry = repo.entries[0]
    assert entry.trace_id == "trace-1"
    assert entry.model == "gpt-5.1"
    assert entry.provider == "openai"
    assert entry.input_tokens == 10
    assert entry.output_tokens == 5


@pytest.mark.anyio
async def test_record_usage_appends_none_tokens_when_unextractable():
    repo = FakeCostLedgerRepository()
    recorder = CostRecorderService(cost_ledger_repository=repo, logger=logging.getLogger("test"))

    await recorder.record_usage(
        trace_id="trace-2",
        model="gpt-5.1",
        provider="openai",
        endpoint="/chat/completions",
        response_body={"error": "boom"},
        response_chunks=[],
    )

    assert len(repo.entries) == 1
    entry = repo.entries[0]
    assert entry.input_tokens is None
    assert entry.output_tokens is None


@pytest.mark.anyio
async def test_record_usage_swallows_repository_errors():
    class BoomRepository(CostLedgerRepository):
        def append(self, entry: CostUsageEntry) -> None:
            raise RuntimeError("disk full")

        def list_entries(self, since: str | None = None) -> list[CostUsageEntry]:
            return []

    recorder = CostRecorderService(cost_ledger_repository=BoomRepository(), logger=logging.getLogger("test"))
    await recorder.record_usage(
        trace_id="trace-3",
        model="gpt-5.1",
        provider="openai",
        endpoint="/chat/completions",
        response_body={"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        response_chunks=[],
    )


@pytest.fixture
def anyio_backend():
    return "asyncio"

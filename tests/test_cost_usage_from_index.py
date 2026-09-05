"""Costs are derived from indexed traces, so they must cover the whole corpus.

These exercise the real SQLite index rather than a stub: the bug they guard
against is costs reflecting only what a live recorder happened to observe.
"""

import logging
from pathlib import Path

import pytest

from openaiproxy.filesystem.fs_trace_repository import FSTraceRepository
from openaiproxy.filesystem.sqlite_trace_index import SqliteTraceIndex
from openaiproxy.interface.repository import LLMProxyConfigRepository
from openaiproxy.models.models import LLMProxyConfig, ModelPricing
from openaiproxy.services.config_service import ConfigService
from openaiproxy.services.cost_service import CostService
from openaiproxy.services.trace_index_service import TraceIndexService


class FakeConfigRepository(LLMProxyConfigRepository):
    def __init__(self, config: LLMProxyConfig):
        self._config = config

    def load(self) -> LLMProxyConfig:
        return self._config

    def save(self, config: LLMProxyConfig) -> None:
        self._config = config


@pytest.fixture
def parts(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    files = FSTraceRepository(trace_dir)
    index = SqliteTraceIndex(trace_dir / "trace_index.db")
    indexer = TraceIndexService(index, files, logging.getLogger("test"))
    config = LLMProxyConfig(
        host=None, port=None, logs_dir=None, trace_dir=None, endpoints=[],
        pricing={"gpt-5.1": ModelPricing(price_input_per_million=2.0, price_output_per_million=8.0)},
    )
    costs = CostService(trace_index=index, config_service=ConfigService(FakeConfigRepository(config)))
    return costs, indexer, index, trace_dir


def test_costs_cover_every_trace_on_disk(parts, write_trace):
    costs, indexer, _, trace_dir = parts
    shard = trace_dir / "completion" / "2026-07-06"
    for i in range(1, 6):
        write_trace(shard, f"sk-a_2026070{i}_111832_00000{i}")

    indexer._catch_up()
    report = costs.get_costs()

    assert report.total_requests == 5
    assert report.priced_requests == 5
    assert report.total_input_tokens == 35   # 5 traces x 7 prompt tokens
    assert report.total_output_tokens == 15  # 5 traces x 3 completion tokens
    assert report.models[0].model == "gpt-5.1"
    assert report.models[0].providers == ["local"]


def test_pricing_is_applied_to_indexed_usage(parts, write_trace):
    costs, indexer, _, trace_dir = parts
    write_trace(trace_dir / "completion" / "2026-07-06", "sk-a_20260706_111832_000001")

    indexer._catch_up()
    report = costs.get_costs()

    # 7 in @ $2/M + 3 out @ $8/M
    assert report.total_cost == pytest.approx(7 / 1_000_000 * 2.0 + 3 / 1_000_000 * 8.0)


def test_traces_without_usage_count_as_unpriced(parts, write_trace):
    costs, indexer, _, trace_dir = parts
    shard = trace_dir / "completion" / "2026-07-06"
    write_trace(shard, "sk-a_20260706_111832_000001")
    write_trace(shard, "sk-a_20260706_111833_000002", with_response=False)

    indexer._catch_up()
    report = costs.get_costs()

    assert report.total_requests == 2
    assert report.priced_requests == 1


def test_models_endpoint_is_excluded_from_costs(parts, write_trace):
    costs, indexer, _, trace_dir = parts
    shard = trace_dir / "completion" / "2026-07-06"
    write_trace(shard, "sk-a_20260706_111832_000001")
    write_trace(shard, "sk-a_20260706_111833_000002", endpoint="/models")

    indexer._catch_up()

    assert costs.get_costs().total_requests == 1


def test_rebuild_reingests_every_trace_into_costs(parts, write_trace):
    costs, indexer, _, trace_dir = parts
    shard = trace_dir / "completion" / "2026-07-06"
    write_trace(shard, "sk-a_20260706_111832_000001")
    indexer._catch_up()
    assert costs.get_costs().total_requests == 1

    # Traces that appeared while the proxy was not watching.
    write_trace(shard, "sk-a_20260706_111833_000002")
    write_trace(shard, "sk-a_20260706_111834_000003")

    indexer._rebuild()

    assert costs.get_costs().total_requests == 3


def test_usage_entries_are_filtered_by_since(parts, write_trace):
    _, indexer, index, trace_dir = parts
    shard = trace_dir / "completion" / "2026-07-06"
    write_trace(shard, "sk-a_20260701_111832_000001", timestamp="2026-07-01T11:18:32.000001")
    write_trace(shard, "sk-a_20260709_111832_000002", timestamp="2026-07-09T11:18:32.000002")

    indexer._catch_up()
    entries = index.usage_entries(since="2026-07-05T00:00:00")

    assert [e.trace_id for e in entries] == ["sk-a_20260709_111832_000002"]
    assert entries[0].id == entries[0].trace_id
    assert entries[0].endpoint == "/chat/completions"

from openaiproxy.interface.cost_ledger_repository import CostLedgerRepository
from openaiproxy.interface.repository import LLMProxyConfigRepository
from openaiproxy.models.cost_models import CostUsageEntry
from openaiproxy.models.models import LLMProxyConfig, ModelPricing
from openaiproxy.services.config_service import ConfigService
from openaiproxy.services.cost_service import CostService


class FakeCostLedgerRepository(CostLedgerRepository):
    def __init__(self, entries: list[CostUsageEntry]):
        self._entries = entries

    def append(self, entry: CostUsageEntry) -> None:
        self._entries.append(entry)

    def list_entries(self, since: str | None = None) -> list[CostUsageEntry]:
        if since is None:
            return list(self._entries)
        return [e for e in self._entries if e.timestamp >= since]


class FakeConfigRepository(LLMProxyConfigRepository):
    def __init__(self, config: LLMProxyConfig):
        self._config = config

    def load(self) -> LLMProxyConfig:
        return self._config

    def save(self, config: LLMProxyConfig) -> None:
        self._config = config


def _make_config(pricing: dict[str, ModelPricing]) -> LLMProxyConfig:
    return LLMProxyConfig(
        host=None, port=None, logs_dir=None, trace_dir=None, endpoints=[], pricing=pricing
    )


def test_get_costs_aggregates_priced_and_unpriced_requests():
    entries = [
        CostUsageEntry(
            id="1", trace_id="t1", timestamp="2026-01-01T00:00:00", model="gpt-5.1",
            provider="openai", endpoint="/chat/completions", input_tokens=1_000_000, output_tokens=1_000_000,
        ),
        CostUsageEntry(
            id="2", trace_id="t2", timestamp="2026-01-01T00:00:01", model="gpt-5.1",
            provider="openai", endpoint="/chat/completions", input_tokens=None, output_tokens=None,
        ),
    ]
    config = _make_config({"gpt-5.1": ModelPricing(price_input_per_million=2.0, price_output_per_million=8.0)})
    cost_service = CostService(
        cost_ledger_repository=FakeCostLedgerRepository(entries),
        config_service=ConfigService(FakeConfigRepository(config)),
    )

    report = cost_service.get_costs()

    assert report.total_requests == 2
    assert report.priced_requests == 1
    assert report.total_cost == 10.0
    assert len(report.models) == 1
    model_cost = report.models[0]
    assert model_cost.model == "gpt-5.1"
    assert model_cost.request_count == 2
    assert model_cost.priced_request_count == 1
    assert model_cost.total_cost == 10.0
    assert model_cost.providers == ["openai"]


def test_get_costs_unknown_model_uses_zero_pricing():
    entries = [
        CostUsageEntry(
            id="1", trace_id="t1", timestamp="2026-01-01T00:00:00", model="mystery-model",
            provider=None, endpoint="/chat/completions", input_tokens=500, output_tokens=500,
        ),
    ]
    cost_service = CostService(
        cost_ledger_repository=FakeCostLedgerRepository(entries),
        config_service=ConfigService(FakeConfigRepository(_make_config({}))),
    )

    report = cost_service.get_costs()

    assert report.total_cost == 0.0
    assert report.models[0].model == "mystery-model"
    assert report.models[0].providers == []


def test_get_costs_filters_by_days():
    entries = [
        CostUsageEntry(
            id="old", trace_id="t1", timestamp="2000-01-01T00:00:00", model="gpt-5.1",
            provider="openai", endpoint="/chat/completions", input_tokens=100, output_tokens=100,
        ),
    ]
    cost_service = CostService(
        cost_ledger_repository=FakeCostLedgerRepository(entries),
        config_service=ConfigService(FakeConfigRepository(_make_config({}))),
    )

    report = cost_service.get_costs(days=7)

    assert report.total_requests == 0

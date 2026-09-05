from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from openaiproxy.interface.cost_ledger_repository import CostLedgerRepository
from openaiproxy.services.config_service import ConfigService


@dataclass(frozen=True)
class ModelCost:
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    request_count: int
    priced_request_count: int
    avg_cost_per_request: float
    price_input_per_million: float
    price_output_per_million: float
    providers: list[str]


@dataclass(frozen=True)
class CostReport:
    models: list[ModelCost]
    total_cost: float
    total_requests: int
    priced_requests: int
    avg_cost_per_request: float
    total_input_tokens: int
    total_output_tokens: int


class CostService:
    def __init__(self, cost_ledger_repository: CostLedgerRepository, config_service: ConfigService):
        self._cost_ledger_repository = cost_ledger_repository
        self._config_service = config_service

    def get_costs(self, days: int | None = None) -> CostReport:
        since = (datetime.now() - timedelta(days=days)).isoformat() if days else None
        entries = self._cost_ledger_repository.list_entries(since=since)

        config = self._config_service.get_config()
        pricing = config.pricing

        buckets: dict[str, dict] = {}
        for entry in entries:
            model = entry.model or "unknown"
            bucket = buckets.setdefault(
                model,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "request_count": 0,
                    "priced_request_count": 0,
                    "providers": set(),
                },
            )
            bucket["request_count"] += 1
            if entry.provider:
                bucket["providers"].add(entry.provider)
            has_usage = entry.input_tokens is not None or entry.output_tokens is not None
            if has_usage:
                bucket["priced_request_count"] += 1
                bucket["input_tokens"] += entry.input_tokens or 0
                bucket["output_tokens"] += entry.output_tokens or 0

        models: list[ModelCost] = []
        grand_total_cost = 0.0
        grand_requests = 0
        grand_priced_requests = 0
        grand_input_tokens = 0
        grand_output_tokens = 0

        for name, bucket in buckets.items():
            model_pricing = pricing.get(name)
            price_input = model_pricing.price_input_per_million if model_pricing else 0.0
            price_output = model_pricing.price_output_per_million if model_pricing else 0.0

            input_cost = bucket["input_tokens"] / 1_000_000 * price_input
            output_cost = bucket["output_tokens"] / 1_000_000 * price_output
            total_cost = input_cost + output_cost
            priced_request_count = bucket["priced_request_count"]
            avg_cost = total_cost / priced_request_count if priced_request_count else 0.0

            models.append(
                ModelCost(
                    model=name,
                    input_tokens=bucket["input_tokens"],
                    output_tokens=bucket["output_tokens"],
                    input_cost=input_cost,
                    output_cost=output_cost,
                    total_cost=total_cost,
                    request_count=bucket["request_count"],
                    priced_request_count=priced_request_count,
                    avg_cost_per_request=avg_cost,
                    price_input_per_million=price_input,
                    price_output_per_million=price_output,
                    providers=sorted(bucket["providers"]),
                )
            )
            grand_total_cost += total_cost
            grand_requests += bucket["request_count"]
            grand_priced_requests += priced_request_count
            grand_input_tokens += bucket["input_tokens"]
            grand_output_tokens += bucket["output_tokens"]

        models.sort(key=lambda m: m.total_cost, reverse=True)

        return CostReport(
            models=models,
            total_cost=grand_total_cost,
            total_requests=grand_requests,
            priced_requests=grand_priced_requests,
            avg_cost_per_request=(grand_total_cost / grand_priced_requests) if grand_priced_requests else 0.0,
            total_input_tokens=grand_input_tokens,
            total_output_tokens=grand_output_tokens,
        )

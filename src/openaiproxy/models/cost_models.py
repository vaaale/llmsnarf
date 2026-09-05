from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostUsageEntry:
    id: str
    trace_id: str
    timestamp: str
    model: str | None
    provider: str | None
    endpoint: str
    input_tokens: int | None
    output_tokens: int | None

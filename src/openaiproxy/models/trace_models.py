from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TraceSummary:
    id: str
    timestamp: str
    api_key: str
    endpoint: str
    model: str | None
    stream: bool
    status_code: int | None
    duration_ms: float | None
    message_count: int
    error: str | None


@dataclass(frozen=True)
class TraceDetail:
    summary: TraceSummary
    request_headers: dict[str, str] = field(default_factory=dict)
    request_payload: Any = None
    response_headers: dict[str, str] = field(default_factory=dict)
    response_body: Any = None
    response_chunks: list[str] = field(default_factory=list)

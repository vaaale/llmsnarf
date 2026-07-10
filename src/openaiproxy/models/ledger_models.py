from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    path: str = ""


@dataclass(frozen=True)
class LedgerEntry:
    id: str
    trace_id: str
    timestamp: str
    model: str | None
    endpoint: str
    issues: list[ValidationIssue] = field(default_factory=list)

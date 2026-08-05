from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_TRACE_CATEGORY_BY_ENDPOINT = {
    "/chat/completions": "completion",
    "/completions": "completion",
    "/responses": "responses",
    "/messages": "messages",
    "/embeddings": "embeddings",
}


def trace_category(endpoint_path: str) -> str:
    """Folder under trace_dir where traces for an inbound API are stored."""
    path = endpoint_path or ""
    for prefix, category in _TRACE_CATEGORY_BY_ENDPOINT.items():
        if path == prefix or path.startswith(prefix + "/"):
            return category
    cleaned = path.strip("/").replace("/", "_").replace("\\", "_")
    return cleaned or "other"


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
    correlation_id: str | None = None
    parent_trace_id: str | None = None


@dataclass(frozen=True)
class TraceDetail:
    summary: TraceSummary
    request_headers: dict[str, str] = field(default_factory=dict)
    request_payload: Any = None
    response_headers: dict[str, str] = field(default_factory=dict)
    response_body: Any = None
    response_chunks: list[str] = field(default_factory=list)
    children: list["TraceDetail"] = field(default_factory=list)

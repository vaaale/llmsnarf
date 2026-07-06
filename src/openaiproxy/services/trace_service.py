from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from openaiproxy.interface.trace_repository import TraceRepository
from openaiproxy.models.trace_models import TraceDetail, TraceSummary


def assemble_stream_content(chunks: list[str]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        for line in chunk.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                parts.append(content)
    return "".join(parts)


class TraceService:
    def __init__(self, trace_repository: TraceRepository):
        self._trace_repository = trace_repository

    def list_traces(
        self,
        limit: int = 100,
        offset: int = 0,
        model: str | None = None,
        api_key: str | None = None,
        status: str | None = None,
        since: str | None = None,
        query: str | None = None,
    ) -> list[TraceSummary]:
        return self._trace_repository.list_traces(
            limit=limit,
            offset=offset,
            model=model,
            api_key=api_key,
            status=status,
            since=since,
            query=query,
        )

    def get_trace(self, trace_id: str) -> tuple[TraceDetail, str] | None:
        detail = self._trace_repository.get_trace(trace_id)
        if detail is None:
            return None
        assembled = assemble_stream_content(detail.response_chunks) if detail.response_chunks else ""
        return detail, assembled

    def stats(self, hours: int = 24, sample_limit: int = 2000) -> dict[str, Any]:
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        summaries = self._trace_repository.list_traces(limit=sample_limit, since=since)

        total = len(summaries)
        errors = sum(
            1
            for s in summaries
            if s.error is not None or (s.status_code is not None and s.status_code >= 400)
        )
        durations = [s.duration_ms for s in summaries if s.duration_ms is not None]
        avg_duration_ms = sum(durations) / len(durations) if durations else None

        by_model = Counter(s.model or "unknown" for s in summaries)
        by_api_key = Counter(s.api_key for s in summaries)

        buckets = [0] * hours
        error_buckets = [0] * hours
        now = datetime.now()
        for s in summaries:
            ts = None
            try:
                ts = datetime.fromisoformat(s.timestamp)
            except (TypeError, ValueError):
                continue
            age_hours = (now - ts).total_seconds() / 3600.0
            if 0 <= age_hours < hours:
                index = hours - 1 - int(age_hours)
                buckets[index] += 1
                if s.error is not None or (s.status_code is not None and s.status_code >= 400):
                    error_buckets[index] += 1

        return {
            "total": total,
            "errors": errors,
            "error_rate": (errors / total) if total else 0.0,
            "avg_duration_ms": avg_duration_ms,
            "by_model": dict(by_model.most_common(10)),
            "by_api_key": dict(by_api_key.most_common(10)),
            "requests_per_hour": buckets,
            "errors_per_hour": error_buckets,
        }

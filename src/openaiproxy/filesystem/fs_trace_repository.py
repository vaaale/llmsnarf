from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from openaiproxy.interface.trace_repository import TraceRepository
from openaiproxy.models.trace_models import TraceDetail, TraceSummary

_TRACE_ID_RE = re.compile(r"^(?P<api_key>.+)_(?P<ts>\d{8}_\d{6}_\d{6})$")


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class FSTraceRepository(TraceRepository):
    def __init__(self, trace_dir: Path):
        self._trace_dir = trace_dir

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _build_summary(self, trace_id: str) -> TraceSummary | None:
        request_data = self._read_json(self._trace_dir / f"{trace_id}_request.json")
        if request_data is None:
            return None
        response_data = self._read_json(self._trace_dir / f"{trace_id}_response.json") or {}

        match = _TRACE_ID_RE.match(trace_id)
        api_key = match.group("api_key") if match else "unknown"

        payload = request_data.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        messages = payload.get("messages")
        message_count = len(messages) if isinstance(messages, list) else 0

        duration_ms: float | None = None
        request_ts = _parse_timestamp(request_data.get("timestamp"))
        response_ts = _parse_timestamp(response_data.get("timestamp"))
        if request_ts and response_ts:
            duration_ms = max((response_ts - request_ts).total_seconds() * 1000.0, 0.0)

        error = response_data.get("error")

        return TraceSummary(
            id=trace_id,
            timestamp=str(request_data.get("timestamp", "")),
            api_key=api_key,
            endpoint=str(request_data.get("endpoint", "")),
            model=payload.get("model"),
            stream=bool(payload.get("stream", False)),
            status_code=response_data.get("status_code"),
            duration_ms=duration_ms,
            message_count=message_count,
            error=str(error) if error is not None else None,
        )

    def _iter_trace_ids(self) -> list[str]:
        if not self._trace_dir.exists():
            return []
        ids = [p.name[: -len("_request.json")] for p in self._trace_dir.glob("*_request.json")]
        return sorted(ids, key=lambda trace_id: trace_id.rsplit("_", 3)[-3:], reverse=True)

    def _matches(
        self,
        summary: TraceSummary,
        model: str | None,
        api_key: str | None,
        status: str | None,
        since: str | None,
        query: str | None,
    ) -> bool:
        if model and summary.model != model:
            return False
        if api_key and summary.api_key != api_key:
            return False
        if status:
            if status == "error":
                if summary.error is None and (summary.status_code is None or summary.status_code < 400):
                    return False
            elif summary.status_code is None or not str(summary.status_code).startswith(status[0]):
                return False
        if since and summary.timestamp and summary.timestamp < since:
            return False
        if query:
            haystack = f"{summary.id} {summary.model or ''} {summary.endpoint}".lower()
            if query.lower() not in haystack:
                return False
        return True

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
        results: list[TraceSummary] = []
        skipped = 0
        for trace_id in self._iter_trace_ids():
            summary = self._build_summary(trace_id)
            if summary is None:
                continue
            if not self._matches(summary, model, api_key, status, since, query):
                continue
            if skipped < offset:
                skipped += 1
                continue
            results.append(summary)
            if len(results) >= limit:
                break
        return results

    def get_trace(self, trace_id: str) -> TraceDetail | None:
        if "/" in trace_id or "\\" in trace_id or ".." in trace_id:
            return None
        summary = self._build_summary(trace_id)
        if summary is None:
            return None

        request_data = self._read_json(self._trace_dir / f"{trace_id}_request.json") or {}
        response_data = self._read_json(self._trace_dir / f"{trace_id}_response.json") or {}

        chunks = response_data.get("chunks")
        chunks = [str(c) for c in chunks] if isinstance(chunks, list) else []

        return TraceDetail(
            summary=summary,
            request_headers=request_data.get("headers") or {},
            request_payload=request_data.get("payload"),
            response_headers=response_data.get("headers") or {},
            response_body=response_data.get("body"),
            response_chunks=chunks,
        )

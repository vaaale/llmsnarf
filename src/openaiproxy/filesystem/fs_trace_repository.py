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

    def _correlation_id_for(self, request_path: Path, request_data: dict[str, Any]) -> str | None:
        # Traces that persist the correlation id (possibly null) are authoritative;
        # the directory name can no longer be trusted now that traces are nested
        # under per-API folders (traces/completion, traces/responses, ...).
        if "correlation_id" in request_data:
            stored = request_data.get("correlation_id")
            return stored if isinstance(stored, str) and stored else None
        # Fallback for traces written before correlation ids were persisted:
        # infer it from the parent directory name (unless it is the trace root).
        parent = request_path.parent
        if parent != self._trace_dir:
            return parent.name
        return None

    def _build_summary_from_path(self, request_path: Path) -> TraceSummary | None:
        request_data = self._read_json(request_path)
        if request_data is None:
            return None
        trace_id = request_path.name[: -len("_request.json")]
        response_data = self._read_json(request_path.with_name(f"{trace_id}_response.json")) or {}

        match = _TRACE_ID_RE.match(trace_id)
        api_key = match.group("api_key") if match else "unknown"

        payload = request_data.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        messages = payload.get("messages")
        if isinstance(messages, list):
            message_count = len(messages)
        elif isinstance(payload.get("input"), list):
            message_count = len(payload["input"])
        elif isinstance(payload.get("input"), str):
            message_count = 1
        else:
            message_count = 0

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
            correlation_id=self._correlation_id_for(request_path, request_data),
            parent_trace_id=request_data.get("parent_trace_id"),
        )

    def _request_path(self, trace_id: str) -> Path | None:
        # Traces may live either at the root or nested under a correlation-id
        # directory, so search recursively for the matching request file.
        flat = self._trace_dir / f"{trace_id}_request.json"
        if flat.exists():
            return flat
        for candidate in self._trace_dir.rglob(f"{trace_id}_request.json"):
            return candidate
        return None

    def _iter_request_files(self) -> list[Path]:
        if not self._trace_dir.exists():
            return []
        files = list(self._trace_dir.rglob("*_request.json"))
        return sorted(
            files,
            key=lambda p: p.name[: -len("_request.json")].rsplit("_", 3)[-3:],
            reverse=True,
        )

    def _matches(
        self,
        summary: TraceSummary,
        model: str | None,
        api_key: str | None,
        status: str | None,
        since: str | None,
        query: str | None,
        correlation_id: str | None = None,
    ) -> bool:
        if model and summary.model != model:
            return False
        if api_key and summary.api_key != api_key:
            return False
        if correlation_id and summary.correlation_id != correlation_id:
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
        correlation_id: str | None = None,
    ) -> list[TraceSummary]:
        results: list[TraceSummary] = []
        skipped = 0
        for request_path in self._iter_request_files():
            summary = self._build_summary_from_path(request_path)
            if summary is None:
                continue
            if summary.endpoint == "/models":
                continue
            if not self._matches(summary, model, api_key, status, since, query, correlation_id):
                continue
            if skipped < offset:
                skipped += 1
                continue
            results.append(summary)
            if len(results) >= limit:
                break
        return results

    def _build_detail_from_path(self, request_path: Path) -> TraceDetail | None:
        summary = self._build_summary_from_path(request_path)
        if summary is None:
            return None
        request_data = self._read_json(request_path) or {}
        trace_id = request_path.name[: -len("_request.json")]
        response_data = self._read_json(request_path.with_name(f"{trace_id}_response.json")) or {}
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

    def _find_children(self, request_path: Path, parent_trace_id: str) -> list[TraceDetail]:
        children: list[TraceDetail] = []
        for candidate in request_path.parent.glob("*_request.json"):
            if candidate == request_path:
                continue
            data = self._read_json(candidate)
            if data is None or data.get("parent_trace_id") != parent_trace_id:
                continue
            detail = self._build_detail_from_path(candidate)
            if detail is not None:
                children.append(detail)
        children.sort(key=lambda d: d.summary.timestamp)
        return children

    def get_trace(self, trace_id: str) -> TraceDetail | None:
        if "/" in trace_id or "\\" in trace_id or ".." in trace_id:
            return None
        request_path = self._request_path(trace_id)
        if request_path is None:
            return None
        detail = self._build_detail_from_path(request_path)
        if detail is None:
            return None
        children = self._find_children(request_path, trace_id)
        if children:
            detail = TraceDetail(
                summary=detail.summary,
                request_headers=detail.request_headers,
                request_payload=detail.request_payload,
                response_headers=detail.response_headers,
                response_body=detail.response_body,
                response_chunks=detail.response_chunks,
                children=children,
            )
        return detail

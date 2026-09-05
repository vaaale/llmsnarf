from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import orjson

from openaiproxy.interface.trace_repository import TraceRepository
from openaiproxy.models.trace_models import TraceDetail, TraceSummary, extract_usage, parse_trace_id

REQUEST_SUFFIX = "_request.json"
RESPONSE_SUFFIX = "_response.json"

# traces/<category>/<YYYY-MM-DD>/ — a grouping directory, never a correlation id.
_DAY_SHARD_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# The filename timestamp is stamped a few microseconds before the timestamp
# written into the trace body, so range-filtering on the filename alone could
# drop a trace that the exact comparison would have kept. A one second slack
# makes the prefilter conservative; the precise check still runs afterwards.
_PREFILTER_SLACK = timedelta(seconds=1)


def trace_id_from_request_path(request_path: Path) -> str:
    return request_path.name[: -len(REQUEST_SUFFIX)]


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def with_children(detail: TraceDetail, children: list[TraceDetail]) -> TraceDetail:
    """Attach children to a trace detail (TraceDetail is frozen)."""
    if not children:
        return detail
    return TraceDetail(
        summary=detail.summary,
        request_headers=detail.request_headers,
        request_payload=detail.request_payload,
        response_headers=detail.response_headers,
        response_body=detail.response_body,
        response_chunks=detail.response_chunks,
        children=children,
    )


def _prefilter_cutoff(since: str | None) -> str | None:
    if not since:
        return None
    parsed = _parse_timestamp(since)
    if parsed is None:
        return None
    return (parsed - _PREFILTER_SLACK).isoformat()


class FSTraceRepository(TraceRepository):
    """Reads traces straight off disk.

    This is the authoritative store: every trace is a pair of JSON files. It is
    also the slow path — building a summary means parsing the response body, and
    response bodies are routinely hundreds of kilobytes. Prefer going through
    the trace index for listing; this class stays the fallback and the source
    the index is rebuilt from.
    """

    def __init__(self, trace_dir: Path):
        self._trace_dir = trace_dir

    @property
    def trace_dir(self) -> Path:
        return self._trace_dir

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            with open(path, "rb") as f:
                data = orjson.loads(f.read())
            return data if isinstance(data, dict) else None
        except (OSError, orjson.JSONDecodeError):
            return None

    def _correlation_id_for(self, request_path: Path, request_data: dict[str, Any]) -> str | None:
        # Traces that persist the correlation id (possibly null) are authoritative;
        # the directory name can no longer be trusted now that traces are nested
        # under per-API folders (traces/completion, traces/responses, ...).
        if "correlation_id" in request_data:
            stored = request_data.get("correlation_id")
            return stored if isinstance(stored, str) and stored else None
        # Fallback for traces written before correlation ids were persisted:
        # infer it from the parent directory name, unless that directory is the
        # trace root or a day shard (which is a grouping, not a thread).
        parent = request_path.parent
        if parent != self._trace_dir and not _DAY_SHARD_RE.match(parent.name):
            return parent.name
        return None

    def build_summary(self, request_path: Path) -> TraceSummary | None:
        request_data = self._read_json(request_path)
        if request_data is None:
            return None
        trace_id = trace_id_from_request_path(request_path)
        response_data = self._read_json(request_path.with_name(f"{trace_id}{RESPONSE_SUFFIX}")) or {}

        api_key, _ = parse_trace_id(trace_id)

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

        response_chunks = response_data.get("chunks")
        response_chunks = [str(c) for c in response_chunks] if isinstance(response_chunks, list) else []
        input_tokens, output_tokens = extract_usage(response_data.get("body"), response_chunks)

        provider = request_data.get("provider")

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
            provider=str(provider) if provider else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def has_response(self, request_path: Path) -> bool:
        """Whether the response half of a trace has been written yet.

        Streaming responses land well after the request, so a trace indexed
        early has no status, duration or token counts until this turns true.
        """
        trace_id = trace_id_from_request_path(request_path)
        return request_path.with_name(f"{trace_id}{RESPONSE_SUFFIX}").exists()

    def request_path_for(self, trace_id: str) -> Path | None:
        # Traces may live either at the root or nested under category/date/
        # correlation-id directories, so search recursively for the match.
        flat = self._trace_dir / f"{trace_id}{REQUEST_SUFFIX}"
        if flat.exists():
            return flat
        for candidate in self._trace_dir.rglob(f"{trace_id}{REQUEST_SUFFIX}"):
            return candidate
        return None

    def iter_request_files(self, since: str | None = None) -> list[Path]:
        """All request files, newest first, optionally limited to ``since``.

        Both the ordering and the ``since`` cut come from the filename, so no
        trace is opened just to be discarded.
        """
        if not self._trace_dir.exists():
            return []
        cutoff = _prefilter_cutoff(since)
        keyed: list[tuple[str, Path]] = []
        for path in self._trace_dir.rglob(f"*{REQUEST_SUFFIX}"):
            _, timestamp = parse_trace_id(trace_id_from_request_path(path))
            if timestamp is None:
                # Unparseable id: it cannot be ordered or range-filtered, so
                # keep it and let the exact comparison decide.
                keyed.append(("", path))
                continue
            if cutoff is not None and timestamp < cutoff:
                continue
            keyed.append((timestamp, path))
        keyed.sort(key=lambda item: item[0], reverse=True)
        return [path for _, path in keyed]

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
        for request_path in self.iter_request_files(since=since):
            summary = self.build_summary(request_path)
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

    def build_call(self, request_path: Path) -> TraceDetail | None:
        """One trace, without its children."""
        summary = self.build_summary(request_path)
        if summary is None:
            return None
        request_data = self._read_json(request_path) or {}
        trace_id = trace_id_from_request_path(request_path)
        response_data = self._read_json(request_path.with_name(f"{trace_id}{RESPONSE_SUFFIX}")) or {}
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

    def build_detail(self, request_path: Path) -> TraceDetail | None:
        detail = self.build_call(request_path)
        if detail is None:
            return None
        children = self._find_children(request_path, trace_id_from_request_path(request_path))
        return with_children(detail, children)

    def _find_children(self, request_path: Path, parent_trace_id: str) -> list[TraceDetail]:
        children: list[TraceDetail] = []
        for candidate in request_path.parent.glob(f"*{REQUEST_SUFFIX}"):
            if candidate == request_path:
                continue
            data = self._read_json(candidate)
            if data is None or data.get("parent_trace_id") != parent_trace_id:
                continue
            child = self._build_call(candidate)
            if child is not None:
                children.append(child)
        children.sort(key=lambda d: d.summary.timestamp)
        return children

    def get_trace(self, trace_id: str) -> TraceDetail | None:
        if "/" in trace_id or "\\" in trace_id or ".." in trace_id:
            return None
        request_path = self.request_path_for(trace_id)
        if request_path is None:
            return None
        return self.build_detail(request_path)

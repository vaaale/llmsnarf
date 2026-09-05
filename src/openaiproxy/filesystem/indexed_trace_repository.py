from __future__ import annotations

import threading

from openaiproxy.filesystem.fs_trace_repository import FSTraceRepository, with_children
from openaiproxy.interface.trace_index import TraceIndex
from openaiproxy.interface.trace_repository import TraceRepository
from openaiproxy.models.trace_models import TraceDetail, TraceSummary


class IndexedTraceRepository(TraceRepository):
    """Lists traces from the index, reads trace bodies from disk.

    Listing is the hot path and the index answers it without opening a file.
    Details still come from the JSON pair on disk — the index only stores the
    path, so opening a trace stays a direct read instead of a directory walk.

    While ``ready`` is unset (initial build, or a rebuild in flight) the index
    holds an incomplete picture, so listing falls back to scanning disk. Slower,
    but it never shows the user a half-populated trace list.
    """

    def __init__(self, index: TraceIndex, files: FSTraceRepository, ready: threading.Event):
        self._index = index
        self._files = files
        self._ready = ready

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
        if not self._ready.is_set():
            return self._files.list_traces(
                limit=limit,
                offset=offset,
                model=model,
                api_key=api_key,
                status=status,
                since=since,
                query=query,
                correlation_id=correlation_id,
            )
        return self._index.query(
            limit=limit,
            offset=offset,
            model=model,
            api_key=api_key,
            status=status,
            since=since,
            query=query,
            correlation_id=correlation_id,
        )

    def get_trace(self, trace_id: str) -> TraceDetail | None:
        if "/" in trace_id or "\\" in trace_id or ".." in trace_id:
            return None
        entry = self._index.get(trace_id)
        if entry is None or not entry.path.exists():
            # Not indexed yet (or the file moved): fall back to searching disk.
            return self._files.get_trace(trace_id)
        detail = self._files.build_call(entry.path)
        if detail is None:
            return self._files.get_trace(trace_id)
        # Map/reduce sub-calls are found via the index rather than by reading
        # every neighbouring trace file to check its parent id.
        children = [
            child
            for child in (
                self._files.build_call(row.path)
                for row in self._index.children_of(trace_id)
                if row.path.exists()
            )
            if child is not None
        ]
        return with_children(detail, children)

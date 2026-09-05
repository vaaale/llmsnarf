from __future__ import annotations

from pathlib import Path

from openaiproxy.filesystem.fs_ledger_repository import LEDGER_FILENAME
from openaiproxy.services.trace_index_service import TraceIndexService


def _file_token(path: Path) -> str:
    """A value that changes whenever the file does, without reading it."""
    try:
        stat = path.stat()
    except OSError:
        return "0"
    return f"{stat.st_mtime_ns}:{stat.st_size}"


class RevisionService:
    """Change-detection tokens the UI polls to decide whether to refetch.

    Every token is a counter or a stat() result, so polling this a few times a
    second costs nothing — which is the whole point: the expensive endpoints are
    only called when something actually changed.

    Tokens are opaque. Callers compare them for equality and must not read
    meaning into the value.
    """

    def __init__(self, trace_index_service: TraceIndexService, trace_dir: Path):
        self._trace_index_service = trace_index_service
        self._trace_dir = trace_dir

    def revisions(self) -> dict[str, str | bool]:
        status = self._trace_index_service.status()
        return {
            # Costs are derived from the same index, so this token covers the
            # cost view too; there is no separate store to watch.
            "traces": str(status.revision),
            "ledger": _file_token(self._trace_dir / LEDGER_FILENAME),
            # Surfaced here so the UI can show an indexing banner without a
            # second request on every poll.
            "indexing": status.building or not status.ready,
        }

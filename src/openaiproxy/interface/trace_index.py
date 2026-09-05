from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from openaiproxy.models.cost_models import CostUsageEntry
from openaiproxy.models.trace_models import TraceSummary


@dataclass(frozen=True)
class IndexedTrace:
    """An index row: the summary plus what the index needs to maintain it."""

    summary: TraceSummary
    path: Path
    # False while only the request half of the trace exists on disk. Such rows
    # carry no status code, duration or token counts and are revisited by the
    # indexer until the response lands.
    complete: bool
    # Filename-derived timestamp. Ordering uses this rather than the timestamp
    # inside the trace body so listings match the on-disk repository exactly.
    sort_key: str


class TraceIndex(ABC):
    @abstractmethod
    def upsert_many(self, entries: list[IndexedTrace]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(
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
        raise NotImplementedError

    @abstractmethod
    def get(self, trace_id: str) -> IndexedTrace | None:
        raise NotImplementedError

    @abstractmethod
    def children_of(self, parent_trace_id: str) -> list[IndexedTrace]:
        """Sub-calls of a trace (map/reduce fan-out), oldest first."""
        raise NotImplementedError

    @abstractmethod
    def usage_entries(self, since: str | None = None) -> list[CostUsageEntry]:
        """Token usage for every indexed trace, for cost reporting.

        Usage is already extracted when a trace is indexed, so cost totals come
        from the same rows that back the trace list rather than from a separate
        ledger that only covers whatever was recorded live.
        """
        raise NotImplementedError

    @abstractmethod
    def known_trace_ids(self) -> set[str]:
        raise NotImplementedError

    @abstractmethod
    def incomplete(self, newer_than: str) -> list[tuple[str, Path]]:
        """Trace ids (with request paths) still awaiting a response file.

        Bounded by ``newer_than`` (a sort key) because a trace whose response
        never arrived — the proxy died mid-request — would otherwise be
        re-checked on every pass forever.
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def revision(self) -> int:
        """Monotonic counter bumped on every write; drives UI change detection."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

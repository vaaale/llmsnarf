from __future__ import annotations

from abc import ABC, abstractmethod

from openaiproxy.models.trace_models import TraceDetail, TraceSummary


class TraceRepository(ABC):
    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def get_trace(self, trace_id: str) -> TraceDetail | None:
        raise NotImplementedError

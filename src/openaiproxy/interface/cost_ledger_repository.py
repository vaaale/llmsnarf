from __future__ import annotations

from abc import ABC, abstractmethod

from openaiproxy.models.cost_models import CostUsageEntry


class CostLedgerRepository(ABC):
    @abstractmethod
    def append(self, entry: CostUsageEntry) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_entries(self, since: str | None = None) -> list[CostUsageEntry]:
        raise NotImplementedError

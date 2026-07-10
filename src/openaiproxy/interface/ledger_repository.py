from __future__ import annotations

from abc import ABC, abstractmethod

from openaiproxy.models.ledger_models import LedgerEntry


class LedgerRepository(ABC):
    @abstractmethod
    def append(self, entry: LedgerEntry) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_entries(
        self,
        limit: int = 100,
        offset: int = 0,
        model: str | None = None,
        since: str | None = None,
    ) -> list[LedgerEntry]:
        raise NotImplementedError

    @abstractmethod
    def get_entry(self, entry_id: str) -> LedgerEntry | None:
        raise NotImplementedError

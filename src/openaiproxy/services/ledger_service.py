from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

import anyio

from openaiproxy.interface.ledger_repository import LedgerRepository
from openaiproxy.models.ledger_models import LedgerEntry
from openaiproxy.services.validation_service import ValidationService


class LedgerService:
    def __init__(
        self,
        ledger_repository: LedgerRepository,
        validation_service: ValidationService,
        logger: logging.Logger,
    ):
        self._ledger_repository = ledger_repository
        self._validation_service = validation_service
        self._logger = logger

    async def validate_and_record(
        self,
        trace_id: str,
        model: str | None,
        endpoint: str,
        request_payload: Any,
        response_body: Any,
        response_chunks: list[str],
    ) -> None:
        def _run() -> None:
            try:
                issues = self._validation_service.validate_trace(
                    request_payload=request_payload,
                    response_body=response_body,
                    response_chunks=response_chunks,
                )
                if not issues:
                    return
                entry = LedgerEntry(
                    id=uuid.uuid4().hex,
                    trace_id=trace_id,
                    timestamp=datetime.now().isoformat(),
                    model=model,
                    endpoint=endpoint,
                    issues=issues,
                )
                self._ledger_repository.append(entry)
                self._logger.info(
                    "ledger_entry trace_id=%s model=%s issues=%d",
                    trace_id,
                    model,
                    len(issues),
                )
            except Exception as exc:
                self._logger.warning("ledger_validate_error trace_id=%s: %s", trace_id, exc)

        await anyio.to_thread.run_sync(_run)

    def list_entries(
        self,
        limit: int = 100,
        offset: int = 0,
        model: str | None = None,
        since: str | None = None,
    ) -> list[LedgerEntry]:
        return self._ledger_repository.list_entries(limit=limit, offset=offset, model=model, since=since)

    def get_entry(self, entry_id: str) -> LedgerEntry | None:
        return self._ledger_repository.get_entry(entry_id)

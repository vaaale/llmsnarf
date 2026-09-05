from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

import anyio

from openaiproxy.interface.cost_ledger_repository import CostLedgerRepository
from openaiproxy.models.cost_models import CostUsageEntry
from openaiproxy.models.trace_models import extract_usage


class CostRecorderService:
    """Extracts token usage from a completed request/response pair and appends
    it to the cost ledger in real time, so the cost page never has to rescan
    the full trace history to stay up to date."""

    def __init__(self, cost_ledger_repository: CostLedgerRepository, logger: logging.Logger):
        self._cost_ledger_repository = cost_ledger_repository
        self._logger = logger

    async def record_usage(
        self,
        trace_id: str,
        model: str | None,
        provider: str | None,
        endpoint: str,
        response_body: Any,
        response_chunks: list[str],
    ) -> None:
        def _run() -> None:
            try:
                input_tokens, output_tokens = extract_usage(response_body, response_chunks)
                entry = CostUsageEntry(
                    id=uuid.uuid4().hex,
                    trace_id=trace_id,
                    timestamp=datetime.now().isoformat(),
                    model=model,
                    provider=provider,
                    endpoint=endpoint,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                self._cost_ledger_repository.append(entry)
            except Exception as exc:
                self._logger.warning("cost_record_error trace_id=%s: %s", trace_id, exc)

        await anyio.to_thread.run_sync(_run)

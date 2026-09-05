from __future__ import annotations

import json
import threading
from pathlib import Path

from openaiproxy.interface.cost_ledger_repository import CostLedgerRepository
from openaiproxy.models.cost_models import CostUsageEntry

_COST_LEDGER_FILENAME = "cost_ledger.ndjson"


class FSCostLedgerRepository(CostLedgerRepository):
    def __init__(self, trace_dir: Path):
        self._path = trace_dir / _COST_LEDGER_FILENAME
        self._lock = threading.Lock()

    def append(self, entry: CostUsageEntry) -> None:
        record = {
            "id": entry.id,
            "trace_id": entry.trace_id,
            "timestamp": entry.timestamp,
            "model": entry.model,
            "provider": entry.provider,
            "endpoint": entry.endpoint,
            "input_tokens": entry.input_tokens,
            "output_tokens": entry.output_tokens,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)

    def list_entries(self, since: str | None = None) -> list[CostUsageEntry]:
        if not self._path.exists():
            return []
        with self._lock:
            try:
                lines = self._path.read_text(encoding="utf-8").splitlines()
            except OSError:
                return []

        entries: list[CostUsageEntry] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            timestamp = record.get("timestamp", "")
            if since and timestamp and timestamp < since:
                continue
            entries.append(
                CostUsageEntry(
                    id=record.get("id", ""),
                    trace_id=record.get("trace_id", ""),
                    timestamp=timestamp,
                    model=record.get("model"),
                    provider=record.get("provider"),
                    endpoint=record.get("endpoint", ""),
                    input_tokens=record.get("input_tokens"),
                    output_tokens=record.get("output_tokens"),
                )
            )
        return entries

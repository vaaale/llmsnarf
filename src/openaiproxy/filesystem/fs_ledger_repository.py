from __future__ import annotations

import json
import threading
from pathlib import Path

from openaiproxy.interface.ledger_repository import LedgerRepository
from openaiproxy.models.ledger_models import LedgerEntry, ValidationIssue

_LEDGER_FILENAME = "validation_ledger.ndjson"


class FSLedgerRepository(LedgerRepository):
    def __init__(self, trace_dir: Path):
        self._path = trace_dir / _LEDGER_FILENAME
        self._lock = threading.Lock()

    def append(self, entry: LedgerEntry) -> None:
        record = {
            "id": entry.id,
            "trace_id": entry.trace_id,
            "timestamp": entry.timestamp,
            "model": entry.model,
            "endpoint": entry.endpoint,
            "issues": [
                {"code": i.code, "message": i.message, "path": i.path}
                for i in entry.issues
            ],
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)

    def _iter_entries(self) -> list[LedgerEntry]:
        if not self._path.exists():
            return []
        entries: list[LedgerEntry] = []
        with self._lock:
            try:
                lines = self._path.read_text(encoding="utf-8").splitlines()
            except OSError:
                return []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            issues = [
                ValidationIssue(
                    code=i.get("code", ""),
                    message=i.get("message", ""),
                    path=i.get("path", ""),
                )
                for i in (record.get("issues") or [])
            ]
            entries.append(
                LedgerEntry(
                    id=record.get("id", ""),
                    trace_id=record.get("trace_id", ""),
                    timestamp=record.get("timestamp", ""),
                    model=record.get("model"),
                    endpoint=record.get("endpoint", ""),
                    issues=issues,
                )
            )
        return entries

    def list_entries(
        self,
        limit: int = 100,
        offset: int = 0,
        model: str | None = None,
        since: str | None = None,
    ) -> list[LedgerEntry]:
        all_entries = list(reversed(self._iter_entries()))
        filtered: list[LedgerEntry] = []
        for entry in all_entries:
            if model and entry.model != model:
                continue
            if since and entry.timestamp and entry.timestamp < since:
                continue
            filtered.append(entry)
        return filtered[offset : offset + limit]

    def get_entry(self, entry_id: str) -> LedgerEntry | None:
        for entry in self._iter_entries():
            if entry.id == entry_id:
                return entry
        return None

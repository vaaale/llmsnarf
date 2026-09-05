from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from openaiproxy.interface.trace_index import IndexedTrace, TraceIndex
from openaiproxy.models.cost_models import CostUsageEntry
from openaiproxy.models.trace_models import TraceSummary

INDEX_FILENAME = "trace_index.db"

# Bump when the table layout changes; a mismatch drops and rebuilds the index
# rather than trying to migrate, since every row is derivable from disk.
_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id        TEXT PRIMARY KEY,
    path            TEXT NOT NULL,
    sort_key        TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    api_key         TEXT NOT NULL,
    endpoint        TEXT NOT NULL,
    model           TEXT,
    stream          INTEGER NOT NULL DEFAULT 0,
    status_code     INTEGER,
    duration_ms     REAL,
    message_count   INTEGER NOT NULL DEFAULT 0,
    error           TEXT,
    correlation_id  TEXT,
    parent_trace_id TEXT,
    provider        TEXT,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    complete        INTEGER NOT NULL DEFAULT 0,
    search          TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_traces_sort ON traces(sort_key DESC);
CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp);
CREATE INDEX IF NOT EXISTS idx_traces_model ON traces(model);
CREATE INDEX IF NOT EXISTS idx_traces_api_key ON traces(api_key);
CREATE INDEX IF NOT EXISTS idx_traces_correlation ON traces(correlation_id);
CREATE INDEX IF NOT EXISTS idx_traces_parent ON traces(parent_trace_id);
CREATE INDEX IF NOT EXISTS idx_traces_incomplete ON traces(complete);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

_COLUMNS = (
    "trace_id, path, sort_key, timestamp, api_key, endpoint, model, stream, status_code, "
    "duration_ms, message_count, error, correlation_id, parent_trace_id, provider, "
    "input_tokens, output_tokens, complete, search"
)


def _row_to_summary(row: sqlite3.Row) -> TraceSummary:
    return TraceSummary(
        id=row["trace_id"],
        timestamp=row["timestamp"],
        api_key=row["api_key"],
        endpoint=row["endpoint"],
        model=row["model"],
        stream=bool(row["stream"]),
        status_code=row["status_code"],
        duration_ms=row["duration_ms"],
        message_count=row["message_count"],
        error=row["error"],
        correlation_id=row["correlation_id"],
        parent_trace_id=row["parent_trace_id"],
        provider=row["provider"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
    )


def _search_text(summary: TraceSummary) -> str:
    return f"{summary.id} {summary.model or ''} {summary.endpoint}".lower()


class SqliteTraceIndex(TraceIndex):
    """Trace metadata mirrored into SQLite so listing never touches trace files.

    Every row is derived from the JSON pair on disk, which stays authoritative —
    the index is a cache that can be thrown away and rebuilt at any time.
    Connections are per-thread (sqlite3 objects are not shareable) and the
    database runs in WAL mode so the background indexer can write while request
    threads read.
    """

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._ensure_schema()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            return connection
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=30000")
        self._local.connection = connection
        return connection

    def _ensure_schema(self) -> None:
        connection = self._connect()
        with self._write_lock:
            version = self._read_meta(connection, "schema_version")
            if version is not None and version != str(_SCHEMA_VERSION):
                connection.executescript("DROP TABLE IF EXISTS traces; DROP TABLE IF EXISTS meta;")
            connection.executescript(_SCHEMA)
            connection.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(_SCHEMA_VERSION),),
            )
            connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('revision', '0')"
            )
            connection.commit()

    @staticmethod
    def _read_meta(connection: sqlite3.Connection, key: str) -> str | None:
        try:
            row = connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        except sqlite3.OperationalError:
            return None
        return row["value"] if row else None

    @staticmethod
    def _bump_revision(connection: sqlite3.Connection) -> None:
        connection.execute(
            "UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'revision'"
        )

    def upsert_many(self, entries: list[IndexedTrace]) -> None:
        if not entries:
            return
        rows = [
            (
                entry.summary.id,
                str(entry.path),
                entry.sort_key,
                entry.summary.timestamp,
                entry.summary.api_key,
                entry.summary.endpoint,
                entry.summary.model,
                1 if entry.summary.stream else 0,
                entry.summary.status_code,
                entry.summary.duration_ms,
                entry.summary.message_count,
                entry.summary.error,
                entry.summary.correlation_id,
                entry.summary.parent_trace_id,
                entry.summary.provider,
                entry.summary.input_tokens,
                entry.summary.output_tokens,
                1 if entry.complete else 0,
                _search_text(entry.summary),
            )
            for entry in entries
        ]
        placeholders = ", ".join(["?"] * 19)
        connection = self._connect()
        with self._write_lock:
            connection.executemany(
                f"INSERT INTO traces ({_COLUMNS}) VALUES ({placeholders}) "
                f"ON CONFLICT(trace_id) DO UPDATE SET "
                + ", ".join(
                    f"{column} = excluded.{column}"
                    for column in _COLUMNS.split(", ")
                    if column != "trace_id"
                ),
                rows,
            )
            self._bump_revision(connection)
            connection.commit()

    def _where(
        self,
        model: str | None,
        api_key: str | None,
        status: str | None,
        since: str | None,
        query: str | None,
        correlation_id: str | None,
    ) -> tuple[str, list]:
        # Traces for /models are noise in the UI and were never listed by the
        # on-disk repository either; they are still indexed so the incremental
        # scan does not keep rediscovering them.
        clauses = ["endpoint <> '/models'"]
        params: list = []
        if model:
            clauses.append("model = ?")
            params.append(model)
        if api_key:
            clauses.append("api_key = ?")
            params.append(api_key)
        if correlation_id:
            clauses.append("correlation_id = ?")
            params.append(correlation_id)
        if status:
            if status == "error":
                clauses.append("(error IS NOT NULL OR status_code >= 400)")
            else:
                clauses.append("substr(CAST(status_code AS TEXT), 1, 1) = ?")
                params.append(status[0])
        if since:
            # An empty timestamp means the trace body had none; the on-disk
            # repository lets those through the since filter, so mirror that.
            clauses.append("(timestamp = '' OR timestamp >= ?)")
            params.append(since)
        if query:
            clauses.append("search LIKE ? ESCAPE '\\'")
            escaped = query.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params.append(f"%{escaped}%")
        return " AND ".join(clauses), params

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
        where, params = self._where(model, api_key, status, since, query, correlation_id)
        sql = (
            f"SELECT {_COLUMNS} FROM traces WHERE {where} "
            f"ORDER BY sort_key DESC LIMIT ? OFFSET ?"
        )
        rows = self._connect().execute(sql, [*params, limit, offset]).fetchall()
        return [_row_to_summary(row) for row in rows]

    def get(self, trace_id: str) -> IndexedTrace | None:
        row = self._connect().execute(
            f"SELECT {_COLUMNS} FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if row is None:
            return None
        return IndexedTrace(
            summary=_row_to_summary(row),
            path=Path(row["path"]),
            complete=bool(row["complete"]),
            sort_key=row["sort_key"],
        )

    def children_of(self, parent_trace_id: str) -> list[IndexedTrace]:
        rows = self._connect().execute(
            f"SELECT {_COLUMNS} FROM traces WHERE parent_trace_id = ? ORDER BY timestamp ASC",
            (parent_trace_id,),
        ).fetchall()
        return [
            IndexedTrace(
                summary=_row_to_summary(row),
                path=Path(row["path"]),
                complete=bool(row["complete"]),
                sort_key=row["sort_key"],
            )
            for row in rows
        ]

    def usage_entries(self, since: str | None = None) -> list[CostUsageEntry]:
        clauses = ["endpoint <> '/models'"]
        params: list = []
        if since:
            clauses.append("(timestamp = '' OR timestamp >= ?)")
            params.append(since)
        rows = self._connect().execute(
            "SELECT trace_id, timestamp, model, provider, endpoint, input_tokens, output_tokens "
            f"FROM traces WHERE {' AND '.join(clauses)} ORDER BY sort_key DESC",
            params,
        ).fetchall()
        return [
            CostUsageEntry(
                # Trace id and usage entry are one-to-one, so it doubles as the id.
                id=row["trace_id"],
                trace_id=row["trace_id"],
                timestamp=row["timestamp"],
                model=row["model"],
                provider=row["provider"],
                endpoint=row["endpoint"],
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
            )
            for row in rows
        ]

    def known_trace_ids(self) -> set[str]:
        rows = self._connect().execute("SELECT trace_id FROM traces").fetchall()
        return {row["trace_id"] for row in rows}

    def incomplete(self, newer_than: str) -> list[tuple[str, Path]]:
        rows = self._connect().execute(
            "SELECT trace_id, path FROM traces WHERE complete = 0 AND sort_key >= ?",
            (newer_than,),
        ).fetchall()
        return [(row["trace_id"], Path(row["path"])) for row in rows]

    def count(self) -> int:
        row = self._connect().execute("SELECT COUNT(*) AS n FROM traces").fetchone()
        return int(row["n"]) if row else 0

    def revision(self) -> int:
        value = self._read_meta(self._connect(), "revision")
        return int(value) if value is not None and value.isdigit() else 0

    def clear(self) -> None:
        connection = self._connect()
        with self._write_lock:
            connection.execute("DELETE FROM traces")
            self._bump_revision(connection)
            connection.commit()

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from openaiproxy.filesystem.fs_trace_repository import (
    REQUEST_SUFFIX,
    FSTraceRepository,
    trace_id_from_request_path,
)
from openaiproxy.interface.trace_index import IndexedTrace, TraceIndex
from openaiproxy.models.trace_models import parse_trace_id

# How long a trace may wait for its response file before the indexer stops
# revisiting it. Long streaming responses finish well inside this.
_RESPONSE_GRACE = timedelta(minutes=30)

# Rows written per transaction during a rebuild.
_BATCH_SIZE = 500


@dataclass(frozen=True)
class IndexStatus:
    ready: bool
    building: bool
    indexed: int
    scanned: int
    total: int
    revision: int
    last_build_seconds: float | None
    last_error: str | None


class TraceIndexService:
    """Keeps the trace index in step with the trace directory.

    Runs one background thread. On start it catches up on everything written
    while the proxy was down (a full build the first time), then polls for new
    traces. Traces indexed before their response file landed are revisited until
    they complete.

    A rebuild drops the index and re-reads every trace from disk. It runs on the
    same thread so it can never overlap an incremental pass, and it clears the
    ready flag so listings fall back to disk while it works.
    """

    def __init__(
        self,
        index: TraceIndex,
        files: FSTraceRepository,
        logger: logging.Logger,
        poll_interval: float = 2.0,
        full_sweep_interval: float = 60.0,
    ):
        self._index = index
        self._files = files
        self._logger = logger
        self._poll_interval = poll_interval
        self._full_sweep_interval = full_sweep_interval

        self.ready = threading.Event()
        self._stopping = threading.Event()
        self._wake = threading.Event()
        self._rebuild_requested = threading.Event()
        self._thread: threading.Thread | None = None

        self._state_lock = threading.Lock()
        self._building = False
        self._scanned = 0
        self._total = 0
        self._last_build_seconds: float | None = None
        self._last_error: str | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="trace-indexer", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stopping.set()
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None

    def request_rebuild(self) -> None:
        self._rebuild_requested.set()
        # Cleared here rather than on the indexer thread so the caller sees the
        # rebuild reflected in the status it gets back, instead of a stale
        # "idle" until the thread next wakes.
        self.ready.clear()
        self._wake.set()

    def status(self) -> IndexStatus:
        with self._state_lock:
            return IndexStatus(
                ready=self.ready.is_set(),
                building=self._building,
                indexed=self._index.count(),
                scanned=self._scanned,
                total=self._total,
                revision=self._index.revision(),
                last_build_seconds=self._last_build_seconds,
                last_error=self._last_error,
            )

    # ── background loop ──────────────────────────────────────────────────────

    def _run(self) -> None:
        last_full_sweep = 0.0
        while not self._stopping.is_set():
            delay = self._poll_interval
            try:
                if self._rebuild_requested.is_set():
                    self._rebuild_requested.clear()
                    self._rebuild()
                    last_full_sweep = time.monotonic()
                elif not self.ready.is_set():
                    # First pass: everything written while the proxy was down.
                    # Only flip ready on success — until then listings fall back
                    # to disk, which is correct, just slower.
                    self._catch_up()
                    last_full_sweep = time.monotonic()
                    self.ready.set()
                else:
                    due_for_sweep = time.monotonic() - last_full_sweep >= self._full_sweep_interval
                    self._index_paths(self._scan(hot_only=not due_for_sweep))
                    if due_for_sweep:
                        last_full_sweep = time.monotonic()
                    self._refresh_incomplete()
            except Exception as exc:  # a broken index must not take the proxy down
                self._record_error(exc, "trace index update failed")
                # Back off rather than hammering a persistent failure.
                delay = self._full_sweep_interval
            self._wake.wait(delay)
            self._wake.clear()

    def _record_error(self, exc: Exception, message: str) -> None:
        self._logger.exception(message)
        with self._state_lock:
            self._last_error = f"{type(exc).__name__}: {exc}"

    # ── scanning ─────────────────────────────────────────────────────────────

    def _hot_dirs(self) -> list[Path]:
        """Directories new traces can land in.

        New traces are written under ``<category>/<YYYY-MM-DD>/``, so the
        incremental pass only has to look at today's and yesterday's shards
        instead of walking the whole corpus. The trace root itself is included
        for the flat legacy layout.
        """
        root = self._files.trace_dir
        if not root.exists():
            return []
        today = datetime.now().date()
        days = {today.isoformat(), (today - timedelta(days=1)).isoformat()}
        dirs: list[Path] = []
        for category in root.iterdir():
            if not category.is_dir():
                continue
            for day in days:
                shard = category / day
                if shard.is_dir():
                    dirs.append(shard)
        return dirs

    def _scan(self, hot_only: bool) -> list[Path]:
        """Request files on disk that the index has not seen yet."""
        if hot_only:
            candidates: list[Path] = []
            root = self._files.trace_dir
            if root.exists():
                candidates.extend(root.glob(f"*{REQUEST_SUFFIX}"))
            for directory in self._hot_dirs():
                candidates.extend(directory.rglob(f"*{REQUEST_SUFFIX}"))
        else:
            candidates = self._files.iter_request_files()

        known = self._index.known_trace_ids()
        return [
            path for path in candidates if trace_id_from_request_path(path) not in known
        ]

    def _build_entry(self, request_path: Path) -> IndexedTrace | None:
        summary = self._files.build_summary(request_path)
        if summary is None:
            return None
        _, sort_key = parse_trace_id(summary.id)
        return IndexedTrace(
            summary=summary,
            path=request_path,
            complete=self._files.has_response(request_path),
            sort_key=sort_key or summary.timestamp,
        )

    def _index_paths(self, paths: list[Path], track_progress: bool = False) -> int:
        if not paths:
            return 0
        if track_progress:
            with self._state_lock:
                self._total = len(paths)
                self._scanned = 0

        indexed = 0
        # Deliberately single-threaded. Reading a trace is dominated by JSON
        # parsing, which holds the GIL, so a reader pool made the build slower
        # *and* starved foreground requests of the interpreter while it ran.
        for start in range(0, len(paths), _BATCH_SIZE):
            chunk = paths[start:start + _BATCH_SIZE]
            batch = [entry for entry in map(self._build_entry, chunk) if entry is not None]
            if track_progress:
                with self._state_lock:
                    self._scanned += len(chunk)
            if batch:
                self._index.upsert_many(batch)
                indexed += len(batch)
            # Checked per batch so shutdown is prompt on a large corpus.
            if self._stopping.is_set():
                break
        return indexed

    def _refresh_incomplete(self) -> None:
        """Re-read traces whose response file has since appeared."""
        cutoff = (datetime.now() - _RESPONSE_GRACE).isoformat()
        pending = self._index.incomplete(cutoff)
        ready = [path for _, path in pending if self._files.has_response(path)]
        if ready:
            self._index_paths(ready)

    # ── builds ───────────────────────────────────────────────────────────────

    def _catch_up(self) -> None:
        """Index everything on disk the index is missing."""
        started = time.monotonic()
        with self._state_lock:
            self._building = True
            self._last_error = None
        try:
            missing = self._scan(hot_only=False)
            if missing:
                self._logger.info("indexing %d trace(s)", len(missing))
            indexed = self._index_paths(missing, track_progress=True)
            self._refresh_incomplete()
            elapsed = time.monotonic() - started
            if indexed:
                self._logger.info("indexed %d trace(s) in %.1fs", indexed, elapsed)
            with self._state_lock:
                self._last_build_seconds = elapsed
        finally:
            with self._state_lock:
                self._building = False

    def _rebuild(self) -> None:
        """Drop the index and re-read every trace from disk."""
        self._logger.info("rebuilding trace index from %s", self._files.trace_dir)
        # Cleared first: a half-rebuilt index must never be served, and if the
        # rebuild fails the main loop retries the catch-up before setting ready.
        self.ready.clear()
        self._index.clear()
        self._catch_up()
        self.ready.set()

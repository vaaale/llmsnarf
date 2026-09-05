from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from fastapi.staticfiles import StaticFiles

from openaiproxy.api.endpoints.router_main import router as main_router
from openaiproxy.api.endpoints.router_ui import router as ui_router
from openaiproxy.filesystem.fs_ledger_repository import FSLedgerRepository
from openaiproxy.filesystem.fs_trace_repository import FSTraceRepository
from openaiproxy.filesystem.indexed_trace_repository import IndexedTraceRepository
from openaiproxy.filesystem.sqlite_trace_index import INDEX_FILENAME, SqliteTraceIndex
from openaiproxy.interface.repository import LLMProxyConfigRepository
from openaiproxy.logging_config import configure_logging
from openaiproxy.middleware.exception_logging import ExceptionLoggingMiddleware
from openaiproxy.services.anthropic_service import AnthropicService
from openaiproxy.services.config_service import ConfigService
from openaiproxy.services.cost_service import CostService
from openaiproxy.services.embeddings_service import EmbeddingsService
from openaiproxy.services.ledger_service import LedgerService
from openaiproxy.services.model_tracker_service import ModelTrackerService
from openaiproxy.services.proxy_service import ProxyService
from openaiproxy.services.responses_service import ResponsesService
from openaiproxy.services.revision_service import RevisionService
from openaiproxy.services.slot_cache_service import SlotAllocator, SlotCacheService
from openaiproxy.services.trace_index_service import TraceIndexService
from openaiproxy.services.trace_service import TraceService
from openaiproxy.services.validation_service import ValidationService
from openaiproxy.services.web_search_service import WebSearchService


def create_app(
    config_repository: LLMProxyConfigRepository,
    logs_dir: Path,
    trace_dir: Path,
    frontend_dir: Path | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.trace_index_service.start()
        try:
            yield
        finally:
            application.state.trace_index_service.stop()

    app = FastAPI(title="OpenAI Proxy", lifespan=lifespan)
    logger = configure_logging(Path(logs_dir))

    # Add exception logging middleware first to catch all exceptions
    app.add_middleware(ExceptionLoggingMiddleware, logger=logger)
    
    model_tracker = ModelTrackerService(logger=logger)
    app.state.model_tracker = model_tracker
    # shared between the proxy and responses paths so a session holds one slot
    slot_cache_service = SlotCacheService(logger=logger)
    slot_allocator = SlotAllocator(logger=logger)
    ledger_service = LedgerService(
        ledger_repository=FSLedgerRepository(trace_dir=Path(trace_dir)),
        validation_service=ValidationService(),
        logger=logger,
    )
    app.state.ledger_service = ledger_service
    app.state.proxy_service = ProxyService(
        config_repository=config_repository,
        trace_dir=Path(trace_dir),
        logger=logger,
        model_tracker=model_tracker,
        ledger_service=ledger_service,
        slot_cache_service=slot_cache_service,
        slot_allocator=slot_allocator,
    )
    web_search_service = WebSearchService(config_repository=config_repository, logger=logger)
    app.state.web_search_service = web_search_service
    app.state.responses_service = ResponsesService(
        config_repository=config_repository,
        trace_dir=Path(trace_dir),
        logger=logger,
        web_search_service=web_search_service,
        model_tracker=model_tracker,
        ledger_service=ledger_service,
        slot_cache_service=slot_cache_service,
        slot_allocator=slot_allocator,
    )
    app.state.anthropic_service = AnthropicService(
        config_repository=config_repository,
        trace_dir=Path(trace_dir),
        logger=logger,
        model_tracker=model_tracker,
        ledger_service=ledger_service,
    )
    app.state.embeddings_service = EmbeddingsService(
        config_repository=config_repository,
        trace_dir=Path(trace_dir),
        logger=logger,
        model_tracker=model_tracker,
        ledger_service=ledger_service,
    )
    app.state.config_service = ConfigService(config_repository=config_repository)
    # Traces on disk stay authoritative; the SQLite index is a rebuildable cache
    # that keeps listing off the filesystem entirely.
    trace_files = FSTraceRepository(trace_dir=Path(trace_dir))
    trace_index = SqliteTraceIndex(db_path=Path(trace_dir) / INDEX_FILENAME)
    trace_index_service = TraceIndexService(index=trace_index, files=trace_files, logger=logger)
    app.state.trace_index = trace_index
    app.state.trace_index_service = trace_index_service
    trace_repository = IndexedTraceRepository(
        index=trace_index, files=trace_files, ready=trace_index_service.ready
    )
    app.state.trace_service = TraceService(trace_repository=trace_repository)
    app.state.cost_service = CostService(
        trace_index=trace_index, config_service=app.state.config_service
    )
    app.state.revision_service = RevisionService(
        trace_index_service=trace_index_service, trace_dir=Path(trace_dir)
    )
    app.include_router(ui_router)
    app.include_router(main_router)
    if frontend_dir is not None and Path(frontend_dir).is_dir():
        app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")
    return app

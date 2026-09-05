from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from fastapi.staticfiles import StaticFiles

from openaiproxy.api.endpoints.router_main import router as main_router
from openaiproxy.api.endpoints.router_ui import router as ui_router
from openaiproxy.filesystem.fs_cost_ledger_repository import FSCostLedgerRepository
from openaiproxy.filesystem.fs_ledger_repository import FSLedgerRepository
from openaiproxy.filesystem.fs_trace_repository import FSTraceRepository
from openaiproxy.interface.repository import LLMProxyConfigRepository
from openaiproxy.logging_config import configure_logging
from openaiproxy.middleware.exception_logging import ExceptionLoggingMiddleware
from openaiproxy.services.anthropic_service import AnthropicService
from openaiproxy.services.config_service import ConfigService
from openaiproxy.services.cost_recorder_service import CostRecorderService
from openaiproxy.services.cost_service import CostService
from openaiproxy.services.embeddings_service import EmbeddingsService
from openaiproxy.services.ledger_service import LedgerService
from openaiproxy.services.model_tracker_service import ModelTrackerService
from openaiproxy.services.proxy_service import ProxyService
from openaiproxy.services.responses_service import ResponsesService
from openaiproxy.services.slot_cache_service import SlotAllocator, SlotCacheService
from openaiproxy.services.trace_service import TraceService
from openaiproxy.services.validation_service import ValidationService
from openaiproxy.services.web_search_service import WebSearchService


def create_app(
    config_repository: LLMProxyConfigRepository,
    logs_dir: Path,
    trace_dir: Path,
    frontend_dir: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="OpenAI Proxy")
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
    cost_ledger_repository = FSCostLedgerRepository(trace_dir=Path(trace_dir))
    cost_recorder_service = CostRecorderService(
        cost_ledger_repository=cost_ledger_repository,
        logger=logger,
    )
    app.state.cost_recorder_service = cost_recorder_service
    app.state.proxy_service = ProxyService(
        config_repository=config_repository,
        trace_dir=Path(trace_dir),
        logger=logger,
        model_tracker=model_tracker,
        ledger_service=ledger_service,
        cost_recorder=cost_recorder_service,
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
        cost_recorder=cost_recorder_service,
        slot_cache_service=slot_cache_service,
        slot_allocator=slot_allocator,
    )
    app.state.anthropic_service = AnthropicService(
        config_repository=config_repository,
        trace_dir=Path(trace_dir),
        logger=logger,
        model_tracker=model_tracker,
        ledger_service=ledger_service,
        cost_recorder=cost_recorder_service,
    )
    app.state.embeddings_service = EmbeddingsService(
        config_repository=config_repository,
        trace_dir=Path(trace_dir),
        logger=logger,
        model_tracker=model_tracker,
        ledger_service=ledger_service,
        cost_recorder=cost_recorder_service,
    )
    app.state.config_service = ConfigService(config_repository=config_repository)
    trace_repository = FSTraceRepository(trace_dir=Path(trace_dir))
    app.state.trace_service = TraceService(trace_repository=trace_repository)
    app.state.cost_service = CostService(
        cost_ledger_repository=cost_ledger_repository, config_service=app.state.config_service
    )
    app.include_router(ui_router)
    app.include_router(main_router)
    if frontend_dir is not None and Path(frontend_dir).is_dir():
        app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")
    return app

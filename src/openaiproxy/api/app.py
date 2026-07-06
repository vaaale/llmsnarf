from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from fastapi.staticfiles import StaticFiles

from openaiproxy.api.endpoints.router_main import router as main_router
from openaiproxy.api.endpoints.router_ui import router as ui_router
from openaiproxy.filesystem.fs_trace_repository import FSTraceRepository
from openaiproxy.interface.repository import LLMProxyConfigRepository
from openaiproxy.logging_config import configure_logging
from openaiproxy.middleware.exception_logging import ExceptionLoggingMiddleware
from openaiproxy.services.config_service import ConfigService
from openaiproxy.services.proxy_service import ProxyService
from openaiproxy.services.trace_service import TraceService


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
    
    app.state.proxy_service = ProxyService(config_repository=config_repository, trace_dir=Path(trace_dir), logger=logger)
    app.state.config_service = ConfigService(config_repository=config_repository)
    app.state.trace_service = TraceService(trace_repository=FSTraceRepository(trace_dir=Path(trace_dir)))
    app.include_router(ui_router)
    app.include_router(main_router)
    if frontend_dir is not None and Path(frontend_dir).is_dir():
        app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")
    return app

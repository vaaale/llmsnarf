from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from openaiproxy.api.endpoints.router_main import router as main_router
from openaiproxy.interface.repository import LLMProxyConfigRepository
from openaiproxy.logging_config import configure_logging
from openaiproxy.services.proxy_service import ProxyService


def create_app(config_repository: LLMProxyConfigRepository, logs_dir: Path, trace_dir: Path) -> FastAPI:
    app = FastAPI(title="OpenAI Proxy")
    logger = configure_logging(Path(logs_dir))
    app.state.proxy_service = ProxyService(config_repository=config_repository, trace_dir=Path(trace_dir), logger=logger)
    app.include_router(main_router)
    return app

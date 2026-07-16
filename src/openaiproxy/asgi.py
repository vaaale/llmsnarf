from __future__ import annotations

import logging

from openaiproxy.bootstrap import resolve_logs_trace_dirs
from openaiproxy.api.app import create_app
from openaiproxy.settings import Settings
from openaiproxy.yaml_repository.yaml_repository import YAMLLLMProxyConfigRepository

logger = logging.getLogger("openaiproxy")

try:
    settings = Settings()
    config_repository = YAMLLLMProxyConfigRepository(settings.config_file_path)
    logs_dir, trace_dir = resolve_logs_trace_dirs(settings, config_repository)
    logs_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    app = create_app(
        config_repository=config_repository,
        logs_dir=logs_dir,
        trace_dir=trace_dir,
        frontend_dir=settings.frontend_dir,
    )

    _config = config_repository.load()
    logger.info(
        "asgi startup config_file=%s logs_dir=%s trace_dir=%s endpoints=%s",
        settings.config_file_path,
        logs_dir,
        trace_dir,
        [e.name for e in _config.endpoints],
    )
    if not _config.endpoints:
        logger.warning(
            "asgi startup no endpoints configured in %s - all proxy requests will fail with 500",
            settings.config_file_path,
        )
except Exception:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("openaiproxy").exception("Fatal error during ASGI application startup")
    raise

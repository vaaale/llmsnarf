from __future__ import annotations

from pathlib import Path

from openaiproxy.interface.repository import LLMProxyConfigRepository
from openaiproxy.settings import Settings


def resolve_listen_host_port(settings: Settings, config_repository: LLMProxyConfigRepository) -> tuple[str, int]:
    config = config_repository.load()

    host = settings.host or config.host or "0.0.0.0"
    port = settings.port or config.port or 8000

    return host, port


def resolve_logs_trace_dirs(settings: Settings, config_repository: LLMProxyConfigRepository) -> tuple[Path, Path]:
    config = config_repository.load()

    logs_dir = settings.logs_dir or Path(config.logs_dir) if config.logs_dir else Path("./logs")
    trace_dir = settings.trace_dir or Path(config.trace_dir) if config.trace_dir else Path("./logs")

    return Path(logs_dir), Path(trace_dir)

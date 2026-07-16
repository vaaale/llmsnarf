from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FILENAME = "routing.log"

_LEVEL_NAMES = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

# Uvicorn/Starlette loggers whose records we also want in our file/console handlers.
_PROPAGATED_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


def _resolve_level(level: int | str | None) -> int:
    """Resolve the effective log level, honouring the LOG_LEVEL env var.

    An explicit ``level`` argument wins; otherwise the ``LOG_LEVEL`` environment
    variable is consulted (useful for tuning verbosity in a docker-compose
    deployment) before falling back to INFO.
    """
    if isinstance(level, int):
        return level
    candidate = level or os.getenv("LOG_LEVEL")
    if candidate is None:
        return logging.INFO
    return _LEVEL_NAMES.get(str(candidate).strip().upper(), logging.INFO)


def configure_logging(logs_dir: Path, level: int | str | None = None) -> logging.Logger:
    """Configure application logging with both console and rotating-file output.

    Logs are written to ``<logs_dir>/routing.log`` and to stdout so they are
    visible via ``docker compose logs`` as well as persisted on the mounted
    ``logs/`` volume. Uvicorn's own loggers are routed through the same handlers
    so startup errors and access logs share one destination.
    """
    resolved_level = _resolve_level(level)

    logger = logging.getLogger("openaiproxy")
    logger.setLevel(resolved_level)

    if logger.handlers:
        # Already configured (e.g. reload / repeated create_app); just sync level.
        logger.setLevel(resolved_level)
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s [%(module)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(resolved_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_file = Path(logs_dir) / LOG_FILENAME
    file_error: Exception | None = None
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        # Never let a bad/read-only logs volume take the whole app down: keep
        # console logging and make the failure loud instead of silent.
        file_error = exc

    # Route uvicorn's loggers through our handlers so their output also lands in
    # the log file and shares the same format.
    for name in _PROPAGATED_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = list(logger.handlers)
        uvicorn_logger.setLevel(resolved_level)
        uvicorn_logger.propagate = False

    logger.propagate = False

    if file_error is not None:
        logger.error(
            "Failed to open log file %s (%s); continuing with console logging only",
            log_file,
            file_error,
        )
    else:
        logger.info(
            "Logging initialized level=%s console=stdout file=%s",
            logging.getLevelName(resolved_level),
            log_file,
        )

    return logger


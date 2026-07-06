from __future__ import annotations

import logging
import traceback
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class ExceptionLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all exceptions with full stack traces."""
    
    def __init__(self, app, logger: logging.Logger | None = None):
        super().__init__(app)
        self.logger = logger or logging.getLogger("openaiproxy")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            # Log the full exception with stack trace
            self.logger.error(
                "Unhandled exception in request %s %s: %s",
                request.method,
                request.url,
                str(exc),
                exc_info=True
            )
            
            # Also log the traceback for clarity
            self.logger.error("Exception traceback:\n%s", traceback.format_exc())
            
            # Re-raise the exception so FastAPI can handle it and return 500
            raise

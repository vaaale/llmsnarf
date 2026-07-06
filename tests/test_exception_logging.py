from __future__ import annotations

import logging
import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from starlette.responses import PlainTextResponse

from openaiproxy.middleware.exception_logging import ExceptionLoggingMiddleware


class TestExceptionLoggingMiddleware:
    """Test suite for ExceptionLoggingMiddleware."""

    def test_exception_logging_middleware_logs_exception_with_stack_trace(self):
        """Test that exceptions are logged with full stack traces."""
        # Create a logger that captures log output
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.ERROR)
        
        logger = logging.getLogger("test_exception_logging")
        logger.setLevel(logging.ERROR)
        logger.handlers.clear()
        logger.addHandler(handler)
        
        # Create FastAPI app with middleware
        app = FastAPI()
        app.add_middleware(ExceptionLoggingMiddleware, logger=logger)
        
        # Add a route that raises an exception
        @app.get("/test-error")
        async def test_error():
            raise ValueError("Test exception for logging")
        
        # Make request to trigger exception - TestClient will raise the exception
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test-error")
        
        # The response should be 500 due to the exception
        assert response.status_code == 500
        
        # Check that the exception was logged
        log_output = log_stream.getvalue()
        assert "Unhandled exception in request GET http://testserver/test-error" in log_output
        assert "Test exception for logging" in log_output
        assert "Traceback" in log_output
        assert "ValueError" in log_output

    def test_exception_logging_middleware_uses_default_logger_when_none_provided(self):
        """Test that middleware uses default logger when none is provided."""
        app = FastAPI()
        app.add_middleware(ExceptionLoggingMiddleware)  # No logger provided
        
        @app.get("/test-error")
        async def test_error():
            raise RuntimeError("Test error")
        
        client = TestClient(app)
        with pytest.raises(Exception):
            client.get("/test-error")

    def test_exception_logging_middleware_passes_through_success(self):
        """Test that middleware allows normal requests to pass through."""
        app = FastAPI()
        app.add_middleware(ExceptionLoggingMiddleware)
        
        @app.get("/test-success")
        async def test_success():
            return PlainTextResponse("Success")
        
        client = TestClient(app)
        response = client.get("/test-success")
        assert response.status_code == 200
        assert response.text == "Success"

    @pytest.mark.asyncio
    async def test_exception_logging_middleware_async_dispatch(self):
        """Test middleware dispatch with async call_next."""
        # Create mock request and response
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "https://example.com/test"
        
        mock_response = MagicMock(spec=Response)
        
        # Create async mock for call_next
        call_next = AsyncMock(return_value=mock_response)
        
        # Create middleware with test logger
        logger = logging.getLogger("test_async")
        logger.addHandler(logging.NullHandler())
        
        middleware = ExceptionLoggingMiddleware(app=None, logger=logger)
        
        # Test successful dispatch
        result = await middleware.dispatch(mock_request, call_next)
        assert result == mock_response
        call_next.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_exception_logging_middleware_async_dispatch_with_exception(self):
        """Test middleware dispatch with exception in call_next."""
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = "https://example.com/test"
        
        # Create async mock that raises exception
        test_exception = ValueError("Async test exception")
        call_next = AsyncMock(side_effect=test_exception)
        
        # Create middleware with test logger that captures output
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.ERROR)
        
        logger = logging.getLogger("test_async_exception")
        logger.setLevel(logging.ERROR)
        logger.handlers.clear()
        logger.addHandler(handler)
        
        middleware = ExceptionLoggingMiddleware(app=None, logger=logger)
        
        # Test dispatch with exception
        with pytest.raises(ValueError, match="Async test exception"):
            await middleware.dispatch(mock_request, call_next)
        
        # Check that exception was logged
        log_output = log_stream.getvalue()
        assert "Unhandled exception in request POST https://example.com/test" in log_output
        assert "Async test exception" in log_output
        assert "Traceback" in log_output

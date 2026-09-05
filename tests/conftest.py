import json
from pathlib import Path

import pytest


def _write_trace(
    directory: Path,
    trace_id: str,
    *,
    model: str = "gpt-5.1",
    endpoint: str = "/chat/completions",
    status_code: int = 200,
    correlation_id: str | None = None,
    parent_trace_id: str | None = None,
    with_response: bool = True,
    timestamp: str = "2026-07-06T11:18:32.127209",
) -> Path:
    """Write a request/response trace pair the way the proxy does."""
    directory.mkdir(parents=True, exist_ok=True)
    request_path = directory / f"{trace_id}_request.json"
    request_path.write_text(json.dumps({
        "timestamp": timestamp,
        "endpoint": endpoint,
        "correlation_id": correlation_id,
        "parent_trace_id": parent_trace_id,
        "provider": "local",
        "headers": {},
        "payload": {"model": model, "messages": [{"role": "user", "content": "hi"}]},
    }))
    if with_response:
        (directory / f"{trace_id}_response.json").write_text(json.dumps({
            "timestamp": "2026-07-06T11:18:33.127209",
            "status_code": status_code,
            "headers": {},
            "body": {"usage": {"prompt_tokens": 7, "completion_tokens": 3}},
        }))
    return request_path


@pytest.fixture
def write_trace():
    return _write_trace

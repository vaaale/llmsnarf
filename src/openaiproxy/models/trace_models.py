from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

_TRACE_CATEGORY_BY_ENDPOINT = {
    "/chat/completions": "completion",
    "/completions": "completion",
    "/responses": "responses",
    "/messages": "messages",
    "/embeddings": "embeddings",
}


def trace_category(endpoint_path: str) -> str:
    """Folder under trace_dir where traces for an inbound API are stored."""
    path = endpoint_path or ""
    for prefix, category in _TRACE_CATEGORY_BY_ENDPOINT.items():
        if path == prefix or path.startswith(prefix + "/"):
            return category
    cleaned = path.strip("/").replace("/", "_").replace("\\", "_")
    return cleaned or "other"


@dataclass(frozen=True)
class TraceSummary:
    id: str
    timestamp: str
    api_key: str
    endpoint: str
    model: str | None
    stream: bool
    status_code: int | None
    duration_ms: float | None
    message_count: int
    error: str | None
    correlation_id: str | None = None
    parent_trace_id: str | None = None
    provider: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


def _usage_from_dict(usage: Any) -> tuple[int | None, int | None]:
    if not isinstance(usage, dict):
        return None, None
    input_tokens = usage.get("input_tokens")
    if input_tokens is None:
        input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("output_tokens")
    if output_tokens is None:
        output_tokens = usage.get("completion_tokens")
    return (
        int(input_tokens) if isinstance(input_tokens, (int, float)) else None,
        int(output_tokens) if isinstance(output_tokens, (int, float)) else None,
    )


def _usage_from_timings(timings: Any) -> tuple[int | None, int | None]:
    """llama.cpp servers report token counts under "timings" (prompt_n/predicted_n)
    instead of an OpenAI-style "usage" object, even when stream_options.include_usage
    is requested."""
    if not isinstance(timings, dict):
        return None, None
    prompt_n = timings.get("prompt_n")
    predicted_n = timings.get("predicted_n")
    return (
        int(prompt_n) if isinstance(prompt_n, (int, float)) else None,
        int(predicted_n) if isinstance(predicted_n, (int, float)) else None,
    )


def extract_usage(response_body: Any, response_chunks: list[str] | None) -> tuple[int | None, int | None]:
    """Best-effort extraction of (input_tokens, output_tokens) from a trace's response.

    Supports non-streaming OpenAI chat completions (prompt_tokens/completion_tokens),
    the Responses API and Anthropic messages (input_tokens/output_tokens), as well as
    their streaming SSE equivalents (a trailing usage-bearing chunk, or Anthropic's
    message_start/message_delta events). Falls back to llama.cpp's "timings"
    (prompt_n/predicted_n) when no "usage" object is present.
    """
    if isinstance(response_body, dict):
        input_tokens, output_tokens = _usage_from_dict(response_body.get("usage"))
        if input_tokens is not None or output_tokens is not None:
            return input_tokens, output_tokens
        input_tokens, output_tokens = _usage_from_timings(response_body.get("timings"))
        if input_tokens is not None or output_tokens is not None:
            return input_tokens, output_tokens

    if not response_chunks:
        return None, None

    input_tokens: int | None = None
    output_tokens: int | None = None
    timings_input: int | None = None
    timings_output: int | None = None
    for chunk in response_chunks:
        for line in chunk.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue

            event_usage = event.get("usage")
            if isinstance(event_usage, dict):
                it, ot = _usage_from_dict(event_usage)
                if it is not None:
                    input_tokens = it
                if ot is not None:
                    output_tokens = ot
                continue

            event_type = event.get("type")
            if event_type == "message_start":
                message = event.get("message") or {}
                it, ot = _usage_from_dict(message.get("usage"))
                if it is not None:
                    input_tokens = it
                if ot is not None:
                    output_tokens = ot
            elif event_type == "message_delta":
                _, ot = _usage_from_dict(event.get("usage"))
                if ot is not None:
                    output_tokens = ot

            it, ot = _usage_from_timings(event.get("timings"))
            if it is not None:
                timings_input = it
            if ot is not None:
                timings_output = ot

    if input_tokens is None:
        input_tokens = timings_input
    if output_tokens is None:
        output_tokens = timings_output

    return input_tokens, output_tokens


@dataclass(frozen=True)
class TraceDetail:
    summary: TraceSummary
    request_headers: dict[str, str] = field(default_factory=dict)
    request_payload: Any = None
    response_headers: dict[str, str] = field(default_factory=dict)
    response_body: Any = None
    response_chunks: list[str] = field(default_factory=list)
    children: list["TraceDetail"] = field(default_factory=list)

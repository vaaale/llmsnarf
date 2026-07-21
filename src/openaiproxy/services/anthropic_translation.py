"""Bidirectional translation between the Anthropic Messages API and the
OpenAI Chat Completions API.

Four request/response translators cover the non-streaming cases and two
incremental translator classes cover server-sent-event streams:

    anthropic_request_to_chat   Anthropic client  -> OpenAI upstream
    chat_response_to_anthropic  OpenAI upstream   -> Anthropic client
    chat_request_to_anthropic   OpenAI client     -> Anthropic upstream
    anthropic_response_to_chat  Anthropic upstream -> OpenAI client
    ChatToAnthropicStream       OpenAI SSE chunks -> Anthropic SSE events
    AnthropicToChatStream       Anthropic SSE events -> OpenAI SSE chunks
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

ANTHROPIC_VERSION = "2023-06-01"
# Anthropic requires max_tokens; used when the OpenAI-side request omits it.
DEFAULT_MAX_TOKENS = 4096

_STOP_REASON_TO_FINISH_REASON = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}

_FINISH_REASON_TO_STOP_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def stop_reason_to_finish_reason(stop_reason: str | None) -> str | None:
    if stop_reason is None:
        return None
    return _STOP_REASON_TO_FINISH_REASON.get(stop_reason, "stop")


def finish_reason_to_stop_reason(finish_reason: str | None) -> str | None:
    if finish_reason is None:
        return None
    return _FINISH_REASON_TO_STOP_REASON.get(finish_reason, "end_turn")


def _parse_json_or_empty(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


# ---------------------------------------------------------------------------
# Anthropic request -> OpenAI chat request
# ---------------------------------------------------------------------------


def _system_to_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = [
            block.get("text", "")
            for block in system
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n\n".join(p for p in parts if p)
    return ""


def _tool_result_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    if content is None:
        return ""
    return json.dumps(content)


def _image_block_to_chat_part(block: dict) -> dict | None:
    source = block.get("source")
    if not isinstance(source, dict):
        return None
    source_type = source.get("type")
    if source_type == "base64":
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
    if source_type == "url":
        return {"type": "image_url", "image_url": {"url": source.get("url", "")}}
    return None


def _anthropic_message_to_chat(message: dict) -> list[dict]:
    role = message.get("role", "user")
    content = message.get("content")

    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        return [{"role": role, "content": ""}]

    if role == "assistant":
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id") or _new_id("call"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    }
                )
            # thinking / redacted_thinking blocks carry no signal for chat backends
        chat_message: dict = {"role": "assistant", "content": "".join(text_parts) or None}
        if tool_calls:
            chat_message["tool_calls"] = tool_calls
        if chat_message["content"] is None and not tool_calls:
            return []
        return [chat_message]

    # user message: tool_result blocks become tool messages, the rest a user message
    tool_messages: list[dict] = []
    user_parts: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "tool_result":
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": _tool_result_content_to_text(block.get("content")),
                }
            )
        elif block_type == "text":
            user_parts.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "image":
            part = _image_block_to_chat_part(block)
            if part:
                user_parts.append(part)

    messages = tool_messages
    if user_parts:
        if all(p["type"] == "text" for p in user_parts):
            messages = messages + [
                {"role": role, "content": "\n".join(p["text"] for p in user_parts)}
            ]
        else:
            messages = messages + [{"role": role, "content": user_parts}]
    return messages


def translate_anthropic_tools_to_chat(tools: Any) -> list[dict]:
    chat_tools: list[dict] = []
    for tool in tools or []:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        fn: dict = {"name": tool["name"]}
        if tool.get("description"):
            fn["description"] = tool["description"]
        fn["parameters"] = tool.get("input_schema") or {"type": "object", "properties": {}}
        chat_tools.append({"type": "function", "function": fn})
    return chat_tools


def translate_anthropic_tool_choice_to_chat(tool_choice: Any) -> tuple[Any, bool | None]:
    """Returns (chat tool_choice, parallel_tool_calls or None)."""
    if not isinstance(tool_choice, dict):
        return None, None
    parallel: bool | None = None
    if tool_choice.get("disable_parallel_tool_use") is True:
        parallel = False
    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto", parallel
    if choice_type == "any":
        return "required", parallel
    if choice_type == "none":
        return "none", parallel
    if choice_type == "tool":
        return {"type": "function", "function": {"name": tool_choice.get("name", "")}}, parallel
    return None, parallel


def anthropic_request_to_chat(payload: dict) -> dict:
    messages: list[dict] = []
    system_text = _system_to_text(payload.get("system"))
    if system_text:
        messages.append({"role": "system", "content": system_text})
    for message in payload.get("messages") or []:
        if isinstance(message, dict):
            messages.extend(_anthropic_message_to_chat(message))

    chat: dict = {"model": payload.get("model"), "messages": messages}

    if payload.get("max_tokens") is not None:
        chat["max_tokens"] = payload["max_tokens"]
    for key in ("temperature", "top_p"):
        if payload.get(key) is not None:
            chat[key] = payload[key]
    if payload.get("stop_sequences"):
        chat["stop"] = payload["stop_sequences"]
    if payload.get("stream") is not None:
        chat["stream"] = payload["stream"]
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata.get("user_id"):
        chat["user"] = metadata["user_id"]

    chat_tools = translate_anthropic_tools_to_chat(payload.get("tools"))
    if chat_tools:
        chat["tools"] = chat_tools
        tool_choice, parallel = translate_anthropic_tool_choice_to_chat(payload.get("tool_choice"))
        if tool_choice is not None:
            chat["tool_choice"] = tool_choice
        if parallel is not None:
            chat["parallel_tool_calls"] = parallel
    return chat


# ---------------------------------------------------------------------------
# OpenAI chat response -> Anthropic response
# ---------------------------------------------------------------------------


def _chat_usage_to_anthropic(usage: Any) -> dict:
    usage = usage if isinstance(usage, dict) else {}
    prompt_tokens = usage.get("prompt_tokens") or 0
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") or 0 if isinstance(details, dict) else 0
    return {
        "input_tokens": max(prompt_tokens - cached, 0),
        "output_tokens": usage.get("completion_tokens") or 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
    }


def chat_response_to_anthropic(data: dict, requested_model: str | None = None) -> dict:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}

    content: list[dict] = []
    reasoning_text = message.get("reasoning_content") or message.get("reasoning")
    if reasoning_text:
        content.append({"type": "thinking", "thinking": reasoning_text, "signature": ""})
    text = message.get("content")
    if isinstance(text, list):  # some backends return content as parts
        text = _chat_content_to_text(text)
    if text:
        content.append({"type": "text", "text": text})
    for tool_call in message.get("tool_calls") or []:
        fn = tool_call.get("function") or {}
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id") or _new_id("toolu"),
                "name": fn.get("name", ""),
                "input": _parse_json_or_empty(fn.get("arguments")),
            }
        )

    stop_reason = finish_reason_to_stop_reason(choice.get("finish_reason")) or "end_turn"
    # some OpenAI-compatible backends report finish_reason "stop" even when
    # tool calls are present; Anthropic agent loops key off tool_use
    if stop_reason == "end_turn" and message.get("tool_calls"):
        stop_reason = "tool_use"

    return {
        "id": _new_id("msg"),
        "type": "message",
        "role": "assistant",
        "model": requested_model or data.get("model") or "",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": _chat_usage_to_anthropic(data.get("usage")),
    }


# ---------------------------------------------------------------------------
# OpenAI chat request -> Anthropic request
# ---------------------------------------------------------------------------


def _chat_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "\n".join(p for p in parts if p)
    return ""


def _chat_image_part_to_anthropic(part: dict) -> dict | None:
    image_url = part.get("image_url")
    url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url or "")
    if not url:
        return None
    if url.startswith("data:"):
        try:
            header, data = url.split(",", 1)
            media_type = header[5:].split(";", 1)[0] or "image/png"
        except ValueError:
            return None
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _chat_user_content_to_blocks(content: Any) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        return []
    blocks: list[dict] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text":
            if part.get("text"):
                blocks.append({"type": "text", "text": part["text"]})
        elif part_type == "image_url":
            block = _chat_image_part_to_anthropic(part)
            if block:
                blocks.append(block)
    return blocks


def _append_blocks(messages: list[dict], role: str, blocks: list[dict]) -> None:
    """Append blocks, merging into the previous message when roles match.

    The Anthropic API requires user/assistant turns to alternate, while chat
    payloads may contain e.g. several consecutive tool messages.
    """
    if not blocks:
        return
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].extend(blocks)
    else:
        messages.append({"role": role, "content": blocks})


def translate_chat_tool_choice_to_anthropic(tool_choice: Any, parallel_tool_calls: Any) -> dict | None:
    choice: dict | None = None
    if tool_choice == "auto":
        choice = {"type": "auto"}
    elif tool_choice == "required":
        choice = {"type": "any"}
    elif tool_choice == "none":
        choice = {"type": "none"}
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = (tool_choice.get("function") or {}).get("name", "")
        choice = {"type": "tool", "name": name}
    if parallel_tool_calls is False:
        if choice is None:
            choice = {"type": "auto"}
        if choice["type"] in ("auto", "any", "tool"):
            choice["disable_parallel_tool_use"] = True
    return choice


def chat_request_to_anthropic(payload: dict) -> dict:
    system_parts: list[str] = []
    messages: list[dict] = []

    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role in ("system", "developer"):
            text = _chat_content_to_text(message.get("content"))
            if text:
                system_parts.append(text)
        elif role == "tool":
            _append_blocks(
                messages,
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id", ""),
                        "content": _chat_content_to_text(message.get("content")),
                    }
                ],
            )
        elif role == "assistant":
            blocks: list[dict] = []
            text = _chat_content_to_text(message.get("content"))
            if text:
                blocks.append({"type": "text", "text": text})
            for tool_call in message.get("tool_calls") or []:
                fn = tool_call.get("function") or {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.get("id") or _new_id("toolu"),
                        "name": fn.get("name", ""),
                        "input": _parse_json_or_empty(fn.get("arguments")),
                    }
                )
            _append_blocks(messages, "assistant", blocks)
        else:
            _append_blocks(messages, "user", _chat_user_content_to_blocks(message.get("content")))

    # Anthropic requires a non-empty conversation starting with a user turn
    if not messages or messages[0]["role"] != "user":
        messages.insert(0, {"role": "user", "content": [{"type": "text", "text": "."}]})

    max_tokens = payload.get("max_tokens") or payload.get("max_completion_tokens") or DEFAULT_MAX_TOKENS
    anthropic: dict = {
        "model": payload.get("model"),
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system_parts:
        anthropic["system"] = "\n\n".join(system_parts)
    if payload.get("temperature") is not None:
        # Anthropic accepts 0..1 while OpenAI accepts 0..2
        anthropic["temperature"] = min(float(payload["temperature"]), 1.0)
    if payload.get("top_p") is not None:
        anthropic["top_p"] = payload["top_p"]
    stop = payload.get("stop")
    if isinstance(stop, str):
        anthropic["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        anthropic["stop_sequences"] = [str(s) for s in stop]
    if payload.get("stream") is not None:
        anthropic["stream"] = payload["stream"]
    if payload.get("user"):
        anthropic["metadata"] = {"user_id": payload["user"]}

    tools: list[dict] = []
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        fn = tool.get("function") or {}
        if not fn.get("name"):
            continue
        anthropic_tool: dict = {
            "name": fn["name"],
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        }
        if fn.get("description"):
            anthropic_tool["description"] = fn["description"]
        tools.append(anthropic_tool)
    if tools:
        anthropic["tools"] = tools
        tool_choice = translate_chat_tool_choice_to_anthropic(
            payload.get("tool_choice"), payload.get("parallel_tool_calls")
        )
        if tool_choice is not None:
            anthropic["tool_choice"] = tool_choice
    return anthropic


# ---------------------------------------------------------------------------
# Anthropic response -> OpenAI chat response
# ---------------------------------------------------------------------------


def _anthropic_usage_to_chat(usage: Any) -> dict:
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = usage.get("input_tokens") or 0
    cache_read = usage.get("cache_read_input_tokens") or 0
    cache_creation = usage.get("cache_creation_input_tokens") or 0
    output_tokens = usage.get("output_tokens") or 0
    prompt_tokens = input_tokens + cache_read + cache_creation
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
        "prompt_tokens_details": {"cached_tokens": cache_read},
    }


def anthropic_response_to_chat(data: dict, requested_model: str | None = None) -> dict:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in data.get("content") or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "thinking":
            reasoning_parts.append(block.get("thinking", ""))
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id") or _new_id("call"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )

    message: dict = {"role": "assistant", "content": "".join(text_parts) or None}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model or data.get("model") or "",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": stop_reason_to_finish_reason(data.get("stop_reason")) or "stop",
            }
        ],
        "usage": _anthropic_usage_to_chat(data.get("usage")),
    }


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------

_STATUS_TO_ANTHROPIC_ERROR_TYPE = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    529: "overloaded_error",
}


def anthropic_error_body(status_code: int, message: str) -> dict:
    error_type = _STATUS_TO_ANTHROPIC_ERROR_TYPE.get(status_code, "api_error")
    return {"type": "error", "error": {"type": error_type, "message": message}}


def openai_error_body(status_code: int, message: str) -> dict:
    error_type = "invalid_request_error" if status_code < 500 else "api_error"
    return {"error": {"message": message, "type": error_type, "code": None, "param": None}}


def extract_error_message(body: Any) -> str:
    """Best-effort extraction of a human readable message from an upstream
    error body in either OpenAI or Anthropic format."""
    if isinstance(body, (bytes, str)):
        try:
            body = json.loads(body)
        except Exception:
            return body.decode("utf-8", errors="replace") if isinstance(body, bytes) else body
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str):
            return error
    return json.dumps(body) if body is not None else "upstream error"


# ---------------------------------------------------------------------------
# Streaming: OpenAI chat SSE chunks -> Anthropic SSE events
# ---------------------------------------------------------------------------


def sse_event(event_type: str, data: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


class ChatToAnthropicStream:
    """Incrementally converts OpenAI chat completion chunks into Anthropic
    Messages SSE events. Feed parsed chunk dicts to process_chunk() and call
    finalize() when the upstream stream ends."""

    def __init__(self, model: str):
        self._model = model
        self._message_id = _new_id("msg")
        self._started = False
        self._block_index = -1
        self._block_type: str | None = None  # text | thinking | tool_use
        self._tool_index_to_block: dict[int, int] = {}
        self._finish_reason: str | None = None
        self._input_tokens = 0
        self._output_tokens = 0
        self._finalized = False

    def _start_events(self) -> list[bytes]:
        if self._started:
            return []
        self._started = True
        message = {
            "id": self._message_id,
            "type": "message",
            "role": "assistant",
            "model": self._model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
        return [sse_event("message_start", {"type": "message_start", "message": message})]

    def _close_block(self) -> list[bytes]:
        if self._block_type is None:
            return []
        events = [
            sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": self._block_index},
            )
        ]
        self._block_type = None
        return events

    def _open_block(self, block_type: str, content_block: dict) -> list[bytes]:
        events = self._close_block()
        self._block_index += 1
        self._block_type = block_type
        events.append(
            sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._block_index,
                    "content_block": content_block,
                },
            )
        )
        return events

    def process_chunk(self, chunk: dict) -> list[bytes]:
        events = self._start_events()

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            if usage.get("prompt_tokens"):
                self._input_tokens = usage["prompt_tokens"]
            if usage.get("completion_tokens"):
                self._output_tokens = usage["completion_tokens"]

        choices = chunk.get("choices") or []
        if not choices:
            return events
        choice = choices[0]
        if choice.get("finish_reason"):
            self._finish_reason = choice["finish_reason"]
        delta = choice.get("delta") or {}

        reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning_delta:
            if self._block_type != "thinking":
                events.extend(
                    self._open_block("thinking", {"type": "thinking", "thinking": "", "signature": ""})
                )
            events.append(
                sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._block_index,
                        "delta": {"type": "thinking_delta", "thinking": reasoning_delta},
                    },
                )
            )

        content_delta = delta.get("content")
        if content_delta:
            if self._block_type != "text":
                events.extend(self._open_block("text", {"type": "text", "text": ""}))
            events.append(
                sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._block_index,
                        "delta": {"type": "text_delta", "text": content_delta},
                    },
                )
            )

        for tc_delta in delta.get("tool_calls") or []:
            tool_index = tc_delta.get("index", 0)
            fn = tc_delta.get("function") or {}
            if tool_index not in self._tool_index_to_block:
                events.extend(
                    self._open_block(
                        "tool_use",
                        {
                            "type": "tool_use",
                            "id": tc_delta.get("id") or _new_id("toolu"),
                            "name": fn.get("name", ""),
                            "input": {},
                        },
                    )
                )
                self._tool_index_to_block[tool_index] = self._block_index
            if fn.get("arguments"):
                events.append(
                    sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": self._tool_index_to_block[tool_index],
                            "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]},
                        },
                    )
                )
        return events

    def finalize(self) -> list[bytes]:
        if self._finalized:
            return []
        self._finalized = True
        events = self._start_events()
        events.extend(self._close_block())
        stop_reason = finish_reason_to_stop_reason(self._finish_reason) or "end_turn"
        if stop_reason == "end_turn" and self._tool_index_to_block:
            stop_reason = "tool_use"
        events.append(
            sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {
                        "input_tokens": self._input_tokens,
                        "output_tokens": self._output_tokens,
                    },
                },
            )
        )
        events.append(sse_event("message_stop", {"type": "message_stop"}))
        return events

    def error(self, message: str) -> list[bytes]:
        return [
            sse_event(
                "error",
                {"type": "error", "error": {"type": "api_error", "message": message}},
            )
        ]


# ---------------------------------------------------------------------------
# Streaming: Anthropic SSE events -> OpenAI chat SSE chunks
# ---------------------------------------------------------------------------


class AnthropicToChatStream:
    """Incrementally converts Anthropic Messages SSE events into OpenAI chat
    completion SSE chunks. Feed parsed event dicts to process_event() and call
    finalize() when the upstream stream ends."""

    def __init__(self, model: str):
        self._model = model
        self._chat_id = f"chatcmpl-{uuid.uuid4().hex}"
        self._created = int(time.time())
        self._role_sent = False
        self._block_to_tool_index: dict[int, int] = {}
        self._next_tool_index = 0
        self._finish_reason: str | None = None
        self._usage: dict = {}
        self._done_sent = False

    def _chunk(self, delta: dict, finish_reason: str | None = None, usage: dict | None = None) -> bytes:
        body: dict = {
            "id": self._chat_id,
            "object": "chat.completion.chunk",
            "created": self._created,
            "model": self._model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            body["usage"] = usage
        return f"data: {json.dumps(body)}\n\n".encode("utf-8")

    def _role_chunk(self) -> list[bytes]:
        if self._role_sent:
            return []
        self._role_sent = True
        return [self._chunk({"role": "assistant", "content": ""})]

    def process_event(self, event: dict) -> list[bytes]:
        event_type = event.get("type")

        if event_type == "message_start":
            message = event.get("message") or {}
            usage = message.get("usage") or {}
            self._usage["input_tokens"] = usage.get("input_tokens") or 0
            self._usage["cache_read"] = usage.get("cache_read_input_tokens") or 0
            self._usage["cache_creation"] = usage.get("cache_creation_input_tokens") or 0
            return self._role_chunk()

        if event_type == "content_block_start":
            index = event.get("index", 0)
            block = event.get("content_block") or {}
            block_type = block.get("type", "text")
            chunks = self._role_chunk()
            if block_type == "tool_use":
                tool_index = self._next_tool_index
                self._next_tool_index += 1
                self._block_to_tool_index[index] = tool_index
                chunks.append(
                    self._chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": tool_index,
                                    "id": block.get("id") or _new_id("call"),
                                    "type": "function",
                                    "function": {"name": block.get("name", ""), "arguments": ""},
                                }
                            ]
                        }
                    )
                )
            return chunks

        if event_type == "content_block_delta":
            index = event.get("index", 0)
            delta = event.get("delta") or {}
            delta_type = delta.get("type")
            chunks = self._role_chunk()
            if delta_type == "text_delta" and delta.get("text"):
                chunks.append(self._chunk({"content": delta["text"]}))
            elif delta_type == "thinking_delta" and delta.get("thinking"):
                chunks.append(self._chunk({"reasoning_content": delta["thinking"]}))
            elif delta_type == "input_json_delta" and delta.get("partial_json"):
                tool_index = self._block_to_tool_index.get(index, 0)
                chunks.append(
                    self._chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": tool_index,
                                    "function": {"arguments": delta["partial_json"]},
                                }
                            ]
                        }
                    )
                )
            return chunks

        if event_type == "message_delta":
            delta = event.get("delta") or {}
            usage = event.get("usage") or {}
            if usage.get("output_tokens"):
                self._usage["output_tokens"] = usage["output_tokens"]
            if usage.get("input_tokens"):
                self._usage["input_tokens"] = usage["input_tokens"]
            stop_reason = delta.get("stop_reason")
            if stop_reason:
                self._finish_reason = stop_reason_to_finish_reason(stop_reason)
                return [self._chunk({}, finish_reason=self._finish_reason)]
            return []

        if event_type == "message_stop":
            return self.finalize()

        if event_type == "error":
            error = event.get("error") or {}
            message = str(error.get("message") or "upstream error")
            body = openai_error_body(500, message)
            self._done_sent = True
            return [
                f"data: {json.dumps(body)}\n\n".encode("utf-8"),
                b"data: [DONE]\n\n",
            ]

        # ping and unknown events are ignored
        return []

    def finalize(self) -> list[bytes]:
        if self._done_sent:
            return []
        self._done_sent = True
        chunks: list[bytes] = []
        if self._finish_reason is None:
            chunks.append(self._chunk({}, finish_reason="stop"))
        input_tokens = (
            (self._usage.get("input_tokens") or 0)
            + (self._usage.get("cache_read") or 0)
            + (self._usage.get("cache_creation") or 0)
        )
        output_tokens = self._usage.get("output_tokens") or 0
        usage = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "prompt_tokens_details": {"cached_tokens": self._usage.get("cache_read") or 0},
        }
        body = {
            "id": self._chat_id,
            "object": "chat.completion.chunk",
            "created": self._created,
            "model": self._model,
            "choices": [],
            "usage": usage,
        }
        chunks.append(f"data: {json.dumps(body)}\n\n".encode("utf-8"))
        chunks.append(b"data: [DONE]\n\n")
        return chunks


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------


def extract_client_credential(headers: dict[str, str]) -> str:
    """Extract the client's API credential from either auth scheme."""
    api_key = headers.get("x-api-key", "")
    if api_key:
        return api_key
    auth_header = headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    return ""


def build_anthropic_headers(client_headers: dict[str, str], endpoint_api_key: str) -> dict[str, str]:
    """Headers for a request to an Anthropic-protocol upstream. A native
    x-api-key from the client wins, then the configured endpoint key, then a
    bearer token from the client."""
    headers = {
        "content-type": "application/json",
        "anthropic-version": client_headers.get("anthropic-version", ANTHROPIC_VERSION),
    }
    key = client_headers.get("x-api-key") or endpoint_api_key or extract_client_credential(client_headers)
    if key:
        headers["x-api-key"] = key
    return headers


def build_openai_headers(client_headers: dict[str, str], endpoint_api_key: str) -> dict[str, str]:
    """Headers for a request to an OpenAI-protocol upstream. A bearer token
    from the client wins, then the configured endpoint key, then an x-api-key
    from the client."""
    headers = {"content-type": "application/json"}
    if client_headers.get("authorization", "").startswith("Bearer "):
        headers["Authorization"] = client_headers["authorization"]
    elif endpoint_api_key:
        headers["Authorization"] = f"Bearer {endpoint_api_key}"
    elif client_headers.get("x-api-key"):
        headers["Authorization"] = f"Bearer {client_headers['x-api-key']}"
    return headers

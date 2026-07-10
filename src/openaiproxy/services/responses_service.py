from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import HTTPException

from openaiproxy.interface.repository import LLMProxyConfigRepository
from openaiproxy.models.models import EndpointConfig
from openaiproxy.services.model_tracker_service import ModelTrackerService
from openaiproxy.services.proxy_service import (
    extract_api_key,
    filter_response_headers,
    save_request_trace,
    save_response_trace,
)
from openaiproxy.services.web_search_service import (
    WEB_FETCH_TOOL_DEFINITION,
    WEB_FETCH_TOOL_NAME,
    WEB_SEARCH_TOOL_DEFINITION,
    WEB_SEARCH_TOOL_NAME,
    WebSearchService,
)

MAX_TOOL_ITERATIONS = 5


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _content_part_to_chat(part: dict) -> dict | None:
    part_type = part.get("type")
    if part_type in ("input_text", "output_text", "text", "summary_text"):
        return {"type": "text", "text": part.get("text", "")}
    if part_type == "refusal":
        return {"type": "text", "text": part.get("refusal", "")}
    if part_type == "input_image":
        image_url = part.get("image_url")
        if isinstance(image_url, str):
            return {"type": "image_url", "image_url": {"url": image_url}}
        if isinstance(image_url, dict):
            return {"type": "image_url", "image_url": image_url}
    return None


def _message_item_to_chat(item: dict) -> dict:
    role = item.get("role", "user")
    content = item.get("content")
    if isinstance(content, str):
        return {"role": role, "content": content}
    if isinstance(content, list):
        parts = [p for p in (_content_part_to_chat(part) for part in content if isinstance(part, dict)) if p]
        if parts and all(p["type"] == "text" for p in parts):
            return {"role": role, "content": "\n".join(p["text"] for p in parts)}
        return {"role": role, "content": parts}
    return {"role": role, "content": ""}


def build_chat_messages(payload: dict) -> list[dict]:
    messages: list[dict] = []
    instructions = payload.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    input_value = payload.get("input")
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
    elif isinstance(input_value, list):
        for item in input_value:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type") or "message"
            if item_type == "message":
                messages.append(_message_item_to_chat(item))
            elif item_type == "function_call":
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": item.get("call_id") or item.get("id") or _new_id("call"),
                                "type": "function",
                                "function": {
                                    "name": item.get("name", ""),
                                    "arguments": item.get("arguments") or "{}",
                                },
                            }
                        ],
                    }
                )
            elif item_type == "function_call_output":
                output = item.get("output")
                if not isinstance(output, str):
                    output = json.dumps(output)
                messages.append({"role": "tool", "tool_call_id": item.get("call_id", ""), "content": output})
            # reasoning / web_search_call echoes from previous turns carry no context for chat backends
    return messages


def translate_tools(payload: dict, web_search_available: bool) -> tuple[list[dict], bool]:
    chat_tools: list[dict] = []
    wants_web_search = False
    client_function_names: set[str] = set()
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        if tool_type in ("web_search", "web_search_preview"):
            wants_web_search = True
        elif tool_type == "function":
            fn: dict = {"name": tool.get("name", "")}
            for key in ("description", "parameters", "strict"):
                if tool.get(key) is not None:
                    fn[key] = tool[key]
            client_function_names.add(fn["name"])
            chat_tools.append({"type": "function", "function": fn})
    web_search_active = (
        wants_web_search and web_search_available and WEB_SEARCH_TOOL_NAME not in client_function_names
    )
    if web_search_active:
        chat_tools.append(WEB_SEARCH_TOOL_DEFINITION)
        if WEB_FETCH_TOOL_NAME not in client_function_names:
            chat_tools.append(WEB_FETCH_TOOL_DEFINITION)
    return chat_tools, web_search_active


def translate_tool_choice(tool_choice: Any) -> Any:
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "function":
            return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
        if choice_type in ("web_search", "web_search_preview"):
            return {"type": "function", "function": {"name": WEB_SEARCH_TOOL_NAME}}
    return None


def translate_text_format(payload: dict) -> dict | None:
    text = payload.get("text")
    fmt = text.get("format") if isinstance(text, dict) else None
    if not isinstance(fmt, dict):
        return None
    fmt_type = fmt.get("type")
    if fmt_type == "json_object":
        return {"type": "json_object"}
    if fmt_type == "json_schema":
        json_schema = {key: fmt[key] for key in ("name", "schema", "strict") if key in fmt}
        return {"type": "json_schema", "json_schema": json_schema}
    return None


def build_chat_payload(payload: dict, forwarded_model: Any, chat_tools: list[dict]) -> dict:
    chat: dict = {"model": forwarded_model, "messages": build_chat_messages(payload)}
    if chat_tools:
        chat["tools"] = chat_tools
        tool_choice = translate_tool_choice(payload.get("tool_choice"))
        if tool_choice is not None:
            chat["tool_choice"] = tool_choice
        if payload.get("parallel_tool_calls") is not None:
            chat["parallel_tool_calls"] = payload["parallel_tool_calls"]
    for source, target in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("max_output_tokens", "max_tokens"),
        ("user", "user"),
    ):
        if payload.get(source) is not None:
            chat[target] = payload[source]
    reasoning = payload.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        chat["reasoning_effort"] = reasoning["effort"]
    response_format = translate_text_format(payload)
    if response_format:
        chat["response_format"] = response_format
    return chat


def _base_response(payload: dict, response_id: str, model: str) -> dict:
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "in_progress",
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": payload.get("instructions"),
        "max_output_tokens": payload.get("max_output_tokens"),
        "model": model,
        "output": [],
        "parallel_tool_calls": payload.get("parallel_tool_calls", True),
        "previous_response_id": None,
        "reasoning": payload.get("reasoning"),
        "store": bool(payload.get("store", False)),
        "temperature": payload.get("temperature"),
        "text": payload.get("text"),
        "tool_choice": payload.get("tool_choice", "auto"),
        "tools": payload.get("tools") or [],
        "top_p": payload.get("top_p"),
        "truncation": payload.get("truncation") or "disabled",
        "usage": None,
        "user": payload.get("user"),
        "metadata": payload.get("metadata") or {},
    }


def _empty_usage() -> dict:
    return {
        "input_tokens": 0,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 0,
    }


def _accumulate_usage(total: dict, chat_usage: Any) -> None:
    if not isinstance(chat_usage, dict):
        return
    total["input_tokens"] += chat_usage.get("prompt_tokens") or 0
    total["output_tokens"] += chat_usage.get("completion_tokens") or 0
    total["total_tokens"] += chat_usage.get("total_tokens") or 0


def _parse_search_query(tool_call: dict) -> str:
    arguments = (tool_call.get("function") or {}).get("arguments") or "{}"
    try:
        args = json.loads(arguments)
        return str(args.get("query") or "")
    except Exception:
        return str(arguments)


def _parse_fetch_url(tool_call: dict) -> str:
    arguments = (tool_call.get("function") or {}).get("arguments") or "{}"
    try:
        args = json.loads(arguments)
        return str(args.get("url") or "")
    except Exception:
        return str(arguments)


def _reset_forced_tool_choice(chat_payload: dict) -> None:
    if isinstance(chat_payload.get("tool_choice"), dict) or chat_payload.get("tool_choice") == "required":
        chat_payload["tool_choice"] = "auto"


class ResponsesService:
    def __init__(
        self,
        config_repository: LLMProxyConfigRepository,
        trace_dir: Path,
        logger: logging.Logger,
        web_search_service: WebSearchService,
        model_tracker: ModelTrackerService,
    ):
        self._config_repository = config_repository
        self._trace_dir = trace_dir
        self._logger = logger
        self._web_search_service = web_search_service
        self._model_tracker = model_tracker

    def _select_endpoint_for_model(self, model: str | None) -> EndpointConfig | None:
        config = self._config_repository.load()
        endpoints = [endpoint for endpoint in config.endpoints if endpoint.enabled]

        if model:
            for endpoint in endpoints:
                if model in endpoint.models or model in endpoint.aliases:
                    return endpoint

        for endpoint in endpoints:
            if "*" in endpoint.models:
                return endpoint

        return None

    async def handle(self, body: bytes, headers: dict[str, str]) -> dict:
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = None
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail={"error": "Request body must be a JSON object"})

        requested_model = payload.get("model")
        endpoint = self._select_endpoint_for_model(requested_model)
        if not endpoint:
            config = self._config_repository.load()
            if not config.endpoints:
                raise HTTPException(status_code=500, detail={"error": "No llmproxy routes configured"})
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "message": "No route configured for requested model",
                        "type": "routing_error",
                        "model": requested_model,
                    }
                },
            )

        forwarded_model = endpoint.aliases.get(str(requested_model), requested_model)

        web_search_config = self._web_search_service.get_config()
        chat_tools, web_search_active = translate_tools(payload, web_search_config.enabled)
        chat_payload = build_chat_payload(payload, forwarded_model, chat_tools)

        substitutions = endpoint.substitute_role or {}
        for message in chat_payload["messages"]:
            if message["role"] in substitutions:
                message["role"] = substitutions[message["role"]]

        self._logger.info(
            "responses_route requested_model=%s forwarded_model=%s endpoint=%s web_search=%s stream=%s",
            requested_model,
            forwarded_model,
            endpoint.name,
            web_search_active,
            bool(payload.get("stream")),
        )

        if forwarded_model is not None:
            await self._model_tracker.ensure_capacity(endpoint, str(forwarded_model))

        should_log = bool(endpoint.log)
        api_key = extract_api_key(headers) if "authorization" in headers else (endpoint.api_key or "unknown")
        base_filename = (
            await save_request_trace(self._trace_dir, "/responses", payload, headers, api_key) if should_log else ""
        )

        forward_headers = {"Content-Type": "application/json"}
        if "authorization" in headers:
            forward_headers["Authorization"] = headers["authorization"]
        elif endpoint.api_key:
            forward_headers["Authorization"] = f"Bearer {endpoint.api_key}"

        chat_url = f"{endpoint.base_url}/chat/completions"

        if payload.get("stream"):
            return {
                "type": "stream",
                "iterator": self._stream(
                    payload, chat_payload, chat_url, forward_headers, web_search_active, should_log, base_filename
                ),
            }

        return await self._complete(
            payload, chat_payload, chat_url, forward_headers, web_search_active, should_log, base_filename
        )

    async def _trace_response(self, should_log: bool, base_filename: str, response_data: dict) -> None:
        if should_log:
            await save_response_trace(self._trace_dir, base_filename, response_data)

    async def _complete(
        self,
        payload: dict,
        chat_payload: dict,
        chat_url: str,
        forward_headers: dict[str, str],
        web_search_active: bool,
        should_log: bool,
        base_filename: str,
    ) -> dict:
        response_id = _new_id("resp")
        result = _base_response(payload, response_id, str(payload.get("model") or chat_payload["model"]))
        usage = _empty_usage()
        output: list[dict] = []
        status = "completed"
        incomplete_reason: str | None = None

        async with httpx.AsyncClient(timeout=300.0) as client:
            for _ in range(MAX_TOOL_ITERATIONS):
                try:
                    upstream = await client.post(chat_url, headers=forward_headers, json=chat_payload)
                except Exception as exc:
                    await self._trace_response(
                        should_log, base_filename, {"timestamp": datetime.now().isoformat(), "error": str(exc)}
                    )
                    return {
                        "type": "response",
                        "content": json.dumps({"error": str(exc)}).encode("utf-8"),
                        "status_code": 502,
                        "headers": {"content-type": "application/json"},
                    }

                if upstream.status_code != 200:
                    await self._trace_response(
                        should_log,
                        base_filename,
                        {
                            "timestamp": datetime.now().isoformat(),
                            "status_code": upstream.status_code,
                            "headers": dict(upstream.headers),
                            "body": upstream.text,
                        },
                    )
                    return {
                        "type": "response",
                        "content": upstream.content,
                        "status_code": upstream.status_code,
                        "headers": filter_response_headers(dict(upstream.headers)),
                    }

                data = upstream.json()
                _accumulate_usage(usage, data.get("usage"))
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}

                reasoning_text = message.get("reasoning_content") or message.get("reasoning")
                if reasoning_text:
                    output.append(
                        {
                            "type": "reasoning",
                            "id": _new_id("rs"),
                            "summary": [{"type": "summary_text", "text": reasoning_text}],
                        }
                    )

                content = message.get("content")
                if content:
                    output.append(
                        {
                            "type": "message",
                            "id": _new_id("msg"),
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": content, "annotations": []}],
                        }
                    )

                tool_calls = message.get("tool_calls") or []
                proxy_tool_names = {WEB_SEARCH_TOOL_NAME, WEB_FETCH_TOOL_NAME}
                web_search_calls = [
                    tc
                    for tc in tool_calls
                    if web_search_active and (tc.get("function") or {}).get("name") in proxy_tool_names
                ]
                client_calls = [tc for tc in tool_calls if tc not in web_search_calls]

                if web_search_calls and not client_calls:
                    chat_payload["messages"].append(message)
                    for tool_call in web_search_calls:
                        fn_name = (tool_call.get("function") or {}).get("name")
                        if fn_name == WEB_FETCH_TOOL_NAME:
                            fetch_url = _parse_fetch_url(tool_call)
                            output.append(
                                {
                                    "type": "web_search_call",
                                    "id": _new_id("ws"),
                                    "status": "completed",
                                    "action": {"type": "fetch", "url": fetch_url},
                                }
                            )
                            fetch_result = await self._web_search_service.fetch(fetch_url)
                            chat_payload["messages"].append(
                                {"role": "tool", "tool_call_id": tool_call.get("id"), "content": fetch_result}
                            )
                        else:
                            query = _parse_search_query(tool_call)
                            output.append(
                                {
                                    "type": "web_search_call",
                                    "id": _new_id("ws"),
                                    "status": "completed",
                                    "action": {"type": "search", "query": query},
                                }
                            )
                            search_result = await self._web_search_service.search(query)
                            chat_payload["messages"].append(
                                {"role": "tool", "tool_call_id": tool_call.get("id"), "content": search_result}
                            )
                    _reset_forced_tool_choice(chat_payload)
                    continue

                for tool_call in client_calls:
                    fn = tool_call.get("function") or {}
                    output.append(
                        {
                            "type": "function_call",
                            "id": _new_id("fc"),
                            "call_id": tool_call.get("id") or _new_id("call"),
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments") or "{}",
                            "status": "completed",
                        }
                    )

                if choice.get("finish_reason") == "length":
                    status = "incomplete"
                    incomplete_reason = "max_output_tokens"
                break
            else:
                status = "incomplete"
                incomplete_reason = "max_tool_iterations"

        result["status"] = status
        if incomplete_reason:
            result["incomplete_details"] = {"reason": incomplete_reason}
        result["output"] = output
        result["usage"] = usage

        await self._trace_response(
            should_log,
            base_filename,
            {
                "timestamp": datetime.now().isoformat(),
                "status_code": 200,
                "headers": {"content-type": "application/json"},
                "body": result,
            },
        )

        return {
            "type": "response",
            "content": json.dumps(result).encode("utf-8"),
            "status_code": 200,
            "headers": {"content-type": "application/json"},
        }

    async def _stream(
        self,
        payload: dict,
        chat_payload: dict,
        chat_url: str,
        forward_headers: dict[str, str],
        web_search_active: bool,
        should_log: bool,
        base_filename: str,
    ) -> AsyncIterator[bytes]:
        response_id = _new_id("resp")
        snapshot = _base_response(payload, response_id, str(payload.get("model") or chat_payload["model"]))
        usage = _empty_usage()
        sequence = 0
        chunks_log: list[str] = []
        output_index = 0

        def event(event_type: str, data: dict) -> bytes:
            nonlocal sequence
            body = {"type": event_type, "sequence_number": sequence, **data}
            sequence += 1
            text = f"event: {event_type}\ndata: {json.dumps(body)}\n\n"
            if should_log:
                chunks_log.append(text)
            return text.encode("utf-8")

        chat_payload = dict(chat_payload)
        chat_payload["stream"] = True
        chat_payload["stream_options"] = {"include_usage": True}

        yield event("response.created", {"response": snapshot})
        yield event("response.in_progress", {"response": snapshot})

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                for _ in range(MAX_TOOL_ITERATIONS):
                    reasoning_item: dict | None = None
                    reasoning_acc = ""
                    message_item: dict | None = None
                    text_acc = ""
                    tool_calls_acc: dict[int, dict] = {}
                    finish_reason: str | None = None

                    def close_reasoning() -> list[bytes]:
                        nonlocal reasoning_item, reasoning_acc, output_index
                        if reasoning_item is None:
                            return []
                        item_id = reasoning_item["id"]
                        done_item = {
                            **reasoning_item,
                            "summary": [{"type": "summary_text", "text": reasoning_acc}],
                        }
                        events = [
                            event(
                                "response.reasoning_summary_text.done",
                                {
                                    "item_id": item_id,
                                    "output_index": output_index,
                                    "summary_index": 0,
                                    "text": reasoning_acc,
                                },
                            ),
                            event(
                                "response.reasoning_summary_part.done",
                                {
                                    "item_id": item_id,
                                    "output_index": output_index,
                                    "summary_index": 0,
                                    "part": {"type": "summary_text", "text": reasoning_acc},
                                },
                            ),
                            event("response.output_item.done", {"output_index": output_index, "item": done_item}),
                        ]
                        snapshot["output"].append(done_item)
                        output_index += 1
                        reasoning_item = None
                        reasoning_acc = ""
                        return events

                    def close_message() -> list[bytes]:
                        nonlocal message_item, text_acc, output_index
                        if message_item is None:
                            return []
                        item_id = message_item["id"]
                        part = {"type": "output_text", "text": text_acc, "annotations": []}
                        done_item = {**message_item, "status": "completed", "content": [part]}
                        events = [
                            event(
                                "response.output_text.done",
                                {
                                    "item_id": item_id,
                                    "output_index": output_index,
                                    "content_index": 0,
                                    "text": text_acc,
                                },
                            ),
                            event(
                                "response.content_part.done",
                                {
                                    "item_id": item_id,
                                    "output_index": output_index,
                                    "content_index": 0,
                                    "part": part,
                                },
                            ),
                            event("response.output_item.done", {"output_index": output_index, "item": done_item}),
                        ]
                        snapshot["output"].append(done_item)
                        output_index += 1
                        message_item = None
                        text_acc = ""
                        return events

                    async with client.stream(
                        "POST", chat_url, headers=forward_headers, json=chat_payload
                    ) as upstream:
                        if upstream.status_code != 200:
                            error_body = (await upstream.aread()).decode("utf-8", errors="replace")
                            snapshot["status"] = "failed"
                            snapshot["error"] = {"code": "upstream_error", "message": error_body[:2000]}
                            yield event("response.failed", {"response": snapshot})
                            return

                        async for line in upstream.aiter_lines():
                            line = line.strip()
                            if not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                            except Exception:
                                continue
                            _accumulate_usage(usage, chunk.get("usage"))
                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            choice = choices[0]
                            if choice.get("finish_reason"):
                                finish_reason = choice["finish_reason"]
                            delta = choice.get("delta") or {}

                            reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning")
                            if reasoning_delta:
                                if reasoning_item is None:
                                    reasoning_item = {"id": _new_id("rs"), "type": "reasoning", "summary": []}
                                    yield event(
                                        "response.output_item.added",
                                        {"output_index": output_index, "item": reasoning_item},
                                    )
                                    yield event(
                                        "response.reasoning_summary_part.added",
                                        {
                                            "item_id": reasoning_item["id"],
                                            "output_index": output_index,
                                            "summary_index": 0,
                                            "part": {"type": "summary_text", "text": ""},
                                        },
                                    )
                                reasoning_acc += reasoning_delta
                                yield event(
                                    "response.reasoning_summary_text.delta",
                                    {
                                        "item_id": reasoning_item["id"],
                                        "output_index": output_index,
                                        "summary_index": 0,
                                        "delta": reasoning_delta,
                                    },
                                )

                            content_delta = delta.get("content")
                            if content_delta:
                                for evt in close_reasoning():
                                    yield evt
                                if message_item is None:
                                    message_item = {
                                        "id": _new_id("msg"),
                                        "type": "message",
                                        "status": "in_progress",
                                        "role": "assistant",
                                        "content": [],
                                    }
                                    yield event(
                                        "response.output_item.added",
                                        {"output_index": output_index, "item": message_item},
                                    )
                                    yield event(
                                        "response.content_part.added",
                                        {
                                            "item_id": message_item["id"],
                                            "output_index": output_index,
                                            "content_index": 0,
                                            "part": {"type": "output_text", "text": "", "annotations": []},
                                        },
                                    )
                                text_acc += content_delta
                                yield event(
                                    "response.output_text.delta",
                                    {
                                        "item_id": message_item["id"],
                                        "output_index": output_index,
                                        "content_index": 0,
                                        "delta": content_delta,
                                    },
                                )

                            for tc_delta in delta.get("tool_calls") or []:
                                index = tc_delta.get("index", 0)
                                acc = tool_calls_acc.setdefault(index, {"id": None, "name": "", "arguments": ""})
                                if tc_delta.get("id"):
                                    acc["id"] = tc_delta["id"]
                                fn = tc_delta.get("function") or {}
                                if fn.get("name"):
                                    acc["name"] += fn["name"]
                                if fn.get("arguments"):
                                    acc["arguments"] += fn["arguments"]

                    for evt in close_reasoning():
                        yield evt
                    final_text = text_acc
                    for evt in close_message():
                        yield evt

                    tool_calls = [
                        {
                            "id": acc["id"] or _new_id("call"),
                            "type": "function",
                            "function": {"name": acc["name"], "arguments": acc["arguments"]},
                        }
                        for _, acc in sorted(tool_calls_acc.items())
                    ]
                    proxy_tool_names = {WEB_SEARCH_TOOL_NAME, WEB_FETCH_TOOL_NAME}
                    web_search_calls = [
                        tc
                        for tc in tool_calls
                        if web_search_active and tc["function"]["name"] in proxy_tool_names
                    ]
                    client_calls = [tc for tc in tool_calls if tc not in web_search_calls]

                    if web_search_calls and not client_calls:
                        chat_payload["messages"].append(
                            {"role": "assistant", "content": final_text or None, "tool_calls": tool_calls}
                        )
                        for tool_call in web_search_calls:
                            fn_name = tool_call["function"]["name"]
                            ws_item = {"id": _new_id("ws"), "type": "web_search_call", "status": "in_progress"}
                            yield event(
                                "response.output_item.added", {"output_index": output_index, "item": ws_item}
                            )
                            yield event(
                                "response.web_search_call.in_progress",
                                {"output_index": output_index, "item_id": ws_item["id"]},
                            )
                            yield event(
                                "response.web_search_call.searching",
                                {"output_index": output_index, "item_id": ws_item["id"]},
                            )
                            if fn_name == WEB_FETCH_TOOL_NAME:
                                fetch_url = _parse_fetch_url(tool_call)
                                fetch_result = await self._web_search_service.fetch(fetch_url)
                                chat_payload["messages"].append(
                                    {"role": "tool", "tool_call_id": tool_call["id"], "content": fetch_result}
                                )
                                done_action = {"type": "fetch", "url": fetch_url}
                            else:
                                query = _parse_search_query(tool_call)
                                search_result = await self._web_search_service.search(query)
                                chat_payload["messages"].append(
                                    {"role": "tool", "tool_call_id": tool_call["id"], "content": search_result}
                                )
                                done_action = {"type": "search", "query": query}
                            yield event(
                                "response.web_search_call.completed",
                                {"output_index": output_index, "item_id": ws_item["id"]},
                            )
                            done_ws = {
                                **ws_item,
                                "status": "completed",
                                "action": done_action,
                            }
                            yield event(
                                "response.output_item.done", {"output_index": output_index, "item": done_ws}
                            )
                            snapshot["output"].append(done_ws)
                            output_index += 1
                        _reset_forced_tool_choice(chat_payload)
                        continue

                    for tool_call in client_calls:
                        arguments = tool_call["function"]["arguments"] or ""
                        fc_item = {
                            "id": _new_id("fc"),
                            "type": "function_call",
                            "status": "in_progress",
                            "call_id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "arguments": "",
                        }
                        yield event("response.output_item.added", {"output_index": output_index, "item": fc_item})
                        if arguments:
                            yield event(
                                "response.function_call_arguments.delta",
                                {"item_id": fc_item["id"], "output_index": output_index, "delta": arguments},
                            )
                        yield event(
                            "response.function_call_arguments.done",
                            {"item_id": fc_item["id"], "output_index": output_index, "arguments": arguments},
                        )
                        done_fc = {**fc_item, "status": "completed", "arguments": arguments}
                        yield event("response.output_item.done", {"output_index": output_index, "item": done_fc})
                        snapshot["output"].append(done_fc)
                        output_index += 1

                    snapshot["usage"] = usage
                    if finish_reason == "length":
                        snapshot["status"] = "incomplete"
                        snapshot["incomplete_details"] = {"reason": "max_output_tokens"}
                        yield event("response.incomplete", {"response": snapshot})
                    else:
                        snapshot["status"] = "completed"
                        yield event("response.completed", {"response": snapshot})
                    return
                else:
                    snapshot["usage"] = usage
                    snapshot["status"] = "incomplete"
                    snapshot["incomplete_details"] = {"reason": "max_tool_iterations"}
                    yield event("response.incomplete", {"response": snapshot})
        except Exception as exc:
            self._logger.exception("responses_stream_error: %s", exc)
            snapshot["status"] = "failed"
            snapshot["error"] = {"code": "server_error", "message": str(exc)}
            yield event("response.failed", {"response": snapshot})
        finally:
            await self._trace_response(
                should_log,
                base_filename,
                {
                    "timestamp": datetime.now().isoformat(),
                    "status_code": 200,
                    "headers": {"content-type": "text/event-stream"},
                    "chunks": chunks_log,
                },
            )

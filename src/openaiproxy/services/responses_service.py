from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

import anyio
import httpx
from fastapi import HTTPException

from openaiproxy.interface.repository import LLMProxyConfigRepository
from openaiproxy.models.models import EndpointConfig
from openaiproxy.services.model_tracker_service import ModelTrackerService
from openaiproxy.services.proxy_service import (
    extract_api_key,
    extract_correlation_id,
    filter_response_headers,
    save_request_trace,
    save_response_trace,
)
from openaiproxy.services.ledger_service import LedgerService
from openaiproxy.services.slot_cache_service import SlotAllocator, SlotCacheService
from openaiproxy.services.web_search_service import (
    WEB_FETCH_TOOL_DEFINITION,
    WEB_FETCH_TOOL_NAME,
    WEB_SEARCH_TOOL_DEFINITION,
    WEB_SEARCH_TOOL_NAME,
    WebSearchService,
)

MAX_TOOL_ITERATIONS = 5
SEARCH_BUDGET_MESSAGE = "Search budget reached — answer with the information already gathered."


def _strip_tools(chat_payload: dict) -> None:
    chat_payload.pop("tools", None)
    chat_payload.pop("tool_choice", None)
    chat_payload.pop("parallel_tool_calls", None)


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


def _effective_call_limit(ws_config: Any) -> int:
    if ws_config.map_reduce_call_limit > 0:
        return ws_config.map_reduce_call_limit
    return ws_config.map_reduce_context_limit


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3)


def _build_citations(text: str, sources: list[dict]) -> tuple[str, list[dict]]:
    seen: set[str] = set()
    unique: list[dict] = []
    for s in sources:
        url = s.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(s)
    if not unique:
        return text, []
    annotations: list[dict] = []
    new_text = text + "\n\n---"
    for i, source in enumerate(unique, 1):
        url = source["url"]
        title = source.get("title") or url
        label = f"[{i}] {title}"
        start_index = len(new_text) + 1  # +1 to skip the leading \n
        new_text += "\n" + label
        end_index = len(new_text)
        annotations.append({
            "type": "url_citation",
            "start_index": start_index,
            "end_index": end_index,
            "url": url,
            "title": title,
        })
    return new_text, annotations


def _lean_base_messages(messages: list[dict]) -> list[dict]:
    """Conversational context for map/reduce calls: keep system/user turns and
    assistant text, drop bulky prior tool results and their tool-call stubs so
    each map call stays bounded regardless of how much history has accumulated."""
    lean: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            continue
        if role == "assistant" and m.get("tool_calls") and not m.get("content"):
            continue
        lean.append(m)
    return lean


def _chunk_text(text: str, chunk_tokens: int) -> list[str]:
    if not text:
        return []
    char_budget = max(10, chunk_tokens * 3)  # 1 token ≈ 3 chars
    boundaries = ["\n# ", "\n## ", "\n### ", "\n\n", ". ", " "]
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + char_budget, n)
        if end == n:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break
        # Find the rightmost occurrence of any boundary within the budget
        best_pos = -1
        for boundary in boundaries:
            pos = text.rfind(boundary, start, end)
            if pos > start and pos > best_pos:
                best_pos = pos
        # Cut after the newline/period so the heading/sentence starts the next chunk
        cut = best_pos + 1 if best_pos > start else end
        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)
        start = cut
    return chunks


class ResponsesService:
    def __init__(
        self,
        config_repository: LLMProxyConfigRepository,
        trace_dir: Path,
        logger: logging.Logger,
        web_search_service: WebSearchService,
        model_tracker: ModelTrackerService,
        ledger_service: LedgerService | None = None,
        slot_cache_service: SlotCacheService | None = None,
        slot_allocator: SlotAllocator | None = None,
    ):
        self._config_repository = config_repository
        self._trace_dir = trace_dir
        self._logger = logger
        self._web_search_service = web_search_service
        self._model_tracker = model_tracker
        self._ledger_service = ledger_service
        self._slot_cache = slot_cache_service or SlotCacheService(logger)
        self._slot_allocator = slot_allocator or SlotAllocator(logger)

    async def _map_reduce_search_result(
        self,
        base_messages: list[dict],
        tool_call_id: str,
        content: str,
        model: str,
        chat_url: str,
        forward_headers: dict,
        chunk_size: int,
        do_reduce: bool,
        context_limit: int = 0,
        should_log: bool = False,
        api_key: str = "unknown",
        correlation_id: str | None = None,
        parent_trace_id: str | None = None,
        cache_prompt: bool = False,
    ) -> str:
        # Each map/reduce call carries only the conversational context (the user's
        # question), not the accumulated raw results of previous searches — so the
        # per-call size stays bounded no matter how long the conversation grows.
        lean_base = _lean_base_messages(base_messages)
        base_tokens = _estimate_tokens(json.dumps({"messages": lean_base}))
        if context_limit > 0:
            available = context_limit - base_tokens
            if available <= 0:
                self._logger.warning(
                    "map_reduce_lean_base_too_large base_tokens=%d context_limit=%d", base_tokens, context_limit
                )
                return content  # caller's final trim clamps it to the real budget
            effective_chunk = min(chunk_size, available)
        else:
            effective_chunk = chunk_size
        chunks = _chunk_text(content, effective_chunk)
        total_chunks = len(chunks)
        self._logger.info(
            "map_reduce_start chunks=%d effective_chunk=%d base_tokens=%d tool_call_id=%s",
            total_chunks, effective_chunk, base_tokens, tool_call_id,
        )

        def _map_messages(payload_content: str) -> list[dict]:
            return [
                *lean_base,
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": WEB_SEARCH_TOOL_NAME, "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": tool_call_id, "content": payload_content},
            ]

        results: list[str] = [""] * total_chunks

        async def map_one(chunk: str, chunk_index: int) -> None:
            endpoint = f"/responses/map-reduce/{chunk_index + 1}/{total_chunks}"
            payload = {"model": model, "messages": _map_messages(chunk)}
            if cache_prompt:
                payload["cache_prompt"] = True
            base_fn = ""
            if should_log:
                base_fn = await save_request_trace(
                    self._trace_dir, endpoint, payload, forward_headers, api_key, correlation_id, parent_trace_id
                )
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    resp = await client.post(chat_url, headers=forward_headers, json=payload)
                    if should_log and base_fn:
                        await save_response_trace(
                            self._trace_dir, base_fn,
                            {
                                "timestamp": datetime.now().isoformat(),
                                "status_code": resp.status_code,
                                "headers": dict(resp.headers),
                                "body": resp.text,
                            },
                            correlation_id,
                            endpoint_path=endpoint,
                        )
                    if resp.status_code == 200:
                        data = resp.json()
                        results[chunk_index] = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            except Exception as exc:
                self._logger.warning("map_reduce_map_call_error: %s", exc)
                if should_log and base_fn:
                    await save_response_trace(
                        self._trace_dir, base_fn,
                        {"timestamp": datetime.now().isoformat(), "error": str(exc)},
                        correlation_id,
                        endpoint_path=endpoint,
                    )

        async with anyio.create_task_group() as tg:
            for i, chunk in enumerate(chunks):
                tg.start_soon(map_one, chunk, i)

        combined = "\n\n".join(r for r in results if r.strip())

        if not do_reduce or not combined:
            return combined

        if context_limit > 0:
            combined_limit = context_limit - base_tokens
            if _estimate_tokens(combined) > combined_limit:
                combined = combined[: combined_limit * 3]

        reduce_payload = {"model": model, "messages": _map_messages(combined)}
        if cache_prompt:
            reduce_payload["cache_prompt"] = True
        reduce_fn = ""
        if should_log:
            reduce_fn = await save_request_trace(
                self._trace_dir, "/responses/map-reduce/final", reduce_payload, forward_headers, api_key, correlation_id, parent_trace_id
            )
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(chat_url, headers=forward_headers, json=reduce_payload)
                if should_log and reduce_fn:
                    await save_response_trace(
                        self._trace_dir, reduce_fn,
                        {
                            "timestamp": datetime.now().isoformat(),
                            "status_code": resp.status_code,
                            "headers": dict(resp.headers),
                            "body": resp.text,
                        },
                        correlation_id,
                        endpoint_path="/responses/map-reduce/final",
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        except Exception as exc:
            self._logger.warning("map_reduce_reduce_call_error: %s", exc)
            if should_log and reduce_fn:
                await save_response_trace(
                    self._trace_dir, reduce_fn,
                    {"timestamp": datetime.now().isoformat(), "error": str(exc)},
                    correlation_id,
                    endpoint_path="/responses/map-reduce/final",
                )
        return combined

    def _select_endpoint_for_model(self, model: str | None) -> EndpointConfig | None:
        config = self._config_repository.load()
        endpoints = [
            endpoint for endpoint in config.endpoints if endpoint.enabled and endpoint.mode != "local"
        ]

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
        correlation_id = extract_correlation_id(headers)
        endpoint = self._select_endpoint_for_model(requested_model)
        if not endpoint:
            config = self._config_repository.load()
            if not config.endpoints:
                self._logger.error(
                    "responses_no_routes_configured requested_model=%s: no endpoints defined in config",
                    requested_model,
                )
                raise HTTPException(status_code=500, detail={"error": "No llmproxy routes configured"})
            self._logger.warning(
                "responses_no_route_for_model requested_model=%s available_endpoints=%s",
                requested_model,
                [e.name for e in config.endpoints],
            )
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
        if endpoint.cache_prompt and endpoint.backend == "llamacpp":
            chat_payload["cache_prompt"] = True

        substitutions = endpoint.substitute_role or {}
        for message in chat_payload["messages"]:
            if message["role"] in substitutions:
                message["role"] = substitutions[message["role"]]

        self._logger.info(
            "responses_route requested_model=%s forwarded_model=%s endpoint=%s web_search=%s stream=%s correlation_id=%s",
            requested_model,
            forwarded_model,
            endpoint.name,
            web_search_active,
            bool(payload.get("stream")),
            correlation_id,
        )

        if forwarded_model is not None:
            await self._model_tracker.ensure_capacity(endpoint, str(forwarded_model))

        should_log = bool(endpoint.log)
        api_key = extract_api_key(headers) if "authorization" in headers else (endpoint.api_key or "unknown")
        base_filename = (
            await save_request_trace(self._trace_dir, "/responses", payload, headers, api_key, correlation_id)
            if should_log
            else ""
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
                    payload, chat_payload, chat_url, forward_headers, web_search_active,
                    should_log, base_filename, correlation_id, web_search_config, api_key,
                    endpoint,
                ),
            }

        return await self._complete(
            payload, chat_payload, chat_url, forward_headers, web_search_active,
            should_log, base_filename, correlation_id, web_search_config, api_key,
            endpoint,
        )

    async def _trace_response(
        self, should_log: bool, base_filename: str, response_data: dict, correlation_id: str | None = None
    ) -> None:
        if should_log:
            await save_response_trace(
                self._trace_dir, base_filename, response_data, correlation_id, endpoint_path="/responses"
            )

    async def _complete(
        self,
        payload: dict,
        chat_payload: dict,
        chat_url: str,
        forward_headers: dict[str, str],
        web_search_active: bool,
        should_log: bool,
        base_filename: str,
        correlation_id: str | None = None,
        web_search_config: Any = None,
        api_key: str = "unknown",
        endpoint: EndpointConfig | None = None,
    ) -> dict:
        # KV slot persistence: lease this conversation's slot for the whole
        # restore -> tool loop -> save sequence
        slot_model = str(chat_payload.get("model") or "") or None
        async with self._slot_allocator.lease(endpoint, correlation_id, model=slot_model) as lease:
            if lease is not None:
                chat_payload["id_slot"] = lease.slot
                if not lease.resident:
                    await self._slot_cache.restore(endpoint, correlation_id, lease.slot, slot_model)
            return await self._complete_inner(
                payload, chat_payload, chat_url, forward_headers, web_search_active,
                should_log, base_filename, correlation_id, web_search_config, api_key,
                endpoint, lease,
            )

    async def _complete_inner(
        self,
        payload: dict,
        chat_payload: dict,
        chat_url: str,
        forward_headers: dict[str, str],
        web_search_active: bool,
        should_log: bool,
        base_filename: str,
        correlation_id: str | None = None,
        web_search_config: Any = None,
        api_key: str = "unknown",
        endpoint: EndpointConfig | None = None,
        lease: Any = None,
    ) -> dict:
        response_id = _new_id("resp")
        result = _base_response(payload, response_id, str(payload.get("model") or chat_payload["model"]))
        usage = _empty_usage()
        output: list[dict] = []
        status = "completed"
        incomplete_reason: str | None = None
        all_sources: list[dict] = []
        max_searches = web_search_config.max_searches if web_search_config is not None else 0
        agent_loop = web_search_config.agent_loop if web_search_config is not None else True
        searches_done = 0

        async with httpx.AsyncClient(timeout=3000.0) as client:
            for _ in range(MAX_TOOL_ITERATIONS):
                try:
                    upstream = await client.post(chat_url, headers=forward_headers, json=chat_payload)
                except Exception as exc:
                    self._logger.exception(
                        "responses_upstream_connect_error chat_url=%s: %s", chat_url, exc
                    )
                    await self._trace_response(
                        should_log, base_filename, {"timestamp": datetime.now().isoformat(), "error": str(exc)},
                        correlation_id,
                    )
                    return {
                        "type": "response",
                        "content": json.dumps({"error": str(exc)}).encode("utf-8"),
                        "status_code": 502,
                        "headers": {"content-type": "application/json"},
                    }

                if upstream.status_code != 200:
                    self._logger.warning(
                        "responses_upstream_error chat_url=%s status=%s body=%s",
                        chat_url,
                        upstream.status_code,
                        upstream.text[:500],
                    )
                    await self._trace_response(
                        should_log,
                        base_filename,
                        {
                            "timestamp": datetime.now().isoformat(),
                            "status_code": upstream.status_code,
                            "headers": dict(upstream.headers),
                            "body": upstream.text,
                        },
                        correlation_id,
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
                    cited_content, annotations = _build_citations(content, all_sources)
                    output.append(
                        {
                            "type": "message",
                            "id": _new_id("msg"),
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": cited_content, "annotations": annotations}],
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
                    raw_results: list[str] = []
                    for tool_call in web_search_calls:
                        if max_searches > 0 and searches_done >= max_searches:
                            chat_payload["messages"].append(
                                {"role": "tool", "tool_call_id": tool_call.get("id"), "content": SEARCH_BUDGET_MESSAGE}
                            )
                            continue
                        searches_done += 1
                        fn_name = (tool_call.get("function") or {}).get("name")
                        if fn_name == WEB_FETCH_TOOL_NAME:
                            fetch_url = _parse_fetch_url(tool_call)
                            tool_result = await self._web_search_service.fetch(fetch_url)
                            output.append(
                                {
                                    "type": "web_search_call",
                                    "id": _new_id("ws"),
                                    "status": "completed",
                                    "action": {"type": "fetch", "url": fetch_url},
                                }
                            )
                        else:
                            query = _parse_search_query(tool_call)
                            tool_result, sources = await self._web_search_service.search(query)
                            all_sources.extend(sources)
                            output.append(
                                {
                                    "type": "web_search_call",
                                    "id": _new_id("ws"),
                                    "status": "completed",
                                    "action": {"type": "search", "query": query},
                                }
                            )
                        if not agent_loop:
                            raw_results.append(tool_result)
                        if agent_loop and web_search_config is not None:
                            current_tokens = _estimate_tokens(json.dumps(chat_payload))
                            if current_tokens + _estimate_tokens(tool_result) > web_search_config.map_reduce_context_limit:
                                self._logger.info(
                                    "map_reduce_triggered estimated_tokens=%d limit=%d",
                                    current_tokens + _estimate_tokens(tool_result),
                                    web_search_config.map_reduce_context_limit,
                                )
                                tool_result = await self._map_reduce_search_result(
                                    list(chat_payload["messages"]),
                                    tool_call.get("id") or "",
                                    tool_result,
                                    str(chat_payload.get("model", "")),
                                    chat_url,
                                    forward_headers,
                                    web_search_config.map_reduce_chunk_size,
                                    web_search_config.map_reduce_reduce,
                                    context_limit=_effective_call_limit(web_search_config),
                                    should_log=should_log,
                                    api_key=api_key,
                                    correlation_id=correlation_id,
                                    parent_trace_id=base_filename,
                                    cache_prompt=bool(chat_payload.get("cache_prompt")),
                                )
                        if agent_loop and web_search_config is not None:
                            remaining = _effective_call_limit(web_search_config) - _estimate_tokens(json.dumps(chat_payload))
                            if remaining > 0 and _estimate_tokens(tool_result) > remaining:
                                tool_result = tool_result[: remaining * 3]
                        chat_payload["messages"].append(
                            {"role": "tool", "tool_call_id": tool_call.get("id"), "content": tool_result}
                        )
                    if not agent_loop:
                        output.append(
                            {
                                "type": "message",
                                "id": _new_id("msg"),
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "\n\n".join(r for r in raw_results if r),
                                        "annotations": [],
                                    }
                                ],
                            }
                        )
                        break
                    if max_searches > 0 and searches_done >= max_searches:
                        _strip_tools(chat_payload)
                    else:
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
            correlation_id,
        )
        if self._ledger_service and base_filename:
            await self._ledger_service.validate_and_record(
                trace_id=base_filename,
                model=payload.get("model"),
                endpoint="/responses",
                request_payload=payload,
                response_body=result,
                response_chunks=[],
            )

        if lease is not None and endpoint is not None:
            await self._slot_cache.save(
                endpoint, correlation_id, lease.slot, str(chat_payload.get("model") or "") or None
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
        correlation_id: str | None = None,
        web_search_config: Any = None,
        api_key: str = "unknown",
        endpoint: EndpointConfig | None = None,
    ) -> AsyncIterator[bytes]:
        # KV slot persistence: lease this conversation's slot for the whole
        # restore -> tool loop -> save sequence
        slot_model = str(chat_payload.get("model") or "") or None
        async with self._slot_allocator.lease(endpoint, correlation_id, model=slot_model) as lease:
            if lease is not None:
                chat_payload["id_slot"] = lease.slot
                if not lease.resident:
                    await self._slot_cache.restore(endpoint, correlation_id, lease.slot, slot_model)
            async for chunk in self._stream_inner(
                payload, chat_payload, chat_url, forward_headers, web_search_active,
                should_log, base_filename, correlation_id, web_search_config, api_key,
                endpoint, lease,
            ):
                yield chunk

    async def _stream_inner(
        self,
        payload: dict,
        chat_payload: dict,
        chat_url: str,
        forward_headers: dict[str, str],
        web_search_active: bool,
        should_log: bool,
        base_filename: str,
        correlation_id: str | None = None,
        web_search_config: Any = None,
        api_key: str = "unknown",
        endpoint: EndpointConfig | None = None,
        lease: Any = None,
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

        all_sources: list[dict] = []
        max_searches = web_search_config.max_searches if web_search_config is not None else 0
        agent_loop = web_search_config.agent_loop if web_search_config is not None else True
        searches_done = 0

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
                        cited_text, annotations = _build_citations(text_acc, all_sources)
                        events = []
                        if len(cited_text) > len(text_acc):
                            events.append(event(
                                "response.output_text.delta",
                                {
                                    "item_id": item_id,
                                    "output_index": output_index,
                                    "content_index": 0,
                                    "delta": cited_text[len(text_acc):],
                                },
                            ))
                        part = {"type": "output_text", "text": cited_text, "annotations": annotations}
                        done_item = {**message_item, "status": "completed", "content": [part]}
                        events.extend([
                            event(
                                "response.output_text.done",
                                {
                                    "item_id": item_id,
                                    "output_index": output_index,
                                    "content_index": 0,
                                    "text": cited_text,
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
                        ])
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
                            self._logger.warning(
                                "responses_stream_upstream_error chat_url=%s status=%s body=%s",
                                chat_url,
                                upstream.status_code,
                                error_body[:500],
                            )
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
                        raw_results: list[str] = []
                        for tool_call in web_search_calls:
                            if max_searches > 0 and searches_done >= max_searches:
                                chat_payload["messages"].append(
                                    {"role": "tool", "tool_call_id": tool_call["id"], "content": SEARCH_BUDGET_MESSAGE}
                                )
                                continue
                            searches_done += 1
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
                                result = await self._web_search_service.fetch(fetch_url)
                                done_action = {"type": "fetch", "url": fetch_url}
                            else:
                                query = _parse_search_query(tool_call)
                                result, sources = await self._web_search_service.search(query)
                                all_sources.extend(sources)
                                done_action = {"type": "search", "query": query}
                            if not agent_loop:
                                raw_results.append(result)
                            if agent_loop and web_search_config is not None:
                                current_tokens = _estimate_tokens(json.dumps(chat_payload))
                                if current_tokens + _estimate_tokens(result) > web_search_config.map_reduce_context_limit:
                                    self._logger.info(
                                        "map_reduce_triggered estimated_tokens=%d limit=%d",
                                        current_tokens + _estimate_tokens(result),
                                        web_search_config.map_reduce_context_limit,
                                    )
                                    result = await self._map_reduce_search_result(
                                        list(chat_payload["messages"]),
                                        tool_call["id"],
                                        result,
                                        str(chat_payload.get("model", "")),
                                        chat_url,
                                        forward_headers,
                                        web_search_config.map_reduce_chunk_size,
                                        web_search_config.map_reduce_reduce,
                                        context_limit=_effective_call_limit(web_search_config),
                                        should_log=should_log,
                                        api_key=api_key,
                                        correlation_id=correlation_id,
                                        parent_trace_id=base_filename,
                                        cache_prompt=bool(chat_payload.get("cache_prompt")),
                                    )
                            if agent_loop and web_search_config is not None:
                                remaining = _effective_call_limit(web_search_config) - _estimate_tokens(json.dumps(chat_payload))
                                if remaining > 0 and _estimate_tokens(result) > remaining:
                                    result = result[: remaining * 3]
                            chat_payload["messages"].append(
                                {"role": "tool", "tool_call_id": tool_call["id"], "content": result}
                            )
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
                        if not agent_loop:
                            raw_text = "\n\n".join(r for r in raw_results if r)
                            msg_item = {
                                "id": _new_id("msg"),
                                "type": "message",
                                "status": "in_progress",
                                "role": "assistant",
                                "content": [],
                            }
                            yield event(
                                "response.output_item.added", {"output_index": output_index, "item": msg_item}
                            )
                            yield event(
                                "response.content_part.added",
                                {
                                    "item_id": msg_item["id"],
                                    "output_index": output_index,
                                    "content_index": 0,
                                    "part": {"type": "output_text", "text": "", "annotations": []},
                                },
                            )
                            if raw_text:
                                yield event(
                                    "response.output_text.delta",
                                    {
                                        "item_id": msg_item["id"],
                                        "output_index": output_index,
                                        "content_index": 0,
                                        "delta": raw_text,
                                    },
                                )
                            part = {"type": "output_text", "text": raw_text, "annotations": []}
                            yield event(
                                "response.output_text.done",
                                {
                                    "item_id": msg_item["id"],
                                    "output_index": output_index,
                                    "content_index": 0,
                                    "text": raw_text,
                                },
                            )
                            yield event(
                                "response.content_part.done",
                                {
                                    "item_id": msg_item["id"],
                                    "output_index": output_index,
                                    "content_index": 0,
                                    "part": part,
                                },
                            )
                            done_msg = {**msg_item, "status": "completed", "content": [part]}
                            yield event(
                                "response.output_item.done", {"output_index": output_index, "item": done_msg}
                            )
                            snapshot["output"].append(done_msg)
                            output_index += 1
                            snapshot["usage"] = usage
                            snapshot["status"] = "completed"
                            yield event("response.completed", {"response": snapshot})
                            return
                        if max_searches > 0 and searches_done >= max_searches:
                            _strip_tools(chat_payload)
                        else:
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
            if lease is not None and endpoint is not None:
                await self._slot_cache.save(
                    endpoint, correlation_id, lease.slot, str(chat_payload.get("model") or "") or None
                )
            await self._trace_response(
                should_log,
                base_filename,
                {
                    "timestamp": datetime.now().isoformat(),
                    "status_code": 200,
                    "headers": {"content-type": "text/event-stream"},
                    "chunks": chunks_log,
                },
                correlation_id,
            )
            if self._ledger_service and base_filename:
                await self._ledger_service.validate_and_record(
                    trace_id=base_filename,
                    model=payload.get("model"),
                    endpoint="/responses",
                    request_payload=payload,
                    response_body=None,
                    response_chunks=chunks_log,
                )

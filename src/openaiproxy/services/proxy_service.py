from __future__ import annotations

import base64
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import anyio
import httpx
from fastapi import HTTPException

from openaiproxy.interface.repository import LLMProxyConfigRepository
from openaiproxy.models.models import EndpointConfig
from openaiproxy.models.trace_models import trace_category
from openaiproxy.services.anthropic_translation import (
    AnthropicToChatStream,
    anthropic_response_to_chat,
    build_anthropic_headers,
    chat_request_to_anthropic,
    extract_error_message,
    openai_error_body,
)
from openaiproxy.services.cost_recorder_service import CostRecorderService
from openaiproxy.services.model_tracker_service import ModelTrackerService
from openaiproxy.services.ledger_service import LedgerService
from openaiproxy.services.slot_cache_service import SlotAllocator, SlotCacheService


CORRELATION_ID_HEADER = "x-correlation-id"


def extract_api_key(headers: dict[str, str]) -> str:
    auth_header = headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    return "unknown"


def sanitize_api_key_for_filename(api_key: str) -> str:
    return api_key.replace("/", "_").replace("\\", "_").replace(":", "_")


def sanitize_correlation_id(value: str) -> str:
    """Make a correlation id safe to use as a single directory name.

    Path separators and traversal sequences are neutralised so the value can
    never escape the trace directory. Returns an empty string when nothing
    usable remains.
    """
    cleaned = value.strip().replace("/", "_").replace("\\", "_").replace(":", "_")
    cleaned = cleaned.replace("\x00", "").strip(". ")
    if cleaned in ("", ".", ".."):
        return ""
    return cleaned[:200]


def extract_correlation_id(headers: dict[str, str]) -> str | None:
    """Return the sanitized X-Correlation-Id header value, or None if absent."""
    raw = headers.get(CORRELATION_ID_HEADER)
    if not raw:
        return None
    sanitized = sanitize_correlation_id(raw)
    return sanitized or None


def trace_subdir(trace_dir: Path, correlation_id: str | None, endpoint_path: str | None = None) -> Path:
    """Resolve the directory a trace should be written to.

    Traces are grouped per inbound API (traces/completion, traces/responses,
    ...). When a correlation id is present the trace is nested one level
    deeper so all messages of the same thread are grouped together.
    """
    target = trace_dir
    if endpoint_path:
        target = target / trace_category(endpoint_path)
    if correlation_id:
        target = target / correlation_id
    return target


_RESPONSE_FRAMING_HEADERS = {
    "content-length",
    "transfer-encoding",
    "content-encoding",
    "connection",
    "keep-alive",
    "trailer",
    "upgrade",
}


def filter_response_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _RESPONSE_FRAMING_HEADERS}


async def save_request_trace(
    trace_dir: Path,
    endpoint_path: str,
    payload: Any,
    headers: dict[str, str],
    api_key: str,
    correlation_id: str | None = None,
    parent_trace_id: str | None = None,
    provider: str | None = None,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_api_key = sanitize_api_key_for_filename(api_key)
    base_filename = f"{safe_api_key}_{timestamp}"
    target_dir = trace_subdir(trace_dir, correlation_id, endpoint_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = target_dir / f"{base_filename}_request.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint_path,
        "correlation_id": correlation_id,
        "parent_trace_id": parent_trace_id,
        "provider": provider,
        "headers": dict(headers),
        "payload": payload,
    }

    def _write() -> None:
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    await anyio.to_thread.run_sync(_write)
    return base_filename


async def save_response_trace(
    trace_dir: Path,
    base_filename: str,
    response_data: dict[str, Any],
    correlation_id: str | None = None,
    endpoint_path: str | None = None,
) -> None:
    target_dir = trace_subdir(trace_dir, correlation_id, endpoint_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = target_dir / f"{base_filename}_response.json"

    def _write() -> None:
        with open(filename, "w") as f:
            json.dump(response_data, f, indent=2)

    await anyio.to_thread.run_sync(_write)


class ProxyService:
    def __init__(
        self,
        config_repository: LLMProxyConfigRepository,
        trace_dir: Path,
        logger: logging.Logger,
        model_tracker: ModelTrackerService,
        ledger_service: LedgerService | None = None,
        cost_recorder: CostRecorderService | None = None,
        slot_cache_service: SlotCacheService | None = None,
        slot_allocator: SlotAllocator | None = None,
    ):
        self._config_repository = config_repository
        self._trace_dir = trace_dir
        self._logger = logger
        self._model_tracker = model_tracker
        self._ledger_service = ledger_service
        self._cost_recorder = cost_recorder
        self._slot_cache = slot_cache_service or SlotCacheService(logger)
        self._slot_allocator = slot_allocator or SlotAllocator(logger)

    def _select_endpoint_for_model(self, model: str | None) -> EndpointConfig | None:
        config = self._config_repository.load()
        endpoints = [
            endpoint for endpoint in config.endpoints if endpoint.enabled and endpoint.mode != "local"
        ]
        if not endpoints:
            return None

        if model:
            for endpoint in endpoints:
                if model in endpoint.models or model in endpoint.aliases:
                    return endpoint

        for endpoint in endpoints:
            if "*" in endpoint.models:
                return endpoint

        return None

    async def _save_request(
        self,
        endpoint_path: str,
        payload: Any,
        headers: dict[str, str],
        api_key: str,
        correlation_id: str | None = None,
        provider: str | None = None,
    ) -> str:
        return await save_request_trace(
            self._trace_dir, endpoint_path, payload, headers, api_key, correlation_id, provider=provider
        )

    async def _save_response(
        self,
        base_filename: str,
        response_data: dict[str, Any],
        correlation_id: str | None = None,
        endpoint_path: str | None = None,
    ) -> None:
        await save_response_trace(self._trace_dir, base_filename, response_data, correlation_id, endpoint_path)

    async def substitute_role(self, payload: dict, endpoint_config: EndpointConfig) -> dict:
        substitutions = endpoint_config.substitute_role
        if "messages" in payload and substitutions:
            for message in payload["messages"]:
                if message["role"] in substitutions:
                    message["role"] = substitutions.get(message["role"], message["role"])
        return payload

    async def proxy(self, method: str, endpoint_path: str, body: bytes, headers: dict[str, str], query_params: dict[str, str]):
        payload: Any = {}
        body_to_forward = body
        payload_is_json_object = False
        if body:
            try:
                payload = json.loads(body)
                payload_is_json_object = isinstance(payload, dict)
            except Exception:
                payload = {
                    "raw_body_base64": base64.b64encode(body).decode("ascii"),
                    "content_type": headers.get("content-type", ""),
                }
        requested_model = payload.get("model") if isinstance(payload, dict) else None
        forwarded_model = requested_model
        correlation_id = extract_correlation_id(headers)

        selected_endpoint = self._select_endpoint_for_model(requested_model)

        if selected_endpoint and isinstance(payload, dict) and forwarded_model is not None:
            forwarded_model = selected_endpoint.aliases.get(str(forwarded_model), forwarded_model)
            if forwarded_model != requested_model:
                payload["model"] = forwarded_model
                body_to_forward = json.dumps(payload).encode("utf-8")

        self._logger.info(
            "route_resolve requested_model=%s forwarded_model=%s endpoint=%s correlation_id=%s",
            requested_model,
            forwarded_model,
            (selected_endpoint.name if selected_endpoint else None),
            correlation_id,
        )
        if not selected_endpoint:
            config = self._config_repository.load()
            if not config.endpoints:
                self._logger.error(
                    "proxy_no_routes_configured requested_model=%s endpoint_path=%s: no endpoints defined in config",
                    requested_model,
                    endpoint_path,
                )
                raise HTTPException(status_code=500, detail={"error": "No llmproxy routes configured"})

            self._logger.warning(
                "proxy_no_route_for_model requested_model=%s endpoint_path=%s available_endpoints=%s",
                requested_model,
                endpoint_path,
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

        payload = await self.substitute_role(payload, selected_endpoint)

        if selected_endpoint.cache_prompt and selected_endpoint.backend == "llamacpp" and payload_is_json_object:
            payload["cache_prompt"] = True
            body_to_forward = json.dumps(payload).encode("utf-8")

        if forwarded_model is not None:
            await self._model_tracker.ensure_capacity(selected_endpoint, str(forwarded_model))

        should_log = bool(selected_endpoint.log) and endpoint_path != "/models"
        api_key = extract_api_key(headers) if "authorization" in headers else (selected_endpoint.api_key or "unknown")
        base_filename = (
            await self._save_request(
                endpoint_path, payload, headers, api_key, correlation_id, provider=selected_endpoint.name
            )
            if should_log
            else ""
        )

        if (
            selected_endpoint.protocol == "anthropic"
            and endpoint_path == "/chat/completions"
            and isinstance(payload, dict)
        ):
            return await self._proxy_chat_via_anthropic(
                payload, selected_endpoint, headers, should_log, base_filename, correlation_id
            )

        use_slot_cache = payload_is_json_object and selected_endpoint.slot_cache

        hop_by_hop_headers = {
            "host",
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
            "content-length",
            "accept-encoding",
        }
        forward_headers = {k: v for k, v in headers.items() if k.lower() not in hop_by_hop_headers}

        if selected_endpoint.protocol == "anthropic":
            forward_headers.pop("authorization", None)
            forward_headers.update(build_anthropic_headers(headers, selected_endpoint.api_key))
        elif "authorization" in headers:
            forward_headers["Authorization"] = headers["authorization"]
        elif selected_endpoint.api_key:
            forward_headers["Authorization"] = f"Bearer {selected_endpoint.api_key}"

        if (
            payload_is_json_object
            and payload.get("stream")
            and endpoint_path in ("/chat/completions", "/completions")
            and selected_endpoint.protocol != "anthropic"
            and "stream_options" not in payload
        ):
            payload["stream_options"] = {"include_usage": True}
            body_to_forward = json.dumps(payload).encode("utf-8")

        target_url = f"{selected_endpoint.base_url}{endpoint_path}"
        is_streaming = isinstance(payload, dict) and payload.get("stream", False)
        self._logger.debug(
            "proxy_forward method=%s target_url=%s streaming=%s should_log=%s",
            method,
            target_url,
            is_streaming,
            should_log,
        )

        # the model as the upstream knows it — a router routes /props and /slots by it
        slot_model = str(forwarded_model) if forwarded_model else None

        def pin_to_slot(slot: int) -> bytes:
            payload["id_slot"] = slot
            return json.dumps(payload).encode("utf-8")

        if is_streaming:
            async def stream_response():
                # KV slot persistence: lease this conversation's slot for the whole
                # restore -> inference -> save sequence
                async with self._slot_allocator.lease(
                    selected_endpoint, correlation_id, enabled=use_slot_cache, model=slot_model
                ) as lease:
                    content = body_to_forward
                    if lease is not None:
                        content = pin_to_slot(lease.slot)
                        if not lease.resident:
                            await self._slot_cache.restore(
                                selected_endpoint, correlation_id, lease.slot, slot_model
                            )
                    async with httpx.AsyncClient(timeout=300.0) as client:
                        try:
                            async with client.stream(
                                method,
                                target_url,
                                headers=forward_headers,
                                params=query_params,
                                content=content if content else None,
                            ) as response:
                                response_data = {
                                    "timestamp": datetime.now().isoformat(),
                                    "status_code": response.status_code,
                                    "headers": dict(response.headers),
                                    "chunks": [],
                                }

                                async for chunk in response.aiter_bytes():
                                    if should_log:
                                        try:
                                            response_data["chunks"].append(chunk.decode("utf-8"))
                                        except Exception:
                                            response_data["chunks"].append(str(chunk))
                                    yield chunk

                                if lease is not None and response.status_code < 400:
                                    await self._slot_cache.save(
                                        selected_endpoint, correlation_id, lease.slot, slot_model
                                    )
                                if should_log:
                                    await self._save_response(base_filename, response_data, correlation_id, endpoint_path)
                                if self._ledger_service and base_filename:
                                    await self._ledger_service.validate_and_record(
                                        trace_id=base_filename,
                                        model=requested_model,
                                        endpoint=endpoint_path,
                                        request_payload=payload,
                                        response_body=None,
                                        response_chunks=response_data.get("chunks") or [],
                                    )
                                if self._cost_recorder and base_filename:
                                    await self._cost_recorder.record_usage(
                                        trace_id=base_filename,
                                        model=requested_model,
                                        provider=selected_endpoint.name,
                                        endpoint=endpoint_path,
                                        response_body=None,
                                        response_chunks=response_data.get("chunks") or [],
                                    )
                        except Exception as e:
                            self._logger.exception(
                                "proxy_stream_error method=%s target_url=%s: %s",
                                method,
                                target_url,
                                e,
                            )
                            error_msg = f"data: {json.dumps({'error': str(e)})}\n\n"
                            response_data = {
                                "timestamp": datetime.now().isoformat(),
                                "error": str(e),
                            }
                            if should_log:
                                await self._save_response(base_filename, response_data, correlation_id, endpoint_path)
                            yield error_msg.encode()

            return {
                "type": "stream",
                "iterator": stream_response(),
            }

        async with self._slot_allocator.lease(
            selected_endpoint, correlation_id, enabled=use_slot_cache, model=slot_model
        ) as lease:
            content = body_to_forward
            if lease is not None:
                content = pin_to_slot(lease.slot)
                if not lease.resident:
                    await self._slot_cache.restore(selected_endpoint, correlation_id, lease.slot, slot_model)
            async with httpx.AsyncClient(timeout=300.0) as client:
                try:
                    response = await client.request(
                        method,
                        target_url,
                        headers=forward_headers,
                        params=query_params,
                        content=content if content else None,
                    )

                    response_data = {
                        "timestamp": datetime.now().isoformat(),
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                        "body": response.json()
                        if response.headers.get("content-type", "").startswith("application/json")
                        else response.text,
                    }
                    if lease is not None and response.status_code < 400:
                        await self._slot_cache.save(selected_endpoint, correlation_id, lease.slot, slot_model)
                    if should_log:
                        await self._save_response(base_filename, response_data, correlation_id, endpoint_path)
                    if self._ledger_service and base_filename:
                        body_json = response_data.get("body") if isinstance(response_data.get("body"), dict) else None
                        await self._ledger_service.validate_and_record(
                            trace_id=base_filename,
                            model=requested_model,
                            endpoint=endpoint_path,
                            request_payload=payload,
                            response_body=body_json,
                            response_chunks=[],
                        )
                    if self._cost_recorder and base_filename:
                        await self._cost_recorder.record_usage(
                            trace_id=base_filename,
                            model=requested_model,
                            provider=selected_endpoint.name,
                            endpoint=endpoint_path,
                            response_body=response_data.get("body"),
                            response_chunks=[],
                        )

                    if response.status_code >= 400:
                        self._logger.warning(
                            "proxy_upstream_error method=%s target_url=%s status=%s",
                            method,
                            target_url,
                            response.status_code,
                        )

                    return {
                        "type": "response",
                        "content": response.content,
                        "status_code": response.status_code,
                        "headers": filter_response_headers(dict(response.headers)),
                    }
                except Exception as e:
                    self._logger.exception(
                        "proxy_request_error method=%s target_url=%s: %s",
                        method,
                        target_url,
                        e,
                    )
                    response_data = {
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e),
                    }
                    if should_log:
                        await self._save_response(base_filename, response_data, correlation_id, endpoint_path)

                    return {
                        "type": "response",
                        "content": json.dumps({"error": str(e)}).encode("utf-8"),
                        "status_code": 500,
                        "headers": {"content-type": "application/json"},
                    }

    async def _proxy_chat_via_anthropic(
        self,
        payload: dict,
        endpoint: EndpointConfig,
        headers: dict[str, str],
        should_log: bool,
        base_filename: str,
        correlation_id: str | None,
    ) -> dict:
        """Forward an OpenAI chat completions request to an Anthropic-protocol
        upstream, translating the request, response, and stream formats."""
        anthropic_payload = chat_request_to_anthropic(payload)
        forward_headers = build_anthropic_headers(headers, endpoint.api_key)
        target_url = f"{endpoint.base_url}/messages"
        model = str(payload.get("model") or "")
        self._logger.debug(
            "proxy_forward_anthropic target_url=%s streaming=%s should_log=%s",
            target_url,
            bool(payload.get("stream")),
            should_log,
        )

        if payload.get("stream"):
            return {
                "type": "stream",
                "iterator": self._anthropic_chat_stream(
                    anthropic_payload, target_url, forward_headers, model,
                    payload, should_log, base_filename, correlation_id, endpoint.name,
                ),
            }

        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                upstream = await client.post(
                    target_url, headers=forward_headers, json=anthropic_payload
                )
            except Exception as exc:
                self._logger.exception(
                    "proxy_anthropic_request_error target_url=%s: %s", target_url, exc
                )
                if should_log:
                    await self._save_response(
                        base_filename,
                        {"timestamp": datetime.now().isoformat(), "error": str(exc)},
                        correlation_id,
                        "/chat/completions",
                    )
                return {
                    "type": "response",
                    "content": json.dumps(openai_error_body(502, str(exc))).encode("utf-8"),
                    "status_code": 502,
                    "headers": {"content-type": "application/json"},
                }

        if upstream.status_code != 200:
            self._logger.warning(
                "proxy_anthropic_upstream_error target_url=%s status=%s body=%s",
                target_url,
                upstream.status_code,
                upstream.text[:500],
            )
            error = openai_error_body(
                upstream.status_code, extract_error_message(upstream.content)
            )
            if should_log:
                await self._save_response(
                    base_filename,
                    {
                        "timestamp": datetime.now().isoformat(),
                        "status_code": upstream.status_code,
                        "headers": dict(upstream.headers),
                        "body": error,
                    },
                    correlation_id,
                    "/chat/completions",
                )
            return {
                "type": "response",
                "content": json.dumps(error).encode("utf-8"),
                "status_code": upstream.status_code,
                "headers": {"content-type": "application/json"},
            }

        try:
            upstream_data = upstream.json()
        except Exception:
            self._logger.warning(
                "proxy_anthropic_upstream_non_json target_url=%s body=%s",
                target_url,
                upstream.text[:200],
            )
            error = openai_error_body(502, "Upstream returned a non-JSON response")
            return {
                "type": "response",
                "content": json.dumps(error).encode("utf-8"),
                "status_code": 502,
                "headers": {"content-type": "application/json"},
            }
        result = anthropic_response_to_chat(upstream_data, model)
        if should_log:
            await self._save_response(
                base_filename,
                {
                    "timestamp": datetime.now().isoformat(),
                    "status_code": 200,
                    "headers": {"content-type": "application/json"},
                    "body": result,
                },
                correlation_id,
                "/chat/completions",
            )
        if self._ledger_service and base_filename:
            await self._ledger_service.validate_and_record(
                trace_id=base_filename,
                model=payload.get("model"),
                endpoint="/chat/completions",
                request_payload=payload,
                response_body=result,
                response_chunks=[],
            )
        if self._cost_recorder and base_filename:
            await self._cost_recorder.record_usage(
                trace_id=base_filename,
                model=payload.get("model"),
                provider=endpoint.name,
                endpoint="/chat/completions",
                response_body=upstream_data,
                response_chunks=[],
            )

        return {
            "type": "response",
            "content": json.dumps(result).encode("utf-8"),
            "status_code": 200,
            "headers": {"content-type": "application/json"},
        }

    async def _anthropic_chat_stream(
        self,
        anthropic_payload: dict,
        target_url: str,
        forward_headers: dict[str, str],
        model: str,
        request_payload: dict,
        should_log: bool,
        base_filename: str,
        correlation_id: str | None,
        provider: str | None = None,
    ):
        translator = AnthropicToChatStream(model=model)
        chunks_log: list[str] = []

        def emit(chunks: list[bytes]) -> list[bytes]:
            if should_log:
                for chunk in chunks:
                    chunks_log.append(chunk.decode("utf-8", errors="replace"))
            return chunks

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST", target_url, headers=forward_headers, json=anthropic_payload
                ) as upstream:
                    if upstream.status_code != 200:
                        error_body = (await upstream.aread()).decode("utf-8", errors="replace")
                        self._logger.warning(
                            "proxy_anthropic_stream_upstream_error target_url=%s status=%s body=%s",
                            target_url,
                            upstream.status_code,
                            error_body[:500],
                        )
                        error = openai_error_body(
                            upstream.status_code, extract_error_message(error_body)
                        )
                        chunk = f"data: {json.dumps(error)}\n\ndata: [DONE]\n\n"
                        chunks_log.append(chunk)
                        yield chunk.encode("utf-8")
                        return

                    async for line in upstream.aiter_lines():
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        try:
                            event = json.loads(data_str)
                        except Exception:
                            continue
                        for chunk in emit(translator.process_event(event)):
                            yield chunk

            for chunk in emit(translator.finalize()):
                yield chunk
        except Exception as exc:
            self._logger.exception("proxy_anthropic_stream_error target_url=%s: %s", target_url, exc)
            error = openai_error_body(500, str(exc))
            chunk = f"data: {json.dumps(error)}\n\ndata: [DONE]\n\n"
            chunks_log.append(chunk)
            yield chunk.encode("utf-8")
        finally:
            if should_log:
                await self._save_response(
                    base_filename,
                    {
                        "timestamp": datetime.now().isoformat(),
                        "status_code": 200,
                        "headers": {"content-type": "text/event-stream"},
                        "chunks": chunks_log,
                    },
                    correlation_id,
                    "/chat/completions",
                )
            if self._ledger_service and base_filename:
                await self._ledger_service.validate_and_record(
                    trace_id=base_filename,
                    model=request_payload.get("model"),
                    endpoint="/chat/completions",
                    request_payload=request_payload,
                    response_body=None,
                    response_chunks=chunks_log,
                )
            if self._cost_recorder and base_filename:
                await self._cost_recorder.record_usage(
                    trace_id=base_filename,
                    model=request_payload.get("model"),
                    provider=provider,
                    endpoint="/chat/completions",
                    response_body=None,
                    response_chunks=chunks_log,
                )

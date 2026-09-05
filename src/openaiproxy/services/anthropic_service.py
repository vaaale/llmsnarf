"""Inbound Anthropic Messages API endpoint (/v1/messages).

Requests routed to an Anthropic-protocol upstream are passed through
natively so provider-specific content (thinking signatures, cache_control,
etc.) survives the round trip. Requests routed to an OpenAI-protocol
upstream are translated to chat completions and the response is translated
back, for both streaming and non-streaming requests.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

import httpx

from openaiproxy.interface.repository import LLMProxyConfigRepository
from openaiproxy.models.models import EndpointConfig
from openaiproxy.services.anthropic_translation import (
    AnthropicToChatStream,
    ChatToAnthropicStream,
    anthropic_error_body,
    anthropic_request_to_chat,
    build_anthropic_headers,
    build_openai_headers,
    chat_response_to_anthropic,
    extract_client_credential,
    extract_error_message,
)
from openaiproxy.services.ledger_service import LedgerService
from openaiproxy.services.model_tracker_service import ModelTrackerService
from openaiproxy.services.proxy_service import (
    extract_correlation_id,
    filter_response_headers,
    save_request_trace,
    save_response_trace,
)


def _json_response(body: dict, status_code: int) -> dict:
    return {
        "type": "response",
        "content": json.dumps(body).encode("utf-8"),
        "status_code": status_code,
        "headers": {"content-type": "application/json"},
    }


class AnthropicService:
    def __init__(
        self,
        config_repository: LLMProxyConfigRepository,
        trace_dir: Path,
        logger: logging.Logger,
        model_tracker: ModelTrackerService,
        ledger_service: LedgerService | None = None,
    ):
        self._config_repository = config_repository
        self._trace_dir = trace_dir
        self._logger = logger
        self._model_tracker = model_tracker
        self._ledger_service = ledger_service

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
            return _json_response(
                anthropic_error_body(400, "Request body must be a JSON object"), 400
            )

        requested_model = payload.get("model")
        correlation_id = extract_correlation_id(headers)
        endpoint = self._select_endpoint_for_model(requested_model)
        if not endpoint:
            config = self._config_repository.load()
            if not config.endpoints:
                self._logger.error(
                    "anthropic_no_routes_configured requested_model=%s: no endpoints defined in config",
                    requested_model,
                )
                return _json_response(
                    anthropic_error_body(500, "No llmproxy routes configured"), 500
                )
            self._logger.warning(
                "anthropic_no_route_for_model requested_model=%s available_endpoints=%s",
                requested_model,
                [e.name for e in config.endpoints],
            )
            return _json_response(
                anthropic_error_body(
                    404, f"No route configured for requested model: {requested_model}"
                ),
                404,
            )

        forwarded_model = endpoint.aliases.get(str(requested_model), requested_model)
        self._logger.info(
            "anthropic_route requested_model=%s forwarded_model=%s endpoint=%s protocol=%s stream=%s correlation_id=%s",
            requested_model,
            forwarded_model,
            endpoint.name,
            endpoint.protocol,
            bool(payload.get("stream")),
            correlation_id,
        )

        if forwarded_model is not None:
            await self._model_tracker.ensure_capacity(endpoint, str(forwarded_model))

        should_log = bool(endpoint.log)
        api_key = extract_client_credential(headers) or endpoint.api_key or "unknown"
        base_filename = (
            await save_request_trace(
                self._trace_dir, "/messages", payload, headers, api_key, correlation_id, provider=endpoint.name
            )
            if should_log
            else ""
        )

        if endpoint.protocol == "anthropic":
            return await self._passthrough(
                payload, forwarded_model, endpoint, headers,
                should_log, base_filename, correlation_id,
            )
        return await self._via_chat_completions(
            payload, forwarded_model, endpoint, headers,
            should_log, base_filename, correlation_id,
        )

    async def _trace_response(
        self, should_log: bool, base_filename: str, response_data: dict, correlation_id: str | None
    ) -> None:
        if should_log:
            await save_response_trace(
                self._trace_dir, base_filename, response_data, correlation_id, endpoint_path="/messages"
            )

    # ------------------------------------------------------------------
    # Anthropic client -> Anthropic upstream (native passthrough)
    # ------------------------------------------------------------------

    async def _passthrough(
        self,
        payload: dict,
        forwarded_model: str | None,
        endpoint: EndpointConfig,
        headers: dict[str, str],
        should_log: bool,
        base_filename: str,
        correlation_id: str | None,
    ) -> dict:
        if forwarded_model is not None and forwarded_model != payload.get("model"):
            payload = {**payload, "model": forwarded_model}
        forward_headers = build_anthropic_headers(headers, endpoint.api_key)
        target_url = f"{endpoint.base_url}/messages"

        if payload.get("stream"):
            return {
                "type": "stream",
                "iterator": self._passthrough_stream(
                    payload, target_url, forward_headers,
                    should_log, base_filename, correlation_id, endpoint.name,
                ),
            }

        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                upstream = await client.post(target_url, headers=forward_headers, json=payload)
            except Exception as exc:
                self._logger.exception(
                    "anthropic_passthrough_connect_error target_url=%s: %s", target_url, exc
                )
                await self._trace_response(
                    should_log, base_filename,
                    {"timestamp": datetime.now().isoformat(), "error": str(exc)},
                    correlation_id,
                )
                return _json_response(anthropic_error_body(502, str(exc)), 502)

        body_json = None
        if upstream.headers.get("content-type", "").startswith("application/json"):
            try:
                body_json = upstream.json()
            except Exception:
                body_json = None
        await self._trace_response(
            should_log,
            base_filename,
            {
                "timestamp": datetime.now().isoformat(),
                "status_code": upstream.status_code,
                "headers": dict(upstream.headers),
                "body": body_json if body_json is not None else upstream.text,
            },
            correlation_id,
        )
        if upstream.status_code >= 400:
            self._logger.warning(
                "anthropic_passthrough_upstream_error target_url=%s status=%s",
                target_url,
                upstream.status_code,
            )
        return {
            "type": "response",
            "content": upstream.content,
            "status_code": upstream.status_code,
            "headers": filter_response_headers(dict(upstream.headers)),
        }

    async def _passthrough_stream(
        self,
        payload: dict,
        target_url: str,
        forward_headers: dict[str, str],
        should_log: bool,
        base_filename: str,
        correlation_id: str | None,
        provider: str | None = None,
    ) -> AsyncIterator[bytes]:
        chunks_log: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST", target_url, headers=forward_headers, json=payload
                ) as upstream:
                    if upstream.status_code != 200:
                        error_body = (await upstream.aread()).decode("utf-8", errors="replace")
                        self._logger.warning(
                            "anthropic_passthrough_stream_error target_url=%s status=%s body=%s",
                            target_url,
                            upstream.status_code,
                            error_body[:500],
                        )
                        message = extract_error_message(error_body)
                        event = anthropic_error_body(upstream.status_code, message)
                        chunk = f"event: error\ndata: {json.dumps(event)}\n\n"
                        chunks_log.append(chunk)
                        yield chunk.encode("utf-8")
                        return
                    async for chunk in upstream.aiter_bytes():
                        if should_log:
                            chunks_log.append(chunk.decode("utf-8", errors="replace"))
                        yield chunk
        except Exception as exc:
            self._logger.exception("anthropic_passthrough_stream_error: %s", exc)
            event = anthropic_error_body(502, str(exc))
            chunk = f"event: error\ndata: {json.dumps(event)}\n\n"
            chunks_log.append(chunk)
            yield chunk.encode("utf-8")
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
                correlation_id,
            )

    # ------------------------------------------------------------------
    # Anthropic client -> OpenAI upstream (translated)
    # ------------------------------------------------------------------

    def _build_chat_payload(
        self, payload: dict, forwarded_model: str | None, endpoint: EndpointConfig
    ) -> dict:
        chat_payload = anthropic_request_to_chat(payload)
        chat_payload["model"] = forwarded_model
        substitutions = endpoint.substitute_role or {}
        for message in chat_payload["messages"]:
            if message["role"] in substitutions:
                message["role"] = substitutions[message["role"]]
        return chat_payload

    async def _via_chat_completions(
        self,
        payload: dict,
        forwarded_model: str | None,
        endpoint: EndpointConfig,
        headers: dict[str, str],
        should_log: bool,
        base_filename: str,
        correlation_id: str | None,
    ) -> dict:
        chat_payload = self._build_chat_payload(payload, forwarded_model, endpoint)
        forward_headers = build_openai_headers(headers, endpoint.api_key)
        chat_url = f"{endpoint.base_url}/chat/completions"

        if payload.get("stream"):
            return {
                "type": "stream",
                "iterator": self._translated_stream(
                    payload, chat_payload, chat_url, forward_headers,
                    should_log, base_filename, correlation_id, endpoint.name,
                ),
            }

        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                upstream = await client.post(chat_url, headers=forward_headers, json=chat_payload)
            except Exception as exc:
                self._logger.exception(
                    "anthropic_upstream_connect_error chat_url=%s: %s", chat_url, exc
                )
                await self._trace_response(
                    should_log, base_filename,
                    {"timestamp": datetime.now().isoformat(), "error": str(exc)},
                    correlation_id,
                )
                return _json_response(anthropic_error_body(502, str(exc)), 502)

        if upstream.status_code != 200:
            self._logger.warning(
                "anthropic_upstream_error chat_url=%s status=%s body=%s",
                chat_url,
                upstream.status_code,
                upstream.text[:500],
            )
            error = anthropic_error_body(
                upstream.status_code, extract_error_message(upstream.content)
            )
            await self._trace_response(
                should_log,
                base_filename,
                {
                    "timestamp": datetime.now().isoformat(),
                    "status_code": upstream.status_code,
                    "headers": dict(upstream.headers),
                    "body": error,
                },
                correlation_id,
            )
            return _json_response(error, upstream.status_code)

        try:
            data = upstream.json()
        except Exception:
            self._logger.warning(
                "anthropic_upstream_non_json chat_url=%s body=%s", chat_url, upstream.text[:200]
            )
            return _json_response(
                anthropic_error_body(502, "Upstream returned a non-JSON response"), 502
            )
        result = chat_response_to_anthropic(data, str(payload.get("model") or ""))

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
                endpoint="/messages",
                request_payload=chat_payload,
                response_body=data,
                response_chunks=[],
            )

        return _json_response(result, 200)

    async def _translated_stream(
        self,
        payload: dict,
        chat_payload: dict,
        chat_url: str,
        forward_headers: dict[str, str],
        should_log: bool,
        base_filename: str,
        correlation_id: str | None,
        provider: str | None = None,
    ) -> AsyncIterator[bytes]:
        chat_payload = dict(chat_payload)
        chat_payload["stream"] = True
        chat_payload["stream_options"] = {"include_usage": True}

        translator = ChatToAnthropicStream(model=str(payload.get("model") or ""))
        chunks_log: list[str] = []

        def emit(events: list[bytes]) -> list[bytes]:
            if should_log:
                for event in events:
                    chunks_log.append(event.decode("utf-8", errors="replace"))
            return events

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST", chat_url, headers=forward_headers, json=chat_payload
                ) as upstream:
                    if upstream.status_code != 200:
                        error_body = (await upstream.aread()).decode("utf-8", errors="replace")
                        self._logger.warning(
                            "anthropic_stream_upstream_error chat_url=%s status=%s body=%s",
                            chat_url,
                            upstream.status_code,
                            error_body[:500],
                        )
                        for event in emit(translator.error(extract_error_message(error_body))):
                            yield event
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
                        for event in emit(translator.process_chunk(chunk)):
                            yield event

            for event in emit(translator.finalize()):
                yield event
        except Exception as exc:
            self._logger.exception("anthropic_stream_error: %s", exc)
            for event in emit(translator.error(str(exc))):
                yield event
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
                correlation_id,
            )
            if self._ledger_service and base_filename:
                await self._ledger_service.validate_and_record(
                    trace_id=base_filename,
                    model=payload.get("model"),
                    endpoint="/messages",
                    request_payload=chat_payload,
                    response_body=None,
                    response_chunks=chunks_log,
                )

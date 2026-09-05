"""OpenAI-compatible /v1/embeddings endpoint.

Requests are routed by model name with the same rules as the other APIs
(models list, aliases, wildcard fallback). Endpoints in *remote* mode forward
the request to {base_url}/embeddings — protocol "openai" upstreams get Bearer
auth, protocol "anthropic" upstreams get x-api-key headers (Anthropic itself
has no embeddings API; this matches Voyage-style providers that accept the
same request shape). Endpoints in *local* mode serve the model in-process
with sentence-transformers.
"""

from __future__ import annotations

import base64
import json
import logging
import struct
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import anyio
import httpx

from openaiproxy.interface.repository import LLMProxyConfigRepository
from openaiproxy.models.models import EndpointConfig
from openaiproxy.services.anthropic_translation import build_anthropic_headers
from openaiproxy.services.cost_recorder_service import CostRecorderService
from openaiproxy.services.ledger_service import LedgerService
from openaiproxy.services.model_tracker_service import ModelTrackerService
from openaiproxy.services.proxy_service import (
    extract_api_key,
    extract_correlation_id,
    filter_response_headers,
    save_request_trace,
    save_response_trace,
)

EMBEDDINGS_PATH = "/embeddings"


class LocalEmbeddingUnavailable(RuntimeError):
    """Raised when local embedding support is requested but not installed."""


def _error_body(message: str, error_type: str = "invalid_request_error") -> dict:
    return {"error": {"message": message, "type": error_type}}


def _json_response(body: dict, status_code: int) -> dict:
    return {
        "type": "response",
        "content": json.dumps(body).encode("utf-8"),
        "status_code": status_code,
        "headers": {"content-type": "application/json"},
    }


def _vector_to_base64(vector: list[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")


def _parse_input_texts(payload: dict) -> list[str]:
    """Normalize the OpenAI 'input' field to a list of strings.

    Token-array inputs (lists of ints) are rejected: local models embed text,
    and forwarding already covers upstreams that accept token arrays.
    """
    value = payload.get("input")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return list(value)
    raise ValueError(
        "'input' must be a string or an array of strings for locally served embedding models"
    )


class LocalEmbeddingBackend:
    """Serves embedding models in-process via sentence-transformers."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _get_model(self, model_id: str) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise LocalEmbeddingUnavailable(
                "Local embedding endpoints require the 'sentence-transformers' package. "
                "Install it with: pip install 'openaiproxy[local-embeddings]'"
            ) from exc
        with self._lock:
            model = self._models.get(model_id)
            if model is None:
                self._logger.info("local_embedding_load model=%s", model_id)
                model = SentenceTransformer(model_id)
                self._models[model_id] = model
            return model

    @staticmethod
    def _count_tokens(model: Any, texts: list[str]) -> int:
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None:
            try:
                return sum(len(tokenizer.encode(text)) for text in texts)
            except Exception:
                pass
        return sum(max(1, len(text) // 4) for text in texts)

    async def embed(self, model_id: str, texts: list[str]) -> tuple[list[list[float]], int]:
        """Return (vectors, prompt_tokens) for the given texts."""

        def _encode() -> tuple[list[list[float]], int]:
            model = self._get_model(model_id)
            vectors = model.encode(texts)
            return [[float(x) for x in vector] for vector in vectors], self._count_tokens(model, texts)

        return await anyio.to_thread.run_sync(_encode)


class EmbeddingsService:
    def __init__(
        self,
        config_repository: LLMProxyConfigRepository,
        trace_dir: Path,
        logger: logging.Logger,
        model_tracker: ModelTrackerService,
        ledger_service: LedgerService | None = None,
        cost_recorder: CostRecorderService | None = None,
        local_backend: LocalEmbeddingBackend | None = None,
    ):
        self._config_repository = config_repository
        self._trace_dir = trace_dir
        self._logger = logger
        self._model_tracker = model_tracker
        self._ledger_service = ledger_service
        self._cost_recorder = cost_recorder
        self._local_backend = local_backend or LocalEmbeddingBackend(logger)

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
            return _json_response(_error_body("Request body must be a JSON object"), 400)

        requested_model = payload.get("model")
        correlation_id = extract_correlation_id(headers)
        endpoint = self._select_endpoint_for_model(requested_model)
        if not endpoint:
            config = self._config_repository.load()
            if not config.endpoints:
                self._logger.error(
                    "embeddings_no_routes_configured requested_model=%s: no endpoints defined in config",
                    requested_model,
                )
                return _json_response(_error_body("No llmproxy routes configured", "routing_error"), 500)
            self._logger.warning(
                "embeddings_no_route_for_model requested_model=%s available_endpoints=%s",
                requested_model,
                [e.name for e in config.endpoints],
            )
            return _json_response(
                _error_body(f"No route configured for requested model: {requested_model}", "routing_error"),
                404,
            )

        forwarded_model = endpoint.aliases.get(str(requested_model), requested_model)
        self._logger.info(
            "embeddings_route requested_model=%s forwarded_model=%s endpoint=%s mode=%s correlation_id=%s",
            requested_model,
            forwarded_model,
            endpoint.name,
            endpoint.mode,
            correlation_id,
        )

        should_log = bool(endpoint.log)
        api_key = extract_api_key(headers) if "authorization" in headers else (endpoint.api_key or "unknown")
        base_filename = (
            await save_request_trace(
                self._trace_dir, EMBEDDINGS_PATH, payload, headers, api_key, correlation_id, provider=endpoint.name
            )
            if should_log
            else ""
        )

        if endpoint.mode == "local":
            return await self._embed_locally(
                payload, str(requested_model or ""), str(forwarded_model or ""),
                should_log, base_filename, correlation_id, endpoint.name,
            )
        return await self._forward(
            payload, forwarded_model, endpoint, headers,
            should_log, base_filename, correlation_id,
        )

    async def _trace_response(
        self, should_log: bool, base_filename: str, response_data: dict, correlation_id: str | None
    ) -> None:
        if should_log:
            await save_response_trace(
                self._trace_dir, base_filename, response_data, correlation_id, endpoint_path=EMBEDDINGS_PATH
            )

    async def _record_ledger(
        self, base_filename: str, payload: dict, response_body: dict | None, provider: str | None = None
    ) -> None:
        if self._ledger_service and base_filename:
            await self._ledger_service.validate_and_record(
                trace_id=base_filename,
                model=payload.get("model"),
                endpoint=EMBEDDINGS_PATH,
                request_payload=payload,
                response_body=response_body,
                response_chunks=[],
            )
        if self._cost_recorder and base_filename:
            await self._cost_recorder.record_usage(
                trace_id=base_filename,
                model=payload.get("model"),
                provider=provider,
                endpoint=EMBEDDINGS_PATH,
                response_body=response_body,
                response_chunks=[],
            )

    # ------------------------------------------------------------------
    # Local mode — models served in-process
    # ------------------------------------------------------------------

    async def _embed_locally(
        self,
        payload: dict,
        requested_model: str,
        model_id: str,
        should_log: bool,
        base_filename: str,
        correlation_id: str | None,
        provider: str | None = None,
    ) -> dict:
        if not model_id:
            return await self._fail(
                _error_body("Request has no 'model' field"), 400,
                should_log, base_filename, correlation_id, payload, provider,
            )
        try:
            texts = _parse_input_texts(payload)
        except ValueError as exc:
            return await self._fail(
                _error_body(str(exc)), 400, should_log, base_filename, correlation_id, payload, provider
            )

        try:
            vectors, prompt_tokens = await self._local_backend.embed(model_id, texts)
        except LocalEmbeddingUnavailable as exc:
            self._logger.error("local_embedding_unavailable model=%s: %s", model_id, exc)
            return await self._fail(
                _error_body(str(exc), "server_error"), 501,
                should_log, base_filename, correlation_id, payload, provider,
            )
        except Exception as exc:
            self._logger.exception("local_embedding_error model=%s: %s", model_id, exc)
            return await self._fail(
                _error_body(f"Local embedding failed: {exc}", "server_error"), 500,
                should_log, base_filename, correlation_id, payload, provider,
            )

        encoding_format = payload.get("encoding_format") or "float"
        data = []
        for index, vector in enumerate(vectors):
            embedding: Any = _vector_to_base64(vector) if encoding_format == "base64" else vector
            data.append({"object": "embedding", "index": index, "embedding": embedding})

        result = {
            "object": "list",
            "data": data,
            "model": requested_model or model_id,
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        }
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
        await self._record_ledger(base_filename, payload, result, provider)
        return _json_response(result, 200)

    async def _fail(
        self,
        error: dict,
        status_code: int,
        should_log: bool,
        base_filename: str,
        correlation_id: str | None,
        payload: dict,
        provider: str | None = None,
    ) -> dict:
        await self._trace_response(
            should_log,
            base_filename,
            {
                "timestamp": datetime.now().isoformat(),
                "status_code": status_code,
                "headers": {"content-type": "application/json"},
                "body": error,
            },
            correlation_id,
        )
        await self._record_ledger(base_filename, payload, error, provider)
        return _json_response(error, status_code)

    # ------------------------------------------------------------------
    # Remote mode — forward to the upstream endpoint
    # ------------------------------------------------------------------

    async def _forward(
        self,
        payload: dict,
        forwarded_model: Any,
        endpoint: EndpointConfig,
        headers: dict[str, str],
        should_log: bool,
        base_filename: str,
        correlation_id: str | None,
    ) -> dict:
        if forwarded_model is not None and forwarded_model != payload.get("model"):
            payload = {**payload, "model": forwarded_model}

        if forwarded_model is not None:
            await self._model_tracker.ensure_capacity(endpoint, str(forwarded_model))

        forward_headers = {"Content-Type": "application/json"}
        if endpoint.protocol == "anthropic":
            forward_headers.update(build_anthropic_headers(headers, endpoint.api_key))
        elif "authorization" in headers:
            forward_headers["Authorization"] = headers["authorization"]
        elif endpoint.api_key:
            forward_headers["Authorization"] = f"Bearer {endpoint.api_key}"

        target_url = f"{endpoint.base_url}{EMBEDDINGS_PATH}"
        self._logger.debug("embeddings_forward target_url=%s should_log=%s", target_url, should_log)

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(target_url, headers=forward_headers, json=payload)
        except Exception as exc:
            self._logger.exception("embeddings_upstream_error target_url=%s: %s", target_url, exc)
            await self._trace_response(
                should_log,
                base_filename,
                {"timestamp": datetime.now().isoformat(), "error": str(exc)},
                correlation_id,
            )
            return _json_response(_error_body(str(exc), "server_error"), 502)

        body_json: dict | None = None
        if response.headers.get("content-type", "").startswith("application/json"):
            try:
                parsed = response.json()
                body_json = parsed if isinstance(parsed, dict) else None
            except Exception:
                body_json = None

        if response.status_code >= 400:
            self._logger.warning(
                "embeddings_upstream_status target_url=%s status=%s", target_url, response.status_code
            )

        await self._trace_response(
            should_log,
            base_filename,
            {
                "timestamp": datetime.now().isoformat(),
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": body_json if body_json is not None else response.text,
            },
            correlation_id,
        )
        await self._record_ledger(base_filename, payload, body_json, endpoint.name)

        return {
            "type": "response",
            "content": response.content,
            "status_code": response.status_code,
            "headers": filter_response_headers(dict(response.headers)),
        }

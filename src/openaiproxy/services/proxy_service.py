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


def extract_api_key(headers: dict[str, str]) -> str:
    auth_header = headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    return "unknown"


def sanitize_api_key_for_filename(api_key: str) -> str:
    return api_key.replace("/", "_").replace("\\", "_").replace(":", "_")


class ProxyService:
    def __init__(self, config_repository: LLMProxyConfigRepository, trace_dir: Path, logger: logging.Logger):
        self._config_repository = config_repository
        self._trace_dir = trace_dir
        self._logger = logger

    def _select_endpoint_for_model(self, model: str | None) -> EndpointConfig | None:
        config = self._config_repository.load()
        endpoints = config.endpoints
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

    async def _save_request(self, endpoint_path: str, payload: Any, headers: dict[str, str], api_key: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_api_key = sanitize_api_key_for_filename(api_key)
        base_filename = f"{safe_api_key}_{timestamp}"
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        filename = self._trace_dir / f"{base_filename}_request.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint_path,
            "headers": dict(headers),
            "payload": payload,
        }

        def _write() -> None:
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)

        await anyio.to_thread.run_sync(_write)
        return base_filename

    async def _save_response(self, base_filename: str, response_data: dict[str, Any]) -> None:
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        filename = self._trace_dir / f"{base_filename}_response.json"

        def _write() -> None:
            with open(filename, "w") as f:
                json.dump(response_data, f, indent=2)

        await anyio.to_thread.run_sync(_write)

    async def proxy(self, method: str, endpoint_path: str, body: bytes, headers: dict[str, str], query_params: dict[str, str]):
        payload: Any = {}
        body_to_forward = body
        if body:
            try:
                payload = json.loads(body)
            except Exception:
                payload = {
                    "raw_body_base64": base64.b64encode(body).decode("ascii"),
                    "content_type": headers.get("content-type", ""),
                }

        requested_model = payload.get("model") if isinstance(payload, dict) else None
        forwarded_model = requested_model

        selected_endpoint = self._select_endpoint_for_model(requested_model)

        if selected_endpoint and isinstance(payload, dict) and forwarded_model is not None:
            forwarded_model = selected_endpoint.aliases.get(str(forwarded_model), forwarded_model)
            if forwarded_model != requested_model:
                payload["model"] = forwarded_model
                body_to_forward = json.dumps(payload).encode("utf-8")

        self._logger.info(
            "route_resolve requested_model=%s forwarded_model=%s endpoint=%s",
            requested_model,
            forwarded_model,
            (selected_endpoint.name if selected_endpoint else None),
        )
        if not selected_endpoint:
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

        should_log = bool(selected_endpoint.log)
        api_key = extract_api_key(headers) if "authorization" in headers else (selected_endpoint.api_key or "unknown")
        base_filename = await self._save_request(endpoint_path, payload, headers, api_key) if should_log else ""

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

        if "authorization" in headers:
            forward_headers["Authorization"] = headers["authorization"]
        elif selected_endpoint.api_key:
            forward_headers["Authorization"] = f"Bearer {selected_endpoint.api_key}"

        target_url = f"{selected_endpoint.base_url}{endpoint_path}"
        is_streaming = isinstance(payload, dict) and payload.get("stream", False)

        if is_streaming:
            async def stream_response():
                async with httpx.AsyncClient(timeout=300.0) as client:
                    try:
                        async with client.stream(
                            method,
                            target_url,
                            headers=forward_headers,
                            params=query_params,
                            content=body_to_forward if body_to_forward else None,
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

                            if should_log:
                                await self._save_response(base_filename, response_data)
                    except Exception as e:
                        error_msg = f"data: {json.dumps({'error': str(e)})}\n\n"
                        response_data = {
                            "timestamp": datetime.now().isoformat(),
                            "error": str(e),
                        }
                        if should_log:
                            await self._save_response(base_filename, response_data)
                        yield error_msg.encode()

            return {
                "type": "stream",
                "iterator": stream_response(),
            }

        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                response = await client.request(
                    method,
                    target_url,
                    headers=forward_headers,
                    params=query_params,
                    content=body_to_forward if body_to_forward else None,
                )

                response_data = {
                    "timestamp": datetime.now().isoformat(),
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": response.json()
                    if response.headers.get("content-type", "").startswith("application/json")
                    else response.text,
                }
                if should_log:
                    await self._save_response(base_filename, response_data)

                return {
                    "type": "response",
                    "content": response.content,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                }
            except Exception as e:
                response_data = {
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                }
                if should_log:
                    await self._save_response(base_filename, response_data)

                return {
                    "type": "response",
                    "content": json.dumps({"error": str(e)}).encode("utf-8"),
                    "status_code": 500,
                    "headers": {"content-type": "application/json"},
                }

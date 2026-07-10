from __future__ import annotations

import asyncio
import logging
import time

import httpx

from openaiproxy.models.models import EndpointConfig


def _management_root(base_url: str) -> str:
    # base_url is normally "http://host:port/v1"; /models/unload lives at the server root
    return base_url[: -len("/v1")] if base_url.endswith("/v1") else base_url


class ModelTrackerService:
    """Tracks which models are loaded per endpoint and unloads the least recently
    used model when an endpoint's max_models limit would be exceeded."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._loaded: dict[str, dict[str, float]] = {}  # endpoint name -> {model id: last used}
        self._initialized: set[str] = set()
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, endpoint_name: str) -> asyncio.Lock:
        if endpoint_name not in self._locks:
            self._locks[endpoint_name] = asyncio.Lock()
        return self._locks[endpoint_name]

    @staticmethod
    def _headers(endpoint: EndpointConfig) -> dict[str, str]:
        if endpoint.api_key:
            return {"Authorization": f"Bearer {endpoint.api_key}"}
        return {}

    async def _initialize(self, client: httpx.AsyncClient, endpoint: EndpointConfig) -> None:
        loaded: dict[str, float] = {}
        try:
            response = await client.get(f"{endpoint.base_url}/models", headers=self._headers(endpoint))
            response.raise_for_status()
            data = response.json()
            for entry in data.get("data") or []:
                if not isinstance(entry, dict) or not entry.get("id"):
                    continue
                status = entry.get("status")
                if isinstance(status, dict) and status.get("value") == "loaded":
                    loaded[str(entry["id"])] = time.monotonic()
        except Exception as exc:
            self._logger.warning("model_tracker init failed endpoint=%s error=%s", endpoint.name, exc)
        self._loaded[endpoint.name] = loaded
        self._initialized.add(endpoint.name)
        self._logger.info("model_tracker init endpoint=%s loaded=%s", endpoint.name, sorted(loaded))

    async def _unload(self, client: httpx.AsyncClient, endpoint: EndpointConfig, model: str) -> None:
        url = f"{_management_root(endpoint.base_url)}/models/unload"
        try:
            response = await client.post(url, headers=self._headers(endpoint), json={"model": model})
            if response.status_code >= 400:
                self._logger.warning(
                    "model_tracker unload failed endpoint=%s model=%s status=%s body=%s",
                    endpoint.name,
                    model,
                    response.status_code,
                    response.text[:300],
                )
            else:
                self._logger.info("model_tracker unloaded endpoint=%s model=%s", endpoint.name, model)
        except Exception as exc:
            self._logger.warning("model_tracker unload error endpoint=%s model=%s error=%s", endpoint.name, model, exc)

    async def ensure_capacity(self, endpoint: EndpointConfig, model: str | None) -> None:
        if endpoint.max_models < 1 or not model:
            return
        async with self._lock(endpoint.name):
            async with httpx.AsyncClient(timeout=60.0) as client:
                if endpoint.name not in self._initialized:
                    await self._initialize(client, endpoint)

                loaded = self._loaded.setdefault(endpoint.name, {})
                if model in loaded:
                    loaded[model] = time.monotonic()
                    return

                while len(loaded) >= endpoint.max_models:
                    lru_model = min(loaded, key=loaded.__getitem__)
                    await self._unload(client, endpoint, lru_model)
                    del loaded[lru_model]

                loaded[model] = time.monotonic()

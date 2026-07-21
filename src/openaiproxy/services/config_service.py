from __future__ import annotations

from dataclasses import replace

from openaiproxy.interface.repository import LLMProxyConfigRepository
from openaiproxy.models.models import EndpointConfig, LLMProxyConfig, WebFetchConfig, WebSearchConfig


class ConfigValidationError(ValueError):
    pass


class ConfigService:
    def __init__(self, config_repository: LLMProxyConfigRepository):
        self._config_repository = config_repository

    def get_config(self) -> LLMProxyConfig:
        return self._config_repository.load()

    def update_endpoints(self, endpoints: list[EndpointConfig]) -> LLMProxyConfig:
        self._validate(endpoints)
        current = self._config_repository.load()
        existing_keys = {endpoint.name: endpoint.api_key for endpoint in current.endpoints}

        resolved: list[EndpointConfig] = []
        for endpoint in endpoints:
            if endpoint.api_key == "":
                endpoint = replace(endpoint, api_key=existing_keys.get(endpoint.name, ""))
            resolved.append(endpoint)

        updated = replace(current, endpoints=resolved)
        self._config_repository.save(updated)
        return updated

    def update_web_search(self, web_search: WebSearchConfig) -> LLMProxyConfig:
        current = self._config_repository.load()
        updated = replace(current, web_search=web_search)
        self._config_repository.save(updated)
        return updated

    def update_web_fetch(self, web_fetch: WebFetchConfig) -> LLMProxyConfig:
        current = self._config_repository.load()
        updated = replace(current, web_fetch=web_fetch)
        self._config_repository.save(updated)
        return updated

    def resolve_route(self, model: str) -> EndpointConfig | None:
        config = self._config_repository.load()
        endpoints = [endpoint for endpoint in config.endpoints if endpoint.enabled]

        for endpoint in endpoints:
            if model in endpoint.models or model in endpoint.aliases:
                return endpoint

        for endpoint in endpoints:
            if "*" in endpoint.models:
                return endpoint

        return None

    @staticmethod
    def _validate(endpoints: list[EndpointConfig]) -> None:
        seen: set[str] = set()
        for endpoint in endpoints:
            if not endpoint.name:
                raise ConfigValidationError("Endpoint name is required")
            if endpoint.name in seen:
                raise ConfigValidationError(f"Duplicate endpoint name: {endpoint.name}")
            seen.add(endpoint.name)
            if not endpoint.base_url:
                raise ConfigValidationError(f"Endpoint '{endpoint.name}': base_url is required")
            if not endpoint.models:
                raise ConfigValidationError(f"Endpoint '{endpoint.name}': at least one model is required")
            if endpoint.protocol not in ("openai", "anthropic"):
                raise ConfigValidationError(
                    f"Endpoint '{endpoint.name}': protocol must be 'openai' or 'anthropic'"
                )

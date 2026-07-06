from __future__ import annotations

from pathlib import Path

import yaml

from openaiproxy.interface.repository import LLMProxyConfigRepository
from openaiproxy.models.models import EndpointConfig, LLMProxyConfig


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


class YAMLLLMProxyConfigRepository(LLMProxyConfigRepository):
    def __init__(self, config_path: Path):
        self._config_path = config_path

    def load(self) -> LLMProxyConfig:
        if not self._config_path.exists():
            return LLMProxyConfig(host=None, port=None, logs_dir=None, trace_dir=None, endpoints=[])

        with open(self._config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        llmproxy = data.get("llmproxy") or {}
        host = llmproxy.get("host")
        port = llmproxy.get("port")
        logs_dir = llmproxy.get("logs_dir")
        trace_dir = llmproxy.get("trace_dir")
        endpoints_raw = llmproxy.get("endpoints") or {}

        endpoints: list[EndpointConfig] = []
        if isinstance(endpoints_raw, dict):
            for name, entry in endpoints_raw.items():
                if not isinstance(entry, dict):
                    continue

                base_url = entry.get("base_url")
                api_key = entry.get("api_key")
                log_enabled = entry.get("log")
                models = entry.get("models")
                aliases_raw = entry.get("aliases")
                substitute_role_raw = entry.get("substitute_role")
                enabled = entry.get("enabled")

                if not base_url or not models:
                    continue

                if isinstance(models, str):
                    models = [models]
                if not isinstance(models, list):
                    continue

                aliases: dict[str, str] = {}
                if isinstance(aliases_raw, dict):
                    aliases = {str(k): str(v) for k, v in aliases_raw.items()}

                substitute_role: dict[str, str] = {}
                if isinstance(substitute_role_raw, dict):
                    substitute_role = {str(k): str(v) for k, v in substitute_role_raw.items()}

                endpoints.append(
                    EndpointConfig(
                        name=str(name),
                        base_url=_normalize_base_url(str(base_url)),
                        api_key=str(api_key) if api_key else "",
                        models=[str(m) for m in models],
                        aliases=aliases,
                        log=bool(True if log_enabled is None else log_enabled),
                        substitute_role=substitute_role,
                        enabled=bool(True if enabled is None else enabled),
                    )
                )

        return LLMProxyConfig(
            host=str(host) if host is not None else None,
            port=int(port) if port is not None else None,
            logs_dir=str(logs_dir) if logs_dir is not None else None,
            trace_dir=str(trace_dir) if trace_dir is not None else None,
            endpoints=endpoints,
        )

    def save(self, config: LLMProxyConfig) -> None:
        endpoints: dict[str, dict] = {}
        for endpoint in config.endpoints:
            entry: dict = {
                "base_url": endpoint.base_url,
                "models": list(endpoint.models),
                "log": endpoint.log,
                "enabled": endpoint.enabled,
            }
            if endpoint.api_key:
                entry["api_key"] = endpoint.api_key
            if endpoint.aliases:
                entry["aliases"] = dict(endpoint.aliases)
            if endpoint.substitute_role:
                entry["substitute_role"] = dict(endpoint.substitute_role)
            endpoints[endpoint.name] = entry

        llmproxy: dict = {}
        if config.host is not None:
            llmproxy["host"] = config.host
        if config.port is not None:
            llmproxy["port"] = config.port
        if config.logs_dir is not None:
            llmproxy["logs_dir"] = config.logs_dir
        if config.trace_dir is not None:
            llmproxy["trace_dir"] = config.trace_dir
        llmproxy["endpoints"] = endpoints

        tmp_path = self._config_path.with_suffix(".yaml.tmp")
        with open(tmp_path, "w") as f:
            yaml.safe_dump({"llmproxy": llmproxy}, f, sort_keys=False, default_flow_style=False)
        tmp_path.replace(self._config_path)

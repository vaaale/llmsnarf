from __future__ import annotations

from pathlib import Path

import yaml

from openaiproxy.interface.repository import LLMProxyConfigRepository
from openaiproxy.models.models import EndpointConfig, LLMProxyConfig, WebFetchConfig, WebSearchConfig


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _parse_web_search(raw: object) -> WebSearchConfig:
    defaults = WebSearchConfig()
    if not isinstance(raw, dict):
        return defaults
    raw_engines = raw.get("engines")
    if isinstance(raw_engines, list):
        engines = [str(e) for e in raw_engines]
    elif isinstance(raw_engines, str) and raw_engines:
        engines = [e.strip() for e in raw_engines.split(",") if e.strip()]
    else:
        engines = list(defaults.engines)
    return WebSearchConfig(
        enabled=bool(defaults.enabled if raw.get("enabled") is None else raw.get("enabled")),
        base_url=_normalize_base_url(str(raw.get("base_url") or defaults.base_url)),
        format=str(raw.get("format") or defaults.format),
        extract=int(defaults.extract if raw.get("extract") is None else raw.get("extract")),
        extract_mode=str(raw.get("extract_mode") or defaults.extract_mode),
        limit=int(defaults.limit if raw.get("limit") is None else raw.get("limit")),
        filter=bool(defaults.filter if raw.get("filter") is None else raw.get("filter")),
        mode=str(raw.get("mode") or defaults.mode),
        engines=engines,
        agent_loop=bool(defaults.agent_loop if raw.get("agent_loop") is None else raw.get("agent_loop")),
        max_searches=int(defaults.max_searches if raw.get("max_searches") is None else raw.get("max_searches")),
        map_reduce_context_limit=int(defaults.map_reduce_context_limit if raw.get("map_reduce_context_limit") is None else raw.get("map_reduce_context_limit")),
        map_reduce_call_limit=int(defaults.map_reduce_call_limit if raw.get("map_reduce_call_limit") is None else raw.get("map_reduce_call_limit")),
        map_reduce_chunk_size=int(defaults.map_reduce_chunk_size if raw.get("map_reduce_chunk_size") is None else raw.get("map_reduce_chunk_size")),
        map_reduce_reduce=bool(defaults.map_reduce_reduce if raw.get("map_reduce_reduce") is None else raw.get("map_reduce_reduce")),
    )


def _parse_web_fetch(raw: object) -> WebFetchConfig:
    defaults = WebFetchConfig()
    if not isinstance(raw, dict):
        return defaults
    return WebFetchConfig(
        base_url=_normalize_base_url(str(raw.get("base_url") or defaults.base_url)),
        format=str(raw.get("format") or defaults.format),
        mode=str(raw.get("mode") or defaults.mode),
    )


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
                max_models = entry.get("max_models")
                protocol = entry.get("protocol")
                mode = str(entry.get("mode") or "remote")

                # local endpoints serve models in-process and need no base_url
                if not models or (not base_url and mode != "local"):
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
                        base_url=_normalize_base_url(str(base_url)) if base_url else "",
                        api_key=str(api_key) if api_key else "",
                        models=[str(m) for m in models],
                        aliases=aliases,
                        log=bool(True if log_enabled is None else log_enabled),
                        substitute_role=substitute_role,
                        enabled=bool(True if enabled is None else enabled),
                        max_models=int(max_models) if max_models is not None else 0,
                        protocol=str(protocol) if protocol else "openai",
                        mode=mode,
                    )
                )

        return LLMProxyConfig(
            host=str(host) if host is not None else None,
            port=int(port) if port is not None else None,
            logs_dir=str(logs_dir) if logs_dir is not None else None,
            trace_dir=str(trace_dir) if trace_dir is not None else None,
            endpoints=endpoints,
            web_search=_parse_web_search(llmproxy.get("web_search")),
            web_fetch=_parse_web_fetch(llmproxy.get("web_fetch")),
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
            if endpoint.max_models > 0:
                entry["max_models"] = endpoint.max_models
            if endpoint.protocol and endpoint.protocol != "openai":
                entry["protocol"] = endpoint.protocol
            if endpoint.mode and endpoint.mode != "remote":
                entry["mode"] = endpoint.mode
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
        llmproxy["web_search"] = {
            "enabled": config.web_search.enabled,
            "base_url": config.web_search.base_url,
            "format": config.web_search.format,
            "extract": config.web_search.extract,
            "extract_mode": config.web_search.extract_mode,
            "limit": config.web_search.limit,
            "filter": config.web_search.filter,
            "mode": config.web_search.mode,
            "engines": list(config.web_search.engines),
            "agent_loop": config.web_search.agent_loop,
            "max_searches": config.web_search.max_searches,
            "map_reduce_context_limit": config.web_search.map_reduce_context_limit,
            "map_reduce_call_limit": config.web_search.map_reduce_call_limit,
            "map_reduce_chunk_size": config.web_search.map_reduce_chunk_size,
            "map_reduce_reduce": config.web_search.map_reduce_reduce,
        }
        llmproxy["web_fetch"] = {
            "base_url": config.web_fetch.base_url,
            "format": config.web_fetch.format,
            "mode": config.web_fetch.mode,
        }
        llmproxy["endpoints"] = endpoints

        tmp_path = self._config_path.with_suffix(".yaml.tmp")
        with open(tmp_path, "w") as f:
            yaml.safe_dump({"llmproxy": llmproxy}, f, sort_keys=False, default_flow_style=False)
        tmp_path.replace(self._config_path)

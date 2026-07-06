from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EndpointConfig:
    name: str
    base_url: str
    api_key: str
    models: list[str]
    aliases: dict[str, str]
    log: bool
    substitute_role: dict[str, str] | None
    enabled: bool = True


@dataclass(frozen=True)
class LLMProxyConfig:
    host: str | None
    port: int | None
    logs_dir: str | None
    trace_dir: str | None
    endpoints: list[EndpointConfig]

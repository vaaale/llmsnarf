from __future__ import annotations

from dataclasses import dataclass, field


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
    max_models: int = 0  # max loaded models on the endpoint; < 1 disables tracking


@dataclass(frozen=True)
class WebSearchConfig:
    enabled: bool = True
    base_url: str = "http://wingman.akhbar.lan:7000"
    format: str = "markdown"  # json | markdown | text | ndjson
    extract: int = 3  # 0-5 top results to enrich with page content
    extract_mode: str = "auto"  # auto | fast | rendered
    limit: int = 25  # 1-100
    filter: bool = False
    mode: str = "balanced"  # any | fast | balanced
    engines: list[str] = field(default_factory=list)  # bing, google, yandex, baidu, duckduckgo, ecosia


@dataclass(frozen=True)
class WebFetchConfig:
    format: str = "markdown"  # json | markdown | text | ndjson
    mode: str = "auto"  # auto | fast | rendered


@dataclass(frozen=True)
class LLMProxyConfig:
    host: str | None
    port: int | None
    logs_dir: str | None
    trace_dir: str | None
    endpoints: list[EndpointConfig]
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    web_fetch: WebFetchConfig = field(default_factory=WebFetchConfig)

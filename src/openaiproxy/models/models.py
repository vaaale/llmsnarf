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
    protocol: str = "openai"  # wire format of the upstream: openai | anthropic
    mode: str = "remote"  # remote = forward requests; local = serve embeddings in-process (no base_url needed)
    backend: str = "generic"  # server implementation: generic | llamacpp — orthogonal to protocol, gates backend-specific features
    cache_prompt: bool = False  # add "cache_prompt": true to forwarded request bodies (llama.cpp only)
    slot_cache: bool = False  # save/restore llama-server KV cache slots per conversation (llama.cpp only, X-Correlation-Id)
    slot_count: int = 0  # llama.cpp server slots (-np); 0 = auto-detect via GET /props


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
    agent_loop: bool = True  # feed results back to the model; False = one search round, raw results returned
    max_searches: int = 3  # max total search/fetch calls the proxy runs per client request; <=0 = unlimited
    map_reduce_context_limit: int = 16000  # approximate token threshold to trigger map-reduce
    map_reduce_call_limit: int = 0  # per-call token budget for map/reduce LLM calls; 0 = same as context_limit
    map_reduce_chunk_size: int = 4000  # approximate tokens per chunk
    map_reduce_reduce: bool = True  # do a final reduce call; False = concatenate map results


@dataclass(frozen=True)
class WebFetchConfig:
    base_url: str = "http://wingman.akhbar.lan:7000"
    format: str = "markdown"  # json | markdown | text | ndjson
    mode: str = "auto"  # auto | fast | rendered


@dataclass(frozen=True)
class ModelPricing:
    price_input_per_million: float = 0.0  # $ per 1M input/prompt tokens
    price_output_per_million: float = 0.0  # $ per 1M output/completion tokens


@dataclass(frozen=True)
class LLMProxyConfig:
    host: str | None
    port: int | None
    logs_dir: str | None
    trace_dir: str | None
    endpoints: list[EndpointConfig]
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    web_fetch: WebFetchConfig = field(default_factory=WebFetchConfig)
    pricing: dict[str, ModelPricing] = field(default_factory=dict)  # keyed by model name

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class WebSearchSchema(BaseModel):
    enabled: bool = True
    base_url: str = "http://wingman.akhbar.lan:7000"
    format: Literal["json", "markdown", "text", "ndjson"] = "markdown"
    extract: int = Field(default=3, ge=0, le=5)
    extract_mode: Literal["auto", "fast", "rendered"] = "auto"
    limit: int = Field(default=25, ge=1, le=100)
    filter: bool = False
    mode: Literal["any", "fast", "balanced"] = "balanced"
    engines: list[str] = Field(default_factory=list)
    agent_loop: bool = True
    max_searches: int = Field(default=3, ge=0)
    map_reduce_context_limit: int = Field(default=16000, ge=1000)
    map_reduce_call_limit: int = Field(default=0, ge=0)
    map_reduce_chunk_size: int = Field(default=4000, ge=500)
    map_reduce_reduce: bool = True


class WebFetchSchema(BaseModel):
    base_url: str = "http://wingman.akhbar.lan:7000"
    format: Literal["json", "markdown", "text", "ndjson"] = "markdown"
    mode: Literal["auto", "fast", "rendered"] = "auto"


class EndpointSchema(BaseModel):
    name: str
    base_url: str = ""
    api_key: str | None = Field(
        default=None,
        description="New API key. Null or empty means keep the existing key.",
    )
    api_key_masked: str = ""
    models: list[str] = Field(default_factory=list)
    aliases: dict[str, str] = Field(default_factory=dict)
    substitute_role: dict[str, str] = Field(default_factory=dict)
    log: bool = True
    enabled: bool = True
    max_models: int = Field(default=0, description="Max loaded models on the endpoint. Less than 1 disables tracking.")
    protocol: Literal["openai", "anthropic"] = Field(
        default="openai", description="Wire format of the upstream endpoint."
    )
    mode: Literal["remote", "local"] = Field(
        default="remote",
        description="remote = forward requests to base_url; local = serve embedding models in-process.",
    )


class ConfigResponse(BaseModel):
    host: str | None
    port: int | None
    logs_dir: str | None
    trace_dir: str | None
    endpoints: list[EndpointSchema]
    web_search: WebSearchSchema = Field(default_factory=WebSearchSchema)
    web_fetch: WebFetchSchema = Field(default_factory=WebFetchSchema)


class ConfigUpdateRequest(BaseModel):
    endpoints: list[EndpointSchema]


class RouteResolveResponse(BaseModel):
    model: str
    endpoint: str | None
    forwarded_model: str | None
    base_url: str | None
    log: bool | None
    wildcard: bool


class TraceSummarySchema(BaseModel):
    id: str
    timestamp: str
    api_key: str
    endpoint: str
    model: str | None
    stream: bool
    status_code: int | None
    duration_ms: float | None
    message_count: int
    error: str | None
    correlation_id: str | None = None
    parent_trace_id: str | None = None


class TraceCallSchema(BaseModel):
    summary: TraceSummarySchema
    request_payload: Any
    response_body: Any
    response_chunks: list[str]
    assembled_content: str


class TraceDetailResponse(BaseModel):
    summary: TraceSummarySchema
    request_headers: dict[str, str]
    request_payload: Any
    response_headers: dict[str, str]
    response_body: Any
    response_chunks: list[str]
    assembled_content: str
    children: list[TraceCallSchema] = []


class StatsResponse(BaseModel):
    total: int
    errors: int
    error_rate: float
    avg_duration_ms: float | None
    by_model: dict[str, int]
    by_api_key: dict[str, int]
    requests_per_hour: list[int]
    errors_per_hour: list[int]


class ValidationIssueSchema(BaseModel):
    code: str
    message: str
    path: str


class LedgerEntrySchema(BaseModel):
    id: str
    trace_id: str
    timestamp: str
    model: str | None
    endpoint: str
    issues: list[ValidationIssueSchema]

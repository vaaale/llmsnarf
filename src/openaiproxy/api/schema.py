from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EndpointSchema(BaseModel):
    name: str
    base_url: str
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


class ConfigResponse(BaseModel):
    host: str | None
    port: int | None
    logs_dir: str | None
    trace_dir: str | None
    endpoints: list[EndpointSchema]


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


class TraceDetailResponse(BaseModel):
    summary: TraceSummarySchema
    request_headers: dict[str, str]
    request_payload: Any
    response_headers: dict[str, str]
    response_body: Any
    response_chunks: list[str]
    assembled_content: str


class StatsResponse(BaseModel):
    total: int
    errors: int
    error_rate: float
    avg_duration_ms: float | None
    by_model: dict[str, int]
    by_api_key: dict[str, int]
    requests_per_hour: list[int]
    errors_per_hour: list[int]

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from openaiproxy.api.schema import (
    ConfigResponse,
    ConfigUpdateRequest,
    EndpointSchema,
    RouteResolveResponse,
    StatsResponse,
    TraceDetailResponse,
    TraceSummarySchema,
    WebSearchSchema,
)
from openaiproxy.models.models import EndpointConfig, LLMProxyConfig, WebSearchConfig
from openaiproxy.models.trace_models import TraceSummary
from openaiproxy.services.config_service import ConfigService, ConfigValidationError
from openaiproxy.services.trace_service import TraceService

router = APIRouter(prefix="/ui/api")


def get_config_service(request: Request) -> ConfigService:
    return request.app.state.config_service


def get_trace_service(request: Request) -> TraceService:
    return request.app.state.trace_service


def _mask_api_key(api_key: str) -> str:
    if not api_key:
        return ""
    return api_key[:4] + "••••"


def _endpoint_to_schema(endpoint: EndpointConfig) -> EndpointSchema:
    return EndpointSchema(
        name=endpoint.name,
        base_url=endpoint.base_url,
        api_key=None,
        api_key_masked=_mask_api_key(endpoint.api_key),
        models=list(endpoint.models),
        aliases=dict(endpoint.aliases),
        substitute_role=dict(endpoint.substitute_role or {}),
        log=endpoint.log,
        enabled=endpoint.enabled,
        max_models=endpoint.max_models,
    )


def _schema_to_endpoint(schema: EndpointSchema) -> EndpointConfig:
    return EndpointConfig(
        name=schema.name.strip(),
        base_url=schema.base_url.strip().rstrip("/"),
        api_key=schema.api_key if schema.api_key else "",
        models=[m.strip() for m in schema.models if m.strip()],
        aliases=dict(schema.aliases),
        log=schema.log,
        substitute_role=dict(schema.substitute_role),
        enabled=schema.enabled,
        max_models=schema.max_models,
    )


def _config_to_response(config: LLMProxyConfig) -> ConfigResponse:
    return ConfigResponse(
        host=config.host,
        port=config.port,
        logs_dir=config.logs_dir,
        trace_dir=config.trace_dir,
        endpoints=[_endpoint_to_schema(endpoint) for endpoint in config.endpoints],
        web_search=WebSearchSchema(
            enabled=config.web_search.enabled,
            base_url=config.web_search.base_url,
            format=config.web_search.format,
            extract=config.web_search.extract,
            extract_mode=config.web_search.extract_mode,
            limit=config.web_search.limit,
            filter=config.web_search.filter,
        ),
    )


def _summary_to_schema(summary: TraceSummary) -> TraceSummarySchema:
    return TraceSummarySchema(
        id=summary.id,
        timestamp=summary.timestamp,
        api_key=summary.api_key,
        endpoint=summary.endpoint,
        model=summary.model,
        stream=summary.stream,
        status_code=summary.status_code,
        duration_ms=summary.duration_ms,
        message_count=summary.message_count,
        error=summary.error,
    )


@router.get("/config", response_model=ConfigResponse)
def get_config(config_service: ConfigService = Depends(get_config_service)) -> ConfigResponse:
    return _config_to_response(config_service.get_config())


@router.put("/config", response_model=ConfigResponse)
def update_config(
    payload: ConfigUpdateRequest,
    config_service: ConfigService = Depends(get_config_service),
) -> ConfigResponse:
    endpoints = [_schema_to_endpoint(schema) for schema in payload.endpoints]
    try:
        updated = config_service.update_endpoints(endpoints)
    except ConfigValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _config_to_response(updated)


@router.put("/config/web_search", response_model=ConfigResponse)
def update_web_search_config(
    payload: WebSearchSchema,
    config_service: ConfigService = Depends(get_config_service),
) -> ConfigResponse:
    web_search = WebSearchConfig(
        enabled=payload.enabled,
        base_url=payload.base_url.strip().rstrip("/"),
        format=payload.format,
        extract=payload.extract,
        extract_mode=payload.extract_mode,
        limit=payload.limit,
        filter=payload.filter,
    )
    if not web_search.base_url:
        raise HTTPException(status_code=422, detail="Web search base_url is required")
    return _config_to_response(config_service.update_web_search(web_search))


@router.get("/config/resolve", response_model=RouteResolveResponse)
def resolve_route(
    model: str = Query(min_length=1),
    config_service: ConfigService = Depends(get_config_service),
) -> RouteResolveResponse:
    endpoint = config_service.resolve_route(model)
    if endpoint is None:
        return RouteResolveResponse(
            model=model,
            endpoint=None,
            forwarded_model=None,
            base_url=None,
            log=None,
            wildcard=False,
        )
    forwarded = endpoint.aliases.get(model, model)
    wildcard = model not in endpoint.models and model not in endpoint.aliases
    return RouteResolveResponse(
        model=model,
        endpoint=endpoint.name,
        forwarded_model=forwarded,
        base_url=endpoint.base_url,
        log=endpoint.log,
        wildcard=wildcard,
    )


@router.get("/traces", response_model=list[TraceSummarySchema])
def list_traces(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    model: str | None = None,
    api_key: str | None = None,
    status: str | None = None,
    since: str | None = None,
    q: str | None = None,
    trace_service: TraceService = Depends(get_trace_service),
) -> list[TraceSummarySchema]:
    summaries = trace_service.list_traces(
        limit=limit,
        offset=offset,
        model=model,
        api_key=api_key,
        status=status,
        since=since,
        query=q,
    )
    return [_summary_to_schema(summary) for summary in summaries]


@router.get("/traces/{trace_id}", response_model=TraceDetailResponse)
def get_trace(
    trace_id: str,
    trace_service: TraceService = Depends(get_trace_service),
) -> TraceDetailResponse:
    result = trace_service.get_trace(trace_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    detail, assembled = result
    return TraceDetailResponse(
        summary=_summary_to_schema(detail.summary),
        request_headers=detail.request_headers,
        request_payload=detail.request_payload,
        response_headers=detail.response_headers,
        response_body=detail.response_body,
        response_chunks=detail.response_chunks,
        assembled_content=assembled,
    )


@router.get("/stats", response_model=StatsResponse)
def get_stats(
    hours: int = Query(default=24, ge=1, le=168),
    trace_service: TraceService = Depends(get_trace_service),
) -> StatsResponse:
    return StatsResponse(**trace_service.stats(hours=hours))

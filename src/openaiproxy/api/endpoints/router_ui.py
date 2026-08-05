from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from openaiproxy.api.schema import (
    ConfigResponse,
    ConfigUpdateRequest,
    EndpointSchema,
    LedgerEntrySchema,
    RouteResolveResponse,
    StatsResponse,
    TraceCallSchema,
    TraceDetailResponse,
    TraceSummarySchema,
    ValidationIssueSchema,
    WebFetchSchema,
    WebSearchSchema,
)
from openaiproxy.models.models import EndpointConfig, LLMProxyConfig, WebFetchConfig, WebSearchConfig
from openaiproxy.models.trace_models import TraceSummary
from openaiproxy.services.config_service import ConfigService, ConfigValidationError
from openaiproxy.services.ledger_service import LedgerService
from openaiproxy.services.trace_service import TraceService, assemble_stream_content

router = APIRouter(prefix="/ui/api")


def get_config_service(request: Request) -> ConfigService:
    return request.app.state.config_service


def get_trace_service(request: Request) -> TraceService:
    return request.app.state.trace_service


def get_ledger_service(request: Request) -> LedgerService:
    return request.app.state.ledger_service


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
        protocol=endpoint.protocol if endpoint.protocol in ("openai", "anthropic") else "openai",
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
        protocol=schema.protocol,
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
            mode=config.web_search.mode,
            engines=list(config.web_search.engines),
            agent_loop=config.web_search.agent_loop,
            max_searches=config.web_search.max_searches,
            map_reduce_context_limit=config.web_search.map_reduce_context_limit,
            map_reduce_call_limit=config.web_search.map_reduce_call_limit,
            map_reduce_chunk_size=config.web_search.map_reduce_chunk_size,
            map_reduce_reduce=config.web_search.map_reduce_reduce,
        ),
        web_fetch=WebFetchSchema(
            base_url=config.web_fetch.base_url,
            format=config.web_fetch.format,
            mode=config.web_fetch.mode,
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
        correlation_id=summary.correlation_id,
        parent_trace_id=summary.parent_trace_id,
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
        mode=payload.mode,
        engines=list(payload.engines),
        agent_loop=payload.agent_loop,
        max_searches=payload.max_searches,
        map_reduce_context_limit=payload.map_reduce_context_limit,
        map_reduce_call_limit=payload.map_reduce_call_limit,
        map_reduce_chunk_size=payload.map_reduce_chunk_size,
        map_reduce_reduce=payload.map_reduce_reduce,
    )
    if not web_search.base_url:
        raise HTTPException(status_code=422, detail="Web search base_url is required")
    return _config_to_response(config_service.update_web_search(web_search))


@router.put("/config/web_fetch", response_model=ConfigResponse)
def update_web_fetch_config(
    payload: WebFetchSchema,
    config_service: ConfigService = Depends(get_config_service),
) -> ConfigResponse:
    web_fetch = WebFetchConfig(
        base_url=payload.base_url.strip().rstrip("/"),
        format=payload.format,
        mode=payload.mode,
    )
    if not web_fetch.base_url:
        raise HTTPException(status_code=422, detail="Web fetch base_url is required")
    return _config_to_response(config_service.update_web_fetch(web_fetch))


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
    correlation_id: str | None = None,
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
        correlation_id=correlation_id,
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
    children = [
        TraceCallSchema(
            summary=_summary_to_schema(child.summary),
            request_payload=child.request_payload,
            response_body=child.response_body,
            response_chunks=child.response_chunks,
            assembled_content=assemble_stream_content(child.response_chunks) if child.response_chunks else "",
        )
        for child in detail.children
    ]
    return TraceDetailResponse(
        summary=_summary_to_schema(detail.summary),
        request_headers=detail.request_headers,
        request_payload=detail.request_payload,
        response_headers=detail.response_headers,
        response_body=detail.response_body,
        response_chunks=detail.response_chunks,
        assembled_content=assembled,
        children=children,
    )


@router.get("/stats", response_model=StatsResponse)
def get_stats(
    hours: int = Query(default=24, ge=1, le=168),
    trace_service: TraceService = Depends(get_trace_service),
) -> StatsResponse:
    return StatsResponse(**trace_service.stats(hours=hours))


def _entry_to_schema(entry) -> LedgerEntrySchema:
    return LedgerEntrySchema(
        id=entry.id,
        trace_id=entry.trace_id,
        timestamp=entry.timestamp,
        model=entry.model,
        endpoint=entry.endpoint,
        issues=[ValidationIssueSchema(code=i.code, message=i.message, path=i.path) for i in entry.issues],
    )


@router.get("/ledger", response_model=list[LedgerEntrySchema])
def list_ledger(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    model: str | None = None,
    since: str | None = None,
    ledger_service: LedgerService = Depends(get_ledger_service),
) -> list[LedgerEntrySchema]:
    entries = ledger_service.list_entries(limit=limit, offset=offset, model=model, since=since)
    return [_entry_to_schema(e) for e in entries]


@router.get("/ledger/{entry_id}", response_model=LedgerEntrySchema)
def get_ledger_entry(
    entry_id: str,
    ledger_service: LedgerService = Depends(get_ledger_service),
) -> LedgerEntrySchema:
    entry = ledger_service.get_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Ledger entry not found")
    return _entry_to_schema(entry)

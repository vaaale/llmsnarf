from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from openaiproxy.services.anthropic_service import AnthropicService
from openaiproxy.services.embeddings_service import EmbeddingsService
from openaiproxy.services.proxy_service import ProxyService
from openaiproxy.services.responses_service import ResponsesService


router = APIRouter()

logger = logging.getLogger("openaiproxy")


def get_proxy_service(request: Request) -> ProxyService:
    return request.app.state.proxy_service


def get_responses_service(request: Request) -> ResponsesService:
    return request.app.state.responses_service


def get_anthropic_service(request: Request) -> AnthropicService:
    return request.app.state.anthropic_service


def get_embeddings_service(request: Request) -> EmbeddingsService:
    return request.app.state.embeddings_service


@router.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request, proxy_service: ProxyService = Depends(get_proxy_service)):
    return await proxy_request(request, "/chat/completions", proxy_service)


@router.post("/v1/completions")
async def proxy_completions(request: Request, proxy_service: ProxyService = Depends(get_proxy_service)):
    return await proxy_request(request, "/completions", proxy_service)


@router.post("/v1/responses")
async def proxy_responses(request: Request, responses_service: ResponsesService = Depends(get_responses_service)):
    body = await request.body()
    headers = dict(request.headers)

    logger.info(
        "incoming_request method=%s path=/responses client=%s body_bytes=%d",
        request.method,
        request.client.host if request.client else "unknown",
        len(body),
    )

    result = await responses_service.handle(body=body, headers=headers)

    if result["type"] == "stream":
        return StreamingResponse(result["iterator"], media_type="text/event-stream")

    status_code = result["status_code"]
    if status_code >= 400:
        logger.warning("response_status method=%s path=/responses status=%s", request.method, status_code)

    return Response(
        content=result["content"],
        status_code=status_code,
        headers=result.get("headers"),
    )


@router.post("/v1/messages")
async def proxy_anthropic_messages(
    request: Request, anthropic_service: AnthropicService = Depends(get_anthropic_service)
):
    body = await request.body()
    headers = dict(request.headers)

    logger.info(
        "incoming_request method=%s path=/messages client=%s body_bytes=%d",
        request.method,
        request.client.host if request.client else "unknown",
        len(body),
    )

    result = await anthropic_service.handle(body=body, headers=headers)

    if result["type"] == "stream":
        return StreamingResponse(result["iterator"], media_type="text/event-stream")

    status_code = result["status_code"]
    if status_code >= 400:
        logger.warning("response_status method=%s path=/messages status=%s", request.method, status_code)

    return Response(
        content=result["content"],
        status_code=status_code,
        headers=result.get("headers"),
    )


@router.post("/v1/embeddings")
async def proxy_embeddings(
    request: Request, embeddings_service: EmbeddingsService = Depends(get_embeddings_service)
):
    body = await request.body()
    headers = dict(request.headers)

    logger.info(
        "incoming_request method=%s path=/embeddings client=%s body_bytes=%d",
        request.method,
        request.client.host if request.client else "unknown",
        len(body),
    )

    result = await embeddings_service.handle(body=body, headers=headers)

    status_code = result["status_code"]
    if status_code >= 400:
        logger.warning("response_status method=%s path=/embeddings status=%s", request.method, status_code)

    return Response(
        content=result["content"],
        status_code=status_code,
        headers=result.get("headers"),
    )


@router.api_route(
    "/v1/{endpoint_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy_passthrough(request: Request, endpoint_path: str, proxy_service: ProxyService = Depends(get_proxy_service)):
    return await proxy_request(request, f"/{endpoint_path}", proxy_service)


async def proxy_request(request: Request, endpoint_path: str, proxy_service: ProxyService):
    body = await request.body()
    headers = dict(request.headers)

    logger.info(
        "incoming_request method=%s path=%s client=%s body_bytes=%d",
        request.method,
        endpoint_path,
        request.client.host if request.client else "unknown",
        len(body),
    )

    result = await proxy_service.proxy(
        method=request.method,
        endpoint_path=endpoint_path,
        body=body,
        headers=headers,
        query_params=dict(request.query_params),
    )

    if result["type"] == "stream":
        return StreamingResponse(result["iterator"], media_type="text/event-stream")

    status_code = result["status_code"]
    if status_code >= 400:
        logger.warning("response_status method=%s path=%s status=%s", request.method, endpoint_path, status_code)
    else:
        logger.debug("response_status method=%s path=%s status=%s", request.method, endpoint_path, status_code)

    return Response(
        content=result["content"],
        status_code=status_code,
        headers=result.get("headers"),
    )


@router.get("/")
async def root():
    return {
        "status": "running",
        "service": "OpenAI Proxy",
        "endpoints": [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/responses",
            "/v1/messages",
            "/v1/embeddings",
            "/v1/{endpoint_path:path}",
        ],
    }

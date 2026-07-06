from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from openaiproxy.services.proxy_service import ProxyService
from openaiproxy.services.responses_service import ResponsesService


router = APIRouter()


def get_proxy_service(request: Request) -> ProxyService:
    return request.app.state.proxy_service


def get_responses_service(request: Request) -> ResponsesService:
    return request.app.state.responses_service


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

    result = await responses_service.handle(body=body, headers=headers)

    if result["type"] == "stream":
        return StreamingResponse(result["iterator"], media_type="text/event-stream")

    return Response(
        content=result["content"],
        status_code=result["status_code"],
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

    result = await proxy_service.proxy(
        method=request.method,
        endpoint_path=endpoint_path,
        body=body,
        headers=headers,
        query_params=dict(request.query_params),
    )

    if result["type"] == "stream":
        return StreamingResponse(result["iterator"], media_type="text/event-stream")

    return Response(
        content=result["content"],
        status_code=result["status_code"],
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
            "/v1/{endpoint_path:path}",
        ],
    }

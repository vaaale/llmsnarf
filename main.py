import json
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    openai_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for OpenAI API"
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key"
    )
    logs_dir: Path = Field(
        default=Path("./logs"),
        description="Directory to save request logs"
    )
    listen_host: str = Field(
        default="0.0.0.0",
        description="Host to listen on"
    )
    listen_port: int = Field(
        default=8000,
        description="Port to listen on"
    )


# Initialize settings
settings = Settings()

# Create logs directory if it doesn't exist
settings.logs_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="OpenAI Proxy")


def extract_api_key(headers: dict[str, str]) -> str:
    """Extract API key from authorization header."""
    auth_header = headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]  # Remove 'Bearer ' prefix
    return "unknown"


def sanitize_api_key_for_filename(api_key: str) -> str:
    """Sanitize API key to make it safe for filenames."""
    # Replace any characters that might be problematic in filenames
    return api_key.replace("/", "_").replace("\\", "_").replace(":", "_")


def save_request(endpoint: str, payload: dict[str, Any], headers: dict[str, str], api_key: str) -> str:
    """Save the request payload to a file and return the base filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_api_key = sanitize_api_key_for_filename(api_key)
    base_filename = f"{safe_api_key}_{timestamp}"
    filename = settings.logs_dir / f"{base_filename}_request.json"
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint,
        "headers": dict(headers),
        "payload": payload
    }
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved request to {filename}")
    return base_filename


def save_response(base_filename: str, response_data: dict[str, Any]) -> None:
    """Save the response data to a file."""
    filename = settings.logs_dir / f"{base_filename}_response.json"
    
    with open(filename, "w") as f:
        json.dump(response_data, f, indent=2)
    
    print(f"Saved response to {filename}")


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    """Proxy endpoint for OpenAI chat completions."""
    return await proxy_request(request, "/chat/completions")


@app.post("/v1/completions")
async def proxy_completions(request: Request):
    """Proxy endpoint for OpenAI completions."""
    return await proxy_request(request, "/completions")


async def proxy_request(request: Request, endpoint_path: str):
    """Generic proxy function to handle requests."""
    # Get the request body
    body = await request.body()
    payload = json.loads(body) if body else {}
    
    # Get headers and prepare them for forwarding
    headers = dict(request.headers)
    
    # Extract API key and save the request
    api_key = extract_api_key(headers)
    base_filename = save_request(endpoint_path, payload, headers, api_key)
    
    # Prepare headers for OpenAI request
    forward_headers = {
        "Content-Type": "application/json",
    }
    
    # Use API key from environment if not provided in request
    if "authorization" in headers:
        forward_headers["Authorization"] = headers["authorization"]
    elif settings.openai_api_key:
        forward_headers["Authorization"] = f"Bearer {settings.openai_api_key}"
    
    # Construct the target URL
    target_url = f"{settings.openai_api_base}{endpoint_path}"
    
    # Check if streaming is requested
    is_streaming = payload.get("stream", False)
    
    if is_streaming:
        # Handle streaming response - client must stay open during streaming
        async def stream_response():
            response_chunks = []
            async with httpx.AsyncClient(timeout=300.0) as client:
                try:
                    async with client.stream(
                        "POST",
                        target_url,
                        headers=forward_headers,
                        json=payload
                    ) as response:
                        response_data = {
                            "timestamp": datetime.now().isoformat(),
                            "status_code": response.status_code,
                            "headers": dict(response.headers),
                            "chunks": []
                        }
                        
                        async for chunk in response.aiter_bytes():
                            # Save chunk for logging
                            try:
                                response_data["chunks"].append(chunk.decode('utf-8'))
                            except:
                                response_data["chunks"].append(str(chunk))
                            yield chunk
                        
                        # Save complete response after streaming
                        save_response(base_filename, response_data)
                except Exception as e:
                    error_msg = f"data: {json.dumps({'error': str(e)})}\n\n"
                    response_data = {
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e)
                    }
                    save_response(base_filename, response_data)
                    yield error_msg.encode()
        
        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream"
        )
    else:
        # Handle regular response
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                response = await client.post(
                    target_url,
                    headers=forward_headers,
                    json=payload
                )
                
                # Save the response
                response_data = {
                    "timestamp": datetime.now().isoformat(),
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
                }
                save_response(base_filename, response_data)
                
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )
            except Exception as e:
                print(e)
                # Save error response
                response_data = {
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
                save_response(base_filename, response_data)
                
                return Response(
                    content=json.dumps({"error": str(e)}),
                    status_code=500,
                    media_type="application/json"
                )


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "OpenAI Proxy",
        "endpoints": [
            "/v1/chat/completions",
            "/v1/completions"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.listen_host,
        port=settings.listen_port,
        reload=False
    )

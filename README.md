# OpenAI Proxy

A FastAPI-based proxy server for OpenAI-compatible completion endpoints. This proxy intercepts requests, saves the payloads to files for logging/debugging, and forwards them to the actual OpenAI API.

## Features

- **Request Logging**: Automatically saves all request payloads with timestamps to JSON files
- **OpenAI Compatible**: Works with standard OpenAI API endpoints
- **Streaming Support**: Handles both regular and streaming responses
- **Easy Configuration**: Configure via environment variables
- **FastAPI**: Built with modern async Python framework

## Installation

1. Install dependencies:
```bash
pip install -e .
```

Or using uv:
```bash
uv pip install -e .
```

2. Create a `.env` file from the example:
```bash
cp .env.example .env
```

3. Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

## Usage

### Starting the Server

Run the proxy server:
```bash
python src/main.py
```

Or with uvicorn directly:
```bash
uvicorn openaiproxy.asgi:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on `http://localhost:8000`

### Making Requests

Instead of sending requests to `https://api.openai.com/v1`, send them to your proxy:

**Example with curl:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Example with Python OpenAI client:**
```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Supported Endpoints

- `POST /v1/chat/completions` - Chat completions endpoint
- `POST /v1/completions` - Text completions endpoint
- `GET /` - Health check endpoint

## Configuration

Configure the proxy using environment variables in your `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | (required) |
| `OPENAI_API_BASE` | Base URL for OpenAI API | `https://api.openai.com/v1` |
| `LOGS_DIR` | Directory to save request logs | `./logs` |
| `HOST` | Host address to bind the server | `0.0.0.0` |
| `PORT` | Port to bind the server | `8000` |

## Request and Response Logs

All requests and responses are saved to the `logs/` directory with the following format:
- **Request filename**: `{api_key}_YYYYMMDD_HHMMSS_microseconds_request.json`
- **Response filename**: `{api_key}_YYYYMMDD_HHMMSS_microseconds_response.json`

Each request-response pair shares the same base filename (api_key + timestamp) for easy matching.

### Request Log Structure
```json
{
  "timestamp": "2024-11-19T07:21:00.123456",
  "endpoint": "/chat/completions",
  "headers": {
    "content-type": "application/json",
    "authorization": "Bearer sk-..."
  },
  "payload": {
    "model": "gpt-4",
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }
}
```

### Response Log Structure

**Non-streaming response:**
```json
{
  "timestamp": "2024-11-19T07:21:01.234567",
  "status_code": 200,
  "headers": {
    "content-type": "application/json"
  },
  "body": {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "choices": [...]
  }
}
```

**Streaming response:**
```json
{
  "timestamp": "2024-11-19T07:21:01.234567",
  "status_code": 200,
  "headers": {
    "content-type": "text/event-stream"
  },
  "chunks": [
    "data: {...}",
    "data: {...}",
    "data: [DONE]"
  ]
}
```

## Development

The proxy is built with:
- **FastAPI**: Modern web framework
- **httpx**: Async HTTP client for forwarding requests
- **uvicorn**: ASGI server
- **pydantic-settings**: Type-safe environment variable management with validation

## License

MIT

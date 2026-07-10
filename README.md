# LLM Snarf

LLM Snarf is a lightweight, self-hosted proxy for OpenAI-compatible LLM APIs. It sits between your client application and any OpenAI-compatible backend (local or remote), and gives you full visibility into every request and response.

**Intended use:** Personal or team use alongside tools like [Open WebUI](https://github.com/open-webui/open-webui), [Continue.dev](https://www.continue.dev/), or any OpenAI SDK. It is not a production API gateway — it's an observability and experimentation tool.

**What it does:**
- Routes requests to one or more configured backend endpoints, with model-based routing and aliasing
- Traces every request/response pair to disk as JSON (toggle per endpoint)
- Validates responses after the connection closes and records failures in a **Validation Ledger** for later inspection
- Optionally injects web search and web fetch capabilities via [OpenSerp](https://github.com/karust/openserp)
- Provides a built-in web UI for live tail, trace inspection, configuration, and the validation ledger

---

## Quickstart (local)

**Requirements:** Python 3.10–3.12, [uv](https://github.com/astral-sh/uv)

```bash
# 1. Clone and install
git clone https://github.com/vaaale/llmsnarf
cd llmsnarf
uv sync

# 2. Create your config
cp .env.example .env          # optional — supply HOST/PORT overrides
cd config
cp llmproxy.yaml.example llmsnarf.yaml  # then edit to point at your backend
# (Or configure endpoints via the UI after starting)
```

Edit `config/llmsnarf.yaml` to point at your LLM backend (see [Configuration](#configuration)), or use the UI at `http://localhost:8080/ui` after starting.

```bash
# 3. Run
uv run llmproxy
```

The proxy starts on `http://0.0.0.0:8080` by default. The web UI is at **`http://localhost:8080/ui`**.

Point your client at `http://localhost:8080/v1` or `http://localhost:8080/v1/responses` instead of the upstream API and everything else stays the same.

---

## Running with Docker

A `Dockerfile` is included. Build and run:

```bash
docker build -t llmsnarf .

docker run -d \
  --name llmsnarf \
  -p 8080:8080 \
  -e LLMSNARF_CONFIG=/app/config \
  -v $(pwd)/config:/app/config \
  alexakhbar/llmsnarf
```

> Mount `./config` so you can edit `config/llmsnarf.yaml` without rebuilding. Traces and logs are written inside the container by default; add `-v $(pwd)/traces:/app/traces` and `-v $(pwd)/logs:/app/logs` if you want them on the host.

### OpenSerp (required for web search)

Web search and web fetch are powered by [OpenSerp](https://github.com/karust/openserp). If you don't need web search, set `web_search.enabled: false` in your config and skip this step.

Run OpenSerp alongside LLM Snarf using Docker Compose:

A `docker-compose.yml` is included. Just run:

```bash
docker compose up -d
```

The compose file mounts `./config` into the container and sets `LLMSNARF_CONFIG=/app/config`, so `config/llmsnarf.yaml` is picked up automatically. Port is read from `${PORT}` in your `.env` (default `8080`).

Then set `web_search.base_url: http://openserp:7001` in `config/llmsnarf.yaml` to reach the OpenSerp sidecar.

---

## Configuration

All configuration lives in `config/llmsnarf.yaml`. Changes take effect immediately — no restart required (the config is reloaded on each request).

```yaml
llmproxy:
  host: 0.0.0.0
  port: 8080
  logs_dir: ./logs        # application logs
  trace_dir: ./traces     # request/response traces and validation ledger

  endpoints:
    my-backend:
      base_url: http://my-llm-server:8000/v1
      api_key: sk-...       # forwarded as Authorization: Bearer; optional
      models:
        - qwen3.6-27b
        - ornith-1.0-35b
      aliases:              # rewrite model names before forwarding
        qwen3.6-27b: gpt-5.5
        ornith-1.0-35b: gpt-codex
      substitute_role:      # rewrite message roles
        system: user
      log: true             # enable trace logging for this endpoint
      enabled: true
      max_models: 0         # 0 = unlimited; >0 caps concurrent model slots

    openai:
      base_url: https://api.openai.com/v1
      api_key: sk-...
      models:
        - gpt-5.6-sol
        - gpt-5.6-luna
      log: true
      enabled: true

    default:
      base_url: http://my-llm-server:8080/v1
      models:
        - '*'               # wildcard: catches any unmatched model
      log: false
      enabled: true
```

### Endpoint routing

Requests are routed by the `model` field in the request body:

1. Exact match against `models` list → use that endpoint
2. Alias match against `aliases` map → rewrite model name and forward
3. Wildcard endpoint (`*`) → catch-all fallback
4. No match → `404 routing_error`

If `api_key` is set on the endpoint, it overrides the client's `Authorization` header. If omitted, the client's key is forwarded as-is.

### Web search

`web_search` and `web_fetch` are two separate tools backed by [OpenSerp](https://github.com/karust/openserp), each with its own base URL (they can point at different OpenSerp instances/endpoints if needed).

```yaml
web_search:                          # -> GET {base_url}/mega/search
  enabled: true
  base_url: http://openserp:7000     # OpenSerp instance
  format: markdown                   # json | markdown | text | ndjson
  extract: 3                         # 0-5 top results to enrich with inline page content
  extract_mode: auto                 # auto | fast | rendered
  limit: 25                          # 1-100 max search results
  filter: false                      # deduplicate results
  mode: balanced                     # any | fast | balanced
  engines: []                        # bing, google, yandex, baidu, duckduckgo, ecosia (empty = all)

web_fetch:                           # -> GET {base_url}/extract
  base_url: http://openserp:7000     # OpenSerp instance
  format: markdown                   # json | markdown | text | ndjson
  mode: auto                         # auto | fast | rendered
```

When enabled, the proxy injects `web_search` and `web_fetch` tool definitions into every request. The model can invoke them and the proxy will execute the search/fetch transparently before returning the final response to the client.

---

## Usage

### Pointing your client at the proxy

Replace the base URL in any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-key",           # passed through to the backend
    base_url="http://localhost:8080/v1"
)

response = client.chat.completions.create(
    model="gpt-codex",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-key" \
  -d '{"model": "gpt-codex", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### Supported endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | Chat completions (streaming + non-streaming) |
| `POST /v1/completions` | Text completions |
| `POST /v1/responses` | Responses API (OpenAI responses format) |
| `GET/POST /v1/{path}` | Passthrough for any other `/v1/` path |
| `GET /ui` | Web UI |
| `GET /` | Health check |

---

## Web UI

Visit **`http://localhost:8080/ui`** after starting the proxy.

| Page | Description |
|---|---|
| **Dashboard** | Traffic KPIs, requests-per-hour chart, model and API key distribution, recent requests |
| **Traces** | Full request/response inspector — conversation view, raw JSON, SSE chunks, headers |
| **Live Tail** | Real-time stream of incoming requests |
| **Validation Ledger** | Log of requests that failed post-response validation, grouped by issue type, with a direct link to the corresponding trace |
| **Configuration** | Add/edit/delete endpoints, toggle tracing per endpoint, configure web search/fetch, test model routing |

---

## Traces

When `log: true` is set on an endpoint, every request/response pair is written to `trace_dir` as a pair of JSON files:

```
traces/
  sk1234_20250110_142301_000123_request.json
  sk1234_20250110_142301_000123_response.json
```

The base filename encodes the API key prefix, date, time, and microseconds. Request and response files share the same base name.

**Request file:**
```json
{
  "timestamp": "2025-01-10T14:23:01.000123",
  "endpoint": "/chat/completions",
  "headers": { "content-type": "application/json" },
  "payload": {
    "model": "llama3",
    "messages": [{ "role": "user", "content": "Hello!" }]
  }
}
```

**Response file (non-streaming):**
```json
{
  "timestamp": "2025-01-10T14:23:01.987654",
  "status_code": 200,
  "headers": { "content-type": "application/json" },
  "body": { "id": "chatcmpl-...", "choices": [...] }
}
```

**Response file (streaming):**
```json
{
  "timestamp": "2025-01-10T14:23:01.987654",
  "status_code": 200,
  "headers": { "content-type": "text/event-stream" },
  "chunks": ["data: {...}", "data: {...}", "data: [DONE]"]
}
```

---

## Validation Ledger

After every traced response, the proxy automatically validates the request and response. If issues are found, an entry is appended to `traces/validation_ledger.ndjson` and surfaced in the **Validation Ledger** UI page.

Detected issue types:

| Code | Description |
|---|---|
| `response.unclosed_tag` | Unclosed XML/model tag (e.g. `<thinking>`, `<tool_call>`) in assistant content |
| `response.tool_call.invalid_json` | Tool call arguments are not valid JSON |
| `response.tool_call.args_not_object` | Tool call arguments parsed but are not a JSON object |
| `response.tool_call.null_arg` | A tool call argument value is `null` |
| `response.tool_call.autolink_arg` | A tool call argument looks like a Markdown autolink |
| `response.tool_call.stringified_array` | A tool call argument is a JSON-stringified array |
| `response.truncated` | Response was cut off due to `max_tokens` limit |
| `request.missing_model` | Request has no `model` field |
| `request.messages_not_array` | `messages` field is not an array |

Each ledger entry links back to the trace ID so you can inspect the full request/response from the UI.

---


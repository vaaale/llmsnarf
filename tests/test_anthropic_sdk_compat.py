"""Drive the proxy through the official anthropic and openai SDKs.

The SDKs strictly parse responses and SSE streams into typed models, so these
tests verify wire-format correctness of the translation layer without a live
backend: the SDK talks to the FastAPI app over an ASGI transport while the
app's own upstream calls are served by a mock transport.
"""

import asyncio
import json
from pathlib import Path

import anthropic
import httpx
import openai
import pytest

from openaiproxy.api.app import create_app


CONFIG = """
llmproxy:
  endpoints:
    anthropic-backend:
      base_url: http://anthropic.test/v1
      protocol: anthropic
      api_key: sk-ant-endpoint
      models: ['claude-x']
      log: false
    openai-backend:
      base_url: http://openai.test/v1
      api_key: sk-oai-endpoint
      models: ['gpt-x']
      log: false
"""

ANTHROPIC_MESSAGE_RESPONSE = {
    "id": "msg_upstream",
    "type": "message",
    "role": "assistant",
    "model": "claude-x",
    "content": [
        {"type": "text", "text": "Hello from Claude"},
        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Oslo"}},
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 11, "output_tokens": 7},
}

CHAT_COMPLETION_RESPONSE = {
    "id": "chatcmpl-upstream",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-x",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello from GPT",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}

ANTHROPIC_STREAM = (
    'event: message_start\ndata: {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x", "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 4, "output_tokens": 0}}}\n\n'
    'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}\n\n'
    'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello "}}\n\n'
    'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "world"}}\n\n'
    'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 0}\n\n'
    'event: content_block_start\ndata: {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_s", "name": "get_weather", "input": {}}}\n\n'
    'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\\"city\\": \\"Oslo\\"}"}}\n\n'
    'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 1}\n\n'
    'event: message_delta\ndata: {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": null}, "usage": {"output_tokens": 6}}\n\n'
    'event: message_stop\ndata: {"type": "message_stop"}\n\n'
)

CHAT_STREAM = (
    'data: {"id": "c1", "object": "chat.completion.chunk", "created": 1700000000, "model": "gpt-x", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]}\n\n'
    'data: {"id": "c1", "object": "chat.completion.chunk", "created": 1700000000, "model": "gpt-x", "choices": [{"index": 0, "delta": {"content": "Hello "}, "finish_reason": null}]}\n\n'
    'data: {"id": "c1", "object": "chat.completion.chunk", "created": 1700000000, "model": "gpt-x", "choices": [{"index": 0, "delta": {"content": "world"}, "finish_reason": null}]}\n\n'
    'data: {"id": "c1", "object": "chat.completion.chunk", "created": 1700000000, "model": "gpt-x", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_s", "type": "function", "function": {"name": "get_weather", "arguments": ""}}]}, "finish_reason": null}]}\n\n'
    'data: {"id": "c1", "object": "chat.completion.chunk", "created": 1700000000, "model": "gpt-x", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"city\\": \\"Oslo\\"}"}}]}, "finish_reason": null}]}\n\n'
    'data: {"id": "c1", "object": "chat.completion.chunk", "created": 1700000000, "model": "gpt-x", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}\n\n'
    'data: {"id": "c1", "object": "chat.completion.chunk", "created": 1700000000, "model": "gpt-x", "choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}}\n\n'
    "data: [DONE]\n\n"
)


@pytest.fixture
def app(tmp_path: Path, monkeypatch):
    from openaiproxy.yaml_repository.yaml_repository import YAMLLLMProxyConfigRepository

    config_path = tmp_path / "llmproxy.yaml"
    config_path.write_text(CONFIG)

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content) if request.content else {}
        streaming = payload.get("stream")
        if request.url.host == "anthropic.test":
            if streaming:
                return httpx.Response(
                    200, content=ANTHROPIC_STREAM, headers={"Content-Type": "text/event-stream"}
                )
            return httpx.Response(200, json=ANTHROPIC_MESSAGE_RESPONSE)
        if request.url.host == "openai.test":
            if streaming:
                return httpx.Response(
                    200, content=CHAT_STREAM, headers={"Content-Type": "text/event-stream"}
                )
            return httpx.Response(200, json=CHAT_COMPLETION_RESPONSE)
        return httpx.Response(500, json={"error": "unexpected host"})

    mock_transport = httpx.MockTransport(upstream_handler)

    class PatchedAsyncClient(httpx.AsyncClient):
        # SDK clients pass an explicit ASGI transport; the proxy's internal
        # upstream clients do not and get the mock upstream.
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("transport", mock_transport)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", PatchedAsyncClient)

    repository = YAMLLLMProxyConfigRepository(config_path)
    return create_app(
        config_repository=repository,
        logs_dir=tmp_path / "logs",
        trace_dir=tmp_path / "traces",
    )


def _anthropic_client(app) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        base_url="http://proxy.test",
        api_key="sk-test",
        http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app)),
    )


def _openai_client(app) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url="http://proxy.test/v1",
        api_key="sk-test",
        http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app)),
    )


def test_anthropic_sdk_nonstreaming_against_openai_backend(app):
    async def main():
        client = _anthropic_client(app)
        message = await client.messages.create(
            model="gpt-x",
            max_tokens=64,
            system="Be nice",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert message.role == "assistant"
        assert message.content[0].type == "text"
        assert message.content[0].text == "Hello from GPT"
        assert message.content[1].type == "tool_use"
        assert message.content[1].name == "get_weather"
        assert message.content[1].input == {"city": "Oslo"}
        assert message.stop_reason == "tool_use"
        assert message.usage.input_tokens == 5
        assert message.usage.output_tokens == 3

    asyncio.run(main())


def test_anthropic_sdk_streaming_against_openai_backend(app):
    async def main():
        client = _anthropic_client(app)
        async with client.messages.stream(
            model="gpt-x",
            max_tokens=64,
            messages=[{"role": "user", "content": "Hello"}],
        ) as stream:
            text = ""
            async for chunk in stream.text_stream:
                text += chunk
            final = await stream.get_final_message()
        assert text == "Hello world"
        assert final.content[0].type == "text"
        assert final.content[0].text == "Hello world"
        assert final.content[1].type == "tool_use"
        assert final.content[1].name == "get_weather"
        assert final.content[1].input == {"city": "Oslo"}
        assert final.stop_reason == "tool_use"
        assert final.usage.output_tokens == 6

    asyncio.run(main())


def test_anthropic_sdk_against_anthropic_backend_passthrough(app):
    async def main():
        client = _anthropic_client(app)
        message = await client.messages.create(
            model="claude-x",
            max_tokens=64,
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert message.id == "msg_upstream"
        assert message.content[0].text == "Hello from Claude"
        assert message.content[1].input == {"city": "Oslo"}
        assert message.usage.input_tokens == 11

    asyncio.run(main())


def test_anthropic_sdk_streaming_against_anthropic_backend_passthrough(app):
    async def main():
        client = _anthropic_client(app)
        async with client.messages.stream(
            model="claude-x",
            max_tokens=64,
            messages=[{"role": "user", "content": "Hello"}],
        ) as stream:
            text = ""
            async for chunk in stream.text_stream:
                text += chunk
            final = await stream.get_final_message()
        assert text == "Hello world"
        assert final.stop_reason == "tool_use"

    asyncio.run(main())


def test_openai_sdk_nonstreaming_against_anthropic_backend(app):
    async def main():
        client = _openai_client(app)
        completion = await client.chat.completions.create(
            model="claude-x",
            max_tokens=64,
            messages=[
                {"role": "system", "content": "Be nice"},
                {"role": "user", "content": "Hello"},
            ],
        )
        choice = completion.choices[0]
        assert choice.message.content == "Hello from Claude"
        assert choice.message.tool_calls[0].function.name == "get_weather"
        assert json.loads(choice.message.tool_calls[0].function.arguments) == {"city": "Oslo"}
        assert choice.finish_reason == "tool_calls"
        assert completion.usage.prompt_tokens == 11
        assert completion.usage.completion_tokens == 7

    asyncio.run(main())


def test_openai_sdk_streaming_against_anthropic_backend(app):
    async def main():
        client = _openai_client(app)
        stream = await client.chat.completions.create(
            model="claude-x",
            max_tokens=64,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        text = ""
        tool_name = ""
        tool_args = ""
        finish_reason = None
        usage = None
        async for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                text += delta.content
            for tc in delta.tool_calls or []:
                if tc.function and tc.function.name:
                    tool_name += tc.function.name
                if tc.function and tc.function.arguments:
                    tool_args += tc.function.arguments
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        assert text == "Hello world"
        assert tool_name == "get_weather"
        assert json.loads(tool_args) == {"city": "Oslo"}
        assert finish_reason == "tool_calls"
        assert usage is not None
        assert usage.prompt_tokens == 4
        assert usage.completion_tokens == 6

    asyncio.run(main())


def test_openai_sdk_against_openai_backend_still_works(app):
    async def main():
        client = _openai_client(app)
        completion = await client.chat.completions.create(
            model="gpt-x",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert completion.choices[0].message.content == "Hello from GPT"

    asyncio.run(main())


def test_anthropic_sdk_error_from_proxy_is_parseable(app):
    async def main():
        client = _anthropic_client(app)
        try:
            await client.messages.create(
                model="unknown-model",
                max_tokens=64,
                messages=[{"role": "user", "content": "Hello"}],
            )
        except anthropic.NotFoundError as exc:
            assert "No route configured" in str(exc)
        else:
            raise AssertionError("expected NotFoundError")

    asyncio.run(main())

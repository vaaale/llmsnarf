import json

from openaiproxy.services.anthropic_translation import (
    AnthropicToChatStream,
    ChatToAnthropicStream,
    anthropic_error_body,
    anthropic_request_to_chat,
    anthropic_response_to_chat,
    build_anthropic_headers,
    build_openai_headers,
    chat_request_to_anthropic,
    chat_response_to_anthropic,
    extract_error_message,
    finish_reason_to_stop_reason,
    openai_error_body,
    stop_reason_to_finish_reason,
)


def parse_sse_events(raw: bytes) -> list[dict]:
    events = []
    for block in raw.decode("utf-8").split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                data = line[5:].strip()
                if data == "[DONE]":
                    events.append({"type": "[DONE]"})
                else:
                    events.append(json.loads(data))
    return events


# ---------------------------------------------------------------------------
# Stop reason mapping
# ---------------------------------------------------------------------------


def test_stop_reason_mapping_round_trip():
    assert stop_reason_to_finish_reason("end_turn") == "stop"
    assert stop_reason_to_finish_reason("max_tokens") == "length"
    assert stop_reason_to_finish_reason("tool_use") == "tool_calls"
    assert stop_reason_to_finish_reason("stop_sequence") == "stop"
    assert stop_reason_to_finish_reason("refusal") == "content_filter"
    assert stop_reason_to_finish_reason(None) is None
    assert stop_reason_to_finish_reason("unknown_future_reason") == "stop"

    assert finish_reason_to_stop_reason("stop") == "end_turn"
    assert finish_reason_to_stop_reason("length") == "max_tokens"
    assert finish_reason_to_stop_reason("tool_calls") == "tool_use"
    assert finish_reason_to_stop_reason("content_filter") == "refusal"
    assert finish_reason_to_stop_reason(None) is None


# ---------------------------------------------------------------------------
# Anthropic request -> chat request
# ---------------------------------------------------------------------------


def test_anthropic_request_basic():
    chat = anthropic_request_to_chat(
        {
            "model": "claude-opus-4-7",
            "max_tokens": 1024,
            "system": "Be helpful",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.5,
            "top_p": 0.9,
            "stop_sequences": ["END"],
            "metadata": {"user_id": "u-1"},
            "stream": True,
        }
    )
    assert chat["model"] == "claude-opus-4-7"
    assert chat["messages"] == [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Hello"},
    ]
    assert chat["max_tokens"] == 1024
    assert chat["temperature"] == 0.5
    assert chat["top_p"] == 0.9
    assert chat["stop"] == ["END"]
    assert chat["user"] == "u-1"
    assert chat["stream"] is True


def test_anthropic_request_system_blocks():
    chat = anthropic_request_to_chat(
        {
            "model": "m",
            "max_tokens": 10,
            "system": [
                {"type": "text", "text": "Part one"},
                {"type": "text", "text": "Part two", "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert chat["messages"][0] == {"role": "system", "content": "Part one\n\nPart two"}


def test_anthropic_request_content_blocks_and_images():
    chat = anthropic_request_to_chat(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "abc123",
                            },
                        },
                        {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
                    ],
                }
            ],
        }
    )
    parts = chat["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "What is this?"}
    assert parts[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,abc123"},
    }
    assert parts[2] == {"type": "image_url", "image_url": {"url": "https://x/y.png"}}


def test_anthropic_request_text_only_blocks_collapse_to_string():
    chat = anthropic_request_to_chat(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
            ],
        }
    )
    assert chat["messages"][0] == {"role": "user", "content": "a\nb"}


def test_anthropic_request_tool_use_and_result():
    chat = anthropic_request_to_chat(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "weather in Oslo?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {"city": "Oslo"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "12C, rain"}
                    ],
                },
            ],
        }
    )
    assistant = chat["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "Let me check."
    assert assistant["tool_calls"][0]["id"] == "toolu_1"
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"city": "Oslo"}

    tool_msg = chat["messages"][2]
    assert tool_msg == {"role": "tool", "tool_call_id": "toolu_1", "content": "12C, rain"}


def test_anthropic_request_tool_result_block_content():
    chat = anthropic_request_to_chat(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}],
                        },
                        {"type": "text", "text": "and then?"},
                    ],
                },
            ],
        }
    )
    assert chat["messages"][0] == {"role": "tool", "tool_call_id": "toolu_1", "content": "line1\nline2"}
    assert chat["messages"][1] == {"role": "user", "content": "and then?"}


def test_anthropic_request_tools_and_tool_choice():
    chat = anthropic_request_to_chat(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            ],
            "tool_choice": {"type": "tool", "name": "get_weather", "disable_parallel_tool_use": True},
        }
    )
    assert chat["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    assert chat["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}
    assert chat["parallel_tool_calls"] is False


def test_anthropic_request_tool_choice_variants():
    base = {
        "model": "m",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "f", "input_schema": {"type": "object"}}],
    }
    assert anthropic_request_to_chat({**base, "tool_choice": {"type": "auto"}})["tool_choice"] == "auto"
    assert anthropic_request_to_chat({**base, "tool_choice": {"type": "any"}})["tool_choice"] == "required"
    assert anthropic_request_to_chat({**base, "tool_choice": {"type": "none"}})["tool_choice"] == "none"


def test_anthropic_request_thinking_blocks_dropped():
    chat = anthropic_request_to_chat(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                        {"type": "text", "text": "answer"},
                    ],
                },
            ],
        }
    )
    assert chat["messages"][1] == {"role": "assistant", "content": "answer"}


# ---------------------------------------------------------------------------
# Chat response -> Anthropic response
# ---------------------------------------------------------------------------


def test_chat_response_text():
    result = chat_response_to_anthropic(
        {
            "id": "chatcmpl-1",
            "model": "gpt-x",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
        requested_model="claude-x",
    )
    assert result["type"] == "message"
    assert result["role"] == "assistant"
    assert result["model"] == "claude-x"
    assert result["content"] == [{"type": "text", "text": "Hello!"}]
    assert result["stop_reason"] == "end_turn"
    assert result["stop_sequence"] is None
    assert result["usage"]["input_tokens"] == 10
    assert result["usage"]["output_tokens"] == 5
    assert result["id"].startswith("msg_")


def test_chat_response_tool_calls():
    result = chat_response_to_anthropic(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
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
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
    )
    assert result["content"] == [
        {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Oslo"}}
    ]
    assert result["stop_reason"] == "tool_use"


def test_chat_response_malformed_tool_arguments_become_empty_input():
    result = chat_response_to_anthropic(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{bad json"}}
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    assert result["content"][0]["input"] == {}


def test_chat_response_reasoning_becomes_thinking():
    result = chat_response_to_anthropic(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hi", "reasoning_content": "deep thought"},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    assert result["content"][0] == {"type": "thinking", "thinking": "deep thought", "signature": ""}
    assert result["content"][1] == {"type": "text", "text": "hi"}


def test_chat_response_tool_calls_with_stop_finish_reason_forces_tool_use():
    # LiteLLM/Ollama sometimes report finish_reason "stop" despite tool calls
    result = chat_response_to_anthropic(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                        ],
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    )
    assert result["stop_reason"] == "tool_use"


def test_chat_response_length_maps_to_max_tokens():
    result = chat_response_to_anthropic(
        {"choices": [{"message": {"role": "assistant", "content": "trunc"}, "finish_reason": "length"}]}
    )
    assert result["stop_reason"] == "max_tokens"


def test_chat_response_cached_tokens():
    result = chat_response_to_anthropic(
        {
            "choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 60},
            },
        }
    )
    assert result["usage"]["input_tokens"] == 40
    assert result["usage"]["cache_read_input_tokens"] == 60


# ---------------------------------------------------------------------------
# Chat request -> Anthropic request
# ---------------------------------------------------------------------------


def test_chat_request_basic():
    result = chat_request_to_anthropic(
        {
            "model": "claude-x",
            "messages": [
                {"role": "system", "content": "Be terse"},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.95,
            "stop": "END",
            "user": "u-9",
            "stream": False,
        }
    )
    assert result["model"] == "claude-x"
    assert result["system"] == "Be terse"
    assert result["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    assert result["max_tokens"] == 256
    assert result["temperature"] == 0.7
    assert result["top_p"] == 0.95
    assert result["stop_sequences"] == ["END"]
    assert result["metadata"] == {"user_id": "u-9"}
    assert result["stream"] is False


def test_chat_request_max_tokens_defaulted():
    result = chat_request_to_anthropic({"model": "m", "messages": [{"role": "user", "content": "x"}]})
    assert result["max_tokens"] == 4096


def test_chat_request_max_completion_tokens():
    result = chat_request_to_anthropic(
        {"model": "m", "max_completion_tokens": 77, "messages": [{"role": "user", "content": "x"}]}
    )
    assert result["max_tokens"] == 77


def test_chat_request_temperature_clamped():
    result = chat_request_to_anthropic(
        {"model": "m", "temperature": 1.8, "messages": [{"role": "user", "content": "x"}]}
    )
    assert result["temperature"] == 1.0


def test_chat_request_multiple_system_messages_joined():
    result = chat_request_to_anthropic(
        {
            "model": "m",
            "messages": [
                {"role": "system", "content": "one"},
                {"role": "developer", "content": "two"},
                {"role": "user", "content": "hi"},
            ],
        }
    )
    assert result["system"] == "one\n\ntwo"


def test_chat_request_tool_flow():
    result = chat_request_to_anthropic(
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "12C"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "d",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": "auto",
        }
    )
    assert result["messages"][1] == {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Oslo"}}
        ],
    }
    assert result["messages"][2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "12C"}],
    }
    assert result["tools"] == [
        {"name": "get_weather", "input_schema": {"type": "object", "properties": {}}, "description": "d"}
    ]
    assert result["tool_choice"] == {"type": "auto"}


def test_chat_request_consecutive_tool_messages_merged():
    result = chat_request_to_anthropic(
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                        {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r1"},
                {"role": "tool", "tool_call_id": "c2", "content": "r2"},
            ],
        }
    )
    # both tool results must land in a single user turn (Anthropic requires alternating roles)
    assert len(result["messages"]) == 3
    tool_results = result["messages"][2]["content"]
    assert result["messages"][2]["role"] == "user"
    assert [b["tool_use_id"] for b in tool_results] == ["c1", "c2"]


def test_chat_request_tool_choice_variants():
    base = {
        "model": "m",
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
    }
    assert chat_request_to_anthropic({**base, "tool_choice": "required"})["tool_choice"] == {"type": "any"}
    assert chat_request_to_anthropic({**base, "tool_choice": "none"})["tool_choice"] == {"type": "none"}
    assert chat_request_to_anthropic(
        {**base, "tool_choice": {"type": "function", "function": {"name": "f"}}}
    )["tool_choice"] == {"type": "tool", "name": "f"}
    assert chat_request_to_anthropic({**base, "tool_choice": "auto", "parallel_tool_calls": False})[
        "tool_choice"
    ] == {"type": "auto", "disable_parallel_tool_use": True}


def test_chat_request_image_data_url():
    result = chat_request_to_anthropic(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAAA"},
                        },
                        {"type": "image_url", "image_url": {"url": "https://a/b.jpg"}},
                    ],
                }
            ],
        }
    )
    blocks = result["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "look"}
    assert blocks[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
    }
    assert blocks[2] == {"type": "image", "source": {"type": "url", "url": "https://a/b.jpg"}}


def test_chat_request_system_only_gets_placeholder_user_turn():
    result = chat_request_to_anthropic(
        {"model": "m", "messages": [{"role": "system", "content": "sys only"}]}
    )
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"][0]["type"] == "text"
    assert result["messages"][0]["content"][0]["text"]


def test_chat_request_assistant_first_gets_placeholder_user_turn():
    result = chat_request_to_anthropic(
        {
            "model": "m",
            "messages": [
                {"role": "assistant", "content": "I begin"},
                {"role": "user", "content": "ok"},
            ],
        }
    )
    assert [m["role"] for m in result["messages"]] == ["user", "assistant", "user"]


def test_chat_request_response_format_dropped():
    result = chat_request_to_anthropic(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "response_format": {"type": "json_object"},
        }
    )
    assert "response_format" not in result


# ---------------------------------------------------------------------------
# Anthropic response -> chat response
# ---------------------------------------------------------------------------


def test_anthropic_response_text():
    result = anthropic_response_to_chat(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-x",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 4},
        },
        requested_model="my-alias",
    )
    assert result["object"] == "chat.completion"
    assert result["model"] == "my-alias"
    choice = result["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "Hello!"}
    assert choice["finish_reason"] == "stop"
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 4
    assert result["usage"]["total_tokens"] == 14


def test_anthropic_response_tool_use():
    result = anthropic_response_to_chat(
        {
            "content": [
                {"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "toolu_9", "name": "get_weather", "input": {"city": "Oslo"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
    )
    message = result["choices"][0]["message"]
    assert message["content"] == "Checking."
    tc = message["tool_calls"][0]
    assert tc["id"] == "toolu_9"
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Oslo"}
    assert result["choices"][0]["finish_reason"] == "tool_calls"


def test_anthropic_response_thinking_becomes_reasoning():
    result = anthropic_response_to_chat(
        {
            "content": [
                {"type": "thinking", "thinking": "let me think", "signature": "s"},
                {"type": "text", "text": "done"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    message = result["choices"][0]["message"]
    assert message["reasoning_content"] == "let me think"
    assert message["content"] == "done"


def test_anthropic_response_cache_tokens_folded_into_prompt():
    result = anthropic_response_to_chat(
        {
            "content": [{"type": "text", "text": "x"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 90,
                "cache_creation_input_tokens": 20,
            },
        }
    )
    assert result["usage"]["prompt_tokens"] == 120
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 90
    assert result["usage"]["total_tokens"] == 125


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_round_trip_anthropic_chat_anthropic_request():
    original = {
        "model": "claude-x",
        "max_tokens": 512,
        "system": "sys",
        "messages": [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {"k": 1}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "r"}]},
        ],
    }
    chat = anthropic_request_to_chat(original)
    back = chat_request_to_anthropic(chat)
    assert back["system"] == "sys"
    assert back["max_tokens"] == 512
    assert back["messages"][0] == {"role": "user", "content": [{"type": "text", "text": "q"}]}
    assert back["messages"][1]["content"] == [
        {"type": "text", "text": "a"},
        {"type": "tool_use", "id": "t1", "name": "f", "input": {"k": 1}},
    ]
    assert back["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "r"}
    ]


# ---------------------------------------------------------------------------
# Streaming: chat chunks -> Anthropic events
# ---------------------------------------------------------------------------


def _chat_chunk(delta: dict, finish_reason=None, usage=None) -> dict:
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def test_chat_to_anthropic_stream_text():
    stream = ChatToAnthropicStream(model="claude-x")
    raw = b""
    raw += b"".join(stream.process_chunk(_chat_chunk({"role": "assistant", "content": ""})))
    raw += b"".join(stream.process_chunk(_chat_chunk({"content": "Hel"})))
    raw += b"".join(stream.process_chunk(_chat_chunk({"content": "lo"})))
    raw += b"".join(stream.process_chunk(_chat_chunk({}, finish_reason="stop")))
    raw += b"".join(
        stream.process_chunk(
            {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
        )
    )
    raw += b"".join(stream.finalize())

    events = parse_sse_events(raw)
    types = [e["type"] for e in events]
    assert types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[0]["message"]["model"] == "claude-x"
    assert events[1]["content_block"] == {"type": "text", "text": ""}
    assert events[2]["delta"] == {"type": "text_delta", "text": "Hel"}
    assert events[3]["delta"] == {"type": "text_delta", "text": "lo"}
    assert events[5]["delta"]["stop_reason"] == "end_turn"
    assert events[5]["usage"] == {"input_tokens": 7, "output_tokens": 3}

    # the raw stream must carry event: lines matching data types (SDK requirement)
    text = raw.decode("utf-8")
    assert "event: message_start\n" in text
    assert "event: content_block_delta\n" in text
    assert "event: message_stop\n" in text


def test_chat_to_anthropic_stream_tool_calls():
    stream = ChatToAnthropicStream(model="claude-x")
    raw = b""
    raw += b"".join(
        stream.process_chunk(
            _chat_chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }
                    ]
                }
            )
        )
    )
    raw += b"".join(
        stream.process_chunk(
            _chat_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]})
        )
    )
    raw += b"".join(
        stream.process_chunk(
            _chat_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"Oslo"}'}}]})
        )
    )
    raw += b"".join(stream.process_chunk(_chat_chunk({}, finish_reason="tool_calls")))
    raw += b"".join(stream.finalize())

    events = parse_sse_events(raw)
    starts = [e for e in events if e["type"] == "content_block_start"]
    assert starts[0]["content_block"]["type"] == "tool_use"
    assert starts[0]["content_block"]["id"] == "call_1"
    assert starts[0]["content_block"]["name"] == "get_weather"

    deltas = [e for e in events if e["type"] == "content_block_delta"]
    partial = "".join(d["delta"]["partial_json"] for d in deltas)
    assert json.loads(partial) == {"city": "Oslo"}

    message_delta = next(e for e in events if e["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "tool_use"


def test_chat_to_anthropic_stream_text_then_tool():
    stream = ChatToAnthropicStream(model="claude-x")
    raw = b""
    raw += b"".join(stream.process_chunk(_chat_chunk({"content": "checking..."})))
    raw += b"".join(
        stream.process_chunk(
            _chat_chunk(
                {
                    "tool_calls": [
                        {"index": 0, "id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                    ]
                }
            )
        )
    )
    raw += b"".join(stream.finalize())

    events = parse_sse_events(raw)
    starts = [e for e in events if e["type"] == "content_block_start"]
    stops = [e for e in events if e["type"] == "content_block_stop"]
    assert [s["content_block"]["type"] for s in starts] == ["text", "tool_use"]
    assert [s["index"] for s in starts] == [0, 1]
    assert [s["index"] for s in stops] == [0, 1]


def test_chat_to_anthropic_stream_reasoning():
    stream = ChatToAnthropicStream(model="claude-x")
    raw = b""
    raw += b"".join(stream.process_chunk(_chat_chunk({"reasoning_content": "thinking..."})))
    raw += b"".join(stream.process_chunk(_chat_chunk({"content": "answer"})))
    raw += b"".join(stream.finalize())

    events = parse_sse_events(raw)
    starts = [e for e in events if e["type"] == "content_block_start"]
    assert starts[0]["content_block"]["type"] == "thinking"
    assert starts[1]["content_block"]["type"] == "text"
    thinking_delta = next(e for e in events if e["type"] == "content_block_delta")
    assert thinking_delta["delta"] == {"type": "thinking_delta", "thinking": "thinking..."}


def test_chat_to_anthropic_stream_length_finish():
    stream = ChatToAnthropicStream(model="m")
    stream.process_chunk(_chat_chunk({"content": "x"}, finish_reason="length"))
    events = parse_sse_events(b"".join(stream.finalize()))
    message_delta = next(e for e in events if e["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "max_tokens"


def test_chat_to_anthropic_stream_tool_use_with_stop_finish_forces_tool_use():
    stream = ChatToAnthropicStream(model="m")
    stream.process_chunk(
        _chat_chunk(
            {
                "tool_calls": [
                    {"index": 0, "id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ]
            },
            finish_reason="stop",
        )
    )
    events = parse_sse_events(b"".join(stream.finalize()))
    message_delta = next(e for e in events if e["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "tool_use"


def test_chat_to_anthropic_stream_empty_produces_valid_envelope():
    stream = ChatToAnthropicStream(model="m")
    events = parse_sse_events(b"".join(stream.finalize()))
    assert [e["type"] for e in events] == ["message_start", "message_delta", "message_stop"]


# ---------------------------------------------------------------------------
# Streaming: Anthropic events -> chat chunks
# ---------------------------------------------------------------------------


def test_anthropic_to_chat_stream_text():
    stream = AnthropicToChatStream(model="claude-x")
    raw = b""
    raw += b"".join(
        stream.process_event(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "usage": {"input_tokens": 9, "output_tokens": 0},
                },
            }
        )
    )
    raw += b"".join(
        stream.process_event(
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
        )
    )
    raw += b"".join(
        stream.process_event(
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}
        )
    )
    raw += b"".join(stream.process_event({"type": "content_block_stop", "index": 0}))
    raw += b"".join(
        stream.process_event(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 3},
            }
        )
    )
    raw += b"".join(stream.process_event({"type": "message_stop"}))

    chunks = parse_sse_events(raw)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert chunks[1]["choices"][0]["delta"] == {"content": "Hi"}
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"
    usage_chunk = chunks[3]
    assert usage_chunk["usage"] == {
        "prompt_tokens": 9,
        "completion_tokens": 3,
        "total_tokens": 12,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    assert chunks[-1]["type"] == "[DONE]"


def test_anthropic_to_chat_stream_tool_use():
    stream = AnthropicToChatStream(model="claude-x")
    raw = b""
    raw += b"".join(stream.process_event({"type": "message_start", "message": {"usage": {}}}))
    raw += b"".join(
        stream.process_event(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}},
            }
        )
    )
    raw += b"".join(
        stream.process_event(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"city": "Oslo"}'},
            }
        )
    )
    raw += b"".join(stream.process_event({"type": "content_block_stop", "index": 0}))
    raw += b"".join(
        stream.process_event(
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 2}}
        )
    )
    raw += b"".join(stream.process_event({"type": "message_stop"}))

    chunks = parse_sse_events(raw)
    tc_start = chunks[1]["choices"][0]["delta"]["tool_calls"][0]
    assert tc_start["index"] == 0
    assert tc_start["id"] == "toolu_1"
    assert tc_start["function"] == {"name": "get_weather", "arguments": ""}
    tc_args = chunks[2]["choices"][0]["delta"]["tool_calls"][0]
    assert tc_args["function"]["arguments"] == '{"city": "Oslo"}'
    finish = next(c for c in chunks if c.get("choices") and c["choices"][0].get("finish_reason"))
    assert finish["choices"][0]["finish_reason"] == "tool_calls"


def test_anthropic_to_chat_stream_thinking():
    stream = AnthropicToChatStream(model="m")
    raw = b""
    raw += b"".join(stream.process_event({"type": "message_start", "message": {"usage": {}}}))
    raw += b"".join(
        stream.process_event(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            }
        )
    )
    raw += b"".join(
        stream.process_event(
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}}
        )
    )
    chunks = parse_sse_events(raw)
    assert chunks[1]["choices"][0]["delta"] == {"reasoning_content": "hmm"}


def test_anthropic_to_chat_stream_multiple_tools_get_sequential_indices():
    stream = AnthropicToChatStream(model="m")
    stream.process_event({"type": "message_start", "message": {"usage": {}}})
    first = parse_sse_events(
        b"".join(
            stream.process_event(
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "tool_use", "id": "a", "name": "f1", "input": {}},
                }
            )
        )
    )
    second = parse_sse_events(
        b"".join(
            stream.process_event(
                {
                    "type": "content_block_start",
                    "index": 2,
                    "content_block": {"type": "tool_use", "id": "b", "name": "f2", "input": {}},
                }
            )
        )
    )
    assert first[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert second[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 1


def test_anthropic_to_chat_stream_ping_ignored():
    stream = AnthropicToChatStream(model="m")
    assert stream.process_event({"type": "ping"}) == []


def test_anthropic_to_chat_stream_error_event():
    stream = AnthropicToChatStream(model="m")
    raw = b"".join(
        stream.process_event(
            {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
        )
    )
    chunks = parse_sse_events(raw)
    assert chunks[0]["error"]["message"] == "busy"
    assert chunks[-1]["type"] == "[DONE]"
    # finalize after error must not emit anything further
    assert stream.finalize() == []


def test_anthropic_to_chat_stream_finalize_without_stop_emits_stop():
    stream = AnthropicToChatStream(model="m")
    stream.process_event({"type": "message_start", "message": {"usage": {"input_tokens": 1}}})
    chunks = parse_sse_events(b"".join(stream.finalize()))
    assert chunks[0]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["type"] == "[DONE]"


# ---------------------------------------------------------------------------
# Errors and headers
# ---------------------------------------------------------------------------


def test_error_bodies():
    body = anthropic_error_body(404, "no route")
    assert body == {"type": "error", "error": {"type": "not_found_error", "message": "no route"}}
    assert anthropic_error_body(429, "x")["error"]["type"] == "rate_limit_error"
    assert anthropic_error_body(418, "x")["error"]["type"] == "api_error"

    body = openai_error_body(400, "bad")
    assert body["error"]["message"] == "bad"
    assert body["error"]["type"] == "invalid_request_error"
    assert openai_error_body(500, "x")["error"]["type"] == "api_error"


def test_extract_error_message():
    assert extract_error_message(b'{"error": {"message": "boom"}}') == "boom"
    assert extract_error_message({"type": "error", "error": {"type": "api_error", "message": "m"}}) == "m"
    assert extract_error_message("plain text") == "plain text"
    assert extract_error_message(None) == "upstream error"


def test_build_anthropic_headers():
    headers = build_anthropic_headers({"x-api-key": "client-key"}, "endpoint-key")
    assert headers["x-api-key"] == "client-key"
    assert headers["anthropic-version"] == "2023-06-01"

    headers = build_anthropic_headers({}, "endpoint-key")
    assert headers["x-api-key"] == "endpoint-key"

    headers = build_anthropic_headers({"authorization": "Bearer tok"}, "")
    assert headers["x-api-key"] == "tok"

    headers = build_anthropic_headers({"anthropic-version": "2024-01-01"}, "k")
    assert headers["anthropic-version"] == "2024-01-01"


def test_build_openai_headers():
    headers = build_openai_headers({"authorization": "Bearer tok"}, "endpoint-key")
    assert headers["Authorization"] == "Bearer tok"

    headers = build_openai_headers({}, "endpoint-key")
    assert headers["Authorization"] == "Bearer endpoint-key"

    headers = build_openai_headers({"x-api-key": "anth-key"}, "")
    assert headers["Authorization"] == "Bearer anth-key"

    headers = build_openai_headers({}, "")
    assert "Authorization" not in headers

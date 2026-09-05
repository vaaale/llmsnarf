import json

from openaiproxy.models.trace_models import extract_usage


def test_extract_usage_openai_non_streaming():
    body = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    assert extract_usage(body, None) == (10, 5)


def test_extract_usage_openai_streaming_trailing_chunk():
    chunks = [
        'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}\n\n'
        "data: [DONE]\n\n"
    ]
    assert extract_usage(None, chunks) == (7, 3)


def test_extract_usage_anthropic_streaming_events():
    chunks = [
        'data: {"type":"message_start","message":{"usage":{"input_tokens":12,"output_tokens":0}}}\n\n'
        'data: {"type":"message_delta","usage":{"output_tokens":8}}\n\n'
    ]
    assert extract_usage(None, chunks) == (12, 8)


def test_extract_usage_llamacpp_timings_non_streaming():
    body = {
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        "timings": {"prompt_n": 456, "predicted_n": 468},
    }
    assert extract_usage(body, None) == (456, 468)


def test_extract_usage_llamacpp_timings_streaming_final_chunk():
    final_chunk = json.dumps(
        {
            "choices": [{"finish_reason": "stop", "index": 0, "delta": {}}],
            "timings": {"prompt_n": 456, "predicted_n": 468},
        }
    )
    chunks = [
        'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        f"data: {final_chunk}\n\n"
        "data: [DONE]\n\n"
    ]
    assert extract_usage(None, chunks) == (456, 468)


def test_extract_usage_prefers_usage_over_timings_when_both_present():
    chunks = [
        'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3},'
        '"timings":{"prompt_n":456,"predicted_n":468}}\n\n'
    ]
    assert extract_usage(None, chunks) == (7, 3)


def test_extract_usage_returns_none_when_no_usage_information():
    chunks = ['data: {"choices":[{"delta":{"content":"hi"}}]}\n\n']
    assert extract_usage(None, chunks) == (None, None)

from __future__ import annotations

import json
import re
from typing import Any

from openaiproxy.models.ledger_models import ValidationIssue

_UNCLOSED_TAG_RE = re.compile(r"<([a-zA-Z_][a-zA-Z0-9_]*)(?:\s[^>]*)?>(?!.*</\1>)", re.DOTALL)
_AUTOLINK_RE = re.compile(r"^\[[^\]]+]\(https?://[^)]+\)$")


def _issues_from_request(payload: Any) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not isinstance(payload, dict):
        issues.append(ValidationIssue(code="request.not_object", message="Request body is not a JSON object"))
        return issues

    if not payload.get("model"):
        issues.append(ValidationIssue(code="request.missing_model", message="Request has no 'model' field", path="model"))

    messages = payload.get("messages")
    if messages is not None:
        if not isinstance(messages, list):
            issues.append(ValidationIssue(code="request.messages_not_array", message="'messages' is not an array", path="messages"))
        else:
            for idx, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    issues.append(ValidationIssue(code="request.message_not_object", message=f"Message at index {idx} is not an object", path=f"messages[{idx}]"))
                    continue
                if not msg.get("role"):
                    issues.append(ValidationIssue(code="request.message_missing_role", message=f"Message at index {idx} has no 'role'", path=f"messages[{idx}].role"))

    return issues


def _issues_from_chat_response(body: Any) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not isinstance(body, dict):
        return issues

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return issues

    for ci, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue

        content = message.get("content")
        if isinstance(content, str) and content:
            for match in _UNCLOSED_TAG_RE.finditer(content):
                tag = match.group(1)
                issues.append(ValidationIssue(
                    code="response.unclosed_tag",
                    message=f"Unclosed XML/model tag <{tag}> in assistant content",
                    path=f"choices[{ci}].message.content",
                ))

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for ti, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                arguments = fn.get("arguments")
                if not isinstance(arguments, str):
                    continue
                path = f"choices[{ci}].message.tool_calls[{ti}].function.arguments"

                try:
                    args = json.loads(arguments)
                except json.JSONDecodeError:
                    issues.append(ValidationIssue(
                        code="response.tool_call.invalid_json",
                        message=f"tool_call '{fn.get('name', '?')}' arguments are not valid JSON",
                        path=path,
                    ))
                    continue

                if not isinstance(args, dict):
                    issues.append(ValidationIssue(
                        code="response.tool_call.args_not_object",
                        message=f"tool_call '{fn.get('name', '?')}' arguments parsed to {type(args).__name__}, expected object",
                        path=path,
                    ))
                    continue

                for key, value in args.items():
                    if value is None:
                        issues.append(ValidationIssue(
                            code="response.tool_call.null_arg",
                            message=f"tool_call '{fn.get('name', '?')}' argument '{key}' is null",
                            path=f"{path}.{key}",
                        ))
                    elif isinstance(value, str) and _AUTOLINK_RE.match(value):
                        issues.append(ValidationIssue(
                            code="response.tool_call.autolink_arg",
                            message=f"tool_call '{fn.get('name', '?')}' argument '{key}' looks like a Markdown autolink",
                            path=f"{path}.{key}",
                        ))
                    elif isinstance(value, str):
                        try:
                            parsed = json.loads(value)
                            if isinstance(parsed, list):
                                issues.append(ValidationIssue(
                                    code="response.tool_call.stringified_array",
                                    message=f"tool_call '{fn.get('name', '?')}' argument '{key}' is a JSON-stringified array",
                                    path=f"{path}.{key}",
                                ))
                        except (json.JSONDecodeError, ValueError):
                            pass

        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            issues.append(ValidationIssue(
                code="response.truncated",
                message="Response was truncated due to max_tokens limit",
                path=f"choices[{ci}].finish_reason",
            ))

    return issues


def _issues_from_stream_chunks(chunks: list[str]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    accumulated_content = ""
    accumulated_args: dict[int, dict[str, str]] = {}
    last_finish_reason: str | None = None

    for chunk in chunks:
        for line in chunk.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            if choice.get("finish_reason"):
                last_finish_reason = choice["finish_reason"]
            content_delta = delta.get("content")
            if isinstance(content_delta, str):
                accumulated_content += content_delta
            for tc_delta in delta.get("tool_calls") or []:
                idx = tc_delta.get("index", 0)
                acc = accumulated_args.setdefault(idx, {"name": "", "arguments": ""})
                fn = tc_delta.get("function") or {}
                if fn.get("name"):
                    acc["name"] += fn["name"]
                if fn.get("arguments"):
                    acc["arguments"] += fn["arguments"]

    if accumulated_content:
        for match in _UNCLOSED_TAG_RE.finditer(accumulated_content):
            tag = match.group(1)
            issues.append(ValidationIssue(
                code="response.unclosed_tag",
                message=f"Unclosed XML/model tag <{tag}> in streamed assistant content",
                path="stream.content",
            ))

    for idx, acc in accumulated_args.items():
        arguments = acc["arguments"]
        name = acc["name"] or "?"
        path = f"stream.tool_calls[{idx}].arguments"
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            issues.append(ValidationIssue(
                code="response.tool_call.invalid_json",
                message=f"Streamed tool_call '{name}' arguments are not valid JSON",
                path=path,
            ))
            continue
        if not isinstance(args, dict):
            issues.append(ValidationIssue(
                code="response.tool_call.args_not_object",
                message=f"Streamed tool_call '{name}' arguments parsed to {type(args).__name__}, expected object",
                path=path,
            ))
            continue
        for key, value in args.items():
            if value is None:
                issues.append(ValidationIssue(
                    code="response.tool_call.null_arg",
                    message=f"Streamed tool_call '{name}' argument '{key}' is null",
                    path=f"{path}.{key}",
                ))
            elif isinstance(value, str) and _AUTOLINK_RE.match(value):
                issues.append(ValidationIssue(
                    code="response.tool_call.autolink_arg",
                    message=f"Streamed tool_call '{name}' argument '{key}' looks like a Markdown autolink",
                    path=f"{path}.{key}",
                ))
            elif isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        issues.append(ValidationIssue(
                            code="response.tool_call.stringified_array",
                            message=f"Streamed tool_call '{name}' argument '{key}' is a JSON-stringified array",
                            path=f"{path}.{key}",
                        ))
                except (json.JSONDecodeError, ValueError):
                    pass

    if last_finish_reason == "length":
        issues.append(ValidationIssue(
            code="response.truncated",
            message="Streamed response was truncated due to max_tokens limit",
            path="stream.finish_reason",
        ))

    return issues


class ValidationService:
    def validate_trace(
        self,
        request_payload: Any,
        response_body: Any,
        response_chunks: list[str],
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        issues.extend(_issues_from_request(request_payload))
        if response_chunks:
            issues.extend(_issues_from_stream_chunks(response_chunks))
        elif response_body is not None:
            issues.extend(_issues_from_chat_response(response_body))
        return issues

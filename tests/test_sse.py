"""Tests for app.api.sse."""

import json

from app.api.sse import format_delta_chunk, format_done, format_error_event


def test_delta_chunk_openai_shape() -> None:
    """Delta 事件为 OpenAI 兼容结构：data: 前缀 + choices[0].delta.content + 空行。"""
    payload = format_delta_chunk("亲")

    assert payload == 'data: {"choices":[{"index":0,"delta":{"content":"亲"}}]}\n\n'
    event = json.loads(payload.removeprefix("data: "))
    assert event["choices"][0]["index"] == 0
    assert event["choices"][0]["delta"]["content"] == "亲"


def test_delta_chunk_index_param() -> None:
    """index 参数写入 choices[0].index。"""
    payload = format_delta_chunk("您好", index=2)

    event = json.loads(payload.removeprefix("data: "))
    assert event["choices"][0]["index"] == 2
    assert event["choices"][0]["delta"]["content"] == "您好"


def test_error_event_shape() -> None:
    """错误事件：data: 前缀 + error.message + 空行。"""
    payload = format_error_event("服务暂时不可用")

    assert payload == 'data: {"error": {"message": "服务暂时不可用"}}\n\n'
    event = json.loads(payload.removeprefix("data: "))
    assert event["error"]["message"] == "服务暂时不可用"


def test_done_marker() -> None:
    """结束标记为字面量 data: [DONE] 加空行。"""
    assert format_done() == "data: [DONE]\n\n"

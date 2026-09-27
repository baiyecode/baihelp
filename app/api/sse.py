"""OpenAI 兼容的 SSE 事件格式化。

每个函数返回一条可直接写入 ``text/event-stream`` 响应体的完整 SSE 事件
（含结尾空行）；流式端点把返回值原样下发即可。
"""

import json


def format_delta_chunk(text: str, index: int = 0) -> str:
    """把一段增量文本格式化为 OpenAI 兼容的 delta 事件。"""
    payload = json.dumps(
        {"choices": [{"index": index, "delta": {"content": text}}]},
        ensure_ascii=False,
        separators=(",", ":"),  # 紧凑风格，与规范 §5.1 的线上示例逐字节一致
    )
    return f"data: {payload}\n\n"


def format_error_event(message: str) -> str:
    """把错误消息格式化为流内 error 事件。"""
    payload = json.dumps({"error": {"message": message}}, ensure_ascii=False)
    return f"data: {payload}\n\n"


def format_done() -> str:
    """返回 SSE 流的结束标记。"""
    return "data: [DONE]\n\n"

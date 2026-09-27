"""售后消息结构化抽取编排：prompt → with_structured_output → parsed。"""

from langchain_core.language_models.chat_models import BaseChatModel

from app.prompts.loader import load_extraction_prompt
from app.schemas.extraction import AfterSalesExtraction


class ExtractionError(RuntimeError):
    """抽取失败：上游未产出可解析的结构化结果（端点映射为 502）。"""


async def extract_after_sales(model: BaseChatModel, text: str) -> AfterSalesExtraction:
    """把一条顾客消息交给抽取 prompt + 结构化输出，返回解析结果。

    - 链形状：``load_extraction_prompt() | model.with_structured_output(...,
      include_raw=True)``，ainvoke 得到 ``{"raw", "parsed", "parsing_error"}``；
    - ``parsing_error`` 非空或 ``parsed`` 为空时抛 ``ExtractionError``；
      模型调用本身的异常原样向上传播，均由端点映射为 502。
    """
    prompt = load_extraction_prompt()
    structured = model.with_structured_output(AfterSalesExtraction, include_raw=True)
    result = await (prompt | structured).ainvoke({"text": text})
    if result["parsed"] is None or result["parsing_error"] is not None:
        msg = (
            "售后信息抽取失败：上游未返回可解析的结构化结果"
            f"（parsing_error={result['parsing_error']!r}）"
        )
        raise ExtractionError(msg)
    return result["parsed"]

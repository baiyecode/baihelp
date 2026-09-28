r"""问法规范化:全半角归一 + 去空白标点 + 小写,产出用于去重比对的规范键。

约定:
- 只保留字母数字与 CJK 汉字,其余(空白、全半角标点、符号)一律剔除;
- 归一结果仅用于等值比对(问法去重),不用于展示;
- 挖矿去重(mine_qa)与测试共用本实现,勿在他处复制逻辑。
"""

import re
import unicodedata

# 归一保留集:先 NFKC 折叠全角为半角再 lower,故只需匹配小写字母/数字/CJK 基本区
_KEEP_RE = re.compile(r"[^0-9a-z\u4e00-\u9fff]+")


def normalize_question(text: str) -> str:
    """问法规范化:NFKC 全半角归一 → 小写 → 剔除字母数字与 CJK 以外的全部字符。

    「优惠券怎么用!」与「优惠券怎么用？」归一为同一键;纯标点串归一为空串。
    """
    return _KEEP_RE.sub("", unicodedata.normalize("NFKC", text).lower())

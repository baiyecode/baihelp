"""normalize_question 规范化测试:全半角归一、去空白标点、小写,产出去重比对键。"""

from app.knowledge.normalize import normalize_question


def test_full_and_half_width_punctuation_normalize_equal() -> None:
    """半角 ! 与全角 ? 结尾的同句问法归一相等,且等于纯内容串。"""
    assert normalize_question("优惠券怎么用!") == normalize_question("优惠券怎么用？")
    assert normalize_question("优惠券怎么用!") == "优惠券怎么用"


def test_full_width_letters_fold_to_ascii_lowercase() -> None:
    """全角字母折叠为半角并小写,内部空白一并剔除。"""
    assert normalize_question("ｃａｓｅ ＡＢＣ") == normalize_question("case ABC")
    assert normalize_question("ｃａｓｅ ＡＢＣ") == "caseabc"


def test_pure_punctuation_normalizes_to_empty() -> None:
    """纯标点/空白串(含全角句读)归一为空串——空串彼此视为同一问法。"""
    assert normalize_question("!!!？？？。。。~~ 　") == ""

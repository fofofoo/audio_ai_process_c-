# -*- coding: utf-8 -*-
"""关键词前缀树（ContextGraph）对拍 —— 搬上游 gtest 的期望值。

============================================================
这份代码是干什么的
============================================================
`src/kws_decode.py` 里的 `_ContextGraph` 是
`sherpa-onnx/csrc/context-graph.cc` 的逐行移植。上游唯一带真实断言的单元测试
就是 `context-graph-test.cc`，它用 9 条互为前后缀的字符串（S / HE / SHE /
SHELL / HIS / HERS / HELLO / THIS / THEM）压 fail 链和 output 弧，并把
`ForwardOneStep + Finalize` 的累计分数与写死的期望值比对。

本脚本把那 4 组用例原样搬过来（strict / 非 strict × 默认分 / 自定义分），
再加一项关键词文件解析检查 —— 因为它不需要 onnx，纯 numpy 就能跑。

用法：
    python -I test_context_graph_mini.py

全部通过时最后一行打印 OK；任一断言失败会直接抛 AssertionError 并指出用例名。
"""

import math
import os
import sys

# -I 隔离模式下脚本目录不进 sys.path（也忽略 PYTHONPATH），
#   不显式插一次，下面的 from src.* 会直接 ImportError
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.kws_decode import KeywordSpotter, _ContextGraph

# 上游 context-graph-test.cc 的 9 条关键词
CONTEXTS = ["S", "HE", "SHE", "SHELL", "HIS", "HERS", "HELLO", "THIS", "THEM"]

# 4 组用例：{查询串: 期望的累计总分}
CASES = [
    # TestBasic：所有关键词都是同一档分数（score=0 表示退化成 context_score=1）
    ("TestBasic", 0, True, {
        "HEHERSHE": 14, "HERSHE": 12, "HISHE": 9,
        "SHED": 6, "SHELF": 6, "HELL": 2,
        "HELLO": 7, "DHRHISQ": 4, "THEN": 2}),
    ("TestBasicNonStrict", 0, False, {
        "HEHERSHE": 7, "HERSHE": 5, "HISHE": 5, "SHED": 3, "SHELF": 3,
        "HELL": 2, "HELLO": 2, "DHRHISQ": 3, "THEN": 2}),
    ("TestCustomize", 5, True, {
        "HEHERSHE": 35.84, "HERSHE": 30.84, "HISHE": 24.18,
        "SHED": 18.34, "SHELF": 18.34, "HELL": 5,
        "HELLO": 13, "DHRHISQ": 10.84, "THEN": 5}),
    ("TestCustomizeNonStrict", 5, False, {
        "HEHERSHE": 20, "HERSHE": 15, "HISHE": 10.84,
        "SHED": 10, "SHELF": 10, "HELL": 5,
        "HELLO": 5, "DHRHISQ": 5.84, "THEN": 5}),
]


def setup_stdout_utf8():
    """把 stdout/stderr 切成 UTF-8（Windows 控制台默认 GBK）。"""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:      # 极老的 Python 才没有 reconfigure
            pass


def _round2(x):
    """C++ std::round(x * 100) / 100 的 Python 版。

    Python 的 round() 是「银行家舍入」（.5 进偶数），和 std::round 的
    「.5 远离零」不一致；本文件的期望值是在 C++ 里算出来的，必须照抄。
    """
    return math.floor(x * 100 + 0.5) / 100.0


def build_graph(base_score):
    """按上游 TestHelper 的方式建图：每条关键词的边分 = 总分 / 字数。"""
    token_ids = [[ord(c) for c in w] for w in CONTEXTS]
    scores = [_round2(base_score / len(w)) for w in CONTEXTS]
    return _ContextGraph(token_ids, 1.0, 0.0, scores)


def run_case(name, base_score, strict_mode, expected):
    """跑一组查询，累计 ForwardOneStep + Finalize 的分数并比对。"""
    graph = build_graph(base_score)
    for query, want in expected.items():
        state = graph.root
        total = 0.0
        for ch in query:
            score, state, _ = graph.forward_one_step(
                state, ord(ch), strict_mode)
            total += score
        score, state = graph.finalize(state)
        total += score
        assert state.token == -1, "%s：Finalize 没回到根" % name
        assert abs(total - want) < 1e-3, (
            "%s：查询 %r 总分 %.2f，期望 %.2f" % (name, query, total, want))
    print("  %-24s OK（%d 条查询）" % (name, len(expected)))


def check_keyword_file():
    """用真实 tokens.txt 解析真实 keywords.txt，确认字段没被吃掉。"""
    kws_dir = os.path.join(
        _HERE, "weight", "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20")
    tokens_txt = os.path.join(kws_dir, "tokens.txt")
    keywords_txt = os.path.join(kws_dir, "test_wavs", "keywords.txt")

    _, token2id = KeywordSpotter._load_tokens(tokens_txt)
    ids, scores, phrases, thresholds = KeywordSpotter._load_keywords(
        keywords_txt, token2id)

    want_phrases = ["LIGHT_UP", "LOVELY_CHILD", "文森特卡索", "周望军",
                    "朱丽楠", "蒋友伯", "女儿", "法国", "见面会", "落实"]
    assert phrases == want_phrases, "关键词原文解析不对：%r" % (phrases,)
    assert len(ids) == len(want_phrases)
    # 文件里没写 :score / #threshold，应当全部落到 0，由构造参数兜底
    assert all(s == 0.0 for s in scores), "意外的 boosting score：%r" % (scores,)
    assert all(t == 0.0 for t in thresholds), \
        "意外的触发阈值：%r" % (thresholds,)
    print("  关键词文件              OK（%d 条，token 全部能查表）" % len(ids))


def main():
    setup_stdout_utf8()
    print("=" * 60)
    print("ContextGraph 对拍（期望值来自上游 context-graph-test.cc）")
    print("=" * 60)
    for name, base_score, strict_mode, expected in CASES:
        run_case(name, base_score, strict_mode, expected)
    check_keyword_file()
    print("-" * 60)
    print("OK")


if __name__ == "__main__":
    main()

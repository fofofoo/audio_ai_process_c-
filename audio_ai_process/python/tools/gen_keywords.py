# -*- coding: utf-8 -*-
"""从「原文清单」生成 sherpa-onnx KWS 关键词文件。

============================================================
这份脚本是干什么的
============================================================
KWS 的词典分两级，这个脚本负责第一级：

    keywords/kws_keywords_raw.txt       人写：一行一个唤醒词
              │
              │  sherpa_onnx.utils.text2token(phone+ppinyin)
              │     中文 -> pypinyin 注音表拆声母/带调韵母
              │     英文 -> en.phone（CMU 发音词典）
              ▼
    keywords/kws_keywords.txt           生成物，识别侧唯一消费者是
                                        src/kws_decode.py: _load_keywords()

第二级（音素符号 -> 整数 id）由识别侧在加载时用 tokens.txt 完成，本脚本
不参与 —— 它只保证「符号」这一侧对得上。

============================================================
raw 文件格式
============================================================
每行::

    原文 [:boosting_score] [#threshold]

- `原文` 可以含空格（英文短语），生成时自动把空格换成 `_` 写进 `@原文`；
- `:x` 关键词加分、`#x` 触发阈值，都可省略，省略时识别侧回落到全局默认
  （score=1.0 / threshold=0.25）；
- `#` 开头的整行是注释，不会生成关键词。

`@原文` 由脚本自动补，不要在 raw 里手写。

============================================================
为什么逐条调用 text2token
============================================================
`sherpa_onnx.utils.text2token` 对转写失败的词是 `continue`（见
sherpa_onnx/utils.py:133 的非 CJK 跳过、:173 的词表 OOV 跳过），**返回值
长度可能小于输入条数**。批量传会让关键词与 token 行错位，而且错得无声无息。
所以这里一条一次调用，空列表立刻报错并指出行号。

用法::

    python -I tools/gen_keywords.py            # 生成，默认不覆盖已有文件
    python -I tools/gen_keywords.py --force    # 覆盖
    python -I tools/gen_keywords.py --dry-run  # 只打印，不写盘

依赖（仅生成期，不进运行时）::

    pip install pypinyin sentencepiece
"""

import argparse
import os
import sys

# -I 隔离模式下脚本目录不进 sys.path，项目根要手动插
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.kws_decode import KeywordSpotter

# ---- 路径：全部写死，不接受命令行参数以外的输入 ----
_KWS_DIR = os.path.join(
    _ROOT, "weight", "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20")
TOKENS_TXT = os.path.join(_KWS_DIR, "tokens.txt")
EN_PHONE = os.path.join(_KWS_DIR, "en.phone")

RAW_TXT = os.path.join(_ROOT, "keywords", "kws_keywords_raw.txt")
OUT_TXT = os.path.join(_ROOT, "keywords", "kws_keywords.txt")

# 这个模型的词表是「CMU 音素 + 部分拼音」，只能用 phone+ppinyin
TOKENS_TYPE = "phone+ppinyin"


def setup_stdout_utf8():
    """把 stdout/stderr 切成 UTF-8（Windows 控制台默认 GBK）。"""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:      # 极老的 Python 才没有 reconfigure
            pass


def check_deps():
    """text2token 需要 pypinyin；它对 sentencepiece 是函数内无条件 import。"""
    try:
        import pypinyin                 # noqa: F401
    except ImportError:
        raise SystemExit(
            "缺少 pypinyin：\n"
            "  C:/Users/Liyun/anaconda3/envs/py310-gpu/python.exe -m pip "
            "install pypinyin sentencepiece")
    try:
        import sentencepiece            # noqa: F401
    except ImportError:
        raise SystemExit(
            "缺少 sentencepiece（text2token 会无条件 import 它）：\n"
            "  C:/Users/Liyun/anaconda3/envs/py310-gpu/python.exe -m pip "
            "install sentencepiece")


def load_tokens(path):
    """读 tokens.txt 的第 1 列，返回 {token: id}，用于生成后的自检。"""
    table = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                table[parts[0]] = int(parts[1])
    return table


def parse_raw(path):
    """解析 raw 文件，返回 [(lineno, phrase, score, threshold)]。

    `:` / `#` 出现之前的所有 word 都算原文（支持英文多词短语）。
    参数写 0 会被识别侧当成「未设置」而回落全局默认，所以显式拒绝。
    """
    items = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            words = line.split()
            split_at = len(words)
            for i, w in enumerate(words):
                if w[0] in ":#@":
                    split_at = i
                    break
            phrase = " ".join(words[:split_at]).strip()
            params = words[split_at:]

            if not phrase:
                raise SystemExit("raw 第 %d 行：缺少原文" % lineno)

            score, threshold = 0.0, 0.0
            for w in params:
                if w[0] == ":":
                    score = float(w[1:])
                elif w[0] == "#":
                    threshold = float(w[1:])
                elif w[0] == "@":
                    raise SystemExit(
                        "raw 第 %d 行：不要手写 @原文，脚本会自动补" % lineno)
                else:
                    raise SystemExit(
                        "raw 第 %d 行：无法识别的参数 %r" % (lineno, w))

            if score == 0.0 and any(w.startswith(":") for w in params):
                raise SystemExit(
                    "raw 第 %d 行：:0 会被当成「未设置」，请直接省略该参数"
                    % lineno)
            if threshold == 0.0 and any(w.startswith("#") for w in params):
                raise SystemExit(
                    "raw 第 %d 行：#0 会被当成「未设置」，请直接省略该参数"
                    % lineno)

            items.append((lineno, phrase, score, threshold))

    if not items:
        raise SystemExit("raw 文件里没有有效关键词：%s" % path)
    return items


def transpose(phrase):
    """原文 -> 音素符号列表。转写失败直接抛，不允许静默跳过。

    调的库：sherpa_onnx.utils.text2token（仅开发期依赖，运行时不加载）。
    它做的事（tokens_type = "phone+ppinyin"）：
      1. 读 tokens.txt 建 {符号: id}，只用于末尾的 OOV 校验；
      2. 读 en.phone（CMU 发音词典）建 {英文词: [音素...]}；
      3. 按空格切词：英文词查词典直出音素；纯 CJK 词走 pypinyin
         拆成「声母 + 带调韵母」；两者都不是则整条文本丢弃。
      4. 返回「符号」列表（output_ids 默认 False，id 映射留给识别侧）。
    注意：它遇到 OOV 是静默 continue，不抛异常，所以这里一次只传一条、
    返回空就自己报错。
    """
    from sherpa_onnx.utils import text2token

    # 单条调用：批量传时 OOV 文本会被静默跳过，返回列表变短，
    # 关键词与 token 行就错位了
    res = text2token([phrase], TOKENS_TXT, TOKENS_TYPE, lexicon=EN_PHONE)

    # 库的失败方式是「返回空」而不是抛异常：[] 或 [[]] 都算失败
    if not res or not res[0]:
        raise SystemExit(
            "转写失败：%r\n"
            "  常见原因：英文词不在 en.phone 里；拼音单元不在 tokens.txt 里。\n"
            "  跑 sherpa-onnx 原生的 text2token 同样会静默跳过这条，"
            "所以这里直接报错。" % phrase)

    # 库的返回元素类型是 Union[str, int]（output_ids=True 时给 id），
    # 这里统一成符号字符串
    return [str(t) for t in res[0]]


def format_line(tokens, score, threshold, phrase):
    """拼最终一行：token... [:score] [#threshold] @原文。"""
    parts = list(tokens)
    if score:
        parts.append(":%g" % score)
    if threshold:
        parts.append("#%g" % threshold)
    # 上游约定：@ 后面不能有空格，用 _ 代替
    parts.append("@%s" % phrase.replace(" ", "_"))
    return " ".join(parts)


def main():
    setup_stdout_utf8()

    ap = argparse.ArgumentParser(
        description="从 kws_keywords_raw.txt 生成 kws_keywords.txt")
    ap.add_argument("--force", action="store_true",
                    help="覆盖已存在的输出文件")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印结果，不写盘")
    args = ap.parse_args()

    check_deps()

    for p in (TOKENS_TXT, EN_PHONE, RAW_TXT):
        if not os.path.isfile(p):
            raise SystemExit("找不到文件：%s" % p)
    if os.path.exists(OUT_TXT) and not (args.force or args.dry_run):
        raise SystemExit(
            "输出文件已存在，未覆盖：%s\n"
            "  确认要重写请加 --force" % OUT_TXT)

    token2id = load_tokens(TOKENS_TXT)
    items = parse_raw(RAW_TXT)

    print("=" * 68)
    print("生成 KWS 关键词：%d 条  (tokens_type=%s)" % (len(items), TOKENS_TYPE))
    print("=" * 68)

    lines, expected_phrases = [], []
    #   lineno    raw 文件行号，仅用于报错时定位
    #   phrase    原文，可能含空格（英文短语），生成时换成 _ 写进 @原文
    #   score     关键词加分 :x；0 表示未设置，识别侧回落全局默认 1.0
    #   threshold 触发阈值 #x；0 表示未设置，识别侧回落全局默认 0.25
    for i, (lineno, phrase, score, threshold) in enumerate(items, 1):

        # 第 1 步：原文 -> 音素/拼音符号。
        # 一次只传一条：批量传时 text2token 会静默丢弃 OOV 文本，
        # 返回值变短后关键词与 token 行就错位了。
        tokens = transpose(phrase)

        # 第 2 步：符号必须在 tokens.txt 里，否则模型没有对应 id。
        # 正常情况下上一步已挡住（text2token 内部也会跳 OOV），这里是防御性检查。
        missing = [t for t in tokens if t not in token2id]
        if missing:
            raise SystemExit(
                "raw 第 %d 行 %r：生成的 token %r 不在 tokens.txt 里"
                % (lineno, phrase, missing))

        # 第 3 步：拼成 kws_keywords.txt 的一行
        # token... [:score] [#threshold] @原文
        lines.append(format_line(tokens, score, threshold, phrase))

        # 第 4 步：记下期望原文，写盘后给 _load_keywords 回读自检比对
        expected_phrases.append(phrase.replace(" ", "_"))
        print("[%d/%d] %-8s -> %s" % (i, len(items), phrase, " ".join(tokens)))

    # 第 5 步：拼文件正文，末尾保留一个换行
    body = "\n".join(lines) + "\n"

    if args.dry_run:
        print("-" * 68)
        print(body)
        print("-" * 68)
        print("dry-run：未写盘。")
        return

    header = (
        "# 自动生成，请勿手改 —— 改 keywords/kws_keywords_raw.txt 后重跑：\n"
        "#   python -I tools/gen_keywords.py --force\n"
        "# 词表：%s\n" % os.path.basename(_KWS_DIR))
    with open(OUT_TXT, "w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        f.write(body)

    # 生成后自检：用识别侧真正的解析函数读回来，确认条数、原文都对得上
    ids, scores, phrases, thresholds = KeywordSpotter._load_keywords(
        OUT_TXT, token2id)
    assert len(ids) == len(items), (
        "回读条数 %d != 生成条数 %d" % (len(ids), len(items)))
    assert phrases == expected_phrases, (
        "回读原文 %r != 期望 %r" % (phrases, expected_phrases))

    print("-" * 68)
    print("已写入：%s" % OUT_TXT)
    print("自检：_load_keywords 回读 %d 条，原文全部对得上" % len(ids))
    print("下一步：python -I test_mic_kws_asr.py  说唤醒词看 avg_prob")


if __name__ == "__main__":
    main()

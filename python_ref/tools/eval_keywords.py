# -*- coding: utf-8 -*-
"""批量评估录好的唤醒词样本，输出 avg_prob 分布 —— 调 #threshold 的依据。

============================================================
这份脚本是干什么的
============================================================
tools/record_keywords.py 把「说唤醒词」录成 wav，这个脚本把每一条都喂进
KWS 解码器，把触发判定用到的那个数摊开：

    每条样本 -> (是否触发, avg_prob, 被阈值挡住的 avg_prob)

三种结局要区分开，它们指向不同的调参方向：

    触发                avg_prob >= 阈值且尾随 blank 够
    被阈值挡住          trie 已走到关键词末端，但 avg_prob < 阈值
                        -> 说明路径选对了，只是门太窄：调 #threshold
    什么都没发生        trie 根本没走到末端
                        -> 说明束搜索没把这条路径推上去：调 :score

第三种没有任何输出，所以「连 debug 都没有」本身就是信息。

============================================================
为什么直接读 recordings/ 目录
============================================================
目录名就是关键词原文（record_keywords.py 建的），不用额外维护清单。
每个词一个子目录，里面是 01.wav 02.wav ……

用法::

    python -I tools/eval_keywords.py
    python -I tools/eval_keywords.py --threshold 0.3   # 临时改全局阈值
    python -I tools/eval_keywords.py --score 1.5       # 临时改全局加分
    python -I tools/eval_keywords.py --only 小智同学
    python -I tools/eval_keywords.py --dir recordings2

`--threshold` / `--score` 只影响本次评估，不改任何文件 —— 可以反复扫，
定下来之后再回填到 keywords/kws_keywords_raw.txt。
"""

import argparse
import contextlib
import io
import os
import sys

import numpy as np
import onnxruntime as ort

# -I 隔离模式下脚本目录不进 sys.path，项目根要手动插
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.audio_preprocess import AudioPreprocess
from src.kws_decode import KeywordSpotter

# ---- 路径：与 test_wav_kws_mini.py 用同一套权重 ----
_KWS_DIR = os.path.join(
    _ROOT, "weight", "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20")
ENCODER_ONNX = os.path.join(
    _KWS_DIR, "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx")
DECODER_ONNX = os.path.join(
    _KWS_DIR, "decoder-epoch-13-avg-2-chunk-8-left-64.onnx")
JOINER_ONNX = os.path.join(
    _KWS_DIR, "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx")
TOKENS_TXT = os.path.join(_KWS_DIR, "tokens.txt")

# 仓库自有关键词文件（gen_keywords.py 生成），不是 weight/ 里的测试词
KEYWORDS_TXT = os.path.join(_ROOT, "keywords", "kws_keywords.txt")
REC_DIR = os.path.join(_ROOT, "recordings")

PROVIDERS = ["CPUExecutionProvider"]
BLOCK = 160                 # 10 ms @16 kHz，与麦克风链路一致
TAIL_SILENCE_S = 0.5        # 末尾补真静音，凑出触发所需的 trailing blanks


def setup_stdout_utf8():
    """把 stdout/stderr 切成 UTF-8（Windows 控制台默认 GBK）。"""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:      # 极老的 Python 才没有 reconfigure
            pass


def build_sessions():
    """建三个 onnx session。KWS 只有 3M，CPU 单线程足够。"""
    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    so.log_severity_level = 3

    def _sess(path):
        if not os.path.isfile(path):
            raise FileNotFoundError("onnx 不存在：%s" % path)
        return ort.InferenceSession(path, so, providers=PROVIDERS)

    return _sess(ENCODER_ONNX), _sess(DECODER_ONNX), _sess(JOINER_ONNX)


def load_wav(path, sample_rate):
    """读 wav，返回 float32 单声道样本；采样率不对直接报错。"""
    import soundfile as sf

    data, sr = sf.read(path, always_2d=True, dtype="float32")
    if sr != sample_rate:
        raise ValueError("%s 采样率 %d != %d" % (path, sr, sample_rate))
    return data[:, 0]


def run_one(kws, samples):
    """按 BLOCK 切块喂完，返回 (命中的 KeywordHit 列表, 走到的最深 trie 层级)。

    第二项用于诊断未命中：level=0 说明一个音都没对上，level=7（对 8 token
    的关键词）说明只差最后一个音。
    """
    front = AudioPreprocess(sample_rate=kws.sample_rate, n_mels=kws.n_mels)
    hits = []
    max_level = 0
    for i in range(0, len(samples), BLOCK):
        feats = front.accept_waveform(samples[i:i + BLOCK])
        hits.extend(kws.accept_features(feats))
        if kws.best_trie_level > max_level:
            max_level = kws.best_trie_level
    return hits, max_level


def evaluate_file(kws, path):
    """评估一条 wav，返回 (hits, blocked)。

    blocked 是这条样本里「走到关键词末端但没过阈值」的事件；一次语音可能
    对多个关键词都走到末端，这里保留全部。
    """
    samples = load_wav(path, kws.sample_rate)
    padded = np.concatenate(
        [samples, np.zeros(int(TAIL_SILENCE_S * kws.sample_rate),
                           dtype=np.float32)])

    kws.reset()
    kws.debug_events.clear()
    # debug=True 会把每条「被阈值挡住」写成一行 stderr；这里是批量评估，
    # 结果统一走 stdout 的表格，stderr 噪声直接吞掉（事件仍在 debug_events 里）
    with contextlib.redirect_stderr(io.StringIO()):
        hits, max_level = run_one(kws, padded)
    return hits, list(kws.debug_events), max_level


def fmt_probs(values):
    """把一组 avg_prob 打成 min/mean/max。"""
    if not values:
        return "-"
    a = np.asarray(values, dtype=np.float64)
    return "%.3f/%.3f/%.3f" % (a.min(), a.mean(), a.max())


def main():
    setup_stdout_utf8()

    ap = argparse.ArgumentParser(description="批量评估录制的唤醒词样本")
    ap.add_argument("--dir", default=REC_DIR, help="样本根目录")
    ap.add_argument("--threshold", type=float, default=None,
                    help="临时覆盖全局触发阈值（默认用构造函数默认值）")
    ap.add_argument("--score", type=float, default=None,
                    help="临时覆盖全局 boosting score（默认用构造函数默认值）")
    ap.add_argument("--only", default=None, help="只评估指定的词")
    ap.add_argument("--keywords", default=KEYWORDS_TXT,
                    help="关键词文件（默认仓库里生成的那份）。做「只留一个词」"
                         "的对照实验时，指向一份临时文件即可，不用改 raw")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        raise SystemExit(
            "找不到样本目录：%s\n  先跑 python -I tools/record_keywords.py"
            % args.dir)

    # ---- 1. 装配 ----
    kw = {}
    if args.threshold is not None:
        kw["keywords_threshold"] = args.threshold
    if args.score is not None:
        kw["keywords_score"] = args.score

    enc, dec, joi = build_sessions()
    # debug=True：需要「被阈值挡住」的事件，那才是调阈值的依据
    kws = KeywordSpotter(enc, dec, joi, TOKENS_TXT, args.keywords,
                         debug=True, **kw)
    info = kws.info()

    print("=" * 74)
    print("唤醒词样本评估")
    print("=" * 74)
    print("样本目录 : %s" % args.dir)
    print("关键词   : %d 条  %s" % (info["n_keywords"],
                                    " / ".join(kws.keywords)))
    print("全局参数 : score=%.2f  threshold=%.3f%s"
          % (kws._graph.context_score, kws._graph.ac_threshold,
             "  (命令行覆盖)" if kw else "  (默认)"))
    print()

    # ---- 2. 逐词逐条评估 ----
    dirs = sorted(d for d in os.listdir(args.dir)
                  if os.path.isdir(os.path.join(args.dir, d)))
    if args.only:
        dirs = [d for d in dirs if d == args.only]
    if not dirs:
        raise SystemExit("样本目录下没有子目录：%s" % args.dir)

    summary = []        # [(phrase, n, n_hit, hit_probs, blocked_probs, wrong)]
    for phrase in dirs:
        sub = os.path.join(args.dir, phrase)
        wavs = sorted(n for n in os.listdir(sub) if n.lower().endswith(".wav"))
        if not wavs:
            print("[%s] 目录里没有 wav，跳过" % phrase)
            continue

        print("-" * 74)
        print("[%s]  %d 条" % (phrase, len(wavs)))
        print("-" * 74)

        hit_probs, blocked_probs, wrong = [], [], []
        # 每条样本的结局，用于最后的「该调哪个参数」判断。
        # 注意这是样本级：一次语音可能对多个关键词都被挡住，但只算一条样本。
        outcomes = []
        for name in wavs:
            path = os.path.join(sub, name)
            hits, blocked, max_level = evaluate_file(kws, path)

            if hits:
                # 以第一条命中为准；一条音频里多次命中同一词也算一次样本
                h = hits[0]
                hit_probs.append(h.avg_prob)
                outcomes.append("hit")
                tag = "命中 %.3f" % h.avg_prob
                if h.keyword != phrase:
                    wrong.append((name, h.keyword))
                    tag += "  <- 触发的是 %s" % h.keyword
                print("  %-8s %s" % (name, tag))
            elif blocked:
                # 可能对多个关键词都走到末端，取最高的那个
                b = max(blocked, key=lambda e: e["avg_prob"])
                blocked_probs.append(b["avg_prob"])
                outcomes.append("blocked")
                print("  %-8s 未命中  被阈值挡住 %.3f < %.3f  (%s)"
                      % (name, b["avg_prob"], b["threshold"], b["keyword"]))
            else:
                outcomes.append("none")
                # max_level 是整条音频里 trie 走到的最深处：
                # 0 = 一个音都没对上，接近关键词长度 = 只差最后一个音
                print("  %-8s 未命中  没走到关键词末端（最远到第 %d 个 token）"
                      % (name, max_level))

        n = len(wavs)
        print("  -- 命中 %d/%d (%.0f%%)  命中时 avg_prob: %s"
              % (len(hit_probs), n, 100.0 * len(hit_probs) / n,
                 fmt_probs(hit_probs)))
        if blocked_probs:
            print("  -- 被阈值挡住: %s  （最高 %.3f，阈值降到它以下才会命中）"
                  % (fmt_probs(blocked_probs), max(blocked_probs)))
        if wrong:
            print("  -- 触发错词: %s"
                  % " / ".join("%s->%s" % (a, b) for a, b in wrong))
        print()

        summary.append((phrase, n, len(hit_probs), hit_probs,
                        blocked_probs, wrong, outcomes))

    # ---- 3. 汇总表 ----
    print("=" * 74)
    print("汇总")
    print("=" * 74)
    print("关键词        样本  命中  命中率  命中时 avg_prob(min/mean/max)  最高被挡")
    for phrase, n, n_hit, hp, bp, _wrong, _out in summary:
        pad = " " * max(0, 10 - len(phrase) * 2)   # 中文按 2 字宽粗略对齐
        print("%s%s  %4d  %4d  %5.0f%%  %-26s  %s"
              % (phrase, pad, n, n_hit, 100.0 * n_hit / n,
                 fmt_probs(hp), "%.3f" % max(bp) if bp else "-"))

    # ---- 4. 给出下一步该动哪个参数 ----
    # 按样本级结局统计：一条样本只算一次，避免「多个词同时被挡住」重复计数
    print()
    outcomes = [o for *_x, out in summary for o in out]
    n_hit = outcomes.count("hit")
    n_blocked = outcomes.count("blocked")
    n_none = outcomes.count("none")

    if n_blocked:
        lowest = min(p for *_x, bp, _w, _o in summary for p in bp)
        print("提示：%d 条走到末端但被阈值挡住 -> 试着降阈值："
              % n_blocked)
        print("        python -I tools/eval_keywords.py --threshold %.2f"
              % max(lowest - 0.02, 0.01))
    if n_none:
        print("提示：%d 条连 trie 末端都没走到 -> 这不是阈值问题，"
              "该给对应关键词加 :score" % n_none)
    if n_blocked == 0 and n_none == 0:
        print("提示：全部命中，当前阈值下没有漏检。想验证抗误触，"
              "需要录一些「不该触发」的负样本（说相似音、日常对话）。")


if __name__ == "__main__":
    main()

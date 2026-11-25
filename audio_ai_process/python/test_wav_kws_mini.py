# -*- coding: utf-8 -*-
"""离线 wav 关键词唤醒 —— mini 版入口（KWS 流水线的第一层验收）。

============================================================
这份代码是干什么的
============================================================
读模型自带的 9 条 `test_wavs/*.wav`，用 `src/kws_decode.py` 的
`KeywordSpotter` 跑流式关键词唤醒，打印每条 wav 命中的关键词、时间戳和
平均声学概率。它和 `test_wav_asr_mini.py` 同构，区别只在解码器：

    test_wav_asr_mini.py   AudioPreprocess -> AsrDecoder        -> 整句文本
    test_wav_kws_mini.py   AudioPreprocess -> KeywordSpotter    -> 命中的关键词

三个 onnx 与两份文本（tokens.txt / keywords.txt）都来自
`weight/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/`，默认用 chunk-8
（每 160 ms 推进一次，唤醒延迟低于 chunk-16）。

============================================================
为什么要在 wav 末尾补静音
============================================================
触发条件要求关键词后面至少出现 2 个 blank 输出帧（80 ms）。wav 里关键词
后面可能紧跟着别的词，没有静音就永远不触发。真机麦克风是连续流，这个问题
被掩盖了；离线验收必须在末尾补一段真静音（0 值，不是特征域的 0），
这里补 0.5 秒。

============================================================
验收内容
============================================================
    1) 逐条 wav 跑一遍，打印命中的关键词 / 时间戳 / 平均概率；
    2) 命中的关键词集合与 EXPECTED 完全一致：zh_0~2 是负样本（不许误触发），
       zh_3~6 各含两个关键词，en_0/en_1 各含一个；
    3) 流式不变性：同一条 wav 用 160 / 1600 两种块长喂，命中结果必须一致
       （块长只影响喂入节奏，不该影响解码结果）。

用法：
    python -I test_wav_kws_mini.py

全部通过时最后一行打印 OK；有 wav 没命中会打印 FAIL 并以非零码退出。
"""

import os
import sys

# -I 隔离模式下脚本目录不进 sys.path（也忽略 PYTHONPATH），
#   不显式插一次，下面的 from src.* 会直接 ImportError
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import onnxruntime as ort

from src.audio_preprocess import AudioPreprocess
from src.kws_decode import KeywordSpotter

# onnxruntime 推理用cpu或gpu
PROVIDERS = ["CPUExecutionProvider"]    #"CUDAExecutionProvider",

# KWS 权重目录（chunk-8 默认）
_KWS_DIR = os.path.join(
    _HERE, "weight", "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20")
ENCODER_ONNX = os.path.join(_KWS_DIR, "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx")
DECODER_ONNX = os.path.join(_KWS_DIR, "decoder-epoch-13-avg-2-chunk-8-left-64.onnx")
JOINER_ONNX = os.path.join(_KWS_DIR, "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx")
TOKENS_TXT = os.path.join(_KWS_DIR, "tokens.txt")
KEYWORDS_TXT = os.path.join(_KWS_DIR, "test_wavs", "keywords.txt")
WAV_DIR = os.path.join(_KWS_DIR, "test_wavs")

# 等流式节奏的块长：160 样本 = 10ms @16kHz，与 MicClient 默认值一致
BLOCK = 160
# 流式不变性复测用的块长：1600 样本 = 100ms
BLOCK_LARGE = 1600
# wav 末尾补的静音时长（秒）：用于凑出触发所需的 trailing blanks
TAIL_SILENCE_S = 0.5

# 每条 wav 期望命中的关键词（原文来自 keywords.txt 的 @ 后缀）。
# 用 ASR 模型转写过这 9 条音频：zh_0~2 讲的是经济话题，不含任何关键词，
# 是负样本；zh_3~6 每条各含两个关键词。
EXPECTED = {
    "en_0.wav": {"LIGHT_UP"},
    "en_1.wav": {"LOVELY_CHILD"},
    "zh_0.wav": set(),                              # 负样本
    "zh_1.wav": set(),                              # 负样本
    "zh_2.wav": set(),                              # 负样本
    "zh_3.wav": {"文森特卡索", "法国"},
    "zh_4.wav": {"蒋友伯", "女儿"},
    "zh_5.wav": {"周望军", "落实"},
    "zh_6.wav": {"朱丽楠", "见面会"},
}


def build_ort_sessions():
    """按写死的三条路径建三个 onnx session。"""
    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    so.log_severity_level = 3

    def _sess(path):
        if not os.path.isfile(path):
            raise FileNotFoundError("onnx 不存在：%s" % path)
        return ort.InferenceSession(path, so, providers=PROVIDERS)

    return _sess(ENCODER_ONNX), _sess(DECODER_ONNX), _sess(JOINER_ONNX)


def load_wav(path):
    """读 wav，返回 float32 单声道样本。多声道只取第一声道。"""
    import soundfile as sf

    if not os.path.isfile(path):
        raise FileNotFoundError("wav 不存在：%s" % path)
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return data[:, 0], int(sr)


def run_one(kws, samples, sample_rate, block):
    """把一段音频按 block 样本切块喂完，返回命中的关键词列表。"""
    front = AudioPreprocess(sample_rate=sample_rate, n_mels=kws.n_mels)
    hits = []
    for i in range(0, len(samples), block):
        feats = front.accept_waveform(samples[i:i + block])
        hits.extend(kws.accept_features(feats))
    return hits


def setup_stdout_utf8():
    """把 stdout/stderr 切成 UTF-8（Windows 控制台默认 GBK）。"""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:      # 极老的 Python 才没有 reconfigure
            pass


def main():
    setup_stdout_utf8()

    # ========================================================
    # 1.建三个 session，装配关键词唤醒器
    # ========================================================
    enc, dec_sess, joi = build_ort_sessions()
    kws = KeywordSpotter(enc, dec_sess, joi, TOKENS_TXT, KEYWORDS_TXT)
    info = kws.info()

    print("=" * 74)
    print("mini 离线 wav KWS —— 文件 -> 特征 -> 命中的关键词")
    print("=" * 74)
    print("权重  : %s" % _KWS_DIR)
    print("onnx  : encoder=%s  decoder=%s  joiner=%s"
          % (os.path.basename(ENCODER_ONNX), os.path.basename(DECODER_ONNX),
             os.path.basename(JOINER_ONNX)))
    print("执行器: %s" % " -> ".join(
        p.replace("ExecutionProvider", "") for p in info["providers"]))
    print("模型  : T=%d(输入帧数)  decode_chunk_len=%d(每轮前移)  "
          "context_size=%d  vocab=%d  状态张量=%d  帧长=%.0fms"
          % (info["T"], info["decode_chunk_len"], info["context_size"],
             info["vocab_size"], info["n_states"], info["frame_shift_s"] * 1e3))
    print("关键词: %d 条  %s" % (info["n_keywords"], " / ".join(kws.keywords)))
    print()

    # ========================================================
    # 2.逐条 wav 跑，收集结果
    # ========================================================
    wav_names = sorted(n for n in os.listdir(WAV_DIR)
                       if n.endswith(".wav") and n.startswith(("en_", "zh_")))
    results = {}
    all_ok = True

    print("-" * 74)
    for name in wav_names:
        path = os.path.join(WAV_DIR, name)
        samples, sr = load_wav(path)
        if sr != kws.sample_rate:
            print("%s: 采样率 %d != %d，跳过" % (name, sr, kws.sample_rate))
            all_ok = False
            continue

        # 末尾补真静音，凑出触发所用的 trailing blanks
        padded = np.concatenate(
            [samples, np.zeros(int(TAIL_SILENCE_S * sr), dtype=np.float32)])

        kws.reset()
        hits = run_one(kws, padded, sr, BLOCK)
        results[name] = hits

        dur = len(samples) / float(sr)
        if not hits:
            print("%-9s %5.2fs  未命中" % (name, dur))
        else:
            desc = "  ".join(
                "%s@%.2fs(p=%.2f)" % (h.keyword, h.start_time, h.avg_prob)
                for h in hits)
            print("%-9s %5.2fs  %s" % (name, dur, desc))

        # 期望集合检查：命中的关键词集合必须与期望完全一致
        # （负样本期望空集，多出一个词也算 FAIL）
        want = EXPECTED.get(name)
        got = {h.keyword for h in hits}
        if want is None:
            ok = bool(hits)
        else:
            ok = (got == want)
            missing = want - got
            extra = got - want
            if missing:
                print("          缺少期望的关键词：%s"
                      % " / ".join(sorted(missing)))
            if extra:
                print("          出现期望之外的关键词：%s"
                      % " / ".join(sorted(extra)))
        if not ok:
            all_ok = False
            print("          FAIL")

    # ========================================================
    # 3.流式不变性：换块长重跑第一条 wav，看命中是否一致
    # ========================================================
    if wav_names:
        name = wav_names[0]
        samples, sr = load_wav(os.path.join(WAV_DIR, name))
        padded = np.concatenate(
            [samples, np.zeros(int(TAIL_SILENCE_S * sr), dtype=np.float32)])
        kws.reset()
        hits_large = run_one(kws, padded, sr, BLOCK_LARGE)

        def _sig(hits):
            return [(h.keyword, round(h.start_time, 2)) for h in hits]

        same = _sig(results[name]) == _sig(hits_large)
        print("-" * 74)
        print("流式不变性 %s：块 160 -> %s   块 1600 -> %s   %s"
              % (name, _sig(results[name]), _sig(hits_large),
                 "OK" if same else "FAIL"))
        all_ok = all_ok and same

    # ========================================================
    # 4.汇总
    # ========================================================
    total_hits = sum(len(v) for v in results.values())
    print("-" * 74)
    print("encoder 调用 %d 次 / 共 %d 帧   累计触发 %d 次"
          % (kws.n_chunks, kws.n_enc_frames, total_hits))
    print("OK" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

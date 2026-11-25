# -*- coding: utf-8 -*-
"""KWS 对拍 —— 用 sherpa-onnx 官方 wheel 当解码 oracle。

============================================================
为什么要有这个脚本
============================================================
上游没有逐 token / 逐帧的 golden 数据，`context-graph-test.cc` 只覆盖 trie。
解码器（束搜索 + 触发判定）最可靠的对照物就是 sherpa-onnx 自己：pip 装下来的
wheel 里跑的就是同一份 C++ `TransducerKeywordDecoder`。

本脚本对同一条 wav 跑两条链路，比较：
    1) 命中的关键词集合是否一致；
    2) 关键词第一个 token 的时间戳是否在容差内（默认 0.2 s）。

============================================================
一个已知的前端口径差异
============================================================
本项目 `AudioPreprocess` 默认 high_freq=0（8000 Hz，跟模型作者 test_onnx.py
一致）；sherpa-onnx C++ 默认 high_freq=-400（7600 Hz），而 wheel 的 Python API
不暴露这个参数。所以两边的特征在 7600~8000 Hz 带略有不同 —— 这是**预期内**的
差异。若结果在边缘样本上不一致，先看时间戳偏差是否只是几十毫秒。

依赖：`pip install sherpa-onnx`（仅开发期对拍用，不是运行时依赖）。
用法：python -I test_kws_oracle_mini.py
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
import soundfile as sf

from src.audio_preprocess import AudioPreprocess
from src.kws_decode import KeywordSpotter

# onnxruntime 推理用cpu或gpu
PROVIDERS = ["CPUExecutionProvider"]

_KWS_DIR = os.path.join(
    _HERE, "weight", "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20")
ENCODER_ONNX = os.path.join(_KWS_DIR, "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx")
DECODER_ONNX = os.path.join(_KWS_DIR, "decoder-epoch-13-avg-2-chunk-8-left-64.onnx")
JOINER_ONNX = os.path.join(_KWS_DIR, "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx")
TOKENS_TXT = os.path.join(_KWS_DIR, "tokens.txt")
KEYWORDS_TXT = os.path.join(_KWS_DIR, "test_wavs", "keywords.txt")
WAV_DIR = os.path.join(_KWS_DIR, "test_wavs")

BLOCK = 160              # 与 MicClient 默认块长一致
TAIL_SILENCE_S = 0.5     # 末尾补静音，凑出触发所需的 trailing blanks
TIME_TOL = 0.2           # 时间戳容差（秒）；约等于 5 个输出帧


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
    """读 wav，返回 (float32 单声道样本, 采样率)。"""
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return data[:, 0], int(sr)


def run_ours(kws, samples):
    """本项目实现：按 BLOCK 切块喂完，返回 [(keyword, 首 token 时间)]。"""
    front = AudioPreprocess(sample_rate=16000, n_mels=kws.n_mels)
    out = []
    for i in range(0, len(samples), BLOCK):
        for h in kws.accept_features(front.accept_waveform(samples[i:i + BLOCK])):
            out.append((h.keyword, h.timestamps[0] if h.timestamps else 0.0))
    return out


def run_oracle(kws, samples, sr):
    """sherpa-onnx wheel：整段喂入，逐个解码 step 取结果。"""
    stream = kws.create_stream()
    stream.accept_waveform(sr, samples)
    stream.input_finished()
    out = []
    while kws.is_ready(stream):
        kws.decode_stream(stream)
        # 必须一次调用同时取到 keyword 和 timestamps：Python 封装里的
        # get_result() / timestamps() 各自都会调一次底层的 GetKeywordResult(true)，
        # 第二次调用会因为去重逻辑返回空结果。
        r = kws.keyword_spotter.get_result(stream)
        if r.keyword:
            out.append((r.keyword.strip(),
                        r.timestamps[0] if r.timestamps else 0.0))
            # 注意：**不要**在这里调 reset_stream。它会把帧偏移清零，
            # 同一个文件里第二个关键词的时间戳会变成「相对上次重置」，
            # 和本项目的绝对时间戳对不上。底层的去重逻辑已经保证
            # 同一个结果不会被重复返回。
    return out


def setup_stdout_utf8():
    """把 stdout/stderr 切成 UTF-8（Windows 控制台默认 GBK）。"""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:      # 极老的 Python 才没有 reconfigure
            pass


def _fmt(hits):
    if not hits:
        return "-"
    return "  ".join("%s@%.2fs" % (k, t) for k, t in hits)


def main():
    setup_stdout_utf8()

    try:
        import sherpa_onnx
    except ImportError:
        print("没装 sherpa-onnx，先执行：")
        print("  C:/Users/Liyun/anaconda3/envs/py310-gpu/python.exe -m pip install sherpa-onnx")
        return 2

    enc, dec_sess, joi = build_ort_sessions()
    ours = KeywordSpotter(enc, dec_sess, joi, TOKENS_TXT, KEYWORDS_TXT)
    oracle = sherpa_onnx.KeywordSpotter(
        tokens=TOKENS_TXT,
        encoder=ENCODER_ONNX,
        decoder=DECODER_ONNX,
        joiner=JOINER_ONNX,
        keywords_file=KEYWORDS_TXT,
        num_threads=1,
        max_active_paths=4,
        num_trailing_blanks=1,
        keywords_score=1.0,
        keywords_threshold=0.25,
        provider="cpu",
    )

    print("=" * 74)
    print("KWS 对拍：本项目 (high_freq=0) vs sherpa-onnx %s (high_freq=-400)"
          % sherpa_onnx.__version__)
    print("=" * 74)

    names = sorted(n for n in os.listdir(WAV_DIR)
                   if n.endswith(".wav") and n.startswith(("en_", "zh_")))
    all_ok = True
    for name in names:
        samples, sr = load_wav(os.path.join(WAV_DIR, name))
        padded = np.concatenate(
            [samples, np.zeros(int(TAIL_SILENCE_S * sr), dtype=np.float32)])

        ours.reset()
        a = run_ours(ours, padded)
        b = run_oracle(oracle, padded, sr)

        set_a = {k for k, _ in a}
        set_b = {k for k, _ in b}
        ok = (set_a == set_b)
        # 集合一致时再比时间戳（同名关键词按出现顺序配对）
        if ok:
            for k, t in a:
                # 找到 oracle 里同名、最近的一次
                cand = [tb for kb, tb in b if kb == k]
                if not cand or min(abs(t - tb) for tb in cand) > TIME_TOL:
                    ok = False
                    break

        print("%-9s 本项目: %-46s oracle: %s   %s"
              % (name, _fmt(a), _fmt(b), "OK" if ok else "DIFF"))
        all_ok = all_ok and ok

    print("-" * 74)
    print("OK" if all_ok else "DIFF（先看时间戳偏差，再判断是否前端口径导致）")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

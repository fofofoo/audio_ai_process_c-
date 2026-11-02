#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C++ wav ASR 与 Python 参考实现的逐 token 对拍。

流程：
    同一条 wav
      ├─ C++   ：test_wav_asr.out --no-window --dump-tokens → name<TAB>id,id,...
      └─ Python：stdlib wave 解码 → python_ref/src/audio_preprocess.py
                 → python_ref/src/asr_decode.py（同一个 160 样本块长）
    逐 token 比较两侧的 id 序列，同时打印 Python 侧文本供肉眼核对。

为什么要用参考实现而不是只比文本：
    detokenize 会把 ▁、中英文空格规则都抹平，不同 token 序列也可能拼出
    同一句话。逐 token 比较能直接把差异定位到「第几个 token、哪个 id」，
    这是确认 C++ 移植（分块判据、状态张量、blank 处理）真的等价的最短路径。

用法（模型权重与 C++ token 导出文件都不入库，路径按需覆盖）：
    /root/miniconda3/envs/pytorch-gpu/bin/python -I tools/compare_asr.py
    ... tools/compare_asr.py --cpp /tmp/cpp_tokens.txt --wav-dir <test_wavs>

前置：先跑 C++ 侧导出
    ./build/test_wav_asr.out --no-window --dump-tokens /tmp/cpp_tokens.txt

判定：
    每个 wav 的 token 序列完全相同 → 返回 0；
    出现缺失文件、长度不同或首个差异位置 → 打印细节并返回 1。
"""

import argparse
import importlib.util
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REF_PRE_PATH = os.path.join(ROOT, "python_ref", "src", "audio_preprocess.py")
REF_ASR_PATH = os.path.join(ROOT, "python_ref", "src", "asr_decode.py")
DEFAULT_MODEL_DIR = os.path.join(
    ROOT, "weight",
    "sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-"
    "2026-06-05")

# 与 test_wav_asr.cpp 对齐：CPU、intra/inter 各 1 线程、160 样本一块。
BLOCK_SIZE = 160
TAIL_PAD_FRAMES = 100


def load_module(name, path):
    """按文件路径加载参考模块，避免依赖 sys.path / 包结构。"""
    if not os.path.isfile(path):
        raise SystemExit("找不到参考实现：%s" % path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_wav_pcm16(path):
    """读 PCM16 wav，返回 (单声道 float32 [-1,1], 采样率)。

    与 C++ tools/wav_reader.h 同口径：多声道取第 0 声道，/32768.0，
    不重采样。C++ 侧对非 16 kHz 只提示不转换，这里保持一致。
    """
    with wave.open(path, "rb") as w:
        channels = w.getnchannels()
        sample_width = w.getsampwidth()
        sample_rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    if sample_width != 2:
        raise SystemExit("%s：只支持 PCM16 wav，当前每样本 %d 字节"
                         % (path, sample_width))

    data = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        data = data.reshape(-1, channels)[:, 0]
    return data.astype(np.float32) / 32768.0, sample_rate


def load_cpp_tokens(path):
    """读 C++ 导出的 name<TAB>id,id,...，返回 {文件名: [id, ...]}。

    允许空 token 列表（整段没识别出字），此时逗号后为空。
    """
    if not os.path.isfile(path):
        raise SystemExit("找不到 C++ token 导出文件：%s\n"
                         "先运行 ./build/test_wav_asr.out --no-window "
                         "--dump-tokens %s" % (path, path))
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            name, sep, ids = line.partition("\t")
            if not sep:
                raise SystemExit("token 导出格式不对（缺少 TAB）：%r" % line)
            result[name] = [int(x) for x in ids.split(",") if x != ""]
    return result


def python_tokens(front_cls, decoder_cls, sessions, tokens_path, samples, sample_rate):
    """按 C++ 入口的节奏跑参考实现，返回 (token_ids, text, n_mels)。

    每次喂 160 样本，空特征也照喂（accept_features 能处理 0 帧）；
    收尾调用 input_finished()，把尾部帧和 100 帧特征域 0 一并交给解码器。
    """
    decoder = decoder_cls(*sessions, tokens_path=tokens_path,
                          sample_rate=sample_rate)
    front = front_cls(sample_rate=sample_rate, n_mels=decoder.n_mels,
                      tail_pad_frames=TAIL_PAD_FRAMES)

    for start in range(0, len(samples), BLOCK_SIZE):
        block = samples[start:start + BLOCK_SIZE]
        decoder.accept_features(front.accept_waveform(block))
    decoder.accept_features(front.input_finished())

    return [int(x) for x in decoder.token_ids], decoder.text, decoder.n_mels


def main():
    parser = argparse.ArgumentParser(
        description="C++ wav ASR 与 Python 参考实现逐 token 对拍")
    parser.add_argument("--cpp", default="/tmp/cpp_tokens.txt",
                        help="C++ --dump-tokens 导出的文件（默认 /tmp/cpp_tokens.txt）")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help="模型目录（含三个 onnx 与 tokens.txt）")
    parser.add_argument("--wav-dir", default=None,
                        help="wav 目录（默认 <模型目录>/test_wavs）")
    parser.add_argument("--wavs", nargs="*", default=None,
                        help="只对拍指定 wav 文件名；默认按 C++ 导出文件里的顺序")
    args = parser.parse_args()

    wav_dir = args.wav_dir or os.path.join(args.model_dir, "test_wavs")
    encoder_path = os.path.join(args.model_dir, "encoder.int8.onnx")
    decoder_path = os.path.join(args.model_dir, "decoder.onnx")
    joiner_path = os.path.join(args.model_dir, "joiner.int8.onnx")
    tokens_path = os.path.join(args.model_dir, "tokens.txt")

    # —— 一、参考实现与 ONNX 会话（CPU，线程数与 C++ 一致）——
    pre_mod = load_module("audio_preprocess_ref", REF_PRE_PATH)
    asr_mod = load_module("asr_decode_ref", REF_ASR_PATH)

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sessions = (
        ort.InferenceSession(encoder_path, options, providers=["CPUExecutionProvider"]),
        ort.InferenceSession(decoder_path, options, providers=["CPUExecutionProvider"]),
        ort.InferenceSession(joiner_path, options, providers=["CPUExecutionProvider"]),
    )

    cpp_tokens = load_cpp_tokens(args.cpp)
    names = args.wavs if args.wavs else list(cpp_tokens)
    if not names:
        raise SystemExit("C++ 导出文件里没有任何 wav 记录：%s" % args.cpp)

    print("=" * 74)
    print("C++ / Python 逐 token 对拍")
    print("=" * 74)
    print("C++ 导出 : %s" % args.cpp)
    print("wav 目录 : %s" % wav_dir)
    print("模型目录 : %s" % args.model_dir)
    print("执行器   : CPU（intra/inter=1，块长 %d 样本）" % BLOCK_SIZE)
    print()

    # —— 二、逐个 wav 跑参考实现并比较 ——
    failures = 0
    for name in names:
        wav_path = os.path.join(wav_dir, name)
        if name not in cpp_tokens:
            print("[%s] C++ 导出里没有这个文件，跳过" % name)
            failures += 1
            continue
        if not os.path.isfile(wav_path):
            print("[%s] 找不到 wav：%s" % (name, wav_path))
            failures += 1
            continue

        samples, sample_rate = read_wav_pcm16(wav_path)
        if sample_rate != 16000:
            print("[%s] 注意：%d Hz，模型按 16 kHz 训练" % (name, sample_rate))

        py_ids, py_text, n_mels = python_tokens(
            pre_mod.AudioPreprocess, asr_mod.AsrDecoder, sessions,
            tokens_path, samples, sample_rate)
        cpp_ids = cpp_tokens[name]

        same = py_ids == cpp_ids
        print("------------------------------------------------------------")
        print("%s  %.2f 秒 / %d 样本 / %d Mel"
              % (name, len(samples) / float(sample_rate), len(samples), n_mels))
        print("  C++    : %d token" % len(cpp_ids))
        print("  Python : %d token" % len(py_ids))

        if same:
            print("  逐 token 一致")
            print("  文本   : %s" % py_text)
        else:
            failures += 1
            first = 0
            while (first < len(py_ids) and first < len(cpp_ids)
                   and py_ids[first] == cpp_ids[first]):
                first += 1
            print("  不一致：首个差异在第 %d 个 token（从 0 数）" % first)
            lo = max(0, first - 3)
            print("    Python[%d:%d] = %s" % (lo, first + 3, py_ids[lo:first + 3]))
            print("    C++   [%d:%d] = %s" % (lo, first + 3, cpp_ids[lo:first + 3]))
            print("  Python 文本：%s" % py_text)

    print()
    print("=" * 74)
    if failures == 0:
        print("全部 %d 个 wav 的 token 序列完全一致" % len(names))
        return 0
    print("有 %d / %d 个 wav 未通过" % (failures, len(names)))
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C++ 前端与 Python 参考实现的逐点对拍。

流程：
    同一条 wav
      ├─ Python：stdlib wave 解码 -> python_ref/src/audio_preprocess.py
      └─ C++   ：裸 float32 -> build/fbank_dump.out -> AudioPreprocess
    两边按相同块长喂入，逐帧逐 Mel 比较 mean|Δ| / max|Δ|。

只比较「波形 -> LogFBank」这一层，不涉及 wav 容器解析和绘图，
所以差异只可能来自分帧、窗、FFT、Mel 滤波器组、取对数这些前端步骤。

用法（项目根目录）：
    python3 -I tools/compare_fbank.py
    python3 -I tools/compare_fbank.py --wav results/xxx.wav --chunks 160,1000,4096,0

判定：
    mean|Δ| <= --mean-tol（默认 1e-5）且 max|Δ| <= --max-tol（默认 1e-3）
    视为通过。全部块长通过时脚本返回 0。
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REF_PATH = os.path.join(ROOT, "python_ref", "src", "audio_preprocess.py")


def load_reference_class():
    """按文件路径加载 Python 参考实现，避免依赖 sys.path / 包结构。"""
    spec = importlib.util.spec_from_file_location("audio_preprocess_ref",
                                                  REF_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AudioPreprocess


def read_wav_pcm16(path):
    """读 PCM16 wav，返回 (单声道 float32 [-1,1], 采样率, 原始声道数)。

    多声道只取第 0 声道，与 C++ 入口和 Python 版 soundfile 的行为一致。
    这里不做重采样：采样率直接决定 Mel 滤波器组，喂错会静默改变特征。
    """
    with wave.open(path, "rb") as w:
        channels = w.getnchannels()
        sample_width = w.getsampwidth()
        sample_rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    if sample_width != 2:
        raise SystemExit("只支持 PCM16 wav，当前每样本 %d 字节" % sample_width)

    data = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        data = data.reshape(-1, channels)[:, 0]
    # 与 soundfile 读 int16 到 float32 的比例一致。
    return data.astype(np.float32) / 32768.0, sample_rate, channels


def python_features(front_cls, samples, sample_rate, chunk, n_mels):
    """按固定块长喂 Python 参考实现，返回 (n_frames, n_mels) float32。"""
    front = front_cls(sample_rate=sample_rate, n_mels=n_mels,
                      tail_pad_frames=0)
    parts = []
    for start in range(0, samples.size, chunk):
        feats = front.accept_waveform(samples[start:start + chunk])
        if feats.size:
            parts.append(feats)
    tail = front.input_finished()
    if tail.size:
        parts.append(tail)
    if not parts:
        return np.zeros((0, n_mels), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def run_cpp(cpp_bin, in_path, out_path, chunk, sample_rate, n_mels):
    """调用 C++ dump 工具；失败时把 stderr 原样抛给调用方。"""
    cmd = [cpp_bin, in_path, out_path, str(chunk), str(sample_rate),
           str(n_mels)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)
    if proc.returncode != 0:
        raise SystemExit("C++ 工具失败（%d）：%s" % (proc.returncode,
                                                     proc.stderr.strip()))
    return proc.stdout.strip()


def compare_one(front_cls, samples, sample_rate, n_mels, chunk, cpp_bin,
                tmp_dir, mean_tol, max_tol):
    """跑一种块长并打印一行结果；返回是否通过。"""
    if chunk <= 0:
        chunk = samples.size
    if chunk <= 0:
        print("wav 为空，跳过")
        return True

    in_path = os.path.join(tmp_dir, "input_%d.f32" % chunk)
    out_path = os.path.join(tmp_dir, "cpp_%d.f32" % chunk)
    samples.astype("<f4").tofile(in_path)

    ref = python_features(front_cls, samples, sample_rate, chunk, n_mels)
    message = run_cpp(cpp_bin, in_path, out_path, chunk, sample_rate, n_mels)

    cpp = np.fromfile(out_path, dtype="<f4")
    if cpp.size % n_mels != 0:
        raise SystemExit("C++ 输出长度 %d 不是 n_mels=%d 的整数倍"
                         % (cpp.size, n_mels))
    cpp = cpp.reshape(-1, n_mels)

    if ref.shape != cpp.shape:
        print("块长 %-6d 帧数不一致：Python=%d C++=%d  FAIL"
              % (chunk, ref.shape[0], cpp.shape[0]))
        return False

    if ref.size == 0:
        print("块长 %-6d 两边都是 0 帧  PASS" % chunk)
        return True

    diff = np.abs(ref.astype(np.float64) - cpp.astype(np.float64))
    mean_diff = float(diff.mean())
    max_diff = float(diff.max())
    frame_idx, mel_idx = np.unravel_index(int(np.argmax(diff)), diff.shape)
    ok = mean_diff <= mean_tol and max_diff <= max_tol

    print("块长 %-6d 帧=%-5d mean|Δ|=%.3e  max|Δ|=%.3e  "
          "max@(帧%d,mel%d)  %s"
          % (chunk, ref.shape[0], mean_diff, max_diff,
             frame_idx, mel_idx, "PASS" if ok else "FAIL"))
    # 顺带打印特征本身的量级，排除「两边都空 / 全零」造成的假通过。
    print("           特征范围 [%.3f, %.3f]  均值 %.3f"
          % (float(ref.min()), float(ref.max()), float(ref.mean())))
    print("           %s" % message)
    return ok


def main():
    parser = argparse.ArgumentParser(description="C++ / Python LogFBank 对拍")
    parser.add_argument(
        "--wav", default=os.path.join(ROOT, "results",
                                      "hk_audio_20261102_085654.wav"),
        help="待比较的 PCM16 wav")
    parser.add_argument(
        "--cpp", default=os.path.join(ROOT, "build", "fbank_dump.out"),
        help="C++ 特征 dump 工具")
    parser.add_argument(
        "--chunks", default="160,1000,4096,0",
        help="块长列表，逗号分隔；0 表示整段一次喂完")
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--mean-tol", type=float, default=1e-5)
    parser.add_argument("--max-tol", type=float, default=1e-3)
    args = parser.parse_args()

    if not os.path.isfile(args.wav):
        raise SystemExit("wav 不存在：%s" % args.wav)
    if not os.path.isfile(args.cpp):
        raise SystemExit("C++ 工具不存在：%s（先编译 fbank_dump.out）"
                         % args.cpp)

    samples, sample_rate, channels = read_wav_pcm16(args.wav)
    print("wav : %s" % args.wav)
    print("      %d 样本 @ %d Hz  %d 声道  %.3f 秒"
          % (samples.size, sample_rate, channels,
             samples.size / float(sample_rate)))
    print("参考: %s" % REF_PATH)

    front_cls = load_reference_class()
    chunks = [int(x) for x in args.chunks.split(",") if x.strip()]

    all_ok = True
    with tempfile.TemporaryDirectory() as tmp_dir:
        for chunk in chunks:
            ok = compare_one(front_cls, samples, sample_rate, args.n_mels,
                             chunk, args.cpp, tmp_dir,
                             args.mean_tol, args.max_tol)
            all_ok = all_ok and ok

    print("结论：%s" % ("全部通过" if all_ok else "存在超差"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

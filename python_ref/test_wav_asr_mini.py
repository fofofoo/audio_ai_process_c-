# -*- coding: utf-8 -*-
"""离线 wav 语音识别 —— mini 版入口（与 test_mic_asr_mini.py 同构）。

============================================================
这份代码是干什么的
============================================================
把 src/ 下的流水线按顺序接起来，读一个 wav 文件从头解到尾。
和 test_mic_asr_mini.py 的区别只有一个 —— 音频从**文件**来，不是从麦克风来：

    test_mic_asr_mini.py   麦克风 -> 裸音频块（MicClient 管采集）
    test_wav_asr_mini.py   wav 文件 -> 裸音频块（本脚本自己切）

预处理与解码两条流水线原封不动（同一个 AudioPreprocess / AsrDecoder），
连每次喂入的块长都取和 MicClient 默认值一样的 160 样本（10ms），所以同一段
音频走这两条路径，结果应当完全一致 —— 这也是本脚本的另一个用途：互为对照。

不模拟实时节奏（不按音频时长 sleep）：wav 是离线场景，尽快跑完更有用。
语谱图窗口照常接收同一份特征，但解码太快时只会画到最后几屏；解码结束后
最后一屏会停住，按任意键关闭。想看完整滚动过程，用 test_spectrogram_cv_mini.py。

============================================================
调用顺序（对照 main 里的编号）
============================================================
    1) 读 wav              soundfile -> float32 单声道
    2) 建三个 onnx session build_ort_sessions()（路径写在文件顶部）
       装配解码器          AsrDecoder(encoder, decoder, joiner, tokens_path)
       装配语谱图          SpectrogramImage(...)
    3) 预处理              front.accept_waveform(block)
    4) 模型推理 + 5) 后处理 dec.accept_features(feats)
    6) 可视化              cv_frame = spec_view.accept_features(feats)
                          cv2.putText + cv2.imshow
    7) 打印结果            stdout 原地刷新 + 窗口底部识别条

结束（喂完 wav 或窗口里按 q/Esc）后还要走一次收尾：补完尾部帧、并在特征
尾部补 100 帧 0，再把剩下的一起解掉 —— 否则最后几个字会丢。那 100 帧 0
只喂解码器、不喂可视化，否则语谱图末尾会多出一块比静音还亮的假色块。

用法：
    python -I test_wav_asr_mini.py

不收命令行参数 —— 运行配置全部硬编码在文件里：
三个 onnx 与词表的路径、执行器顺序（GPU/CPU）、wav 文件、块长，都在那里改。
模型怎么建、跑在哪块硬件上，也由入口这一层决定；AsrDecoder 只接收现成的
session，既不定位权重文件也不碰 onnxruntime。
"""

import os
import sys
import time

# -I 隔离模式下脚本目录不进 sys.path（也忽略 PYTHONPATH），
#   不显式插一次，下面的 from src.* 会直接 ImportError
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cv2

from src.audio_preprocess import AudioPreprocess
from src.asr_decode import AsrDecoder
from src.spectrogram_image import SpectrogramImage
import onnxruntime as ort

# onnxruntime 推理用cpu或gpu
PROVIDERS = ["CPUExecutionProvider"]    #"CUDAExecutionProvider",

# 语谱图窗口：400 列 x 10ms = 4 秒滚动窗口；底部 56px 用来画识别结果
SPEC_COLS = 400
SPEC_FOOTER_H = 56
# 窗口标题只能用 ASCII：部分 OpenCV 构建对非 ASCII 标题处理不一致
SPEC_WINDOW = "mini spectrogram"


def build_ort_sessions(ENCODER_ONNX,DECODER_ONNX,JOINER_ONNX):
    """按写死的三条路径建三个 onnx session，返回 (encoder, decoder, joiner)。

    建会话放在入口这一层、而不是 AsrDecoder 内部：「模型从哪来、跑在哪块
    硬件上」是运行环境的决定，不是解码逻辑 —— 解码类只接收现成的 session。
    """

    # 单线程 + 静音日志：多个 chunk 之间本来就串行，
    # 让 onnxruntime 再开线程池只会增加调度开销。
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
    """读 wav，返回 (float32 单声道样本, 采样率)。

    多声道只取第一声道（模型是单声道模型）。这里**不做重采样** —— 采样率
    不对时宁可明确报出来，也不要悄悄喂进错误的数据。
    """
    import soundfile as sf

    if not os.path.isfile(path):
        raise FileNotFoundError("wav 不存在：%s" % path)
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return data[:, 0], int(sr)


def setup_stdout_utf8():
    """把 stdout/stderr 切成 UTF-8。

    Windows 控制台默认 GBK，不转的话打印中文会乱码或直接抛
    UnicodeEncodeError（重定向到文件或管道时尤其明显）。
    """
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:      # 极老的 Python 才没有 reconfigure
            pass


def main():
    setup_stdout_utf8()

    # ========================================================
    # 1.读 wav
    # ========================================================
    # 要识别的 wav。权重包里自带 test_wavs/0.wav ~ 3.wav，换文件名即可换样本
    _ASR_DIR = os.path.join(
        _HERE, "weight",
        "sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05")
    WAV_PATH = os.path.join(_ASR_DIR, "test_wavs", "1.wav")
    # 每次喂给前端的样本数：160 = 10ms @16kHz，与 MicClient 的默认 block 相同，
    # 这样「文件路径」与「麦克风路径」的喂入节奏完全一致，两边结果可以互相对照
    BLOCK = 160
    # 读音频
    samples, sr = load_wav(WAV_PATH)
    duration = len(samples) / float(sr)

    # ========================================================
    #  2.加载三个onnx模型，初始化预处理和推理实例
    # ========================================================
    #  asr各个模型路径（与 test_mic_asr_mini.py 用的是同一套）
    ENCODER_ONNX = os.path.join(_ASR_DIR, "encoder.int8.onnx")
    DECODER_ONNX = os.path.join(_ASR_DIR, "decoder.onnx")
    JOINER_ONNX = os.path.join(_ASR_DIR, "joiner.int8.onnx")
    TOKENS_TXT = os.path.join(_ASR_DIR, "tokens.txt")
    # 2.1 加载onnx模型，返回三个session
    enc, dec_sess, joi = build_ort_sessions(ENCODER_ONNX, DECODER_ONNX, JOINER_ONNX)
    # 2.2 初始化asr推理器，组合三个session和词表的语言转文字工作流水线
    dec = AsrDecoder(enc, dec_sess, joi, TOKENS_TXT)
    # 2.3 初始化预处理类，将音频转换为特征，输入asr模型，n_mels 跟着模型走，避免两边维度对不上
    #     注意：这里保留默认 tail_pad_frames=100，收尾时解码要用；那 100 帧 0 不喂可视化（见第 5 节）
    front = AudioPreprocess(sample_rate=sr, n_mels=dec.n_mels)
    # 2.4 初始化语谱图：和 AsrDecoder 一样只接收特征，
    #     底部预留 SPEC_FOOTER_H 像素白条，识别文字由入口用 cv2.putText 画进去
    spec_view = SpectrogramImage(
        n_mels=front.n_mels,
        cols=SPEC_COLS,
        frame_shift=front.frame_step / float(sr),
        footer_h=SPEC_FOOTER_H,
        label=os.path.basename(WAV_PATH))

    # ========================================================
    # 3.打印模型信息、预处理参数、wav 参数
    # ========================================================
    info = dec.info()
    print("=" * 74)
    print("mini 离线 wav ASR —— 文件 -> 特征 -> 文字 + 语谱图")
    print("=" * 74)
    print("wav   : %s" % WAV_PATH)
    print("        %.2f 秒 / %d 样本 @ %d Hz" % (duration, len(samples), sr))
    print("权重  : %s" % _ASR_DIR)
    print("onnx  : encoder=%s  decoder=%s  joiner=%s" % (os.path.basename(ENCODER_ONNX), os.path.basename(DECODER_ONNX), os.path.basename(JOINER_ONNX)))
    print("执行器: %s" % " -> ".join(p.replace("ExecutionProvider", "") for p in info["providers"]))
    print("模型  : T=%d(输入帧数)  decode_chunk_len=%d(每轮前移)  ""context_size=%d  vocab=%d  状态张量=%d"
          % (info["T"], info["decode_chunk_len"], info["context_size"], info["vocab_size"], info["n_states"]))
    print("预处理端  : %d Hz  帧长 %d 点(%.0fms)  帧移 %d 点(%.0fms)  n_mels=%d"
          % (front.sample_rate, front.frame_size, front.frame_size / front.sample_rate * 1e3,  front.frame_step, front.frame_step / front.sample_rate * 1e3,  front.n_mels))
    print("词表  : %d 个  blank_id=%d (%s)"% (len(dec.id2token), dec.blank_id, dec.id2token.get(dec.blank_id, "?")))
    print()
    if sr != front.sample_rate:
        print("注意：wav 是 %d Hz，前端按 %d Hz 算，结果会不对。\n" % (sr, front.sample_rate))

    # ========================================================
    # 4.主循环：切块 -> 预处理 -> 推理 -> 后处理 -> 打印 -> 刷新语谱图
    # ========================================================
    # 4.1 开始语音转文字工作流
    shown = ""
    quit_now = False
    t_start = time.perf_counter()
    for i in range(0, len(samples), BLOCK):
        # 4.2 取一块（末尾那块可能不满）
        blk = samples[i:i + BLOCK]
        # 4.3 预处理
        feats = front.accept_waveform(blk)
        # 4.4 可视化返回伪彩语谱图
        cv_frame = spec_view.accept_features(feats)
        # 4.5 推理 + 后处理
        text = dec.accept_features(feats)
        # 4.6 打印：只在变化时刷新 stdout（窗口每帧都重画）
        if text != shown:
            shown = text
            sys.stdout.write("\r%s" % text.ljust(72))
            sys.stdout.flush()
        # 4.7 识别结果画到底部白条，再显示 + 处理按键（putText 只支持 ASCII）
        cv2.putText(cv_frame, text, (8, cv_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow(SPEC_WINDOW, cv_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            quit_now = True

    # ========================================================
    # 5.结束，收尾并统计结果
    # ========================================================
    # 5.1 收尾：让前端补出尾部帧、并在特征尾补 100 帧 0（log 域能量 1，不是静音值），
    #     使最后不满一个 chunk 的尾段也能被解码 —— 少了这一步，最后几个字会被整段丢掉。
    #     这一步的返回值尾部带着 100 帧 0，只喂解码器；喂可视化会多一块假亮尾
    text = dec.accept_features(front.input_finished())
    # 最后一屏按当前缓存重画（不喂补 0 特征），把完整识别结果写到底部
    cv_frame = spec_view.frame()
    cv2.putText(cv_frame, text, (8, cv_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow(SPEC_WINDOW, cv_frame)
    # 5.2 统计结果
    dur = time.perf_counter() - t_start
    print("\n")
    print("=" * 74)
    print("识别结果：%s" % text)
    print("=" * 74)
    print("音频 %.2fs   耗时 %.2fs   RTF %.3f（< 1 即快于实时）" % (duration, dur, dur / duration if duration else 0.0))
    print("喂入样本 %d   产出帧 %d" % (front.n_samples, front.n_frames))
    print("encoder 调用 %d 次 / 共 %d 帧   token %d 个"% (dec.n_chunks, dec.n_enc_frames, len(dec.token_ids)))
    # 5.3 最后一屏语谱图：按 q/Esc 退出就直接关窗，否则停住等按键
    if quit_now:
        cv2.destroyAllWindows()
    else:
        print("\n按任意键关闭语谱图窗口……")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

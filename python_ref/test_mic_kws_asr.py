# -*- coding: utf-8 -*-
"""实时麦克风语音识别 —— mini 版入口（关键词唤醒 + 语音识别 + 一个可视化窗口）。

============================================================
这份代码是干什么的
============================================================
把 src/ 下的流水线按顺序接起来，做一个「先说唤醒词、再说话」的实时识别
程序。不用 Qt：识别结果用 print 原地刷新，同时开一个 OpenCV 窗口，把同一份
特征画成滚动语谱图，当前模式与识别文字也画在窗口底部。

四条流水线 + 一个显示单元，各管一段，彼此不知道对方的存在：

    mic_client.py           MicClient           声卡 -> 裸音频块
    audio_preprocess.py     AudioPreprocess     波形块 -> LogFBank 特征
    kws_decode.py           KeywordSpotter      特征 -> 命中的关键词（唤醒）
    asr_decode.py           AsrDecoder          特征 -> 中文文本
    spectrogram_image.py    SpectrogramImage    特征 -> BGR 滚动语谱图画布

主循环有两种模式，默认从关键词唤醒开始：

    KWS --命中唤醒词--> ASR
    ASR --空闲超时 / 解码出结束语--> KWS

同一时刻只有一个解码器在吃特征：KWS 模式只喂 KeywordSpotter，ASR 模式只喂
AsrDecoder；切换时两边各自 reset()，前端实例始终连续，特征时间轴不断。

主循环干的事非常直白：

    取一块音频 -> 算特征 -> 按模式喂解码器 -> 画字 -> 显示，如此往复。
    accept_features() 直接返回整幅画布，main 里画完字就 imshow；
    同一份特征同时喂给「当前模式的解码器」和 SpectrogramImage。

============================================================
调用顺序（对照 main 里的编号）
============================================================
    1) 连接麦克风          mic.start()
    2) 建两组 onnx session build_ort_sessions()（路径写在文件顶部）
       装配 ASR 解码器     AsrDecoder(...)
       装配唤醒器          KeywordSpotter(...)
       装配语谱图          SpectrogramImage(...)
    3) 预处理              preprocessor.accept_waveform(samples)
    4) 模型推理 + 5) 后处理 KWS 模式：keyword_spotter.accept_features(feats)
                          ASR 模式：asr_decoder.accept_features(feats)
    6) 可视化              cv_frame = spec_view.accept_features(feats)
                          cv2.putText + cv2.imshow
    7) 打印结果            stdout 原地刷新 + 窗口底部状态条

结束（Ctrl+C 或窗口里按 q/Esc）后还要走一次收尾：取走麦克风里残余的音频。
如果此时在 ASR 模式，再让前端补完尾部帧、并在特征尾补 100 帧 0，把剩下的
一起解掉 —— 否则最后几个字会丢。那 100 帧 0 只喂解码器、不喂可视化。
KWS 模式不需要补帧：唤醒判定只看已经吐出来的输出帧。

用法：
    python -I test_mic_kws_asr.py

不收命令行参数 —— 运行配置全部硬编码在文件顶部的「运行配置」一节：
两组 onnx 与词表的路径、关键词文件、执行器顺序（GPU/CPU）、麦克风设备、
初始模式、ASR 空闲超时与结束语，都在那里改。模型怎么建、跑在哪块硬件上，
也由入口这一层决定；两个解码类只接收现成的 session，既不定位权重文件
也不碰 onnxruntime。
"""

import os
import sys
import time
from collections import deque

# -I 隔离模式下脚本目录不进 sys.path（也忽略 PYTHONPATH），
#   不显式插一次，下面的 from src.* 会直接 ImportError
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cv2
import numpy as np

from src.audio_preprocess import AudioPreprocess
from src.asr_decode import AsrDecoder
from src.cv_text import fit_text_tail
from src.kws_decode import KeywordSpotter
from src.mic_client import MicClient, list_devices
from src.spectrogram_image import SpectrogramImage
import onnxruntime as ort

# onnxruntime 推理用cpu或gpu
PROVIDERS = ["CPUExecutionProvider"]    #"CUDAExecutionProvider",

# ============================================================
# 运行配置：模式与切换规则
# ============================================================
MODE_KWS = "kws"
MODE_ASR = "asr"

# 初始模式：调试旧行为（纯 ASR）时改成 MODE_ASR
INITIAL_MODE = MODE_KWS

# ASR 模式里连续多久没有新 token 就切回 KWS（秒）
ASR_IDLE_TIMEOUT = 10.0

# 解出的文本（去掉尾部标点后）以这些词结尾，就立刻切回 KWS
END_PHRASES = ("再见", "拜拜", "结束", "退出", "没事了", "滚吧", "滚", "退下", "闭嘴", "退下吧")

# 切到 ASR 时回填最近多少帧特征（1 帧 = 10ms）。0 = 不回填；
# 回填的音频里包含唤醒词本身，ASR 文本开头可能出现唤醒词，属正常现象。
ASR_BACKFILL_FRAMES = 100

# KWS 观测开关：True 时把「trie 已走到关键词末端、但 avg_prob 没过 #threshold」
# 的候选打到 stderr，退出时再汇总。调 #threshold 只能靠这批「差一点」的分数，
# 命中时的 avg_prob 单独打印，不受此开关影响。
KWS_DEBUG = True

# 判断结束语前先剥掉的尾部空白与中英文标点
_END_PUNCT = " \t\r\n，。！？、,.!?;；:：\"'“”‘’"

# 麦克风设备：None = 系统默认输入设备；也可以写设备编号（如 1）或设备名
MIC_DEVICE = 3


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

    # 单线程 + 静音日志：这是实时语音场景，多个 chunk 之间本来就串行，
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


def normalize_device(device=MIC_DEVICE):
    """设备号既可以是编号（"1"）也可以是名字，能转成整数就转。"""
    if isinstance(device, str):
        try:
            return int(device)
        except ValueError:
            return device
    return device


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


def is_end_phrase(text):
    """识别文本（去掉尾部标点后）是否以某个结束语结尾。"""
    t = text.rstrip(_END_PUNCT)
    return any(t.endswith(p) for p in END_PHRASES)


def main():
    setup_stdout_utf8()

    # ========================================================
    # 1.连接麦克风
    # ========================================================
    # list_devices()        # 想看本机有哪些设备，把这一行取消注释
    device = normalize_device(MIC_DEVICE)

    # 初始化麦克风客户端：
    # 请求 sample_rate = 16kHz 单声道，PortAudio 以 16k 打开device的流，声卡原生不支持时由 PortAudio 内部重采样。
    # 不用管客户端内部块大小与回调节奏，默认每块 160 样本(10ms)回调一次，最多缓存 256 块(≈2.56s，满了淘汰最旧块)。
    # 主循环只需定期调 get_last_audio()，返回自上次调用以来累积所有缓存块拼成的一维 float32 波形(长度不固定，无新数据为 None)，
    mic = MicClient(sample_rate = 16000, device = device)
   
    # ========================================================
    #  2.加载ASR模型
    # ========================================================
    # 2.1 asr各个模型路径
    _ASR_DIR = os.path.join(
        _HERE, "weight",
        "sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05")
    ENCODER_ONNX = os.path.join(_ASR_DIR, "encoder.int8.onnx")
    DECODER_ONNX = os.path.join(_ASR_DIR, "decoder.onnx")
    JOINER_ONNX = os.path.join(_ASR_DIR, "joiner.int8.onnx")
    TOKENS_TXT = os.path.join(_ASR_DIR, "tokens.txt")
    # 2.2 加载onnx模型，返回三个session
    encoder_session, decoder_session, joiner_session = build_ort_sessions(ENCODER_ONNX, DECODER_ONNX, JOINER_ONNX)
    # 2.3 初始化asr推理器，组合三个session和词表的语言转文字工作流水线
    asr_decoder = AsrDecoder(encoder_session, decoder_session, joiner_session, TOKENS_TXT)

    # ========================================================
    #  3.加载KWS模型
    # ========================================================
    # 3.1 关键词唤醒模型路径（chunk-8：160ms 推进，唤醒延迟低于 chunk-16）
    _KWS_DIR = os.path.join( _HERE, "weight","sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20")
    KWS_ENCODER_ONNX = os.path.join(
        _KWS_DIR, "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx")
    KWS_DECODER_ONNX = os.path.join(
        _KWS_DIR, "decoder-epoch-13-avg-2-chunk-8-left-64.onnx")
    KWS_JOINER_ONNX = os.path.join(
        _KWS_DIR, "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx")
    KWS_TOKENS_TXT = os.path.join(_KWS_DIR, "tokens.txt")
    # 仓库自有的关键词文件：改唤醒词改这里，不动 weight/ 里的原始文件
    KWS_KEYWORDS_TXT = os.path.join(_HERE, "keywords", "kws_keywords.txt")
    # 3.2 加载onnx模型，返回三个session
    kws_enc, kws_dec, kws_joi = build_ort_sessions(KWS_ENCODER_ONNX, KWS_DECODER_ONNX, KWS_JOINER_ONNX)
    # 3.3 初始化关键词唤醒器：束搜索 + 关键词 trie，只吐命中的关键词
    # debug=True：记录被 ac_threshold 挡住的候选（调阈值用，见 KWS_DEBUG）
    keyword_spotter = KeywordSpotter(kws_enc, kws_dec, kws_joi, KWS_TOKENS_TXT, KWS_KEYWORDS_TXT,
                                     debug=KWS_DEBUG)

    # ========================================================
    #  4.加载预处理类和语谱图可视化类
    # ========================================================
    # 4.1 初始化预处理类，将音频转换为LogFBank特征，输入asr模型，n_mels 跟着模型走，避免两边维度对不上
    #     注意：这里保留默认 tail_pad_frames=100，收尾时解码要用；那 100 帧 0 不喂可视化（见第 5 节）
    preprocessor = AudioPreprocess(sample_rate=16000, n_mels=asr_decoder.n_mels)
    # KWS 与 ASR 共用同一份特征，n_mels 必须一致
    if keyword_spotter.n_mels != preprocessor.n_mels:
        raise RuntimeError("KWS 的 n_mels=%d 与 ASR 的 n_mels=%d 不一致"
                           % (keyword_spotter.n_mels, preprocessor.n_mels))
    # 4.2 初始化语谱图：和 AsrDecoder 一样只接收特征，底部预留 SPEC_FOOTER_H 像素白条，识别文字由入口用 cv2.putText 画进去
    spec_view = SpectrogramImage(
        n_mels=preprocessor.n_mels,
        cols=SPEC_COLS,
        frame_shift=preprocessor.frame_step / float(preprocessor.sample_rate),
        footer_h=SPEC_FOOTER_H,
        label="mic")

    # ========================================================
    # 5.打印模型信息、预处理参数、麦克风参数
    # ========================================================
    info = asr_decoder.info()
    kws_info = keyword_spotter.info()
    print("=" * 74)
    print("mini 流式 KWS + ASR —— 预处理 / 唤醒 / 解码 / 麦克风 + 语谱图")
    print("=" * 74)
    print("ASR 权重: %s" % _ASR_DIR)
    print("ASR onnx: encoder=%s  decoder=%s  joiner=%s" % (os.path.basename(ENCODER_ONNX), os.path.basename(DECODER_ONNX), os.path.basename(JOINER_ONNX)))
    print("KWS 权重: %s" % _KWS_DIR)
    print("KWS onnx: encoder=%s  decoder=%s  joiner=%s"
          % (os.path.basename(KWS_ENCODER_ONNX), os.path.basename(KWS_DECODER_ONNX), os.path.basename(KWS_JOINER_ONNX)))
    # 执行器：硬编码 CUDA 优先、CPU 兜底
    print("执行器: %s" % " -> ".join(p.replace("ExecutionProvider", "") for p in info["providers"]))
    print("ASR 模型: T=%d(输入帧数)  decode_chunk_len=%d(每轮前移)  ""context_size=%d  vocab=%d  状态张量=%d"
          % (info["T"], info["decode_chunk_len"], info["context_size"], info["vocab_size"], info["n_states"]))
    print("KWS 模型: T=%d  decode_chunk_len=%d  context_size=%d  vocab=%d  状态张量=%d  帧长=%.0fms"
          % (kws_info["T"], kws_info["decode_chunk_len"], kws_info["context_size"],
             kws_info["vocab_size"], kws_info["n_states"], kws_info["frame_shift_s"] * 1e3))
    print("预处理端  : %d Hz  帧长 %d 点(%.0fms)  帧移 %d 点(%.0fms)  n_mels=%d"
          % (preprocessor.sample_rate, preprocessor.frame_size, preprocessor.frame_size / preprocessor.sample_rate * 1e3,  preprocessor.frame_step, preprocessor.frame_step / preprocessor.sample_rate * 1e3,  preprocessor.n_mels))
    print("ASR 词表  : %d 个  blank_id=%d (%s)"% (len(asr_decoder.id2token), asr_decoder.blank_id, asr_decoder.id2token.get(asr_decoder.blank_id, "?")))
    print("唤醒词    : %d 条  %s" % (kws_info["n_keywords"],
                                     " / ".join(keyword_spotter.keywords)))
    print("切换规则  : 初始=%s  空闲超时=%.0fs  结束语=%s"
          % (INITIAL_MODE, ASR_IDLE_TIMEOUT, " / ".join(END_PHRASES)))
    print()

    # ========================================================
    # 6.主循环：取音频 -> 预处理 -> 推理 -> 后处理 -> 打印 -> 刷新语谱图
    # ========================================================
    # 6.1 打开麦克风
    mic.start()
    print("麦克风已开：device=%s  每块 %d 样本 (%.1f ms)  缓冲上限 %d 块"
            % (device if device is not None else "系统默认", mic.block, mic.block / mic.sample_rate * 1e3, mic.max_blocks))
    # 6.2 模式状态
    mode = INITIAL_MODE
    text = ""               # 当前 ASR 文本（KWS 模式下为空）
    shown = ""              # 控制台原地刷新用
    last_asr_text = ""      # 最近一次非空 ASR 文本，收尾时打印
    last_n_tokens = 0       # 上一次看到的 token 数，用来判断「有没有新字」
    last_activity = time.perf_counter()
    wake_count = 0          # 唤醒次数
    back_count = 0          # 切回 KWS 的次数
    # 两个解码器每次 reset() 都会把自带的计数器清零，所以这里另存一份累计值，
    # 每次切回 KWS 前把「刚结束的这一段会话」加进来，收尾时再加当前会话
    asr_calls_total = 0
    asr_frames_total = 0
    asr_tokens_total = 0
    kws_calls_total = 0
    kws_frames_total = 0
    kws_hits_total = 0
    # 最近 ASR_BACKFILL_FRAMES 帧特征；maxlen=0 时 append 自动丢弃
    recent = deque(maxlen=ASR_BACKFILL_FRAMES)
    t_start = time.perf_counter()
    print("=" * 74)
    print("识别结果(ctrl+C 或窗口里按 q/Esc 退出)：" )
    quit_now = False
    try:
        while not quit_now:
            # 6.3 取音频
            samples = mic.get_last_audio()
            if samples is not None:
                # 6.4 预处理
                feats = preprocessor.accept_waveform(samples)
                # 6.5 可视化返回伪彩语谱图
                cv_frame = spec_view.accept_features(feats)
                # 6.6 留一份特征尾巴，切到 ASR 时回填。
                #     按「帧」而不是按「块」存，deque 的 maxlen 才等于帧数
                for row in feats:
                    recent.append(row)

                if mode == MODE_KWS:
                    # ---- KWS 模式：只喂唤醒器 ----
                    hits = keyword_spotter.accept_features(feats)
                    status = "KWS | listening"
                    if hits:
                        wake_count += len(hits)
                        print("\r%s" % (" " * 72))
                        # avg_prob 是关键词各 token 的纯声学概率均值，
                        # 调 #threshold 时拿它当命中侧基准
                        print("[唤醒] %s  (avg_prob=%.3f)"
                              % (hits[0].keyword, hits[0].avg_prob))
                        # 切到 ASR：清空 ASR 状态，回填最近约 1 秒特征，
                        # 免得唤醒词末尾和命令开头之间被 chunk 边界吃掉
                        asr_decoder.reset()
                        text = ""
                        if ASR_BACKFILL_FRAMES > 0 and len(recent):
                            text = asr_decoder.accept_features(
                                np.stack(list(recent)))
                        last_n_tokens = len(asr_decoder.token_ids)
                        if text:
                            last_asr_text = text
                            shown = text
                            sys.stdout.write("\r%s" % text.ljust(72))
                            sys.stdout.flush()
                        else:
                            shown = ""
                        last_activity = time.perf_counter()
                        mode = MODE_ASR
                        status = "ASR | listening"
                else:
                    # ---- ASR 模式：只喂解码器 ----
                    text = asr_decoder.accept_features(feats)
                    n_tokens = len(asr_decoder.token_ids)
                    if n_tokens > last_n_tokens:
                        last_n_tokens = n_tokens
                        last_activity = time.perf_counter()
                    if text:
                        last_asr_text = text
                    # 打印：只在变化时刷新 stdout（窗口每帧都重画）
                    if text != shown:
                        shown = text
                        sys.stdout.write("\r%s" % text.ljust(72))
                        sys.stdout.flush()
                    status = "ASR | listening"

                    # 6.7 切回 KWS：结束语优先，其次空闲超时
                    ended = is_end_phrase(text)
                    idle = time.perf_counter() - last_activity
                    if ended or idle > ASR_IDLE_TIMEOUT:
                        back_count += 1
                        if ended:
                            print("\n[结束语] %s" % text.rstrip(_END_PUNCT))
                        else:
                            print("\n[超时 %.0fs] 切回关键词模式" % idle)
                        # 先把本段会话的计数加进累计值，再 reset
                        asr_calls_total += asr_decoder.n_chunks
                        asr_frames_total += asr_decoder.n_enc_frames
                        asr_tokens_total += len(asr_decoder.token_ids)
                        kws_calls_total += keyword_spotter.n_chunks
                        kws_frames_total += keyword_spotter.n_enc_frames
                        kws_hits_total += keyword_spotter.n_hits
                        asr_decoder.reset()
                        keyword_spotter.reset()
                        mode = MODE_KWS
                        text = ""
                        shown = ""
                        last_n_tokens = 0
                        last_activity = time.perf_counter()
                        status = "KWS | listening"

                # 6.8 底部状态条：第一行模式，第二行 ASR 文本
                #     （putText 只支持 ASCII，中文会画成方框，以控制台为准）
                #     文本先按画布宽度裁掉开头的旧字，保证刚说的内容始终可见
                h = cv_frame.shape[0]
                cv2.putText(cv_frame, status, (8, h - 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0),
                            1, cv2.LINE_AA)
                if mode == MODE_ASR and text:
                    draw_text = fit_text_tail(text, cv_frame.shape[1] - 16)
                    cv2.putText(cv_frame, draw_text, (8, h - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                                1, cv2.LINE_AA)
                cv2.imshow(SPEC_WINDOW, cv_frame)
            else:
                time.sleep(0.002)      # 还没攒到新数据，让出 CPU 别空转
            # 6.9 处理按键（没新数据时也要泵一次事件，窗口才不会假死）
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                quit_now = True
    except KeyboardInterrupt:
        pass

    # ========================================================
    # 7.结束，关闭麦克风，统计结果
    # ========================================================
    # 7.1 关闭麦克风
    mic.stop()
    # 7.2 收尾：关流后缓冲里可能还压着几块，先取走喂给当前模式的解码器
    leftover = mic.get_last_audio()
    if leftover is not None:
        feats = preprocessor.accept_waveform(leftover)
        spec_view.accept_features(feats)
        if mode == MODE_ASR:
            text = asr_decoder.accept_features(feats)
        else:
            keyword_spotter.accept_features(feats)
    if mode == MODE_ASR:
        # 补尾帧 + 100 帧 0（log 域能量 1，不是静音值），使最后不满一个
        # chunk 的尾段也能被解码 —— 少了这一步，最后几个字会被整段丢掉。
        # 只喂解码器；喂给可视化的话，语谱图末尾会多出一块比静音还亮的假色块
        text = asr_decoder.accept_features(preprocessor.input_finished())
        if text:
            last_asr_text = text
    else:
        # KWS 模式不补 0：那 100 帧是 log 域的 0，不是静音
        text = last_asr_text
    # 最后一屏按当前缓存重画（不喂补 0 特征），把完整识别结果写到底部
    cv_frame = spec_view.frame()
    cv2.putText(cv_frame, text, (8, cv_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow(SPEC_WINDOW, cv_frame)
    # 7.3 正常退出，统计结果
    dur = time.perf_counter() - t_start
    print("\n")
    print("=" * 74)
    print("识别结果：%s" % text)
    print("=" * 74)
    print("时长 %.1fs   喂入样本 %d   产出帧 %d" % (dur, preprocessor.n_samples, preprocessor.n_frames))
    # 累计值 = 已经结束的各段会话 + 当前这一段
    print("ASR：encoder 调用 %d 次 / 共 %d 帧   token %d 个"
          % (asr_calls_total + asr_decoder.n_chunks,
             asr_frames_total + asr_decoder.n_enc_frames,
             asr_tokens_total + len(asr_decoder.token_ids)))
    print("KWS：encoder 调用 %d 次 / 共 %d 帧   触发 %d 次"
          % (kws_calls_total + keyword_spotter.n_chunks,
             kws_frames_total + keyword_spotter.n_enc_frames,
             kws_hits_total + keyword_spotter.n_hits))
    print("唤醒 %d 次   切回关键词模式 %d 次" % (wake_count, back_count))
    # KWS 调参参考：被阈值挡住的候选。降序排 —— 排在最前的是「差一点」的，
    # 它们离阈值越近，说明 #threshold 越该往下调。
    if KWS_DEBUG and keyword_spotter.debug_events:
        evs = sorted(keyword_spotter.debug_events, key=lambda e: -e["avg_prob"])
        print("\n[KWS debug] 被阈值挡住 %d 次（avg_prob 高 → 低）："
              % len(evs))
        for ev in evs[:20]:
            print("  %-8s avg_prob=%.3f  threshold=%.3f  level=%d"
                  % (ev["keyword"], ev["avg_prob"], ev["threshold"],
                     ev["level"]))
        if len(evs) > 20:
            print("  ... 其余 %d 条略" % (len(evs) - 20))
    print("丢块 %d   峰值幅度 %.3e" % (mic.dropped, mic.peak))
    # 7.4 try异常退出提示
    if mic.dropped:
        print("提示：出现丢块说明主循环跟不上采集速度，可调大 MicClient 的 "
              "block 参数（在 mic_client.py 里）。")
    if mic.peak == 0.0:
        print("提示：峰值幅度为 0，麦克风可能没采到声音，"
              "可改顶部 MIC_DEVICE 指定别的设备；"
              "取消第 1 节里 list_devices() 那行的注释即可列出本机设备。")
    # 7.5 最后一屏语谱图：按 q/Esc 退出就直接关窗，否则停住等按键
    if quit_now:
        cv2.destroyAllWindows()
    else:
        print("\n按任意键关闭语谱图窗口……")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

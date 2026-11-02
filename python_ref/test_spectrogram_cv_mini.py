# -*- coding: utf-8 -*-
"""实时滚动语谱图 —— mini 版入口（OpenCV 窗口）。

============================================================
这份代码是干什么的
============================================================
把 wav 文件按 10ms 一块、**按真实时间节奏**喂进 src/ 的预处理流水线，
再用 OpenCV 窗口把 LogFBank 特征画成滚动的语谱图：

    wav --切块--> AudioPreprocess --特征--> SpectrogramImage --BGR 画布--> 窗口

前端与可视化各管一段、互不知道对方：AudioPreprocess 只出特征，
SpectrogramImage 只收特征、出画布。和 test_wav_asr_mini.py 的区别只在
最后一步：那边把特征喂给解码器出文字，这边把特征画成图。喂块大小
（160 样本 = 10ms）与预处理类完全相同。

============================================================
「实时」是模拟出来的
============================================================
wav 其实已经整段读进内存，但主循环按 1x 节奏 sleep，每 10ms 才喂一块，
所以看到的是「边放边出图」的滚动效果，和麦克风实时输入的画面节奏一致。
把顶部 FEED_REALTIME 改成 False，就会尽快喂完、画面直接到结尾，
想快速检查一段音频的完整语谱图时用这个。

============================================================
画面怎么摆
============================================================
    横轴：时间，左旧右新，最新一列贴在最右端
    纵轴：Mel 通道，低频在**下**、高频在上（Mel 轴非线性，故不标 Hz 刻度）
    亮度：对数能量；配色 viridis
    顶部：白色信息条，显示当前时间 / 窗口 / 色阶（ASCII，cv2 字体限制）
    底部：本入口没有识别结果，所以不预留文字条

色阶（lo/hi）在启动后前 CALIBRATE_FRAMES 帧上标定一次，之后**锁死**。
不锁的话，安静段和说话段的亮度会互相挤压，画面看起来一直在「呼吸」。
整段几乎无声、动态范围太小时，标定会退回库里的默认色阶。

============================================================
按键
============================================================
    q / Esc   退出
    s         把当前画面存成 spectrogram_mini.png（当前工作目录）

用法（在项目根目录执行）：
    python -I test_spectrogram_cv_mini.py

不收命令行参数 —— wav 路径、画布列数、窗口尺寸、colormap 都硬编码在
下面的「运行配置」里，改哪里一目了然，和另外两个 mini 入口保持一致。
"""

import os
import sys
import time

import cv2

# -I 隔离模式下脚本目录不进 sys.path（也忽略 PYTHONPATH）。
# 本脚本在项目根，而实现库在 src/ 下，所以要把项目根插进去，
# 下面的 from src.* 才找得到。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.audio_preprocess import AudioPreprocess
from src.spectrogram_image import DEFAULT_LEVEL, SpectrogramImage

# ========================================================
# 运行配置
# ========================================================
# 要可视化的 wav。权重包里自带 test_wavs/0.wav ~ 3.wav，换文件名即可换样本
WAV_PATH = os.path.join(
    _HERE, "weight",
    "sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05",
    "test_wavs", "0.wav")

N_MELS = 80            # Mel 通道数，也是图像行数；纯可视化用 80 即可
BLOCK = 160            # 每次喂 10ms（160 样本 @16kHz），与 MicClient 默认块长一致
COLS = 400             # 滚动窗口宽度（列）。400 列 x 10ms = 4 秒
FEED_REALTIME = True   # True = 按 1x 节奏喂；False = 尽快喂完，画面直接到结尾
CALIBRATE_FRAMES = 50  # 用前 0.5 秒的特征标定色阶，之后锁死
PLOT_SIZE = (1200, 300)     # 语谱图本身的显示尺寸（宽, 高），80 行太矮需要放大
HEADER_H = 52               # 顶部信息条高度：文字放这里，不遮挡语谱图
# viridis 伪彩 + 白底黑字信息条；OpenCV 的 COLORMAP_VIRIDIS 就是那张 matplotlib 色表。
COLORMAP = cv2.COLORMAP_VIRIDIS
SAVE_PATH = "spectrogram_mini.png"
# 窗口标题只能用 ASCII：部分 OpenCV 构建对非 ASCII 标题处理不一致
WINDOW = "mini spectrogram"


# ========================================================
# 工具函数
# ========================================================
def load_wav(path):
    """读 wav，返回 (float32 单声道样本, 采样率)。

    多声道只取第一声道（识别模型与前端都按单声道设计）。
    这里**不做重采样** —— 采样率不对时宁可明确提示，也不要悄悄喂错数据。
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


# ========================================================
# 主流程
# ========================================================
def main():
    setup_stdout_utf8()

    # ========================================================
    # 1.读 wav
    # ========================================================
    samples, sr = load_wav(WAV_PATH)
    duration = samples.size / float(sr)

    # ========================================================
    # 2.装配前端与可视化（各管一段，只靠特征相连）
    # ========================================================
    # 采样率用 wav 的实际值：这张图是给眼睛看的，按真实采样率建 Mel 滤波器
    # 才对得上音频的频段。喂模型仍要求 16kHz，不对就先重采样。
    # tail_pad_frames=0：可视化不补尾部那 100 帧 0，否则图上多一块假亮尾
    front = AudioPreprocess(sample_rate=sr,
                            n_mels=N_MELS,
                            tail_pad_frames=0)
    view = SpectrogramImage(n_mels=N_MELS,
                            cols=COLS,
                            frame_shift=front.frame_step / float(sr),
                            level=DEFAULT_LEVEL,
                            calibrate_frames=CALIBRATE_FRAMES,
                            plot_size=PLOT_SIZE,
                            header_h=HEADER_H,
                            colormap=COLORMAP,
                            label=os.path.basename(WAV_PATH))
    frame_shift = front.frame_step / float(sr)   # 一帧对应的时间（秒）

    # ========================================================
    # 3.打印参数
    # ========================================================
    print("=" * 74)
    print("mini 实时语谱图 —— wav -> LogFBank -> OpenCV 窗口")
    print("=" * 74)
    print("wav   : %s" % WAV_PATH)
    print("        %.2f 秒 / %d 样本 @ %d Hz" % (duration, samples.size, sr))
    print("前端  : 帧长 %d 点(%.0fms)  帧移 %d 点(%.0fms)  n_mels=%d"
          % (front.frame_size,
             front.frame_size / float(sr) * 1e3,
             front.frame_step,
             frame_shift * 1e3,
             front.n_mels))
    print("画布  : %d 列 x %d 行 = %.1f 秒窗口"
          % (COLS, N_MELS, COLS * frame_shift))
    print("按键  : q / Esc 退出   s 保存当前画面")
    print("=" * 74)
    if sr != 16000:
        print("注意：wav 是 %d Hz。本图按实际采样率画，仅用于观察；"
              "喂 ASR 模型前要先重采样到 16000 Hz。\n" % sr)

    # ========================================================
    # 4.主循环：1x 节奏喂块 -> 预处理 -> 滚动出图 -> 显示
    # ========================================================
    t0 = time.perf_counter()   # 1x 节奏的时间基准
    fed = 0                    # 已喂入的样本数
    quit_now = False
    cv_frame = None

    while fed < samples.size and not quit_now:
        # 4.1 1x 节奏：等到"这一块音频本该到达"的时刻。
        #     每块都重新按绝对时间算，偶尔睡过头也不会越拖越慢
        if FEED_REALTIME:
            wait = t0 + fed / float(sr) - time.perf_counter()
            if wait > 0:
                time.sleep(wait)

        # 4.2 喂一块（末尾那块可能不满）：前端出特征，可视化直接返回画布
        blk = samples[fed:fed + BLOCK]
        fed += blk.size
        cv_frame = view.accept_features(front.accept_waveform(blk))

        # 4.3 显示 + 处理按键
        cv2.imshow(WINDOW, cv_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):          # Esc / q
            quit_now = True
        elif key == ord("s"):              # s：存图
            cv2.imwrite(SAVE_PATH, cv_frame)
            print("已保存 %s" % SAVE_PATH)

    # ========================================================
    # 5.收尾：补完尾部帧，把最后一屏停住
    # ========================================================
    # tail_pad_frames=0，这里只把最后不足一帧的样本补完，不会多出假数据
    cv_frame = view.accept_features(front.input_finished())
    cv2.imshow(WINDOW, cv_frame)
    print("\n结束：喂入 %d 样本，产出 %d 帧（%.2f 秒）"
          % (fed, view.n_frames, view.n_frames * frame_shift))

    if not quit_now:
        print("按任意键关闭窗口……")
        cv2.waitKey(0)      # 停住最后一屏，方便慢慢看
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

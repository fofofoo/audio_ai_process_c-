# -*- coding: utf-8 -*-
"""语谱图：LogFBank 特征 -> 可直接 imshow 的 BGR 画布（mini 版可视化单元）。

============================================================
这份代码是干什么的
============================================================
src/ 里每个单元只干一件事，彼此靠数据接口相连：

    audio_preprocess.py   AudioPreprocess   波形块           -> (帧数, n_mels) 特征
    asr_decode.py         AsrDecoder        特征             -> 文本
    本模块                 SpectrogramImage  特征             -> BGR 画布（viridis 语谱图）

和 AsrDecoder 只接收特征、不碰麦克风是同一个边界：本模块**只接收特征**，
不 import audio_preprocess，也不碰波形。谁产生特征、怎么切块，由入口脚本
决定（见 test_spectrogram_cv_mini.py / test_mic_asr_mini.py）。

主流程只有三步，不藏节流、不藏窗口管理：

    cv_frame = spec_view.accept_features(feats)   # 收一块特征，返回整幅画布
    cv2.putText(cv_frame, text, ...)              # 要写文字就自己写
    cv2.imshow("spectrogram", cv_frame)           # 显示由入口自己决定

依赖：只用 numpy + OpenCV。本模块不负责写文字，底部白条（footer_h）只是
给调用方留一块背景，写什么、怎么写在入口里用 cv2.putText 决定。
三个类（AudioPreprocess / AsrDecoder / 本模块）互不引用，只靠数据接口相连，
保持每个文件都能独立读懂、独立移植。

============================================================
一、坐标与朝向（最容易画反的地方）
============================================================
特征矩阵的形状是 (时间, Mel)，而图像的形状是 (行, 列) = (y, x)。
要做两次变换：

    1. 转置：    (帧数, n_mels) -> (n_mels, 帧数)，行号就是 Mel 通道号；
    2. 上下翻转：让 Mel 0（最低频）落在最后一行，也就是画面**底部**。

OpenCV / 常见图像库的 row 0 在顶部。少了第 2 步，低频会跑到屏幕上方，
整张图看起来「高音在下」，和所有语谱图的惯例相反 —— 而且不会报错，
只会让人对着一张上下颠倒的图分析半天。

注意纵轴是 Mel 通道序号，**不是线性 Hz**：Mel 滤波器低频密、高频疏，
第 0 个通道和靠近 8kHz 的通道带宽差很多。所以本模块不画 Hz 刻度 ——
要标频率得先做 Mel 反变换，很容易给出「看起来精确、实际有偏」的刻度。

============================================================
二、色阶：对数能量怎么变成 0~255
============================================================
特征值本身是 ln(能量)，静音帧停在地板 ln(1.19e-7) = -15.94。
映射是一次线性拉伸 + 截断：

    像素 = clip((值 - lo) / (hi - lo) * 255, 0, 255)

实时滚动图必须**提前传固定 lo/hi**，或者在启动时标定一次后锁定。
如果每帧都按当前窗口重新算，安静段和说话段会互相挤压亮度，
画面会一直「呼吸」。

本类的做法：构造时传一个兜底色阶，前 calibrate_frames 帧（默认 50 =
0.5 秒）用 estimate_levels() 按实际数据标定一次，之后**锁死**；
calibrate_frames=0 表示完全不标定，一直用传入的固定值。

单位说明：lo/hi 是 ln(能量)，不是 dB；这里只说「对数能量」，
不要和 10·log10 的 dB 混用。

============================================================
三、可视化时尾部不要补 100 帧（调用方注意）
============================================================
AudioPreprocess.input_finished() 默认在尾部补 100 帧 0（对数域能量 1），
这是给解码用的：不补的话最后不满一个 chunk 的尾段会被整段丢掉。
但补出来的 0 在图上是一块**比静音还亮**的色块（静音地板是 -15.94，
0 比它高），纯属视觉噪声。本模块不建前端，管不到这一步 ——
只能在这里记一笔：可视化路径构造 AudioPreprocess 时要传
tail_pad_frames=0，解码路径仍然用默认 100，两边互不影响。

============================================================
四、滚动缓存与画布组成
============================================================
本类自己维护一个特征缓存 _feat_buf（形状 (帧数, n_mels)），
每来一块特征就拼接上去，只保留最近 cols 帧：

    cols=400、帧移 10ms  ->  画布固定 400 列 = 4 秒滚动窗口；
                            新帧永远贴右端，还没到的时间在左边补黑。

每帧画布 = 顶部信息条（白色）+ 语谱图 + 可选底部白条。
信息条和底部白条都只提供背景；写什么由入口用 cv2.putText 决定。

注意 cv2.putText 只能画 ASCII：中文会变成一串问号，而且不报错。
所以中文识别结果仍以终端输出为准，窗口里的文字只作辅助。
"""

import cv2
import numpy as np

# 默认色阶（对数能量域）。下限取 kaldi 的静音地板 ln(float32 eps)：
# 这样「没声音」就是最暗，不会因为下限拍脑袋取错而整片发灰。
# 本模块不 import 前端，这里直接用 numpy 算同一个数（≈ -15.94）。
# 上限是经验值，只作为「还没标定 / 标定失败」时的兜底。
DEFAULT_LEVEL_LO = float(np.log(np.finfo(np.float32).eps))
DEFAULT_LEVEL_HI = 8.0
DEFAULT_LEVEL = (DEFAULT_LEVEL_LO, DEFAULT_LEVEL_HI)


def estimate_levels(feats, p_lo=5.0, p_hi=99.5, min_range=6.0,
                    fallback=DEFAULT_LEVEL):
    """用一组特征的百分位估色阶 (lo, hi)。

    参数:
        feats     —— (帧数, n_mels) 特征矩阵
        p_lo/p_hi —— 取百分位的位置。5 / 99.5 能避开个别尖峰把上限拉飞
        min_range —— 动态范围低于这个值就认为「几乎没声音」，退回 fallback。
                     对着一小段静音强行拉伸，只会把数值噪声放大成一片雪花
        fallback  —— 标定失败时用的固定色阶

    返回:
        (lo, hi)，单位为 ln(能量)
    """
    feats = np.asarray(feats, dtype=np.float32)
    if feats.size == 0:
        return tuple(float(v) for v in fallback)

    lo, hi = np.percentile(feats, [p_lo, p_hi])
    lo, hi = float(lo), float(hi)
    if hi - lo < min_range:
        return tuple(float(v) for v in fallback)
    return lo, hi


def features_to_image(feats, level=None):
    """特征矩阵 -> 2D 灰度图。

    参数:
        feats —— (帧数, n_mels) 的 LogFBank 特征
        level —— (lo, hi) 色阶；None 表示用这批特征自己的百分位（静态图适用）

    返回:
        uint8 数组，形状 (n_mels, 帧数)：行 = Mel 通道（低频在下），
        列 = 时间（左旧右新）。0 = 最暗，255 = 最亮。
    """
    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim != 2:
        raise ValueError("feats 必须是二维 (帧数, n_mels)，实际形状 %r"
                         % (feats.shape,))

    n_frames, n_mels = feats.shape
    if n_frames == 0:
        return np.zeros((n_mels, 0), dtype=np.uint8)

    lo, hi = level if level is not None else estimate_levels(feats)
    if hi <= lo:
        # 防御：调用方传了反的或相等的色阶时，至少别让除法炸掉
        hi = lo + 1.0

    # 线性拉伸到 0~255 再截断；截断而不是回绕，超出的部分保持黑白两端
    x = (feats - float(lo)) * (255.0 / (float(hi) - float(lo)))
    np.clip(x, 0.0, 255.0, out=x)

    # .T：时间 -> 列；[::-1]：低频 -> 底部。
    # 转置 + 反向切片得到的是负步长视图，OpenCV 可能不认，先落成连续内存。
    img = x.astype(np.uint8).T[::-1]
    return np.ascontiguousarray(img)


class SpectrogramImage:
    """特征 -> 可直接 imshow 的 BGR 语谱图画布（整段或滚动）。

    只接收特征，不碰波形，也不自己建 AudioPreprocess —— 和 AsrDecoder
    只接收特征、不碰麦克风是同一个边界。缓存由本类自己维护。

    典型用法（前端是入口脚本自己装的）::

        front = AudioPreprocess(tail_pad_frames=0)   # 可视化路径不补尾帧
        view = SpectrogramImage(n_mels=80, cols=400, footer_h=56, label="0.wav")

        cv_frame = view.accept_features(front.accept_waveform(blk))
        cv2.putText(cv_frame, text, (8, cv_frame.shape[0] - 20), ...)
        cv2.imshow("spectrogram", cv_frame)
        cv2.waitKey(1)

    收尾时调用 view.frame() 渲染当前缓存，而不要把 input_finished()
    的补 0 特征喂进来（见模块开头第三节）。
    """

    def __init__(self, n_mels=80, cols=400, frame_shift=0.01,
                 level=DEFAULT_LEVEL, calibrate_frames=50,
                 plot_size=(1200, 300), header_h=52, footer_h=0,
                 colormap=cv2.COLORMAP_VIRIDIS, label=""):
        """
        参数:
            n_mels         —— Mel 通道数，也是灰度图的行数
            cols           —— 滚动窗口宽度（帧），同时决定画布固定宽度
            frame_shift    —— 一帧对应的时间（秒），只用于信息条上的时间
            level          —— 兜底色阶 (lo, hi)；标定之前用它
            calibrate_frames —— 收到多少帧后按实际数据标定一次并锁死；
                              0 = 不标定，一直用 level
            plot_size      —— 语谱图的显示尺寸（宽, 高）；n_mels 行太矮，需要放大
            header_h       —— 顶部白色信息条高度（ASCII 文字）
            footer_h       —— 底部白色条高度，留给调用方写文字；0 = 不预留
            colormap       —— OpenCV colormap；默认 viridis
            label          —— 信息条第一行；非 ASCII 字符会被替换成 ?
        """
        self.n_mels = int(n_mels)
        self.cols = max(1, int(cols))
        self.frame_shift = float(frame_shift)
        self.plot_size = (int(plot_size[0]), int(plot_size[1]))
        self.header_h = int(header_h)
        self.footer_h = int(footer_h)
        self.colormap = colormap
        self.label = ("" if label is None else str(label))
        self.label = self.label.encode("ascii", "replace").decode("ascii")

        self.calibrate_frames = int(calibrate_frames)
        self._level0 = (None if level is None
                        else (float(level[0]), float(level[1])))
        self.level = self._level0
        # 传 0 表示不标定，构造完就直接「已标定」
        self._calibrated = self.calibrate_frames <= 0

        # 内部特征缓存：_feat_buf 只保存窗口内的帧；
        # _n_total 是累计接收的帧数（不受滚动裁剪影响）
        self._feat_buf = np.zeros((0, self.n_mels), dtype=np.float32)
        self._n_total = 0

    # ========================================================
    # 对外接口
    # ========================================================
    def accept_features(self, feats):
        """喂一块特征，返回**整幅 BGR 画布**（可直接 cv2.imshow）。

        参数:
            feats —— float32 特征矩阵 (n, n_mels)，可能是 0 帧

        返回:
            uint8 数组 (header_h + plot_h + footer_h, plot_w, 3)，BGR。
            每喂一块就重画一次；调用方拿到后可直接画字、显示或存图。
        """
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim != 2 or feats.shape[1] != self.n_mels:
            raise ValueError("feats 形状应为 (n, %d)，实际 %r"
                             % (self.n_mels, feats.shape))

        n_new = int(feats.shape[0])
        if n_new:
            if self._feat_buf.shape[0]:
                self._feat_buf = np.concatenate((self._feat_buf, feats))
            else:
                self._feat_buf = feats
            self._n_total += n_new

            if self._feat_buf.shape[0] > self.cols:
                # 只留最近 cols 帧，旧帧立刻可以回收
                self._feat_buf = np.ascontiguousarray(
                    self._feat_buf[-self.cols:])
        return self.frame()

    def frame(self):
        """按当前缓存渲染一帧 BGR 画布。

        收尾时想显示最后一屏就调用它 —— 不要为了出图去喂
        input_finished() 的补 0 特征。
        """
        self._calibrate()

        gray = features_to_image(self._feat_buf, level=self.level)
        if gray.shape[1] < self.cols:
            # 右对齐：新帧永远贴右端，左边空出来的是"还没到的时间"
            pad = np.zeros((self.n_mels, self.cols - gray.shape[1]),
                           dtype=np.uint8)
            gray = np.hstack((pad, gray))

        # 80 行直接显示太矮，先按显示尺寸放大；再上 viridis 伪彩
        gray = cv2.resize(gray, self.plot_size, interpolation=cv2.INTER_LINEAR)
        plot = cv2.applyColorMap(gray, self.colormap)
        cv2.rectangle(plot, (0, 0),
                      (self.plot_size[0] - 1, self.plot_size[1] - 1),
                      (0, 0, 0), 1)

        parts = [self._header(), plot]
        if self.footer_h > 0:
            footer = np.full((self.footer_h, self.plot_size[0], 3),
                             255, dtype=np.uint8)
            cv2.line(footer, (0, 0),
                     (self.plot_size[0] - 1, 0), (0, 0, 0), 1)
            parts.append(footer)
        return np.vstack(parts)

    def reset(self):
        """清空特征缓存，色阶回到构造时的值。"""
        self._feat_buf = np.zeros((0, self.n_mels), dtype=np.float32)
        self._n_total = 0
        self.level = self._level0
        self._calibrated = self.calibrate_frames <= 0

    def features(self):
        """当前缓存内的特征矩阵 (帧数, n_mels)，float32。

        滚动模式下只含最近 cols 帧。返回的是内部数组本身，
        调用方只读，不要原地修改。
        """
        return self._feat_buf

    # ========================================================
    # 内部
    # ========================================================
    def _calibrate(self):
        """攒够 calibrate_frames 帧后标定一次色阶，之后不再改。"""
        if self._calibrated or self._n_total < self.calibrate_frames:
            return
        fallback = self.level if self.level is not None else DEFAULT_LEVEL
        self.level = estimate_levels(self._feat_buf, fallback=fallback)
        self._calibrated = True

    def _header(self):
        """顶部白色信息条：两行 ASCII 文字 + 底部黑分隔线。"""
        w = self.plot_size[0]
        h = self.header_h
        if h <= 0:
            return np.zeros((0, w, 3), dtype=np.uint8)

        header = np.full((h, w, 3), 255, dtype=np.uint8)
        if self.level is None:
            level_txt = "auto"
        else:
            level_txt = "[%.2f, %.2f]" % self.level
        line1 = self.label or "spectrogram"
        line2 = ("t=%5.2fs  win=%.2fs  n_mels=%d  level=%s"
                 % (self._n_total * self.frame_shift,
                    self.cols * self.frame_shift,
                    self.n_mels, level_txt))

        cv2.putText(header, line1, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
        if h >= 48:
            cv2.putText(header, line2, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)
        cv2.line(header, (0, h - 1), (w - 1, h - 1), (0, 0, 0), 1)
        return header

    # ========================================================
    # 状态
    # ========================================================
    @property
    def n_frames(self):
        """累计接收的帧数（不受滚动裁剪影响）。"""
        return self._n_total

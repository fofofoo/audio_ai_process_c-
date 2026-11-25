# -*- coding: utf-8 -*-
"""麦克风采集：输出裸音频流（mini 版第三条流水线）。
打开声卡，把麦克风的音频一块一块收进缓冲，等主循环主动来取。
    PortAudio 采集线程 ──callback()──> deque 缓冲 ── get_last_audio()拼接一维波形 ──> 主循环获取最近的音频数据（长度不固定）
"""

import collections

import numpy as np
import sounddevice as sd


class MicClient:
    """麦克风裸流客户端。
    用法::

        mic = MicClient()
        mic.start()
        while 想继续:
            samples = mic.get_last_audio()    # 没有新数据时返回 None
            if samples is None:
                continue
            ...                               # 拿去做前端
        mic.stop()
    """

    def __init__(self, sample_rate=16000, channels=1, block=160, device=None, max_blocks=256):
        """
        参数:
            sample_rate —— 采样率（Hz），模型要求 16000
            channels    —— 采集声道数，1 = 单声道
            block       —— 每次回调交付的样本数。160 = 10ms @16kHz = 16000 Hz * 0.01 s = 160
            device      —— 设备编号或名字；None = 系统默认输入设备
            max_blocks  —— 缓冲最多存几块；满了淘汰最旧的（约 2.56s @256 块），一块block有160个样本
        """
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.block = int(block)
        self.device = device
        self.max_blocks = int(max_blocks)

        # 采集线程写右端、主循环读左端，无需加锁
        # deque 是 collections 里的双端队列，支持从两端以 O(1) 增删：append / appendleft / pop / popleft，maxlen 满时 append 淘汰最左
        self._block_buf = collections.deque(maxlen=self.max_blocks)
        self._stream = None      # PortAudio 输入流句柄；start() 之后才有值
        self._dropped = 0        # 因缓冲满被淘汰的块数
        self._peak = 0.0         # 至今为止的最大样本幅度
        self._n_samples = 0      # 已被取走的累计样本数

    # ========================================================
    # 采集线程：唯一的入口就是这个回调，只做 copy + 入队
    # sd.InputStream 的触发方式：start() 之后，PortAudio 自己的实时采集
    # 线程每攒够 block 个样本（160 帧 = 10 ms @16 kHz）就调用一次，
    # 一直循环到 stop()/close()。不是本类起的 Python 线程，主线程不参与。
    # ========================================================
    def _callback(self, indata, frames, time_info, status):
        if status:
            # 溢出/欠载之类的状态提示，不致命，报一声方便排查
            print("[音频状态] %s" % status)

        # 必须 copy：indata 的底层缓冲会被 PortAudio 立刻复用
        # indata 形状是 (frames, channels) = (160, 1)，160 帧即 10 ms
        blk = indata[:, 0].copy()

        # deque 满了的话，接下来这次 append 会淘汰最旧的块 —— 记账一下
        if len(self._block_buf) == self._block_buf.maxlen:
            self._dropped += 1

        # 把一帧音频写入缓存
        self._block_buf.append(blk)

        # 一帧内的峰值幅度：只看不写临界，偶尔的竞争无伤大雅，就不加锁了
        blk_max = float(np.abs(blk).max())
        if blk_max > self._peak:
            self._peak = blk_max

    # ========================================================
    # 对外接口
    # ========================================================
    def start(self):
        """打开麦克风，采集线程开始往里送数据。"""
        # 初始化 sounddevice 输入流：从 device 设备按 sample_rate、channels
        # 采集音频，每满 block 个样本（160 帧 = 10 ms）回调一次 _callback。
        # dtype 固定 float32（范围约 [-1, 1]），与模型输入口径一致；
        # device=None 表示系统默认输入设备。
        # 这里只是构造，start() 之后 PortAudio 才真正开始采集和回调。
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.block,
            device=self.device,
            callback=self._callback)

        # 开始采集和回调音频数据
        self._stream.start()

    # 返回最近积累的全部一维音频波形数据
    def get_last_audio(self):
        """取走自上次调用以来累积的全部样本，并将其拼接成连续的一维波形输出。

        返回:
            一维 float32 波形；缓冲为空时返回 None
        """
        # 1. 收集本次取出的所有音频块
        chunks = []
        # 2. 从队头逐块取出，直到队列为空；空队列 popleft 会抛 IndexError
        while True:
            try:
                chunks.append(self._block_buf.popleft())
            except IndexError:
                break
        # 3. 一块都没有：自上次调用以来还没有新数据
        if not chunks:
            return None
        # 4. 记账：累计已被取走的样本数
        self._n_samples += sum(c.size for c in chunks)
        # 5. 只有一块时直接返回原数组，省掉一次 concatenate 拷贝；
        #    多块时拼成一个连续的一维数组再返回，保证波形连续不丢样本
        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate(chunks)

    def stop(self):
        """关闭麦克风。之后仍可调 get_last_audio() 取走残余样本。"""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # ========================================================
    # 状态（给入口脚本打印 / 判断用）
    # ========================================================
    @property
    def dropped(self):
        """因缓冲上限被淘汰的块数。持续增长说明主循环跟不上。"""
        return self._dropped

    @property
    def peak(self):
        """至今为止的最大样本幅度，用来确认麦克风真的收到了声音。"""
        return self._peak

    @property
    def running(self):
        """麦克风是否还开着。"""
        return self._stream is not None and self._stream.active

    @property
    def n_samples(self):
        """已被取走的累计样本数。"""
        return self._n_samples


def list_devices():
    """打印本机所有音频设备，以及默认输入设备。"""
    print(sd.query_devices())
    print("\n默认输入设备：%s" % (sd.query_devices(kind="input"),))

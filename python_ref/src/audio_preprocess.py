# -*- coding: utf-8 -*-
"""流式语音预处理：波形 -> LogFBank 特征（mini 版第一条流水线）。

============================================================
这份代码是干什么的
============================================================
把麦克风一小块一小块送来的波形，转成模型能吃的 LogFBank 特征矩阵。

每个 10ms 的音频块进来，就立刻产出一帧 80 维特征；来不及凑满一帧就攒着，
等下一块。整个过程只依赖 numpy，没有任何"未来样本"的假设 —— 这是流式
（online）与批量（offline）唯一的本质差别。

    LogFBank 流水线（7 步，全部写在下面的 _proc_frame 里）：
        去直流 -> 逐帧预加重 -> 加窗 -> FFT -> 功率谱 -> Mel 滤波 -> 取对数

============================================================
一、分帧：kaldi 的 FirstSampleOfFrame 语义
============================================================
帧长 Frame_len = 400 个样本（25ms @16kHz），帧移 Frame_step_len = 160 个样本（10ms）。
第 m 帧覆盖的样本区间是

    lo = m·Frame_step_len + Frame_step_len//2 − Frame_len//2 = 160m − 120
    hi = lo + Frame_len    = 160m + 280

注意 lo 可以是**负数**（第 0 帧就是 −120），hi 也可能越过已到达的样本数。
kaldi 的规矩是：越界的部分用**镜像反射**补出来，而不是丢帧或补零。
这就是本项目默认 snip_edges=False 的含义 —— 帧中心严格钉在 t·Frame_step_len 上，
每 10ms 必出一帧，两端缺的靠镜像补齐。

左补齐量恒为 120，且**只有第 0 帧**用得到：m=1 时 lo 已经是 40 > 0。

============================================================
二、跨块攒样本：为什么要用绝对索引
============================================================
一帧要 400 个点，而一块只有 160 个，所以得攒。缓冲区 _buf 只保存
[_base, _base+len(_buf)) 这一段样本，其中 _base 是 _buf[0] 在**整条音频**
里的绝对序号。用绝对序号而不是"相对当前块"，跨块拼接才不会错位。

攒够一帧就立刻吐出去，并把再也用不到的旧样本丢掉（_trim），否则跑一晚上
内存会一直涨。

============================================================
三、一个不显然的退化分支
============================================================
numpy 的镜像补齐（symmetric）要求补齐宽度小于原数组长度，太短的信号会
直接报错。kaldi 的做法在极短信号下会退化成 edge（左右都复制边缘样本）。
代入上面的公式能算出，这个退化只在整段样本数 n <= 140（不到 9ms）时触发
—— 实时流里只可能出现在最开头那一两块。_pad_mode 复刻了这个判据。

============================================================
四、七项必须对齐的 kaldi 口径（写错任何一项都会静默产生偏差）
============================================================
    1. snip_edges=False 的分帧语义（见上）
    2. povey 窗，且**分母是 Frame_len−1，不是 Frame_len**
    3. 窗函数手算，不用 scipy（scipy 默认 periodic 约定与 kaldi 不同）
    4. **逐帧**预加重（不是对整段波形做），且帧内必须**倒序**执行
    5. Mel 滤波器组要遍历**整数 FFT bin**、在 **Mel 域**插值
    6. 镜像补齐用 symmetric（重复边缘样本），不是 reflect
    7. 取对数前的地板是 log(max(能量, float32_eps))，不是 log(能量 + eps)

依赖：只用 numpy。
"""

import numpy as np

# kaldi 在给 Mel 能量取对数前，先把下限抬到 float32 的 epsilon：
#   1.1920929e-07 -> log = -15.9424。静音帧就停在这个地板上。
# 注意这跟"加一个 eps 再取对数"是两回事：加 1e-10 会得到 -23.03。
FLOAT32_EPS = float(np.finfo(np.float32).eps)


def hz_to_mel(hz):
    """赫兹 -> Mel 刻度。kaldi / HTK 的写法：1127·ln(1 + f/700)，
    与 2595·log10(1 + f/700) 数学上完全等价（2595/ln10 = 1127.01…），
    但这里照抄 kaldi 的 1127 取整常数与自然对数，才能逐点对齐它的滤波器组。"""
    return 1127.0 * np.log(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


class AudioPreprocess:
    """流式 LogFBank 前端。

    典型用法（和 kaldi_native_fbank.OnlineFbank 的节奏一致）::

        front = AudioPreprocess()
        while 还有音频:
            feats = front.accept_waveform(chunk)   # (n_new, 80) float32，可能 0 帧
        feats = front.input_finished()             # 收尾：补完尾帧 + 尾部补 0

    单帧的因果延迟 = Frame_len//2 = 200 个样本 = 12.5ms：第 m 帧要等样本攒到
    160m+280 才能算出来，而它的中心在 160m+80。
    """

    def __init__(self,
                 sample_rate=16000,
                 n_mels=80,
                 frame_len_ms=25.0,
                 frame_shift_ms=10.0,
                 n_fft=512,
                 preemph=0.97,
                 low_freq=20.0,
                 high_freq=0.0,
                 tail_pad_frames=100):
        """
        参数:
            sample_rate     —— 采样率（Hz）
            n_mels          —— Mel 滤波器个数，即输出特征维度
            frame_len_ms    —— 帧长（毫秒），25ms 是语音的经典取值
            frame_shift_ms  —— 帧移（毫秒），10ms 保证时间轴平滑连续
            n_fft           —— FFT 点数，取 >= 与帧长（400）接近的 2 的幂（512 @16kHz -> 257 bin）
            preemph         —— 预加重系数，通常 0.95 ~ 0.97
            low_freq        —— Mel 滤波器组下限（Hz），kaldi 默认 20
            high_freq       —— Mel 滤波器组上限；0 = Nyquist，负数 = Nyquist + 该值。
                               kaldi C++ 的默认是 -400（即 7600Hz），本模块跟sharpe官方 test_onnx.py 取 0（Nyquist = 8000Hz）。
            tail_pad_frames —— input_finished() 时在尾部补几帧 0（特征域 0， 不是静音），默认 100 帧 ≈ 1 秒，与官方一致
        """
        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        self.n_fft = int(n_fft)
        self.preemph = float(preemph)
        self.low_freq = float(low_freq)
        self.high_freq = float(high_freq)
        self.tail_pad_frames = int(tail_pad_frames)

        # kaldi FrameExtractionOptions::WindowSize() 是**截断**而不是四舍五入
        self.frame_size = int(self.sample_rate * 0.001 * frame_len_ms)     # Frame_len = 400
        self.frame_step = int(self.sample_rate * 0.001 * frame_shift_ms)   # Frame_step_len = 160
        self.n_freq = self.n_fft // 2 + 1                                  # 257 个正频率 bin

        # 第 m 帧的起点 = m·Frame_step_len + first；
        # 第一步，确定输出数据的起始点 first 在输入数据的哪个位置。常见方案对比：
        # 方案 A: 第一帧范围 [0, 400]，等待 25ms 输出，无镜像补齐。中心点落在 200 + m*160。
        # 方案 B: 第一帧范围 [-120, 280]，等待 17.5ms 输出，镜像补齐 120 个点。中心点落在 80 + m*160。
        # 方案 C: 第一帧范围 [-200, 200]，等待 12.5ms 输出，镜像补齐 200 个点。中心点落在 0 + m*160。
        # 三种方案仅对初始化的第一帧有影响，对整体的影响很小，因此三种方案都可以实施。
        # 本模块选择【方案 B】。原因：严格对齐 Kaldi 库的 snip_edges=False 行为，
        # 且在首帧等待延迟 (17.5ms) 和镜像补齐带来的假信号污染 (120点) 之间取得了最佳平衡。
   
        # 第 m 帧的起点 = m·Frame_step_len + first；
        # first 对 Frame_step_len=160/Frame_len=400 就是 -120
        # self.first = self.frame_step // 2 - self.frame_size // 2
        self.first = -120   # 可选 0, -120, -200 等，对应上述方案 A, B, C

        # 左补齐量（恒为 120，只有第 0 帧用得到），_pad_mode 判据里要用
        # self.left_pad = self.frame_size // 2 - self.frame_step // 2
        self.left_pad = - self.first 

        # 窗与 Mel 滤波器组都是一次算好、之后反复复用
        self._win = self._build_window()
        self._filt, self._filt_bins = self._build_mel_filterbank()

        self.reset()

    # ========================================================
    # 状态与重置
    # ========================================================
    def reset(self):
        """清空全部状态，回到「一个样本都没喂过」。"""
        # _buf 保存绝对序号 [_base, _base+len(_buf)) 范围内的样本
        self._buf = np.zeros(0, dtype=np.float64)
        self._base = 0     # _buf[0] 对应的绝对样本序号
        self._n = 0        # 累计已喂入的样本总数
        self._next = 0     # 下一个要输出的帧号
        self._done = False # input_finished() 是否已调用
        self._nf_total = 0 # 收尾时才确定的批量帧数

    @property
    def n_samples(self):
        """累计已喂入的样本数。"""
        return self._n

    @property
    def n_frames(self):
        """累计已产出的帧数。"""
        return self._next

    # ========================================================
    # 对外接口
    # ========================================================
    def accept_waveform(self, samples):
        """喂入一块波形，返回本次新凑出来的特征。

        参数:
            samples —— 任意长度的一维波形（float32/float64/整数都行）
        返回:
            float32 特征矩阵 (n_new, n_mels)；这一块还没凑够一帧时是 (0, n_mels)
        """
        chunk = np.asarray(samples, dtype=np.float64).ravel()
        if chunk.size:
            # 每秒只拼 100 次、每次不到 1KB，这点拷贝可以忽略
            self._buf = (np.concatenate((self._buf, chunk))
                         if self._buf.size else chunk.copy())
            # 更新已喂入的样本总数
            self._n += chunk.size

        out = []
        # while 判断已喂入的样本总数是否足够凑出下一帧，能凑就立刻产出
        while self._can_emit():
            # 根据索引从缓存中提取出一帧原始样本
            cur_frame = self._get_frame(self._next)
            # 把这一帧的原始样本走完 7 步流水线，得到一个 n_mels 维特征向量
            proc_frame = self._proc_frame(cur_frame)
            # 把这一帧的特征放到输出列表里
            out.append(proc_frame)
            # _next 号帧已经定下来了，下一帧是 _next+1
            self._next += 1
            # _buf 里有些旧样本再也用不到了，丢掉它们释放内存
            self._trim()          # 丢掉后面任何一帧都用不到的旧样本

        if not out:
            return np.zeros((0, self.n_mels), dtype=np.float32)
        return np.stack(out).astype(np.float32)

    def input_finished(self):
        """宣告音频结束：补出尾部帧，再补 tail_pad_frames 帧特征域 0。

        补的 0 是**特征域**的 0（log 域对应能量 1），不是静音值 —— 照抄官方
        test_onnx.py 的做法，不要自作主张改成静音。不补的话，解码循环末尾
        凑不满一个 chunk 的那段会被整段丢掉，尾字就识别不出来。

        返回:
            float32 特征矩阵 (n_rest [+ tail_pad_frames], n_mels)
        """
        if self._done:
            return np.zeros((0, self.n_mels), dtype=np.float32)
        self._done = True
        # 样本总数此时已定，批量版的帧数公式可以用了：round(N/Frame_step_len)
        self._nf_total = (self._n + self.frame_step // 2) // self.frame_step

        out = []
        while self._next < self._nf_total:
            # 根据索引从缓存中提取出一帧原始样本
            cur_frame = self._get_frame(self._next)
            # 把这一帧的原始样本走完 7 步流水线，得到一个 n_mels 维特征向量
            proc_frame = self._proc_frame(cur_frame)
            # 把这一帧的特征放到输出列表里
            out.append(proc_frame)
            self._next += 1

        if out:
            # 情况 1：out 列表非空，把列表里的多个 (80,) 一维数组，堆叠成 (帧数, 80) 的二维矩阵
            feats = np.stack(out).astype(np.float32)
        else:
            # 情况 2：out 列表为空，返回一个形状为 (0, 80) 的空矩阵
            feats = np.zeros((0, self.n_mels), dtype=np.float32)

        # 无条件补：跟官方 test_onnx.py 一致。整段不足 80 个样本时本就没有
        # 任何帧，此处会补成 (tail_pad_frames, n_mels) 全 0 —— 这是刻意的，
        # 不要加 "feats.shape[0] > 0" 之类的守卫，否则与批量版语义不一致。
        if self.tail_pad_frames > 0:
            feats = np.pad(feats, ((0, self.tail_pad_frames), (0, 0)),
                           mode="constant", constant_values=0.0)
            feats = np.ascontiguousarray(feats, dtype=np.float32)
        return feats

    # ========================================================
    # 出帧
    # ========================================================
    def _can_emit(self):
        """现在能不能把 _next 号帧定下来。

        流未结束时，只有 hi <= n 的帧能出（不需要右补齐）；收尾之后帧数上界
        由批量公式给定，尾部帧允许右补齐。
        """
        if self._done:
            return self._next < self._nf_total

        # 如果累计缓存长度小于下一帧末尾对应的索引，返回false
        next_frame_end_idx = self._next * self.frame_step + self.first + self.frame_size
        if self._n < next_frame_end_idx:
            return False
        else:
            return True


    def _get_frame(self, m):
        """取第 m 帧的 Frame_len 个原始样本，两端越界处镜像补齐。

        只补这一帧自己越界的那一段。因为补出来的点只取决于贴边的那几个样本、
        且镜像长度固定，逐帧补与"整段补一次"得到的结果完全相同。
        """
        # 帧起点索引
        lo = m * self.frame_step + self.first
        # 帧终点索引
        hi = lo + self.frame_size
        # 累计输入的总样本数
        n = self._n
        # 真实已有缓存的绝对索引
        abs_start_idx = max(lo, 0)          # 真实样本区间的绝对起点
        abs_end_idx = min(hi, n)          # 真实样本区间的绝对终点

        # 取出缓存里这一段样本的切片，
        # 缓存会释放旧样本，self._base是缓存里上一个样本在整段音频里的绝对索引
        seg = self._buf[abs_start_idx - self._base: abs_end_idx - self._base]

        # 补帧：nl/nr 是需要补齐的点数，只有 m=0 或流末尾才非 0
        nl = abs_start_idx - lo             # 需要左补几个点（只有 m == 0 时非 0）
        nr = hi - abs_end_idx             # 需要右补几个点（只有流末尾非 0）
        # 中间帧不需要补点，直接返回seg作为当前帧
        if nl == 0 and nr == 0:
            return seg          # 绝大多数帧走这里，直接是一段切片
        
        # 第一帧时向左补帧，最后一帧向右补帧
        mode = self._pad_mode(n)
        parts = []
        if nl:
            parts.append(self._left_pad(nl, mode))
        parts.append(seg)
        if nr:
            parts.append(self._right_pad(nr, mode))

        return np.concatenate(parts)

    def _pad_mode(self, n):
        """决定这一帧用哪种补齐方式：'symmetric' 或退化的 'edge'。

        symmetric（默认）：镜像时**重复**边缘样本，即 sig[-k] = sig[k-1]。
        edge：左右都复制边缘样本，只在极短信号（n <= 140）时触发 ——
              因为 numpy 的镜像拷贝不允许补齐宽度超过原数组长度。
        """
        nf = (n + self.frame_step // 2) // self.frame_step
        if n <= 0 or nf <= 0:
            return "edge"
        right = max(0, (nf - 1) * self.frame_step + self.first
                    + self.frame_size - n)
        if self.left_pad < n and right < n:
            return "symmetric"
        return "edge"

    def _left_pad(self, k, mode):
        """补在帧首的 k 个点。来源是信号开头。

        只有第 0 帧会调用它，此时 _base 必然是 0（_trim 还没机会跑过）。
        """
        if mode == "edge":
            return np.full(k, self._buf[0 - self._base], dtype=np.float64)
        # symmetric: sig[-j] = sig[j-1]，所以取前 k 个样本倒序
        return self._buf[0 - self._base: k - self._base][::-1]

    def _right_pad(self, k, mode):
        """补在帧尾的 k 个点。来源是信号结尾。"""
        n = self._n
        if mode == "edge":
            return np.full(k, self._buf[n - 1 - self._base], dtype=np.float64)
        # symmetric: sig[n+j] = sig[n-1-j]，所以取末尾 k 个样本倒序
        return self._buf[n - k - self._base: n - self._base][::-1]

    def _trim(self):
        """丢掉后面任何一帧都用不到的旧样本。

        下一帧的起点是 _next·Frame_step_len + first，它之前的样本再也不会被读到。
        用 .copy() 而不是切片：切片是视图，会让整块原始内存一直挂着不释放。
        """
        # 下一帧的起点
        keep_from = max(0, self._next * self.frame_step + self.first)

        # 下一帧的起点 - 减去已经删掉的数据索引_base ，就是下一帧的起点的相对索引
        k = keep_from - self._base

        if k > 0:
            # 用需要保留的样本直接覆盖掉旧样本，释放内存
            self._buf = self._buf[k:].copy()
            # 已经删掉的数据索引_base，更新为下一帧的起点索引
            self._base = keep_from

    # ========================================================
    # 帧内处理：7 步 LogFBank，显式逐帧
    # ========================================================
    def _proc_frame(self, frame):
        """把一帧 Frame_len 个原始样本走完 7 步，得到一个 n_mels 维特征向量。

        这个函数就是 C++ 主循环体的逐行对应物：每一步都是一次显式运算，
        没有大矩阵批量。

        参数:
            frame —— 一帧原始样本 (Frame_len,)
        返回:
            LogFBank 特征 (n_mels,)，float64
        """
        x = np.asarray(frame, dtype=np.float64).copy()

        # --- 1) 去直流：减掉本帧均值 -------------------------------------
        # 麦克风有直流偏置，不去掉会在 0Hz 处堆一个虚假的大峰值，
        # 经 Mel 滤波器泄漏到低频通道上。
        x -= x.mean()

        # --- 2) 预加重：y[i] = x[i] - α·x[i-1] -----------------------
        # 当前帧内的逐个样本，采用倒序的方式进行预加重计算
        # 语音频谱整体按 -6dB/倍频程衰减，预加重把高频抬回来，让各频段量级均衡。
        # 必须**倒序**：正序的话 x[i-1] 已经被改写成新值，结果就错了。
        # 帧首没有"前一个样本"，kaldi 约定用帧首样本自己顶替，于是 x[0] - α·x[0] = x[0]·(1-α)，也就是下面这一行。
        for i in range(self.frame_size - 1, 0, -1):
            x[i] -= self.preemph * x[i - 1]
        x[0] *= (1.0 - self.preemph)

        # --- 3) 加窗 ----------------------------------------------------
        # FFT 隐含假设输入首尾相接。不加窗的话，帧两端的不连续会造成频谱
        # 泄漏（主频两侧拖出虚假谱线）。povey 窗两端严格归零，泄漏最小。
        x *= self._win

        # --- 4) FFT -----------------------------------------------------
        # 实信号只需正频率半边：rfft 输出 n_fft//2 + 1 = 257 个 bin。
        # 对应 C++ 侧就是一次 FFT 库调用（如 kissfft / pffft），不用手写。
        # 这里音频的采样率是 sample_rate =16000,因此fft的最大频率是sample_rate/2 = 8 khz （奈奎斯特频率）
        # dft 的定义要求 self.n_fft > self.frame_size
        spec = np.fft.rfft(x, n=self.n_fft)

        # --- 5) 功率谱 --------------------------------------------------
        # 周期图定义：|X|²。平方后各频率分量的强弱被拉开，更贴近"能量"。
        power = spec.real * spec.real + spec.imag * spec.imag

        # --- 6) Mel 滤波：257 维功率谱 × 80×257 三角权重 -> 80 维 ----------
        # 降维就发生在这一步，做的是「相乘累加」：
        #   power[k]          —— 第 k 个 FFT bin 的功率（bin k 对应 k·fs/n_fft Hz）
        #   self._filt[m, k]  —— 第 m 个三角滤波器在 bin k 上的权重（0 ~ 1）
        #   mel[m]            —— 第 m 个 Mel 通道的能量
        # 每个通道 = 三角权重那一行与整条功率谱的点积，也就是把三角盖住的
        # 那几个 bin 按权重加权求和：
        #     mel[m] = Σ_k self._filt[m, k] · power[k]
        # 权重表在 _build_mel_filterbank 里一次算好，这里只做乘加。
        #
        # 8kHz 内 257 维均匀频谱对语音冗余度很高，人耳对频率的感知又非线性
        # （低频分辨力强、高频弱）；Mel 滤波器组低频密、高频疏，正好模拟这点。
        # 滤波器组 80×257 里绝大多数是 0，全扫的话每帧两万次内层迭代，纯 Python
        # 撑不住；_filt_bins[m] 预先记下每个滤波器非零的 bin 区间，只扫这一段。
        mel = np.empty(self.n_mels, dtype=np.float64)
        for m in range(self.n_mels):
            lo, hi = self._filt_bins[m]      # 第 m 个三角的非零 bin 区间 [lo, hi)
            acc = 0.0
            for k in range(lo, hi):
                # 权重 × 该 bin 的功率，逐 bin 累加成一个通道的总能量
                acc += self._filt[m, k] * power[k]

            # --- 7) 取对数，并抬到 kaldi 的地板 --------------------------
            # 用 max 而不是加 eps：静音帧才会停在 log(1.19e-7) = -15.94，
            # 而不是 log(能量 + 1e-10) 得到的 -23 一带。
            mel[m] = np.log(acc) if acc > FLOAT32_EPS else np.log(FLOAT32_EPS)

        return mel

    # ========================================================
    # 构造：窗函数与 Mel 滤波器组（各算一次，之后复用）
    # ========================================================
    def _build_window(self):
        """povey 窗：(0.5 − 0.5·cos(2πn/(Frame_len−1)))^0.85。
        它是在汉宁窗的基础上，整体做了一个 0.85 次方 的变换。
        手算而不用 scipy.signal.get_window，原因只有一个：**分母**。
        kaldi 的窗都用 (Frame_len−1) 作分母（首尾都取到零点），scipy 默认
        periodic=True 用 Frame_len 作分母 —— Frame_len=400 时两者最大差 5.5e-3，
        足以把 log-mel 特征拉开可见的差距。
        """
        n = np.arange(self.frame_size, dtype=np.float64)
        a = 2.0 * np.pi / (self.frame_size - 1)
        return (0.5 - 0.5 * np.cos(a * n)) ** 0.85

    def _build_mel_filterbank(self):
        """构造 Mel 三角滤波器组，形状 (n_mels, n_freq)。

            kaldi 的做法（和"先算浮点 bin 位置、再按 bin 序号插值"不同）：

                1. 把 low_freq / high_freq 换到 Mel 刻度，在 Mel 刻度上等分
                    n_mels + 1 段，每个滤波器得到 left / center / right 三个边界；
                2. **遍历每一个整数 FFT bin**，把它的中心频率 i·(fs/n_fft) 换到 Mel；
                3. 若该 Mel 值严格落在 (left, right) 内，就按它到 center 的距离在
                    **Mel 域**做线性插值。

            为什么必须这样：Hz -> Mel 是非线性的，第 3 步必须拿 Mel 值（而不是
            bin 序号）去插值才能得到 kaldi 的形状；另外边界用严格不等号，
            落在边界上的 bin 直接不参与（权重 0），而不是插值出 0.5。

            返回 (滤波器组, 每个滤波器的非零 bin 区间列表)。
        """
        nyquist = 0.5 * self.sample_rate
        # 滤波器组的上限：high_freq > 0 直接用；0 表示取到 Nyquist（8000Hz）；
        # 负数表示从 Nyquist 再往回收，如 -400 -> 7600Hz（kaldi C++ 的默认）
        high = self.high_freq if self.high_freq > 0.0 else nyquist + self.high_freq
        # 每个 FFT bin 的带宽 = fs/n_fft = 31.25Hz；bin k 的中心频率 = k · bin_width
        bin_width = self.sample_rate / float(self.n_fft)

        # 三角滤波器的摆放全在 Mel 刻度上完成：低频密、高频疏
        mel_low = hz_to_mel(self.low_freq)  # 最小频率对应的梅尔刻度
        mel_high = hz_to_mel(high)          # 最大频率对应的梅尔刻度
        # 相邻等距点的 Mel 间距；一个三角横跨两段（left->center + center->right）
        mel_delta = (mel_high - mel_low) / (self.n_mels + 1)

        # 257 个 bin 的中心频率先统一换成 Mel，免得下面 80 个滤波器重复算
        # 换算后的bin_mels序列低频点对应梅尔值间距小，高频点对应梅尔值间距大
        bin_mels = hz_to_mel(bin_width * np.arange(self.n_freq))

        # filt[m, k] = 第 m 个三角在 bin k 上的权重；bins[m] = 该行非零的 bin 区间
        filt = np.zeros((self.n_mels, self.n_freq), dtype=np.float64)
        bins = []
        # 相邻等距点的 Mel 间距，
        # 前半段一个三角滤波区间中的低频响应点少，后段一个滤波区间会有多个高频点。
        # 每个三角形是一个带通滤波器，将落入其区间的各个 FFT bin 的【能量】，
        # 乘以对应的【三角形权重】进行加权求和，最终压缩成 1 个代表该频带总能量的标量。
        for m in range(self.n_mels):
            # 第 m 个三角的三个端点（都是 Mel 值）：左脚 / 顶点 / 右脚
            left = mel_low + m * mel_delta
            center = mel_low + (m + 1) * mel_delta
            right = mel_low + (m + 2) * mel_delta

            # 只有中心频率严格落在 (left, right) 里的 bin 归本滤波器，其余权重恒 0。
            # bin_mels 单调递增，所以命中的 bin 一定是连续的一段
            inside = np.nonzero((bin_mels > left) & (bin_mels < right))[0]
            if inside.size == 0:
                # 三角窄到罩不住任何一个 bin 中心时，整行全 0，区间记成空
                bins.append((0, 0))
                continue
            # 记成半开区间 [lo, hi)，供运行期只扫非零部分
            lo, hi = int(inside[0]), int(inside[-1]) + 1
            for k in range(lo, hi):
                bm = bin_mels[k]
                # 以顶点为界分两段线性插值：上升沿 0->1，下降沿 1->0
                filt[m, k] = ((bm - left) / (center - left) if bm <= center
                              else (right - bm) / (right - center))
            bins.append((lo, hi))
        return filt, bins

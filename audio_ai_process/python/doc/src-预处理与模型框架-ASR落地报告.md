# src 预处理与模型框架 —— 输入输出特征、ASR 落地报告

> 本文由 `src/` 四个模块的注释与三个入口脚本的装配注释整理而成，只描述
> **当前代码实际做了什么**、**每层的输入输出契约**、**移植到 C++ 时要守住什么**。
> 七项前端差异的完整推导与实测数据见
> [工作流报告](sherpa-onnx-ASR预处理及后处理工作流报告.md)，两份文档互为补充：
> 工作流报告回答「为什么这么对齐」，本文回答「对齐之后代码长什么样」。

---

## 一、总览

### 1.1 三条流水线

```
                      ┌──────────────────────────────────────────┐
 麦克风 ── PortAudio │ MicClient.get_last_audio()               │
                      │   → 一维 float32 波形（长度不固定）        │
                      └──────────────────┬───────────────────────┘
 wav 文件 ── soundfile ──► 按 BLOCK=160 切块 ┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────────┐
                      │ AudioPreprocess.accept_waveform(blk)     │
                      │   → feats (n_new, 80) float32             │
                      └───────────┬──────────────────┬───────────┘
                                  │                  │
                                  ▼                  ▼
              ┌───────────────────────┐  ┌─────────────────────────┐
              │ AsrDecoder            │  │ SpectrogramImage        │
              │  .accept_features()   │  │  .accept_features()     │
              │   → text (str)        │  │   → BGR 画布 (H, W, 3)  │
              └───────────────────────┘  └─────────────────────────┘
```

三个处理单元**只靠特征矩阵相连，互不引用**：`AudioPreprocess` 不知道谁在消费特征，
`AsrDecoder` 不知道特征从哪来、模型跑在哪块硬件上，`SpectrogramImage` 只接收特征、
不碰波形。入口脚本只负责「配置 + 数据源」。

### 1.2 模块职责

| 文件 | 类 | 职责 | 依赖 |
|---|---|---|---|
| `src/audio_preprocess.py` | `AudioPreprocess` | 波形块 → LogFBank；维护跨块样本尾巴、分帧、7 步帧内处理 | numpy |
| `src/asr_decode.py` | `AsrDecoder` | 三个 ONNX 会话 + 贪心搜索 + 文本还原；维护特征缓冲与全部跨块状态 | numpy（会话由外部建好传入） |
| `src/mic_client.py` | `MicClient` | PortAudio 回调采集，主循环主动拉取全部累积样本 | numpy + sounddevice |
| `src/spectrogram_image.py` | `SpectrogramImage` | 特征 → BGR 滚动语谱图画布（viridis） | numpy + OpenCV |

代码风格是刻意的：显式逐帧循环、不引入 scipy（窗函数手算）、不起 Python 线程，
目标是能逐行对照移植。

---

## 二、预处理链路：波形 → LogFBank

### 2.1 参数

| 参数 | 取值 | 说明 |
|---|---|---|
| `sample_rate` | 16000 Hz | 模型固定 16 kHz |
| `frame_len_ms` | 25 ms → `frame_size = 400` | **截断**取整，不是四舍五入 |
| `frame_shift_ms` | 10 ms → `frame_step = 160` | 每 10 ms 必出一帧 |
| `n_fft` | 512 | 正频率 bin 数 `n_freq = 257` |
| `n_mels` | 80 | 输出特征维度，入口从 `dec.n_mels` 取，避免两边对不上 |
| `preemph` | 0.97 | 预加重系数 |
| `low_freq` / `high_freq` | 20 Hz / 0（= Nyquist 8000 Hz） | `high_freq` 为负时表示 Nyquist + 该值（kaldi C++ 默认 −400） |
| `tail_pad_frames` | 100 | 收尾补的特征域 0，约 1 秒 |

### 2.2 帧内七步流水线（`_proc_frame`）

一帧 400 个原始样本，依次经过：

| # | 步骤 | 实现 | 为什么 |
|---|---|---|---|
| 1 | 去直流 | `x -= mean(x)` | 麦克风直流偏置会在 0 Hz 堆虚假峰值，经 Mel 泄漏到低频通道 |
| 2 | 逐帧预加重 | **倒序** `x[i] -= α·x[i-1]`，最后 `x[0] *= (1−α)` | 语音频谱按 −6 dB/倍频程衰减；倒序是因为正序会用到已被改写的前值。帧首没有前一个样本，按 kaldi 约定用自身顶替 |
| 3 | 加窗 | `povey = (0.5 − 0.5·cos(2πn/(400−1)))^0.85` | FFT 隐含首尾相接；povey 两端严格归零，泄漏最小。分母必须是 `Frame_len−1`（kaldi 口径），不是 `Frame_len` |
| 4 | FFT | `np.fft.rfft(x, n=512)` → 257 bin | 实信号只需正频率半边；C++ 侧对应一次 FFT 库调用 |
| 5 | 功率谱 | `|X|² = re² + im²` | 周期图定义，平方后各频率分量强弱拉开 |
| 6 | Mel 滤波 | 80 个三角 × 257 bin 加权求和 → 80 维 | 降维发生在这里；低频密、高频疏，模拟人耳非线性 |
| 7 | 取对数 | `log(max(acc, float32_eps))` | 静音地板停在 `log(1.192e-7) = −15.94`；不是 `log(e + eps)`（那会到 −23 一带） |

第 6 步的内层循环只扫每个三角的非零 bin 区间 `_filt_bins[m]`，否则 80×257
每帧两万次迭代，纯 Python 撑不住。

### 2.3 分帧：kaldi 的 `FirstSampleOfFrame` 语义

第 m 帧覆盖的绝对样本区间：

```
lo = m·160 + first = 160m − 120
hi = lo + 400     = 160m + 280
```

`lo` 可以是负数（第 0 帧就是 −120），`hi` 也可能越过已到达样本数。kaldi 的规矩是
**越界处镜像补齐**，而不是丢帧或补零——这就是 `snip_edges=False`：帧中心严格钉在
`t·160` 上，每 10 ms 必出一帧。

`first` 的三种取值（只影响第一帧及初始化等待）：

| 方案 | `first` | 首帧范围 | 首帧等待 | 镜像点数 | 中心点 |
|---|---|---|---|---|---|
| A | 0 | [0, 400] | 25 ms | 0 | 200 + 160m |
| **B（本项目）** | **−120** | **[−120, 280]** | **17.5 ms** | **120** | **80 + 160m** |
| C | −200 | [−200, 200] | 12.5 ms | 200 | 0 + 160m |

选 B：严格对齐 kaldi `snip_edges=False`，并在首帧等待（17.5 ms）与镜像假信号污染
（120 点）之间取得平衡。

**稳态因果延迟**是 200 个样本 = 12.5 ms：第 m 帧要等样本攒到 `160m + 280`，而它的
中心在 `160m + 80`。左补齐量恒为 120，且只有第 0 帧用得到（m=1 时 `lo = 40 > 0`）。

### 2.4 流式实现：绝对索引 + 增量出帧

一帧要 400 点、一块只有 160 点，所以必须攒。缓冲区 `_buf` 只保存
`[_base, _base + len(_buf))` 这一段，`_base` 是 `_buf[0]` 在**整条音频**里的绝对序号——
用绝对序号而不是「相对当前块」，跨块拼接才不会错位。

三个关键判据：

```python
# 出帧：已喂样本数够到该帧末尾
self._n >= self._next * 160 - 120 + 400      # 即 _next*160 + 280

# 收尾之后的帧数上界：批量公式
_nf_total = (self._n + 80) // 160            # 等价 round(N/160)

# 裁掉旧样本：下一帧起点之前的都再用不到
keep_from = max(0, self._next * 160 - 120)
```

`_trim` 用 `.copy()` 而不是切片：切片是视图，会让整块原始内存一直挂着不释放。

### 2.5 边界补齐：`symmetric` 与退化 `edge`

- **symmetric（默认）**：镜像时**重复**边缘样本，`sig[−k] = sig[k−1]`；
- **edge**：左右都复制边缘样本。只在极短信号（整段 `n <= 140`，不到 9 ms）时触发，
  因为 numpy 的镜像拷贝不允许补齐宽度超过原数组长度；实流里只可能出现在最开头。

因为补出来的点只取决于贴边的几个样本、镜像宽度固定，**逐帧补**与「整段补一次」
结果完全相同——这是流式与批量能逐位一致的前提之一。

### 2.6 七个必须对齐的 kaldi 口径

| # | 口径 | 本项目实现 |
|---|---|---|
| 1 | `snip_edges=False` 分帧语义 | 帧中心钉在 `t·160`，两端镜像 |
| 2 | povey 窗，分母 `Frame_len−1` | 手算 `(0.5−0.5cos)^0.85` |
| 3 | 不用 scipy 窗 | scipy 默认 periodic（分母 400），最大差 5.5e-3 |
| 4 | **逐帧**且**倒序**预加重 | 见 2.2 第 2 步 |
| 5 | Mel 滤波器组遍历整数 bin、Mel 域插值 | 见 2.7 |
| 6 | 镜像用 symmetric 而非 reflect | 重复边缘样本 |
| 7 | 对数地板 `log(max(e, eps))` | 静音停在 −15.94 |

### 2.7 Mel 滤波器组

kaldi 的做法与「先算浮点 bin 位置、再按 bin 序号插值」不同：

1. `low_freq`/`high_freq` 换到 Mel 刻度，等分 `n_mels + 1` 段，每个滤波器得到
   `left / center / right` 三个边界；
2. **遍历每一个整数 FFT bin**，把中心频率 `i·(fs/n_fft)` 换到 Mel；
3. 若 Mel 值**严格**落在 `(left, right)` 内，按到 `center` 的距离在 **Mel 域**线性插值；
   落在边界上的 bin 权重直接为 0，而不是插值出 0.5。

Hz→Mel 用 kaldi 的写法 `1127·ln(1 + f/700)`（与 2595·log10 等价，但常数与自然对数
照抄才能逐点对齐）。滤波器组只算一次，运行期只做乘加。

### 2.8 收尾：补尾帧 + 补 100 帧特征域 0

`input_finished()` 做两件事：

1. 用批量帧数公式补出剩余尾帧（允许右镜像）；
2. **无条件**补 `tail_pad_frames=100` 帧 0，约 1 秒。

补的 0 是**特征域**的 0（log 域对应能量 1），不是静音值（静音是 −15.94）——
照抄官方 `test_onnx.py`。不补的话，最后凑不满一个 chunk 的尾段会被整段丢掉，
尾字识别不出来。整段不足 80 个样本时也会补成全 0，**不要**加
`feats.shape[0] > 0` 之类的守卫，否则与批量语义不一致。

---

## 三、输入输出特征契约

### 3.1 接口一览

| 模块 | 方法 | 输入 | 输出 | 形状 / dtype |
|---|---|---|---|---|
| `AudioPreprocess` | `accept_waveform(samples)` | 任意长度一维波形 | 新特征 | `(n_new, 80)` float32，可能 0 行 |
| | `input_finished()` | — | 尾帧 + 100 帧 0 | `(n_rest + 100, 80)` float32 |
| | `reset()` | — | 回到未喂状态 | — |
| `AsrDecoder` | `accept_features(feats)` | 特征块 | 当前完整文本 | `feats: (n, 80)` float32 → `str` |
| | `run_encoder(x)` | 一个 chunk | 声学编码 | `x: (1, 61, 80)` → `(1, nf, 512)`，`nf = 12` |
| | `run_decoder(hyp)` | 最近 `context_size` 个 id | 语言编码 | `(1, 2)` int64 → `(1, 512)` float32 |
| | `run_joiner(enc, dec)` | 单帧 + 语言编码 | 全词表打分 | `(1, 1, 512)` + `(1, 512)` → `(1, 5000)` |
| | `reset()` / `text` / `token_ids` / `info()` | — | 文本与尺寸信息 | `token_ids` 不含开头 2 个占位 blank |
| `SpectrogramImage` | `accept_features(feats)` | 特征块 | 整幅画布 | `(H, W, 3)` uint8 BGR，`H = header_h + plot_h + footer_h` |
| | `frame()` | — | 按当前缓存重画 | 同上 |
| | `features()` | — | 窗口内特征（只读） | `(≤cols, 80)` float32 |
| `MicClient` | `get_last_audio()` | — | 自上次调用以来的**全部**样本 | 一维 float32；无新数据时 `None` |

### 3.2 特征域的语义

- 数值是 **`ln(能量)`**，不是 dB（不要与 `10·log10` 混用）；
- 静音地板 **−15.94**（`ln(float32_eps)`）；
- 补出来的 0 表示能量 1，比静音还亮——只该喂解码器，不该喂给眼睛；
- 可视化色阶的 `lo/hi` 单位同样是 `ln(能量)`，默认 `lo = −15.94`、`hi = 8.0`。

### 3.3 两条消费路径的补帧差异

| 路径 | 构造参数 | 原因 |
|---|---|---|
| 解码 | `tail_pad_frames=100`（默认） | 尾段要凑满 chunk 才能被解码 |
| 可视化 | `tail_pad_frames=0` | 补的 0 在语谱图上是一块比静音还亮的假色块 |

可视化收尾时调 `SpectrogramImage.frame()` 重画最后一屏，**不要**把
`input_finished()` 的返回值喂进来。

### 3.4 时序

| 量 | 值 |
|---|---|
| 帧移 | 10 ms / 160 样本 |
| 稳态单帧因果延迟 | 12.5 ms（200 样本） |
| 首帧等待（方案 B） | 17.5 ms |
| 解码时间轴步进 | 480 ms（48 帧） |
| encoder 每次实际产出 | 12 帧（整体 4 倍 = encoder_embed 2× + U-Net 内部 2×） |
| 尾部补帧 | 100 帧 ≈ 1 秒 |

---

## 四、模型框架结构

模型：`sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05`。

### 4.1 transducer 三段分工

| 模型 | 角色 | 输入 | 输出 | 职责 |
|---|---|---|---|---|
| `encoder.int8.onnx` | 声学编码 | 61 帧特征 + 116 个状态 | 声学表征 + 116 个新状态 | 负责「听到什么」，跨块累积记忆 |
| `decoder.onnx` | 预测网络 / predictor | 最近 2 个 token | 512 维语言编码 | 负责「上一字接龙」，辅助去重、断句；轻量 2-token CNN，不是长记忆 LSTM |
| `joiner.int8.onnx` | 联合网络 | 单帧声学表征 + 语言编码 | 词表 5000 维打分 | 综合两边快速决策出字 |

### 4.2 encoder：滑动窗口 + 4 倍时间压缩 + transformer

1. **滑动窗口**：每次喂 `T = 61` 帧，但时间轴只前进 `decode_chunk_len = 48` 帧
   （480 ms）。多出的 13 帧是卷积模块必需的重叠上下文——旧 13 帧 + 新 48 帧拼接后
   送入，避免边界补零失真。
2. **encoder_embed：2D 卷积 + ConvNeXt，时间只压 2 倍**。输入 `(1, 61, 80)` 先当作
   `(N, 1, T, 80)` 的图像，依次经过（参数取自 onnx 图）：

   | 层 | 类型 | 形状变化 |
   |---|---|---|
   | `conv.0` | Conv2d 3×3，stride (1,1)，频率维 pad 1 | `61×80` → `59×80` |
   | `conv.4` | Conv2d 3×3，stride (2,2) | `59×80` → `29×39` |
   | `conv.7` | Conv2d 3×3，stride (1,2) | `29×39` → `27×19` |
   | `convnext` | 7×7 depthwise（128 组）+ 两个 1×1 | `27×19×128` 不变 |
   | `out` | 投影 | 通道 `128 → 192` |

   输出 `(1, 27, 192)`。**时间维在开头的 encoder_embed 里只压缩 ~2 倍**（只有
   `conv.4` 的 stride 是 2），频率维 80 → 19（对应状态 `embed_states (1,128,3,19)`）。
   通道也不是一步到 512：`80 → 8 → 32 → 128 → 192`（192 即第一个 stack 的宽度）。
3. **U-Net 内部：6 个 stack，时间先压缩再恢复**。维度 `192 → 256 → 512 → 768 →
   512 → 256`，分别 2/2/4/5/4/2 个 block、共 19 个。各 stack 用 reshape 合并/拆分
   相邻帧做时间下采样与上采样，因此不同 stack 的帧率不同（注意力缓存的 `T` 分别是
   256/128/64/32）。512 出现在第 3 个 stack，`encoder_out` 最终也是 512 维。
4. **总时间压缩是 4 倍，但分摊在两处**：encoder_embed 2 倍 + 内部 U-Net 再 2 倍，
   所以输入 48 帧（一个 chunk 的非重叠部分）最终产出 **12** 帧，而不是开头一口气压 4 倍。
5. **上下文建模**：encoder_embed 输出的 token 序列进入 Transformer，自注意力捕获
   长距离依赖，得到融合全局上下文的高级时序声学表征。

### 4.3 116 个状态张量

按注释拆成三类：

| 类别 | 数量 | 构成 | 形状 / 说明 |
|---|---|---|---|
| 全局状态 | 2 | `processed_lens`、`embed_states` | `(1,)` int64；`(1, 128, 3, 19)` float（频率维 80 经两次 stride 2 → 19） |
| 注意力缓存 | 76 | 19 block × 4 | key `(T, 1, key_dim)`、value `(T, 1, value_dim)`、value2 `(T, 1, value_dim)`、non-linear attention `(1, 1, T, 3·C/4)`；`T` 按 stack 为 256/128/64/32，`key_dim` 128/256，`value_dim` 48/96 |
| CNN 边界状态 | 38 | 19 block × 2（conv1/conv2） | `(1, C, kernel/2)`，`C` 为 192/256/512/768，`T = 15/7`；流式架构在边界**不做 padding**，直接拼历史特征做因果卷积 |

合计 `2 + 76 + 38 = 116`（每个 block 6 个：4 个注意力缓存 + 2 个卷积缓存），
与 sherpa-onnx `OnlineZipformer2TransducerModel::GetEncoderInitStates()` 的构造一一对应。
这些状态**除 `reset()` 外绝不清零**——流式能记住前文的全部机制就在这里。

形状与 dtype 全部由 ONNX 输入签名自动推导：字符串维（`'N'`）按 batch=1 取 1，
其余取字面值。换模型不用改代码。

### 4.4 decoder 与 joiner

- decoder 元数据：`context_size = 2`、`vocab_size = 5000`；
- 内置 Embedding 表把 token id 查成隐藏向量，**外部 `tokens.txt` 的大小与顺序必须
  与之严格对应**；
- joiner 接收「一帧声学表征 + 上一次接龙得到的语言编码」，输出整个词表的 logit。

### 4.5 关键参数全部来自模型自身

| 来源 | 参数 |
|---|---|
| encoder `custom_metadata_map` | `model_type=zipformer2`、`T=61`、`decode_chunk_len=48`、`cnn_module_kernels`、`encoder_dims`、`num_encoder_layers`、`num_heads`、`left_context_len`、`query_head_dims`、`value_head_dims` |
| decoder `custom_metadata_map` | `context_size=2`、`vocab_size=5000` |
| 输入签名 | `n_mels=80`、116 个状态的形状与 dtype |
| `tokens.txt` | `blank_id`（查 `<blk>`，不硬编码 0） |

代码里没有任何硬编码的帧数/状态数——这是换模型不用改代码的前提。

---

## 五、ASR 解码落地

### 5.1 跨块必须存活的状态

| 状态 | 初值 | 作用 |
|---|---|---|
| `states`（116 个张量） | 全 0，形状由签名推导 | encoder 的全部记忆 |
| `hyp` | `[blank_id] * context_size` | 已确定的 token 序列 |
| `decoder_out` | `run_decoder(hyp)` | 最近一次语言编码，blank 时直接复用 |
| `_feat_buf` | 空 | 攒够一个 chunk 的特征 |

`reset()` 是唯一的清零入口。

### 5.2 贪心搜索主循环

```python
# 初始化
states = zeros(116 个状态的签名)
hyp = [blank_id] * context_size
decoder_out = run_decoder(hyp)

# 每喂一块特征
feat_buf = concat(feat_buf, feats)
while len(feat_buf) > T:                  # 严格大于，见 5.3
    x = feat_buf[:T][None]                # (1, 61, 80)
    feat_buf = feat_buf[decode_chunk_len:] # 前进 48 帧

    enc_feats, states = run_encoder(x, states)   # (1, 12, 512)
    for k in range(enc_feats.shape[1]):          # 取运行时帧数，不写死 T
        cur = enc_feats[0, k:k+1]                # (1, 1, 512)
        tid = argmax(run_joiner(cur, decoder_out))
        if tid != blank_id:
            hyp.append(tid)
            decoder_out = run_decoder(hyp[-context_size:])   # 只有出字才刷新
```

要点：

- **blank 不刷新 decoder**：绝大多数帧都是 blank，只有真出字才跑一次 decoder 前向；
- **`encoder_out.shape[1]` 必须取运行时值**（实测 12），写死 `T` 会错；
- `hyp` 是「尚未最终确认的候选序列」，在贪心下即为正式序列；
- 贪心的代价是抛弃了其他高概率字，没有纠错可能；束搜索会保留前 K 条路径，
  用累计得分淘汰错误接龙，但要多等一个 chunk 周期。

### 5.3 流式判据为什么必须是 `len(buf) > T`

```
批量第 k 次喂 encoder 的条件   48k + 61 < L      （L = 总帧数）
等价于                        L − 48k > 61
流式喂完第 k 次后的剩余帧数    L − 48k
=> 流式判据必须是 len(buf) > T   （严格大于）
```

写成 `>=` 会在 `L` 恰好等于 `48k + 61` 时多喂一次，尾部多吐出几个 token。
代价是实时场景下多攒 1 帧（10 ms）才喂。

### 5.4 后处理：token → 文本

1. 从 `tokens.txt` 读 `{id: token}`（每行 `token id`，用 `rpartition` 切）；
2. `▁`（U+2581）还原成空格；
3. 按位置去空格：
   - **CJK 之间**去空格：`昨 天` → `昨天`；
   - **全角标点两侧**去空格：`Monday ，today` → `Monday，today`；
   - **CJK 与 ASCII 之间保留**：`是 Monday` 不能粘成 `是Monday`；
4. `strip()` 收尾。

查表时用 `id2token.get(i, "?")` 兜底，遇到词表外 id 不崩。

### 5.5 收尾

`front.input_finished()` 返回的尾帧 + 100 帧 0 再喂一次 `accept_features`，
把最后不满一个 chunk 的尾段解掉。这一步之后不再喂 encoder 以外的任何东西
（可视化用 `spec_view.frame()` 重画）。

---

## 六、可视化与麦克风（落地辅助链路）

### 6.1 麦克风：为什么取「全部」而不是「最新一块」

`get_last_audio()` 取走自上次调用以来累积的**全部**样本。只保留最新一块会让两次
拉取之间的语音出现断口；去直流和预加重会把断口当突变，Mel 谱错乱、吞字。
波形连续无损比单次延迟更重要。

- 回调线程只做 copy + 入队，`deque(maxlen=256)` 约 2.56 s；满了淘汰最旧块并记
  `_dropped`，持续增长说明主循环跟不上；
- `dtype="float32"`、`blocksize=160`、单声道，与模型输入口径一致；
- 主循环主动拉取，不额外起 Python 线程。

### 6.2 语谱图

1. **转置 + 上下翻转**：`(帧数, 80)` → `(80, 帧数)`，Mel 0 落在**底部**。
   OpenCV row 0 在顶部，少了翻转整张图上下颠倒且不报错；
2. **线性拉伸 + 截断**：`clip((值 − lo)/(hi − lo) × 255, 0, 255)`；
3. **色阶标定一次后锁定**：前 50 帧（0.5 s）按 5/99.5 百分位估一次，动态范围
   小于 6.0 时退回固定色阶；每帧重算会让画面「呼吸」；
4. **viridis 伪彩 + 放大**：80 行太矮，先 `cv2.resize` 再 `applyColorMap`；
5. **滚动**：默认 400 列 = 4 秒，新帧贴右、未到的时间在左侧补黑。

纵轴是 Mel 通道序号，**不是线性 Hz**，所以不画频率刻度。`cv2.putText` 只能画
ASCII，中文识别结果以终端为准。

---

## 七、C++ 落地映射

### 7.1 模块 → 编译单元

| Python | C++ 单元 | 边界 |
|---|---|---|
| `AudioPreprocess` | `preprocess.cpp/h` | 只依赖 FFT 库；维护样本缓冲与帧游标 |
| `AsrDecoder` | `decoder.cpp/h` | 只依赖 onnxruntime C API；session 由外部创建 |
| `MicClient` | 独立采集库 `mic_client.cpp/h` | 与 ASR 库零耦合：打开设备、按块输出 PCM；ASR 库只接收 PCM |
| `SpectrogramImage` | 可选 `viz.cpp`（调试用） | 不属于交付库 |

### 7.2 建议的对外库接口

```cpp
// 初始化：传入三个 onnx 路径、词表路径、执行器
AsrEngine::AsrEngine(enc_path, dec_path, joi_path, tokens_path, provider);

// 复位：清零 116 个状态、hyp、decoder_out、样本与特征缓冲
void reset();

// 喂 PCM：任意长度 float32 单声道 16 kHz，内部增量分帧、增量解码
// 有新的确定文本时通过回调/返回值给出
bool feed(const float* pcm, size_t n_samples, std::string* new_text);

// 收尾：补尾帧 + 100 帧 0，解出最后一段
std::string finish();

// 当前完整文本
const std::string& text() const;
```

### 7.3 移植检查清单

- [ ] 状态初值从 ONNX 输入签名推导，不硬编码数量与形状；
- [ ] `blank_id` 从 `tokens.txt` 查 `<blk>`；
- [ ] 解码循环用运行时 `encoder_out.shape[1]`；
- [ ] 流式喂 encoder 判据是 `len(buf) > T`（严格大于）；
- [ ] 116 个状态 + `hyp` + `decoder_out` 跨块存活，`reset()` 之外不清零；
- [ ] 收尾补 100 帧特征域 0（不是静音），且只喂解码器；
- [ ] 前端逐帧、倒序预加重，povey 窗分母 `N−1`；
- [ ] Mel 滤波遍历整数 bin、Mel 域插值、严格不等号；
- [ ] 文本还原保留 CJK 与 ASCII 之间的空格；
- [ ] 可视化路径不补尾帧（若移植可视化）。

### 7.4 性能与依赖

- 三个会话 `intra_op`/`inter_op` 线程数都为 1：chunk 之间本就串行，线程池只增加调度开销；
- int8 模型可直接加载，无需额外量化步骤；
- Python 侧实测：离线 RTF 0.059~0.068、实时 RTF ≈ 0.10、单次喂 encoder 峰值负载 6~9%、
  会话加载 ≈ 1.5 s。C++ 侧应以此为对照基线。

---

## 八、不变量清单（写错不报错，只会静默出错）

1. `T = 61` 是**输入**帧数，时间轴每次只前进 48 帧；实测每 chunk 输出 12 帧。
2. `encoder_out.shape[1]` 必须取运行时值。
3. blank 不刷新 decoder。
4. `blank_id` 从词表查，不硬编码 0。
5. 解码循环尾部补 100 帧**特征域 0**（log 域能量 1）。
6. 流式喂 encoder 的判据是 `len(buf) > T`。
7. 116 个状态张量 + `hyp` + `decoder_out` 跨块存活。
8. 文本还原：CJK 之间去空格、全角标点前后去空格、**CJK 与 ASCII 之间保留**。
9. 可视化路径 `tail_pad_frames=0`。
10. 预加重必须逐帧且倒序；窗函数分母是 `Frame_len−1`；对数地板用 `max` 不用加法。

---

## 九、参数速查

| 类别 | 参数 | 值 |
|---|---|---|
| 前端 | 采样率 / 帧长 / 帧移 | 16000 Hz / 400 点 / 160 点 |
| | FFT 点数 / bin 数 | 512 / 257 |
| | Mel 数 / 频带 | 80 / 20 Hz ~ 8000 Hz |
| | 预加重 / 窗 | 0.97 / povey |
| | 首帧偏移 / 左补齐 | −120 / 120 |
| | 对数地板 | −15.94 |
| | 尾部补帧 | 100（解码）/ 0（可视化） |
| 模型 | 输入帧数 `T` | 61 |
| | 时间轴步进 `decode_chunk_len` | 48 |
| | encoder 输出帧数 | 12 |
| | 状态张量数 | 116 |
| | decoder `context_size` | 2 |
| | 词表大小 | 5000 |
| 解码 | 喂 encoder 判据 | `len(buf) > 61` |
| | 出字策略 | 逐帧 argmax，非 blank 才刷新 decoder |

---

## 十、相关文档

| 文档 | 内容 |
|---|---|
| [工作流报告](sherpa-onnx-ASR预处理及后处理工作流报告.md) | 七项前端差异的完整推导、流式等价性证明、全部实测数据 |
| [部署调研报告](sherpa-onnx-ASR部署调研报告.md) | 调用路径与可行性调研 |
| [README](../README.md) | 成果速览、快速开始、项目结构、C++ 移植计划 |

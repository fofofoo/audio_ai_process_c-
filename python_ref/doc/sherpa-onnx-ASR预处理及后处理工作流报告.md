# sherpa-onnx ASR 预处理及后处理工作流报告

> 项目：audio_ai_process
> 模型：`sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05`
> 日期：2026-10-13
> 前置文档：[sherpa-onnx-ASR部署调研报告.md](sherpa-onnx-ASR部署调研报告.md)（调用路径与可行性调研）

---

## 一、目标与结论

**目标**：不依赖 sherpa-onnx 运行时，用**自研传统预处理模块**（`../src/traditional_features.py`）产出模型输入，用 **onnxruntime** 完成推理与后处理，跑通「离线批量」与「实时流式」两条语音识别链路。

**结论：两条链路都已跑通。** 在权重包自带的四个测试音频上，自研前端与 `kaldi_native_fbank` 参考实现的识别结果**逐 token 完全相同**；流式链路（按 160 样本一块喂）产出的 token 序列与离线批量链路**也逐位相同**。此外 `mini/`（第六节末尾）作为同一件事的第三种独立写法，与 `src/` 同样逐位吻合。

| 指标 | 实测结果 |
|---|---|
| 前端特征偏差（vs `kaldi_native_fbank`） | `mean\|Δ\|` ≈ 4×10⁻⁶，`max\|Δ\|` ≤ 5.4×10⁻⁴ |
| 四条测试音频识别一致性 | **4/4 逐 token 相同** |
| 流式 vs 批量（分帧层 / 特征层） | **逐位相同（max\|Δ\| = 0）** |
| 流式 vs 批量（token 序列） | **4/4 完全一致** |
| `mini/` 独立重写 vs `src/`（前端） | **逐位相同（max\|Δ\| = 0）** |
| `mini/` 独立重写 vs `src/`（token 序列） | **4/4 完全一致** |
| 离线实时率 RTF（单线程 CPU，int8 权重） | 0.059 ~ 0.068 |
| 实时识别 RTF（`mini/` 流式链路） | ≈ 0.10 |
| 流式峰值负载（单次喂 encoder / 喂间隔） | 6 ~ 9% |
| 模型会话加载耗时 | ≈ 1.5 s |
| 端到端运行时依赖 | `onnxruntime` + `numpy` + `soundfile`（**不再需要** `kaldi_native_fbank`） |
| `mini/` 端到端运行时依赖 | `onnxruntime` + `numpy` + `sounddevice` |

也就是说，`kaldi_native_fbank` 在本项目中从此只保留「数值标尺」这一个用途，交付形态不依赖它。

---

## 二、模型契约

权重目录下三个 ONNX 文件与词表：

| 文件 | 大小 | 说明 |
|---|---|---|
| `encoder.int8.onnx` | ≈ 148 MiB | 流式 zipformer2 编码器（int8 量化） |
| `decoder.onnx` | ≈ 10.8 MiB | 预测网络（未量化） |
| `joiner.int8.onnx` | ≈ 2.5 MiB | 联合网络（int8 量化） |
| `tokens.txt` | 5000 行 | `token id` 格式，`<blk> 0`、`<sos/eos> 1` |

张量签名：

| 模型 | 输入 | 输出 |
|---|---|---|
| encoder | `x` float32 `['N', 61, 80]` + **116 个状态张量** | `encoder_out` float32 `['N', ?, 512]` + 116 个新状态 |
| decoder | `y` int64 `['N', 2]` | `decoder_out` float32 `['N', 512]` |
| joiner | `encoder_out` `['N', 512]` + `decoder_out` `['N', 512]` | `logit` float32 `['N', 5000]` |

encoder 的 116 个输入/输出是成对的流式状态（卷积缓存、注意力键值缓存等）。**形状与 dtype 全部可以从输入签名自动推出**，无需硬编码：签名里的字符串维（`'N'`）按 batch=1 取 1，其余取字面值。

encoder 自定义元数据（解码循环的全部关键参数都在这里）：

| 键 | 值 |
|---|---|
| `model_type` | `zipformer2` |
| `T` | 61 |
| `decode_chunk_len` | 48 |
| `version` | 1 |
| `comment` | `streaming zipformer2` |
| `model_author` | `k2-fsa` |
| `cnn_module_kernels` | 31,31,15,15,15,31 |
| `encoder_dims` | 192,256,512,768,512,256 |
| `num_encoder_layers` | 2,2,4,5,4,2 |
| `num_heads` | 4,4,4,8,4,4 |
| `left_context_len` | 256,128,64,32,64,128 |
| `query_head_dims` | 32,32,32,32,32,32 |
| `value_head_dims` | 12,12,12,12,12,12 |

decoder 自定义元数据：

| 键 | 值 |
|---|---|
| `context_size` | 2 |
| `vocab_size` | 5000 |

**`T` 与 `decode_chunk_len` 的关系**：每次向 encoder 喂 61 帧特征，但时间轴只前进 48 帧（480 ms）。多出的 13 帧是卷积模块需要的重叠上下文，输出时会被 encoder 内部裁掉。

---

## 三、预处理：与 kaldi 前端的对齐

### 3.1 基线口径

以 `kaldi_native_fbank` 的输出为 ground truth。它与模型训练时的前端完全一致，参数为：

| 参数 | 值 |
|---|---|
| 采样率 | 16000 Hz |
| 帧长 / 帧移 | 25 ms / 10 ms（400 / 160 采样点） |
| FFT 点数 | 512 → rfft 输出 257 bin |
| Mel 滤波器数 | 80 |
| 频率范围 | low = 20 Hz，high = 0（= Nyquist = 8000 Hz） |
| 窗函数 | `povey` |
| `snip_edges` | `False`（`kaldi_native_fbank` 自身默认是 `True`，官方脚本显式改掉） |
| `remove_dc_offset` | `True` |
| `preemph_coeff` | 0.97 |
| `dither` | 0（推理时必须为 0，否则结果不可复现） |
| 输入量纲 | 保持 `[-1, 1]`，**不**乘 32768 |

### 3.2 差异清单与修复

初版实现在 `hamming` 窗下与 kaldi 的偏差是 `max|Δ| = 10.44`、`mean|Δ| = 0.92`。定位到七项差异，逐项修复：

| # | 差异项 | 原实现 | kaldi 口径 | 实测偏差 |
|---|---|---|---|---|
| 1 | **分帧语义** | `snip_edges=True`，帧数 `(N−W)//S + 1` | `snip_edges=False`，帧数 `round(N/S)`，帧中心钉在 `t·S`，越界部分镜像补齐 | 帧数与对齐位置全不同 |
| 2 | **窗函数** | `hamming` | `povey = (0.5 − 0.5·cos(2πn/(N−1)))^0.85` | — |
| 3 | **窗函数分母** | `N`（scipy 的 `periodic=True`） | `N − 1`（对称窗，首尾都取到零点） | hamming `max\|Δ\| = 5.54×10⁻³` |
| 4 | **Mel 滤波器组** | 在 bin 序号域线性插值 | 遍历整数 FFT bin，在 **Mel 域**插值 | `max\|Δ\| = 0.83` |
| 5 | **边界填充** | `reflect` | `symmetric`（镜像时**重复**边缘样本） | reflect `6.21×10⁻¹` / symmetric `1.37×10⁻⁴` |
| 6 | **对数下限** | `log(energy + 1e−10)` | `log(max(energy, float32_eps))` | 最小值 −21.06 vs **−15.94** |
| 7 | **预加重顺序** | 先对整段波形预加重，再分帧 | 分帧后**逐帧**预加重，帧首样本乘 `(1−α)` | `max\|Δ\| ≈ 5.0` |

其中第 5 项是实测确定的。`reflect` 与 `symmetric` 的差别只在边缘样本是否重复一次，肉眼极难分辨，但把两条路径的最大偏差位置打出来就很清楚：`reflect` 的最差点落在**第 0 帧**（边缘帧），而 `symmetric` 的最差点落在第 117 帧（普通中间帧）——说明前者是边界模型错误，后者只剩浮点残差。

第 6 项的 −15.94 这个数字是 `ln(float32_eps) = ln(1.1920929×10⁻⁷)`。kaldi 用 `max` 而不是「加 eps」，所以静音帧的能量地板停在 −15.94；原实现会在静音处一路跌到 −21 以下，特征分布的形状都不一样。

### 3.3 Mel 滤波器组算法（最关键的一项）

原实现按「先算出浮点 bin 位置、再按 bin 序号插值」来做，与 kaldi 差异 `max|Δ| = 0.83`——这不是精度问题，是算法不同。kaldi 的 `InitKaldiMelBanks` 做法是：

1. 把 `low_freq` / `high_freq` 换到 Mel 刻度，在 Mel 刻度上等分 `n_mels + 1` 段，得到每个滤波器的 `left` / `center` / `right` 三个 Mel 边界；
2. **遍历每一个整数 FFT bin**，把它的中心频率 `i·(fs/N)` 换到 Mel；
3. 若该 Mel 值**严格**落在 `(left, right)` 内，就按它到 `center` 的距离在 **Mel 域**做线性插值。

```python
# src/traditional_features.py :: _build_mel_filterbank
mel_low = hz_to_mel_kaldi(self.low_freq)
mel_high = hz_to_mel_kaldi(high_freq)
mel_delta = (mel_high - mel_low) / (self.n_mels + 1)

# 每个整数 bin 的中心频率先统一换算到 Mel，避免内层循环重复算
bin_mels = hz_to_mel_kaldi(fft_bin_width * np.arange(self.n_freq))

filters = np.zeros((self.n_mels, self.n_freq), dtype=np.float64)
for m in range(self.n_mels):
    left   = mel_low + m * mel_delta
    center = mel_low + (m + 1) * mel_delta
    right  = mel_low + (m + 2) * mel_delta
    inside  = (bin_mels > left) & (bin_mels < right)
    rising  = inside & (bin_mels <= center)
    falling = inside & (bin_mels > center)
    filters[m, rising]  = (bin_mels[rising] - left) / (center - left)
    filters[m, falling] = (right - bin_mels[falling]) / (right - center)
```

两个容易踩的点：

- **必须拿 Mel 值去插值**，不能拿 bin 序号。Hz→Mel 是非线性的，用 bin 序号插值会得到形状不对的三角；
- **边界用严格不等号**（`> left` 且 `< right`）。落在边界上的 bin 权重直接是 0，而不是插值出 0.5。

Mel 刻度公式用 kaldi 的原式 `1127·ln(1 + f/700)`（与教科书的 `2595·log10(1 + f/700)` 数学等价，`2595/ln10 = 1127.01`），避免常数进位带来的末位差异。

改完后与 `knf.MelBanks.get_matrix()` 逐点比对：`max|Δ| = 1.07×10⁻⁵`（纯 float32 精度）。

> 注：`num_fft_bins` 必须取 **257**（`n_fft/2 + 1`），不是 256。

### 3.4 `high_freq` 的口径分歧（未完全验证）

官方 `test_onnx.py` 不设 `mel_opts.high_freq`，于是取 knf 的默认值 **0**（= Nyquist = 8000 Hz）。而 sherpa-onnx C++ 的 `FeatureExtractorConfig` 默认 `high_freq = -400.0`（= Nyquist − 400 = **7600 Hz**），且 zipformer transducer 的实现路径不会覆盖它。

两条口径在本项目里都能与 kaldi 良好对齐（已分别验证），但**识别结果是否一致尚未实测**。目前实现默认取 8000 Hz，与 `test_onnx.py` 保持一致；需要时可改 `TraditionalFeatureExtractor(high_freq=-400.0)`。

### 3.5 对齐结果

四个测试音频，自研前端（默认参数）vs `kaldi_native_fbank`：

| 文件 | 时长 (s) | 采样点 | 特征形状 | `max\|Δ\|` | `mean\|Δ\|` |
|---|---|---|---|---|---|
| `0.wav` | 10.053 | 160850 | (1005, 80) | 1.37×10⁻⁴ | 4.62×10⁻⁶ |
| `1.wav` | 5.100 | 81600 | (510, 80) | 2.99×10⁻⁴ | 4.44×10⁻⁶ |
| `2.wav` | 4.690 | 75040 | (469, 80) | 1.39×10⁻⁴ | 3.89×10⁻⁶ |
| `3.wav` | 8.830 | 141280 | (883, 80) | 5.43×10⁻⁴ | 4.16×10⁻⁶ |

`mean|Δ|` 从最初的 **0.92 降到 4×10⁻⁶**（约 22 万倍）。残余偏差集中在普通中间帧（不是边缘帧），量级在 float32 的表示精度附近，属于纯浮点运算顺序差异。

另外 `logfbank()` 与手工走 `frame_signal(boundary="symmetric") → process_window → power_spectrum → log_mel_spectrogram` 的路径**逐位相同**，说明封装没有引入额外误差。

---

## 四、后处理：onnxruntime 解码链路

实现见 `../src/asr_onnx_decode.py`。整体是一次 transducer 贪心搜索（greedy search），逻辑与官方 `test_onnx.py` 等价，但有四处刻意不同（见第四节末）。

### 4.1 会话与状态张量

三个 ONNX 各开一个 `InferenceSession`，统一 `intra_op_num_threads = 1`、`inter_op_num_threads = 1`、`CPUExecutionProvider`。

116 个状态张量的初值构造**完全由输入签名驱动**：

```python
for n in self.encoder.get_inputs()[1:]:
    dtype = {"tensor(float)": np.float32,
             "tensor(int64)": np.int64,
             "tensor(int32)": np.int32}[n.type]
    shape = [1 if isinstance(s, str) else int(s) for s in n.shape]
    states.append(np.zeros(shape, dtype=dtype))
```

签名里的字符串维（`'N'`）按 batch=1 取 1。这样换了模型也不用改代码。

### 4.2 解码循环

```
hyp = [blank_id] * context_size        # 初始假设：全是 blank
decoder_out = run_decoder(hyp)

start = 0
while start + T < n_feat_frames:
    x = features[start : start + T]    # 取 T=61 帧
    start += decode_chunk_len          # 时间轴前进 48 帧
    encoder_out, states = run_encoder(x, states)

    for k in range(encoder_out.shape[1]):       # 注意：是运行时帧数，不是 T
        joiner_out = run_joiner(encoder_out[:, k:k+1], decoder_out)
        token_id = argmax(joiner_out)
        if token_id != blank_id:
            hyp.append(token_id)
            decoder_out = run_decoder(hyp[-context_size:])   # 用最近 2 个 token 刷新

result = hyp[context_size:]            # 去掉初始的 blank 前缀
```

三个细节：

1. **逐帧喂 joiner，而不是整块。** 每个 encoder 输出帧都要独立过一次 joiner 参与竞争，这是 transducer 的定义。
2. **`encoder_out.shape[1]` 是运行时值。** 元数据里的 `T = 61` 是输入帧数，encoder 输出会被内部卷积模块裁掉一部分，实测每 chunk 输出 12 帧（22 chunk × 12 = 264 帧）。
3. **blank 不刷新 decoder。** 只有产生非 blank token 才重新跑一次 decoder，这是贪心搜索省算力的关键。

`blank_id` 从 `tokens.txt` 里查 `<blk>` / `<blank>` 得到（实测是 0），不硬编码。

### 4.3 尾部补帧

特征在送入解码前，尾部统一补 **100 帧 0**（约 1 秒）：

```python
feats = np.pad(feats, ((0, 100), (0, 0)), mode="constant", constant_values=0.0)
```

原因：解码循环的终止条件是 `while start + T < n_frames`，尾部不补的话，最后一段音频凑不满一个 chunk 会被整段丢掉，尾字识别不出来。

需要注意补的 `0` 是**特征域的 0**（log 域对应能量 1），不是静音值。这是照抄官方做法，不要"自作主张改成静音"。

### 4.4 文本还原

`tokens.txt` 里的 token 用 `▁`（U+2581）标记词首。它同时承担中文单字切分的标记，所以直接 `"".join()` 会得到 `▁昨▁天▁是▁Monday` 这种串。后处理分三步：

```python
text = "".join(tokens).replace("▁", " ")
# CJK 字符之间不留空格（「昨 天」→「昨天」）
text = re.sub(r"(?<=[一-鿿㐀-䶿])\s+(?=[一-鿿㐀-䶿])", "", text)
# 全角标点前不留空格（「Monday ，」→「Monday，」）
text = re.sub(r"\s+(?=[　-〿＀-￯])", "", text)
# 全角标点后不留空格（「， today」→「，today」）
text = re.sub(r"(?<=[　-〿＀-￯])\s+", "", text)
```

CJK 与 ASCII 之间的空格**要保留**（`是 Monday` 不能粘成 `是Monday`）。

### 4.5 与官方 `test_onnx.py` 的四处差异

| | 官方 | 本实现 |
|---|---|---|
| 前端 | 硬编码 `kaldi_native_fbank` | `--frontend ours\|knf\|both`，默认 `both` 互相验证 |
| 依赖 | 在权重包目录内运行、读同目录文件 | 只读三个 onnx 与 `tokens.txt`，路径全部参数化 |
| `blank_id` | 硬编码 `0` | 从 `tokens.txt` 查 |
| 输出 | 只打印文本 | 另加编码耗时与 **RTF（实时率）**，为实时麦克风铺路 |

「不在权重包目录内运行」是安全要求：解释器启动时会从脚本所在目录加载模块，如果在外部下载解压的目录里直接执行，同目录下被投毒的 `json.py` 之流会被 `import` 钩子带进来。

---

## 五、端到端验证结果

`--frontend both` 同时跑自研前端与 `kaldi_native_fbank` 两条链路：

| 文件 | 时长 (s) | chunk 数 | encoder 输出帧数 | ours 耗时 (s) | ours RTF | knf RTF |
|---|---|---|---|---|---|---|
| `0.wav` | 10.053 | 22 | 264 | 0.66 | 0.066 | 0.061 |
| `1.wav` | 5.100 | 12 | 144 | 0.34 | 0.066 | 0.071 |
| `2.wav` | 4.690 | 11 | 132 | 0.28 | 0.059 | 0.067 |
| `3.wav` | 8.830 | 20 | 240 | 0.60 | 0.068 | 0.063 |

识别结果：

```
0.wav  昨天是 Monday，today is 礼拜二，the day after tomorrow 是星期三
1.wav  这是第一种，第二种叫呃与 always always 什么意思啊？
2.wav  就是频繁的啊，不认识记下来，frequently 频繁的
3.wav  第一句是个什么时态？加了 yes 是一般现代时，对，我们把它时态写上去。
```

**两条前端的 token 序列在全部四个文件上逐位相同**（`4/4 一致 ✓`）。也就是说，`mean|Δ| ≈ 4×10⁻⁶` 的前端残差没有在任何一帧上翻转 argmax。

RTF ≈ 0.06 意味着比实时快约 16 倍。当时推断「即使后续加上麦克风采集、实时特征提取，CPU 算力仍有充裕余量」—— 这一点在第六节得到了证实：接上流式链路后，峰值负载也只有 6 ~ 9%。

---

## 六、流式（实时）链路

第五节的链路是离线的：一次性拿到整段波形，特征一次算完、解码一次跑完。实时麦克风做不到这一点 —— 音频每 10 ms 到一块，特征和解码都必须能「接着上次继续」。

新增两个模块解决这件事，都遵循同一条原则：**能复用的绝不重写**。

| 模块 | 职责 | 复用什么 |
|---|---|---|
| `../src/streaming_features.py` | 增量分帧 + 增量 LogFBank | 帧内处理全用 `traditional_features.py` |
| `../src/streaming_asr.py` | 增量解码 + 文本还原 + 麦克风采集 | 会话与 joiner 全用 `asr_onnx_decode.py` |

### 6.1 增量特征提取

批量版 `frame_signal` 是「对整段做一次 `np.pad`、再滑窗取帧」。流式版拿不到「未来」的样本，于是把它拆成三件事，**只接管分帧这一层**：

1. 维护跨块的样本尾巴（一帧 400 点，一块只有 160 点，得攒）；
2. 自己实现两端镜像补齐，补法必须与批量版的 `np.pad` 逐点一致；
3. 丢掉再也用不到的旧样本，否则跑一晚上内存就爆。

帧区间按 kaldi `FirstSampleOfFrame`：

```
第 m 帧的样本区间   lo = m·S + S//2 - W//2 = 160m - 120
                    hi = lo + W            = 160m + 280
```

左边补 120 个点，**只有第 0 帧**用得到（m=1 时 lo 已经是 40 > 0）；右边只在流末尾补。单帧的因果延迟 = `W//2` = 200 样本 = 12.5 ms。

这里埋着一个不明显的坑。批量版 `frame_signal` 有这一行：

```python
mode = boundary if (left < n and right < n) else "edge"
```

因为 numpy 的 `reflect` / `symmetric` 都不允许 pad 宽度超过原数组长度，短信号只能退回 `edge`（左右都复制边缘样本）。代入 kaldi 的补齐公式算出来，它实际只在 **n ≤ 140**（不到 9 ms）时触发。流式版在收尾、样本总数已确定时才判边界模式，用的是同一个判据。

### 6.2 增量解码

批量解码循环是：

```python
start = 0
while start + T < n_frames:            # T = 61
    encoder_out = run_encoder(features[start:start + T])
    start += decode_chunk_len           # 48
    # 逐帧过 joiner，非 blank 就接进假设
```

流式版不知道最终的 `n_frames`，只能一边攒一边喂。关键是**什么时候可以喂** —— 判据必须严格等价，否则尾部会多喂一次 encoder：

```
批量第 k 次喂的条件      48k + 61 < L        （L = 总帧数）
等价于                   L - 48k > 61
而流式喂完第 k 次后剩余   L - 48k 帧

=> 流式判据必须是  len(buf) > T   （严格大于）
```

写成 `>=` 就会在 `L` 恰好等于 `48k + 61` 时多喂一次，尾部多吐出几个 token。代价是麦克风场景下多攒 1 帧（10 ms）才喂。

三样状态必须跨块存活，`reset()` 之外绝不清零：

| 状态 | 作用 |
|---|---|
| encoder 的 116 个状态张量 | 流式编码器「记住前文」的全部机制 |
| 假设序列 `hyp` | 已确定的 token；decoder 要取最近 `context_size` 个当输入 |
| `decoder_out` | 最近一次 decoder 输出；没吐新 token 前不重算，省下大量重复前向 |

### 6.3 验证结果

**分帧层与特征层逐位相同。** 同一条音频，一路走批量、一路按块喂：

| 验证 | 规模 | 结果 |
|---|---|---|
| `streaming_features.py` 自测 | 合成信号 × 6 种块长（160/100/257/400/1000/33） | 分帧与特征均 bit-exact |
| 短信号边界 | n = 120/140/141/200/279/280/281（打 edge 退化分支） | bit-exact |
| `scripts/probe_streaming_features.py` | 4 条真实音频 × 3 种块长 | 分帧与特征均 bit-exact |
| 流式 vs kaldi | 4 条真实音频 | 残差与批量路径对 kaldi 的残差**逐位相同** |

最后一行是关键：流式没有引入任何额外偏差。它同时说明，比较浮点结果时**必须先统一 dtype** —— 流式输出 float32、批量返回 float64，直接相减量到的是 float32 的表示精度（≈ 5×10⁻⁷），与代码是否正确无关。

**增量解码与批量解码的 token 序列完全一致：**

| 文件 | 时长 (s) | 批量 token | 流式 token | 序列一致 |
|---|---|---|---|---|
| `0.wav` | 10.053 | 27 | 27 | ✓ |
| `1.wav` | 5.100 | 20 | 20 | ✓ |
| `2.wav` | 4.690 | 23 | 23 | ✓ |
| `3.wav` | 8.830 | 32 | 32 | ✓ |

块长取 160 / 320 / 1600 样本，结果都一致。

**实时性。** 分母不能取「单块时长」：绝大多数块根本不喂 encoder，真正费时的只有喂的那几次，而两次喂之间隔着 `decode_chunk_len` = 48 帧 = 480 ms 的音频。

| 指标 | 实测 |
|---|---|
| 单次喂 encoder 最坏耗时 | ≈ 40 ms |
| 两次喂之间的音频时长 | 480 ms |
| **峰值负载** | **6 ~ 9%** |
| 整体 RTF | 0.066 ~ 0.085 |

文本是边听边长出来的，不是等音频结束才一次吐出：`3.wav` 全程刷新 12 次，1.59 s 时已出「第一句」，8.83 s 出全句。

### 6.4 麦克风实测

I/O 通路验证（`--mic --seconds 2`），静音环境：

| 指标 | 结果 |
|---|---|
| 默认输入设备 | 麦克风阵列 (Realtek(R) Audio)，编号 1，2 声道 |
| 收块 / 丢弃 | 162 块 / **0 块** |
| 喂 encoder 次数 | 5（2.3 s ÷ 480 ms ≈ 4.8 → 5 ✓） |
| 处理耗时占比 | 5.5% |
| 峰值幅度 | 3.05×10⁻⁵（环境底噪，无语音输入） |

采集线程有两条铁律（`realtime_mic_monitor.py` 里踩过）：回调线程里**只能**做一次 `.copy()` 再塞进缓冲（PortAudio 会复用 `indata` 的底层缓冲，不 copy 会读到脏数据），且消费端跟不上时丢**最旧**的块（`deque(maxlen=...)`）—— 实时识别要的是「现在」，堆积只会让延迟越来越大。

### 6.5 `mini/`：同一件事的第三种写法

`src/` 的流式实现是为生产写的：分层抽象、跨模块复用（`StreamingFramer` 只接管分帧，帧内处理全部复用批量版）。它跑得对，但读者要在多个文件间来回跳才能拼出「一帧音频到一段文字」的完整路径，不利于学习与 C++ 移植。

`mini/` 因此另写了一套：三个类零耦合（`AudioPreprocess` / `AsrDecoder` / `MicClient`），**不 import `src/` 任何模块**；前端与推理全部写成显式逐帧循环（去直流 → 逐帧倒序预加重 → 加窗 → FFT → 功率谱 → Mel 滤波 → log），不用 Qt、不起 Python 线程、不引入 scipy（窗函数手算）。复刻的是同样的 kaldi 口径与解码不变量，但代码组织完全不同。

它同时是第三把数值标尺 —— 独立重写、且不共享任何一行代码，与 `src/` 一致才能说明两份实现都没写错。`scripts/verify_mini.py` 逐点对拍：

| 对拍项 | 覆盖范围 | 结果 |
|---|---|---|
| 前端 vs `frontend_for_onnx` | 4 段真实音频 × 块长 160/257/400/1000/一次全喂；短信号 n = 0/60/79/80/120/139/140/141/280/281（含 `edge` 退化分支） | `max\|Δ\| = 0`，**逐位相同** |
| 解码 token 序列 vs 批量 | 4 段音频 × 每次喂 1/7/37 帧/一次全喂 | **完全一致**，chunk 数与 encoder 帧数也逐项相同 |
| 实时麦克风 | RTF ≈ 0.10 | 识别正确，丢块 0 |

逐位相同是能解释的：两边在实现细节上唯一的差异，只有 Mel 能量的求和次序（`power @ filters.T` 走 BLAS，mini 用显式稀疏累加）与功率谱的算法（`np.abs(z)**2` vs `real² + imag²`）。这些都只差最后几个 bit，在 float64 下实测为 0。

**一处刻意的反向取舍**：`MicClient.get_last_audio()` 取走的是自上次调用以来累积的**全部**样本，与 6.4 节 `streaming_asr.py` 的「丢最旧块」相反。两边目标不同 —— GUI 那条线要画面低延迟，mini 要波形连续无损。只保留最新一块会让两次拉取之间的 30 ms 语音出现断口，去直流与预加重会把断口当突变，Mel 谱错乱、吞字。

---

## 七、关键文件与用法

| 文件 | 职责 |
|---|---|
| `../src/traditional_features.py` | 自研前端。默认参数已对齐 kaldi；`frontend_for_onnx()` 是直通模型的工作流入口 |
| `../src/asr_onnx_decode.py` | 端到端解码：前端 + 会话管理 + 贪心搜索 + 文本还原 |
| `../src/streaming_features.py` | 增量特征：`StreamingFramer`（只分帧）+ `StreamingLogFBank`（分帧 + 帧内处理） |
| `../src/streaming_asr.py` | 流式解码：`StreamingAsr`（增量解码）+ 离线对拍 + 实时麦克风 |
| `../src/live_audio_view.py` | 实时可视化控件：波形 / RMS / 滚动语谱图三块图 |
| `../src/console_utf8.py` | 把 stdout/stderr 切成 UTF-8，所有带中文输出的脚本共用 |
| `../scripts/probe_knf_reference.py` | 读 ONNX 元数据；导出 knf 参考特征 |
| `../scripts/probe_knf_components.py` | 逐组件比对（窗函数 / Mel 矩阵 / Mel 刻度） |
| `../scripts/probe_mel_iterate.py` | 用 numpy 复现 kaldi Mel 算法并迭代到一致 |
| `../scripts/probe_align_features.py` | 端到端特征对齐验证（含边界模式 A/B 对比） |
| `../scripts/probe_streaming_features.py` | 流式特征对齐验证（真实音频，分帧层与特征层 bit-exact） |
| `../scripts/smoke_test_spectrogram.py` | 离线链路冒烟：offscreen 跑完整「播放 + 三图 + 识别」，并统计帧间隔 |
| `../scripts/smoke_test_mic_audio.py` | 麦克风链路冒烟：假 `InputStream` 按 1× 速度回放真实 wav，`realtime_mic_monitor.py` 一行不改地跑完 |
| `../scripts/smoke_test_mic_monitor.py` | 麦克风链路冒烟：真开麦克风跑 N 秒后自动退出（验证设备与回调线程） |
| `../scripts/verify_mini.py` | `mini/` 与 `src/` 逐点对拍（前端特征 + token 序列；`--part a\|b\|ab`） |
| `../decode_result.txt` | 最近一次解码输出（UTF-8） |

上面是核心库，都在 `src/` 下；命令行入口仍是 `src/` 里的模块（`python -I src/xxx.py`）。

`mini/` 是与 `src/` 平行的**另一套自包含实现**（见 6.5 节），两者互不引用 —— `mini/` 不 import `src/`，`src/` 也不知道 `mini/` 的存在：

| 文件 | 职责 |
|---|---|
| `../mini/audio_preprocess.py` | `AudioPreprocess`：波形块 → LogFBank 特征（显式逐帧循环） |
| `../mini/asr_decode.py` | `AsrDecoder`：特征 → 文本（三个 session + 贪心搜索 + 文本还原） |
| `../mini/mic_client.py` | `MicClient`：回调采集 + `get_last_audio()` 拉取累积样本 |

项目根目录留三个入口，它们各自把 `src/` 或 `mini/` 加进 `sys.path` 后再 import：

| 根目录入口 | 职责 |
|---|---|
| `../test_spectrogram.py` | 离线：播一条 aishell1 样本，边放边显示三块图 + 流式识别文字 |
| `../realtime_mic_monitor.py` | 实时：麦克风采音，三块图 + 流式识别文字同屏 |
| `../test_mic_asr_mini.py` | 实时：`mini/` 那套实现的纯终端入口（不用 Qt，全程 `print`） |

运行方式（权重路径按需替换）：

```bash
# ---- 两个 GUI 入口（根目录）----
python test_spectrogram.py            # 离线样本回放 + 实时识别
python realtime_mic_monitor.py        # 实时麦克风 + 实时识别

# ---- 核心库的命令行入口 ----
# 离线端到端解码，两条前端互验
python -I src/asr_onnx_decode.py \
  --model-dir <权重目录> \
  --frontend both \
  --out result.txt

# 只跑自研前端
python -I src/asr_onnx_decode.py --model-dir <权重目录> --frontend ours

# 指定音频
python -I src/asr_onnx_decode.py --model-dir <权重目录> --wav a.wav b.wav

# 流式特征对齐验证
python -I scripts/probe_streaming_features.py <权重目录>/test_wavs/*.wav

# 流式解码：按块喂 wav，与批量结果对拍
python -I src/streaming_asr.py --model-dir <权重目录> --batch-check --chunk 160 320 1600

# 实时麦克风（Ctrl+C 结束；--seconds N 可限时自动结束）
python -I src/streaming_asr.py --model-dir <权重目录> --mic
python -I src/streaming_asr.py --model-dir <权重目录> --list-devices

# ---- mini/ 那套实现（自包含，不 import src/）----
python -I test_mic_asr_mini.py                  # 实时麦克风，纯 print，Ctrl+C 结束
python -I test_mic_asr_mini.py --seconds 8      # 跑 8 秒自动退出
python -I test_mic_asr_mini.py --list-devices   # 列出音频设备

# ---- 冒烟测试（回归用，不需要人工看界面）----
python -I scripts/smoke_test_spectrogram.py     # 离线链路 + 帧间隔统计
python -I scripts/smoke_test_mic_audio.py       # 麦克风链路，假音频回放
python -I scripts/smoke_test_mic_monitor.py     # 麦克风链路，真开麦克风
python -I scripts/verify_mini.py                # mini/ 与 src/ 逐点对拍
```

不指定 `--wav` 时自动跑 `<权重目录>/test_wavs/*.wav`。

`frontend_for_onnx()` 的单独用法：

```python
from traditional_features import TraditionalFeatureExtractor

ext = TraditionalFeatureExtractor(sample_rate=16000)   # 默认即 kaldi 口径
feats = ext.frontend_for_onnx(samples)                 # float32 (n_frames+100, 80)
```

流式用法（调用节奏与 `kaldi_native_fbank.OnlineFbank` 一致）：

```python
from streaming_features import StreamingLogFBank

fbank = StreamingLogFBank(sample_rate=16000, tail_pad_frames=100)
while 还有音频:
    feats = fbank.accept_waveform(chunk)   # (n_new, 80) float32，可能 0 帧
feats = fbank.input_finished()             # 收尾：补完尾部帧 + 补 100 帧 0
```

```python
from streaming_asr import StreamingAsr

asr = StreamingAsr("<权重目录>")
while 还有音频:
    text = asr.accept_audio(chunk)         # 返回当前完整识别文本
text = asr.input_finished()                # 收尾，返回最终文本
```

---

## 八、遗留问题与后续工作

### 8.1 已知问题

1. **`high_freq` 口径分歧未实测**（见 3.4 节）。8000 Hz 与 7600 Hz 两条路径都能对齐 kaldi，但识别结果是否会分叉没有验证。

2. **`dither` 语义**。本模块的 `dither` 是高斯噪声的标准差、直接作用于输入量纲。`kaldi_native_fbank` 的默认值是 `2.9999999e-05`（float32 的 3e-5，该值假设输入是 16-bit 整数域，对归一化到 `[-1, 1]` 的信号几乎不起作用）。推理一律用 0 —— 非 0 时结果不可复现，流式版更是直接拒绝 `dither != 0`。

3. **`sounddevice` 的设备编号与 `query_devices()` 打印的表对不上**。`--device 12` 会被拒（`No input device matching '12'`），尽管表里 12 号确实有 2 个输入通道。默认路径不受影响，暂未深究。

4. ~~**`live_audio_view.py` 的 MFCC 列失效**~~ —— **已解决**：MFCC 可视化整块删除（该列本来就算错了，为旧的 `snip_edges=True` 语义所写），文件从 655 行缩到 428 行，两个调用方从未启用过它，不受影响。

### 8.2 后续工作

1. ~~流式增量特征提取器~~ —— **已完成**，见第六节，与批量路径逐位相同。
2. ~~实时麦克风 ASR~~ —— **已完成**，见第六节。算力余量充裕（峰值负载 6~9%）。
3. **基准测试**：对 `data/` 下的 aishell1 子集批量解码，与标注比 WER，出具准确率与 RTF 基线。
4. 可选：VAD / 端点检测，避免静音段空转。
5. 可选：把实时识别结果接进 `live_audio_view.py`，波形 / RMS / 语谱图与识别文字同屏。

---

## 附录：前端参数速查

| 参数 | 默认值 | 备注 |
|---|---|---|
| `sample_rate` | 16000 | |
| `frame_len` / `frame_shift` | 0.025 / 0.01 秒 | 实际取整用 kaldi 的**截断**语义（`int(sr × ms / 1000)`），非四舍五入 |
| `n_fft` | 512 | → `n_freq = 257` |
| `n_mels` | 80 | |
| `n_mfcc` | 13 | |
| `preemph` | 0.97 | 逐帧执行 |
| `window` | `povey` | 支持 povey / hamming / hanning / hann / sine / blackman / rectangular |
| `lifter` | 22 | MFCC 用 |
| `log_eps` | `None` | `None` → kaldi 口径 `log(max(e, float32_eps))`；给数值 → `log(e + log_eps)` |
| `low_freq` / `high_freq` | 20.0 / 0.0 | `high_freq = 0` 表示取到 Nyquist；负数表示 Nyquist + 该值 |
| `snip_edges` | `False` | |
| `remove_dc_offset` | `True` | 逐帧减均值 |
| `dither` | 0.0 | 推理必须为 0 |
| `frame_preemphasis` | `True` | `True` → 分帧后逐帧预加重（kaldi）；`False` → 整段先预加重（教科书写法） |
| `boundary` | `symmetric` | 帧越界补齐方式，kaldi 为 `symmetric` |

工作流方法：

| 方法 | 输出 |
|---|---|
| `logfbank(signal)` | (n_frames, 80) |
| `logfbank(signal, with_delta=True, do_normalize=True)` | (n_frames, 240) |
| `mfcc(signal, with_delta=False, do_normalize=False)` | (n_frames, 13) |
| `mfcc(signal)` | (n_frames, 39) |
| `frontend_for_onnx(signal)` | float32 (n_frames + 100, 80)，直通模型 |

中间步骤方法（可单独调用）：`pre_emphasis` / `frame_and_window`（教科书写法）、`num_frames` / `frame_signal` / `process_window`（kaldi 口径）、`power_spectrum` / `log_mel_spectrogram` / `dct` / `dynamic_features` / `normalize`。

流式模块 API：

| 类 / 方法 | 说明 |
|---|---|
| `StreamingFramer(ext)` | 只管分帧。`accept_waveform(块)` → `(n_new, 400)` 原始帧；`input_finished()` 收尾 |
| `StreamingLogFBank(extractor=…, tail_pad_frames=0)` | 分帧 + 帧内处理。`accept_waveform(块)` → `(n_new, 80)` float32 |
| `StreamingAsr(model_dir)` | 流式解码。`accept_audio(块)` → 当前完整文本；`input_finished()` → 最终文本 |
| `StreamingAsr.reset()` | 清空全部跨块状态，回到「刚开始听」 |
| `StreamingAsr.token_ids` / `.text` / `.n_chunks` | 已确定的 token / 当前文本 / 已喂 encoder 次数 |
| `StreamingAsr.batch_reference(samples)` | 用同一套权重跑批量解码，供对拍用（**会打乱流式状态**，之后须 `reset()`） |

流式模块只接受 `frame_preemphasis=True` + `snip_edges=False` + `dither=0`（即推理口径），其余组合直接抛异常并说明原因。

# sherpa-onnx ASR 模型部署调研报告

> 调研日期：2026-10-13
> 调研对象：`sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05`
> 目标：找到官方 ASR 模型的调用方式，产出 Python（onnxruntime）与 C++ 两条部署路径，并为「用自研传统预处理模块 + onnxruntime 后处理」这条路线做可行性评估。

---

## 一、结论先行

官方调用方式分**三个层次**，与本次目标的关系如下：

| 层次 | 位置 | 用途 | 对本项目的价值 |
|---|---|---|---|
| ① 纯 onnxruntime 参考实现 | `../weight/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05/test_onnx.py`（7.1 KB） | 官方随权重下发的、不依赖 sherpa-onnx 库的完整前后处理 | **核心资产**，自研路线的直接蓝本 |
| ② Python 封装 API | `../sharpa-onnx/sherpa-onnx/python/sherpa_onnx/` | 一行 `OnlineRecognizer.from_transducer(...)` 完成全部流程 | 快速验证、作为结果基准 |
| ③ C++ / C API | `../sharpa-onnx/sherpa-onnx/csrc/` | 端侧部署主力 | 后续移植 Jetson 用 |

**一句话**：`test_onnx.py` 就是「用 onnxruntime 直接跑」的官方答案，sherpa-onnx 库本身是可选的。

---

## 二、官方示例索引（Python / C++ / C）

### 2.1 Python

| 文件 | 说明 |
|---|---|
| `../sharpa-onnx/python-api-examples/online-decode-files.py` | **流式 transducer 文件解码**，官方 API 的标准用法 |
| `../sharpa-onnx/python-api-examples/streaming-paraformer-asr-microphone.py` | **流式 + 麦克风**，结构上最接近本项目需求 |
| `../sharpa-onnx/python-api-examples/speech-recognition-from-microphone.py` | 麦克风（VAD 分段、非严格流式） |
| `../sharpa-onnx/python-api-examples/speech-recognition-from-microphone-with-endpoint-detection.py` | 麦克风 + 端点检测 |
| `../sharpa-onnx/python-api-examples/simulate-streaming-*.py` | 用文件模拟流式喂数据，便于离线调试 |

标准用法（`online-decode-files.py:342`）：

```python
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=..., encoder=..., decoder=..., joiner=...,
    num_threads=1, provider="cpu",
    sample_rate=16000, feature_dim=80,
    decoding_method="greedy_search", max_active_paths=4,
)
s = recognizer.create_stream()
s.accept_waveform(sample_rate, samples)          # samples 为 [-1,1] float32
s.accept_waveform(sample_rate, tail_paddings)    # 尾部补 0.66s 静音
s.input_finished()
while recognizer.is_ready(s):
    recognizer.decode_streams([s])
text = recognizer.get_result(s)
```

### 2.2 C++

| 文件 | 说明 |
|---|---|
| `../sharpa-onnx/cxx-api-examples/streaming-zipformer-cxx-api.cc` | 流式 zipformer 文件解码 |
| `../sharpa-onnx/cxx-api-examples/zipformer-transducer-simulate-streaming-microphone-cxx-api.cc` | **流式 zipformer2 transducer + 模拟麦克风**，与本模型完全对应 |
| `../sharpa-onnx/cxx-api-examples/streaming-zipformer-rtf-cxx-api.cc` | 带 RTF 测速，可参考其基准测试写法 |

### 2.3 C

| 文件 | 说明 |
|---|---|
| `../sharpa-onnx/c-api-examples/streaming-zipformer-c-api.c` | C 接口流式解码 |
| `../sharpa-onnx/c-api-examples/asr-microphone-example/` | 麦克风完整工程 |

---

## 三、模型契约（必须逐位对齐的部分）

以下配置提取自 `test_onnx.py` 与 `../sharpa-onnx/sherpa-onnx/csrc/features.h`，是该模型期望的**精确输入规格**：

| 项 | 值 | 来源 |
|---|---|---|
| 采样率 | 16000 Hz | features.h:19 |
| 特征维度 | 80 mel | features.h:25 |
| 帧长 / 帧移 | 25 ms / 10 ms | features.h:56-57 |
| **snip_edges** | **false** | features.h:55 |
| **窗函数** | **`povey`**（非 hamming） | features.h:61 |
| **normalize_samples** | **true** → 波形保持 [-1,1]，**不乘 32768** | features.h:53 |
| dither | 0（关闭） | features.h:46 |
| remove_dc_offset | true（逐帧去均值） | features.h:59 |
| preemph_coeff | 0.97 | features.h:60 |
| low_freq | 20 Hz | features.h:28 |
| high_freq | Nyquist − 400 = 7600 Hz | features.h:38 |
| 对数 | 自然对数，mel 能量 floor 到 float epsilon | kaldi 标准行为 |

### 3.1 关于 normalize_samples 的重要澄清

`features.h:53` 中 `bool normalize_samples = true;`——**默认是 true**。

在整个 `csrc/` 目录中检索 `normalize_samples`，被显式设为 `false` 的只有以下模型，**zipformer transducer 不在其中**：

- `offline-recognizer-paraformer-impl.h:214` / `online-recognizer-paraformer-impl.h:231`
- `offline-recognizer-ctc-impl.h:110` / `online-recognizer-ctc-impl.h:261`（WeNet）
- `offline-recognizer-fire-red-asr-impl.h:84`
- `offline-recognizer-funasr-nano-impl.cc:228`
- SenseVoice 系列（从 metadata 读）

对应逻辑（`features.cc:73-83`）：

```cpp
void AcceptWaveform(int32_t sampling_rate, const float *waveform, int32_t n) {
  if (config_.normalize_samples) {
    AcceptWaveformImpl(sampling_rate, waveform, n);      // 直接用 [-1,1]
  } else {
    std::vector<float> buf(n);
    for (int32_t i = 0; i != n; ++i) buf[i] = waveform[i] * 32768;   // 放大
    AcceptWaveformImpl(sampling_rate, buf.data(), n);
  }
}
```

**结论**：本模型输入为 **[-1,1] 归一化波形，不做 ×32768 放大**。这与 `test_onnx.py` 直接喂 soundfile 读出的 [-1,1] 样本自洽。

### 3.2 一个待实测的疑点

`test_onnx.py` 使用的是 `knf.FbankOptions()` 的**默认值**，而 kaldi-native-fbank 中 `high_freq` 默认为 `0`（= Nyquist，即 8000 Hz）；sherpa-onnx 自身的 `FeatureExtractorConfig` 默认是 `-400`（= 7600 Hz）。

两者不一致。既然 `test_onnx.py` 是随权重下发且能跑通的版本，说明模型对该频带边界不敏感，但**这也意味着必须以实测为准，不能靠推断**。

---

## 四、模型元数据与解码循环

### 4.1 内嵌元数据

`test_onnx.py` 从 ONNX 文件的 `custom_metadata_map` 中读出四个关键量：

```
encoder.int8.onnx  →  model_type = "zipformer2"   （代码中 assert）
                      decode_chunk_len            ← 每步推进的特征帧数
                      T                           ← 每次喂给 encoder 的特征帧数
decoder.onnx       →  context_size                ← 解码器历史长度
                      vocab_size
```

> 已用字符串检索确认键名存在：`encoder.int8.onnx` 内含 `model_type` / `decode_chunk_len` / `T` / `version` / `model_author` / `comment`；`decoder.onnx` 内含 `context_size` / `vocab_size`。
> **具体数值需运行 onnxruntime 读取**（值以 varint 存储，非 ASCII，无法从二进制直接提取）。

### 4.2 特征侧的两处补丁

```python
# 1) 尾部补 100 帧 0（约 1 秒静音），保证最后一个 chunk 能跑完
features = np.pad(features, ((0, 100), (0, 0)), mode="constant", constant_values=0)
```

### 4.3 贪心搜索主循环

```python
blank_id = 0
hyp = [blank_id] * context_size
decoder_out = run_decoder(hyp)

start = 0
while start + T < features.shape[0]:
    x = features[start : start + T][None]
    start += decode_chunk_len              # 注意：T 与 decode_chunk_len 并非同一量
    encoder_out, states = run_encoder(x, states)
    for k in range(encoder_out.shape[1]):  # 逐帧贪心
        cur = encoder_out[0, k : k + 1]
        joiner_out = run_joiner(cur, decoder_out)
        token_id = joiner_out.argmax()
        if token_id != blank_id:
            hyp.append(token_id)
            decoder_out = run_decoder(hyp[-context_size:])

tokens = [id2token[i] for i in hyp[context_size:]]
```

### 4.4 其他参考实现细节

- `get_encoder_states()`：依据 `encoder.get_inputs()[1:]` 的 shape 构造全零初始状态
- `load_tokens()`：解析 `tokens.txt`（每行 `token id`，共 5000 行；id 0 = `<blk>`，id 1 = `<sos/eos>`）
- `load_audio()`：soundfile 读取，`always_2d=True`，取第 0 通道；非 16000 Hz 时用 librosa 重采样
- **`bpe.model` 在 `test_onnx.py` 中未被使用**——纯贪心搜索不需要它，仅热词（hotwords）场景才需要

---

## 五、目标路线可行性评估

目标路线：**自研传统预处理模块（`../src/traditional_features.py`）产出模型输入 → onnxruntime 完成推理与后处理 → 实时麦克风 ASR + 基准测试**。

结论：**可行**，但存在必须先解决的分帧语义硬伤。差异清单按严重程度排序：

| # | 差异项 | 现状（traditional_features.py） | 模型要求 | 严重度 |
|---|---|---|---|---|
| 1 | **分帧语义** | `sliding_window_view(signal, frame_size)[::frame_step]` → 帧数 `(N-W)//S+1`，即 **snip_edges=True** | **snip_edges=False**：帧数 = `round(N/S)`，帧中心落在 `t·S`，窗两端靠**镜像反射**填充 | 🔴 帧数与对齐位置全不同 |
| 2 | **窗函数** | hamming（默认） | **povey** = `(0.5 − 0.5·cos(2πn/(N−1)))^0.85` | 🔴 逐点数值不同 |
| 3 | **Mel 滤波器组** | 自定义：浮点 bin 位置 + `ceil(left)..floor(right)` | kaldi `MelBanks`：mel 刻度 `1127·ln(1+f/700)`，逐 fft bin 按 `(mel−left)/(center−left)` 计算浮点权重；且 high_freq=7600 而非 8000 | 🔴 每个 mel bin 均有偏差 |
| 4 | **预加重作用域** | 对整段信号做一次 `pre_emphasis` | **逐帧**处理，且帧首样本需引用「帧起点前一个样本」，边界需反射处理 | 🟡 |
| 5 | **去直流** | 无 | 每帧 `x -= mean(x)` | 🟡 |
| 6 | 对数底与保护 | `log_eps = 1e-10` | float epsilon（≈1.19e-7），自然对数 | 🟡 |

> 第 3、4 项所描述的 kaldi 行为基于 kaldi fbank 的标准实现。kaldi-native-fbank 的源码**不在当前代码树中**（`../sharpa-onnx/cmake/kaldi-native-fbank.cmake` 仅为拉取脚本），无法逐行核对，因此这两项**必须通过实测比对定标**，不能仅凭推断。

---

## 六、当前环境现状

```
onnxruntime          1.23.0   ✅ 已安装
numpy                2.1.1    ✅ 已安装
kaldi_native_fbank            ❌ 未安装（纯 Python wheel，pip 可装）
sherpa_onnx                   ❌ 未安装
soundfile / librosa           ❌ 未安装
```

Python 解释器（py310-gpu）：
`C:\Users\Liyun\anaconda3\envs\py310-gpu\python.exe`

---

## 七、建议的推进方案

**核心思路**：先用 `kaldi_native_fbank` 建立一根**数值标尺**，再用它把 `traditional_features.py` 逼近到逐位对齐。对齐不了就量化报告偏差，而不是凭感觉判断「应该能跑」。

| 路线 | 组成 | 定位 |
|---|---|---|
| **C（标尺，先做）** | `kaldi_native_fbank` + `onnxruntime` | 即 `test_onnx.py` 的架构，立即拿到可用的 baseline 特征与识别结果 |
| **B（目标，后做）** | `traditional_features.py` → 特征 → 与 C 逐帧比对 `max｜Δ｜` → 对齐 → 接同一套 onnxruntime 解码 → 实时麦克风 + 基准测试 | 最终交付形态 |

有 C 做标尺，B 的每一步偏差都是可量化的数字。

### 分步计划

1. **环境准备**：`pip install kaldi_native_fbank soundfile`（装入 py310-gpu）
2. **读取元数据 + 建立基准**
   - ⚠️ `test_onnx.py` 属外部下载包解压产物，按不可信数据处理原则，**不在其所在目录直接执行**（解释器会从脚本目录加载代码，等同于执行外部代码、存在被投毒风险）
   - 做法：将其逻辑**重写**到自建目录下的新脚本，以 `-I` 隔离模式运行，权重路径以参数传入
   - 产出：`T` / `decode_chunk_len` / `context_size` / `vocab_size` 实际值；`test_wavs/` 的识别结果（作为后续所有路线的一致性基准）
3. **特征对齐**：以 `kaldi_native_fbank` 输出为 ground truth，逐帧比对 `traditional_features.py`，输出各维 `max|Δ|` 与相对误差；据此在 `traditional_features.py` 中**新增 kaldi 兼容模式**（而非覆盖现有实现——现有实现仍服务于语谱图 / MFCC 可视化）
4. **实时 ASR + 基准测试**：在第 3 步对齐通过后进行

---

## 八、关键文件索引

| 路径 | 说明 |
|---|---|
| `../weight/sherpa-onnx-x-asr-480ms-.../test_onnx.py` | 纯 onnxruntime 参考实现（核心） |
| `../weight/sherpa-onnx-x-asr-480ms-.../encoder.int8.onnx` | 编码器（149 MB） |
| `../weight/sherpa-onnx-x-asr-480ms-.../decoder.onnx` | 解码器（11 MB） |
| `../weight/sherpa-onnx-x-asr-480ms-.../joiner.int8.onnx` | 联合网络（2.5 MB） |
| `../weight/sherpa-onnx-x-asr-480ms-.../tokens.txt` | 词表（5000 行） |
| `../weight/sherpa-onnx-x-asr-480ms-.../bpe.model` | BPE 模型（贪心搜索下未使用） |
| `../sharpa-onnx/sherpa-onnx/csrc/features.h` | 特征提取配置与默认值 |
| `../sharpa-onnx/sherpa-onnx/csrc/features.cc` | 特征提取实现（含 normalize_samples 分支） |
| `../sharpa-onnx/python-api-examples/online-decode-files.py` | Python API 标准用法 |
| `../sharpa-onnx/cxx-api-examples/streaming-zipformer-cxx-api.cc` | C++ 流式示例 |
| `../src/traditional_features.py` | 自研传统预处理模块（待对齐） |

---

## 附：备注

- 权重包内的 `README.md`（96 字节）内容为：`Files in this directory are from https://github.com/Gilgamesh-J/X-ASR/tree/main`。即该权重包为第三方基于 k2-fsa/sherpa-onnx 格式打包，`test_onnx.py` 亦出自该仓库。
- 用户所述源码目录实际名称为 **`sharpa-onnx`**（多一个字母 `a`），非官方仓库名 `sherpa-onnx`。

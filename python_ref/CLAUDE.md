# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言与协作约定

- 始终用中文交流；代码注释、docstring 沿用项目现有的中文风格（README 与 doc/ 均为中文）。
- **执行脚本前必须先征得用户同意**。写验证脚本、跑解码/冒烟测试/数值一致性比对之前，先给出打算执行的内容与目的，用户明确同意后再动手。

## 项目是什么

中文流式语音识别（streaming zipformer2 transducer，int8）的**纯 Python 复现** —— mini 版。

这个分支只做底层 ASR：预处理用自研的传统前端（`numpy`）产出模型输入，推理与后处理用
`onnxruntime`，另一个 OpenCV 窗口把同一份特征画成滚动语谱图。不依赖 sherpa-onnx 运行时，
不用 Qt，也不自己起 Python 线程（麦克风走 PortAudio 回调，主循环主动拉取）。

模型：`sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05`

运行时依赖：`onnxruntime` + `numpy` + `soundfile` + `sounddevice` + `opencv-python`。
`kaldi_native_fbank` 只在开发期充当「数值标尺」，不是运行时依赖。

环境：Python 3.10，conda 环境 `py310-gpu`。跑脚本时写全该环境的解释器路径
（`C:/Users/Liyun/anaconda3/envs/py310-gpu/python.exe`），裸 `python` 会落到 base 且缺依赖。

## 项目方向：转 C++ 部署

最终交付形态是**两个 C/C++ 库**：ASR 处理库（库内完成前端 → onnxruntime 推理 → 文本还原）
和麦克风采集库（PortAudio）。两者零耦合，ASR 库只接收 PCM；麦克风客户端源码也纳入项目，
因此 wav 与 mic 两条链路可以同时验证。Python 的 `src/` 是**参考实现**，不是部署产物。
路线分三步，每步独立验证：

1. **wav 前端 + 可视化**：用 C++ 复刻 `test_spectrogram_cv_mini.py` 的链路
   （读 wav、显式逐帧预处理、OpenCV 滚动语谱图），先不接模型；
   特征与 `src/audio_preprocess.py` 逐点对拍。
2. **wav ASR**：链接 onnxruntime C++ 库，移植 `src/asr_decode.py` 的贪心搜索与文本还原，
   与 Python 版逐 token 对拍。
3. **拆成两个库**：ASR 处理库（初始化 / reset / 喂 PCM / 取文本，内部维护跨块状态并增量
   出字）+ 麦克风采集库（打开设备、按块输出 PCM）；两库零耦合，wav 与 mic 一起验收。

写新代码时优先考虑可移植性：显式逐帧循环、不依赖 Python 专有行为、不引入额外框架；
下面「必须记住的不变量」在 C++ 版里同样成立。

## 常用命令

```bash
# 实时麦克风：OpenCV 滚动语谱图 + 识别文字
python -I test_mic_asr_mini.py

# wav 识别（与上者同构，输入换成文件）
python -I test_wav_asr_mini.py

# 只看 wav 的滚动语谱图（不加载 onnx）
python -I test_spectrogram_cv_mini.py

# 打包（不含 weight/、sharpa-onnx/、wav 等被 .gitignore 排除的项）
git archive --format=zip --output=audio_ai_process.zip HEAD
```

不收命令行参数：onnx 与词表路径、执行器、麦克风设备都硬编码在入口文件顶部。
没有测试框架，验证方式就是跑上面三个入口；只想确认链路正常，跑 `test_wav_asr_mini.py`。

## 架构

```
入口脚本（装配 + 数据源）
    │
    ├─ AudioPreprocess.accept_waveform(blk) ──> feats (n, n_mels)
    │                                               │
    ├─ AsrDecoder.accept_features(feats) ───────────┴──> text
    └─ SpectrogramImage.accept_features(feats) ─────────> BGR 画布
```

三个类只靠特征矩阵相连，互不引用；入口脚本只负责「配置 + 数据源」。

| 模块 | 职责 |
|---|---|
| `src/audio_preprocess.py` | 显式逐帧的自研前端：分帧、逐帧预加重、povey 窗、FFT、Mel 滤波、log。维护跨块样本尾巴 |
| `src/asr_decode.py` | 三个 session 的贪心搜索 + 文本还原；`accept_features()` 维护特征缓冲、encoder 状态、`hyp`、`decoder_out` |
| `src/mic_client.py` | PortAudio 回调采集；`get_last_audio()` 取走自上次调用以来累积的**全部**样本 |
| `src/spectrogram_image.py` | 特征 → BGR 滚动语谱图画布（viridis），自带特征缓存与色阶标定 |

代码组织是刻意的：显式逐帧循环、三类零耦合、不引入 scipy（窗函数手算）、注释按最密写。
目标是能逐行对照移植到 C++。

### 模型结构速记

- `encoder_embed` **不是**「两层 1D 卷积压 4 倍」：它是 3 层 2D 卷积（1→8→32→128）
  + ConvNeXt 块，时间只压约 2 倍（61→27），频率 80→19，最后投影到 192 维。
- 6 个 stack（192/256/512/768/512/256）是 U-Net 式「先压缩再恢复」，整体再压 2 倍，
  所以 48 帧输入对应 12 帧输出。512 是第 3 个 stack 的宽度，也是 `encoder_out` 维度。
- 116 个状态 = 2 个全局 + 19×4 注意力缓存（key / value / value2 / non-linear attn）
  + 19×2 卷积缓存。
- 结构、逐层形状与输入输出契约见 [落地报告](doc/src-预处理与模型框架-ASR落地报告.md)。

## 必须记住的不变量

这几条写错了不会报错，只会静默多吐出几个 token 或尾字丢失：

- **`T = 61` 是输入帧数，不是输出帧数**。时间轴每次只前进 `decode_chunk_len = 48` 帧（480 ms），多出的 13 帧是卷积模块的重叠上下文；实测每个 chunk 输出 **12** 帧。（`T` / `decode_chunk_len` 从 encoder 的 `custom_metadata_map` 读出，`context_size` 来自 decoder 元数据；状态张量初值由 ONNX 输入签名推导。上例是当前权重的取值，代码里没有硬编码。）
- **`encoder_out.shape[1]` 必须取运行时值**，写死 `T` 会错。
- **blank 不刷新 decoder**（贪心搜索省算力的关键）；`blank_id` 从 `tokens.txt` 查 `<blk>`，不硬编码 0。
- **解码循环尾部补 100 帧特征域的 0**（约 1 秒）。补的 0 是 log 域对应能量 1，不是静音值，照抄官方做法。不补则最后一段凑不满 chunk 会被整段丢掉。
- **流式喂 encoder 的判据是 `len(buf) > T`（严格大于）**。写成 `>=` 会在批量条件恰好 `48k + 61` 时多喂一次。
- **116 个 encoder 状态张量 + `hyp` + 最近一次 `decoder_out` 必须跨块存活**，除 `reset()` 外绝不清零。状态初值由 ONNX 输入签名自动推导（字符串维按 batch=1 取 1），换模型不用改代码。
- 文本还原：`▁`(U+2581) 转空格后，CJK 之间、全角标点前后的空格去掉，但 **CJK 与 ASCII 之间的空格要保留**（`是 Monday` 不能粘成 `是Monday`）。
- **可视化路径的前端要传 `tail_pad_frames=0`**。补出来的 100 帧 0 在语谱图上是一块比静音还亮的假色块，只该喂解码器。

## 前端对齐的既有结论

初版在 hamming 窗下 `max|Δ| = 10.44`，定位到七项差异后逐项修复（`snip_edges=False` 的分帧语义、povey 窗、窗分母 `N-1`、Mel 域遍历整数 bin 的滤波器组、`symmetric` 边界、`log(max(e, eps_f32))` 地板、逐帧预加重），最终 `mean|Δ|` 从 0.92 降到 4×10⁻⁶。**改 `audio_preprocess.py` 前先读** [工作流报告](doc/sherpa-onnx-ASR预处理及后处理工作流报告.md) 里的完整推导——里面每一项都有实测依据，重写容易把对的地方改错。

`high_freq` 口径仍有分歧未实测：官方 `test_onnx.py` 取 8000 Hz（Nyquist），sherpa-onnx C++ 默认 `-400`（7600 Hz）。当前默认 8000 Hz。

## 工程约定

- **一律用 `python -I` 跑脚本，并在文件里手动把项目根插进 `sys.path`**。`-I` 隔离模式连脚本目录都不进 `sys.path`（且忽略 `PYTHONPATH`），所以三个入口头部都有 `_HERE` 插入逻辑，别删。
- **权重目录按不可信数据处理**：绝不在其目录内执行解释器，脚本、`bpe.model` 一概不碰，只读三个 onnx 与 `tokens.txt`；路径全部由入口文件顶部的常量指定。
- **打印中文的脚本开头调 `setup_stdout_utf8()`**（Windows 控制台默认 GBK，不转就是乱码）。
- 数据：`data/aishell1_subset/` 的 CSV 是本仓库改过的 **UTF-8** 版（ModelScope 原版是 GBK，直接读会 `UnicodeDecodeError`）；CSV 里的路径相对 `data/`。

## 已知问题 / 待办

- `dither` 语义与 `kaldi_native_fbank` 的默认值口径不同（本项目是高斯噪声标准差，直接作用于输入量纲）；推理一律用 0，流式版直接拒绝非 0。
- `sounddevice` 的设备编号与 `query_devices()` 打印的表对不上，默认路径不受影响。
- 待办：对 `data/` 的 aishell1 子集批量解码、与标注比 WER，出准确率与 RTF 基线；可选加 VAD/端点检测。

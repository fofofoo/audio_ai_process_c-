# audio_ai_process（C++）

**稳定版**：海康拾音器实时音频或 wav 文件 → 流式 LogFBank → 中文流式 ASR / 关键词唤醒 / OpenCV 滚动语谱图。
仓库同时提供「设备实时取流」与「wav 文件」两条数据源，共用同一套前端；
ASR 结果与 `python_ref/` 中的 Python 参考实现逐点、逐 token 对齐，KWS 按同一套
束搜索与 trie 触发语义移植。工程拆成「麦克风采集」与「ASR 处理」两个零耦合
SHARED 库，入口只做装配。

## 功能说明

1. **设备音频采集与保活**：通过 HCNetSDK 语音转发直接获取设备端拾音器的编码音频，
   不依赖本地声卡。G.722.1 由 SDK 解码为 16 kHz 单声道 PCM；客户端每 40 ms 回发一帧
   80 B G.722 静音做上行保活——实测只收不发时设备约 10 秒后停止回调，保活后可长时间连续取流。
   生命周期拆成 `init()` → `start()` → `stop()` → `deinit()` 四段：`init()` 只登录并
   确认编码与采样率，`start()` 才真正开流，模型加载等耗时步骤可以放在中间执行；
   `stop()` / `start()` 只停/起对讲线路，不重跑登录，`start()` 按 200 ms 间隔重试
   以避开设备侧操作节流。
2. **录制与输出格式控制**：`start_record(path)` / `stop_record()` 可随时开始、结束写盘，
   wav 先写 `.wav.part` 再原子改名，异常中断不会留下看似完整的半截文件。
   可选 `set_audio_format` 在启动前主动设置对讲编码（G.722.1 / PCM），写前快照、写后读回复核，
   不一致拒绝初始化，`deinit()` 默认写回原配置；可选 `output_sample_rate=16000` 在采集库内做
   整数倍抗混叠降采样，wav、取流缓冲与统计三处保持一致。
3. **实时取流接口**：`get_last_audio()` 一次取走自上次调用以来的全部 PCM16 样本，
   与是否正在录制无关；接口线程安全，可直接供前端、可视化与 ASR 消费。
4. **流式 LogFBank 前端**：按 kaldi `snip_edges=False` 口径实现分帧、povey 窗、逐帧预加重、
   512 点 FFT、80 通道 Mel 滤波和取对数；输入块长任意，出帧结果与块长无关。
5. **双数据源滚动语谱图**：设备实时流与 wav 文件都能画成 4 秒滚动窗口，viridis 伪彩、
   百分位色阶自动标定，画面可保存到 `results/`。
6. **wav 流式语音识别**：encoder / decoder / joiner 三个 session 串成完整流水线，
   实现贪心搜索与文本还原；支持单文件、多文件或目录批量识别，窗口底部实时显示中文结果
   （FreeType 渲染，缺字体自动回退 ASCII），`--no-window` 只跑文本。
7. **设备实时语音识别**：把设备取流、LogFBank 前端与 `AsrDecoder` 串成主循环，边说话边出字；
   窗口同时滚动显示语谱图与中文识别结果，空格键暂停/恢复语音转发，`--seconds N` 可限时自检。
8. **wav 关键词唤醒**：`KwsDecoder` 在 263 个音素级单元上做束搜索，每条假设挂一个关键词
   trie 状态；走完整条关键词并攒够尾随 blank 帧后输出命中词、首 token 时间与平均声学概率。
   关键词由文本文件配置（`token... :boost #threshold @原文`），换唤醒词不用重训模型；
   `test_wav_kws.out` 跑权重包自带 9 条 wav（含 3 条负样本），并检查命中集合与块长不变性。
9. **模型自检与数值对拍**：`load_onnx_model.out` 校验三模型加载与 metadata；
   `fbank_dump.out` + `compare_fbank.py`、`--dump-tokens` + `compare_asr.py` 分别做
   逐帧逐 Mel 与逐 token 对拍，保证 C++ 移植没有静默偏移。
10. **两个零耦合库与安装**：采集库不依赖 OpenCV / ORT，ASR 库不依赖 HCNetSDK；
   两个库互不 include 对方头文件，`cmake --install build` 输出到 `install/<库名>/`。

**一句话亮点：从设备取流、流式前端到中文 ASR、关键词唤醒与实时可视化全链路 C++，ASR 与 Python 参考实现逐位、逐 token 对齐，KWS 沿用同一套解码语义。**

## 稳定性与验证基线

功能已收敛，接口与数值口径以本文档及三个模块 README 为准。已在 **Linux x86_64（WSL2）**
与 **aarch64（Jetson，JetPack 5.1.2）** 完成全链路验证；Windows 的目录、链接和安装
路径已经配置，但尚未实机验证。

| 验证项 | 方法 | 当前结果 |
| --- | --- | --- |
| C++ 前端 vs Python 参考实现 | `compare_fbank.py`，块长 160 / 17 / 333 / 1000 / 4096 / 0 | 三条实测录音均为**逐位一致**（平均与最大绝对误差都为 0） |
| C++ wav ASR vs Python 参考实现 | `--dump-tokens` + `compare_asr.py` | 权重包四条 wav 全部**逐 token 一致**（token 数 27 / 20 / 23 / 32） |
| C++ wav KWS | `test_wav_kws.out`，9 条自带 wav | 命中集合与期望一致，负样本零误触发；160 / 1600 块长结果相同，累计触发 10 次 |
| 模型加载与契约 | `load_onnx_model.out` | CPU 与 CUDA EP 均通过；`T=61`、`decode_chunk_len=48`、`context_size=2`、`vocab_size=5000` |
| 设备长时取流 | G.722.1 运行 45 s 组 | 完整交付 46.4 s 音频，回调未中断 |
| 设备实时识别 | `bench_perf.sh --realtime 20`（Jetson CPU） | 丢块 **0**，启动积压 **−0.34 s**，吞吐 0.99（20 秒场景） |
| wav 识别性能 | `test_wav_asr.out --no-window` | CPU 上 RTF ≈ 0.06（入口结束打印） |
| 平台 | Linux x86_64 / WSL2、aarch64 / Jetson | 通过；Windows 未实测 |

四条测试音频的识别文本（Python 参考实现与 C++ 一致，可作回归基线）：

```
0.wav  昨天是 Monday，today is 礼拜二，the day after tomorrow 是星期三
1.wav  这是第一种，第二种叫呃与 always always 什么意思啊？
2.wav  就是频繁的啊，不认识记下来，frequently 频繁的
3.wav  第一句是个什么时态？加了 yes 是一般现代时，对，我们把它时态写上去。
```

## 组成

| 类别 | 路径 | 一句话职责 |
| --- | --- | --- |
| 库 | `lib_hk_sdk_audio_client.so` | 麦克风采集：登录、语音转发、G.722.1 解码、可选降采样与 wav 录制、取流缓冲与统计；不依赖 OpenCV/ORT |
| 库 | `lib_audio_ai_process.so` | ASR / KWS 处理：LogFBank 前端、`AsrDecoder` / `KwsDecoder`（ORT）、KWS 关键词前缀树、语谱图、文本渲染；不依赖 HCNetSDK |
| 采集源码 | `hk_mic_client/hk_sdk_audio_client.{h,cpp}` | 设备连接与音频流水线 |
| 采集源码 | `hk_mic_client/audio_resampler.{h,cpp}` | 整数倍降采样（Kaiser 窗 FIR + 抽取），PCM 32 kHz → 16 kHz；零外部依赖 |
| 采集源码 | `hk_mic_client/hk_sdk/` | 厂商 HCNetSDK 头文件与预编译库（按平台分目录） |
| ASR 源码 | `audio_ai_process/audio_preprocess.{h,cpp}` | 流式 LogFBank 前端：PCM 块 → `(n_frames, n_mels)` 特征 |
| ASR 源码 | `audio_ai_process/asr_decode.{h,cpp}` | 流式 ASR 解码：三模型推理、贪心搜索、文本还原、跨块状态 |
| KWS 源码 | `audio_ai_process/kws_decode.{h,cpp}` | 流式关键词唤醒：三模型推理、trie 约束的束搜索、尾随 blank 触发判定、静音软重置 |
| KWS 源码 | `audio_ai_process/kws_context_graph.{h,cpp}` | 关键词前缀树（trie + fail/output 弧）：逐 token 步进、关键词加分与命中判定；纯标准库，不依赖 ORT |
| ASR 源码 | `audio_ai_process/spectrogram_image.{h,cpp}` | 特征 → BGR 滚动语谱图画布 |
| ASR 源码 | `audio_ai_process/text_renderer.{h,cpp}` | UTF-8 文本 → BGR 画布（FreeType 渲染中文，可选依赖） |
| ASR 源码 | `audio_ai_process/text_utils.{h,cpp}` | UTF-8 编解码、字符分类、系统路径等小工具 |
| 入口 | `save_audio_wav.cpp` | 录 10 秒设备音频到 `results/` |
| 入口 | `test_spectrogram_cv_mic.cpp` | 设备实时语谱图入口 |
| 入口 | `test_spectrogram_cv_wav.cpp` | wav 文件语谱图入口 |
| 入口 | `test_wav_asr.cpp` | wav 语音识别入口：文本 +（可选）语谱图与中文识别条 |
| 入口 | `test_wav_kws.cpp` | wav 关键词唤醒入口：批量跑 wav、打印命中词与时间、检查期望集合与块长不变性；不开窗口 |
| 入口 | `test_mic_asr.cpp` | 设备实时语音识别入口：取流 → 特征 → 后台解码 → 文本 + 滚动语谱图（其他入口的线程模型参照此文件） |
| 入口 | `load_onnx_model.cpp` | ASR 三模型加载自检入口 |
| 工具 | `tools/wav_reader.h` | 最小 RIFF 解析，wav 类入口共用 |
| 工具 | `tools/fbank_dump.cpp` | 前端对拍：裸 float32 PCM → 裸 float32 特征 |
| 工具 | `tools/compare_fbank.py` | C++ 与 Python 前端逐点比较 |
| 工具 | `tools/compare_asr.py` | C++ 与 Python ASR 逐 token 比较 |
| 工具 | `tools/test_context_graph.cpp` | KWS 前缀树单测：上游 4 组 gtest 期望值 + phrase/threshold/output 弧字段自检，不需要权重与 ORT |
| 工具 | `tools/bench_perf.sh` | 性能基准：离线 RTF / 实时处理率、取流与后台解码丢块，CPU 与 CUDA 对比 |
| 资源 | `onnxruntime/` | 项目内托管的 ONNX Runtime；`install/` 不入库，接入方式见 `onnxruntime/README.md` |
| 资源 | `weight/` | ASR 模型权重与词表（fp32 / int8 两套），本地放置，不入库 |
| 资源 | `python_ref/` | Python 参考实现（只读基准，不参与 C++ 编译） |
| 资源 | `results/` | 运行产物：录音 wav 与语谱图 png，只提交 `.gitkeep` |

### 依赖链（谁编译在先）

```
HCNetSDK ───────────────> lib_hk_sdk_audio_client.so ──┐
                                                       ├──> 入口（按需链接：save_audio_wav / test_* / load_onnx_model）
OpenCV ──────────────────> lib_audio_ai_process.so ────┤
ONNX Runtime（可选） ────┘                             │
FreeType（可选） ────────┘                             │
                                                       └── test_spectrogram_cv_mic / test_mic_asr 同时链接两个库
```

ORT 缺失时 `asr_decode.cpp` / `kws_decode.cpp` 与四个 ORT 入口自动跳过；
`kws_context_graph.cpp` 与 `test_context_graph.out` 不依赖 ORT，仍可构建。
FreeType 缺失时 `text_renderer` 编译空实现，入口回退 `cv::putText`。

### 数据流（谁把数据给谁）

```
海康拾音器 --HCNetSDK 语音转发--> HkSdkAudioClient --PCM16--+
                                                            +--> /32768.0 --> AudioPreprocess
wav 文件 --tools/wav_reader.h（RIFF 解析）------------------+                      |
                                                                                  v
                                                                    LogfbankFeatures (n,80)
                                                                                  |
                          +---------------------------+---------------------------+
                          v                           v                           v
              AsrDecoder（贪心搜索）      KwsDecoder（trie 束搜索）      SpectrogramImage
                          |                           |                           |
                          v                           v                           v
               文本（stdout / 窗口底部）      命中关键词（stdout）          BGR 画布 --> imshow
```

ASR 与 KWS 是两条并列的解码分支，共用同一个前端：前者输出整句文本，后者在
关键词 trie 约束下只输出命中事件。`test_wav_kws.out` 只接 KWS 分支，不建窗口。

麦克风入口与 wav 入口只在「数据源」不同，后半段完全共用。核心类只靠特征相连：
前端不知道设备，可视化不知道波形，解码器不知道数据源。

## 输入输出契约

### 音频输入

- **设备链路**：G.722.1，16 kHz、单声道、PCM16；设备地址、端口、账号、语音通道
  写在各入口 `main` 顶部，默认 `10.168.2.65:8000`、`channel=1`。
- **PCM 链路**：本机实测只能 32 kHz（请求 16 kHz 不生效）。`output_sample_rate`
  为 `0` 时按设备原始率输出（`device_sample_rate()`）；设为 `16000` 时降采样后输出，
  `sample_rate()` 恒为 16000。只支持整数倍降采样，44.1k 等比例拒绝启动。
- **推荐格式**：实时 ASR 默认用 **G.722.1**（16 kbps、样本连续、无需降采样）。
  PCM 存在**单次 `StartVoiceCom` 会话上限**：约 21.5 秒后停止交付，把 PCM 写成持久
  配置并重启设备后复测结果不变；`stop()` → `start()` 重开语音转发可恢复（2026-11-04
  实测）。PCM 只建议用于短会话或无损采集调试，详见
  [`hk_mic_client/hk_audio_README.md`](hk_mic_client/hk_audio_README.md) 1.7.2。
- **wav 文件**：支持 PCM16 与 float32，多声道只取第一声道；不做重采样，
  非 16 kHz 的 wav 可以画图，但喂 ASR 前必须自行重采样。

### 特征输出

- 类型 `float32`，行主序 `(n_frames, n_mels)`，第 i 帧第 j 维在 `data[i * n_mels + j]`。
- 单位是对数能量 `ln(能量)`，静音地板为 `ln(float32_eps) ≈ -15.9424`。
- 可视化路径构造前端时传 `tail_pad_frames=0`；解码路径保持默认 `100`。
  补的 100 帧特征 0 是 log 域能量 1，不是静音，只用于凑满解码块。

### 公共接口

| 类 | 主要接口 | 说明 |
| --- | --- | --- |
| `hk_isapi::HkSdkAudioClient` | `init()` / `deinit()` | 登录、查询编码、校验采样率 / 恢复编码、登出并释放 SDK；不涉及文件 |
| | `start()` / `stop()` | 只起/停对讲线路（`StartVoiceCom` / `StopVoiceCom` + 保活），保留登录与解码器；`stop()` 返回 `bool`，暂停自动收尾 wav |
| | `start_record(path)` / `stop_record()` | 开始/结束 wav 写盘，可多段录制 |
| | `get_last_audio(samples)` | 取走并清空累积 PCM16；无新数据返回 `false` |
| | `get_stats()` / `last_error()` / `sample_rate()` / `device_sample_rate()` | 统计、错误、输出采样率与设备原始采样率 |
| `audio_ai::AudioPreprocess` | `accept_waveform(samples)` | 喂任意长度 float 波形，返回本次新产出的特征 |
| | `input_finished()` | 补尾帧；可视化传 `tail_pad_frames=0` |
| | `reset()` | 清空流式状态，构造参数与滤波器组保留 |
| `audio_ai::SpectrogramImage` | `accept_features(feats)` | 喂特征，返回整幅 BGR 画布 |
| | `frame()` / `reset()` | 按当前缓存重画 / 清空缓存 |
| `audio_ai::AsrDecoder` | `accept_features(feats)` | 喂特征，返回截至当前的完整文本；跨块状态内部维护 |
| | `text()` / `token_ids()` | 当前文本 / 已确认 token（供对拍） |
| | `reset()` / `info()` | 清空全部跨块状态 / 模型契约与计数 |
| `audio_ai::KwsDecoder` | `accept_features(feats)` | 喂特征，返回本次新命中的 `KeywordHit`（关键词、token 时间、平均概率） |
| | `keywords()` / `info()` | 关键词原文列表 / 模型契约与束宽 |
| | `reset()` / `n_hits()` / `n_chunks()` | 清空束搜索与 encoder 状态 / 累计命中与编码次数 |
| `audio_ai::Utf8TextRenderer` | `init(path, face_index)` / `draw(...)` | 加载字体 / 把 UTF-8 文本画到 BGR 画布；不可用时回退 `cv::putText` |
| | `draw_tail(..., max_width)` | 从尾部按像素宽度截断后绘制（累积文本滑动显示，中英混排同样准确） |

### 模型契约

`load_onnx_model.out` 只验证「能否加载 + 接口是什么」，不做推理。三个模型各自独立加载，
单个失败只报 `ERROR` 并继续，最后按「全部成功且关键 metadata 齐全」返回 0 / 1。

| 模型 | 输入 | 输出 |
| --- | --- | --- |
| encoder | `x` float32 `[-1, 61, 80]` + 116 个状态张量 | `encoder_out` `[-1, -1, 512]` + 116 个新状态 |
| decoder | `y` int64 `[-1, 2]` | `decoder_out` `[-1, 512]` |
| joiner | `encoder_out` + `decoder_out` `[-1, 512]` | `logit` `[-1, 5000]` |

关键 metadata：`model_type=zipformer2`、`T=61`、`decode_chunk_len=48`、
`context_size=2`、`vocab_size=5000`（与 `tokens.txt` 的 5000 行一致）。
即每块编码窗口 610 ms、块间前进 480 ms、重叠 130 ms。
状态张量的动态维对应 batch，建推理循环时填 1；符号维在 `load_onnx_model.out`
里照实打印，不臆测具体值。

KWS 权重包是另一套契约（`T=29`、`decode_chunk_len=16`、`vocab_size=263` 等），
详见下文「关键词唤醒（KWS）」的模型属性表。

## 关键词唤醒（KWS）

### 1. 它是什么

开集关键词唤醒（open-vocabulary KWS）不是关键词分类器，而是**词表被关键词前缀树
约束住的小型流式 ASR**：模型仍是 streaming zipformer2 transducer，「只认这几个词」
完全由解码器完成——每条 beam 路径挂一个 trie 状态，输出只能是某个关键词的发音
序列，或者什么都没有。前端与 ASR 同口径（80 维 fbank、16 kHz），同一个
`AudioPreprocess` 可以直接复用。

C++ 侧三个部件：

| 部件 | 路径 | 职责 |
| --- | --- | --- |
| `KwsContextGraph` | `audio_ai_process/kws_context_graph.{h,cpp}` | 关键词前缀树：加分、逐 token 步进、命中判定；纯标准库 |
| `KwsDecoder` | `audio_ai_process/kws_decode.{h,cpp}` | 分块编码、trie 约束的束搜索、触发判定、静音软重置 |
| `test_wav_kws.out` | `test_wav_kws.cpp` | 离线验收入口：wav → 特征 → 命中事件，不开窗口 |

Python 参考实现在 `python_ref/src/kws_decode.py`；C++ 按它的逐帧语义移植，并按
「图 + 解码器 + 入口」拆成三层。

### 2. 模型与解码要点

模型尺寸从 ONNX metadata 与输入签名读取，不硬编码。默认 chunk-8：
`T=29`、`decode_chunk_len=16`（时间轴 160 ms/块）、每块输出 4 帧（40 ms/帧）、
`context_size=2`、`vocab_size=263`、38 个 encoder 状态；chunk-16 为
`T=45` / 32 帧 / 8 帧，其余不变。263 个 token = 3 个特殊符 + 69 个 CMU 音素 +
191 个部分拼音，**匹配的是发音序列，不是文字**。

与 ASR 的区别集中在解码器：ASR 贪心、非 blank 不刷新 decoder；KWS 用 ≤4 条假设的
束搜索，每帧对全部假设批量跑 decoder / joiner，`blank_id` 硬编码 0、`<unk>` 也按
blank 处理。命中条件是三者同时成立：trie 走完整条关键词、末尾至少 2 个 blank
输出帧（80 ms）、关键词各 token 的平均声学概率不低于该词阈值。触发后只重置 beam，
encoder 状态继续累积；连续 1.5 s 无有效输出时软重置，但保留未消费的特征尾巴。
离线 wav 末尾必须补真静音，否则句尾关键词不触发；麦克风流连续，不受影响。

### 3. 唤醒词设置

一行一个关键词：

```
token token ... [:boost] [#threshold] @原文
```

- token 必须出现在 `tokens.txt`，否则启动报错并带行号；
- `:boost` 是束搜索路径加分（越大越容易走到）；`#threshold` 是触发阈值
  （0~1，越大越难触发，用来压同音误触）；
- `@原文` 是展示文本，中文必须有且不能含空格（用 `_` 代替）；
- 整行 `#` 是注释；`:0` / `#0` 表示使用构造默认值（1.0 / 0.25）。

入口默认读权重包自带的 `test_wavs/keywords.txt`，用 `--keywords` 换文件。
关键词的生成与调参只在 Python 参考实现中完成；C++ 侧只读生成好的 token 文件，
不移植 `text2token` / 录音评估工具。

### 4. 验证

| 入口 | 覆盖 |
| --- | --- |
| `test_context_graph.out` | 上游 4 组 trie 期望值 + 字段自检，不需要权重与 ORT |
| `test_wav_kws.out` | 9 条 wav（`zh_0~2` 为负样本）命中集合一致 + 160/1600 块长不变性 |
| `test_wav_kws.out --dump-hits` | 导出 TSV（文件名 / 关键词 / 首 token 时间 / 平均概率），供与 Python 对拍 |

### 5. 延伸阅读

逐帧流程、易错点、与上游 `context-graph` / `transducer-keyword-decoder` /
`keyword-spotter-transducer-impl` 的逐项对应，见
[python_ref/README.md](python_ref/README.md) 的「束搜索」「唤醒词机制」两章和
[KWS 调研报告](python_ref/doc/sherpa-onnx-KWS关键词唤醒调研报告.md)。

## 运行与集成

### 依赖与编译

CMake ≥ 3.10、C++17、OpenCV（core / imgproc / imgcodecs / highgui）、pthread。
海康 SDK 已放在 `hk_mic_client/hk_sdk/`，CMake 按架构自动选择 `linux_x86` /
`linux64_arm` / `win64_x86`，并为可执行文件设置 RPATH。

FreeType 是 `lib_audio_ai_process` 的可选依赖（供 `text_renderer` 渲染中文）：
找到时定义 `HK_HAVE_FREETYPE`，带窗口的 ASR 入口在底部用系统字体渲染中文；
找不到时该模块编译空实现、入口自动回退 `cv::putText`（只支持 ASCII），不影响识别。
KWS 离线入口不建窗口，不涉及文本渲染。

ASR 相关目标额外依赖 ONNX Runtime，CMake 按架构选择项目内托管版本：
x86_64 用 `onnxruntime/install/onnxruntime-1.23.2`，aarch64 用
`onnxruntime/install/onnxruntime-1.18.0`（需在 Jetson 上源码编译）。
ORT 缺失时只给 WARNING，音频采集 / 语谱图 / fbank 对拍仍可构建，
依赖推理的 `asr_decode.cpp` / `kws_decode.cpp` 与四个 ORT 入口自动跳过。
CUDA EP 的 cuDNN 9 环境配置见 `onnxruntime/README.md`。

```bash
cmake -S . -B build
cmake --build build -j
cmake --install build      # 输出到 install/<库名>/{include,lib}
```

### 安装与集成

```
install/
├── hk_sdk_audio_client/
│   ├── include/   hk_sdk_audio_client.h、audio_resampler.h、HCNetSDK 头文件
│   └── lib/       lib_hk_sdk_audio_client.so + HCNetSDK 运行库（含 HCNetSDKCom/）
└── audio_ai_process/
    ├── include/audio_ai_process/*.h
    └── lib/lib_audio_ai_process.so
```

两个库互不 include 对方头文件；`test_mic_asr.out` / `test_spectrogram_cv_mic.out`
只作为组合层同时链接两者。安装后库自身 `INSTALL_RPATH=$ORIGIN`，运行时能在同目录
找到 HCNetSDK 运行库。ORT、OpenCV、FreeType 不随安装复制，属于外部运行期依赖：
使用安装库的程序需要自行提供对应运行库（或设置 `LD_LIBRARY_PATH`）。

使用方编译时加 `-Iinstall/<库名>/include`，链接
`-Linstall/hk_sdk_audio_client/lib -l_hk_sdk_audio_client` 或
`-Linstall/audio_ai_process/lib -l_audio_ai_process`；只需要采集能力时
不必链接 OpenCV 或 ONNX Runtime。

### 常用命令

```bash
# 录 10 秒设备音频 -> results/hk_audio_<时间戳>.wav
./build/save_audio_wav.out

# 设备实时语谱图：q / Esc 退出，s 存图
DISPLAY=:0 ./build/test_spectrogram_cv_mic.out

# 自检模式：跑 10 秒，存 results/spectrogram_mic.png 后退出
DISPLAY=:0 ./build/test_spectrogram_cv_mic.out 10

# wav 语谱图：默认按 1x 节奏播放；不传路径时读 results/hk_audio_20261102_085654.wav（本机录音，不入库）
DISPLAY=:0 ./build/test_spectrogram_cv_wav.out results/<录音>.wav

# 批量检查：尽快喂完，存 results/spectrogram_wav.png 后退出
DISPLAY=:0 ./build/test_spectrogram_cv_wav.out results/<录音>.wav --fast

# ASR 三模型加载自检（默认非 int8、CPU）
./build/load_onnx_model.out

# 切换 int8 模型目录（encoder / joiner 自动识别 .int8.onnx）
./build/load_onnx_model.out weight/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05

# 验证 CUDA EP（需先按 onnxruntime/README.md 设置 cuDNN 9 的 LD_LIBRARY_PATH）
LD_LIBRARY_PATH=<cuDNN 9 目录> ./build/load_onnx_model.out --cuda

# wav 语音识别：单个文件，窗口底部实时显示中文结果
DISPLAY=:0 ./build/test_wav_asr.out weight/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05/test_wavs/1.wav

# 批量识别：默认目录（权重包 test_wavs/）或目录/多文件，只打印文本
./build/test_wav_asr.out --no-window
./build/test_wav_asr.out --no-window weight/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05/test_wavs

# 导出 token 供 Python 参考实现逐 token 对拍
./build/test_wav_asr.out --no-window --dump-tokens /tmp/cpp_tokens.txt

# CUDA EP（失败自动回退 CPU）
./build/test_wav_asr.out --no-window --cuda weight/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05/test_wavs/1.wav

# 实时麦克风语音识别：窗口底部实时显示中文，q / Esc 退出，空格暂停/恢复语音转发
DISPLAY=:0 ./build/test_mic_asr.out

# 只打印文本；跑 10 秒自动收尾退出（无人值守自检）
./build/test_mic_asr.out --no-window --seconds 10

# KWS：默认跑权重包自带 9 条 wav，检查命中集合与流式不变性
./build/test_wav_kws.out

# KWS：指定文件 / 目录，导出命中（文件名 <TAB> 关键词 <TAB> 时间 <TAB> 概率）
./build/test_wav_kws.out weight/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/test_wavs \
    --dump-hits /tmp/kws_hits.tsv

# KWS：覆盖 boosting score / 触发阈值，并打印被阈值挡住的候选
./build/test_wav_kws.out --score 2.0 --threshold 0.3 --debug

# KWS 关键词前缀树单测：不需要权重 / ORT，通过时返回 0
./build/test_context_graph.out
```

WSLg 下如果窗口打不开，先确认 `DISPLAY=:0`；环境里可能残留无效的远程 `DISPLAY`。
所有运行产物统一写入 `results/`。两个语谱图入口使用相对 CWD 的 `results/` 路径，
运行时需要先 `cd` 到工程根。

### 对拍与自检

没有单元测试框架；验证按改动层次选择下面的最小集合。

KWS 关键词前缀树：

```bash
./build/test_context_graph.out
```

搬上游 `context-graph-test.cc` 的 4 组期望值（strict / 非 strict × 默认分 /
自定义分），并补 phrase、逐关键词阈值、fail/output 弧与空关键词抛异常的
字段自检。纯标准库实现，不加载 onnx，换机也能单独跑。

离线 wav 关键词唤醒：

```bash
./build/test_wav_kws.out
```

跑权重包自带的 9 条 wav，命中集合必须与期望完全一致：

| wav | 期望命中 | wav | 期望命中 |
| --- | --- | --- | --- |
| `en_0.wav` | `LIGHT_UP` | `zh_4.wav` | `蒋友伯`、`女儿` |
| `en_1.wav` | `LOVELY_CHILD` | `zh_5.wav` | `周望军`、`落实` |
| `zh_0.wav` | 空（负样本） | `zh_6.wav` | `朱丽楠`、`见面会` |
| `zh_1.wav` | 空（负样本） | | |
| `zh_2.wav` | 空（负样本） | | |

触发要求关键词后至少 2 个 blank 输出帧（80 ms），所以入口会在每条 wav 末尾补
**0.5 秒真静音**（样本域 0，不是特征域 0）；不补则句末关键词不触发。入口还会用
160 / 1600 两种块长重跑第一条 wav，命中词与首 token 时间必须一致——块长只影响
喂入节奏，不该影响解码结果。当前 9 条全部符合期望，累计触发 10 次。

前端逐点对拍：

```bash
python3 -I tools/compare_fbank.py --wav results/<录音>.wav
python3 -I tools/compare_fbank.py --chunks 160,17,333,1000,4096,0
```

脚本用同一分块节奏分别跑 `python_ref/src/audio_preprocess.py` 和 C++ 的
`fbank_dump.out`，输出 `mean|Δ|`、`max|Δ|` 与超差位置。判定阈值默认
`mean|Δ| ≤ 1e-5`、`max|Δ| ≤ 1e-3`；当前实测为 0。

ASR 逐 token 对拍：

```bash
# 1) C++ 侧导出 token
./build/test_wav_asr.out --no-window --dump-tokens /tmp/cpp_tokens.txt

# 2) Python 参考实现按同一块长（160 样本）解码并逐个 token 比较
#    用带 numpy + onnxruntime 的 Python 环境（本机为 conda 环境 pytorch-gpu）
python -I tools/compare_asr.py --cpp /tmp/cpp_tokens.txt \
    --wav-dir weight/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05/test_wavs
```

`test_wav_asr.out` 每行写「文件名 <TAB> id,id,...」；脚本用 stdlib `wave` 读同一条
wav，依次跑 `python_ref/src/audio_preprocess.py` 与 `asr_decode.py`（尾部同样补
100 帧 0），逐 token 比较并打印 Python 侧文本。当前权重包自带的四个 wav
全部一致；出现差异时会指出第几个 token、两侧 id。对拍脚本依赖 numpy，
`compare_asr.py` 另需带 onnxruntime 的 Python 环境。

性能基准：

```bash
# 离线：默认权重目录 4 条 wav，只测前端 + 三模型推理的纯吞吐
./tools/bench_perf.sh

# 离线：CPU 与 CUDA 对比（CUDA 注册失败时脚本会明确提示「实际仍跑 CPU」）
./tools/bench_perf.sh --cuda --wav weight/<权重包>/test_wavs

# 实时：设备在线，各跑 20 秒无窗识别，看主循环能否跟上 16 kHz 实时流
./tools/bench_perf.sh --realtime 20

# 实时：加 --window 测渲染开销，加 --cuda 做执行器对比
./tools/bench_perf.sh --realtime 20 --window
```

脚本只解析入口自己打印的统计行，不重新计时，口径与入口一致：

- **RTF** = 解码耗时 / 音频时长（离线，< 1 快于实时，越小越好）；
- **主循环吞吐** = 喂入样本 / 16000 / 墙钟秒数（稳态 ≈ 1，> 1 表示正在消化积压）；
- **丢块时长** = 取流丢弃样本 / 16000 秒（> 0 说明取流缓冲已溢出）；
- **解码丢弃** = 后台解码队列因满而丢弃的块数（上限 50 块 ≈ 2 s 缓冲，每块是前端
  一次投递的音频、约 40 ms，不是解码块的 480 ms）。取流没丢但这项 > 0，说明推理
  吞吐低于实时、丢的是识别内容；它衡量「后台解码跟不上」，与取流丢块衡量的
  「采集/主循环跟不上」是两种不同的落后。
- **启动积压** = 实际转发秒数 − 墙钟秒数。入口现在先 `init()` 登录、模型加载完
  才 `start()` 开流，正常情况下这项应接近 0；若明显大于 0，说明开流早于主循环，
  取流缓冲（上限 2.56 s）会先溢出，归因丢块时不能直接当成「主循环慢」。

### Jetson 实测结论

环境：Jetson（`nvidia-desktop`），设备 10.168.2.65 / G.722.1 / 16 kHz，
int8 权重包，2026-11-03 首测；2026-11-04 后台解码改造后补测 20 s 三组。

**离线（4 条 wav，共 28.67 s 音频）**

| 执行器 | 解码耗时 | RTF | 相对实时 |
| --- | --- | --- | --- |
| CPU | 7.78 s | 0.2713 | 3.69 倍 |
| CUDA | 14.75 s | 0.5145 | 1.94 倍 |

CPU 比 CUDA 快 1.90 倍：每块只编码 61 帧、joiner / decoder 逐帧调用、116 个
状态张量每次进出一遍，kernel launch 与显存搬运的开销盖过了 GPU 收益。

**实时（`--realtime 20`）**

| 运行 | 墙钟 | 转发 | 处理音频 | 吞吐 | 丢块 | 解码丢弃 | 启动积压 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CPU 无窗（改造前） | 20.32 s | 27.02 s | 22.60 s | 1.11 | 4.52 s | — | 6.70 s |
| CPU 有窗（改造前） | 20.29 s | 26.95 s | 22.56 s | 1.11 | 4.96 s | — | 6.65 s |
| CUDA 无窗（改造前） | 20.50 s | 34.69 s | 20.52 s | 1.00 | 14.24 s | — | 14.18 s |
| CPU 无窗（改造后） | 20.67 s | 20.10 s | 20.20 s | 0.98 | **0.00 s** | **0** | **−0.58 s** |
| CPU 有窗（改造后） | 20.75 s | 20.04 s | 20.48 s | 0.99 | **0.00 s** | **0** | **−0.71 s** |
| CUDA 无窗（改造后） | 20.69 s | 20.03 s | 20.00 s | 0.97 | **0.00 s** | **5** | **−0.66 s** |

进程 CPU 占用（`top` 单次采样，单进程 100% = 一个核打满；计的是进程全部线程，
含 HCNetSDK 线程池）：无窗约 **0.31 核**、有窗约 **0.70 核**，渲染增加约 0.4 核。
与吞吐 0.98、丢块 0 一致，单核仍有约 30% 余量。

**长跑（CPU 无窗，281 s）**：取流丢块 0、后台解码队列丢弃 0、静音补零 0 次，上行保活
发送 6884 帧 / 失败 0 次；喂入 4,495,360 样本（280.96 s），实际转发 280.55 s，吞吐约
0.999，encoder 587 块（587 × 480 ms ≈ 281.8 s）。说明 4 分半内主循环与后台解码都没有
积压累积，长会话不会越跑越滞后。

改造后的入口把设备会话拆成 `init()` → 模型加载 → `start()`，实测启动积压与取流丢块都归零，
转发时长与处理时长基本相等（20.10 s vs 20.20 s）。启动积压为负是正常现象：主循环进入后才
开始计墙钟，收尾的补尾解码与打印不计入 `forward_seconds`。改造前的两列「解码丢弃」没有
对应统计（后台解码队列是改造后新增的），记 —。

结论：

- **Jetson 用默认 CPU，不加 `--cuda`**：离线慢 1.9 倍；改造后 CUDA 取流不再丢块，
  但后台解码队列在 20 s 内丢了 5 块，且没有任何性能收益。
- **主循环跟得上**：无积压时吞吐天然接近 1，判据是丢块为 0（改造后实测）；离线 RTF
  0.27 还有约 3.7 倍余量。改造前的 1.11 是在消化 6.7 s 启动积压，不代表更快。
- **渲染不是瓶颈**：改造后有窗 / 无窗吞吐 0.99 / 0.98、丢块与解码丢弃都是 0，
  处理音频只差 0.28 s（20 s 的 1.4%）；改造前也同为 1.11，差异在噪声范围内。
- **CUDA 不再持续丢块，但仍不划算**：改造前 CUDA 在主循环期间持续丢 2.62 s；解码挪到
  后台线程后，这条路径不再被推理拖住，取流 0 丢块。代价换到后台队列：20 s 内丢 5 块，
  离线仍比 CPU 慢 1.9 倍、启动更久。这 5 块没有时间戳，未细分是前段预热还是稳态，
  但无论哪种都不改变「默认 CPU」的结论。
- **丢块曾主要来自启动阶段**：改造前的入口先 `start()` 再加载模型，这段时间（CPU ≈ 6.7 s、
  CUDA ≈ 14.2 s）设备已经在转发，取流缓冲上限 2.56 s，超出的部分直接丢。
  CPU 的丢块几乎全部落在这里；CUDA 在启动阶段丢 11.62 s 之外，还在主循环期间
  持续丢了 2.62 s。改造后这部分已消除。

判断口径：不要用「喂入 ÷ 实际转发」估主循环速度，它等价于「1 − 丢块比例」；
看 **吞吐**（喂入 ÷ 墙钟）和 **启动积压**（实际转发 − 墙钟）。

### Jetson 上能不能实时 ASR：能

**CPU 单线程即可，不需要 GPU。** 判断依据是离线 RTF 与实时吞吐两项：

| 指标 | 实测 | 含义 |
| --- | --- | --- |
| 离线 RTF（CPU） | 0.2713 | 1 秒音频只需 0.27 秒算完，约 3.7 倍余量 |
| 实时吞吐（CPU，稳态） | 0.98~0.99 | 跟得上时天然接近 1；真正判据是丢块 |
| 实时丢块（CPU） | **0.00 s / 20 s** | 改造后实测；改造前 4.52 s 全部在启动阶段 |
| 离线 RTF（CUDA） | 0.5145 | 比 CPU 慢 1.9 倍；改造后实时取流不丢块，但后台解码 20 s 内丢 5 块 |

**这两项不能相除换算**：离线 RTF 的分母是音频时长，量的是纯计算成本（每秒音频 0.27 s）；
实时吞吐的分母是墙钟，而分子（喂入样本）被设备 1.0× 实时率封顶，算力再快也不会超过 1。
所以「离线 0.27、实时 0.99」不代表实时比离线慢 3.7 倍，实时路径的纯计算成本同样是
每秒音频约 0.27 s，其余时间在等设备送下一批数据。实时的真实余量看进程 CPU 占用
（无窗 0.31 核，约七成时间空闲）与丢块，不看吞吐距 1 多远；上文 20 s 表也能反证：
有窗多花约 0.4 核做渲染，吞吐只从 0.98 变到 0.99。

**CUDA 不要开**：模型每块只编码 61 帧、joiner / decoder 逐帧调用、116 个状态张量
每次进出一遍，kernel launch 与显存搬运的开销盖过了 GPU 收益。实时场景保持默认 CPU。

**延迟预算**：解码每 480 ms 推进一块（`decode_chunk_len=48` 帧），编码窗口 610 ms，
加上推理（一块约 130 ms）与渲染，从说完话到出字大致 **0.5~1 s**；窗口上 4 秒滚动
语谱图只影响观感，不增加识别延迟。

**启动阶段丢音频的说明**：改造前的入口先 `start()` 开流、再加载模型，加载期间
（Jetson CPU ≈ 6.7 s、CUDA ≈ 14.2 s）设备已经在转发，而取流缓冲上限只有 2.56 s，
超出的部分直接丢。入口现在拆成 `init()` → 模型加载 → `start()`，开流时主循环已就绪；
Jetson 实测丢块从 4.52 s 降到 **0**，启动积压从 6.70 s 降到 **−0.58 s**。

**解码已挪到后台线程（示范脚本）**：`test_mic_asr.cpp` 在主循环外创建
`AsrWorker`，特征块经 `std::deque` 有界队列投递（上限 50 块，满时丢最旧），
主线程只做取流、前端、渲染与显示。理由：Jetson CPU 上推理、渲染、取流同线程时
吞吐约 0.99，恰在实时线上，加渲染就可能丢块。代价是底部文字比语谱图滞后若干块，
且推理真的跟不上时会丢弃最早的音频块（统计行「解码丢弃 N 块」）；尾块走
`push_final()` 阻塞等空位，绝不丢弃。`AsrDecoder` 明确非线程安全，全程只有解码
线程访问它，主线程在 `join` 之后才读 `n_chunks` / `token_ids` 等统计。
其他入口仍是主循环同步解码，需要时参照此脚本改造，不要在多线程间共享同一个
`AsrDecoder`。

### x64（WSL 笔记本）实时基准

环境：WSL2 / 24 逻辑核 / 设备同上（10.168.2.65，G.722.1，16 kHz）/ int8 权重包，
2026-11-04，`--realtime 20`。

| 运行 | 墙钟 | 转发 | 处理音频 | 吞吐 | 取流丢块 | 解码丢弃 | 进程 CPU 占用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CPU 无窗 | 20.30 s | 20.09 s | 20.40 s | 1.00 | 0 | 0 | 0.08 核 |
| CPU 有窗 | 20.33 s | 20.03 s | 20.08 s | 0.99 | 0 | 0 | 3.35 核 |

- 无窗只占 **0.08 核**：x64 单核比 Jetson 快得多，20 s 音频的前端与后台解码摊薄后
  几乎可以忽略。
- 有窗跳到约 **3.35 核**（`top` 单进程 331~360%，与 `/proc` 时间差一致，两口径互验）：
  入口每轮都 `imshow`（5.7，约 100 次/秒），单帧约 30 ms CPU 摊在多个线程上；而
  Jetson 原生 X11 有窗只到 0.70 核、渲染只增加约 0.4 核。线程数不是异常：无窗
  137 个、有窗 164 个，主体是 HCNetSDK/PlayCtrl 的线程池（无窗时也在），有窗只多
  约 27 个 OpenCV 显示线程，不存在每帧新建线程的泄漏。吞吐仍 0.99、两种丢块都为 0，
  说明它不破坏实时性，但带窗运行会明显抢占 CPU；远程 / 无显示器环境优先 `--no-window`。
- CUDA 组未在 x64 补测（本次放弃）。

### 断线与恢复

采集层是回调驱动，当前**不做自动重连**，行为如下：

- **启动流程**：`init()` 依次初始化 SDK、登录、查询编码、校验采样率，任一步失败
  立即返回 `false`，原因写入 `last_error()`；`start()` 再重建解码器并起流。
  登录失败不自动重试，上层可稍后重新调用 `init()`（已初始化的 SDK 与登录会话会复用）。
- **运行中断线判据（调用方自行判断）**：`get_stats()` 的 `packets` / `encoded_bytes`
  停止增长；注意设备静音时可能长时间不下发音频包，不能只凭「一段时间没数据」判定断线。
- **SDK 侧**：未启用 `NET_DVR_SetReconnect`；语音对讲会话断开后 SDK 不会自恢复，
  只能由上层重新 `stop()` 后 `start()`（登录会话仍在时不必重跑 `init()`）。
- **暂停恢复**：就是 `stop()` / `start()`；`start()` 对 error 29 按 200 ms 间隔重试最多
  5 次，其它错误直接失败；`StartVoiceCom` 本身同步阻塞约 1.2 秒。
- **保活失败**：只累加 `uplink_send_errors`，不触发重连；会话真失效后回调会再次停流。
- **断线期间输出**：默认开启静音补零（`pad_silence_gaps`）：按回调到达时刻检测
  停顿，在下一次回调前补等长 PCM16 0，让 wav 与 ASR 时间轴对齐墙钟。判定阈值
  200 ms，单次最多补 10 s；只补「确认丢失」的停顿，SDK 事后补发的积压音频不会
  重复补 0。统计见 `gap_padded_samples` / `gap_events`，正常连续推流应为 0。

## 频率与数据量

| 层级 | G.722.1 | PCM 32 kHz | 决定因素 |
| --- | --- | --- | --- |
| 设备编码 / 解码单元 | 25 单元/秒（40 ms/单元） | 无此概念 | SDK 固定 80 B → 1280 B |
| 回调包频率 | 实测约 12.7 包/秒 | 实测约 21 包/秒 | 设备推送与网络聚合，不固定 |
| 输出采样率 | 16000 采样/秒 | 32000 采样/秒（`output_sample_rate=16000` 时降为 16000） | 设备音频配置 + 输出配置 |
| 前端出帧 | 100 帧/秒（10 ms 帧移） | 同左（降采样后进入前端） | 前端固定 16 kHz、`frame_shift=10 ms` |
| ASR 编码块 | 每块 61 帧（610 ms），前进 48 帧（480 ms） | 同左 | `T` 与 `decode_chunk_len` metadata |
| 语谱图窗口 | 400 列 × 10 ms = 4 秒 | 同左 | `SpectrogramImage` 的 `cols` |

要点：

- **25 单元/秒不是 25 包/秒**。前者是 40 ms 解码单元的节奏，后者是实际回调次数；
  设备会把多帧聚合进一个回调包。完整链路与逐级推导见
  [`hk_mic_client/hk_audio_README.md`](hk_mic_client/hk_audio_README.md) 1.9 节。
- 设备编码单元由 SDK 的 80 B → 1280 B 约定和 16 kbps 码率共同决定；码率改为 32 kbps 后
  「80 B = 40 ms」不再成立。
- 前端出帧只由帧移决定，与回调包长度、读取间隔无关；语谱图 wav 入口默认 1x 播放，
  `--fast` 只改变喂入节奏，不改变特征结果；ASR wav 入口按固定 160 样本块尽快喂完。
- 实时入口主循环无新数据时等待 10 ms，与帧移一致；收到数据就喂前端，不按固定频率重复发送旧帧。

## 已知限制

| 位置 | 问题 | 影响 |
| --- | --- | --- |
| 各入口 `main` | 设备 IP、账号、通道、录制时长写死 | 换设备需改代码重新编译 |
| `HkSdkAudioClient` | 不做自动重连；未启用 `NET_DVR_SetReconnect` | 会话断开后需上层 `stop()` + `start()`；断流期间按到达时刻补 0 对齐墙钟 |
| `HkSdkAudioClient` | 每个实例都会 `NET_DVR_Init/Cleanup` | 同一进程不要同时创建音频与视频/热成像客户端 |
| `HkSdkAudioClient` | PCM 路径采样率被设备固定为 32 kHz（本机实测请求 16 kHz 无效） | 需要 16 kHz 时必须显式设 `output_sample_rate=16000`，由客户端降采样 |
| `HkSdkAudioClient` | 降采样只支持整数倍（32k/48k→16k） | 44.1k 等非整数比会拒绝启动 |
| `HkSdkAudioClient` | PCM 下行约 21.5 秒停止交付（单次 `StartVoiceCom` 会话上限）；持久配置 + 重启设备后复测仍停在 21.52 秒 / 471 包，45 秒运行只收到约 21.5 秒音频；重开语音转发可恢复 | PCM 单次实时 ASR 有效音频上限约 21.5 秒；长会话需每约 20 秒 `stop()` → `start()`；实时场景推荐 G.722.1，详见包 README 1.7.2 |
| `HkSdkAudioClient` | 主动设置编码写的是设备持久配置，且设备可能「成功但没生效」 | 默认关闭；开启后以读回复核为准，`deinit()` 写回原配置 |
| `HkSdkAudioClient` | PCM 下行时上行仍须回发 G.722 静音；持久 PCM + 重启设备后上行仍是 G.722.1（裸 PCM 保活 2560 B 第一秒即 0 包） | 不要按下行编码切换保活 payload；裸 PCM 上行不可用 |
| `HkSdkAudioClient` | `stop()` 会关闭当前 wav；`StartVoiceCom` 同步调用约阻塞 1.2 秒 | 连续录音不能跨暂停；在 UI 线程调用 `start()` 会卡顿，应放后台 |
| `HkSdkAudioClient` | 取流缓冲上限 40960 样本（约 2.56 s） | 调用 `get_last_audio()` 间隔过长会丢弃最旧样本 |
| `HkSdkAudioClient` | 静音补零依赖回调到达时刻（SDK 无设备时间戳），阈值 200 ms、单次上限 10 s；更长的断线只补前 10 s | 小于阈值的抖动不补；超过 10 s 的断线仍会让 wav 短于墙钟，超出部分丢弃后时间轴重新对齐 |
| `tools/wav_reader.h` | RIFF 解析只支持 PCM16 / float32，只取第一声道 | 其他编码或需要混音的 wav 不能直接使用 |
| `SpectrogramImage` | 信息条使用 `cv::putText`，只支持 ASCII | 非 ASCII 标签会被替换成 `?` |
| `test_wav_asr.cpp` | 模型按 16 kHz 训练，非 16 kHz 的 wav 只提示不重采样 | 采样率不对时识别结果不可用 |
| `test_mic_asr.cpp` | 输出采样率由 `output_sample_rate=16000` 保证，配置被误改时会直接报错退出 | 不要把它改回 0，否则 PCM 编码下无法识别 |
| 两个 ASR 入口 | 中文字体候选写死 Linux 常见路径（Noto / 文泉驿 / Droid） | 其他平台或精简系统找不到字体时回退 ASCII，中文显示为 `?` |
| 两个带识别文字的入口 | `draw_tail` 底层是 alpha 混合（`dst = dst * (1 - alpha/255)`），同一串字画在已经画过字的像素上会让边缘逐次变深 | `test_mic_asr.cpp` / `test_wav_asr.cpp` 都在画字前先刷白底条（保留首行黑线）；新增入口若复用「缓存画布 + 每轮重画」必须照做，否则文字会一深一浅地闪 |
| `AsrDecoder` | 非线程安全，跨块状态（116 个状态 + `hyp` + `decoder_out`）必须 `reset()` 才能换音频 | 多文件 / 多路复用时要按流隔离实例或串行调用 |
| `KwsDecoder` | 非线程安全；encoder 状态、beam、每条假设的 trie 状态与未消费特征尾巴都跨块存活 | 换音频必须 `reset()`；不要在多线程间共享同一个实例 |
| `KwsDecoder` / `test_wav_kws.cpp` | 触发要求关键词后至少 2 个 blank 输出帧（80 ms）；KWS 不使用 `input_finished()` 的特征域补 0 | 离线 wav 必须在**样本域**末尾补真静音（默认 0.5 s），否则句末关键词不触发 |
| `test_wav_kws.cpp` | 非 16 kHz 的 wav 直接判 FAIL、不重采样；不在期望表里的自定义文件只打印命中、不做集合检查 | 送检前先自行重采样到 16 kHz；新增回归样本时同步补 `kExpected` |
| `KwsDecoder` | 触发阈值与 boosting score 是逐关键词配置的，默认 0.25 / 1.0 | 误触发或漏触发要改关键词文件的 `#threshold` / `:score`，不能靠调全局参数 |
| KWS | 目前只有离线 wav 入口，尚未接入设备实时麦克风；也没有与 sherpa-onnx wheel 的 oracle 对拍 | 实时双模式与 oracle 对拍在后续计划中，现阶段以期望集合 + 块长不变性验收 |
| `test_mic_asr.cpp` | 解码在后台线程（有界队列 50 块，满时丢最旧，尾块 `push_final()` 永不丢弃）；`AsrDecoder` 只被该线程访问，主线程仅在 `join` 后读统计 | 其他入口仍是主循环同步解码；要改就参照本文件的 `AsrWorker`，不要在多线程里共享同一个 `AsrDecoder` |
| 两个 ASR 入口 | Jetson 实测 CUDA EP 反而更慢：每块只编码 61 帧，joiner/decoder 逐帧调用，116 个状态张量每次进出一遍 | 实时场景默认用 CPU；开 `--cuda` 前先用 `tools/bench_perf.sh --cuda` 实测 |
| 两个 ASR 入口 | 已改为 `init()` → 模型加载 → `start()`：开流时主循环已就绪，启动阶段不再丢音频 | Jetson 实测丢块 0；若再出现启动积压，先确认 `start()` 是否被提前调用 |
| 共享库 | 两个库以 SHARED 构建，Linux 下无需导出宏；ORT / OpenCV 不随安装复制 | Windows 构建需补 `__declspec(dllexport)`；部署机需自备 ORT 与 OpenCV 运行库 |
| `weight/`、`onnxruntime/install/` | 体积大且按模型/平台分发，不入库 | 克隆后需自行放置权重与 ORT 预编译库，否则推理目标无法构建 |
| `hk_mic_client/hk_sdk/` | 厂商预编译库，体积较大且按平台分目录 | 换平台需确认对应目录与运行期依赖完整 |

## 后续计划（未实施）

以下问题已确认但本轮不改，按优先级记录，动手前先用 `tools/bench_perf.sh`
建立基线、避免凭感觉优化。

| 优先级 | 问题 | 现状 | 候选方案 |
| --- | --- | --- | --- |
| 中 | CUDA EP 在 Jetson 上比 CPU 慢 1.9 倍 | 离线 RTF 0.5145 vs 0.2713；原因是每块 61 帧、joiner/decoder 逐帧、116 个状态张量反复搬运 | 试 fp16、TensorRT EP、增大编码块或合并小块推理；每次都用 `bench_perf.sh --cuda` 对比 |
| 中 | PCM 长会话没有自动分段重开 | 已确认 21.5 秒是单次 `StartVoiceCom` 会话上限（重启设备、持久配置后一致）；实测 25 秒 `stop()` → `start()` 后包数与取流恢复，但客户端未自动做 | 在 `HkSdkAudioClient` 内对 PCM 下行每约 20 秒自动重开语音转发；注意与录制收尾、暂停状态和 `StartVoiceCom` 约 1.2 秒阻塞协调 |
| 中 | 不做自动重连 | 断流后需上层 `stop()` → `start()`，登录会话未失效时不必 `init()` | 加 watchdog：`packets` 一段时间不增长即重连；必须抑制「设备静音不发包」的误判，阈值取设备最长静音间隔 |
| 中 | KWS 尚无实时入口 | 离线链路（trie + `KwsDecoder` + `test_wav_kws.out`）已通，设备流还没接 | 参照 `python_ref/test_mic_kws_asr.py` 增加「KWS 唤醒成功 → 切 ASR，ASR 静默 10 s / 结束语 → 切回 KWS」的主循环；切换时两个解码器各自 `reset()`，前端保持连续 |
| 低 | KWS 没有 oracle 对拍 | `--dump-hits` 已能导出关键词 / 首 token 时间 / 平均概率，但还没和 sherpa-onnx wheel 逐条比较 | 移植 `python_ref/test_kws_oracle_mini.py` 的做法，对拍 TSV 的关键词与时间戳 |
| 低 | 仓库自有唤醒词文件未落地 | C++ 入口目前只读权重包自带的 `test_wavs/keywords.txt`，不能改 `weight/` | 增加仓库级 `keywords/kws_keywords.txt`（运行时只读），入口默认指向它，权重包文件保留作回归基准 |
| 低 | 启动失败报错时机 | `init()` 的登录/编码错误能立即报；`StartVoiceCom` 失败要等模型加载完才报 | 若现场更在意早报错，可在 `init()` 后追加一次「起流 → 立即停流」探测，代价是约 1.2 秒 |
| 低 | 没有单元测试框架 | 全靠对拍脚本与自检入口，依赖人工执行 | 把 `compare_fbank.py`、`compare_asr.py` 与三个自检入口接入 CI |
| 低 | Windows 未实机验证 | 目录与链接路径已配置，共享库缺导出宏 | 补 `__declspec(dllexport)` 后在 Windows 上走一遍构建与冒烟 |

## 参考文档

| 文档 | 内容 |
| --- | --- |
| [`hk_mic_client/hk_audio_README.md`](hk_mic_client/hk_audio_README.md) | G.722.1 帧长、回调包、解码单元、PCM 对比、上行保活、init/start/stop/deinit 生命周期与失败分级 |
| [`python_ref/README.md`](python_ref/README.md) | Python 参考实现：前端七项 kaldi 口径、模型结构、贪心搜索、KWS 束搜索与唤醒词设置、流式等价性与实测数据 |
| [`onnxruntime/README.md`](onnxruntime/README.md) | 两个平台的 ONNX Runtime 安装、CUDA EP 与 cuDNN 9 配置 |
| [`weight/`](weight/) 各模型目录的 `README.md` | 权重来源、模型结构说明与官方 `test_onnx.py` |

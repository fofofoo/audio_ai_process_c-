# audio_ai_process（C++）

海康设备麦克风实时音频或 wav 文件 → LogFBank 特征 → 流式 zipformer2 ASR / OpenCV 滚动语谱图。
仓库同时提供「设备实时取流」与「wav 文件」两条数据源，共用同一套前端；
特征与识别结果都与 `python_ref/` 中的 Python 参考实现逐点、逐 token 对齐。

## 功能说明

1. **设备音频采集与录制**：通过 HCNetSDK 语音转发直接拿设备端拾音器的编码音频，
   不依赖本地声卡；G.722.1 由 SDK 解码成 16 kHz 单声道 PCM，录制开关可随时开始/结束，
   wav 先写 `.part` 再原子改名，异常中断不会留下看似完整的半截文件。
2. **实时取流接口**：`get_last_audio()` 一次取走自上次调用以来的全部 PCM16 样本，
   与是否正在录制无关；接口线程安全，可直接供前端、可视化或后续 ASR 消费。
3. **流式 LogFBank 前端**：按 kaldi `snip_edges=False` 口径实现分帧、povey 窗、
   逐帧预加重、512 点 FFT、80 通道 Mel 滤波和取对数；输入块长任意，出帧结果与块长无关。
4. **双数据源滚动语谱图**：设备实时流与 wav 文件都能画成 4 秒滚动窗口，
   viridis 伪彩、百分位色阶自动标定，画面可保存到 `results/`。
5. **数值对拍工具**：裸 float32 特征 dump + Python 参考实现逐点比较，
   用于确认 C++ 前端没有静默偏移；当前三条实测录音、多种块长均为逐位一致。
6. **ASR 模型加载自检**：链接项目内托管的 ONNX Runtime，依次加载 zipformer2
   transducer 的 encoder / decoder / joiner，打印输入输出契约与 metadata，
   并校验 `T`、`decode_chunk_len`、`context_size`、`vocab_size` 与词表行数；
   CPU 与 CUDA EP 两条路径均已实测通过。
7. **wav 流式语音识别**：三个 session 串成完整流水线，实现贪心搜索与文本还原；
   支持单文件、多文件或目录批量识别，窗口底部实时显示中文结果
   （FreeType 渲染，缺字体自动回退 ASCII），`--no-window` 可只跑文本。
   权重包自带四个 wav 与 Python 参考实现**逐 token 一致**，CPU 上 RTF ≈ 0.06。

**一句话亮点：从设备取流、流式前端到中文 ASR 与实时可视化全链路 C++，且与 Python 参考实现逐位、逐 token 对齐。**

## 组成

| 路径 | 一句话职责 |
| --- | --- |
| `hk_mic_client/hk_sdk_audio_client.{h,cpp}` | 海康设备音频采集库：登录、语音转发、G.722.1 解码、可选 wav 录制、取流缓冲与统计 |
| `hk_mic_client/hk_sdk/` | 厂商提供的 HCNetSDK 头文件与预编译库（按平台分目录） |
| `audio_ai_process/audio_preprocess.{h,cpp}` | 流式 LogFBank 前端：PCM 块 → `(n_frames, n_mels)` 特征 |
| `audio_ai_process/spectrogram_image.{h,cpp}` | 特征 → BGR 滚动语谱图画布 |
| `audio_ai_process/asr_decode.{h,cpp}` | 流式 ASR 解码：三模型推理、贪心搜索、文本还原、跨块状态 |
| `audio_ai_process/text_utils.{h,cpp}` | 通用小工具：UTF-8 编解码、字符分类、系统路径 |
| `audio_ai_process/text_renderer.{h,cpp}` | UTF-8 文本 → BGR 画布（FreeType 渲染中文，可选依赖） |
| `tools/wav_reader.h` | 最小 RIFF 解析，wav 入口与 ASR 入口共用 |
| `save_audio_wav.cpp` | 录 10 秒设备音频到 `results/` |
| `test_spectrogram_cv_mic.cpp` | 设备实时语谱图入口 |
| `test_spectrogram_cv_wav.cpp` | wav 文件语谱图入口 |
| `test_wav_asr.cpp` | wav 语音识别入口：文本 +（可选）语谱图与中文识别条 |
| `tools/fbank_dump.cpp` | 前端对拍工具：裸 float32 PCM → 裸 float32 特征 |
| `tools/compare_fbank.py` | 调 Python 参考实现与 C++ 工具逐点比较 |
| `tools/compare_asr.py` | 调 Python 参考实现与 C++ token 导出逐 token 比较 |
| `load_onnx_model.cpp` | ASR 三模型（encoder / decoder / joiner）加载自检入口 |
| `onnxruntime/` | 项目内托管的 ONNX Runtime；`install/` 不入库，接入方式见 `onnxruntime/README.md` |
| `weight/` | ASR 模型权重与词表，本地放置，不入库 |
| `python_ref/` | Python 参考实现（只读基准，不参与 C++ 编译） |
| `results/` | 运行产物：录音 wav 与语谱图 png |

数据流：

```
海康拾音器 --HCNetSDK 语音转发--> HkSdkAudioClient --PCM16--+
                                                            +--> /32768.0 --> AudioPreprocess
wav 文件 --tools/wav_reader.h（RIFF 解析）-------------------+                      |
                                                                                  v
                                                                    LogfbankFeatures (n,80)
                                                                                  |
                                        +-----------------------------------------+
                                        v                                         v
                          AsrDecoder（encoder/decoder/joiner）        SpectrogramImage
                                        |                                         |
                                        v                                         v
                                  文本（stdout / 窗口底部）                 BGR 画布 --> imshow
```

## 输入输出契约

### 音频输入

- **设备链路**：G.722.1，16 kHz、单声道、PCM16；设备地址、端口、账号、语音通道
  写在各入口 `main` 顶部，默认 `10.168.2.65:8000`、`channel=1`。
- **PCM 链路**：部分设备会把采样率固定为 32 kHz（请求 16 kHz 不生效），
  实际采样率以 `sample_rate()` 为准。
- **wav 文件**：支持 PCM16 与 float32，多声道只取第一声道；不做重采样，
  非 16 kHz 的 wav 可以画图，但喂 ASR 前必须自行重采样。

### 特征输出

- 类型 `float32`，行主序 `(n_frames, n_mels)`，第 i 帧第 j 维在 `data[i * n_mels + j]`。
- 单位是对数能量 `ln(能量)`，静音地板为 `ln(float32_eps) ≈ -15.9424`。
- 可视化路径构造前端时传 `tail_pad_frames=0`；解码路径保持默认 `100`。

### 公共接口

| 类 | 主要接口 | 说明 |
| --- | --- | --- |
| `hk_isapi::HkSdkAudioClient` | `start()` / `stop()` | 建立或断开语音转发，不涉及文件 |
| | `start_record(path)` / `stop_record()` | 开始/结束 wav 写盘，可多段录制 |
| | `get_last_audio(samples)` | 取走并清空累积 PCM16；无新数据返回 `false` |
| | `get_stats()` / `last_error()` / `sample_rate()` | 统计、错误与当前采样率 |
| `audio_ai::AudioPreprocess` | `accept_waveform(samples)` | 喂任意长度 float 波形，返回本次新产出的特征 |
| | `input_finished()` | 补尾帧；可视化传 `tail_pad_frames=0` |
| | `reset()` | 清空流式状态，构造参数与滤波器组保留 |
| `audio_ai::SpectrogramImage` | `accept_features(feats)` | 喂特征，返回整幅 BGR 画布 |
| | `frame()` / `reset()` | 按当前缓存重画 / 清空缓存 |
| `audio_ai::AsrDecoder` | `accept_features(feats)` | 喂特征，返回截至当前的完整文本；跨块状态内部维护 |
| | `text()` / `token_ids()` | 当前文本 / 已确认 token（供对拍） |
| | `reset()` / `info()` | 清空全部跨块状态 / 模型契约与计数 |
| `audio_ai::Utf8TextRenderer` | `init(path, face_index)` / `draw(...)` | 加载字体 / 把 UTF-8 文本画到 BGR 画布；不可用时回退 `cv::putText` |

## 运行与集成

### 依赖与编译

CMake ≥ 3.10、C++17、OpenCV（core / imgproc / imgcodecs / highgui）、pthread。
海康 SDK 已放在 `hk_mic_client/hk_sdk/`，CMake 按架构自动选择 `linux_x86` /
`linux64_arm` / `win64_x86`，并为可执行文件设置 RPATH。
FreeType 是 `test_wav_asr.out` 的可选依赖：找到时窗口底部用系统字体渲染中文，
找不到时该目标编译为空实现、自动回退 `cv::putText`，不影响识别。

ASR 相关目标额外依赖 ONNX Runtime，CMake 按架构选择项目内托管版本：
x86_64 用 `onnxruntime/install/onnxruntime-1.23.2`，aarch64 用
`onnxruntime/install/onnxruntime-1.18.0`（需在 Jetson 上源码编译）。
ORT 缺失时只给 WARNING，不影响音频链路；推理目标通过 `ort_link_target()` 接入。
CUDA EP 的 cuDNN 9 环境配置见 `onnxruntime/README.md`。

```bash
cmake -S . -B build
cmake --build build -j
```

已在 Linux x86_64（WSL2）验证；ARM 与 Windows 目录已配置，但未实测。

### 常用命令

```bash
# 录 10 秒设备音频 -> results/hk_audio_<时间戳>.wav
./build/save_audio_wav.out

# 设备实时语谱图：q / Esc 退出，s 存图
DISPLAY=:0 ./build/test_spectrogram_cv_mic.out

# 自检模式：跑 10 秒，存 results/spectrogram_mic.png 后退出
DISPLAY=:0 ./build/test_spectrogram_cv_mic.out 10

# wav 语谱图：默认按 1x 节奏播放，路径可省略
DISPLAY=:0 ./build/test_spectrogram_cv_wav.out results/hk_audio_20261102_085654.wav

# 批量检查：尽快喂完，存 results/spectrogram_wav.png 后退出
DISPLAY=:0 ./build/test_spectrogram_cv_wav.out results/hk_audio_20261102_085654.wav --fast

# ASR 三模型加载自检（默认非 int8、CPU）
./build/load_onnx_model.out

# 切换 int8 模型目录（encoder/joiner 自动识别 .int8.onnx）
./build/load_onnx_model.out weight/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05

# 验证 CUDA EP（需先按 onnxruntime/README.md 设置 cuDNN 9 的 LD_LIBRARY_PATH）
LD_LIBRARY_PATH=/home/liyun/code/AI/3rd/cudnn-9.10/lib ./build/load_onnx_model.out --cuda

# wav 语音识别：单个文件，窗口底部实时显示中文结果
DISPLAY=:0 ./build/test_wav_asr.out weight/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05/test_wavs/1.wav

# 批量识别：默认目录（权重包 test_wavs/）或目录/多文件，只打印文本
./build/test_wav_asr.out --no-window
./build/test_wav_asr.out --no-window weight/.../test_wavs

# 导出 token 供 Python 参考实现逐 token 对拍
./build/test_wav_asr.out --no-window --dump-tokens /tmp/cpp_tokens.txt

# CUDA EP（失败自动回退 CPU）
./build/test_wav_asr.out --no-window --cuda weight/.../test_wavs/1.wav
```

WSLg 下如果窗口打不开，先确认 `DISPLAY=:0`；环境里可能残留无效的远程 `DISPLAY`。
所有运行产物统一写入 `results/`。

### 前端对拍

```bash
python3 -I tools/compare_fbank.py --wav results/hk_audio_20261102_085654.wav
python3 -I tools/compare_fbank.py --chunks 160,17,333,1000,4096,0
```

脚本用同一分块节奏分别跑 `python_ref/src/audio_preprocess.py` 和 C++ 的
`fbank_dump.out`，输出 `mean|Δ|`、`max|Δ|` 与超差位置。判定阈值默认
`mean|Δ| ≤ 1e-5`、`max|Δ| ≤ 1e-3`；当前实测为 0。

### ASR 逐 token 对拍

```bash
# 1) C++ 侧导出 token
./build/test_wav_asr.out --no-window --dump-tokens /tmp/cpp_tokens.txt

# 2) Python 参考实现按同一块长（160 样本）解码并逐个 token 比较
/root/miniconda3/envs/pytorch-gpu/bin/python -I tools/compare_asr.py \
    --cpp /tmp/cpp_tokens.txt --wav-dir weight/.../test_wavs
```

`test_wav_asr.out` 每行写「文件名 <TAB> id,id,...」；脚本用 stdlib `wave` 读同一条
wav，依次跑 `python_ref/src/audio_preprocess.py` 与 `asr_decode.py`（尾部同样补
100 帧 0），逐 token 比较并打印 Python 侧文本。当前权重包自带的四个 wav
（27 / 20 / 23 / 32 token）全部一致；出现差异时会指出第几个 token、两侧 id。

### 模型加载自检

`load_onnx_model.out` 只验证「能否加载 + 接口是什么」，不做推理。三个模型各自
独立加载，单个失败只报 `ERROR` 并继续，最后按「全部成功且关键 metadata 齐全」
返回 0 / 1。默认读取 `weight/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-2026-06-05/`，
目录可用第一个非 `-` 参数覆盖；`--cuda` 尝试注册 CUDA EP，失败自动回退 CPU。

实测契约：

| 模型 | 输入 | 输出 |
| --- | --- | --- |
| encoder | `x` float32 `[-1, 61, 80]` + 116 个状态张量 | `encoder_out` `[-1, -1, 512]` + 116 个新状态 |
| decoder | `y` int64 `[-1, 2]` | `decoder_out` `[-1, 512]` |
| joiner | `encoder_out` + `decoder_out` `[-1, 512]` | `logit` `[-1, 5000]` |

关键 metadata：`model_type=zipformer2`、`T=61`、`decode_chunk_len=48`、
`context_size=2`、`vocab_size=5000`（与 `tokens.txt` 5000 行一致）。
即每块编码窗口 610 ms、块间前进 480 ms、重叠 130 ms。
状态张量的动态维对应 batch，建推理循环时填 1；符号维在 `load_onnx_model.out`
里照实打印，不臆测具体值。

## 已知限制

| 位置 | 问题 | 影响 |
| --- | --- | --- |
| 各入口 `main` | 设备 IP、账号、通道、录制时长写死 | 换设备需改代码重新编译 |
| `HkSdkAudioClient` | 每个实例都会 `NET_DVR_Init/Cleanup` | 同一进程不要同时创建音频与视频/热成像客户端 |
| `HkSdkAudioClient` | PCM 路径采样率可能被设备固定为 32 kHz | 喂 16 kHz ASR 前必须重采样 |
| `HkSdkAudioClient` | 取流缓冲上限 40960 样本（约 2.56 s） | 调用 `get_last_audio()` 间隔过长会丢弃最旧样本 |
| `tools/wav_reader.h` | RIFF 解析只支持 PCM16 / float32，只取第一声道 | 其他编码或需要混音的 wav 不能直接使用 |
| `SpectrogramImage` | 信息条使用 `cv::putText`，只支持 ASCII | 非 ASCII 标签会被替换成 `?` |
| `test_wav_asr.cpp` | 模型按 16 kHz 训练，非 16 kHz 的 wav 只提示不重采样 | 采样率不对时识别结果不可用 |
| `test_wav_asr.cpp` | 中文字体候选写死 Linux 常见路径（Noto / 文泉驿 / Droid） | 其他平台或精简系统找不到字体时回退 ASCII，中文显示为 `?` |
| `AsrDecoder` | 非线程安全，跨块状态（116 个状态 + `hyp` + `decoder_out`）必须 `reset()` 才能换音频 | 多文件 / 多路复用时要按流隔离实例或串行调用 |
| 当前阶段 | 已完成 wav ASR 推理；设备实时链路目前只到特征与语谱图 | 麦克风实时识别还未接入，后续与拆库一起做 |
| `weight/`、`onnxruntime/install/` | 体积大且按模型/平台分发，不入库 | 克隆后需自行放置权重与 ORT 预编译库，否则推理目标无法构建 |
| `hk_mic_client/hk_sdk/` | 厂商预编译库，体积较大且按平台分目录 | 换平台需确认对应目录与运行期依赖完整 |

## 后续计划

1. **麦克风实时识别**：把设备取流、前端与 `AsrDecoder` 串起来，边说话边出字；
   注意设备 PCM 可能被固定为 32 kHz，需要先重采样到 16 kHz。
2. **拆成两个库**：ASR 处理库（初始化 / reset / 喂 PCM / 取文本）与麦克风采集库
   （打开设备、按块输出 PCM），两库零耦合。

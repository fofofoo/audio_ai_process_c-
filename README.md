# audio_ai_process（C++）

海康设备麦克风实时音频 → LogFBank 特征 → OpenCV 滚动语谱图的 C++ 实现。
仓库同时提供「设备实时取流」与「wav 文件」两条数据源，二者共用同一套前端与可视化；
特征口径与 `python_ref/` 中的 Python 参考实现逐点对齐。

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

**一句话亮点：从设备取流、流式前端到实时可视化全链路 C++，且与 Python 参考实现逐位对齐。**

## 组成

| 路径 | 一句话职责 |
| --- | --- |
| `hk_mic_client/hk_sdk_audio_client.{h,cpp}` | 海康设备音频采集库：登录、语音转发、G.722.1 解码、可选 wav 录制、取流缓冲与统计 |
| `hk_mic_client/hk_sdk/` | 厂商提供的 HCNetSDK 头文件与预编译库（按平台分目录） |
| `audio_ai_process/audio_preprocess.{h,cpp}` | 流式 LogFBank 前端：PCM 块 → `(n_frames, n_mels)` 特征 |
| `audio_ai_process/spectrogram_image.{h,cpp}` | 特征 → BGR 滚动语谱图画布 |
| `save_audio_wav.cpp` | 录 10 秒设备音频到 `results/` |
| `test_spectrogram_cv_mic.cpp` | 设备实时语谱图入口 |
| `test_spectrogram_cv_wav.cpp` | wav 文件语谱图入口（自带最小 RIFF 解析） |
| `tools/fbank_dump.cpp` | 前端对拍工具：裸 float32 PCM → 裸 float32 特征 |
| `tools/compare_fbank.py` | 调 Python 参考实现与 C++ 工具逐点比较 |
| `python_ref/` | Python 参考实现（只读基准，不参与 C++ 编译） |
| `results/` | 运行产物：录音 wav 与语谱图 png |

数据流：

```
海康拾音器 --HCNetSDK 语音转发--> HkSdkAudioClient --PCM16--+
                                                            +--> /32768.0 --> AudioPreprocess
wav 文件 --入口内置 RIFF 解析--------------------------------+                      |
                                                                                  v
                                                                    LogfbankFeatures (n,80)
                                                                                  |
                                                                                  v
                                                            SpectrogramImage --> BGR 画布 --> imshow
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

## 运行与集成

### 依赖与编译

CMake ≥ 3.10、C++17、OpenCV（core / imgproc / imgcodecs / highgui）、pthread。
海康 SDK 已放在 `hk_mic_client/hk_sdk/`，CMake 按架构自动选择 `linux_x86` /
`linux64_arm` / `win64_x86`，并为可执行文件设置 RPATH。

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

## 已知限制

| 位置 | 问题 | 影响 |
| --- | --- | --- |
| 各入口 `main` | 设备 IP、账号、通道、录制时长写死 | 换设备需改代码重新编译 |
| `HkSdkAudioClient` | 每个实例都会 `NET_DVR_Init/Cleanup` | 同一进程不要同时创建音频与视频/热成像客户端 |
| `HkSdkAudioClient` | PCM 路径采样率可能被设备固定为 32 kHz | 喂 16 kHz ASR 前必须重采样 |
| `HkSdkAudioClient` | 取流缓冲上限 40960 样本（约 2.56 s） | 调用 `get_last_audio()` 间隔过长会丢弃最旧样本 |
| `test_spectrogram_cv_wav.cpp` | RIFF 解析只支持 PCM16 / float32，只取第一声道 | 其他编码或需要混音的 wav 不能直接使用 |
| `SpectrogramImage` | 信息条使用 `cv::putText`，只支持 ASCII | 非 ASCII 标签会被替换成 `?` |
| 当前阶段 | 尚未接入 ASR 推理 | 目前只能得到特征与语谱图，没有识别文本 |
| `hk_mic_client/hk_sdk/` | 厂商预编译库，体积较大且按平台分目录 | 换平台需确认对应目录与运行期依赖完整 |

## 后续计划

1. **wav ASR**：链接 onnxruntime C++ 库，移植 `python_ref/src/asr_decode.py` 的
   贪心搜索与文本还原，与 Python 版逐 token 对拍。
2. **拆成两个库**：ASR 处理库（初始化 / reset / 喂 PCM / 取文本）与麦克风采集库
   （打开设备、按块输出 PCM），两库零耦合。

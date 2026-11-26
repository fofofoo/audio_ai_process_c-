# kws_asr_processor

海康设备实时语音处理节点：KWS 关键词唤醒 + ASR 连续识别，结果发布到 ROS 话题。
算法实现来自 `audio_ai_process` 仓库的两个库，本包只做 ROS 封装（参数、话题、主循环）。

## 功能说明

1. **设备实时取流**：通过 HCNetSDK 语音转发从海康拾音器取音频，
   默认 G.722.1（16 kHz），客户端内部做降采样与静音补零，取流缓冲约 2.56 s。
2. **双模式识别**：平时只跑 KWS 关键词唤醒，命中唤醒词后切到 ASR 完整识别，
   出现结束语或空闲 10 s 无新 token 自动切回监听。
3. **ROS 话题输出**：唤醒 / 结束语 / 超时事件发布到 `/audio_ai/kws_event`，
   识别文本变化时发布到 `/audio_ai/asr_text`（均为 `std_msgs/String`）。
4. **转写落盘**：ASR 每 64 个 token 结算一段，追加写入 `log/asr_last_text.txt`。
5. **可选可视化**：`show_window:=true` 时弹出滚动语谱图窗口，窗口内
   `q` / `Esc` 退出、空格暂停 / 恢复语音转发。

## 话题

| 话题 | 类型 | 方向 | 说明 |
|---|---|---|---|
| `/audio_ai/asr_text` | `std_msgs/String` | 发布 | 当前 ASR 会话文本，只在变化时发布；切回 KWS 后为空串 |
| `/audio_ai/kws_event` | `std_msgs/String` | 发布 | 唤醒 / 结束语 / 超时切回事件，每个事件发布一次 |

本节点没有订阅话题，音频源是设备语音转发。

## 双模式切换框架

```
KWS 监听 --命中唤醒词--> ASR 识别
   ▲                        │
   └── 结束语 / 空闲 10 秒 ──┘
```

## 核心音频链路

图中的编号对应后面的逐步说明。实线是音频和特征的流向，右下角那条回线是结果的流向。

```
① 设备 / wav
      │ 16 kHz PCM16
      ▼
② HkSdkAudioClient / wav_reader
      │ PCM16
      ▼
③ AudioPreprocess（LogFBank）
      │ LogFbankFeatures (n, 80)
      ├────────────────────────────────┐
      ▼                                ▼
④ SpectrogramImage        ⑤ BlockWorker 音频特征输入队列
   （主线程渲染）            （有界，满时丢最旧）
                                         │
                                         ▼
                                  ⑥ 后台线程（唯一）
                                     session.on_block(输入块)
                                         │
                                         ▼
                                  ⑦ 业务 Session
                                     ├ 单模式：AsrSession
                                     └ 双模式：KwsAsrSession
                                         │
                                         ▼
                                  ⑧ 快照（text / mode / 事件）
                                         │
                                         ▼
                                  ⑨ 主线程读快照显示
```

### 文件级链路：从采集到解码器

落到具体文件，调用关系如下。`hk_sdk_audio_client.h`、`wav_reader.h`、`audio_preprocess.h`、`spectrogram_image.h` 属于 `lib_hk_mic_client`；`block_worker.h` 及其之后的 Session、Decoder 属于 `lib_audio_ai_process`。

```
设备 / wav 文件
   │
   ▼
hk_sdk_audio_client.h   麦克风采集：登录 / 起流 / G.722.1 解码 / 取流缓冲
   │
   ▼
audio_preprocess.h      前端：PCM → LogfbankFeatures (n, 80)
   │
   ├───────────────────────────┐          AI音频处理，语音转文字数据流
   ▼                           ▼
spectrogram_image.h      block_worker.h   数据入口：管理音频特征输入队列：定上限、丢最旧、尾块阻塞、空块过滤、线程生命管理
 可视化:特征 → 画布             │
                               ▼
                            Session       业务分发、数据出口：KWS / ASR 输入分发,汇总输出，写入快照，供主线程提取
                 kws_asr_session.h/asr_session.h
                               │
                               ▼
                             Decoder      业务核心：接受短期音频特征(480ms),调用各类模型协作，输出文字
                 asr_decode.h / kws_decode.h

```

采集库只产出 `LogfbankFeatures`，不依赖 ASR 库；并发worker、Session、解码器与快照都在`audio_ai_process` 一侧。设备与 wav 两条数据源在前端汇合，前端输出再分给可视化与管家。

## ASR文本数据链路与快照机制
```
入口 main（主线程，装配）
  AsrDecoder / KwsDecoder → AsrSession → KwsAsrSession → BlockWorker
  每轮 push(特征块)；收尾 push_final(尾块) → finish_and_join()
        │（有界队列：满时丢最旧，0 帧不入队）
        ▼
Worker 线程（唯一，串行）：KwsAsrSession.on_block(item)
  ├ 1.1 执行 KWS 模式 → kws_decoder_.accept_features()；命中 → enter_asr()
  └ 1.2 执行 ASR 模式 → AsrSession::on_block → asr_decoder_.accept_features()
  │                      ├ all_ids_（全量）→ 切回时 current_text() → last_text / finish()
  │                      ├ segment_ids_（每满 64 → 写一行 txt）
  │                      └ recent_text_（最近 128）→ snap_.text
  └ 2.0 刷新 KwsAsrSession.snapshot 快照（ASR 模式每块刷新；KWS 仅命中唤醒 / 切回时刷新）：
        ├ mode      ← Kws / Asr
        ├ text      ← ASR：recent_text_（最近 128）；KWS：空串
        ├ last_text ← 切回时 text_of(all_ids_) 全量（back_to_kws / finish 时更新）
        ├ 计数      ← wake_count / back_count
        ├ 事件      ← event_seq / last_event（唤醒 / 结束语 / 超时）
        └ snapshot() → 整份拷贝（短锁）→ 主线程

主线程（稳态每轮）
  单模式：latest_text()（128 尾部窗口）
  双模式：snapshot()（整份拷贝，text 为最近 128 窗口）
  ROS 节点：snapshot() → 与上次比较去重 → 发话题
主线程（join 后）：finish() / all_token_ids() / stats()
```
## 三类关键文本数据
```
all_ids_（触发asr识别后的所有文本数据，唯一事实来源）
  ├─ 切回时 current_text() ─► snap_.last_text / 事件（一轮一次）
  └─ finish() ─────────────► 最终完整文本

segment_ids_（all_ids_ 的当前段切片，≤64，用于落盘保存）
  └─ close_segment() ─► text_of(segment_ids_) ─► log/asr_last_text.txt（一行一段）

recent_tokens_（最近 128 个，all_ids_ 的尾部窗口，用于显示）
  ├─ recent_text_ ─► snap_.text ─► snapshot() ─► 窗口 / stdout / 话题（双模式）
  └─ recent_text_ ─► AsrSession::snapshot_ ─► latest_text()（单模式显示）
```
## launch 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `ip` | `10.168.2.65` | 音频设备 IP |
| `sdk_port` | `8000` | HCNetSDK 私有协议端口 |
| `username` / `password` | `admin` / `Password` | 登录凭据 |
| `voice_channel` | `1` | 语音通道号 |
| `audio_ai_root` | `/home/liyun/code/AI/audio_ai_process` | 算法仓库根（库与权重都在其下） |
| `asr_model_dir` | `<root>/weight/sherpa-onnx-x-asr-...-int8-2026-06-05` | ASR 权重目录 |
| `kws_model_dir` | `<root>/weight/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20` | KWS 权重目录 |
| `keywords` | `<root>/audio_ai_process/keywords/kws_keywords.txt` | 唤醒词文件 |
| `transcript_path` | `<root>/log/asr_last_text.txt` | ASR 分段转写路径 |
| `initial_asr` | `false` | true = 从 ASR 模式起步（调试纯识别） |
| `idle_timeout` | `10.0` | ASR 空闲多少秒切回 KWS |
| `backfill_frames` | `100` | 唤醒时回填帧数（10 ms/帧） |
| `use_cuda` | `false` | ASR / KWS 尝试 CUDA EP，失败回退 CPU |
| `show_window` | `false` | 是否显示语谱图窗口 |

## 构建

先准备 `audio_ai_process` 仓库的两个库安装产物（只需一次）：

```bash
cd /home/liyun/code/AI/audio_ai_process
cmake -S hk_mic_client -B hk_mic_client/build && cmake --build hk_mic_client/build -j
cmake --install hk_mic_client/build
cmake -S audio_ai_process -B audio_ai_process/build && cmake --build audio_ai_process/build -j
cmake --install audio_ai_process/build
```

再编译本包（工作区就在本仓库内）：

```bash
cd /home/liyun/code/AI/audio_ai_process/catkin_ws
catkin_make
```

`CMakeLists.txt` 默认从本包位置向上三级定位仓库根（本包 → src → catkin_ws →
仓库根），整个仓库搬目录时不用改；特殊布局下可覆盖：

```bash
catkin_make -DAUDIO_AI_ROOT=/path/to/audio_ai_process
```

## 运行

```bash
cd /home/liyun/code/AI/audio_ai_process/catkin_ws
source devel/setup.bash
roslaunch kws_asr_processor kws_asr_processor.launch
roslaunch kws_asr_processor kws_asr_processor.launch use_cuda:=true
roslaunch kws_asr_processor kws_asr_processor.launch ip:=10.168.2.66
```

观察话题：

```bash
rostopic echo /audio_ai/asr_text
rostopic echo /audio_ai/kws_event
```

## 已知限制

- 设备语音通道同一时刻只能被一个进程占用；`StartVoiceCom` 报 error 29 时先
  `pgrep` 找占用进程，不要直接 kill 别人启动的程序。
- 节点退出才打印收尾统计（时长、编码调用、丢块、峰值幅度）；正常运行只有
  事件与文本的实时打印。
- 模型文件名与权重包版本绑定，换权重包时需要同步改 `src/main.cpp` 里的文件名常量。

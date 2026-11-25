# audio_ai_processor

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
roslaunch audio_ai_processor audio_ai_processor.launch
roslaunch audio_ai_processor audio_ai_processor.launch use_cuda:=true
roslaunch audio_ai_processor audio_ai_processor.launch ip:=10.168.2.66
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

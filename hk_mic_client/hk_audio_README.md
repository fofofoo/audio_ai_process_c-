# 音频链路：帧长、回调包与解码单元

本文说明海康 HCNetSDK 语音转发链路里三个最容易混淆的概念：**G.722.1 协议帧**、**海康 SDK 的解码单元**、**网络回调包**。它们分别是 20 ms、40 ms 和「任意长度」，混为一谈会直接导致误判码率、误算包频率、误写 wav 头。

全文围绕一条链路展开：第一章拆开帧长、回调包与解码单元三个易混概念，并依次展开两个核心接口（`NET_DVR_StartVoiceCom_MR_V30` 与 G.722.1 解码接口族）、两条数据路径、40 ms 的成立条件、频率与数据量、常见误解和已知限制；第二章按生命周期讲客户端的落地实现（start 六步 → 回调分流 → stop 收尾）。接入方式、编码选型、录音流程见根 [README.md](../README.md)，客户端实现见 [hk_sdk_audio_client.cpp](hk_sdk_audio_client.cpp)。

## 结论速查

| 概念 | 由谁定义 | 具体值 | 说明 |
| --- | --- | --- | --- |
| G.722.1 协议帧 | ITU-T 标准 | **20 ms / 320 采样点**（16 kHz 单声道） | 编码器的基本帧，标准码率 24 / 32 kbps |
| 海康解码单元 | SDK 接口约定 | **输入 80 字节 → 输出 1280 字节** | `NET_DVR_DecodeG722Frame` 一次调用的固定比例 |
| 40 ms | 换算结果 | 640 个 PCM16 采样点 | 80 B ÷ 16 kbps = 40 ms，仅当前设备码率成立 |
| G.722.1 回调包 | SDK 回调线程 | 长度任意，实测约 **168 B / 包** | 不是帧，应用层必须重新按 80 B 组帧 |
| PCM 回调包 | 设备 PCM16 流 | 长度任意，实测约 **3031 B / 包** | 没有帧概念，直接按字节追加 |

**一句话：40 ms 是「海康解码接口的 80 字节单元 × 当前 16 kbps 码率」算出来的，不是 G.722.1 协议规定的帧长；回调网络包更是与它无关。**

## 一、音频链路：帧长、回调包与解码单元

这三个概念最常被合成一句话——「G.722.1 一帧 80 字节、40 毫秒」——然后直接拿去算码率、算包频率、写 wav 头。问题在于这句话把三个来源不同的量混在了一起：20 ms 来自 ITU-T 标准，40 ms 来自海康接口的固定换算，而回调包长度根本没有任何一方保证。混用的后果并不抽象：统计表对不上、把 12 秒的录音标成 6 秒、wav 头采样率写错。本章把三层拆开，并在最后说明它们各自影响代码的哪一部分。

```
① 协议层       G.722.1 编码帧：20 ms / 320 采样点（标准）
                         │
② 接口层       海康 SDK：固定 80 B 输入 → 固定 1280 B 输出 = 640 点 / 40 ms
                         │
③ 传输层       NET_DVR_StartVoiceCom_MR_V30 回调：长度任意的字节块
```

三者的关系是：设备把连续码流按**任意长度**推给回调；应用层攒够 80 字节才交给解码器；解码器固定吐出 1280 字节。**40 ms 只出现在第②层。**

### 1.1 协议层：G.722.1 是 20 ms 一帧

ITU-T G.722.1 在 16 kHz 采样下每帧 320 个采样点，即 20 ms。标准码率为 24 / 32 kbps（16 kbps 属于 Siren7 等扩展档）：

- 32 kbps × 20 ms = 640 bit = **80 字节**；
- 16 kbps × 20 ms = 320 bit = **40 字节**。

所以在标准 G.722.1 里，“80 字节”对应的是 **20 ms**，而不是 40 ms。

这里要注意换算的方向：16 kHz 采样下，一帧固定是 320 个采样点、固定 20 ms，与码率无关；码率只改变这一帧被压成多少字节。所以「一帧永远 20 ms」，「多少字节」取决于码率。16 kbps 时标准帧只有 40 字节，而海康接口要求一次喂 80 字节——这 80 字节要么是 40 ms 的私有帧，要么是两个 20 ms 标准帧拼在一起，接口层面无从区分。

### 1.2 接口层：海康硬性规定 80 B → 1280 B

《设备网络SDK使用手册》接口页《NET_DVR_DecodeG722Frame》原文：

> `pInBuffer`：[in] 输入缓冲区，编码数据大小为 **80 字节**
>
> `pOutBuffer`：[out] 输出缓冲区，按采样标准（采样频率为 16000，16 位采样，单通道）获取的 PCM 音频数据，规定输出数据的大小为 **1280 字节**

1280 字节 = 640 个 PCM16 采样点 = 16 kHz 下 **40 ms**。同样的约定也出现在《NET_DVR_EncodeG722Frame》（输入 1280 字节 PCM、输出 80 字节）和《NET_DVR_AUDIODEC_PROCESS_PARAM》（`out_frame_size`：G711 输入 320 字节、G722 输入 1280 字节）。

即：**80 字节是海康解码 API 的调用单元，不是协议帧长。** 手册没有说明设备底层是按 20 ms 帧编码后两帧聚合，还是私有 40 ms 帧；从「16 kbps × 40 ms = 80 B」的数值关系看，更可能是两个标准 20 ms 帧聚合，但这一点属于推断，手册未写明。

SDK 为什么固定这个比例，手册没有解释，但从接口形态可以推断：输入固定、输出固定，调用方就能一次性分配好缓冲，解码函数既不用返回「需要多大空间」，也不用处理变长输出。代价是这套接口隐含了码率假设——它按 16 kbps、40 ms 的数据量设计了 80 字节入口；设备码率一变，字节数与时长的对应关系随之改变，而接口本身不会报错。

### 1.3 传输层：回调包长度任意

`byAudioFlag == 1` 时回调给出的是**连续码流中的一段**，SDK 不保证包长：

| 实测场景 | 包数 | 总字节 | 平均包长 | 折算包时长 |
| --- | --- | --- | --- | --- |
| G.722.1，6 秒录音 | 76 包 | 12800 B | ≈168 B | ≈84 ms（16 kbps） |
| PCM 32 kHz，5 秒录音 | 106 包 | 321280 B | ≈3031 B | ≈47 ms（512 kbps） |

数据来自根 [README.md](../README.md) 的实测记录。两列都说明：**回调包边界与 40 ms、与 80 字节都不对齐**，应用层不能拿“一包”当“一帧”。

三层分开之后，各自的职责就清楚了：协议层的 20 ms 决定编码器的最小处理单位，但我们拿不到也用不上；接口层的 80 B → 1280 B 决定解码怎么调用、一次出来多少 PCM；传输层的包长只影响缓冲怎么攒、统计里「包数」这个指标怎么解读。至于 wav 头里的采样率，它由设备音频配置单独决定，是这三层之外的第四个量，必须另外向设备查询，不能由任何一层推导出来。

### 1.4 核心接口 1：NET_DVR_StartVoiceCom_MR_V30

第二章 2.2 的 start 第 6 步和整条回调链都绕不开这个接口。它是整条链路唯一「出数据」的入口：登录、查编码、开文件都是为它做准备，解码、统计、写 wav 都是在处理它推回来的数据。它位于 [hk_sdk_audio_client.cpp:204](hk_sdk_audio_client.cpp#L204)。

#### 1.4.1 它是什么

海康 HCNetSDK 的**设备端语音对讲（多路，V30）**接口。名字拆开看：

| 片段 | 含义 |
| --- | --- |
| `NET_DVR_` | HCNetSDK 私有协议（默认 8000 端口）的网络接口前缀 |
| `VoiceCom` | 语音对讲，不是「录音」专用接口 |
| `MR` | Multi-Route，多路，具体通道由参数指定 |
| `V30` | 第三代结构，替代旧的 `NET_DVR_StartVoiceCom` |

它**不是**独立客户端或进程，调用它只做三件事：向设备发起对讲请求、登记回调、返回一个句柄。音频不经过返回值，全部由 SDK 回调线程异步推送。

#### 1.4.2 四个参数

| 参数 | 代码中的实际值 | 含义与取值依据 |
| --- | --- | --- |
| `lUserID` | `m_user_id` | `NET_DVR_Login_V40` 成功后的用户 ID，`< 0` 表示尚未登录 |
| `dwVoiceChan` | `m_config.voice_channel`（默认 1） | 语音通道号，本机实测为 1；不是摄像头通道号 |
| `cbVoiceDataCallBack` | `&HkSdkAudioClient::voice_callback` | C 风格函数指针，必须是静态函数；成员函数无法直接取地址 |
| `pUser` | `this` | 回调时原样回传，用来把静态跳板接回实例 |

对应的回调签名由 SDK 固定，不能改：

```cpp
void CALLBACK f(LONG  lVoiceComHandle,   // 与返回值相同的对讲句柄
                char* pRecvDataBuffer,   // 音频数据，长度见下一个参数
                DWORD dwBufSize,         // 本次字节数，长度任意
                BYTE  byAudioFlag,       // 音频来源标志，见 2.4
                void* pUser);            // 调用时传入的 this
```

#### 1.4.3 返回值与失败处理

- 返回 `LONG` 句柄，`< 0` 失败，失败码用 `NET_DVR_GetLastError()` 取；
- 句柄只供 `NET_DVR_StopVoiceCom()` 使用，不参与数据读取；
- 失败收尾（[hk_sdk_audio_client.cpp:224-227](hk_sdk_audio_client.cpp#L224-L227)）：撤销 `m_running` → 记录 `last_error` 并返回 `false`；`start()` 不打开 wav，因此没有半截文件需要收尾。若本次主动改过设备编码，还会先写回原配置（见 [2.9](#29-主动设置编码可选)）。

#### 1.4.4 byAudioFlag 是最关键的一个字节

| 值 | 含义 | 处理 |
| --- | --- | --- |
| 1 | 设备端编码后的音频（拾音器） | 接收，进入解码或直接写盘 |
| 其他 | 本地麦克风、控制数据等 | 丢弃，否则会把噪声或控制帧写进 wav |

客户端在 [hk_sdk_audio_client.cpp:840](hk_sdk_audio_client.cpp#L840) 做的第一个判断就是 `flag != 1` 直接返回。这个判断比帧长、包长都重要：漏判一次，整段 wav 就会被无关数据污染。

#### 1.4.5 为什么是这个接口，而不是另外几个

| 接口 | 行为 | 为什么不用 |
| --- | --- | --- |
| `NET_DVR_StartVoiceCom_V30` | 语音对讲，依赖本地声卡播放 | 服务器 / WSL 无桌面环境没有可用声卡 |
| `NET_DVR_ClientAudioStart_V30` | 客户端广播 / 监听 | Linux 下不可用 |
| `NET_DVR_StartVoiceCom_MR_V30` | 多路对讲，回调直接给设备编码音频 | **本项目的选择**：无需声卡，纯网络录音 |

#### 1.4.6 时序约束

```
start()：m_running.store(true)        ← 必须先置位
           → StartVoiceCom_MR_V30     ← SDK 可能在本函数返回前就触发首包
           → 回调 flag==1 → 检查 m_running → 处理

stop()： NET_DVR_StopVoiceCom         ← 必须先停，阻断新回调
           → 锁内冲刷残包 → close_wav ← 之后才允许 fclose
```

两个顺序都不能反：

- 运行标志晚于启动 → 首包被 `m_running == false` 丢掉；
- `fclose` 早于 `StopVoiceCom` → 已排队的回调仍可能 `fwrite` 到悬垂 `FILE*`。

另外 `NET_DVR_Init` / `NET_DVR_Cleanup` 是实例级的，本类与可见光/热成像客户端混跑会互相清理，详见项目 [CLAUDE.md](../CLAUDE.md) 约束 1。

#### 1.4.7 上行保活：只收不发约 10 秒后设备停流

手册《语音对讲转发广播模块》写明语音转发是**双向**的：「除了接收设备发送过来语音数据，也需要发送语音数据给设备」，发送用 `NET_DVR_VoiceComSendData`（G.722 每次 80 字节）。

实测（`test_mic_asr` 与独立探针，2026-11-03）：只收不发时，设备稳定送满约 **10.5 秒**（180 个回调包 / 168960 样本，`stream_dropped_samples = 0`）后回调彻底停止；窗口停止滚动、识别不再出字，但主循环与取流缓冲都正常。每 40 ms 回发一帧 **G.722 静音**后，连续 30 秒以上回调不断。

实现要点：

- 静音帧由 `NET_DVR_InitG722Encoder` + `NET_DVR_EncodeG722Frame` 把 640 个 0 采样点编成 80 B，只编一次、线程只读；
- 保活线程每 40 ms 调一次 `NET_DVR_VoiceComSendData`，句柄用原子量传递；
- `stop()` 必须先停线程再 `NET_DVR_StopVoiceCom`，否则线程可能往正在销毁的句柄发数据；
- 统计里 `uplink_sent_frames` / `uplink_send_errors` 可判断保活是否正常；
- 配置项 `uplink_keepalive` 默认开启；PCM 路径未实测，暂不启用。

### 1.5 核心接口 2：G.722.1 解码接口族

第二章 2.3 的 G.722.1 分支里，`decode_pending_locked` 循环调用的就是这一族接口。`StartVoiceCom` 负责把编码字节取回来，解码接口负责把编码字节变成 PCM——没有它，G.722.1 路径就只是一个不断变长的二进制缓冲。

#### 1.5.1 三个接口

| 接口 | Linux 签名 | 作用 | 调用次数 |
| --- | --- | --- | --- |
| `NET_DVR_InitG722Decoder` | `void* InitG722Decoder()` | 申请一个解码器句柄，内部保存自适应预测的历史状态 | 每个实例一次 |
| `NET_DVR_DecodeG722Frame` | `BOOL DecodeG722Frame(void* handle, NET_DVR_AUDIODEC_PROCESS_PARAM* param)` | 送入 80 字节编码数据，输出 1280 字节 PCM | 每 80 字节一次 |
| `NET_DVR_ReleaseG722Decoder` | `void ReleaseG722Decoder(void* handle)` | 释放句柄 | 每个实例一次 |

客户端里的位置：

| 动作 | 位置 |
| --- | --- |
| 创建 | [hk_sdk_audio_client.cpp:152](hk_sdk_audio_client.cpp#L152)（`start()` 第 4 步） |
| 解码 | [hk_sdk_audio_client.cpp:904](hk_sdk_audio_client.cpp#L904)（`decode_pending_locked()` 循环内） |
| 释放 | [hk_sdk_audio_client.cpp:76](hk_sdk_audio_client.cpp#L76)（析构） |

创建放在 `start()`、释放在析构，而不是成对放在 `start()`/`stop()`：`stop()` 之后仍然可以再次 `start()`，句柄可以复用，没必要反复申请释放。

#### 1.5.2 参数结构体：哪些是输入，哪些是输出

`NET_DVR_AUDIODEC_PROCESS_PARAM` 是解码的唯一入口，字段方向容易记错：

| 字段 | 方向 | 填什么 |
| --- | --- | --- |
| `in_buf` | 输入 | 指向**恰好一帧**编码数据，本项目中是 `m_encoded_buf.data()` |
| `in_data_size` | 输入 | 这一帧的字节数，固定 80 |
| `out_buf` | 输入 | 输出缓冲区，本项目中是 `m_pcm_buf.data()` |
| `out_frame_size` | 输入 + 输出 | 传入缓冲区容量（5120），解码后回填实际写入字节数 |
| `proc_data_size` | **输出** |  SDK 回填本次实际消费的编码字节数 |
| `dec_info` | 输出 | 解码信息，本项目未使用 |
| `g711_type` | 输入 | 只对 G.711 有意义，G.722.1 忽略；保留 1 与已验证的探针程序一致 |

一个字段的双重身份最容易踩坑：`out_frame_size` 传进去是“我能放多少”，读出来是“实际写了多少”。所以代码里先用 `m_pcm_buf.size()` 赋值，解码后再拿它做统计和 `fwrite` 的长度。

#### 1.5.3 一次解码的完整过程

```
m_encoded_buf 头部 80 字节
  → param.in_buf / in_data_size = 80
  → NET_DVR_DecodeG722Frame(handle, &param)
      ├─ 返回 FALSE → 错误码 NET_DVR_GetLastError，丢一帧
      └─ 返回 TRUE
           ├─ proc_data_size  = 本次消费的编码字节（通常 80）
           ├─ out_frame_size  = 实际写出的 PCM 字节（通常 1280）
           └─ m_pcm_buf 里是 640 个 PCM16 采样点 = 16 kHz 下的 40 ms
  → 按 proc_data_size 从缓冲头部 erase
  → 按 out_frame_size 写 wav
```

数据量映射固定为 **80 B → 1280 B → 640 点 → 40 ms**，这一条在 1.2 节已经说明，是 SDK 的硬性约定，不是协议帧长。解码器本身不关心时间，它只认“进去多少字节、出来多少字节”。

#### 1.5.4 解码器是有状态的

G.722.1 属于自适应预测编码，解码器内部保留前一帧的预测状态。这带来两个约束：

1. **必须按顺序连续喂**：不能跳帧、不能把同一段码流交给两个解码器实例乱序解，否则状态对不上，输出会变成噪声；
2. **丢帧是有代价的**：代码在解码失败时丢掉一整帧继续，这是为了不让同一坏帧原地死循环，但丢帧后解码器状态会与编码器失配，随后若干帧可能出现轻微失真，直到自适应状态重新收敛（这一点属推断，手册未说明具体恢复长度）。

也正因为有状态，解码器句柄不能跨实例共用，必须每个 `HkSdkAudioClient` 一个。

#### 1.5.5 平台差异

同一组接口在不同平台签名不同（见 [HCNetSDK.h:52014-52029](hk_sdk/include/HCNetSDK.h#L52016-L52029)）：

| 平台 | `InitG722Decoder` | `DecodeG722Frame` |
| --- | --- | --- |
| Linux / Win64 | `InitG722Decoder()`，无参数 | 走 `NET_DVR_AUDIODEC_PROCESS_PARAM` 结构体 |
| 旧版 Windows 32 位 | `InitG722Decoder(int nBitrate = 16000)` | 三个裸指针参数：句柄、输入、输出 |

本项目在 Linux 下编译，走的是结构体版本。移植到旧 32 位 Windows SDK 时要改调用形式，数据量映射不变。

#### 1.5.6 失败与边界

| 情况 | 处理 |
| --- | --- |
| 创建失败（返回空指针） | `start()` 直接失败，后续无解码器可用 |
| 解码返回 FALSE | 计一次 `decode_errors`，丢掉这一帧继续；日志只打第一次 |
| `proc_data_size == 0` 或超过缓冲长度 | 按一整帧兜底，保证缓冲一定前进 |
| `out_frame_size == 0` | 只计解码成功，不写数据 |
| 缓冲剩余不足 80 字节 | 退出循环，留着等下一次回调拼齐；`stop()` 时计入 `dropped_partial_bytes` 丢弃 |

解码失败不中断采集，是因为运行中的一次坏帧不应该让整段录音作废；但失败会破坏解码器状态，所以失败次数需要看统计而不是忽略。

#### 1.5.7 为什么不支持其他编码

SDK 里还有 `NET_DVR_DecodeG711Frame`、`NET_DVR_DecodeG726Frame`，但本项目只接受 G.722.1（type 0）和 PCM（type 8）：

- G.711 在设备侧通常是 8 kHz，采样率与 16 kHz 的 ASR 前端不匹配，且码率更高、音质更差；
- G.726 设备配置里没有这一档；
- AAC / MP2L2 网络 SDK **没有解码接口**，收到也无法还原。

所以 `query_audio_format()` 在 `start()` 阶段就直接拒绝这几种编码，而不是录到一半才发现解不了。

### 1.6 G.722.1 路径逐级说明

```
设备编码码流（连续字节流，回调包长度任意）
  → m_encoded_buf 累积
  → 每次取 80 B 调 NET_DVR_DecodeG722Frame
  → 固定解出 1280 B PCM16 = 640 点 = 40 ms
  → 追加写 wav
```

代码位置对应：

| 步骤 | 位置 |
| --- | --- |
| 回调分流（`byAudioFlag == 1` 才收） | [hk_sdk_audio_client.cpp:840](hk_sdk_audio_client.cpp#L840) |
| 编码字节累积 + 触发切帧 | [hk_sdk_audio_client.cpp:870](hk_sdk_audio_client.cpp#L870) |
| 按 80 字节逐帧解码 | [hk_sdk_audio_client.cpp:904](hk_sdk_audio_client.cpp#L904) |
| 解码输出缓冲（1280 B × 4 预留） | [hk_sdk_audio_client.cpp:63](hk_sdk_audio_client.cpp#L63) |
| 追加写 wav | [hk_sdk_audio_client.cpp:860](hk_sdk_audio_client.cpp#L860) |

边界处理：

- 一次回调可能不足 80 字节，也可能包含多帧；`m_encoded_buf` 负责重新对齐。
- `proc_data_size` 用 SDK 回填的实际消费量，不写死 80。
- 解码失败丢一帧继续，避免同一坏帧无限重试。
- `stop()` 时不足 80 字节的残包计入 `dropped_partial_bytes` 后丢弃。

### 1.7 PCM 路径

```
设备 PCM16 连续样本流（回调包长度任意）
  → 直接追加写 wav
```

PCM（`byAudioEncType == 8`）不经过编码器，也就没有“帧”和“最小单元”：

- 回调数据是裸 PCM16 样本流，长度任意；
- 不需要组帧、不需要解码，直接按字节追加（[hk_sdk_audio_client.cpp:860](hk_sdk_audio_client.cpp#L860)）；
- 写入时长完全由「字节数 ÷ 采样率」决定，没有 40 ms 约束；
- 采样率必须取设备实际返回值（本机实测被固定为 32 kHz，请求 16 kHz 不生效）。

### 1.8 “40 ms”成立的条件

40 ms 只在下面两个条件同时满足时成立：

1. SDK 固定按 **80 字节 → 1280 字节** 解码；
2. 设备码率为 **16 kbps**（`BITRATE_ENCODE_INDEX` 中 `BITRATE_ENCODE_16kps = 2`，项目设备当前即 `bitrateEnum=2`）。

推导：80 B × 8 bit ÷ 16000 bit/s = 0.04 s = 40 ms。

如果设备改成 32 kbps（枚举 3），80 字节码流只含 20 ms，而 SDK 仍会固定输出 1280 字节（40 ms），两者就对不上。**这说明海康这套接口默认假设 16 kbps，不能把“80 B = 40 ms”当成协议结论到处套。**

### 1.9 频率与数据量

| 层级 | G.722.1 | PCM 32 kHz | 决定因素 |
| --- | --- | --- | --- |
| 设备编码/解码单元 | 25 单元/秒（40 ms/单元） | 无此概念 | SDK 80 B → 1280 B 约定 |
| 回调包频率 | 实测约 12.7 包/秒 | 实测约 21 包/秒 | 设备推送与网络聚合，不固定 |
| wav 采样率 | 16000 采样/秒 | 32000 采样/秒 | 设备音频配置 |
| 数据率 | 16 kbps = 2000 B/秒 | 512 kbps = 64000 B/秒 | 码率 × 采样位宽 |

注意：

- **25 单元/秒不是 25 包/秒**。前者是 40 ms 解码单元的节奏，后者是实际回调次数；实测 G.722.1 回调只有约 12.7 包/秒，因为设备把多帧聚合在一个回调包里。根 README 早期记录的“25 包/秒”实际指前者。
- 无启动填充：收到第一个完整 80 字节就开始写 wav；残包在 `stop()` 丢弃。
- 设备静音时可能不下发音频包，客户端不补静音，因此 **wav 音频时长可能小于墙钟时长**。

### 1.10 常见误解

| 误解 | 事实 |
| --- | --- |
| “G.722.1 一帧就是 80 字节 / 40 ms” | 标准帧是 20 ms；80 字节是海康解码接口的输入单元 |
| “回调一次就是 40 ms 音频” | 回调包长度任意，实测约 84 ms/包，需重新组帧 |
| “PCM 也是一包 40 ms” | PCM 无帧概念，包长任意，直接按样本流追加 |
| “25 包/秒” | 25 是 40 ms 解码单元频率；实测回调约 12.7 包/秒 |
| “音频数据连续，时长就等于墙钟” | 数据首尾相接，但静音/丢包不补，时长可能更短 |

### 1.11 已知限制

| 位置 | 问题 | 影响 |
| --- | --- | --- |
| 客户端整体 | 回调线程内加锁解码并写文件 | 低延迟实时识别场景下可能反压回调，建议改为入队 + 独立消费线程 |
| G.722.1 路径 | 固定按 80 字节切帧，未做抖动缓冲 | 设备推送节奏突变时可能短暂累积；残包停止时丢弃 |
| 音频链路 | 不补静音 | 静音段在 wav 中被压缩，音频时长小于墙钟 |
| 码率 | 按 16 kbps 假设推导 40 ms | 设备改码率后需重新核对解码单元与时长关系 |
| 上行保活 | `NET_DVR_VoiceComSendData` 回发静音失败只计数 `uplink_send_errors`，不触发重连 | 会话失效后回调会再次停流，需调用方看统计或重开 `start()` |
| 上行保活 | PCM 编码路径未实测，暂不启用（`uplink_keepalive` 只对 G.722.1 生效） | 设备切 PCM 后若同样 10 秒断流，需另行验证 |
| 暂停/恢复 | 暂停会关闭当前 wav，恢复后需重新 `start_record()` | 一段连续录音不能跨暂停；暂停期间的空档不会写进文件 |
| 暂停/恢复 | `StartVoiceCom` 调用本身阻塞约 1.2 秒 | `start()` / `resume_audio_forwarding()` 会阻塞调用线程，UI 线程需自行放到后台 |
| 主动设置编码 | `NET_DVR_SetDVRConfig` 写的是设备持久配置，会影响该设备上所有对讲客户端 | 默认关闭；`stop()` 默认写回原配置，恢复失败只记录错误 |
| 主动设置编码 | 设备可能「返回成功但没接受请求」（实测 PCM 16 kHz 被钉在 32 kHz） | 设置后必须读回复核，不一致时拒绝启动并回写原配置 |
| PCM 路径 | 本机实测只能 32 kHz，请求 16 kHz 无效 | 喂 16 kHz ASR 前必须重采样，或改回 G.722.1 |

## 二、落地实现：客户端工作流（HkSdkAudioClient）

第一章讲数据形态、两个核心接口与各种边界，这一章讲客户端按什么顺序把它们串起来。实现全部在 [hk_sdk_audio_client.cpp](hk_sdk_audio_client.cpp)，下面按生命周期串一遍。流式数据的处理的核心是构建一个缓存池，不断地将输入端数据存入缓存池中，输出端规定需要的数据包大小，每当缓存池累计数据大于一帧输出数据大小就开始从缓存池打捞数据，一次性将满足条件的所有数据移走，在完成数据输出的同时删除缓存池中的对应数据。

### 2.1 总览

```
构造 HkSdkAudioClient(cfg)      只保存配置 + 预分配解码缓冲，不碰 SDK、不连设备
  │
start()                         六步，任一步失败立刻返回 false，不做中途回滚
  ├─ 0. 清零状态                 m_stats / m_encoded_buf / m_stream_buf / 编码默认值
  ├─ 1. init_sdk()               NET_DVR_Init + 连接超时 3 s / 重试 1 次
  ├─ 2. login_device()           NET_DVR_Login_V40 → m_user_id
  ├─ 3. query_audio_format()     只读查询，只接受 G.722.1(type 0) / PCM(type 8)
  ├─ 4. InitG722Decoder          按实例复用，PCM 模式也创建
  ├─ 5. StartVoiceCom_MR_V30     登记回调 + this，返回语音对讲句柄
  └─ 6. 上行保活                 G.722.1 且 uplink_keepalive 时建编码器并起线程
       │                         每 40 ms 回发一帧 80 B G.722 静音
       │   ◀── SDK 回调线程持续触发，与主线程并发
       ▼
   voice_callback → on_voice_data → 加锁 → 过滤 → 分流 → 解码/直写 → 写盘 + 取流缓冲
       │
   start_record(path)            打开 .wav.part 并开始写盘（不写文件也能取流）
       │
stop()                          停保活线程 → 停流 → 冲刷残包 → 回填 wav 头 → 改名
  └─ 析构                         Logout → ReleaseG722Decoder / Encoder → NET_DVR_Cleanup
```

### 2.2 start() 六步：顺序不能换

| 步 | 函数 | 关键动作 | 失败会怎样 |
| --- | --- | --- | --- |
| 0 | 内联 | 清统计、清编码与取流缓冲、编码与采样率给默认值 | — |
| 1 | [`init_sdk()`](hk_sdk_audio_client.cpp#L608) | `NET_DVR_Init`、`SetConnectTime(3000, 1)` | 没有 SDK，后面全部无意义 |
| 2 | [`login_device()`](hk_sdk_audio_client.cpp#L622) | 用 ip / 端口 / 账号 / 密码登录，保存 `m_user_id` | 没有身份，所有 `NET_DVR_*` 都会失败 |
| 3 | [`query_audio_format()`](hk_sdk_audio_client.cpp#L653) | `NET_DVR_GetCurrentAudioCompress` 只读查询 | 编码不支持则拒绝启动，避免写出标错采样率的 wav |
| 4 | `NET_DVR_InitG722Decoder` | 建解码器；当前是 PCM 也建 | 运行中设备切回 G.722.1 就无法解码 |
| 5 | [`NET_DVR_StartVoiceCom_MR_V30`](hk_sdk_audio_client.cpp#L204) | 申请对讲会话、登记回调 | 没有数据来源，撤销运行标志并返回 |
| 6 | `init_uplink_keepalive()` / `start_uplink_keepalive()` | 建 G.722 编码器、编好静音帧、起 40 ms 回发线程 | 只打印提示，不影响取流；但设备约 10 秒后会停流 |

`start()` **不打开文件**：wav 由 `start_record(path)` 单独开启，取流缓冲（`get_last_audio()`）也独立于录制。这样语音转发可以在不录制的情况下长时间运行。

> 第 2 步与第 3 步之间还有一个**可选**的 2.5 步：`set_audio_format` 为真时调用 `apply_audio_format()` 主动写设备编码，写前快照、写后读回复核，不一致直接拒绝启动。默认关闭，详见 [2.9](#29-主动设置编码可选)。

顺序为什么不能调：

- **1 → 2 → 3**：没初始化不能登录，没登录不能查编码；
- **3 → 4/5**：编码类型决定走解码还是直写，也决定第 6 步是否启用上行保活；
- **`m_running` → 5**：SDK 可能在本函数返回前就同步触发第一个回调，标志晚置位会丢首包；
- **5 → 6**：保活线程要用 `StartVoiceCom` 返回的句柄，句柄没拿到之前发不了。

失败策略是「一路返回，不回头清理」：已经登录的会话、已经初始化的 SDK、已经创建的解码器都保留，由析构统一释放；下次 `start()` 直接复用。这也是 `start()` 可重复调用的原因。

#### 逐步展开

**第 0 步 · 清零。** `start()` 一进来先看 `m_running`，已经在跑就直接返回 `true`，因此重复调用是安全的，不会建出第二条流。接着清空上一轮的统计、编码缓冲和已写字节数，并把编码类型与采样率填成保守默认值——真正值等第 3 步查完设备再覆盖。清空编码缓冲不是可选项：上一轮遗留的半帧会让本轮的帧边界从第一帧起就错位。这一步全程持有 `m_mutex`，因为这些字段同时被回调线程和 `stop()` 访问。

**第 1 步 · 初始化 SDK。** 只有 `m_sdk_initialized` 为 `false` 时才调用 `NET_DVR_Init`，并把连接超时设为 3 秒、重试 1 次。超时给太长，按下 Ctrl+C 之后程序会卡在网络等待里收不了尾。这里刻意不调用 `NET_DVR_SetReconnect`：语音对讲会话断开后 SDK 不会自己重建，只能由上层重新 `start()`。初始化失败就直接返回，后面的每一步都不再有意义。

**第 2 步 · 登录设备。** 只有 `m_user_id` 仍是 `-1` 时才登录。IP、端口、用户名、密码全部来自构造时保存的 `AudioCaptureConfig`，字符串按目标缓冲区长度减一拷贝，配置写长了会被截断而不是越界。登录成功后 `m_user_id` 就是后续所有 `NET_DVR_*` 的身份凭据；失败时它保持 `-1`，下一次 `start()` 会重新尝试。

**第 3 步 · 查询音频编码。** 用 `NET_DVR_GetCurrentAudioCompress` 做只读查询，不修改设备配置——改配置会像热成像那样在设备上留下持久改动。返回值只接受两种：`type 0` 是 G.722.1，固定按 16 kHz 处理；`type 8` 是 PCM，按采样率枚举换算，枚举为「默认」时用配置里的兜底值。其余编码（G.711、AAC、MP2L2 等）直接报错拒绝启动，因为网络 SDK 没有对应的解码接口，硬录会写出采样率和内容都对不上的 wav。这一步产出的 `m_codec` 和 `m_output_sample_rate` 同时决定三件事：数据走解码器还是直接写盘、wav 头里写多少、统计怎么换算，所以必须赶在打开文件之前完成。

**第 4 步 · 创建解码器。** 用 `NET_DVR_InitG722Decoder` 申请一个 G.722 解码器句柄，按实例复用，重复 `start()` 不会重建。即便当前编码是 PCM 也照样创建：设备编码是持久配置，运行期间被切回 G.722.1 的情况虽然少见，但没有解码器就没有退路。这个句柄在析构里统一释放，不在 `stop()` 里销毁。

**第 5 步 · 发起语音对讲。** 先把 `m_running` 置为 `true`，再调用 `NET_DVR_StartVoiceCom_MR_V30`。顺序不能反：SDK 可能在这个函数返回前就异步触发第一个回调，而回调里第一个动作就是检查 `m_running`，标志晚置位等于丢掉首包。函数返回语音对讲句柄，小于 0 表示失败，此时撤销运行标志并记录错误。成功后记录起始时刻，作为墙钟时长统计的起点。这一步只建流，不打开任何文件。

**第 6 步 · 上行保活。** 手册要求语音转发双向，只收不发时设备约 10 秒后停止回调（详见 [1.4.7](#147-上行保活只收不发约-10-秒后设备停流)）。当前编码是 G.722.1 且 `uplink_keepalive` 为真时：`init_uplink_keepalive()` 用 `NET_DVR_InitG722Encoder` + `NET_DVR_EncodeG722Frame` 把 640 个 0 采样点编成一帧 80 B 静音（只编一次，线程只读），`start_uplink_keepalive()` 再把句柄写进原子量并起线程，每 40 ms 调一次 `NET_DVR_VoiceComSendData`。初始化失败只打印提示、不影响取流；PCM 路径未实测，暂不启用。

### 2.3 回调线程的一次处理

```
SDK 回调线程
  voice_callback(handle, data, size, flag, pUser)
    → static_cast<HkSdkAudioClient*>(pUser)      把 this 还原
      → on_voice_data(data, size, flag)
          ├─ flag != 1            → 直接丢，不是设备端音频
          ├─ 加锁
          ├─ m_running == false   → 直接丢，stop() 已经发生
          ├─ packets++ / encoded_bytes += size
          ├─ PCM 模式              → pcm_samples += size/2 → fwrite 原始字节
          └─ G.722.1 模式          → 追加进 m_encoded_buf → decode_pending_locked()
                                      while (>= 80 B) 解一帧 → fwrite 1280 B
```

两个「必须」：

1. **必须攒缓冲**：回调包长度任意（实测平均 168 B ≈ 2 帧），一次回调可能不足一帧、刚好一帧、或多帧，不能拿「一包」当「一帧」；
2. **必须在锁内**：`m_encoded_buf`、`m_stats`、`m_wav`、`m_data_bytes` 同时被回调线程和主线程 `stop()` 访问。

[`decode_pending_locked()`](hk_sdk_audio_client.cpp#L904) 的 `_locked` 后缀表示「调用方已持锁」，函数内部不再加锁（`std::mutex` 不可重入）。

#### 逐步展开

**① 进入回调。** SDK 在自己的线程里调用 `voice_callback`，一次性给四样东西：对讲句柄、数据指针、字节数、来源标志，外加当初注册的 `pUser`。这个函数必须是静态的——C 函数指针装不下成员函数——所以它拿不到 `this`，第一件事就是用 `static_cast` 把 `pUser` 还原成 `HkSdkAudioClient*`，再转调 `on_voice_data`。跳板本身不做任何判断，只负责把「对象指针」和「数据」重新拼到一起。

**② 过滤来源。** `on_voice_data` 的第一个判断是 `flag != 1` 就返回。`byAudioFlag` 表示这段音频来自哪里，只有 `1` 是设备端编码后的音频；其他值可能是本地麦克风上行或控制数据，混进来会直接污染录音。这一步比帧长判断更早、也更重要。

**③ 加锁并检查运行标志。** 通过过滤后立刻拿 `m_mutex`，然后读 `m_running`。`stop()` 会把标志置 `false` 并关闭文件，但总可能有已经排队的回调还在路上；在这里先看标志，就不会往一个已经 `fclose` 的 `FILE*` 里写。锁一直持有到本次回调结束，保证缓冲区、统计和文件三者不会被主线程同时改动。

**④ 记账。** 先累加回调包数和编码字节数。这两项不管后面解码成不成功都要记，因为它们是判断链路是否正常的直接依据——包数停住说明设备不发或网络断了，而不是解码出了问题。

**⑤ PCM 分支：直接写。** 设备编码是 PCM 时，回调数据就是目标格式，不需要任何转换：按每样本 2 字节累加采样点数，再把整块字节追加到 wav，函数随即返回。`size` 若不是偶数（异常半样本），统计向下取整，但写入长度仍以 `size` 为准。这条路径不维护缓冲，也不存在帧边界问题。

**⑥ G.722.1 分支：攒够一帧再解。** 编码路径复杂一些。回调给的是一段连续码流，边界和 80 字节的帧边界没有关系，所以先整段追加到 `m_encoded_buf`，再交给 `decode_pending_locked` 处理。后者在 `while` 循环里每次取缓冲头部的 80 字节调用 `NET_DVR_DecodeG722Frame`，解码器固定吐出 1280 字节 PCM（640 个采样点，40 ms）。

每一轮有三件事必须做对。第一，解码失败时计一次错误、丢掉这一帧继续，否则同一坏帧会原地无限重试；错误日志只在第一次打印，避免坏帧刷屏拖慢回调。第二，成功时用 SDK 回填的 `proc_data_size` 而不是写死的 80 作为消费量——它是从 `in_buf` 起算的字节数，而 `in_buf` 正好指向缓冲头部，所以可以直接拿来做 `erase` 的长度；若回填成 0 或超过缓冲长度，就按一整帧兜底，保证缓冲一定前进，循环一定能退出。第三，`out_frame_size` 大于 0 时累加采样点数并写盘，等于 0 时只记解码成功、不写数据。

循环在剩余字节不足一帧时退出，尾巴留在缓冲里，等下一次回调拼够 80 字节再解。`stop()` 时如果缓冲里还有剩余，会单独计入 `dropped_partial_bytes` 后丢弃，不会写进 wav。

**⑦ 返回。** 回调函数返回时锁随作用域释放。整条路径里唯一的阻塞点是 `fwrite`，正常情况下命中页缓存只需几十微秒，比起一帧 40 ms 的音频时长差了三个数量级，所以不需要额外开线程——这也正是解码直接放在回调里完成的原因。

### 2.4 stop()：顺序同样不能反

实现见 [stop()](hk_sdk_audio_client.cpp#L330)。

```
1. stop_uplink_keepalive()        先停保活线程（置标志 → join → 句柄清 -1）
2. m_running.exchange(false)     软闸：回调看到就丢，但拦不住已经进入的调用
3. NET_DVR_StopVoiceCom          硬闸：让 SDK 停止派发回调
4. 锁内：缓冲残包计入 dropped_partial_bytes 后丢弃
5. close_wav：回填 RIFF/data 长度 → fflush → fclose → .part 改名 .wav
6. 记录 wall_seconds（仅本次确实运行过才更新）
```

保活线程必须最先停：它每 40 ms 用语音对讲句柄调用 `NET_DVR_VoiceComSendData`，如果先 `StopVoiceCom` 再停线程，线程可能正拿着已失效的句柄发数据。`stop_uplink_keepalive()` 的 `join` 最多等循环里的一小片 `sleep_for(5 ms)`，不会阻塞很久。

`fclose` 必须晚于 `StopVoiceCom`。反过来的话，已经在排队的回调仍会执行 `fwrite`，写向一个已经关闭的 `FILE*`，属于未定义行为。`m_running` 只是软闸，真正阻断回调的是 `StopVoiceCom`。

### 2.5 线程与锁

| 线程 | 跑哪些函数 |
| --- | --- |
| 调用线程（main） | 构造 / 析构 / `start()` / `stop()` / `get_stats()` / `output_path()` |
| SDK 回调线程 | `voice_callback` → `on_voice_data` → `decode_pending_locked` → `write_pcm_locked` |
| 上行保活线程 | `uplink_keepalive_loop()`：每 40 ms 调 `NET_DVR_VoiceComSendData` |

一把 `m_mutex` 保护 `m_stats`、`m_encoded_buf`、`m_data_bytes`、`m_wav`；保活线程同样在锁内累加 `uplink_sent_frames` / `uplink_send_errors`。回调线程内除了「首次出错打印」之外不做 I/O 等待，避免反压 SDK 造成丢包。保活线程句柄走 `m_voice_handle_atomic`，不碰 `m_mutex` 保护的字段。

另外 `is_running()` 只反映本类标志，**不代表链路在出音频**：设备静音时 SDK 可能长时间不下发回调，此时它仍然是 `true`。

### 2.6 失败分三级

| 级别 | 表现 | 例子 |
| --- | --- | --- |
| 致命 | `start()` 返回 false，原因写进 `last_error()` | 登录失败、编码不支持、开文件失败、起流失败 |
| 可降级 | 只计数 + **首次**打印，继续运行 | 解码失败（丢一帧继续）、`fwrite` 失败（后续帧继续尝试） |
| 静默 | 不影响运行 | `--probe` 不开文件（写入自动变空操作）、采样率取设备值而非配置值 |

运行中的错误不能中断音频：一次坏帧或一次写失败不应该让整段录音作废，这就是「丢一帧继续」和「只打印第一次」的原因。

### 2.7 wav 落盘

- 先写 44 字节标准头，`RIFF` / `data` 两处长度先写 0，`stop()` 时回填；
- 用 `.wav.part` 过渡，收尾才 `rename` 成 `.wav`：录制中途崩溃/强杀不会留下「看起来合法但长度错误」的文件；同目录 `rename` 是原子操作；
- 头字段按**小端**逐字节写（[`write_u16`](hk_sdk_audio_client.cpp#L1117) / [`write_u32`](hk_sdk_audio_client.cpp#L1127)），不直接 `fwrite` 主机字节序，保证大端平台（部分 ARM 板）也能正确解析；
- `m_data_bytes` 只统计 PCM 负载，不含 44 字节头；
- 设备静音可能不下发包，客户端也不补静音，因此 **wav 音频时长可能小于墙钟时长**。

### 2.8 暂停与恢复：只停/起对讲线路

`start()` 里的 SDK 初始化、登录、编码查询、解码器创建都是会话级资源；暂停只需要拆掉「对讲线路」这一层：

```
会话层（保留）  NET_DVR_Init → Login → GetCurrentAudioCompress → G722 解码器/编码器
线路层（暂停）  NET_DVR_StartVoiceCom_MR_V30  ⇄  NET_DVR_StopVoiceCom
```

接口：

| 接口 | 行为 |
| --- | --- |
| [`pause_audio_forwarding()`](hk_sdk_audio_client.cpp#L380) | 停保活线程 → `StopVoiceCom` → 句柄置 `-1` → 自动 `stop_record()` → 丢弃编码缓冲残包 |
| [`resume_audio_forwarding()`](hk_sdk_audio_client.cpp#L423) | 重建 G.722 解码器 → 重新 `StartVoiceCom`（失败每 200 ms 重试，最多 5 次）→ 起保活线程 |

状态拆成三个标志：`m_session_ready`（初始化完成）、`m_paused`（暂停而非停止）、`m_running`（正在转发）。`is_running()` 返回 `m_running`，暂停后为 `false`；暂停中再调 `start()` 等价于 `resume_audio_forwarding()`，不会重跑初始化。

语义约定：

- **暂停自动收尾 wav**：正在录制时 pause 会先 `stop_record()`（回填长度、`.part` 改名），恢复后需要重新 `start_record()`；
- **取流缓冲保留**：暂停前没取走的样本仍可用 `get_last_audio()` 取完，不会被丢弃；
- **恢复重建解码器**：暂停期间设备重新开始编码，旧解码器的自适应预测状态可能失配，重建成本一次分配；
- **统计**：`forward_seconds` 只累计实际转发时长，暂停时间不计入；`wall_seconds` 仍是从 `start()` 到 `stop()` 的墙钟；
- `stop()` 会清掉暂停标志，下次 `start()` 重新走完整初始化。

实测（2026-11-03，设备 `10.168.2.65`）：

| StopVoiceCom 后间隔 | 结果 |
| --- | --- |
| 0 ms | `StartVoiceCom` 失败，`error=29`（`NET_DVR_DVROPRATEFAILED`） |
| 100 ms | 2 次中 1 次成功 |
| 200 ms | 2 / 2 成功；连续 10 次「停 200 ms → 恢复」10 / 10 成功 |
| 1 / 3 / 5 s | 全部成功，恢复后包率与基线一致 |

`StartVoiceCom` 调用本身约阻塞 1.2 秒（探针从调用前计时，首包 1.23–1.31 s；正式库中 `resume` 返回后 20 ms 内即可取到数据），因此暂停开关的响应时间主要来自这个同步调用，而不是暂停造成的额外等待。

重试逻辑放在 [`start_voice_forward()`](hk_sdk_audio_client.cpp#L187) 内部，`start()` 与 `resume` 共用：只对 error 29 重试（5 次 × 200 ms），网络、权限等其它错误立即返回。这样冷启动撞上「上一个进程刚退出、设备还没释放会话」也能自动恢复；如果语音通道被另一客户端占用，则 5 次后如实失败。

### 2.9 主动设置编码（可选）

默认只读设备编码。需要强制编码时打开 `AudioCaptureConfig::set_audio_format`，其余字段在 `main` 里用字面量写死：

| 字段 | 含义 | 取值 |
| --- | --- | --- |
| `set_audio_format` | 是否在启动前写设备编码 | 默认 `false` |
| `audio_enc_type` | 目标编码 | `0`=G.722.1（自动 16 kHz），`8`=PCM（自动 32 kHz） |
| `restore_audio_format` | `stop()` 时是否写回原配置 | 默认 `true` |

采样率不单独开放：跟着编码走固定值。G.722.1 标准就是 16 kHz；PCM 本机实测只能 32 kHz（请求 16 kHz 会被设备忽略），所以 PCM 直接请求 32 kHz，避免暴露一个设备不接受的组合。

调用的是通用配置接口，手册在《NET_DVR_SetDVRConfig》：命令 `NET_DVR_SET_COMPRESSCFG_AUD`（1059），`lChannel` 固定 `0xFFFFFFFF`，结构体 `NET_DVR_COMPRESSION_AUDIO`；读取原值用 `NET_DVR_GET_COMPRESSCFG_AUD`（1058），设备不支持时退回 `NET_DVR_GetCurrentAudioCompress`。

流程（`start()` 第 2 步与第 3 步之间）：

```
GetDVRConfig(1058) 快照原值
  → SetDVRConfig(1059) 写目标编码 + 跟随采样率（其余字段沿用快照）
  → GetCurrentAudioCompress 读回
  → 编码一致 && 采样率一致 ? 继续 : 回写原值并拒绝启动
```

判定口径：

- 只改 `byAudioEncType` / `byAudioSamplingRate`，码率、`byres`、`bySupport` 原样保留；
- 采样率由编码决定：`0`→`1`（16 kHz），`8`→`2`（32 kHz）；G.722.1 的读回 `0` / `1` 都按 16 kHz 接受，PCM 读回必须是 `2`；
- **不相信 `SetDVRConfig` 的返回值**：设备可能返回成功但没生效，wav 头与 ASR 输入都依赖实际采样率，所以以读回值为准；
- `stop()` 在注销登录前写回快照；启动途中任何一步失败也会回写，避免下次 `start()` 把「已改过的值」当成原值；
- 该接口改的是设备持久配置，会影响所有对讲客户端；恢复失败只记录 `last_error`，不阻断收尾。

实测（2026-11-03，设备 `10.168.2.65`）：

| 目标编码 | 实际请求 | 读回 | 结果 |
| --- | --- | --- | --- |
| G.722.1 | `type=0 rate=1` | `type=0 rate=1` | 设置成功，`stop()` 后写回原配置 |
| PCM | `type=8 rate=2` | `type=8 rate=2` | 设置成功，但 ASR 入口会因 32 kHz ≠ 16 kHz 拒绝运行 |

早期试过请求 PCM 16 kHz，`SetDVRConfig` 返回成功但读回仍是 `rate=2`（32 kHz），所以现在固定请求 32 kHz。要喂 16 kHz ASR，要么重采样，要么用默认的 G.722.1 路径。


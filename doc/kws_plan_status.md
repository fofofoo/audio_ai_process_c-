# KWS 接入计划与进度

- 状态：B1、B2、B3 均已完成（2026-11-05）；实时双模式入口已在真机上跑通
- 遗留：KWS oracle 对拍、双模式性能基准脚本、负样本扩充（见第五、七节）
- 相关文档：
  - [BlockWorker 异步处理机制](audio_ai_block_worker.md)
  - [ASR 有界状态与转写落盘改造计划](asr_bounded_state_plan.md)
  - [Python 参考实现 KWS 说明](../python_ref/README.md)

---

## 一、目标与总体路线

KWS 分支的目标是给实时语音链路加一层「先说唤醒词、再说话」的双模式：

```
KWS 监听 --命中唤醒词--> ASR 识别
   ▲                       │
   └── 结束语 / 空闲 10 秒 ──┘
```

平时只把特征喂给 `KwsDecoder`，计算量小、误识别少；命中唤醒词后切到完整 ASR，
识别用户命令；出现结束语或长时间没有新 token 后切回监听。同一时刻只有一个解码器
吃特征，`AudioPreprocess` 始终连续，特征时间轴不断。

整体拆成三步，每步独立验证：

| 步骤 | 内容 | 状态 |
|---|---|---|
| B1 | 关键词前缀树 `KwsContextGraph` + 单测 | 已完成 |
| B2 | 束搜索解码器 `KwsDecoder` + 离线 wav 入口 `test_wav_kws.out` | 已完成 |
| B3 | 设备实时 KWS+ASR 双模式入口 | 已完成（2026-11-05） |

---

## 二、已完成部分

### 2.1 B1：关键词前缀树

**提交**：`facb499 feat: 接入 KWS 关键词前缀树与单测`

**已落地文件**

- `audio_ai_process/kws_context_graph.h/.cpp`：Aho-Corasick 风格的关键词前缀树。
  构造期按 `KeywordSpec` 建 trie、填 fail/output 弧；运行期 `forward_one_step()`
  逐 token 步进，`is_matched()` 判断是否走完整条关键词。纯标准库，不依赖 ORT / ROS。
- `tools/test_context_graph.cpp`：搬上游 `context-graph-test.cc` 的 4 组期望值，
  另补 phrase / threshold / output 弧字段自检；不需要权重和 ORT。

**验证结果**：`./build/test_context_graph.out` 通过，退出码表示成败。

**已固定的语义**

- `KeywordSpec::score == 0` / `threshold == 0` 是「使用构造默认值」的哨兵；
- 公共前缀分支的 `token_score` 取 `max(score, 已有值)`，`is_end` 时重算 `output_score`；
- `fill_fail_output()` 里 `output_score` 用 `+=`，不是 `=`；
- 根节点 `token = -1`、`fail` 指向自己，`-1` 只作标记，不参与下标运算；
- 图不可拷贝 / 不可移动，构造完成后只读，可并发查询；
- 空 `token_ids` 抛 `std::invalid_argument`（本仓库对上游「静默跳过」的刻意差异）。

### 2.2 B2：KWS 解码器与离线入口

**提交**

- `53fbbbe feat(python_ref): 接入 KWS 关键词唤醒参考实现`
- `c339cd6 feat: 接入 KWS 关键词解码器与离线 wav 入口`（已提交并推送）

**已落地文件**

- `audio_ai_process/kws_decode.h/.cpp`：流式关键词唤醒。持有三个 `Ort::Session`
  与 38 个状态张量，按值持有 `KwsContextGraph`；完成分块编码、束搜索（每帧批量跑
  decoder / joiner）、尾随 blank 触发判定与静音软重置。
- `test_wav_kws.cpp`：离线 wav 入口，对应 `python_ref/test_wav_kws_mini.py`。
  逐文件读 wav、末尾补 0.5 秒真静音、按块喂前端与 `KwsDecoder`，打印命中词、
  首 token 时间与平均概率；内嵌 `kExpected` 期望集合，并用 160 / 1600 两种块长复测。
- Python 侧 `python_ref/src/kws_decode.py`、`kws_context_graph.py` 与
  `test_wav_kws_mini.py`、`test_context_graph_mini.py` 作为参考实现。

**验证结果**

- `./build/test_wav_kws.out`：权重包自带 9 条 wav，命中集合与 `kExpected` 一致；
  `zh_0~2` 为负样本，零误触发；160 / 1600 块长结果相同；累计触发 10 次。
- `./build/test_context_graph.out`：通过。

**模型契约（chunk-8，当前默认）**

- `T = 29`、`decode_chunk_len = 16`（时间轴每块前进 160 ms）；
- 每块 4 个输出帧，40 ms/帧；
- 38 个状态张量，`context_size = 2`，`vocab_size = 263`；
- 263 个 token = 3 个特殊符 + 69 个 CMU 音素 + 191 个部分拼音；
- `blank_id = 0`，`<unk>` 也按 blank 处理；
- 触发条件三者同时成立：trie `is_matched()`、最佳假设的尾随 blank 数
  `> num_trailing_blanks_`（默认至少 2 帧 / 80 ms）、关键词前若干 token 的平均
  声学概率 `>= ac_threshold`；
- 命中时间是关键词**首 token** 的时间戳，不是触发时刻；
- 连续静音超过 1.5 秒软重置：重建 encoder 零状态与空白 beam，保留未消费的特征尾巴；
- 离线 wav 末尾必须补**样本域**真静音（默认 0.5 秒）；特征域补 0 是 log 能量 1，
  不是静音，不能替代。

**交付时的限制（B3 已解除其中第一条）**

- ~~只有离线 wav 入口，还没有接设备实时麦克风~~ → B3 已补 `test_mic_kws_asr.out`；
- 还没有和 sherpa-onnx wheel 的 oracle 逐条对拍（`--dump-hits` 已具备导出能力）。

---

## 三、B3：实时双模式入口（已完成，2026-11-05）

### 3.1 目标形态

新增 `test_mic_kws_asr.cpp`，对应 `python_ref/test_mic_kws_asr.py`：

```
海康设备取流 -> AudioPreprocess -> 同一份特征
                                  ├─ SpectrogramImage（始终喂）
                                  ├─ KwsDecoder（KWS 模式）
                                  └─ AsrDecoder / AsrSession（ASR 模式）
```

主循环只负责取音频、算特征、投递、显示；解码与模式机放在后台 Worker 的 Session 里。

### 3.2 模式机

**KWS 模式**

- 每块特征只喂 `KwsDecoder`；
- 维持最近 100 帧特征的回填环（按帧存，约 1 秒）；
- 命中唤醒词时：
  1. `AsrDecoder::reset()`；
  2. 把最近 100 帧特征回填给 ASR；
  3. 切到 ASR 模式；
  4. 发布「唤醒」事件（关键词、首 token 时间、平均概率）；
  5. 主线程下一轮读到模式变化，底部状态从 `KWS | listening` 变为 `ASR`。

回填的音频里包含唤醒词本身，ASR 文本开头可能出现唤醒词，这是可接受的；回填太少
会切掉命令词开头，回填太多会把唤醒词之前的无关内容听进去。100 帧是当前折中值。

**ASR 模式**

- 每块特征只喂 `AsrDecoder`（通过 `AsrSession`，见第四节）；
- 只要产出新 token，就刷新空闲计时；
- 切回 KWS 的两个条件：
  - 识别文本去掉尾部空白与中英文标点后，以某个结束语结尾；
  - 连续 10 秒没有新 token（空闲超时）。

结束语候选沿用 Python 参考实现：

```
再见 / 拜拜 / 结束 / 退出 / 没事了 / 滚吧 / 滚 / 退下 / 闭嘴 / 退下吧
```

**切回 KWS 时**

- 结算当前 ASR 转写段（调用 `AsrSession` 的段结算接口，落盘）；
- 两个解码器都 `reset()`；
- ASR 文本清空，模式回到 KWS；
- 主线程显示回到 `KWS | listening`。

### 3.3 架构：BlockWorker + AsrSession + KwsAsrSession

沿用已经定稿的通用异步框架：

- **BlockWorker**（header-only 模板）：只负责一条有界音频特征输入队列、单消费者线程、
  满时丢最旧、尾块阻塞不丢、异常捕获、收尾 join。不知道 token、模式、关键词。
- **AsrSession**：ASR 分支的 token 累积、64 token 分段、`log/asr_last_text.txt` 落盘、
  显示快照。
- **KwsAsrSession**：双模式业务类，内部持有 `KwsDecoder` 和一个 `AsrSession`，
  维护模式机、回填环、事件、空闲计时、统计和显示快照。

选型理由：只开一个后台线程，同时服务两个解码器，避免两个线程各自抢 CPU、互相同步
特征；切换模式时没有跨线程迁移，解码器始终在同一个线程里串行调用。

入口装配与单模式一致：

```cpp
KwsAsrSession session(asr, kws, cfg);
BlockWorker<LogFbankFeatures> worker(
    [&session](const AudioFeatureInput& item) { session.on_block(item); },
    [](const LogFbankFeatures& f) {
        return f.n_frames <= 0 || f.data.empty();
    });
```

### 3.4 收尾块的处理

程序退出时前端会补 100 帧特征域的 0，用来凑满 ASR 最后一个 chunk。这 100 帧只对
ASR 有意义；KWS 不能用特征域的 0 当静音（那是 log 能量 1，不是真静音）。

BlockWorker 的 `Item` 带 `is_tail` 标记，`KwsAsrSession` 在 KWS 模式下看到尾块就
直接跳过，在 ASR 模式下才正常喂给 `AsrDecoder`。Worker 只负责透传标记，不解释语义。

### 3.5 入口与命令行开关

新增根目录入口 `test_mic_kws_asr.cpp`，设备地址、权重目录、阈值等参数直白写在 `main`，
命令行只暴露运行期常用的 5 个开关：

| 开关 | 作用 |
|---|---|
| `--no-window` | 不开窗口，只打印文本（远程调试） |
| `--cuda` | ASR 使用 CUDA EP，失败回退 CPU |
| `--seconds N` | 跑 N 秒自动收尾退出 |
| `--initial-asr` | 初始进入 ASR 模式（调试纯 ASR 行为） |
| `--keywords <文件>` | 覆盖默认关键词文件 |

窗口底部状态文字：

- KWS 模式：`KWS | listening`
- ASR 模式：`ASR | <识别文本>`，按像素宽度从尾部截断
- 可附带唤醒次数、切回次数、丢块统计

### 3.6 关键词文件

已在仓库根新增 `keywords/kws_keywords.txt`（`test_mic_kws_asr.out` 默认读取，
`--keywords` 可覆盖），三条中文唤醒词沿用 Python 参考实现
已验证的配置：

```
小智同学   score=1.5  threshold=0.30
小智小智   score=1.5  threshold=0.30
你好小智   score=1.5  threshold=0.35
```

C++ 侧读取的是生成后的音素行格式：

```
<token...> :score #threshold @原文
```

直接由 `python_ref/keywords/kws_keywords.txt` 复制或对照生成，`weight/` 里的原始
关键词文件保持只读。参数依据见
`python_ref/keywords/kws_keywords_raw.txt` 的 30 条真机样本扫描记录：score=1.5 时
三个词命中率为 10/10、9/10、8/10；2.0 反而下降。

### 3.7 验收标准

- KWS 唤醒：三条唤醒词各能触发；误触发可控；唤醒事件带关键词、时间、概率；
- 回填：唤醒后紧接着说命令词，命令词开头不丢；
- ASR 文本：窗口与 stdout 正常累积；退出时输出本段完整文本；
- 切回：结束语能立即切回；空闲 10 秒能切回；切回后 KWS 继续正常监听；
- 落盘：ASR 段按 64 token 结算并追加到 `log/asr_last_text.txt`；切回/退出时尾段也落盘；
- 收尾：KWS 模式忽略 `is_tail`，ASR 模式正常解码尾块；
- 回归：`test_context_graph.out`、`test_wav_kws.out`、`test_wav_asr.out` 仍通过；
- 性能：Jetson 上实时处理率、取流丢块、后台队列丢块可观测，不因双模型加载明显掉队。

### 3.8 实施结果（2026-11-05）

**新增 / 改动文件**

- `audio_ai_process/kws_asr_session.h/.cpp`：模式机、100 帧回填环、空闲计时、
  结束语判定、事件序号与统计；只做业务编排，不碰 ORT / 线程 / 队列。
- `test_mic_kws_asr.cpp`：设备取流 → 前端 → `BlockWorker` → `KwsAsrSession`，
  窗口两行状态（模式行 + 文本行），5 个命令行开关照设计落地。
- `audio_ai_process/asr_session.h/.cpp`：补 `settle_segment()` / `token_count()` /
  `current_text()` / `clear_history()`，供模式切换使用。
- `keywords/kws_keywords.txt`：从 `python_ref/keywords/kws_keywords.txt` 复制。
- `CMakeLists.txt`：库源、安装头文件、入口声明与链接分组同步。

**离线验证（临时验程序，不入库）**

- 唤醒 wav + 0.5 s 真静音 + 语音 wav 组合喂入，结束语设为该 wav 结尾词；
- 结果：唤醒 1 次、切回 1 次、回填 100 帧、尾块跳过 0 次、转写 1 段，
  `log/asr_last_text.txt` 行数与段数一致。
- 回归：`test_context_graph.out` 通过；`test_wav_kws.out` 9 条命中集合与
  160/1600 块长不变性通过（累计触发 10 次）；`test_wav_asr.out` 四条 wav 文本与
  改造前一致（27 / 20 / 23 / 32 token）。

**真机 60 秒（x64 / G.722.1 / 16 kHz / int8）**

| 事件 | 内容 |
|---|---|
| 唤醒 1 | `小智同学`（avg_prob 0.467222，t=2.8 s）→ 结束语「再见」切回 |
| 唤醒 2 | `小智同学`（avg_prob 0.657341，t=2.44 s）→ 结束语「再见」切回 |
| 唤醒 3 | `你好小智`（avg_prob 0.401793，t=0.32 s）→ 结束语「还没结束」误命中「结束」后切回 |

- 统计：ASR encoder 34 次 / 408 帧 / 47 token；KWS encoder 284 次 / 1136 帧 /
  3 次命中；唤醒 3、切回 3、回填 300 帧、尾块跳过 1 次；
  取流丢块 0、解码丢块 0、取流缓冲峰值 1。
- `log/asr_last_text.txt` 恰 3 行，与 3 次 ASR 会话一一对应。

**已知行为**

- 回填包含唤醒词本身，ASR 文本开头常带唤醒词（如「小智同学，……」），属预期；
- 结束语按后缀硬匹配，第 3 次会话的「还没结束」被判为结束语，说明短语表需要按
  现场话术收紧（例如改判整句相等，或去掉过于通用的「结束」）。

---

## 四、已完成（2026-11-05）：ASR 缓存与转写落盘改造

B3 原计划紧接着 B2 做，但曾临时插入 ASR 缓存改造，原因是 **B3 的 `AsrSession`
依赖 `AsrDecoder` 的新契约**：

- 旧 `AsrDecoder` 内部用 `hyp_` 保存整段历史，每次返回全量文本；如果先写
  `AsrSession`，只能按旧 API 写一版，等解码器改造时再推翻重写；
- 新契约把「解码器有界状态」和「外部长期文本」分开，`AsrSession` 才能真正成为
  转写缓存的所有者，KwsAsrSession 直接复用它。

改造内容（详见 `doc/asr_bounded_state_plan.md`）：

- `AsrDecoder` 删除 `hyp_`、`text_`、`token_ids()`、`text()`；
- 内部只留最近 128 个 token 与 `recent_text()`；
- `accept_features()` 返回本次新增 token 的 `std::vector<int64_t>`；
- 外部持有 `all_ids`，满 64 token 结算一段并追加到 `log/asr_last_text.txt`；
- 取消所有时间触发（最小段长、静音 reset、最大段长、定时 flush）；
- 解码器只在显式要求时 `reset()`。

**改造分两阶段**：

1. 解码器有界化 + 增量输出，改写 `test_wav_asr` 与 `test_mic_asr` 的最小适配；
2. 通用 `BlockWorker` + `AsrSession`，重写 `test_mic_asr`，加分段与落盘。

阶段 2 完成后，B3 可以直接在 `AsrSession` 之上写 `KwsAsrSession`，不再返工。

**实施结果（2026-11-05，阶段一、二均已完成）**：

- `audio_ai_process/block_worker.h`（header-only 模板）、`audio_ai_process/asr_session.h/.cpp`
  新增；`test_mic_asr.cpp` 的本地 `AsrWorker` 已删除，改用
  `BlockWorker<LogFbankFeatures>` + `AsrSession`；`test_wav_asr.cpp` 改为外部累积
  `all_ids`，收尾用 `text_of(all_ids)`。
- 离线回归：权重包 4 条 wav 的 token 与文本与改造前**逐字节一致**，
  `tools/compare_asr.py` 四条全部一致；`test_context_graph.out`、`test_wav_kws.out`
  （9 条命中集合 + 160/1600 块长不变性）不受影响。
- 分段路径：临时离线程序用 `segment_tokens=8` 跑 27 token 的 wav，得到 3 段、
  txt 3 行，最终文本与非分段一致（`tool` 与 `finish()` 口径一致）。
- 真机：20 s 运行识别出「干嘛你要测试吗？」，8 token、1 段（在 `finish()` 结算），
  `log/asr_last_text.txt` 恰一行且与最终文本一致，取流丢块 0、解码丢块 0、
  静音补零 0。
- `AsrSession` 已补公开的 `settle_segment()`（内部复用 `close_segment()`），
  供 B3 切回 KWS 时人为切断当前 ASR 段。

---

## 五、B3 落地时的口径确认（已完成）

1. **模式切换时如何结算 ASR 段**：`AsrSession` 提供公开的 `settle_segment()`
   （2026-11-05 补），内部复用 `close_segment()`，不足 `segment_tokens` 也结算。
   `KwsAsrSession` 切回 KWS 时在 **Worker 线程内**调用它（不能与 `on_block()` 并发），
   随后 `clear_history()` 清本会话 token，文件与段计数保留。
2. **空闲计时的口径**：沿用 Python 的「距上次新 token 的墙钟时间」，
   `KwsAsrSession` 用 `std::chrono::steady_clock` 计时，`on_block` 里发现
   `token_count()` 增长就刷新。设备无数据时 Worker 不执行 `on_block`，也就不会触发
   超时判断——此时不存在「用户还在说话」的场景。
3. **结束语匹配**：去掉尾部空白与中英文标点后做 UTF-8 后缀比较，已按 Python 列表
   落地（再见 / 拜拜 / 结束 / 退出 / 没事了 / 滚吧 / 滚 / 退下 / 闭嘴 / 退下吧）。
   真机发现「还没结束」会被判为结束语，短语表需按现场话术收紧，见 3.8。
4. **回填实现**：按帧维护 100 帧环形缓冲（`ring_` + `ring_head_` + `ring_frames_`），
   唤醒时按时间顺序拼成一个 `LogFbankFeatures`，直接在同一 Worker 线程调用
   `AsrSession::on_block()`，不经过队列。
5. **初始模式**：默认 KWS；`--initial-asr` 从 ASR 起步，用于复用纯 ASR 调试路径。
6. **关键词文件生成**：仓库根 `keywords/kws_keywords.txt` 已建，内容复制自 Python 侧
   生成物；C++ 侧仍不引入拼音生成工具，现场改词需要重新跑 Python 生成流程。
7. **oracle 对拍**：KWS 与 sherpa-onnx wheel 的逐条对拍仍是低优先级待办，
   `--dump-hits` 的 TSV 已具备对拍所需的字段。

---

## 六、风险与注意事项

- **两套模型同时驻留**：ASR 与 KWS 各自三个 session，Jetson 上内存和初始化时间都会
  增加；加载顺序仍在设备 `start()` 之前，避免启动阶段丢音频。
- **只有一个线程**：两个解码器串行执行，切换模式不会并行；实时性靠「同一时刻只喂
  一个解码器」保证，不能改成两个线程各喂一路特征。
- **KWS 的静音软重置**：`KwsDecoder` 内部 1.5 秒静音会重建 beam 与 encoder 状态，
  这是解码器自身行为，与模式切换无关；切回 KWS 时额外再 `reset()` 一次是干净的。
- **尾块语义**：KWS 模式必须跳过 `is_tail`，否则会把特征域 0 当音频喂进束搜索。
- **回填包含唤醒词**：ASR 文本开头出现唤醒词属正常现象，不要在后处理里强行删除，
  否则可能误删真正的命令词开头。
- **离线回归不受影响**：B3 是新增入口，不改 `test_wav_kws` 与 `test_wav_asr` 的
  默认行为；ASR 缓存改造完成后需重新跑两条离线基线。

---

## 七、原全量计划（六步版）与测试清单

最初规划不是现在的三步，而是六步；后来为了先锁住离线正确性，把「关键词文件加载」
并入解码器、把「离线 wav 入口」并入 B2，才压缩成当前的 B1 / B2 / B3。原计划保留
在这里，便于对照哪些能力已经落地、哪些测试还欠着。

### 7.1 六步版

| 原编号 | 内容 | 产物 | 现状 |
|---|---|---|---|
| B1 | ContextGraph：trie + fail/output 弧 | `kws_context_graph.h/.cpp`、`tools/test_context_graph.cpp` | 完成（`facb499`） |
| B2 | KeywordSpotter：束搜索 + 触发 + 软重置 | `kws_decode.h/.cpp` | 完成（`c339cd6`） |
| B3 | 关键词文件运行时加载，不引入拼音库 | 解析 `token... :score #threshold @原文` | 解析能力随 `KwsDecoder` 落地；仓库根 `keywords/kws_keywords.txt` 已建 |
| B4 | 离线验收入口：9 条自带 wav | `test_wav_kws.cpp` | 完成（`c339cd6`） |
| B5 | 实时双模式入口 | `test_mic_kws_asr.cpp` | 完成（2026-11-05，见第三节） |
| B6 | CMake + 文档 | `CMakeLists.txt`、README、CLAUDE.md | 完成：库与两个实时入口的 CMake、安装头文件、README 与 CLAUDE.md 已同步 |

### 7.2 测试单列的版本

后来又有一版把测试从离线入口里拆出来，编号不同但能力集合一致：

- B1：ContextGraph + `test_context_graph.out` —— 完成；
- B2：KwsDecoder + `test_wav_kws.out` —— 完成；
- B3：`tools/compare_kws.py`，C++ 与 Python 逐条对拍 —— 未做；
- B4：`test_mic_kws_asr.out`，真机双模式 —— 完成（2026-11-05）；
- B5：回归与文档 —— 完成（离线三条回归 + README / CLAUDE.md 同步）。

当前文档统一采用「B1 前缀树、B2 解码器 + 离线入口、B3 实时入口」的简化编号；
看到旧记录里的 B4 / B5 时，按上表换算即可。

### 7.3 原计划的验收顺序与判据

1. ContextGraph：搬上游 4 组期望值逐组比对，退出码 0；
2. 离线 9 条 wav：关键词集合与 Python 完全一致；首 token 时间戳允许 ≤1 帧（40 ms）；
   160 / 1600 块长结果不变；
3. 真机：唤醒 → 命令 → 10 秒超时或结束语切回；取流丢块与后台队列丢块为 0；
   同时观察 CPU；
4. 回归：`test_wav_asr.out` / `test_mic_asr.out` 行为不变；
5. 文档：README、CLAUDE.md 与 CMake 同步。

### 7.4 尚未落地的测试与收尾项

- **`tools/compare_kws.py`**：读取 `test_wav_kws.out --dump-hits` 导出的 TSV，与
  Python 侧逐条比较关键词、首 token 时间、平均概率；时间容差 1 帧。目前只有入口内
  的集合自检，没有和 Python 的逐条对拍。
- **负样本与阈值**：现有负样本只有 `zh_0~2` 三条。原计划还要求录「你好」不接
  「小智」、相似音、日常对话等负样本，把阈值定在正负样本之间；当前 0.30 / 0.35
  的依据是 30 条正样本扫描，不是负样本。
- **oracle 对拍**：Python 侧 `test_kws_oracle_mini.py` 可与 sherpa-onnx wheel 对拍；
  C++ 侧还没有对应入口，`--dump-hits` 的字段已经够用。
- **关键词三处一致**：`小智同学` 的 1.5 已在 raw、生成物、调优报告之间统一；
  改词时仍需三处一起核对，C++ 只读生成物，不读 raw。
- **实时入口统计**：已完成——`test_mic_kws_asr.out` 打印 KWS encoder 次数 / 帧数 /
  命中次数、ASR encoder 次数 / 帧数 / token 数、唤醒 / 切回次数、回填帧数、
  尾块跳过次数、设备丢块与解码丢块。
- **性能基准**：`tools/bench_perf.sh` 目前只覆盖纯 ASR；双模式入口的实时处理率、
  两类编码调用与丢块对比尚未纳入脚本。
- **文档收尾**：已完成——README 的「KWS 尚无实时入口」已替换为 oracle 对拍与
  性能基准两项遗留，CLAUDE.md 的模块表、命令表与不变量已补双模式小节。

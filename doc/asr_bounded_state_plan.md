# 实时 ASR 有界状态与转写落盘改造计划

- 状态：阶段一、二已实施并验证（2026-11-05）；阶段三（双模式入口）待做
- 范围：`audio_ai_process/asr_decode.*`、`test_wav_asr.cpp`、`test_mic_asr.cpp`，以及后续的 BlockWorker / AsrSession / 双模式入口
- 不改：`python_ref/`、`KwsDecoder`、`tools/compare_asr.py` 的对拍口径

---

## 一、背景与目标

当前 `AsrDecoder` 用 `hyp_` 保存整段会话的全部已确认 token。每个 chunk 只要出现新 token，就对全部历史重新 `detokenize` 一次。长时间常开时会出现三个问题：

1. 解码器内部状态随会话线性增长（约 8 B/token，最坏约 720 KB/小时）；
2. 每次刷新文本的代价是 O(总 token 数)，总代价随会话时长呈 O(n²)；
3. 异常退出时，已经识别出的文本没有任何落盘。

本次改造的目标：

- 解码器只保留继续解码必需的固定大小状态；
- 解码器只向外部输出「本次新增 token」，全量 token 与全量文本由外部缓存管理；
- 外部按 token 数分段结算，每次 `text_of` 只处理一段，代价与段长有关、与总时长无关；
- 每结算一段就把该段文本追加到 txt 并 `flush`，异常退出最多丢当前未结算的尾巴；
- 为后续 BlockWorker 异步改造与 KWS+ASR 双模式入口打基础。

---

## 二、职责边界

```
入口 main
 ├─ 持有 Ort::Session / AsrDecoder          （模型与解码）
 ├─ 持有 BlockWorker<LogFbankFeatures>       （线程 / 队列 / 背压）
 └─ 持有 AsrSession                          （token 累积、分段、落盘、快照）

worker 线程：BlockWorker -> AsrSession::on_block -> decoder.accept_features
主线程：只读 AsrSession::latest_text() 快照；join 后读最终结果
```

三条边界：

- **AsrDecoder**：特征 → token。内部状态全部有界；对外只吐新增 token，另外保留一个「最近 K 个 token 的尾部文本」供显示。
- **AsrSession**：全量 token、当前段 token、分段结算、txt 追加、显示快照。它是外部长期缓存的所有者。
- **BlockWorker**：不知道 token、文本、分段，只管排队、串行、背压、尾块、异常、join。

离线 wav 入口不经过 Worker，也不分段、不落盘；主线程直接持有 `all_ids`。

---

## 三、AsrDecoder 改造

### 3.1 对外接口

```cpp
class AsrDecoder {
public:
    /// 本次新增的已确认 token，按时间顺序；没有新 token 时返回空。
    /// 不含 blank、不含开头 context_size 个占位。
    std::vector<int64_t> accept_features(const LogFbankFeatures& feats);

    /// 最近 kRecentTokens 个 token 拼出的尾部文本（有界，供显示）。
    const std::string& recent_text() const;

    /// 对给定的 token 序列做一次完整 detokenize（段结算 / 收尾 / 导出用）。
    std::string text_of(const std::vector<int64_t>& ids) const;

    void reset();

    // 统计接口保持不变：
    // n_chunks() / n_enc_frames() / buffered_frames() / info()
};
```

**删除** `text()` 和 `token_ids()`。这两个接口原来的语义是「全量」，改造后解码器内部只有尾部，保留同名会让调用方在编译通过的情况下拿到错误结果，因此直接删除，让编译器强制所有调用点改过来。

不再需要 `Update` 结构体、`advanced_ms`、`silence_ms`、`has_sentence_end`：分段只按 token 数，不做时间判断，也不需要句末标点检测。

### 3.2 内部状态变化

- `std::vector<int64_t> hyp_` **删除**，换成：

  ```cpp
  static constexpr size_t kRecentTokens = 128;
  std::deque<int64_t> recent_tokens_;  // 只存已确认 token，上限 kRecentTokens
  std::string recent_text_;            // 最近 kRecentTokens 个 token 的文本
  ```

- `text_` **删除**。
- `states_`、`decoder_out_`、`feat_buf_`、`n_buffered_`、统计字段不变。
- `recent_tokens_` 只存已确认 token，不含 blank、不含开头的 `context_size` 个占位。
- 每次有新 token 时：追加、裁剪到 `kRecentTokens`、对窗口重新 `detokenize` 得到 `recent_text_`。代价 O(128)，与总时长无关。

### 3.3 `run_decoder()` 的输入与 blank 补齐

`run_decoder()` 吃最近 `context_size`（当前权重 2）个 token，输入张量形状固定 `(1, 2)`。旧实现靠 `hyp_` 初始化时的两个 blank 占位满足这个契约。

改为只存已确认 token 后，构造输入时补回来：

```
recent = []           -> 输入 [blk, blk]
recent = [t1]         -> 输入 [blk, t1]
recent = [t1, t2]     -> 输入 [t1, t2]
recent = [t1, t2, t3] -> 输入 [t2, t3]
```

这不是新逻辑，只是把旧 `hyp_` 开头两个占位改成「用时再补」，结果与现在完全等价。

与 `run_joiner()` 的关系：`run_joiner()` 的输入是 encoder 单帧和 `decoder_out`，而 `decoder_out` 来自 `run_decoder()`。流开始时如果没有 blank 占位，`decoder_out` 就没有合法的起始语言上下文，joiner 第一帧的打分会错。所以 blank 补齐是解码器内部必须的。

`run_decoder()` 仍然保持 private，**只由 `AsrDecoder::accept_features()`（每个非 blank 输出帧）和 `reset()`（一次）调用**。入口、AsrSession、BlockWorker 都不会直接调用它。

`recent_tokens_` 保留 128 个而不是 2 个，是因为 `run_decoder` 只需要最后 `context_size` 个，多出来的部分只服务 `recent_text()` 的显示窗口；两者互不影响。

### 3.4 `detokenize()` 的归属

`detokenize()` 仍留在解码器内部，因为它依赖构造期从 `tokens.txt` 建好的 `id2token_`。外部只通过两个入口消费：

- `recent_text()`：解码器内部对尾部窗口拼一次，供显示；
- `text_of(ids)`：AsrSession 在段结算和收尾时调用，把一段或全量 token 转成文字。

AsrSession **不重新实现** `▁` 替换、CJK/全宽空格规则，也不重复加载 `tokens.txt`，只决定「什么时候转、转哪一段、转完写到哪里」。

### 3.5 `reset()` 语义

- 重新分配并清零 116 个状态张量；
- `recent_tokens_` 清空，`recent_text_` 清空；
- `decoder_out_` 用 `[blk, blk]` 重算；
- `feat_buf_`、`n_buffered_`、`n_chunks_`、`n_enc_frames_` 清零；
- 不再有 `text_` 需要清。

`reset()` 只在入口显式要求时调用：启动、切换模式、换文件、显式重新开始。**静音、段结算、最大长度都不触发 reset**。

---

## 四、外部缓存与 AsrSession

### 4.1 外部数据

```cpp
std::vector<int64_t> all_ids;      // 全量 token，唯一事实来源
std::vector<int64_t> segment_ids;  // 当前段的 token
```

- `all_ids` 会长，这是外部长期缓存的合理成本（8 B/token，最坏约 720 KB/小时）。
- `segment_ids` 每满 64 个就结算并清空，所以每次 `text_of` 的输入规模固定很小。
- 最终全量文本用 `decoder.text_of(all_ids)` 算一次，与分几段无关，与旧 `text_` 逐字节一致。
- 不保留已结算段的文本数组：段文本结算后直接追加到 txt，最终文本从 `all_ids` 重算，减少一份长期内存。

### 4.2 谁持有谁管理

- `test_wav_asr.cpp`：主线程直接持有 `all_ids`；不分段、不落盘；收尾 `decoder.text_of(all_ids)`。
- `test_mic_asr.cpp`：`AsrSession` 在 worker 线程持有 `all_ids`、`segment_ids`、txt 文件和快照；主线程只读快照。
- 双模式入口：`KwsAsrSession` 内部持有一个 `AsrSession` 成员负责 ASR 分支。
- 最终打印与 `--dump-tokens` 都在 `finish_and_join()` 之后读取，避免并发。

### 4.3 AsrSession 接口

```cpp
using AudioFeatureInput = BlockWorker<LogFbankFeatures>::Item;

struct AsrSessionConfig {
    size_t segment_tokens = 64;  // 当前段满 64 个 token 就结算
    std::string transcript_path = "log/asr_last_text.txt";
};

class AsrSession {
public:
    AsrSession(AsrDecoder& decoder, AsrSessionConfig cfg);

    void on_block(const AudioFeatureInput& item);  // worker 线程
    std::string latest_text() const;               // 有界显示文本

    // worker.finish_and_join() 之后调用
    std::string finish();                          // 结算尾段、关闭文件、返回全量文本
    const std::vector<int64_t>& all_token_ids() const;
    int n_segments() const;
};
```

### 4.4 `on_block()` 流程

```
1. ids = decoder.accept_features(item.block)
2. all_ids.insert(ids)
   segment_ids.insert(ids)
3. if (segment_ids.size() >= cfg.segment_tokens) close_segment()
4. snapshot = decoder.recent_text()
```

AsrSession 不调用 `run_encoder` / `run_joiner` / `run_decoder`，所有模型调用都留在 `accept_features()` 内部；`text_of()` 是纯查表 + 字符串处理，没有推理。

### 4.5 段结算

`close_segment()`：

```
1. 若 segment_ids 为空，直接返回
2. text = decoder.text_of(segment_ids)
3. 追加一行到 transcript_path，flush
4. segment_ids.clear()
5. ++n_segments
```

规则：

- 段内 token 数达到 64 就立即结算，不等句末标点、不按时间判断；
- 解码器在段边界**不 reset**，跨段声学上下文连续；
- 正常退出、切模式时，`finish()` 把不足 64 的尾段也结算；
- txt 一行一段，行内可能断在句子中间，这是「按 token 数分段」的已知取舍；
- 文本中的换行替换为空格，保证一行一段；UTF-8 编码。

### 4.6 txt 落盘

文件名固定为 `log/asr_last_text.txt`，不带时间戳：

- `AsrSession` 构造时创建 `log/`（`std::filesystem::create_directories`），并以 `std::ios::trunc` 清空文件；
- 之后每次段结算以 `std::ios::app` 追加一行并 `flush`；
- 不做 5 秒定时刷写、不做临时文件 + rename、不搞第二个文件；
- 崩溃最多丢当前未结算的尾巴（不足 64 个 token）；
- 正常退出时 `finish()` 结算最后一段，不会丢；
- 写失败只打印一次 stderr，不抛出、不中断解码。

`log/` 加入 `.gitignore`，目录留 `.gitkeep`（与 `results/` 的做法一致）。

### 4.7 `finish()`

```
1. 若 segment_ids 非空，close_segment()
2. 关闭文件
3. 返回 decoder.text_of(all_ids)
```

由主线程在 `worker.finish_and_join()` 之后调用，此时 worker 线程已退出，不存在并发访问。

### 4.8 显示

- 解码器提供 `recent_text()`：最近 128 个 token 的文本，有界，保证解码器单独使用时也有显示能力。
- AsrSession 的快照直接取 `decoder.recent_text()`，不额外拼历史尾部；因为段边界不 reset，`recent_text()` 本身就是跨段滚动的窗口，再拼历史会重复。
- 主线程每帧读取快照，按像素宽度从尾部截断绘制；不拷贝全量文本。
- 解码器显式 reset 时 `recent_text()` 清空，显示随之清空，符合「重新开始」的语义。

---

## 五、BlockWorker 异步改造

新增 `audio_ai_process/block_worker.h`（header-only 模板），语义沿用 `doc/audio_ai_block_worker.md`：

```cpp
template <typename Block>
class BlockWorker {
public:
    struct Item { Block block; bool is_tail = false; };
    using Handler = std::function<void(const Item&)>;
    using Skip = std::function<bool(const Block&)>;

    explicit BlockWorker(Handler handler, Skip skip = {},
                         size_t max_pending = 50);

    bool push(Block block);          // 非阻塞；满时丢最旧并计数
    void push_final(Block block);    // 尾块，阻塞等空位，绝不丢
    void finish_and_join();
    size_t dropped() const;
    size_t max_pending() const;
    std::exception_ptr error() const;
};
```

入口装配：

```cpp
using AudioFeatureInput = BlockWorker<LogFbankFeatures>::Item;

AsrSession session(decoder, cfg);
BlockWorker<LogFbankFeatures> worker(
    [&session](const AudioFeatureInput& item) { session.on_block(item); },
    [](const LogFbankFeatures& f) {
        return f.n_frames <= 0 || f.data.empty();
    });
```

Worker 语义：

- 单消费者线程，handler 串行执行，解码器因此不需要加锁；
- 有界队列，满时丢最旧并计入 `dropped()`；
- `push_final()` 用于 `input_finished()`，阻塞等空位，不丢；
- 0 帧块由 skip 过滤，不占容量、不计丢弃；
- handler 异常记录后停止消费，`error()` 可查；
- `finish_and_join()` 幂等，析构兜底 join。

收尾顺序：

```
worker.push_final(front.input_finished())
worker.finish_and_join()
session.finish()
打印 / dump
```

---

## 六、实施阶段

### 阶段一：解码器有界化 + 增量输出（已完成 2026-11-05）

**改动文件**

1. `audio_ai_process/asr_decode.h`
   - 增加 `recent_text()`、`text_of()`；
   - 增加 `recent_tokens_`、`recent_text_`；
   - 删除 `hyp_`、`text_`、`token_ids()`、`text()`；
   - `accept_features()` 返回值改为 `std::vector<int64_t>`。
2. `audio_ai_process/asr_decode.cpp`
   - 重写 `accept_features()`：追加特征 → 出块 → 贪心 → 新 token 追加并裁剪 → 维护窗口文本 → 返回新增 id；
   - `run_decoder()` 输入改为「最近 `context_size` 个，不足补 blank」；
   - 新增 `text_of()`，复用现有 `detokenize()`；
   - `reset()` 清理新状态。
3. `test_wav_asr.cpp`
   - 主线程维护 `std::vector<int64_t> all_ids`；
   - 实时显示改 `decoder.recent_text()`；
   - 收尾 `final_text = decoder.text_of(all_ids)`；
   - `--dump-tokens` 用 `all_ids`，格式不变。
4. `test_mic_asr.cpp`（最小适配，阶段二整体替换）
   - `AsrWorker` 增加 `all_ids`，每次 append `accept_features` 返回的 id；
   - 快照写 `decoder.recent_text()`；
   - join 后用 `decoder.text_of(all_ids)` 得最终文本；
   - 删除 `token_ids()` 调用。

**验证**

- `cmake --build build -j` 通过；
- 改动前先导出基线：`test_wav_asr.out --no-window --dump-tokens /tmp/before.txt`；
- 改动后导出 `/tmp/after.txt`，逐字节 `diff`；
- 四条 wav 的 stdout「识别结果」与改动前 diff；
- `python3 -I tools/compare_asr.py --cpp /tmp/after.txt --wav-dir <目录>` 四个 wav 全部一致；
- `test_mic_asr.out --no-window --seconds 10` 冒烟。

**回滚**：只涉及上述 4 个文件，整体还原即可。

### 阶段二：BlockWorker + AsrSession + 分段 + 落盘（已完成 2026-11-05）

**改动文件**

1. 新增 `audio_ai_process/block_worker.h`；
2. 新增 `audio_ai_process/asr_session.h/.cpp`；
3. `test_mic_asr.cpp`：删除本地 `AsrWorker`，改为 `AsrSession` + `BlockWorker`；
4. `CMakeLists.txt`：`asr_session.cpp` 加入 `lib_audio_ai_process`，新头文件随库安装；
5. `.gitignore`：加入 `log/`；`log/` 目录放 `.gitkeep`；
6. `doc/audio_ai_block_worker.md`：补 AsrSession 的最终形态。

**验证**

- `test_mic_asr --no-window --seconds 10` 文本正常；
- `log/asr_last_text.txt` 在每满 64 token 后追加一行，行数与段数一致；
- 长跑 3～5 分钟观察段数、文件增长、后台队列丢弃、CPU；
- `kill -9` 后最多丢当前未满 64 token 的尾巴，已结算段都在文件里；
- 正常退出后尾段被结算，文件内容与最终文本一致；
- `test_wav_asr` 不受影响；
- `./build/test_context_graph.out`、`./build/test_wav_kws.out` 仍通过。

**回滚**：阶段二只影响 `test_mic_asr` 与新文件，删除新文件并还原入口即可。

**实施结果（2026-11-05）**

- 阶段一：`asr_decode` 改为只保留 `recent_tokens_`（128）与 `recent_text_`，
  `accept_features()` 返回增量；`test_wav_asr` / `test_mic_asr` 均改为外部累积
  `all_ids`。改动前后 4 条 wav 的 token 导出与「识别结果」文本逐字节一致，
  `compare_asr.py` 四个 wav 全部逐 token 一致。
- 阶段二：新增 `block_worker.h`、`asr_session.h/.cpp`；`test_mic_asr.cpp` 删除本地
  `AsrWorker`，改用 `BlockWorker` + `AsrSession`；CMake 与 `.gitignore` 同步，
  `log/.gitkeep` 入库。真机 20 s 实测识别出 8 个 token，退出时结算 1 段，
  `log/asr_last_text.txt` 恰好一行且与最终文本一致，取流丢块与解码丢块均为 0。
- 分段逻辑另用临时离线驱动验证（`segment_tokens=8`）：27 token 切成 3 段，
  txt 行数 = `n_segments()`，最终文本与不分段时逐字节一致。

### 阶段三：双模式入口（原 B3）

- 新增 `keywords/kws_keywords.txt`；
- 新增 `test_mic_kws_asr.cpp` 与 `KwsAsrSession`；
- `KwsAsrSession` 内部持有 `KwsDecoder` 和一个 `AsrSession` 成员；
- 复用 `BlockWorker`，不新增 Worker 类；
- 命令行开关：`--no-window`、`--cuda`、`--seconds`、`--initial-asr`、`--keywords`；
- 验证 KWS 命中集合、唤醒后 ASR 文本、切回 KWS 后 ASR 段落盘。

**回滚**：纯新增入口与文件，删掉即可。

### 为什么是这个顺序

1. 解码器契约先稳定，`AsrSession` 才能一次写对，不用先按旧全量 API 写一版再推翻；
2. 阶段一可以独立用 wav 对拍验证，不依赖线程改造；
3. 阶段二只影响麦克风入口，出问题容易定位；
4. 阶段三是在前两步基础上的新入口，风险最低。

---

## 七、参数定稿

| 参数 | 值 | 作用 |
|---|---|---|
| `kRecentTokens` | 128 | 解码器内部保留的 token 数，只影响尾部文本 |
| `segment_tokens` | 64 | 当前段满 64 个 token 就结算并追加一行 |
| `transcript_path` | `log/asr_last_text.txt` | 启动截断，之后按段追加 |
| 队列上限 | 50 | BlockWorker 有界队列 |

不再有：最小段长、静音阈值、最大段长、flush 间隔、句末标点判定。

---

## 八、验证矩阵

| 层次 | 命令 | 期望 |
|---|---|---|
| 编译 | `cmake --build build -j` | 通过 |
| 模型契约 | `./build/load_onnx_model.out` | 通过 |
| 前端 | `python3 -I tools/compare_fbank.py` | 逐位一致（不受影响） |
| token 对拍 | `test_wav_asr --dump-tokens` + `compare_asr.py` | 四个 wav 全一致 |
| 文本回归 | 改动前后 stdout「识别结果」diff | 逐字节一致 |
| 实时冒烟 | `test_mic_asr --no-window --seconds 10` | 文本正常、txt 生成 |
| 段结算 | 10 秒实时运行 | txt 行数 = `n_segments` |
| 崩溃恢复 | `kill -9` | 已结算段都在，最多丢未满 64 token 的尾巴 |
| KWS 回归 | `test_context_graph.out`、`test_wav_kws.out` | 通过 |
| 长跑 | 3～5 分钟实时运行 | 内存不随 token 历史增长，队列丢弃可观测 |

---

## 九、风险与边界

- **接口删除是故意的**：`text()` / `token_ids()` 不保留同名，避免静默改变语义。
- **最终文本精确性**：`text_of(all_ids)` 与旧 `text_` 使用同一个 `detokenize` 和同一 token 序列，离线对拍必须逐字节一致。
- **txt 行内可能断句**：按 64 token 硬切，不保证行尾是句号；这是简化换来的确定性，最终文本仍以 `text_of(all_ids)` 为准。
- **崩溃丢失窗口**：最多丢当前未结算的 63 个 token，不是固定时间窗口。
- **解码器不因静音 reset**：与改造前行为一致，长跑不会涨状态；需要重新开始时由入口显式 `reset()`。
- **落盘失败不拖垮解码**：只告警，不抛异常。
- **暂停/恢复暂不结算段**：`stop()` / `start()` 之间如果间隔很长，恢复后解码器上下文偏旧；后续可用控制块触发结算。
- **离线对拍不受分段影响**：`test_wav_asr` 不做分段，`compare_asr.py` 口径不变。
- **线程安全**：`AsrSession` 只在 worker 线程写，`finish()` 在 join 之后由主线程调用；快照用私有 mutex 保护。

---

## 十、文档同步

- `README.md`：更新 AsrDecoder 契约、实时入口行为、txt 落盘与分段说明（随代码提交）；
- `CLAUDE.md`：同步不变量与模块职责（本地维护，不入库）；
- `doc/audio_ai_block_worker.md`：阶段二补 AsrSession 最终形态；
- 本文件：实施过程中如有变更，同步更新。

---

## 十一、决策记录

| 议题 | 结论 |
|---|---|
| 解码器是否保留全量 `hyp_` | 不保留，只留最近 128 个 token |
| `accept_features()` 输出 | 本次新增 token id 的 `std::vector<int64_t>` |
| 全量 token 谁持有 | 外部入口 / AsrSession 的 `all_ids` |
| `detokenize` 归属 | 留在解码器，外部通过 `text_of()` 调用 |
| 句末标点分段 | 取消 |
| 静音 reset | 取消 |
| 最大段长时间 | 取消，由 64 token 上限替代 |
| 段结算规则 | 当前段满 64 token 立即结算；退出/切模式结算尾段 |
| 段结算是否 reset 解码器 | 不 reset |
| 落盘方式 | 单文件追加，每段一行并 flush；启动截断 |
| 落盘路径 | `log/asr_last_text.txt` |
| 显示文本 | 解码器 `recent_text()`，窗口 128 token |
| 改造顺序 | 解码器 → Worker/AsrSession → 双模式 |

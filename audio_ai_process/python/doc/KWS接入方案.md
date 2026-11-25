# 关键词唤醒（KWS）接入 mini 分支方案

> 目标：在 `test_mic_kws_asr.py` 里并存两条识别链路——默认关键词唤醒，命中后切到
> 完整 ASR；长时间无新字或解出结束语后切回 KWS。**不改**现有前端的数值口径，
> **不引入** sherpa-onnx 运行时，KWS 解码逻辑按 C++ 参考实现逐行移植，为后续
> C++ 部署保留同一套语义。

---

## 0. 结论先行

新增和改动如下，其余文件保持不动：

| 文件 | 动作 | 说明 |
|---|---|---|
| `src/kws_decode.py` | 新增 | KWS 流水线：关键词 trie + 束搜索 + 触发判定 |
| `src/dialog_mode.py` | 新增（可选） | 模式状态机；先写在入口里，稳定后再抽 |
| `keywords/kws_keywords.txt` | 新增 | 仓库自有的关键词文件（预生成 token，运行时只读） |
| `tools/gen_keywords.py` | 新增（开发期） | 文本 → token 文件的离线生成器，非运行时依赖 |
| `test_wav_kws_mini.py` | 新增 | 离线验收入口，跑模型自带 9 条 wav |
| `test_mic_kws_asr.py` | 修改 | 同时装配 KWS/ASR，主循环按模式分派 |
| `src/audio_preprocess.py` | 不动 | KWS 与 ASR 共用同一份 80 维 fbank |
| `src/asr_decode.py` | 不动 | ASR 分支保持原行为 |
| `src/mic_client.py` | 不动 | 麦克风仍是“裸 PCM 生产者” |

推荐默认使用 **chunk-8** 模型：每 160 ms 推进一次、130 ms 卷积前瞻，唤醒延迟明显低于
chunk-16（320 ms 推进）。两者输出帧长都是 40 ms。

---

## 1. 从模型文件读到的事实

以下数值全部来自 onnx 的 `custom_metadata_map`，运行时由代码读取，不硬编码。

| 项 | chunk-8 权重 | chunk-16 权重 |
|---|---|---|
| `model_type` | `zipformer2` | `zipformer2` |
| `T`（encoder 输入帧数） | 29 | 45 |
| `decode_chunk_len`（时间轴前移） | 16 帧 = 160 ms | 32 帧 = 320 ms |
| 每块 encoder 输出帧 | 4 | 8 |
| 输出帧长 | 40 ms | 40 ms |
| 卷积重叠上下文 `T - shift` | 13 帧 = 130 ms | 13 帧 = 130 ms |
| `context_size`（decoder） | 2 | 2 |
| `vocab_size`（joiner） | 263 | 263 |

> 注意：输出帧是 **40 ms/帧（25 Hz）**，不是 20 ms。C++ 参考实现里
> `trailing_silence = num_trailing_blanks * 4 * 0.01` 正是按 40 ms 算的。

词表共 263 个音素级单元：3 个特殊符 + 69 个 CMU 音素 + 191 个部分拼音。
关键词匹配是**发音序列匹配**，不是文字匹配。

前端口径：80 维 fbank、16 kHz、25 ms 窗 / 10 ms 移。与
`src/audio_preprocess.py` 的默认输出一致，`dither` 推理时取 0（本仓库已如此）。
因此 KWS 和 ASR 可以**共用同一个 `AudioPreprocess` 实例**。

---

## 2. `src/kws_decode.py`

### 2.1 对外接口

与 `AsrDecoder` 同构：只接收调用方建好的三个 session，不碰 onnxruntime、
不定位权重文件。

```python
class KeywordSpotter:
    def __init__(self, encoder, decoder, joiner, tokens_path, keywords_path,
                 max_active_paths=4, num_trailing_blanks=1,
                 keywords_score=1.0, keywords_threshold=0.25,
                 reset_after_silence=1.5): ...

    def accept_features(self, feats) -> list[KeywordHit]:
        """喂一块 (n, 80) 特征，返回本次新触发的关键词列表（可能为空）。"""
    def reset(self): ...
    def info(self) -> dict: ...
```

```python
@dataclass
class KeywordHit:
    keyword: str          # 原文，来自 @原文；没有则用 token 拼
    tokens: list[str]
    timestamps: list[float]   # 秒
    start_time: float
    avg_prob: float
```

`accept_features` 内部维护：

- `_feat_buf`：未凑满 `T` 的特征尾巴；
- `states`：encoder 的全部状态张量（数量由输入签名推导）；
- `hyps`：当前 beam（≤ `max_active_paths` 条）；
- `_frame_offset`：已解码的输出帧数，用于时间戳；
- `_trailing_blanks`：最佳假设连续 blank 的输出帧数。

### 2.2 内部结构

1. **`_ContextGraph`**：关键词前缀树 + fail 链 + output 弧。
   逐项复刻 `sherpa-onnx/csrc/context-graph.cc`：
   - `ForwardOneStep(state, token) -> (score, new_state, matched_node)`
   - `IsMatched(state) -> (bool, matched_node)`
   - 节点字段：`token / token_score / node_score / output_score / level /
     ac_threshold / is_end / phrase / next / fail / output`
   - fail 与 output 弧在构建完成后用 BFS 填一次（`FillFailOutput`）。

2. **`_Hypothesis`**：`ys`（token 序列）、`ys_probs`（每个 token 的声学概率）、
   `timestamps`、`log_prob`、`context_state`（trie 节点引用）、
   `num_trailing_blanks`。

3. **`_Hypotheses`**：以 `"-".join(map(str, ys))` 为 key 的字典；
   同 key 合并时用 log-sum-exp（对应 C++ 的 `Hypotheses::Add`）。

### 2.3 每帧解码（与 C++ 对齐）

对 encoder 输出的每个时间帧 `t`：

1. 取当前 beam 的所有假设 `prev`（合并后，≤ 4 条）；
2. `decoder_input = [h.ys[-context_size:] for h in prev]`，**一次 batch 跑 decoder**；
3. 把当前 encoder 帧复制成 `len(prev)` 份，**一次 batch 跑 joiner**；
4. 对每行做 log-softmax，加上该假设的 `log_prob`；
5. 在 `len(prev) * 263` 个候选里取 top-`max_active_paths`；
6. 对每个候选：
   - 若 `token != 0 and token != unk_id`（**blank 硬编码 0**，unk 视作 blank）：
     - `ys.append(token)`、记录时间戳与 `exp(logprob)`；
     - `num_trailing_blanks = 0`；
     - `(context_score, new_state, _) = trie.ForwardOneStep(...)`；
     - 若 `new_state` 回到根（`token == -1`），把 `ys` 重置为
       `[-1] * (context_size - 1) + [0]`，清空时间戳与 `ys_probs`
       （对应 C++ 的 “start matching from the start state”）；
   - 否则 `num_trailing_blanks += 1`；
   - `log_prob = 候选分 + context_score`；
   - 加入 `_Hypotheses`（同 key 合并）。
7. 取最佳假设，`IsMatched`：
   - 若 `num_trailing_blanks > num_trailing_blanks_` 且
     `mean(ys_probs[-level:]) >= ac_threshold`，触发；
   - 触发后把 beam 重置为单条 blank 假设（`ys = [-1,...,0]`，`context_state = root`）。

> `-1` 不要 clamp 成 0：ONNX `Gather` 支持负索引，`-1` 取的是词表最后一行。
> 这是 C++ 参考实现的真实行为，照抄即可。

### 2.4 静音自动重置

每处理完一块，若最佳假设的 `num_trailing_blanks * 40 ms > 1.5 s`，调用
`self.reset()`（含 encoder 状态）。与 C++ `DecodeStreams` 的行为一致。

### 2.5 关键词文件解析

每行：`token token ... [:boost] [#threshold] [@原文]`

- token 必须存在于 `tokens.txt`，否则启动即报错并打印行号；
- `:1.5` 是 boosting score，缺省用 `keywords_score`；
- `#0.35` 是触发阈值，缺省用 `keywords_threshold`；
- `@原文` 用于触发后展示；中文必须有，且原文不能含空格（用 `_`）；
- 注释行以 `#` 开头时注意与阈值区分：阈值一定紧跟在 token 之后且形如 `#0.35`，
  建议约定“整行以 `#` 开头且后面不是数字”才是注释。

---

## 3. 关键词文件的生成

**运行时只读预生成的 token 文件**，C++ 侧可直接复用同一份。

三条路径，按优先级：

1. **零依赖验证**：直接复用
   `weight/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/test_wavs/keywords.txt`
   的 9 条关键词。先把 `test_wav_kws_mini.py` 跑通。
2. **开发期生成器** `tools/gen_keywords.py`：
   - 英文：读模型目录的 `en.phone`（CMU 发音词典），`WORD -> phones`；
   - 中文：需要 `pypinyin`（纯 Python，`pip install pypinyin`）；
   - 输出 `tokens... [:score] [#thr] @原文`。
   - 该脚本只在开发期用，不进入运行时依赖，和 `kaldi_native_fbank` 的定位一致。
3. **有 sherpa-onnx 环境时**：
   `sherpa-onnx-cli text2token --tokens tokens.txt --tokens-type phone+ppinyin
   --lexicon en.phone --text raw.txt --output keywords.txt`

> 当前 `py310-gpu` 环境里 `pypinyin` 和 `sherpa_onnx` 都没装。第一版建议先用
> 现成的 `keywords.txt` 验收链路，唤醒词确定后再决定装哪个生成器。

---

## 4. 模式状态机

放在入口脚本里（`main` 的一个小函数或两个常量 + 分支），先不抽象成类。
状态只有两个：

```text
MODE_KWS  --- 命中关键词 --->  MODE_ASR
MODE_ASR  --- 空闲超时 / 结束语 --->  MODE_KWS
```

### 4.1 KWS → ASR

触发条件：`KeywordSpotter.accept_features()` 返回非空。

动作：

1. `asr_decoder.reset()`；
2. 可选：把入口维护的最近 `ASR_BACKFILL_FRAMES`（默认 100 帧 = 1 s）特征回填给
   ASR，避免唤醒词末尾与命令开头之间被 chunk 边界吃掉；
3. 记录 `last_token_count = 0`、`last_activity = now`；
4. `mode = MODE_ASR`；
5. 控制台打印 `[唤醒] <keyword>`，窗口 footer 切成 `ASR`。

> 回填会让 ASR 文本开头可能出现唤醒词本身，通常可接受；如果不想看到，把
> `ASR_BACKFILL_FRAMES` 设为 0，代价是命令第一个音节可能晚 100~300 ms 才进 ASR。

### 4.2 ASR → KWS

满足任一条件：

- **空闲超时**：`now - last_activity > ASR_IDLE_TIMEOUT`（默认 10 s）。
  `last_activity` 在 `len(asr_decoder.token_ids)` 增加时刷新。
- **结束语**：当前累计文本去掉尾部中英文标点后，以 `END_PHRASES` 中任一项结尾。
  默认 `["再见", "拜拜", "结束", "退出", "没事了"]`，在入口顶部可改。

动作：

1. `asr_decoder.reset()`；
2. `keyword_spotter.reset()`；
3. `mode = MODE_KWS`；
4. 打印 `[切回] 关键词模式`。

### 4.3 主循环骨架

```python
while not quit_now:
    samples = mic.get_last_audio()
    if samples is None:
        time.sleep(0.002)
    else:
        feats = preprocessor.accept_waveform(samples)
        cv_frame = spec_view.accept_features(feats)

        if mode == MODE_KWS:
            hits = kws.accept_features(feats)
            if hits:
                ... 切到 ASR ...
            status = "KWS listening"
        else:
            text = asr_decoder.accept_features(feats)
            ... 刷新 last_activity / 检查结束语 / 超时 ...
            status = "ASR"
        ... 画 footer、按键处理 ...
```

两个解码器**只消费自己活跃期间的 feats**；切换时双方 `reset()`，不共享任何状态。
前端实例始终连续，不重置，因此特征时间轴不会断。

### 4.4 收尾

- 退出时若处于 ASR 模式：保持现有收尾逻辑（补尾帧 + 100 帧 0，只喂 ASR）。
- 退出时若处于 KWS 模式：KWS 不需要补 0，也无需最后再解一次，直接停麦、关窗。

---

## 5. 可视化与交互

- 窗口 footer 用 ASCII 状态串：`KWS | listening`、`ASR | <text>`。
  OpenCV 的 `FONT_HERSHEY_SIMPLEX` 不支持中文，识别文本仍以控制台为准；
  现版本把中文直接 `putText` 到窗口本来就会画成方框，这次顺手改成 ASCII 状态即可。
- 触发唤醒时 footer 背景短暂变绿/文字变红 1 帧即可（可选，不影响主流程）。
- 控制台保留原地刷新，并额外打印 `[唤醒]` / `[切回]` 两类事件行。

---

## 6. 验收

所有脚本执行前先征得同意。

### 6.1 离线 KWS：`test_wav_kws_mini.py`

用模型自带的 9 条 `test_wavs/*.wav` + `keywords.txt`，逐条打印命中关键词、
时间戳与平均概率。期望：

| wav | 期望命中 |
|---|---|
| `en_0.wav` | `LIGHT UP` |
| `en_1.wav` | `LOVELY CHILD` |
| `zh_0..6.wav` | 文森特卡索 / 周望军 / 朱丽楠 / 蒋友伯 / 女儿 / 法国 / 见面会 / 落实（按文件顺序核对） |

若个别中文短词不触发，先调 `keywords_score` 与 `#threshold`，再怀疑实现。

### 6.2 端到端麦克风

1. 静音启动：窗口显示 `KWS listening`，控制台无识别文本；
2. 说唤醒词：控制台出现 `[唤醒]`，footer 变 `ASR`；
3. 说一句命令：ASR 文本逐步出现；
4. 停口 10 s：控制台出现 `[切回]`，footer 回到 `KWS listening`；
5. 再说结束语（如“再见”）：立即切回 KWS。

### 6.3 回归

- `test_wav_asr_mini.py` 结果不变；
- `test_mic_kws_asr.py` 在 `INITIAL_MODE = "asr"`（调试开关）时仍等价于原行为。

### 6.4 性能

在 KWS 模式下统计每块 ONNX 调用次数：chunk-8 每 160 ms 触发
4 次 joiner + 4 次 decoder（batch≤4）+ 1 次 encoder。与 ASR 贪心模式同量级，
单线程 CPU 足够；若 CPU 吃紧，优先换 chunk-16 或把 `max_active_paths` 降到 2。

---

## 7. 风险与待定项

| 项 | 现状 | 建议 |
|---|---|---|
| 唤醒词未定 | 只有模型自带示例词 | 先跑通链路，再定 2~4 个词的唤醒词并生成 token |
| 中文生成依赖 | `pypinyin` / `sherpa_onnx` 均未安装 | 第一版用自带 `keywords.txt`；定词后再装 `pypinyin` |
| 回填策略 | 未实现 | 默认回填 100 帧，设 0 可关 |
| 中文同音误触 | 拼音级模型固有属性 | 用多字词 + 提高 `#threshold`；上线前按实测调参 |
| 结束语列表 | 硬编码在入口 | 收敛到顶部常量，后续可外置 |
| 窗口中文 | OpenCV 画不出中文 | 窗口只用 ASCII 状态，中文走控制台 |
| chunk 选择 | 两套权重都在 | 默认 chunk-8；若帧率不足再评估 chunk-16 |

---

## 8. 实施记录（2026-11-04）

### 8.1 已完成并验证

| 项 | 状态 | 证据 |
|---|---|---|
| `src/kws_decode.py` | 已完成 | 见第 2 节；`accept_features(feats) -> list[KeywordHit]` |
| trie（ContextGraph） | 通过 | `test_context_graph_mini.py` 搬了上游 4 组 gtest 用例，全部命中期望总分 |
| 9 条自带 wav | 通过 | `test_wav_kws_mini.py`：2 英 + 7 中，关键词集合与期望完全一致 |
| 解码头对拍 | 通过 | `test_kws_oracle_mini.py`：与 `pip install sherpa-onnx`（2.0.1）逐条一致，时间戳误差 0 ms |
| 流式不变性 | 通过 | 同一 wav 按 160 / 1600 样本喂，命中与时间戳一致 |
| 入口模式切换 | 已实现，待真机联调 | `test_mic_kws_asr.py` 默认 KWS，命中切 ASR，10 s 空闲或结束语切回 |
| 仓库关键词文件 | 已完成 | `keywords/kws_keywords.txt`（10 条，运行时不依赖 weight/ 里的原始文件） |

### 8.2 实施中发现的、和计划不同的事实

1. **这个 KWS 模型的 joiner 要二维输入**：`encoder_out` 是 `(N, 320)`，
   不是常规 transducer 的 `(N, 1, 320)`。以 onnx 输入签名为准，
   `run_joiner()` 已按二维实现。
2. **`zh_0~2.wav` 是负样本**：用 ASR 模型转写过，讲的是经济话题，
   不含任何关键词；`zh_3~6` 每条各含两个关键词。期望表已按此写死。
3. **`KeywordHit.start_time` 的语义和 C++ 不同**：C++ 取「取结果那一刻」的
   帧时间；本项目取关键词第一个 token 的绝对时间，对使用者更有意义。
4. **离线对拍时不要调 `reset_stream()`**：它会把 C++ 侧的帧偏移清零，
   同一文件里第二个关键词的时间戳会变成「相对上次重置」，与本项目的绝对
   时间戳对不上。底层去重逻辑已经保证同一结果不会重复返回。
5. **前端口径差异不影响结论**：本项目 `high_freq=0`（8000 Hz），
   sherpa-onnx wheel 默认 `-400`（7600 Hz），但 9 条 wav 的关键词与时间戳
   完全一致 —— 至少在这批样本上，7600~8000 Hz 的差异可以忽略。

### 8.3 仍待办

- 真机麦克风联调：唤醒 → 命令 → 超时/结束语切回；
- 确定正式的 2~4 个唤醒词，并决定用 `pypinyin` 还是 `sherpa-onnx-cli`
  重新生成 token 文件；
- README 增加 KWS 章节；
- 阈值调参（中文短词的同音误触）。

---

## 9. 实施顺序

1. `src/kws_decode.py` 的 trie 单元自检；
2. `test_wav_kws_mini.py` 跑通 9 条自带 wav（此时不改入口）；
3. `keywords/kws_keywords.txt` + `tools/gen_keywords.py`（按定词结果）；
4. 改 `test_mic_kws_asr.py` 接模式状态机；
5. 联调、调阈值、补 README / CLAUDE.md 说明；
6. 为后续 C++ 移植补一节「Python ↔ C++ 函数对照表」（本文件第 2.3 节已按此组织）。

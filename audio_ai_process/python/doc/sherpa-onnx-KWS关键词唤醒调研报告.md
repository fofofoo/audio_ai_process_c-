# sherpa-onnx KWS 关键词唤醒调研报告

> 调研日期：2026-11-04
> 调研对象：`sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20`（chunk-8 / chunk-16 两套权重）
> 上游版本：sherpa-onnx（本地源码副本 `sharpa-onnx/`）
> 目标：摸清「唤醒词 → 运行时匹配」的完整链路，为确定正式唤醒词和后续 C++ 移植提供依据。
>
> 相关文档：[KWS 接入方案](KWS接入方案.md)、[README · 关键词唤醒](../README.md#关键词唤醒kws)、[上游 Keyword spotting 文档](Keyword_spotting_index.rst)

---

## 阅读地图

本报告按「一个契约、两个阶段」组织：

| 章节 | 属于 | 回答什么 |
|---|---|---|
| 二 ~ 三 | 全局 | KWS 是什么，为什么能切成两半 |
| **四** | **契约** | 263 个符号从哪来，两侧凭什么能对接 |
| 五 | 生成侧 | 汉字/英文怎么变成音素符号 |
| 六 | 接口 | 两侧交接的文件长什么样 |
| 七 | 识别侧 | 声波怎么变成一次唤醒事件 |
| 八 ~ 十二 | 工程 | 与 ASR 的关系、验证、交付结论 |

---

## 一、结论先行

1. **开集 KWS 不是分类器**，而是「词表被关键词前缀树约束住的小型流式 transducer ASR」。模型是通用音素识别器，「只认这几个词」完全靠解码器。
2. **KWS 可以清晰地切成两个阶段**：关键词生成（离线）与关键词识别（运行时）。两者的产物与依赖不同，失败代价也不同。
3. **联系两个阶段的不是某一份字典，而是一套发音符号集**——`tokens.txt` 的符号列。生成侧向它对齐，识别侧从它取 id。
4. **不需要录制唤醒词，也不需要模型输出音素**。从原文到音素符号是查表（`pypinyin` / CMU 词典），从音素到 id 也是查表。
5. **第一级是 Python（`text2token`），第二级是 C++（`EncodeKeywords`）**。前者是 TTS 前端意义上的 G2P，不是 TTS；后者是运行时加载。
6. **声调不是「被强调」，而是被显式编码**：中文音节拆成「声母 + 带调韵母」，声调用 Unicode 符号（`ō ó ǒ ò`）写在韵母 token 上；英文的 `0/1/2` 是 CMU 重音等级，不是声调。
7. **词表 263 = 3 特殊符 + 69 CMU 音素 + 191 拼音单元**。同为 `zh` 与 `ZH`，在拼音体系与 CMU 体系里是**两个不同的 token**。
8. **换唤醒词的成本 = 改一行文本 + 跑一次 text2token**。C++ 运行时不需要引入任何拼音库。

---

## 二、开集 KWS 的定位

传统（闭集）KWS 是一个分类问题：为每个唤醒词准备大量正负样本，训练出来的模型**只能识别训练时见过的词**。

开集（open-vocabulary / customized）KWS 反过来：

> 一个通用音素识别模型不动，靠**解码器**把输出限制在给定的发音序列集合上。

模型大小与唤醒词数量无关，可以随时增删关键词而无需重训。sherpa-onnx 提供两个逐词可调的旋钮来平衡「触发率 ↔ 误报率」：

| 参数 | 作用 | 方向 |
|---|---|---|
| boosting score (`:x`) | 解码时给含关键词的路径加分，帮它们活过束搜索 | 越大越容易触发 |
| trigger threshold (`#x`) | 要求关键词各 token 的平均声学概率不低于该值 | 越大越**难**触发 |

逐词可调意味着能只收紧有问题的词，而不影响其他词。

这个设计直接导致了后面的二分：既然模型不关心具体是哪些词，那么「给出哪些词」就成了一个**独立的、可以先于识别完成的阶段**。

---

## 三、全局结构：一个契约，两个阶段

### 3.1 二分

```
┌──────────── 生成侧（离线、构建期）────────────┐
│  pypinyin 注音表                            │  ← 生成侧独有
│  en.phone（CMU 发音词典）                   │  ← 生成侧独有
│         │                                   │
│         ▼  产出音素符号                     │
│  ★ tokens.txt 的符号列 ★      ← 两侧共用    │
└─────────┬───────────────────────────────────┘
          │
     keywords.txt                 ← 交接接口（纯文本）
          │
┌─────────▼───────────────────────────────────┐
│  tokens.txt 的 id 列                        │  ← 识别侧独有
│  ContextGraph（前缀树）                     │  ← 识别侧独有
│  encoder / decoder / joiner                │
└─────────────────────────────────────────────┘
```

### 3.2 `tokens.txt` 是契约，不是字典

这是最容易混淆的一点。**没有任何一个词典是两侧共同查询的**：

- 生成侧查的是「文字 → 发音」（`pypinyin` 的注音表、`en.phone`）；
- 识别侧查的是「发音 → 整数」（`tokens.txt`）；
- 两侧共用的，只是 `tokens.txt` 的**第 1 列**——那 263 个符号的写法与拆法。

`tokens.txt` 两列被两侧分别使用：

| 列 | 生成侧 | 识别侧 |
|---|---|---|
| 第 1 列（符号） | ✅ 对齐目标：产出的音素必须在这里面 | ✅ 查表的键 |
| 第 2 列（id） | ❌ 不关心 | ✅ 查表的值 |

所以准确的说法是：联系两个阶段的是**同一套发音符号的约定**，不是同一份查表用的字典。

### 3.3 两个阶段性质不对称

| | 关键词生成 | 关键词识别 |
|---|---|---|
| 时机 | 离线、构建期，一次性的 | 在线、运行时，持续进行 |
| 输入 | 人的语言（汉字 / 英文词） | 声波 |
| 依赖 | 语言学词典 | 声学模型 + 前缀树 |
| 输出 | 文本文件 | 唤醒事件（关键词 / 时间戳 / 概率） |
| 失败后果 | 关键词静默消失，可重来 | 误唤醒或漏唤醒 |
| 换模型时 | 需重新生成 | 代码不变 |

**生成侧允许试错，识别侧不行。** 所以生成侧必须有人工核对，识别侧必须有对拍测试。

### 3.4 为什么这个划分重要

它正好对应语音领域的经典概念 **lexicon（发音词典）**——`词 → 音素序列` 的映射表。三种系统的差别就在这里：

| 系统 | 发音词典在哪 |
|---|---|
| 传统闭集 KWS | 训练时固定，写进模型 |
| **开集 KWS** | **运行时作为参数传入** |
| BPE ASR | 不需要显式词典，BPE 子词隐式承担 |

「开集」二字的含义就是：词典从「训练时固定」变成「运行时参数」。这也是为什么增删唤醒词不需要重训，以及为什么生成阶段可以独立成一个离线工具。

---

## 四、契约层：263 个发音符号

这一节是两侧的共同基础，先讲清楚「符号长什么样」，后面两个阶段才有意义。

### 4.1 三段结构

| id 区间 | 数量 | 内容 |
|---|---|---|
| 0 ~ 2 | 3 | `<blk>`、`<sos/eos>`、`<unk>` |
| 3 ~ 71 | 69 | **CMU 音素**，大写，元音带重音数字：`AA0 AA1 AA2 ... ZH` |
| 72 ~ 262 | 191 | **拼音单元**，小写：声母、无调韵母、带调韵母 |

69 = 24 个辅音（无重音档）+ 15 个元音 × 3 档重音（0/1/2）。

### 4.2 同一个音素，两套 token

```
ZH   id=71    CMU 音素（英文用）
zh   id=182   拼音声母（中文用）
```

`B/CH/D/F/G/HH/...` 与大写的 `ZH` 属于 CMU 体系；`b c ch d f g h ... zh` 属于拼音体系。两套并存、互不替代。英文走 CMU，中文走拼音，混排时由生成侧按「是否 CJK 字符」自动选择。

### 4.3 声调是显式维度

中文音节**没有整音节 token**：词表里不存在 `zhōu`，只有 `zh` + `ōu`。

```
周  →  zh + ōu
轴  →  zh + óu        ← 只差韵母 token
```

这也是「声调在 KWS 里更显眼」的真正原因——不是模型额外强调它，而是音节被拆开后，声调**成了 token 身份的一部分**。

### 4.4 词表印证了 `strict=False` 规则

`ü` 的写法在词表里留下了清晰痕迹：

| 场景 | 例 | 词表中的韵母 |
|---|---|---|
| `j/q/x` 后的 `ü` | 军 `jūn` | `ūn`（按实际发音写成 u 系） |
| `n/l` 后的 `ü` | 女 `nǚ` | `ǚ`（保留真 ü） |
| `lüe` / `nüe` | 略 / 虐 | `üè`（id 219） |
| `jue/que/xue` | 觉 / 却 / 学 | `uè`（id 158） |

词表里同时存在 `uè` 与 `üè`、`ūn` 与 `ǘn/ǚn/ǜn`，但**没有 `ǖn`**——因为一声的 `ǖ` 只出现在 `j/q/x` 之后，而那里已经转写成 `u` 系了。

这直接印证生成侧调的是 `to_finals_tone(x, strict=False)`。**这条规则错了，整个关键词文件会静默失效**，这是必须用工具生成而不能手写的原因。

### 4.5 与 ASR 词表的对比

| 维度 | ASR（5000） | KWS（263） |
|---|---|---|
| 中文粒度 | 整字 `▁周` | 声母 + 带调韵母 |
| 英文粒度 | BPE 子词 `▁the` / `ing` / `ed` | 单音素 `L AY1 T` |
| 词边界标记 | `▁`（子词拼接需要） | 无（音素天然线性，边界由 trie 给） |
| 生成方式 | SentencePiece 从语料统计得到 | 语言学先验定死（CMU + 拼音规则） |
| 声调 | 隐含在整字里，词表层面不可见 | 显式写在韵母 token 上 |
| embedding 维度 | 512 | 320 |

ASR 的 BPE 表是**数据驱动**的：高频片段合成一个 token，不含语言学知识。KWS 的音素表是**先验定死**的：单元在任何语料、任何时候都不变。

---

## 五、生成侧：原文 → 音素符号

### 5.1 定位：G2P，不是 TTS

传统 TTS 的流水线：

```
文本 → 分词 → G2P（注音） → 音素序列 → 声学模型 → 波形
                ↑
        text2token 做的正是这一段
```

区别在终点：TTS 要把音素送进声学模型合成波形，`text2token` 把音素送去**查表**。

> 同一份发音表示，TTS 拿它去合成，KWS 拿它去匹配。

它不生成声音，也不处理韵律、时长、音色，只回答「这些字该怎么念」。所以它是**语音合成的前端组件**，而不是合成的简化版。

### 5.2 实现位置

**在 Python，不在 C++。**

| 文件 | 作用 |
|---|---|
| `sherpa-onnx/python/sherpa_onnx/utils.py: text2token()` | 实际算法 |
| `sherpa-onnx/python/sherpa_onnx/cli.py` | `text2token` 子命令与参数解析 |
| `setup.py:106` | 注册 `sherpa-onnx-cli` 命令 |

依赖：`pypinyin`（中文）、`sentencepiece`（仅 bpe 类型）、`en.phone`（英文）。

支持 6 种建模单元，本模型对应 `phone+ppinyin`：

| 类型 | 单元 |
|---|---|
| `cjkchar` | 汉字单字 |
| `bpe` | SentencePiece 子词 |
| `cjkchar+bpe` | 中文单字 + 英文子词 |
| `fpinyin` | 完整拼音（带调） |
| `ppinyin` | 部分拼音：声母 + 带调韵母 |
| `phone+ppinyin` | **CMU 音素 + 部分拼音** ← 本模型 |

### 5.3 中文路径

```python
py = [x[0] for x in pinyin(txt)]           # 周望军 → [zhōu, wàng, jūn]
initial = to_initials(x, strict=False)      # → zh / w / j
final   = to_finals_tone(x, strict=False)   # → ōu / àng / ūn
```

`pypinyin` 自带约 4 万字的注音表，外加词组表处理多音字——「卡」在「卡索」里读 `kǎ` 而不是 `qiǎ`，靠的就是词组表。

### 5.4 英文路径

按空格切词，每个词查 `en.phone`（CMU 发音词典，126,052 行）展开成音素：

```
LIGHT L AY1 T
UP    AH1 P
```

词不在表里、且不是 CJK 字符时，打印警告并**跳过整行**。

### 5.5 声调从哪来

**查表，不录音、不靠模型。**

- 中文：`pypinyin` 的音表给出带调拼音，再按声母/韵母拆开，声调符号留在韵母上；
- 英文：CMU 词典给出音素，`0/1/2` 是重音等级。

整条链路是确定性的文本处理，与声学无关。同一段文字在世界上任何一台机器上生成的 token 行都相同。

### 5.6 两种失败姿势不一致

| 阶段 | 遇到未知单元 |
|---|---|
| 生成侧 `text2token` | 打印警告，**跳过整行**（关键词静默消失） |
| 识别侧 `EncodeKeywords` | 记录 OOV、返回失败，上层报错退出 |

中间没有强校验，所以生成后**必须核对行数与内容**。

### 5.7 生成侧的硬约束

| 约束 | 原因 |
|---|---|
| 中文必须有 `@原文` | 拼音不可逆（`s uǒ` 可能是「所锁索」） |
| 英文词必须在 `en.phone` 里 | 否则整行被静默跳过 |
| 原文不能含空格 | 用 `_` 代替，`@` 后是单个 token |
| 所有音素必须在 `tokens.txt` 里 | 否则识别侧启动报错 |

生成命令：

```bash
sherpa-onnx-cli text2token  INPUT OUTPUT \
    --tokens      weight/.../tokens.txt \
    --tokens-type phone+ppinyin \
    --lexicon     weight/.../en.phone
```

输入文件每行写原文，`:score` / `#threshold` 可一并写上，`cli.py` 会原样保留到输出行。

---

## 六、接口：`keywords.txt`

### 6.1 格式

每行一条关键词：

```
token token ... [:boost] [#threshold] @原文
```

- 前面的 token 必须出现在 `tokens.txt`；
- `:2.0` boosting score，缺省用全局 `keywords_score`；
- `#0.35` 触发阈值，缺省用全局 `keywords_threshold`；
- `@原文` 触发后展示的文本。

实例（本仓库 `keywords/kws_keywords.txt`）：

```
L AY1 T AH1 P @LIGHT_UP
L AH1 V L IY0 CH AY1 L D @LOVELY_CHILD
w én s ēn t è k ǎ s uǒ @文森特卡索
zh ōu w àng j ūn @周望军
n ǚ ér @女儿
```

### 6.2 交接意义

这个文件是**两个阶段唯一的耦合点**，而且耦合的是「文本约定」而不是「代码契约」：

- 生成侧不 import 识别侧的任何东西；
- 识别侧不依赖 `pypinyin`、`jieba` 或任何发音词典；
- 增删唤醒词 = 改这个文件 + 重启进程。

对 C++ 交付的意义：**关键词生成留在 Python 工具链，C++ 运行时只吃文本文件**。C++ 侧只需要 `EncodeKeywords` 加 `ContextGraph`，不必把发音词典搬进部署包。

> 反过来说，一旦有人绕过 `text2token` 手写 token 行，就可能破坏符号约定：
> `zh ōu w àng j ūn` ✓ / `zhōu wàng jūn` ✗（整音节不在符号表里）。
> 前者能查表，后者要到识别侧才暴露。

---

## 七、识别侧：PCM → 关键词事件

### 7.1 模型契约

数值全部来自 onnx 的 `custom_metadata_map`，运行时由代码读取，本项目未硬编码。

| 项 | chunk-8（默认） | chunk-16 |
|---|---|---|
| `model_type` | `zipformer2` | `zipformer2` |
| `T`（encoder 输入帧数） | 29 | 45 |
| `decode_chunk_len`（时间轴前移） | 16 帧 = 160 ms | 32 帧 = 320 ms |
| 每块 encoder 输出帧 | 4 | 8 |
| 输出帧长 | 40 ms（25 Hz） | 40 ms |
| 卷积重叠上下文 `T − shift` | 13 帧 = 130 ms | 13 帧 = 130 ms |
| `context_size`（decoder） | 2 | 2 |
| `vocab_size`（joiner） | 263 | 263 |
| encoder 状态张量 | 38 | 38 |

38 个状态 = 6 个 stack × 6 个缓存 + 2 个全局（`new_embed_states` / `new_processed_lens`）。

**joiner 的输入是二维 `(N, 320)`**，不是常规 transducer 的 `(N, 1, 320)`；`encoder_out` 与 `decoder_out` 也都是 320 维。写移植代码时以 onnx 输入签名为准。

**输出帧长 40 ms 的成因**：`decode_chunk_len = 16` 帧特征 ⇒ 4 帧输出 ⇒ 160 ms / 4 = 40 ms。C++ 参考实现里 `trailing_silence = num_trailing_blanks * 4 * 0.01` 正是按这个算的。

### 7.2 加载：从符号到 id

`csrc/utils.cc:21 EncodeBase()`，核心只有一次查表：

```cpp
if (symbol_table.Contains(word)) {
  int32_t id = symbol_table[word];
  tmp_ids.push_back(id);
} else {
  switch (word[0]) {
    case ':': score     = std::stof(word.substr(1)); break;
    case '#': threshold = std::stof(word.substr(1)); break;
    case '@': phrase    = word.substr(1);            break;
    default:  /* 报错 */ has_oov = true;              break;
  }
}
```

没有语音学、不了解中文，纯粹是「token 字符串 → id」加字段提取。调用点在 `keyword-spotter-transducer-impl.h:122`（构造时加载文件）与 `:284`（`CreateStream(keywords)` 时加载内联关键词）。本项目的 `KeywordSpotter._load_keywords()` 就是逐行移植的它。

拿到 id 序列后，构建 `ContextGraph`（`csrc/context-graph.cc`）。

### 7.3 为什么不是贪心

263 个音素级 token 里发音相近的很多（`zh`/`z`、前后鼻音、清浊），argmax 一旦在关键词中段选错，后面再准也救不回来。束搜索用「多留几条候选、最后再判」换稳定性。

### 7.4 每帧六步

```
prev = 当前 beam 的全部假设（≤ max_active_paths = 4）
  ① decoder([h.ys[-2:] for h in prev])            → (H, 320)
  ② joiner(当前 encoder 帧复制 H 份, decoder_out)  → (H, 263)
  ③ 逐行 log_softmax，加上各假设累计分
  ④ 在 H × 263 上取全局 top-4
       非 blank/unk → 追加 token、记时间戳与概率、走一步 trie
       blank/unk   → num_trailing_blanks += 1
  ⑤ 按 token 序列合并（log-sum-exp）
  ⑥ 取最佳假设做触发判定
```

与贪心相比的三处反向：

- **decoder 每帧都跑**，而且一次喂 H 条历史。blank 候选的分数也要算，否则无法比较「继续听」与「出下一个音素」。
- **joiner 输入是批量的二维张量**。
- **每个候选都要走一次 trie**，各自推进 `context_state`。

### 7.5 前缀树

节点字段与 C++ `ContextState` 一一对应：`token / token_score / node_score / output_score / level / is_end / phrase / ac_threshold / next / fail / output`。

两条辅助链：

- **fail 链**：失配时回退到「仍可能接上当前 token 的最长后缀节点」，不是回根重来；
- **output 弧**：沿 fail 链指向最近的 end 节点，解决「走到一半才发现前面已经完整说出一个短关键词」的情况。

以 `HE` / `SHELL` / `THEM` 为例，走到 `THE` 时该节点不是任何关键词的结尾，但它的 fail 链指向 `HE` 的结尾节点，output 弧把 `HE` 报出来。

**回根即清空**：走完一步若 `new_state` 是根，该路径的 `ys` 重置为空白假设 `[-1] * (context_size-1) + [0]`，时间戳与概率一并清空。关键词匹配不允许在关键词之外积累上下文。

> `-1` 不是哨兵：ONNX `Gather` 支持负索引，`-1` 取词表最后一行。照抄 C++，**不要 clamp 成 0**。

### 7.6 触发判定

```
matched, node = trie.is_matched(best.context_state)
if matched
   and best.num_trailing_blanks > 1          # 关键词后至少 2 个 blank 输出帧（80 ms）
   and mean(best.ys_probs[:node.level]) >= node.ac_threshold
then 触发
```

三个条件缺一不可：trie 说「已经走完某个关键词」、blank 说「这个词说完了、后面是停顿」、平均概率说「声学证据足够强」。注意用的是**算术平均**而非累计 log 概率。

触发后只重置 beam，**encoder 状态继续累积**——唤醒词之后紧接着说话时，声学上下文不用重来。

### 7.7 静音软重置

连续 1.5 s 没有非 blank 输出时整体软重置（保留未消费的特征尾巴，重置 encoder 状态与 beam），对齐 C++ `DecodeStreams`，避免陈旧前缀一直挂着。

---

## 八、两侧咬合：与 ASR 的关系

识别侧与 ASR 共用同一个 `AudioPreprocess` 实例（80 维 fbank、16 kHz、25 ms 窗 / 10 ms 移），从解码器开始全部分叉：

| 维度 | ASR | KWS |
|---|---|---|
| 词表 | 5000 BPE | 263 音素 |
| 搜索 | 贪心，单假设 | 束搜索，≤ 4 |
| 每帧 decoder | 仅非 blank 才刷新 | 每条假设每帧都跑 |
| joiner 输入 | `(1, 1, 512)` | `(H, 320)` 二维 |
| 跨块状态 | `hyp` + `decoder_out` | beam + trie 状态 + 时间戳 |
| 输出 | 累积文本 | 事件（关键词 / 时间戳 / 平均概率） |
| blank | 从 `tokens.txt` 查 | **硬编码 0**，`<unk>` 也当 blank |
| 收尾 | 补 100 帧特征域 0 | 不补；离线 wav 需补真静音 |

**共用一个前端、不共用一个解码器**：输出契约、不变量、词表都不同，拆开后也正好与上游两个 C++ 解码器一一对应。

双模式状态机（`test_mic_kws_asr.py`）：

```
KWS --命中唤醒词--> ASR --10 s 无新 token 或结束语--> KWS
```

同一时刻只有一个解码器吃特征；切换时双方 `reset()`，`AudioPreprocess` 始终连续。

---

## 九、验证与实测

| 层次 | 脚本 | 覆盖 | 结果 |
|---|---|---|---|
| 单元 | `test_context_graph_mini.py` | 搬上游 `context-graph-test.cc` 的 4 组期望值 + 关键词文件解析 | 全部通过，**不依赖 onnx** |
| 离线 | `test_wav_kws_mini.py` | 9 条自带 wav；`zh_0~2` 为负样本；160 / 1600 两种块长 | 关键词集合完全一致，流式不变性成立 |
| 对拍 | `test_kws_oracle_mini.py` | 与 sherpa-onnx wheel（2.0.1）逐条比对 | 关键词与首 token 时间戳 **0 ms 偏差** |
| 端到端 | `test_mic_kws_asr.py` | 真机唤醒 → 命令 → 切回 | 8 次唤醒 / 8 次切回 / 0 丢块 |

前端口径差异（本项目 `high_freq` 取 Nyquist 8000 Hz，sherpa-onnx C++ 默认 7600 Hz）在这批样本上没有造成任何差异，但样本量只有 9 条，不能外推。

---

## 十、工程结论

### 10.1 对 C++ 交付

- 生成侧**不进部署包**：`pypinyin`、`en.phone`、`text2token` 全部留在构建工具链；
- 识别侧只需要 `EncodeKeywords` + `ContextGraph` + 三个 onnx session；
- 关键词是运行时读文本，因此 C++ 库不必暴露「动态添加上下文词」的接口，除非将来要支持在线换词。

KWS 是否纳入 C++ 交付目前**尚未决策**。它不影响前端与 ASR 库的移植（可先行），但会影响最终接口是否需要一个「唤醒事件」回调。

### 10.2 定词建议（经验性，需实测验证）

- **2~4 个音节**。1 个音节太短，声学证据不足且同类词多，误触发率高。
- **避开日常高频词**。「你好」「好的」在对话里随时出现，等于把误唤醒当成常态。
- **避免与常用词只差一个声调的组合**。词表里声调是独立 token，但连续语流中声调识别本就不稳。
- **中英混排可用**，但英文部分要选常见词；人名、品牌名这类 `en.phone` 覆盖不到的词会直接失败。
- 定词后**必须重新生成 token 文件并跑一轮负样本测试**。

### 10.3 阈值

全局默认 `keywords_score=1.0`、`keywords_threshold=0.25`。误触发多就把该行 `#阈值` 提到 0.35 左右；漏触发就调低，或把 `:boost` 调大。逐词可调，所以能只收紧有问题的词。

---

## 十一、风险与待办

| 项 | 现状 | 处理 |
|---|---|---|
| 正式唤醒词未定 | 仓库里仍是模型自带的 10 条示例词 | 定 2~4 个词后重新生成，跑负样本 |
| 阈值未按真实词调过 | 当前 0.25 是 C++ 默认值 | 定词后用实测数据调 |
| 生成器未固化 | 依赖外部 `sherpa-onnx-cli`，本环境未装 pypinyin | 装 pypinyin 写 `tools/gen_keywords.py`，或装 sherpa-onnx |
| 纯 ASR 回归未跑 | 双模式改动碰过入口主循环 | `INITIAL_MODE=MODE_ASR` 跑一次 |
| 文档状态滞后 | `KWS接入方案.md` 8.1/8.3 仍写「真机待联调」「README 待补」 | 两项实际已完成，需更新记录 |
| 样本量小 | 只有 9 条官方 wav | 定词后用真实录音补测 |

---

## 十二、参考索引

### 上游源码

| 文件 | 内容 | 属于 |
|---|---|---|
| `sherpa-onnx/python/sherpa_onnx/utils.py` | `text2token()`：原文 → token 符号 | 生成侧 |
| `sherpa-onnx/python/sherpa_onnx/cli.py` | `text2token` 子命令 | 生成侧 |
| `sherpa-onnx/csrc/utils.cc:21` | `EncodeBase()`：token → id | 识别侧 |
| `sherpa-onnx/csrc/utils.cc:208` | `EncodeKeywords()`：关键词文件入口 | 识别侧 |
| `sherpa-onnx/csrc/context-graph.{h,cc}` | 前缀树 + fail 链 + output 弧 | 识别侧 |
| `sherpa-onnx/csrc/transducer-keyword-decoder.cc` | 束搜索 + 触发判定 | 识别侧 |
| `sherpa-onnx/csrc/keyword-spotter-transducer-impl.h` | 流式装配、关键词加载、静音重置 | 识别侧 |
| `sherpa-onnx/csrc/context-graph-test.cc` | 4 组期望值 | 识别侧 |

### 本仓库

| 文件 | 内容 | 属于 |
|---|---|---|
| `keywords/kws_keywords.txt` | 运行时关键词文件 | 接口 |
| `src/kws_decode.py` | 解码器逐行移植（`_load_keywords` + `_ContextGraph` + 束搜索） | 识别侧 |
| `test_context_graph_mini.py` | trie 单测 | 识别侧 |
| `test_wav_kws_mini.py` | 离线 wav 验收 | 识别侧 |
| `test_kws_oracle_mini.py` | 与 wheel 对拍 | 识别侧 |
| `doc/KWS接入方案.md` | 设计决策与实施记录 | — |

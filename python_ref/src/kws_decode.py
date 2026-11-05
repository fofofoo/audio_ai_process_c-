# -*- coding: utf-8 -*-
"""流式关键词唤醒：LogFBank 特征 -> 命中关键词（mini 版第四条流水线）。

============================================================
这份代码是干什么的
============================================================
开集关键词唤醒（open vocabulary KWS）本质上是一个「词表被关键词约束住的小型
ASR」：给定若干关键词，解码结果只可能是其中某一个，或者什么都没有。模型本身
就是普通的 streaming zipformer2 transducer，约束完全由解码器完成。

本模块把 sherpa-onnx 的 C++ 参考实现逐行搬到 Python：

    csrc/context-graph.{h,cc}               -> _ContextGraph（前缀树 + fail/output 弧）
    csrc/transducer-keyword-decoder.cc      -> KeywordSpotter._decode_frames（束搜索）
    csrc/keyword-spotter-transducer-impl.h  -> KeywordSpotter（流式装配 + 静音重置）
    csrc/utils.cc: EncodeBase/EncodeKeywords -> KeywordSpotter._load_keywords

和 AsrDecoder 的区别只在解码层：ASR 是 5000 词表上的贪心搜索；KWS 是 263 个
音素级单元上的束搜索，并且每条路径都挂着一个 trie 状态，用来给关键词前缀加分
（boosting score），并在走完整条关键词后判定触发（trigger threshold）。

============================================================
一、关键词文件
============================================================
每行::

    L AY1 T AH1 P @LIGHT_UP
    w én s ēn t è k ǎ s uǒ :2.0 #0.35 @文森特卡索

- 前面的 token 必须来自 tokens.txt；
- `:分数` 是 boosting score（越大越容易触发）；
- `#阈值` 是 trigger threshold（0~1，越小越容易触发）；
- `@原文` 是触发后展示的文本。中文（ppinyin/phone+ppinyin）必须带 `@原文`，
  且原文不能有空格，用 `_` 代替。

============================================================
二、每帧解码（对照 C++）
============================================================
对 encoder 吐出的每个时间帧：

    1. 取当前 beam 的全部假设，**一次 batch** 跑 decoder；
    2. 把当前 encoder 帧复制 beam_size 份，**一次 batch** 跑 joiner；
    3. log_softmax 后加上各假设的累计分，在 beam×vocab 上取 top-k；
    4. 对每个候选 token 走一次 trie.ForwardOneStep：
       - 非 blank/unk：追加 token、记录概率与时间戳、刷新 trie 状态；
         若 trie 回到根（没有匹配任何关键词前缀），把 token 历史清空
         （对齐 C++ 的 “start matching from the start state”）；
       - blank/unk：只累加 trailing blanks；
    5. 相同 token 序列的路径按 log-sum-exp 合并（Hypotheses::Add）；
    6. 取最佳假设，若其 trie 状态命中关键词、末尾 blank 帧数足够、
       且关键词的平均声学概率 >= 该关键词的阈值，则触发并重置 beam。

几个不能想当然的地方：

- **blank_id 硬编码为 0**，`<unk>` 也按 blank 处理（C++ 注释如此）；
- 空白假设的 token 历史是 `[-1, ..., -1, 0]`，不是全 0。`-1` 是 ONNX Gather
  的合法负索引，**不要 clamp**；
- 触发判定用的是「关键词各 token 声学概率的算术平均」，不是累计 log 概率；
- 触发后只重置 beam，encoder 状态继续累积（C++ 行为）；
- 连续 1.5 s 没有非 blank 输出时整体软重置，避免陈旧前缀状态一直挂着。

============================================================
三、跨块必须存活的状态
============================================================
    1. encoder 的全部状态张量（数量由 ONNX 输入签名推导）；
    2. 当前 beam（≤ max_active_paths 条假设，含各自的 trie 状态）；
    3. 未凑满一个 chunk 的特征尾巴；
    4. frame_offset（输出帧计数，只用于时间戳）。

依赖：只用 numpy。三个 onnx 会话由调用方（入口脚本）建好传进来，
本模块既不定位权重文件、也不碰 onnxruntime。
"""

import os
import sys
from collections import deque

import numpy as np

# C++ 参考实现里 blank 固定为 0，这里只用来校验 tokens.txt 的自洽性
_EXPECTED_BLANK_ID = 0


# ============================================================
# 一、关键词前缀树（ContextGraph 的 Python 版）
# ============================================================
class _ContextState:
    """trie 节点。

    字段含义与 C++ 的 ContextState 一一对应：

        token        —— 到达本节点所消耗的 token id；根节点是 -1
        token_score  —— 这条边的 boosting score
        node_score   —— 从根走到本节点的累计 score
        output_score —— 若本节点（或 fail 链上的某个 end 节点）构成关键词，
                        累加得到的输出奖励；普通节点为 0
        level        —— 本节点在所属关键词里的序号（1-based）
        ac_threshold —— 关键词的触发阈值，只有 end 节点有效
        is_end       —— 是否是某个关键词的结尾
        phrase       —— 关键词原文，只有 end 节点有效
        next         —— {token_id: _ContextState}
        fail         —— 失配后跳转的节点（Aho-Corasick 的 fail 链）
        output       —— fail 链上最近的 end 节点
    """

    __slots__ = ("token", "token_score", "node_score", "output_score",
                 "level", "ac_threshold", "is_end", "phrase",
                 "next", "fail", "output")

    def __init__(self, token, token_score=0.0, node_score=0.0,
                 output_score=0.0, level=0, ac_threshold=0.0,
                 is_end=False, phrase=""):
        self.token = int(token)
        self.token_score = float(token_score)
        self.node_score = float(node_score)
        self.output_score = float(output_score)
        self.level = int(level)
        self.ac_threshold = float(ac_threshold)
        self.is_end = bool(is_end)
        self.phrase = phrase
        self.next = {}
        self.fail = None
        self.output = None


class _ContextGraph:
    """关键词前缀树 + fail/output 弧。

    逐行对照 `sherpa-onnx/csrc/context-graph.cc`。构建期只做一次，
    运行期每帧对每条假设调一次 forward_one_step。
    """

    def __init__(self, token_ids, context_score, ac_threshold,
                 scores=None, phrases=None, ac_thresholds=None):
        """
        参数:
            token_ids      —— 每条关键词的 token id 列表
            context_score  —— 默认 boosting score
            ac_threshold   —— 默认触发阈值
            scores         —— 每条关键词单独的 boosting score（可为空列表）
            phrases        —— 每条关键词的原文
            ac_thresholds  —— 每条关键词单独的触发阈值
        """
        self.context_score = float(context_score)
        self.ac_threshold = float(ac_threshold)

        self.root = _ContextState(-1)
        self.root.fail = self.root

        self._build(token_ids, scores or [], phrases or [],
                    ac_thresholds or [])
        self._fill_fail_output()

    # --------------------------------------------------------
    def _build(self, token_ids, scores, phrases, ac_thresholds):
        """把每条关键词插进 trie，并合并公共前缀。

        与 C++ Build 的差异只有一处：Python 的 dict 没有“count”，
        这里用 `in` 判断；其余分支（新建 / 复用 / 更新 is_end、output_score、
        phrase、ac_threshold）完全照抄。
        """
        for i, ids in enumerate(token_ids):
            node = self.root
            score = scores[i] if scores else 0.0
            if score == 0.0:
                score = self.context_score
            ac_thr = ac_thresholds[i] if ac_thresholds else 0.0
            if ac_thr == 0.0:
                ac_thr = self.ac_threshold
            phrase = phrases[i] if phrases else ""

            for j, token in enumerate(ids):
                is_last = (j == len(ids) - 1)
                if token not in node.next:
                    node.next[token] = _ContextState(
                        token,
                        token_score=score,
                        node_score=node.node_score + score,
                        output_score=(node.node_score + score) if is_last else 0.0,
                        level=j + 1,
                        ac_threshold=ac_thr if is_last else 0.0,
                        is_end=is_last,
                        phrase=phrase if is_last else "")
                else:
                    child = node.next[token]
                    # 多条关键词共享前缀时，边分取较大值
                    token_score = max(score, child.token_score)
                    child.token_score = token_score
                    child.node_score = node.node_score + token_score
                    is_end = is_last or child.is_end
                    child.output_score = child.node_score if is_end else 0.0
                    child.is_end = is_end
                    if is_last:
                        child.phrase = phrase
                        child.ac_threshold = ac_thr
                node = node.next[token]

            # 空关键词（ids 为空）在 C++ 里也会被收下，但不会进入任何节点；
            # 这里直接跳过，效果相同。

    # --------------------------------------------------------
    def _fill_fail_output(self):
        """BFS 填 fail 链和 output 弧（Aho-Corasick 建树的标准做法）。

        output 指向 fail 链上最近的 end 节点；output_score 把该 end 节点的
        累计分加进来，这样即使当前节点不是关键词结尾，只要回退链上命中，
        也能拿到触发奖励。
        """
        q = deque()
        for child in self.root.next.values():
            child.fail = self.root
            q.append(child)

        while q:
            cur = q.popleft()
            for token, child in cur.next.items():
                fail = cur.fail
                if token in fail.next:
                    fail = fail.next[token]
                else:
                    fail = fail.fail
                    while token not in fail.next:
                        fail = fail.fail
                        if fail.token == -1:      # 回到根
                            break
                    if token in fail.next:
                        fail = fail.next[token]
                child.fail = fail

                # fail 链上最近的 end 节点
                output = fail
                while not output.is_end:
                    output = output.fail
                    if output.token == -1:
                        output = None
                        break
                child.output = output
                if output is not None:
                    child.output_score += output.output_score

                q.append(child)

    # --------------------------------------------------------
    def forward_one_step(self, state, token, strict_mode=True):
        """沿 trie 走一步，返回 (context_score, new_state, matched_state)。

        对应 C++ ContextGraph::ForwardOneStep(state, token, strict_mode)。
        `strict_mode=False` 只在上游的单测里用到：命中关键词后直接回到根，
        并把该关键词的累计分一次性加进来。KWS 解码器用的是默认的 True。

        matched_state 只在本步走到了某个 end 节点（或 output 弧）时非 None，
        解码循环主要用 is_matched() 在最佳假设上再查一次。
        """
        if token in state.next:
            node = state.next[token]
            score = node.token_score
        else:
            # 失配：沿 fail 链找能接上 token 的节点
            node = state.fail
            while token not in node.next:
                node = node.fail
                if node.token == -1:              # 回到根
                    break
            if token in node.next:
                node = node.next[token]
            # 与 C++ 一致：用新节点的累计分减去旧节点的累计分
            score = node.node_score - state.node_score

        if node.is_end:
            matched = node
        elif node.output is not None:
            matched = node.output
        else:
            matched = None

        # 非 strict 模式：命中即回到根，分数按 end 节点的累计分结算
        if not strict_mode and node.output_score != 0:
            if node.is_end:
                output_score = node.node_score
            elif node.output is not None:
                output_score = node.output.node_score
            else:
                output_score = node.node_score
            return (score + output_score - node.node_score,
                    self.root, matched)

        return score + node.output_score, node, matched

    # --------------------------------------------------------
    def finalize(self, state):
        """收尾：把当前节点累计分退掉，回到根（C++ ContextGraph::Finalize）。

        主要给上游 context-graph-test 的对拍用；流式解码不使用。
        """
        return -state.node_score, self.root

    # --------------------------------------------------------
    def is_matched(self, state):
        """判断当前状态是否已经走完一个关键词。返回 (bool, matched_state)。"""
        if state.is_end:
            return True, state
        if state.output is not None:
            return True, state.output
        return False, None


# ============================================================
# 二、束搜索的假设
# ============================================================
class _Hypothesis:
    """一条搜索路径。

    ys / timestamps / ys_probs 三个列表严格同步：ys[i] 是第 i 个非 blank
    token，timestamps[i] 是它的输出帧号，ys_probs[i] 是它在那一帧的声学
    概率（不是 log 概率）。ys 前 context_size 个位置是占位 blank。
    """

    __slots__ = ("ys", "timestamps", "ys_probs", "log_prob",
                 "context_state", "num_trailing_blanks")

    def __init__(self, ys, log_prob=0.0, context_state=None):
        self.ys = list(ys)
        self.timestamps = []
        self.ys_probs = []
        self.log_prob = float(log_prob)
        self.context_state = context_state
        self.num_trailing_blanks = 0

    def clone(self):
        h = _Hypothesis(self.ys, self.log_prob, self.context_state)
        h.timestamps = list(self.timestamps)
        h.ys_probs = list(self.ys_probs)
        h.num_trailing_blanks = self.num_trailing_blanks
        return h


def _log_add(a, b):
    """log(exp(a) + exp(b))，用于合并相同 token 序列的两条路径。"""
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    hi = max(a, b)
    lo = min(a, b)
    return hi + float(np.log1p(np.exp(lo - hi)))


def _log_softmax(x):
    """按行做 log_softmax。x: (H, V) float32。"""
    m = x.max(axis=1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


# ============================================================
# 三、关键词唤醒器
# ============================================================
class KeywordSpotter:
    """三段 onnx 会话 + 关键词 trie + 流式束搜索。

    用法与 AsrDecoder 一致::

        kws = KeywordSpotter(encoder, decoder, joiner, tokens_path, keywords_path)
        hits = kws.accept_features(feats)     # 可能返回 0~N 个 KeywordHit
        kws.reset()                           # 换模式 / 重新开始监听时调用

    `accept_features` 直接返回本次新触发的关键词列表；没有命中就是空列表。
    """

    _DTYPES = {"tensor(float)": np.float32,
               "tensor(int64)": np.int64,
               "tensor(int32)": np.int32}

    def __init__(self, encoder, decoder, joiner, tokens_path, keywords_path,
                 max_active_paths=4, num_trailing_blanks=1,
                 keywords_score=1.0, keywords_threshold=0.25,
                 reset_after_silence=1.5, sample_rate=16000, debug=False):
        """
        参数:
            encoder / decoder / joiner —— 已建好的 onnxruntime.InferenceSession
            tokens_path    —— tokens.txt 路径（263 个音素级单元）
            keywords_path  —— 关键词文件路径
            max_active_paths   —— 束宽，C++ 默认 4
            num_trailing_blanks—— 触发前至少要有多少个 blank 输出帧，默认 1
            keywords_score     —— 默认 boosting score
            keywords_threshold —— 默认触发阈值
            reset_after_silence—— 连续静音多少秒后软重置，C++ 默认 1.5
            sample_rate        —— 仅用于记录，模型固定 16 kHz
            debug              —— True 时记录「trie 已走到末端但被
                                  ac_threshold 挡住」的候选，见 _record_blocked
        """
        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner
        self.sample_rate = int(sample_rate)
        self.providers = encoder.get_providers()

        self.tokens_path = os.path.abspath(tokens_path)
        self.keywords_path = os.path.abspath(keywords_path)
        if not os.path.isfile(self.tokens_path):
            raise FileNotFoundError("找不到词表：%s" % self.tokens_path)
        if not os.path.isfile(self.keywords_path):
            raise FileNotFoundError("找不到关键词文件：%s" % self.keywords_path)

        # ---- encoder 元数据：T / decode_chunk_len / n_mels ----
        meta = self.encoder.get_modelmeta().custom_metadata_map
        if meta.get("model_type", "") != "zipformer2":
            raise RuntimeError("encoder 的 model_type=%r，本模块只支持 zipformer2"
                               % meta.get("model_type", ""))
        self.input_frame_num = int(meta["T"])
        self.T_shift = int(meta["decode_chunk_len"])

        x_shape = self.encoder.get_inputs()[0].shape
        last = x_shape[-1]
        self.n_mels = int(last) if not isinstance(last, str) else 80

        # ---- encoder 状态张量：数量与形状全部从输入签名推导 ----
        self._state_proto = []
        for n in self.encoder.get_inputs()[1:]:
            if n.type not in self._DTYPES:
                raise RuntimeError("状态张量 %s 的类型 %s 未支持" % (n.name, n.type))
            shape = [1 if isinstance(s, str) else int(s) for s in n.shape]
            self._state_proto.append((shape, self._DTYPES[n.type]))

        # ---- decoder / joiner 元数据 ----
        dmeta = self.decoder.get_modelmeta().custom_metadata_map
        self.context_size = int(dmeta["context_size"])
        self.vocab_size = int(dmeta["vocab_size"])

        # ---- 输入输出名字 ----
        self._enc_in = [n.name for n in self.encoder.get_inputs()]
        self._enc_out = [n.name for n in self.encoder.get_outputs()]
        self._dec_in = self.decoder.get_inputs()[0]
        self._dec_out = self.decoder.get_outputs()[0].name
        self._joi_in = [n.name for n in self.joiner.get_inputs()]
        self._joi_out = self.joiner.get_outputs()[0].name

        # ---- 词表 ----
        self.id2token, self.token2id = self._load_tokens(self.tokens_path)
        self.blank_id = self._find_token_id("<blk>", _EXPECTED_BLANK_ID)
        if self.blank_id != _EXPECTED_BLANK_ID:
            raise RuntimeError("KWS 解码器把 blank 硬编码为 0，但 tokens.txt 里 "
                               "<blk> 的 id 是 %d" % self.blank_id)
        self.unk_id = self.token2id.get("<unk>", -1)

        # ---- 关键词 ----
        (self._keyword_ids, self._keyword_scores,
         self._keyword_phrases, self._keyword_thresholds) = \
            self._load_keywords(self.keywords_path, self.token2id)

        self._graph = _ContextGraph(
            self._keyword_ids, keywords_score, keywords_threshold,
            self._keyword_scores, self._keyword_phrases,
            self._keyword_thresholds)

        self.max_active_paths = max(1, int(max_active_paths))
        self.num_trailing_blanks = int(num_trailing_blanks)
        self.reset_after_silence = float(reset_after_silence)

        # ---- 调阈值用的观测：只记「差一点命中」的候选 ----
        # 命中时 avg_prob 总能从 KeywordHit 拿到，真正缺的是被阈值挡住的分数，
        # 那才是决定 #threshold 该写多少的依据。默认关，避免正常使用刷屏。
        self.debug = bool(debug)
        self.debug_events = []      # [{keyword, avg_prob, threshold, level, frame}]
        self._debug_last_key = None  # 去重：同词同分数连续多帧只记一次

        # 输出帧时长：subsampling=4 时 = T_shift/4 * 10ms。第一次 encoder
        # 输出后会按实际帧数校正（一般是 40 ms，不是 20 ms）。
        self._frame_shift_s = self.T_shift * 0.01 / 4.0

        self.reset()

    # ========================================================
    # 构造：词表与关键词
    # ========================================================
    @staticmethod
    def _load_tokens(path):
        """读 tokens.txt，返回 ({id: token}, {token: id})。"""
        id2token = {}
        token2id = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                token, sep, idx = line.rpartition(" ")
                if not sep:
                    continue
                i = int(idx)
                id2token[i] = token
                token2id[token] = i
        return id2token, token2id

    def _find_token_id(self, token, default=-1):
        i = self.token2id.get(token, None)
        if i is None:
            print("警告：tokens.txt 里没找到 %s，退回 id=%d" % (token, default))
            return default
        return i

    @staticmethod
    def _load_keywords(path, token2id):
        """解析关键词文件，返回 (ids, scores, phrases, thresholds)。

        对齐 C++ `utils.cc: EncodeBase` 的字段规则：
            - 能查到 id 的 word 当作 token；
            - `:x` boosting score、`#x` 触发阈值、`@x` 原文；
            - 其余一律报错（避免关键词悄悄少了几个 token）。

        做成 staticmethod 是为了能脱离 onnx session 单测：只需要 token2id。
        """
        ids_all, scores_all, phrases_all, thresholds_all = [], [], [], []
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                # 整行以 # 开头只可能是注释：合法关键词行必须以 token 开头，
                # 阈值 #0.35 只能出现在 token 之后
                if not line or line.startswith("#"):
                    continue
                token_ids = []
                score = 0.0
                threshold = 0.0
                phrase = ""
                for word in line.split():
                    if word in token2id:
                        token_ids.append(token2id[word])
                        continue
                    head = word[0]
                    if head == ":":
                        score = float(word[1:])
                    elif head == "#":
                        threshold = float(word[1:])
                    elif head == "@":
                        phrase = word[1:]
                    else:
                        raise ValueError(
                            "关键词文件第 %d 行：token %r 不在 tokens.txt 里"
                            % (lineno, word))
                if not token_ids:
                    # 纯注释/空行；C++ 会收下一条空关键词，效果相同，这里直接跳过
                    continue
                ids_all.append(token_ids)
                scores_all.append(score)
                phrases_all.append(phrase)
                thresholds_all.append(threshold)

        if not ids_all:
            raise ValueError("关键词文件里没有有效关键词：%s" % path)
        return ids_all, scores_all, phrases_all, thresholds_all

    # ========================================================
    # 状态
    # ========================================================
    def _blank_ys(self):
        """空白假设的 token 历史：[-1]*(context-1) + [blank]。

        -1 不是笔误：C++ 就是这么填的，ONNX Gather 支持负索引。
        """
        ys = [-1] * self.context_size
        ys[-1] = self.blank_id
        return ys

    def _blank_hypotheses(self):
        h = _Hypothesis(self._blank_ys(), 0.0, self._graph.root)
        return {tuple(h.ys): h}

    def _reset_stream(self, keep_feat_buf=False):
        """只重置流状态，不碰统计计数。"""
        self.states = [np.zeros(shape, dtype=dt)
                       for shape, dt in self._state_proto]
        self._hyps = self._blank_hypotheses()
        self._frame_offset = 0
        self._best_trailing_blanks = 0
        self._last_best_level = 0
        self._debug_last_key = None
        self._frame_shift_s = self.T_shift * 0.01 / 4.0
        if not keep_feat_buf:
            self._feat_buf = np.zeros((0, self.n_mels), dtype=np.float32)

    def reset(self):
        """清空全部跨块状态，回到「刚开始监听」。"""
        self._reset_stream(keep_feat_buf=False)
        self.n_chunks = 0        # 喂过几次 encoder
        self.n_enc_frames = 0    # encoder 一共吐了多少输出帧
        self.n_hits = 0          # 累计触发次数

    # --------------------------------------------------------
    # 属性 / 信息
    # --------------------------------------------------------
    @property
    def best_trie_level(self):
        """最近一个 chunk 里最佳假设在 trie 上走到的深度（0 = 在根）。

        未命中时这个值能区分「一个字都没对上」和「只差最后一个音」：
        评估脚本取整条音频的最大值，就知道关键词是在第几个 token 断掉的。
        """
        return self._last_best_level

    @property
    def keywords(self):
        """关键词原文列表（没有 @原文 时用 token 拼接）。"""
        out = []
        for ids, phrase in zip(self._keyword_ids, self._keyword_phrases):
            out.append(phrase if phrase else
                       " ".join(self.id2token.get(i, "?") for i in ids))
        return out

    def info(self):
        """把关键尺寸打成一包，方便入口脚本打印。"""
        return {"T": self.input_frame_num,
                "decode_chunk_len": self.T_shift,
                "context_size": self.context_size,
                "vocab_size": self.vocab_size,
                "n_states": len(self.states),
                "blank_id": self.blank_id,
                "unk_id": self.unk_id,
                "n_mels": self.n_mels,
                "n_keywords": len(self._keyword_ids),
                "max_active_paths": self.max_active_paths,
                "frame_shift_s": self._frame_shift_s,
                "providers": self.providers}

    # ========================================================
    # 三个 session：与 AsrDecoder 同形的低层接口
    # ========================================================
    def run_encoder(self, x):
        """喂 (1, T, n_mels)，返回 encoder 输出；状态就地累积。"""
        feed = {self._enc_in[0]: x}
        for i, name in enumerate(self._enc_in[1:]):
            feed[name] = self.states[i]
        out = self.encoder.run(self._enc_out, feed)
        self.states = out[1:]
        return out[0]

    def run_decoder(self, y):
        """喂 (H, context_size) 的 token id，返回 (H, decoder_dim)。"""
        dtype = (np.int64 if self._dec_in.type == "tensor(int64)" else np.int32)
        return self.decoder.run([self._dec_out],
                                {self._dec_in.name: np.asarray(y, dtype=dtype)})[0]

    def run_joiner(self, encoder_out, decoder_out):
        """喂 (H, enc_dim) + (H, dec_dim)，返回 (H, vocab) 的 logit。

        注意：这个 KWS 模型的 joiner 要的是**二维** encoder_out（N, 320），
        和常规 transducer joiner 的 (N, 1, 320) 不同 —— 以输入签名为准。
        """
        return self.joiner.run([self._joi_out],
                               {self._joi_in[0]: encoder_out,
                                self._joi_in[1]: decoder_out})[0]

    # ========================================================
    # 对外主入口
    # ========================================================
    def accept_features(self, feats):
        """喂一块特征，把能解码的都解掉，返回本次新触发的关键词列表。

        参数:
            feats —— float32 特征矩阵 (n, n_mels)，可能是 0 帧
        返回:
            list[KeywordHit]；没有新触发就是空列表
        """
        hits = []

        if feats.shape[0]:
            if self._feat_buf.shape[0]:
                self._feat_buf = np.concatenate((self._feat_buf, feats))
            else:
                self._feat_buf = feats

        # 与 C++ DecodeStreams 一致：解码前先看上一块结束时最佳假设的
        # trailing blanks，静音足够长就软重置（保留尚未消费的特征尾巴）。
        if (self._best_trailing_blanks * self._frame_shift_s
                > self.reset_after_silence):
            self._reset_stream(keep_feat_buf=True)

        # 严格大于：写成 >= 会在批量条件恰好 16k + T 时多喂一次
        while self._feat_buf.shape[0] > self.input_frame_num:
            x = np.ascontiguousarray(
                self._feat_buf[:self.input_frame_num][None])
            self._feat_buf = np.ascontiguousarray(
                self._feat_buf[self.T_shift:])
            self.n_chunks += 1

            enc_feats = self.run_encoder(x)
            nf = int(enc_feats.shape[1])
            self.n_enc_frames += nf
            if nf == 0:
                continue

            # 每个输出帧代表多少秒：本模型是 T_shift/nf * 10ms（通常 40ms）
            self._frame_shift_s = self.T_shift * 0.01 / nf
            self._decode_frames(enc_feats, hits)
            self._frame_offset += nf

        if hits:
            self.n_hits += len(hits)
        return hits

    # ========================================================
    # 束搜索（对照 TransducerKeywordDecoder::Decode）
    # ========================================================
    def _decode_frames(self, enc_feats, hits):
        """对 encoder 输出的一整块逐帧解码，命中结果追加到 hits。"""
        nf = int(enc_feats.shape[1])
        V = self.vocab_size
        K = self.max_active_paths

        for t in range(nf):
            # 1) 当前 beam（同序列已合并）
            prev = list(self._hyps.values())
            H = len(prev)

            # 2) decoder：一次喂完所有假设最近的 context_size 个 token
            y = np.array([h.ys[-self.context_size:] for h in prev],
                         dtype=np.int64)
            dec_out = self.run_decoder(y)

            # 3) joiner：把当前 encoder 帧复制 H 份。这个模型要二维输入，
            #    所以直接取 (enc_dim,) 再复制成 (H, enc_dim)
            cur = enc_feats[0, t]                       # (enc_dim,)
            enc_rep = np.repeat(cur[None], H, axis=0)   # (H, enc_dim)
            logits = np.asarray(self.run_joiner(enc_rep, dec_out),
                                dtype=np.float32).reshape(H, V)

            # 4) log_softmax + 各假设已有累计分
            logprobs = _log_softmax(logits)
            base = np.array([h.log_prob for h in prev], dtype=np.float32)
            scores = logprobs + base[:, None]

            # 5) 在 H*V 个候选里取 top-k（H、V 都很小，直接 argsort 足够）
            flat = scores.ravel()
            k = min(K, flat.size)
            topk = np.argsort(-flat, kind="stable")[:k]

            new_hyps = {}
            for flat_idx in topk:
                hyp_idx, token = divmod(int(flat_idx), V)
                src = prev[hyp_idx]
                new_hyp = src.clone()
                context_score = 0.0

                if token != self.blank_id and token != self.unk_id:
                    # 非 blank：追加 token 并走 trie
                    new_hyp.ys.append(token)
                    new_hyp.timestamps.append(self._frame_offset + t)
                    new_hyp.ys_probs.append(float(np.exp(
                        logprobs[hyp_idx, token])))
                    new_hyp.num_trailing_blanks = 0

                    context_score, new_state, _ = \
                        self._graph.forward_one_step(src.context_state, token)
                    new_hyp.context_state = new_state

                    # 回到根：当前 token 不属于任何关键词前缀，清空历史。
                    # C++ 连刚 push 的 token 一起丢掉，这里照做。
                    if new_state.token == -1:
                        new_hyp.ys = self._blank_ys()
                        new_hyp.timestamps = []
                        new_hyp.ys_probs = []
                else:
                    # blank / unk：不刷新 trie，只累计静音帧
                    new_hyp.num_trailing_blanks += 1

                new_hyp.log_prob = float(flat[flat_idx]) + context_score
                self._merge(new_hyps, new_hyp)

            best = max(new_hyps.values(), key=lambda h: h.log_prob)
            triggered = False

            # 6) 触发判定
            matched, matched_state = self._graph.is_matched(best.context_state)
            if matched:
                level = matched_state.level
                # 与 C++ 一致：取 ys_probs 的前 level 个求平均。
                # 单一关键词场景下 ys_probs 恰好只有 level 个元素。
                n_prob = min(level, len(best.ys_probs))
                avg_prob = (sum(best.ys_probs[:n_prob]) / n_prob
                            if n_prob else 0.0)
                if (best.num_trailing_blanks > self.num_trailing_blanks
                        and avg_prob >= matched_state.ac_threshold):
                    token_ids = best.ys[-level:]
                    frame_ids = best.timestamps[-level:]
                    tokens = [self.id2token.get(int(i), "?")
                              for i in token_ids]
                    times = [int(f) * self._frame_shift_s for f in frame_ids]
                    hits.append(KeywordHit(
                        keyword=(matched_state.phrase or " ".join(tokens)),
                        tokens=tokens,
                        timestamps=times,
                        # C++ 的 start_time 取的是「取结果那一刻」的帧时间；
                        # 这里改成关键词第一个 token 的时间，对使用者更有意义
                        start_time=times[0] if times else
                        self._frame_offset * self._frame_shift_s,
                        avg_prob=avg_prob))
                    triggered = True
                    self._debug_last_key = None
                    # 触发后重置 beam，但 encoder 状态继续累积
                    new_hyps = self._blank_hypotheses()
                elif self.debug:
                    # 没触发：要么 blank 还不够（正常等待），要么 avg_prob 没过
                    # 阈值（调参依据）。后者由 _record_blocked 自己判断。
                    self._record_blocked(matched_state, avg_prob, level)

            self._hyps = new_hyps
            self._last_best_level = best.context_state.level
            if triggered:
                self._best_trailing_blanks = 0
            else:
                self._best_trailing_blanks = best.num_trailing_blanks

    def _record_blocked(self, state, avg_prob, level):
        """记录「trie 已走到末端，但平均声学概率没过阈值」的候选。

        调 #threshold 的唯一依据。命中时 avg_prob 能从 KeywordHit 拿到，真正
        缺的是「差一点」的那批分数 —— 没有它们就只能靠猜阈值该写多少。

        同一假设会连续多帧重复到这里（trie 不前进，ys_probs 与 blank 计数也
        不变），按 (原文, 分数) 去重，一句话只记一次。

        avg_prob 够高却没触发，说明只是尾随 blank 还没攒够，属正常等待，
        这类不记录，否则日志里全是噪声。
        """
        if avg_prob >= state.ac_threshold:
            return

        phrase = state.phrase or "<未命名关键词>"
        key = (phrase, round(avg_prob, 3))
        if key == self._debug_last_key:
            return
        self._debug_last_key = key

        ev = {"keyword": phrase,
              "avg_prob": avg_prob,
              "threshold": state.ac_threshold,
              "level": level,
              "frame": int(self._frame_offset)}
        # 长时间挂着监听时别让列表无限长；stderr 照样打，只是不再累积
        if len(self.debug_events) < 2000:
            self.debug_events.append(ev)
        sys.stderr.write(
            "\n[KWS debug] 被阈值挡住 %s  avg_prob=%.3f < threshold=%.3f "
            "(level=%d)\n" % (phrase, avg_prob, state.ac_threshold, level))
        sys.stderr.flush()

    @staticmethod
    def _merge(hyps, hyp):
        """同 token 序列的路径按 log-sum-exp 合并（Hypotheses::Add）。"""
        key = tuple(hyp.ys)
        old = hyps.get(key)
        if old is None:
            hyps[key] = hyp
        else:
            old.log_prob = _log_add(old.log_prob, hyp.log_prob)


# ============================================================
# 四、触发结果
# ============================================================
class KeywordHit:
    """一次关键词触发。"""

    __slots__ = ("keyword", "tokens", "timestamps", "start_time", "avg_prob")

    def __init__(self, keyword, tokens, timestamps, start_time, avg_prob):
        self.keyword = keyword
        self.tokens = list(tokens)
        self.timestamps = list(timestamps)   # 秒
        self.start_time = float(start_time)
        self.avg_prob = float(avg_prob)

    def __repr__(self):
        return ("KeywordHit(keyword=%r, t=%.3f, prob=%.3f)"
                % (self.keyword, self.start_time, self.avg_prob))

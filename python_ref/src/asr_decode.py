# -*- coding: utf-8 -*-
"""流式 ASR 解码：LogFBank 特征 -> 文字（mini 版第二条流水线）。

============================================================
这份代码是干什么的
============================================================
拿 onnxruntime 跑三段模型（encoder / decoder / joiner），对每一帧特征做
贪心搜索（greedy search），把得到的 token 序列拼成中文文本。

模型是 streaming zipformer2 transducer，三段的分工：

    encoder  —— 吃特征，吐出"声学编码"；它带着一大堆**状态张量**，跨块累积，这就是"流式"能记住前文的全部机制,主要负责猜字。
    decoder  —— 吃最近若干个 token，吐出"语言编码"（也叫 predictor），2TOKEN轻量化CNN(非长记忆的LSTM)，辅助去重、断句。
    joiner   —— 把 encoder 的一帧和 decoder 的输出合起来，输出整个词表的打分。

=========================================================================================
一、流式 Encoder 编码：部分重叠滑动输入、encoder_embed 与 U-Net 的两段式时间压缩
=========================================================================================
1. 滑动窗口机制：每次喂给 encoder 的是 T = 61 帧，但时间轴只前进 decode_chunk_len = 48 帧（480ms）。
   多出的 13 帧是卷积模块必需的重叠上下文，用于将旧的 13 帧与新的 48 帧拼接后送入模型，避免边界补零失真。

2. encoder_embed：输入 (b, 61, 80) 先当作 (N, 1, T, 80) 的图像，经 3 层 2D 卷积
   1→8→32→128（3x3；只有第二层 conv.4 的 stride=2 作用于时间维，第三层 conv.7 只压频率），
   把频率维 80 压到 19；随后过 ConvNeXt 块（7x7 depthwise + 两个 1x1），
   最后由 out 投影把通道 128 升到 192。输出 (b, 27, 192)。
   **时间维在开头只压缩约 2 倍，不是 4 倍；通道也不是一步到 512。**

3. U-Net 内部：6 个 stack 的维度是 192/256/512/768/512/256，各自用 reshape 合并/拆分
   相邻帧做时间下采样与上采样（先压缩再恢复），不同 stack 帧率不同（KV cache 的 T 为
   256/128/64/32）。512 是第 3 个 stack 的宽度，也是 encoder_out 的最终维度。
   整体时间压缩 4 倍 = encoder_embed 2 倍 + 内部再 2 倍：48 帧输入对应 12 帧输出。
   这两段之后才进入 Transformer，由自注意力捕获长距离依赖，得到融合全局上下文的声学表征。

============================================================================================
二、贪心搜索：参考decoder基于上一个字的接龙预测值和encoder实际听到的结果 ，joiner 综合决策快速出字
============================================================================================
对 encoder 吐出的每一帧，跟当前 decoder 输出一起过 joiner，取 argmax：
    · 如果joiner结果不是 blank，用最近 context_size 个 token 重算一次 decoder 输出，
    · 后续将decoder根据这个选出的字继续预测下一个最可能的字，并不断向后词语接龙，这也是搜索这个概念的来源；
    · 如果joiner结果是 blank —— **什么都不做**，沿用上一次的 decoder 输出。
    · 贪心的由来，argmax是选字最快的，但模型将直接抛弃其他高概率的字，没有纠错可能了。
    · 相比之下，没采用的束搜索就是每一步都同时保留前 K 条最可能的“词语接龙”路径，通过将新词得分累加到历史总分中，历史中错误的接龙被正确接龙替代，多次更新后各路径历史语句完全固化并被删掉，不再进入打分模型，从而在保证速度的情况避免贪心搜索的“一错到底”。
    · joiner输出 blank 不刷新 decoder，绝大多数帧都是 blank，只有真出新 token 才需要跑一次decoder 前向，加速模型推理过程。

============================================================
三、跨块必须存活的状态
============================================================
    1. encoder 的全部状态张量（实测 116 个）—— 除 reset() 外绝不清零；
    2. hyp（已识别出的 token 序列）；
    3. 最近一次的 decoder 输出。

============================================================
四、收尾：尾部补 100 帧特征域 0
============================================================
由前端（AudioPreprocess.input_finished）在特征尾部补 100 帧 0 完成，本模块
不需要额外处理。补的 0 是 log 域对应能量 1 的 0，不是静音 —— 目的是让最后
不满一个 chunk 的尾段也能凑齐、被解码到，否则尾字会丢。

依赖：只用 numpy —— 三个 onnx 会话由调用方（入口脚本）建好传进来，
本模块既不定位权重文件、也不碰 onnxruntime。
"""

import os
import re
import numpy as np


# 文本还原用的字符区间：
#   _CJK      —— 统一表意文字 + 扩展 A
#   _FULLWIDTH —— CJK 标点 + 全角形
_CJK = r"一-鿿㐀-䶿"
_FULLWIDTH = r"　-〿＀-￯"

# ============================================================
# 二、解码器
# ============================================================
class AsrDecoder:
    """三段 onnx 会话 + 流式贪心搜索。
    三个 session 由调用方建好传进来（见入口 test_mic_asr_mini.py）——
    本类只管「怎么用模型」，不管「模型从哪来、跑在哪块硬件上」。
    """

    # onnx 张量类型 -> numpy dtype，用来给状态张量选初值类型
    _DTYPES = {"tensor(float)": np.float32,
               "tensor(int64)": np.int64,
               "tensor(int32)": np.int32}

    def __init__(self, encoder, decoder, joiner, tokens_path, sample_rate=16000):
        """
        参数:
            encoder / decoder / joiner —— 三个已建好的 onnxruntime.InferenceSession，
            tokens_path —— tokens.txt 的路径
            sample_rate —— 采样率，目前只用于记录（模型固定 16kHz）
        """

        # 三个模型
        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner
        # 字典词表，最后idx转文字时使用
        self.tokens_path = os.path.abspath(tokens_path)
        if not os.path.isfile(self.tokens_path):
            raise FileNotFoundError("找不到词表：%s" % self.tokens_path)
        # 音频采样率
        self.sample_rate = int(sample_rate)

        # CUDA / CPU
        self.providers = encoder.get_providers()

        # 直接从 encoder的onnx模型中metadata的参数读取输入张量大小，不硬编码 ----
        meta = self.encoder.get_modelmeta().custom_metadata_map
        model_type = meta.get("model_type", "")
        # 当前使用sharpe的zipformer2框架的asr-encoder流式模型，输入参数需要一一对应才能使用
        if model_type != "zipformer2":
            raise RuntimeError("encoder 的 model_type=%r，本解码循环只支持 zipformer2" % model_type)
        # 输入帧数，如 61
        self.input_frame_num = int(meta["T"])   
        # 每轮前移，如 48                          
        self.T_shift = int(meta["decode_chunk_len"])   
        # 读取asr-encoder 第一个输入的大小， (batch, T, n_mels)
        x_shape = self.encoder.get_inputs()[0].shape
        last_channel_num = x_shape[-1]
        self.n_mels = int(last_channel_num) if not isinstance(last_channel_num, str) else 80  
        
        # 读取asr-encoder第二类输入 状态张量(含KVCache) ----
        # Zipformer2是类U-Net结构的transformer+cnn混合网络，总计6个stack，每层2到5个block，合计19个block层，时间压缩是先压缩再恢复的，不同stack中的T帧长度不一致。
        # encoder规定了512帧（初始状态）的记忆特征状态，对应一帧10ms，总计是5.12s的历史数据。各stack的时间压缩长度不同，折成帧数是 256/128/64/32
        # 第二类输入要求的 116个 状态向量 主要含三类：
        # 1. 2 个全局状态：processed_lens (1,) int64、embed_states (1,128,3,19) float
        # 2. 76 个 KV Cache 状态, 19个block × 每层 4 个，key/val 形状 (T = 256/128/64/32, batch = 1, key_dim = 128 或 256 / value_dim = 48 或 96)，
        # 3.1 38 个 CNN 边界状态, 19个block × 每层 2 个(conv1 / conv2)，每个 block 除注意力、前馈外还有一个卷积模块，流式架构中在边界不使用padding，而是直接使用历史数据进行因果卷积
        # 3.2 因此，需要在输入1d 特征左侧拼接历史特征，历史特征的时间长度与卷积核在边界需要补的数据一致，(batch = 1,Channels = 192/256/512/768, T = kernel_size/2 = 15/7)
        self._state_proto = []
        for n in self.encoder.get_inputs()[1:]:
            if n.type not in self._DTYPES:
                raise RuntimeError("状态张量 %s 的类型 %s 未支持" % (n.name, n.type))
            shape = [1 if isinstance(s, str) else int(s) for s in n.shape]
            self._state_proto.append((shape, self._DTYPES[n.type]))   

        # 读取asr-decoder模型的参数
        dmeta = self.decoder.get_modelmeta().custom_metadata_map
        # 输入token的大小
        self.context_size = int(dmeta["context_size"])
        # Decoder 内置Embedding 字典，通过查表将idx转成对应的隐藏层向量
        # vocab_size是内置字典的词表大小，外部的tokens.txt的词表大小和排列顺序需要与之严格对应
        self.vocab_size = int(dmeta["vocab_size"])

        # ---- 读取各onnx模型的输入输出名字 ----
        self._enc_in = [n.name for n in self.encoder.get_inputs()]
        self._enc_out = [n.name for n in self.encoder.get_outputs()]
        self._dec_in = self.decoder.get_inputs()[0]
        self._dec_out = self.decoder.get_outputs()[0].name
        self._joi_in = [n.name for n in self.joiner.get_inputs()]
        self._joi_out = self.joiner.get_outputs()[0].name

        # ---- 词表 ----
        self.id2token = self.load_tokens(self.tokens_path)
        self.blank_id = self.find_blank(self.id2token)

        self.reset()

    # --------------------------------------------------------
    def reset(self):
        """清空全部跨块状态，回到「刚开始听」。""" 
        self.states = [np.zeros(shape, dtype=dt)
                       for shape, dt in self._state_proto]
        self.hyp = [self.blank_id] * self.context_size
        self.decoder_out = self.run_decoder(self.hyp)
        self._feat_buf = np.zeros((0, self.n_mels), dtype=np.float32)
        self.n_chunks = 0        # 喂过几次 encoder
        self.n_enc_frames = 0    # encoder 一共吐了多少帧
        self._text = ""

    # --------------------------------------------------------
    # 属性
    # --------------------------------------------------------
    @property
    def text(self):
        """当前识别文本。"""
        return self._text

    @property
    def token_ids(self):
        """已确定的 token id，**不含**开头那 context_size 个占位 blank。"""
        return self.hyp[self.context_size:]

    def info(self):
        """把关键尺寸打成一包，方便入口脚本打印。"""
        return {"T": self.input_frame_num, "decode_chunk_len": self.T_shift,
                "context_size": self.context_size,
                "vocab_size": self.vocab_size,
                "n_states": len(self.states), "blank_id": self.blank_id,
                "n_mels": self.n_mels, "providers": self.providers}

    # --------------------------------------------------------
    # 三个 session：C++ 侧逐个对应的低层接口
    # --------------------------------------------------------
    def run_encoder(self, x):
        """喂 (1, T, n_mels) 特征，返回 encoder 输出；状态就地累积到 self.states。

        注意返回值是 out[0]（编码结果），out[1:] 是更新后的状态张量，
        必须原样存回去 —— 这就是流式能记住前文的全部机制。
        """
        # feed 是 模型要求的输入，第一项是音频数据，往后106项是上一次推理保留的特征状态记忆self.states
        feed = {self._enc_in[0]: x}
        # 上一次的特征状态记忆self.states
        for i, name in enumerate(self._enc_in[1:]):
            feed[name] = self.states[i]
        # 输入新音频 + 旧记忆 ，输出 高级时序声学表征 + 新记忆
        out = self.encoder.run(self._enc_out, feed)
        # 新的特征状态记忆self.states，内部流通
        self.states = out[1:]
        # 输出 高级时序声学表征
        return out[0]

    def run_decoder(self, hyp):
        """喂最近 context_size 个 token id，返回 decoder（predictor）输出。"""
        # self.hyp[-self.context_size:]是最近 context_size 个 token id
        dtype = (np.int64 if self._dec_in.type == "tensor(int64)" else np.int32)
        y = np.array([hyp], dtype=dtype)
        return self.decoder.run([self._dec_out], {self._dec_in.name: y})[0]

    def run_joiner(self, encoder_out, decoder_out):
        """把 encoder 一帧与 decoder 输出合起来，返回整个词表的打分。"""
        # 单帧的enc_feats和上一次接龙预测结果self.decoder_out，共同作为输入
        return self.joiner.run([self._joi_out],
                               {self._joi_in[0]: encoder_out,
                                self._joi_in[1]: decoder_out})[0]

    # --------------------------------------------------------
    # 一站式：喂特征，边攒边解码
    # --------------------------------------------------------
    def accept_features(self, feats):
        """喂一块特征，把能解码的都解掉，返回当前完整文本。

        参数:
            feats —— float32 特征矩阵 (n, n_mels)，可能是 0 帧
        返回:
            当前识别文本（没有新 token 时与上次相同）
        """
        # 判断输入特征是否非空（feats.shape[0] > 0）
        if feats.shape[0]:
            # 输入特征非空，需要将其加入缓冲区
            # 判断当前缓冲区是否已有数据
            if self._feat_buf.shape[0]:
                # 情况 1：缓冲区非空，需要将新特征拼接到缓冲区尾部
                self._feat_buf = np.concatenate((self._feat_buf, feats))
            else:
                # 情况 2：缓冲区为空（首次输入或刚被清空）
                self._feat_buf = feats

        #  n_new 表示本段480ms音频成功转出的文字数量
        n_new = 0
        # 等待缓存帧数量大于模型要求的输入帧数量61
        while self._feat_buf.shape[0] > self.input_frame_num:
            # 每次从_feat_buf取最新 input_frame_num 帧给 encoder，np.ascontiguousarray将切片数据转换为内存中连续排列的数组
            x = np.ascontiguousarray(self._feat_buf[:self.input_frame_num][None])
            # _feat_buf 取完抛弃最老的 T_shift 帧，等待新的缓存进来，剩下的input_frame_num - T_shift 帧是前后两次输入共同的重叠帧   
            self._feat_buf = np.ascontiguousarray(self._feat_buf[self.T_shift:])
            # 累计推理输入+1
            self.n_chunks += 1

            # encoder 网络推理，输入新的61帧音频数据， 输出 高级时序声学表征 enc_feat, (b ,T = 48 / 4 = 12 , hidden_dim)
            enc_feats = self.run_encoder(x)          # 状态在 self.states 里累积
            nf = enc_feats.shape[1]                  # nf是非重叠的T_shift = 48帧，时序压缩4倍后的帧数量12
            self.n_enc_frames += nf                 # 已经编码的帧

            # 逐帧提取非重叠的音频压缩编码后的帧，输入到后续的预测网络decoder 和 联合网络joiner中
            # 注意encoder网络是输入多帧数据，decoder和joiner都是输入单帧数据，因此要顺序逐帧输入编码后的音频特征
            for k in range(nf):
                # 逐帧提取非重叠的音频压缩编码后的帧
                cur = enc_feats[0, k:k + 1]          # encoder 的第 k 帧
                # 联合网络 joiner 根据 当前的声学表征帧cur 和 上一次出字时 decoder 预测的下一个字，输出音频对应的文字id
                # .argmax()，贪心算法直接将概率最高的字作为确定的输出字符
                tid = int(self.run_joiner(cur, self.decoder_out).argmax())
                # 如果 joiner 的预测字不是空白，输入到decoder接龙网络中判断是否为重复字并预测下一个字(对应的特征)
                if tid != self.blank_id:
                    # 非 blank：接进假设hyp，并刷新 decoder
                    # hyp 是 Hypothesis（假设）的缩写。在 ASR 解码领域，它专指“当前正在生成中的、尚未最终确认的候选 token 序列”。
                    # 这本模型使用贪心算法，hyp已经是正式token序列了，不用再束搜索了
                    self.hyp.append(tid)
                    # 音频成功转出的文字数量 n_new + 1
                    n_new += 1
                    # 预测网络decoder判断重复并预测下一个字
                    self.decoder_out = self.run_decoder(self.hyp[-self.context_size:])
                    
        # 判断本轮输入的480ms音频，是否识别出了非 blank的token序列 self.hyp
        # 这里单独一个模块，考虑到Beam Search 必须等一个完整的计算周期（Chunk）结束后，才能输出确定的文本。虽然这里是贪心搜索，这里做兼容。
        if n_new:
            # 1. 有新字吐出，将 token id 列表转换为对应的字符串列表
            token_str_list = []
            # self.token_ids方法，从self.hyp中逐个提取预测出的token，查表找到对应字符
            for i in self.token_ids:
                # 从词表中查找 id 对应的 token 字符串
                # 使用 .get(i, "?") 是为了鲁棒性：万一遇到词表外的未知 id，用 "?" 兜底，防止程序崩溃
                token_str = self.id2token.get(i, "?")
                token_str_list.append(token_str)
                
            # 2. 将 token 字符串列表还原为人类可读的文本
            # detokenize 内部会处理 BPE 的 ▁ 符号（替换为空格）以及中英文之间的空格规则
            self._text = self.detokenize(token_str_list)
            
        return self._text

    # ============================================================
    # 一、词表与文本还原（模块级小工具）
    # ============================================================
    def load_tokens(self, path):
        """读 tokens.txt，返回 {id: token}。

        每行的格式是 `token id`（token 在前、id 在后）。用 rpartition 而不是
        split()：BPE token 里用 ▁ 表示词首空格，本身不含空格，但 rpartition
        对这种格式更稳，遇到多余空白的行也不会崩。
        """
        id2token = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                token, sep, idx = line.rpartition(" ")
                if not sep:
                    continue
                id2token[int(idx)] = token
        return id2token


    def find_blank(self, id2token):
        """在词表里找出 blank 的 id。找不到才退回 0 并提示。

        不要硬编码 0：blank 的 id 是模型定的，换个词表就可能变。
        """
        for i, t in id2token.items():
            if t in ("<blk>", "<blank>"):
                return i
        print("警告：tokens.txt 里没找到 <blk>/<blank>，退回 blank_id=0")
        return 0

    def detokenize(self, tokens):
        """把 BPE token 序列拼成可读文本（后处理）。

        不能只做 "".join()，原因有两层：

            1. ▁ 是 BPE 的词首标记，得还原成空格，否则得到「▁昨▁天▁是」。
            2. 本模型的词表把中文按单字切（▁礼▁拜▁二），单把 ▁ 换成空格会散成
            「礼 拜 二」。所以还得去空格，但只能去特定位置：
                - CJK 之间           去   「昨 天」 -> 「昨天」
                - 全角标点两侧       去   「Monday ，today」 -> 「Monday，today」
                - CJK 与 ASCII 之间  保留 「是 Monday」不能粘成「是Monday」
        """
        text = "".join(tokens).replace("▁", " ")
        text = re.sub(r"(?<=[%s])\s+(?=[%s])" % (_CJK, _CJK), "", text)
        text = re.sub(r"\s+(?=[%s])" % _FULLWIDTH, "", text)
        text = re.sub(r"(?<=[%s])\s+" % _FULLWIDTH, "", text)
        return text.strip()

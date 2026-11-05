/**
 * @file kws_decode.cpp
 * @brief 流式关键词唤醒实现：特征 → 三个 ONNX 会话 → 束搜索 → 触发判定。
 *
 * 本文件是 python_ref/src/kws_decode.py 的 C++ 移植，与 Python 版逐函数对应：
 *   run_encoder / run_decoder / run_joiner / accept_features / decode_frames /
 *   load_tokens / load_keywords / reset / record_blocked。
 * Python 里讲清「为什么」的段落这里同样保留，避免 C++ 版只剩代码。
 *
 * ============================================================
 * 一、与 AsrDecoder 的区别
 * ============================================================
 *   ASR 是 5000 词表上的贪心搜索：每帧取 argmax，blank 不刷新 decoder。
 *   KWS 是 263 个音素级单元上的束搜索：每帧保留 max_active_paths 条路径，
 *   每条路径挂一个 KwsContextGraph 状态，用来给关键词前缀加分；走完整条
 *   关键词后还要等足够多的 blank 帧，并检查平均声学概率，才真正触发。
 *
 * ============================================================
 * 二、每帧解码（对照上游 TransducerKeywordDecoder::Decode）
 * ============================================================
 *   1. 取当前 beam 的全部假设，**一次 batch** 跑 decoder；
 *   2. 把当前 encoder 帧复制 beam_size 份，**一次 batch** 跑 joiner；
 *   3. log_softmax 后加上各假设的累计分，在 beam×vocab 上取 top-k；
 *   4. 对每个候选 token 走一次 trie.ForwardOneStep：
 *      - 非 blank/unk：追加 token、记录概率与时间戳、刷新 trie 状态；
 *        若 trie 回到根，把 token 历史清空（对齐 C++ 的
 *        “start matching from the start state”）；
 *      - blank/unk：只累加 trailing blanks；
 *   5. 相同 token 序列的路径按 log-sum-exp 合并（Hypotheses::Add）；
 *   6. 取最佳假设，若 trie 命中、末尾 blank 帧数足够、且关键词的平均声学
 *      概率 >= 该关键词的阈值，则触发并重置 beam。
 *
 * ============================================================
 * 三、与 Python / 上游 C++ 的刻意差异
 * ============================================================
 *   1. 关键词树由独立的 KwsContextGraph 承担，本类只按值组合它；不把树逻辑
 *      复制进解码器，便于脱离 ORT 单测（见 tools/test_context_graph.cpp）；
 *   2. beam 用 `std::vector<Hypothesis>` 保存插入顺序，对应 Python dict 的
 *      迭代顺序；上游 C++ 用 unordered_map，顺序不确定。保持顺序是为了让
 *      top-k 的稳定排序与 Python 版逐位一致；
 *   3. 触发结果的 start_time 取关键词首 token 的时间，与 Python 版一致；
 *      上游 C++ 取的是「取结果那一刻」的帧时间；
 *   4. 计算平均声学概率时对 `ys_probs.size()` 做了 `min` 保护，上游 C++ 直接
 *      按 `matched_state->level` 下标访问；正常路径下两者相同；
 *   5. 监听 1.5 s 静音时软重置，保留未消费的特征尾巴（对齐 Python
 *      `_reset_stream(keep_feat_buf=True)` 与上游 `DecodeStreams`）。
 *
 * 其余分支——blank 固定 0、<unk> 当 blank、空白历史 [-1,...,-1,0]、非 blank
 * 才刷 trie、根节点清空历史、触发只重置 beam——全部照抄，不做“优化”。
 *
 * ============================================================
 * 四、调用顺序（对照各函数内的编号）
 * ============================================================
 *   构造：
 *     1.1 读 encoder / decoder 契约与状态模板
 *     1.2 读词表、关键词并建 trie
 *     1.3 初始化 beam 与 encoder 状态
 *   运行期：
 *     2.1 accept_features() 追加特征、软重置、按块排空
 *     2.2 decode_frames() 逐帧束搜索与触发
 *     3.1 run_encoder / run_decoder / run_joiner 三个会话调用
 *
 * ============================================================
 * 用法
 * ============================================================
 *     KwsDecoder kws(encoder, decoder, joiner, tokens_path, keywords_path);
 *     auto hits = kws.accept_features(feats);
 *     for (const auto& h : hits) { ... }
 *     kws.reset();   // 换文件 / 重新开始监听
 */

#include "audio_ai_process/kws_decode.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace audio_ai {

namespace {

// ============================================================
// 一、小工具
// ============================================================

// 读取 custom metadata 的单个键；不存在返回空串，由 require_int 统一报错。
std::string meta_value(const Ort::Session& session, const char* key,
                       Ort::AllocatorWithDefaultOptions& allocator) {
    auto metadata = session.GetModelMetadata();
    auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
    return value ? std::string(value.get()) : std::string();
}

// 把 metadata 字符串转成整数：空值、非数字、带尾随字符都直接抛，
// 避免带着错参数继续构造。std::stoll 只要求前缀合法，所以必须检查 used。
int64_t require_int(const std::string& value, const char* key) {
    if (value.empty()) {
        throw std::runtime_error(std::string("模型 metadata 缺少 ") + key);
    }
    try {
        size_t used = 0;
        const int64_t result = std::stoll(value, &used);
        if (used != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return result;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("模型 metadata ") + key +
                                 " 不是合法整数：" + value);
    }
}

// 解析关键词行里的 `:x` / `#x` 数值；空串、非数字、带尾随字符都抛，
// 错误信息带上行号，方便直接定位到 keywords.txt。
float parse_float_field(const std::string& text, const std::string& what) {
    if (text.empty()) {
        throw std::runtime_error(what + " 缺少数值");
    }
    try {
        size_t used = 0;
        const float value = std::stof(text, &used);
        if (used != text.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error(what + " 不是合法浮点数：" + text);
    }
}

// log(exp(a) + exp(b))，用于合并相同 token 序列的两条路径。
// 照抄 Python 的 _log_add：先处理 -inf，再用 log1p 保证数值稳定。
float log_add(float a, float b) {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    if (a == neg_inf) {
        return b;
    }
    if (b == neg_inf) {
        return a;
    }
    const float hi = std::max(a, b);
    const float lo = std::min(a, b);
    return hi + std::log1p(std::exp(lo - hi));
}

// 用空格拼接 token 文本；关键词没有 @原文 时作为展示名。
std::string join_tokens(const std::vector<std::string>& tokens) {
    std::string out;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            out += ' ';
        }
        out += tokens[i];
    }
    return out;
}

}  // namespace

// ==================== 构造 ====================

// 绑定三个会话，读取模型契约，加载词表/关键词并建 trie，最后 reset()。
// 所有形状/尺寸都从模型本身读，换权重目录不用改代码。
KwsDecoder::KwsDecoder(Ort::Session& encoder, Ort::Session& decoder,
                       Ort::Session& joiner, const std::string& tokens_path,
                       const std::string& keywords_path, int max_active_paths,
                       int num_trailing_blanks, float keywords_score,
                       float keywords_threshold, float reset_after_silence,
                       int sample_rate, bool debug)
    : encoder_(encoder),
      decoder_(decoder),
      joiner_(joiner),
      memory_info_(
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      id2token_(load_tokens(tokens_path)),
      token2id_(invert_token_map(id2token_)),
      keyword_specs_(load_keywords(keywords_path, token2id_)),
      graph_(keyword_specs_, keywords_score, keywords_threshold),
      sample_rate_(sample_rate),
      max_active_paths_(std::max(1, max_active_paths)),
      num_trailing_blanks_(num_trailing_blanks),
      keywords_score_(keywords_score),
      keywords_threshold_(keywords_threshold),
      reset_after_silence_(reset_after_silence),
      debug_(debug) {
    // —— 1.1 encoder 契约：模型类型、T、decode_chunk_len、n_mels ——
    // 与 AsrDecoder 相同，只有 model_type 是 zipformer2 时下面的状态张量与
    // 束搜索循环才成立。
    const std::string model_type =
        meta_value(encoder_, "model_type", allocator_);
    if (model_type != "zipformer2") {
        throw std::runtime_error(
            "encoder 的 model_type=" + model_type +
            "，本解码循环只支持 zipformer2");
    }
    input_frame_num_ = static_cast<int>(
        require_int(meta_value(encoder_, "T", allocator_), "T"));
    t_shift_ = static_cast<int>(require_int(
        meta_value(encoder_, "decode_chunk_len", allocator_),
        "decode_chunk_len"));

    // encoder 第一个输入是 (batch, T, n_mels)，最后一维是 Mel 通道数。
    {
        const auto shape = encoder_.GetInputTypeInfo(0)
                               .GetTensorTypeAndShapeInfo()
                               .GetShape();
        if (!shape.empty() && shape.back() > 0) {
            n_mels_ = static_cast<int>(shape.back());
        }
    }

    // encoder 输出最后一维 = joiner 的输入宽度（当前权重 320）。
    // 若导出时该维是动态的，再退回 joiner 第一个输入的最后一维。
    {
        const auto shape = encoder_.GetOutputTypeInfo(0)
                               .GetTensorTypeAndShapeInfo()
                               .GetShape();
        if (!shape.empty() && shape.back() > 0) {
            encoder_dim_ = static_cast<int>(shape.back());
        }
    }

    // —— decoder 契约：context_size / vocab_size / 输入类型 ——
    context_size_ = static_cast<int>(require_int(
        meta_value(decoder_, "context_size", allocator_), "context_size"));
    vocab_size_ = static_cast<int>(require_int(
        meta_value(decoder_, "vocab_size", allocator_), "vocab_size"));
    dec_input_type_ = decoder_.GetInputTypeInfo(0)
                          .GetTensorTypeAndShapeInfo()
                          .GetElementType();
    if (dec_input_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
        dec_input_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        throw std::runtime_error("decoder 输入不是 int64/int32");
    }

    // joiner 必须有两个输入；本模型要求二维 (N, dim)，不是常规的
    // (N, 1, dim)，构造期先按签名检查，避免运行期才报形状错误。
    if (joiner_.GetInputCount() < 2) {
        throw std::runtime_error("joiner 输入少于 2 个");
    }
    for (size_t i = 0; i < 2; ++i) {
        const auto shape = joiner_.GetInputTypeInfo(i)
                               .GetTensorTypeAndShapeInfo()
                               .GetShape();
        if (shape.size() != 2) {
            throw std::runtime_error(
                "本 KWS 模型的 joiner 需要二维输入 (N, dim)，当前输入的秩是 " +
                std::to_string(shape.size()));
        }
    }

    // —— 输入输出名字（存储在本对象里，指针指向这些 string）——
    for (size_t i = 0; i < encoder_.GetInputCount(); ++i) {
        auto name = encoder_.GetInputNameAllocated(i, allocator_);
        enc_in_names_.emplace_back(name.get());
    }
    for (size_t i = 0; i < encoder_.GetOutputCount(); ++i) {
        auto name = encoder_.GetOutputNameAllocated(i, allocator_);
        enc_out_names_.emplace_back(name.get());
    }
    for (const std::string& name : enc_in_names_) {
        enc_in_ptrs_.push_back(name.c_str());
    }
    for (const std::string& name : enc_out_names_) {
        enc_out_ptrs_.push_back(name.c_str());
    }

    {
        auto name = decoder_.GetInputNameAllocated(0, allocator_);
        dec_in_name_ = name.get();
        auto out = decoder_.GetOutputNameAllocated(0, allocator_);
        dec_out_name_ = out.get();
    }
    {
        auto in0 = joiner_.GetInputNameAllocated(0, allocator_);
        auto in1 = joiner_.GetInputNameAllocated(1, allocator_);
        auto out = joiner_.GetOutputNameAllocated(0, allocator_);
        joi_in_name_0_ = in0.get();
        joi_in_name_1_ = in1.get();
        joi_out_name_ = out.get();
    }

    // encoder 输出宽度拿不到时，用 joiner 第一个输入的最后一维兜底。
    if (encoder_dim_ <= 0) {
        const auto shape = joiner_.GetInputTypeInfo(0)
                               .GetTensorTypeAndShapeInfo()
                               .GetShape();
        if (!shape.empty() && shape.back() > 0) {
            encoder_dim_ = static_cast<int>(shape.back());
        }
    }
    if (encoder_dim_ <= 0) {
        throw std::runtime_error("无法确定 encoder_out 维度");
    }

    // —— 状态张量模板：形状取自输入签名，动态维按 batch=1 填 1 ——
    // 数量与 AsrDecoder 不同（KWS 当前权重 38 个），因此这里不写死数量。
    if (encoder_.GetInputCount() < 1) {
        throw std::runtime_error("encoder 没有输入");
    }
    for (size_t i = 1; i < encoder_.GetInputCount(); ++i) {
        const auto type_info = encoder_.GetInputTypeInfo(i);
        const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = tensor_info.GetShape();
        for (int64_t& dim : shape) {
            if (dim < 0) {
                dim = 1;  // Python 版：动态维一律取 1
            }
        }
        const auto type = tensor_info.GetElementType();
        if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
            type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
            type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            throw std::runtime_error("encoder 状态张量类型未支持");
        }
        state_proto_.push_back({shape, type});
    }

    // —— 1.2 词表与空白 / unk ——
    // KWS 解码器把 blank 硬编码为 0；词表对不上时直接拒绝，避免静默偏移。
    {
        const auto it = token2id_.find("<blk>");
        if (it == token2id_.end()) {
            throw std::runtime_error(
                "tokens.txt 里没有 <blk>，KWS 解码器要求 blank 固定为 0");
        }
        if (it->second != 0) {
            throw std::runtime_error(
                "tokens.txt 里 <blk> 的 id 是 " + std::to_string(it->second) +
                "，KWS 解码器把 blank 硬编码为 0");
        }
        blank_id_ = it->second;
    }
    {
        const auto it = token2id_.find("<unk>");
        unk_id_ = it != token2id_.end() ? it->second : -1;
    }

    // 关键词展示名：有 @原文 用原文，否则用 token 拼接。
    for (const KeywordSpec& spec : keyword_specs_) {
        if (!spec.phrase.empty()) {
            keywords_display_.push_back(spec.phrase);
            continue;
        }
        std::vector<std::string> tokens;
        tokens.reserve(spec.token_ids.size());
        for (int32_t id : spec.token_ids) {
            const auto it = id2token_.find(id);
            tokens.push_back(it != id2token_.end() ? it->second : "?");
        }
        keywords_display_.push_back(join_tokens(tokens));
    }

    // 初始帧长：subsampling=4 时的估计值，第一次 encoder 输出后会按实际
    // 帧数校正（本模型通常仍是 40 ms）。
    frame_shift_s_ =
        static_cast<float>(t_shift_) * 0.01f / 4.0f;

    // —— 1.3 初始 beam 与 encoder 状态 ——
    reset();
}

// ==================== 状态重置 ====================

// 只重置流状态，不碰统计计数；keep_feat_buf=true 时保留尚未消费的特征尾巴。
// 对应 Python 的 _reset_stream()。
void KwsDecoder::reset_stream(bool keep_feat_buf) {
    states_ = make_zero_states();

    hyps_.clear();
    hyps_.push_back(blank_hypothesis());

    frame_offset_ = 0;
    best_trailing_blanks_ = 0;
    last_best_level_ = 0;
    debug_last_valid_ = false;
    debug_last_key_ = {};
    frame_shift_s_ = static_cast<float>(t_shift_) * 0.01f / 4.0f;

    if (!keep_feat_buf) {
        feat_buf_.clear();
        n_buffered_ = 0;
    }
}

// 清空全部跨块状态与统计，回到「刚开始监听」。模型契约、词表、trie 都不动。
void KwsDecoder::reset() {
    reset_stream(false);
    n_chunks_ = 0;
    n_enc_frames_ = 0;
    n_hits_ = 0;
}

// 按构造期保存的形状/类型重新分配 encoder 状态张量，并全部填 0。
// 与 AsrDecoder::reset() 的分配口径一致：动态维在构造期已填成 1。
std::vector<Ort::Value> KwsDecoder::make_zero_states() {
    std::vector<Ort::Value> states;
    states.reserve(state_proto_.size());
    for (const StateProto& proto : state_proto_) {
        const size_t count = static_cast<size_t>(std::accumulate(
            proto.shape.begin(), proto.shape.end(), int64_t{1},
            std::multiplies<int64_t>()));

        // CreateTensor<T> 是模板，T 必须编译期确定，所以 float / int64 /
        // int32 各写一遍。
        Ort::Value value(nullptr);
        if (proto.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            value = Ort::Value::CreateTensor<float>(
                allocator_, proto.shape.data(), proto.shape.size());
            std::fill_n(value.GetTensorMutableData<float>(), count, 0.0f);
        } else if (proto.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            value = Ort::Value::CreateTensor<int64_t>(
                allocator_, proto.shape.data(), proto.shape.size());
            std::fill_n(value.GetTensorMutableData<int64_t>(), count,
                        int64_t{0});
        } else {
            value = Ort::Value::CreateTensor<int32_t>(
                allocator_, proto.shape.data(), proto.shape.size());
            std::fill_n(value.GetTensorMutableData<int32_t>(), count,
                        int32_t{0});
        }
        states.push_back(std::move(value));
    }
    return states;
}

// 空白假设：ys = [-1]*(context_size-1) + [0]，累计分 0，trie 在根。
// -1 不是笔误：ONNX Gather 支持负索引，不要 clamp。
std::vector<int64_t> KwsDecoder::blank_ys() const {
    std::vector<int64_t> ys(static_cast<size_t>(context_size_), -1);
    if (!ys.empty()) {
        ys.back() = blank_id_;
    }
    return ys;
}

KwsDecoder::Hypothesis KwsDecoder::blank_hypothesis() const {
    Hypothesis hyp;
    hyp.ys = blank_ys();
    hyp.log_prob = 0.0f;
    hyp.context_state = graph_.root();
    return hyp;
}

// ==================== 三个会话的低层调用 ====================

// 喂 (1, T, n_mels) 特征块，返回 encoder_out；新状态就地写回 states_。
// 与 AsrDecoder::run_encoder 同构：第一项是本次音频，往后是记忆状态；
// 输出第 0 项是编码结果，第 1 项起是更新后的状态。
Ort::Value KwsDecoder::run_encoder(Ort::Value x) {
    std::vector<Ort::Value> inputs;
    inputs.reserve(1 + states_.size());
    inputs.push_back(std::move(x));
    for (Ort::Value& state : states_) {
        inputs.push_back(std::move(state));
    }
    states_.clear();

    auto outputs = encoder_.Run(
        Ort::RunOptions{nullptr}, enc_in_ptrs_.data(), inputs.data(),
        inputs.size(), enc_out_ptrs_.data(), enc_out_ptrs_.size());

    if (outputs.size() != enc_out_names_.size() || outputs.empty()) {
        throw std::runtime_error("encoder 输出个数与模型签名不一致");
    }
    states_.reserve(outputs.size() - 1);
    for (size_t i = 1; i < outputs.size(); ++i) {
        states_.push_back(std::move(outputs[i]));
    }
    return std::move(outputs[0]);
}

// 喂 (H, context_size) 的 token id 批次，返回 (H, decoder_dim) 的浮点缓冲。
// decoder 负责「接龙」：判断是否重复，并预测下一个 token 对应的特征。
// 输入 dtype 跟随模型签名（int64 或 int32），不假设一定是 int64。
std::vector<float> KwsDecoder::run_decoder(const std::vector<int64_t>& ys,
                                           size_t n_hyps) {
    if (n_hyps == 0 ||
        ys.size() != n_hyps * static_cast<size_t>(context_size_)) {
        throw std::invalid_argument(
            "KwsDecoder::run_decoder: ys 长度与 (H, context_size) 不一致");
    }

    const int64_t shape[2] = {static_cast<int64_t>(n_hyps),
                              static_cast<int64_t>(context_size_)};
    std::vector<Ort::Value> inputs;
    if (dec_input_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        Ort::Value y =
            Ort::Value::CreateTensor<int64_t>(allocator_, shape, 2);
        std::copy(ys.begin(), ys.end(), y.GetTensorMutableData<int64_t>());
        inputs.push_back(std::move(y));
    } else {
        Ort::Value y =
            Ort::Value::CreateTensor<int32_t>(allocator_, shape, 2);
        int32_t* dst = y.GetTensorMutableData<int32_t>();
        for (size_t i = 0; i < ys.size(); ++i) {
            dst[i] = static_cast<int32_t>(ys[i]);
        }
        inputs.push_back(std::move(y));
    }

    const char* in_names[] = {dec_in_name_.c_str()};
    const char* out_names[] = {dec_out_name_.c_str()};
    auto outputs = decoder_.Run(Ort::RunOptions{nullptr}, in_names,
                                inputs.data(), inputs.size(), out_names, 1);

    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    const size_t count = info.GetElementCount();
    if (count % n_hyps != 0) {
        throw std::runtime_error("decoder 输出元素数不能按 beam 整除");
    }
    const float* data = outputs[0].GetTensorData<float>();
    return std::vector<float>(data, data + count);
}

// encoder 的当前帧复制 H 份 + decoder_out → (H, vocab) 的 logit 缓冲。
// 本模型的 joiner 要二维输入 (N, dim)，不是常规的 (N, 1, dim)。
std::vector<float> KwsDecoder::run_joiner(
    const float* encoder_frame, size_t n_hyps, size_t enc_dim,
    const std::vector<float>& decoder_out, size_t dec_dim) {
    if (n_hyps == 0 || dec_dim == 0 ||
        decoder_out.size() != n_hyps * dec_dim) {
        throw std::invalid_argument(
            "KwsDecoder::run_joiner: decoder_out 长度与 (H, dec_dim) 不一致");
    }

    // 输入 0：(H, enc_dim)，把同一帧重复 H 份。
    const int64_t enc_shape[2] = {static_cast<int64_t>(n_hyps),
                                  static_cast<int64_t>(enc_dim)};
    Ort::Value enc =
        Ort::Value::CreateTensor<float>(allocator_, enc_shape, 2);
    float* enc_dst = enc.GetTensorMutableData<float>();
    for (size_t i = 0; i < n_hyps; ++i) {
        std::memcpy(enc_dst + i * enc_dim, encoder_frame,
                    enc_dim * sizeof(float));
    }

    // 输入 1：(H, dec_dim)，decoder 已经按 beam 顺序排好。
    const int64_t dec_shape[2] = {static_cast<int64_t>(n_hyps),
                                  static_cast<int64_t>(dec_dim)};
    Ort::Value dec =
        Ort::Value::CreateTensor<float>(allocator_, dec_shape, 2);
    std::memcpy(dec.GetTensorMutableData<float>(), decoder_out.data(),
                decoder_out.size() * sizeof(float));

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(enc));
    inputs.push_back(std::move(dec));

    const char* in_names[] = {joi_in_name_0_.c_str(), joi_in_name_1_.c_str()};
    const char* out_names[] = {joi_out_name_.c_str()};
    auto outputs = joiner_.Run(Ort::RunOptions{nullptr}, in_names,
                               inputs.data(), inputs.size(), out_names, 1);

    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    const size_t count = info.GetElementCount();
    if (count != n_hyps * static_cast<size_t>(vocab_size_)) {
        throw std::runtime_error(
            "joiner 输出元素数与 (beam, vocab) 不一致");
    }
    const float* data = outputs[0].GetTensorData<float>();
    return std::vector<float>(data, data + count);
}

// ==================== 一体化接口 ====================

// 喂一块特征：追加进缓冲 → 必要时软重置 → 按 `buffered > T` 排空。
// 每个块：取前 T 帧 → encoder 一次 → 时间轴前进 T_shift 帧 → 逐帧束搜索。
std::vector<KeywordHit> KwsDecoder::accept_features(
    const LogFbankFeatures& feats) {
    // 形状不对时越界读取不会报错，只会静默算出乱码，所以直接抛。
    if (feats.n_mels != n_mels_) {
        throw std::invalid_argument(
            "KwsDecoder: feats.n_mels 与模型输入维度不一致");
    }
    if (feats.n_frames < 0 ||
        feats.data.size() != static_cast<size_t>(feats.n_frames) *
                                 static_cast<size_t>(feats.n_mels)) {
        throw std::invalid_argument(
            "KwsDecoder: feats.data 长度与 n_frames * n_mels 不一致");
    }

    if (feats.n_frames > 0) {
        feat_buf_.insert(feat_buf_.end(), feats.data.begin(), feats.data.end());
        n_buffered_ += feats.n_frames;
    }

    // 与上游 DecodeStreams 一致：解码前先看上一块结束时最佳假设的
    // trailing blanks，静音足够长就软重置（保留尚未消费的特征尾巴）。
    if (static_cast<float>(best_trailing_blanks_) * frame_shift_s_ >
        reset_after_silence_) {
        reset_stream(true);
    }

    std::vector<KeywordHit> hits;

    // 严格大于：写成 >= 会在批量条件恰好 k*T_shift + T 时多喂一次。
    while (n_buffered_ > input_frame_num_) {
        // 1) 取缓冲最前面的 T 帧组成 (1, T, n_mels)。
        const int64_t x_shape[3] = {1, input_frame_num_, n_mels_};
        Ort::Value x =
            Ort::Value::CreateTensor<float>(allocator_, x_shape, 3);
        const size_t x_count = static_cast<size_t>(input_frame_num_) *
                               static_cast<size_t>(n_mels_);
        std::memcpy(x.GetTensorMutableData<float>(), feat_buf_.data(),
                    x_count * sizeof(float));

        // 2) 时间轴前进 T_shift 帧。先移缓冲再编码，与 Python 版顺序一致；
        //    剩下的 T − T_shift 帧是下一次输入要用的重叠上下文。
        const size_t drop =
            static_cast<size_t>(t_shift_) * static_cast<size_t>(n_mels_);
        feat_buf_.erase(feat_buf_.begin(), feat_buf_.begin() + drop);
        n_buffered_ -= t_shift_;
        ++n_chunks_;

        Ort::Value enc_out = run_encoder(std::move(x));
        const auto enc_shape =
            enc_out.GetTensorTypeAndShapeInfo().GetShape();
        if (enc_shape.size() != 3) {
            throw std::runtime_error("encoder_out 不是三维张量");
        }
        const int nf = static_cast<int>(enc_shape[1]);
        const int enc_dim = static_cast<int>(enc_shape[2]);
        n_enc_frames_ += nf;
        if (nf == 0) {
            continue;
        }

        // 3) 每个输出帧代表多少秒：T_shift/nf × 10 ms（通常 40 ms）。
        frame_shift_s_ = static_cast<float>(t_shift_) * 0.01f /
                         static_cast<float>(nf);

        // 4) 逐帧束搜索；命中结果追加到 hits。
        decode_frames(enc_out.GetTensorData<float>(), nf, enc_dim, &hits);
        frame_offset_ += nf;
    }

    if (!hits.empty()) {
        n_hits_ += static_cast<int>(hits.size());
    }
    return hits;
}

// ==================== 束搜索 ====================

// 对 encoder 输出的一整块逐帧解码。每帧：
//   beam 一次 batch 跑 decoder / joiner → top-k → 逐候选走 trie →
//   合并同序列 → 取最佳 → 触发判定。
void KwsDecoder::decode_frames(const float* enc_data, int nf, int enc_dim,
                               std::vector<KeywordHit>* hits) {
    const size_t V = static_cast<size_t>(vocab_size_);
    const size_t K = static_cast<size_t>(max_active_paths_);

    for (int t = 0; t < nf; ++t) {
        const size_t H = hyps_.size();
        if (H == 0) {
            throw std::logic_error("KwsDecoder: beam 为空");
        }

        // 1) decoder 输入：每条假设最近的 context_size 个 token。
        std::vector<int64_t> ys_batch;
        ys_batch.reserve(H * static_cast<size_t>(context_size_));
        for (const Hypothesis& hyp : hyps_) {
            if (hyp.ys.size() < static_cast<size_t>(context_size_)) {
                throw std::logic_error(
                    "KwsDecoder: 假设的 token 历史短于 context_size");
            }
            const size_t begin =
                hyp.ys.size() - static_cast<size_t>(context_size_);
            ys_batch.insert(ys_batch.end(), hyp.ys.begin() + begin,
                            hyp.ys.end());
        }

        const std::vector<float> dec_out = run_decoder(ys_batch, H);
        const size_t dec_dim = dec_out.size() / H;

        // 2) joiner：当前 encoder 帧复制 H 份 + decoder 输出。
        const float* frame =
            enc_data + static_cast<size_t>(t) * static_cast<size_t>(enc_dim);
        const std::vector<float> logits =
            run_joiner(frame, H, static_cast<size_t>(enc_dim), dec_out,
                       dec_dim);

        // 3) 逐行 log_softmax：先减行最大值，再减 log(sum(exp))。
        std::vector<float> logprobs(logits.size());
        for (size_t i = 0; i < H; ++i) {
            const float* row = logits.data() + i * V;
            float max_logit = row[0];
            for (size_t j = 1; j < V; ++j) {
                max_logit = std::max(max_logit, row[j]);
            }
            float sum = 0.0f;
            for (size_t j = 0; j < V; ++j) {
                sum += std::exp(row[j] - max_logit);
            }
            const float log_sum = std::log(sum);
            float* out = logprobs.data() + i * V;
            for (size_t j = 0; j < V; ++j) {
                out[j] = row[j] - max_logit - log_sum;
            }
        }

        // 4) 加上各假设已有的累计分，得到 H*V 个候选分。
        std::vector<float> scores(H * V);
        for (size_t i = 0; i < H; ++i) {
            const float base = hyps_[i].log_prob;
            for (size_t j = 0; j < V; ++j) {
                scores[i * V + j] = logprobs[i * V + j] + base;
            }
        }

        // 5) 稳定排序取 top-k。NumPy 的 argsort(kind="stable") 在分数相同时
        //    保留较小下标，std::stable_sort 同样如此；不要换成 partial_sort。
        std::vector<size_t> order(scores.size());
        std::iota(order.begin(), order.end(), size_t{0});
        const size_t k = std::min(K, order.size());
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b) {
                             return scores[a] > scores[b];
                         });
        order.resize(k);

        std::vector<Hypothesis> next_hyps;
        next_hyps.reserve(k);
        bool triggered = false;

        for (size_t flat_idx : order) {
            const size_t hyp_idx = flat_idx / V;
            const int token = static_cast<int>(flat_idx % V);

            Hypothesis new_hyp = hyps_[hyp_idx];
            float context_score = 0.0f;

            // blank 固定为 0；<unk> 也按 blank 处理。
            if (token != blank_id_ && token != unk_id_) {
                new_hyp.ys.push_back(token);
                new_hyp.timestamps.push_back(frame_offset_ + t);
                new_hyp.ys_probs.push_back(
                    std::exp(logprobs[hyp_idx * V + static_cast<size_t>(token)]));
                new_hyp.num_trailing_blanks = 0;

                const KwsStepResult step = graph_.forward_one_step(
                    hyps_[hyp_idx].context_state, token);
                context_score = step.context_score;
                new_hyp.context_state = step.state;

                // 回到根：当前 token 不属于任何关键词前缀，清空历史。
                // 与 C++ 一致，连刚 push 的 token 一起丢掉。
                if (step.state->token == -1) {
                    new_hyp.ys = blank_ys();
                    new_hyp.timestamps.clear();
                    new_hyp.ys_probs.clear();
                }
            } else {
                ++new_hyp.num_trailing_blanks;
            }

            new_hyp.log_prob = scores[flat_idx] + context_score;
            merge_hypothesis(&next_hyps, std::move(new_hyp));
        }

        // 6) 取累计分最高的假设。std::max_element 返回第一个最大值，
        //    与 Python max() 在并列时的选择一致。
        const Hypothesis best = *std::max_element(
            next_hyps.begin(), next_hyps.end(),
            [](const Hypothesis& a, const Hypothesis& b) {
                return a.log_prob < b.log_prob;
            });

        // 7) 触发判定：trie 命中 + 尾随 blank 足够 + 平均声学概率过阈值。
        const KwsContextState* matched = nullptr;
        if (graph_.is_matched(best.context_state, &matched)) {
            const int level = matched->level;
            const size_t n_prob = std::min<size_t>(
                static_cast<size_t>(std::max(level, 0)), best.ys_probs.size());
            float avg_prob = 0.0f;
            if (n_prob > 0) {
                float sum = 0.0f;
                for (size_t i = 0; i < n_prob; ++i) {
                    sum += best.ys_probs[i];
                }
                avg_prob = sum / static_cast<float>(n_prob);
            }

            if (best.num_trailing_blanks > num_trailing_blanks_ &&
                avg_prob >= matched->ac_threshold) {
                KeywordHit hit;
                // 与 C++ 一致取最后 level 个 token；这里对 ys / timestamps
                // 都做了 min 保护，避免异常状态下越界。
                const size_t n_ys = std::min<size_t>(
                    static_cast<size_t>(std::max(level, 0)), best.ys.size());
                const size_t n_ts = std::min(n_ys, best.timestamps.size());
                const size_t begin_ys = best.ys.size() - n_ys;
                const size_t begin_ts = best.timestamps.size() - n_ts;
                hit.tokens.reserve(n_ys);
                hit.timestamps.reserve(n_ys);
                for (size_t i = 0; i < n_ys; ++i) {
                    const int64_t id = best.ys[begin_ys + i];
                    const auto it =
                        id2token_.find(static_cast<int32_t>(id));
                    hit.tokens.push_back(it != id2token_.end() ? it->second
                                                               : "?");
                    if (i < n_ts) {
                        hit.timestamps.push_back(
                            static_cast<float>(best.timestamps[begin_ts + i]) *
                            frame_shift_s_);
                    }
                }
                hit.keyword = !matched->phrase.empty()
                                  ? matched->phrase
                                  : join_tokens(hit.tokens);
                hit.start_time =
                    !hit.timestamps.empty()
                        ? hit.timestamps[0]
                        : static_cast<float>(frame_offset_) * frame_shift_s_;
                hit.avg_prob = avg_prob;
                hits->push_back(std::move(hit));

                triggered = true;
                debug_last_valid_ = false;
                // 触发后只重置 beam，encoder 状态继续累积。
                next_hyps.clear();
                next_hyps.push_back(blank_hypothesis());
            } else if (debug_) {
                record_blocked(matched, avg_prob, level);
            }
        }

        hyps_ = std::move(next_hyps);
        last_best_level_ = best.context_state->level;
        best_trailing_blanks_ =
            triggered ? 0 : best.num_trailing_blanks;
    }
}

// 同 token 序列的路径按 log-sum-exp 合并（上游 Hypotheses::Add）。
// 已存在时只更新累计分，其余字段保留先插入的那条，与 Python 版一致。
void KwsDecoder::merge_hypothesis(std::vector<Hypothesis>* hyps,
                                  Hypothesis hyp) {
    for (Hypothesis& old : *hyps) {
        if (old.ys == hyp.ys) {
            old.log_prob = log_add(old.log_prob, hyp.log_prob);
            return;
        }
    }
    hyps->push_back(std::move(hyp));
}

// 记录「trie 已走到末端，但平均声学概率没过阈值」的候选。
// avg_prob 够高却没触发只说明尾随 blank 还没攒够，属正常等待，不记录。
// 同词同分数连续多帧只记一次；列表上限 2000，stderr 照常打印。
void KwsDecoder::record_blocked(const KwsContextState* state, float avg_prob,
                                int level) {
    if (avg_prob >= state->ac_threshold) {
        return;
    }

    const std::string phrase =
        state->phrase.empty() ? "<未命名关键词>" : state->phrase;
    const std::pair<std::string, int> key{
        phrase, static_cast<int>(std::lround(avg_prob * 1000.0f))};
    if (debug_last_valid_ && key == debug_last_key_) {
        return;
    }
    debug_last_key_ = key;
    debug_last_valid_ = true;

    if (debug_events_.size() < 2000) {
        debug_events_.push_back(
            {phrase, avg_prob, state->ac_threshold, level, frame_offset_});
    }
    std::fprintf(stderr,
                 "\n[KWS debug] 被阈值挡住 %s  avg_prob=%.3f < "
                 "threshold=%.3f (level=%d)\n",
                 phrase.c_str(), avg_prob, state->ac_threshold, level);
}

// ==================== 词表与关键词 ====================

// 读 tokens.txt，每行 `token id`。用最后一个空格切分（rpartition 口径）：
// token 本身不含空格，遇到多余空白也不会切错 id。
std::map<int32_t, std::string> KwsDecoder::load_tokens(
    const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("找不到词表：" + path);
    }

    std::map<int32_t, std::string> id2token;
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        // rpartition：没有空格的行直接跳过（Python 的 sep 为空）。
        const size_t pos = line.find_last_of(' ');
        if (pos == std::string::npos) {
            continue;
        }
        const std::string token = line.substr(0, pos);
        const std::string idx = line.substr(pos + 1);
        try {
            size_t used = 0;
            const long long value = std::stoll(idx, &used);
            if (used != idx.size() ||
                value < std::numeric_limits<int32_t>::min() ||
                value > std::numeric_limits<int32_t>::max()) {
                throw std::invalid_argument("bad id");
            }
            id2token[static_cast<int32_t>(value)] = token;
        } catch (const std::exception&) {
            throw std::runtime_error("词表第 " + std::to_string(lineno) +
                                     " 行的 id 不是合法整数：" + idx);
        }
    }

    if (id2token.empty()) {
        throw std::runtime_error("词表为空：" + path);
    }
    return id2token;
}

// 由 {id: token} 建 {token: id}；重复 token 时后出现的覆盖先出现的，
// 与 Python dict 的赋值语义一致。
std::unordered_map<std::string, int32_t> KwsDecoder::invert_token_map(
    const std::map<int32_t, std::string>& id2token) {
    std::unordered_map<std::string, int32_t> token2id;
    token2id.reserve(id2token.size());
    for (const auto& kv : id2token) {
        token2id[kv.second] = kv.first;
    }
    return token2id;
}

// 解析关键词文件，对齐上游 utils.cc: EncodeBase 与 Python _load_keywords。
// 能查到 id 的词当 token；`:x` / `#x` / `@x` 分别写分数 / 阈值 / 原文；
// 其余词报错并带行号。没有 token 的行（纯参数 / 注释）跳过。
std::vector<KeywordSpec> KwsDecoder::load_keywords(
    const std::string& path,
    const std::unordered_map<std::string, int32_t>& token2id) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("找不到关键词文件：" + path);
    }

    std::vector<KeywordSpec> specs;
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        // 整行以 # 开头只可能是注释：合法关键词行必须以 token 开头，
        // 阈值 #0.35 只能出现在 token 之后。
        const size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos) {
            continue;
        }
        if (line[first] == '#') {
            continue;
        }

        KeywordSpec spec;
        std::istringstream iss(line);
        std::string word;
        while (iss >> word) {
            const auto it = token2id.find(word);
            if (it != token2id.end()) {
                spec.token_ids.push_back(it->second);
                continue;
            }

            const std::string where =
                "关键词文件第 " + std::to_string(lineno) + " 行";
            switch (word[0]) {
                case ':':
                    spec.score = parse_float_field(
                        word.substr(1), where + " 的 boosting score");
                    break;
                case '#':
                    spec.threshold = parse_float_field(
                        word.substr(1), where + " 的触发阈值");
                    break;
                case '@':
                    spec.phrase = word.substr(1);
                    break;
                default:
                    throw std::runtime_error(
                        where + "：token " + word +
                        " 不在 tokens.txt 里");
            }
        }

        if (spec.token_ids.empty()) {
            // 纯注释 / 纯参数行；C++ 会收下一条空关键词，效果相同。
            continue;
        }
        specs.push_back(std::move(spec));
    }

    if (specs.empty()) {
        throw std::runtime_error("关键词文件里没有有效关键词：" + path);
    }
    return specs;
}

// ==================== 只读信息 ====================

// 把构造期与运行期读到的关键尺寸打成一包，供入口打印。
KwsDecoder::Info KwsDecoder::info() const {
    Info out;
    out.T = input_frame_num_;
    out.decode_chunk_len = t_shift_;
    out.context_size = context_size_;
    out.vocab_size = vocab_size_;
    out.blank_id = blank_id_;
    out.unk_id = unk_id_;
    out.n_mels = n_mels_;
    out.n_states = states_.size();
    out.n_keywords = static_cast<int>(graph_.num_keywords());
    out.max_active_paths = max_active_paths_;
    out.frame_shift_s = frame_shift_s_;
    return out;
}

}  // namespace audio_ai

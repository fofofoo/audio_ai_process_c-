/**
 * @file asr_decode.cpp
 * @brief 流式 ASR 解码实现：LogFBank 特征 → 三个 ONNX 会话 → 贪心搜索 → 文字。
 *
 * 本文件是 python_ref/src/asr_decode.py 的 C++ 移植，与 Python 版逐函数对应：
 *   run_encoder / run_decoder / run_joiner / accept_features /
 *   load_tokens / find_blank / detokenize / reset。
 * Python 里讲清「为什么」的段落这里同样保留，避免 C++ 版只剩代码。
 *
 * ============================================================
 * 一、三段模型的分工（streaming zipformer2 transducer）
 * ============================================================
 *   encoder —— 吃特征，吐出「声学编码」；它带着一大堆**状态张量**，跨块累积，
 *              这就是「流式」能记住前文的全部机制，主要负责猜字。
 *   decoder —— 吃最近若干个 token，吐出「语言编码」（也叫 predictor），
 *              2-token 轻量化 CNN（非长记忆的 LSTM），辅助去重、断句。
 *   joiner  —— 把 encoder 的一帧和 decoder 输出合起来，输出整个词表的打分。
 *
 * ============================================================
 * 二、流式 encoder：部分重叠滑动输入、encoder_embed 与 U-Net 的两段式时间压缩
 * ============================================================
 * 1. 滑动窗口：每次喂 T = 61 帧，但时间轴只前进 decode_chunk_len = 48 帧
 *    （480 ms）。多出的 13 帧是卷积模块必需的重叠上下文，用来把旧 13 帧与
 *    新 48 帧拼接后送入模型，避免边界补零失真。
 *
 * 2. encoder_embed：输入 (b, 61, 80) 先当作 (N, 1, T, 80) 的图像，经 3 层
 *    2D 卷积 1→8→32→128（3x3；只有第二层 conv.4 的 stride=2 作用于时间维，
 *    第三层 conv.7 只压频率），把频率维 80 压到 19；随后过 ConvNeXt 块
 *    （7x7 depthwise + 两个 1x1），最后由 out 投影把通道 128 升到 192。
 *    输出 (b, 27, 192)。
 *    **时间维在开头只压缩约 2 倍，不是 4 倍；通道也不是一步到 512。**
 *
 * 3. U-Net 内部：6 个 stack 的维度是 192/256/512/768/512/256，各自用 reshape
 *    合并/拆分相邻帧做时间下采样与上采样（先压缩再恢复），不同 stack 帧率不同
 *    （KV cache 的 T 为 256/128/64/32）。512 是第 3 个 stack 的宽度，也是
 *    encoder_out 的最终维度。整体时间压缩 4 倍 = encoder_embed 2 倍 + 内部再
 *    2 倍：48 帧输入对应 12 帧输出。这两段之后才进入 Transformer，由自注意力
 *    捕获长距离依赖，得到融合全局上下文的声学表征。
 *
 * ============================================================
 * 三、贪心搜索：decoder 基于上一个字的接龙预测，joiner 综合决策快速出字
 * ============================================================
 * 对 encoder 吐出的每一帧，跟当前 decoder 输出一起过 joiner，取 argmax：
 *   · 结果不是 blank：用最近 context_size 个 token 重算一次 decoder 输出，
 *     后续由这个选出的字继续预测下一个最可能的字，不断向后「词语接龙」；
 *   · 结果是 blank：**什么都不做**，沿用上一次的 decoder 输出。
 *   · argmax 是选字最快的方式，但模型直接抛弃其他高概率候选，没有纠错机会。
 *     未采用的束搜索会同时保留每一步最可能的 K 条「词语接龙」路径，把新词
 *     得分累加到历史总分，多次更新后错误路径被正确路径替代 —— 在保证速度
 *     的前提下避免贪心搜索的「一错到底」。
 *   · 绝大多数帧都是 blank，只有真出新 token 才跑一次 decoder 前向，这是
 *     贪心搜索省算力的关键。
 *
 * ============================================================
 * 四、跨块必须存活的状态（只有 reset() 才清零）
 * ============================================================
 *   1. encoder 的全部状态张量（实测 116 个）；
 *   2. recent_tokens_：最近 kRecentTokens 个已确认 token（有界，只服务显示）；
 *   3. 最近一次的 decoder 输出 decoder_out_。
 * 全量 token 不在这里保存，由调用方累积；本类对外只吐增量。
 *
 * ============================================================
 * 五、收尾：尾部补 100 帧特征域 0
 * ============================================================
 * 由前端 AudioPreprocess::input_finished() 在特征尾部补 100 帧 0 完成，本文件
 * 不需要额外处理。补的 0 是 log 域对应能量 1 的 0，不是静音 —— 目的是让最后
 * 不满一个 chunk 的尾段也能凑齐、被解码到，否则尾字会丢。
 *
 * 依赖：三个 Ort::Session 由调用方（入口）建好传进来；本文件不定位权重、
 * 不碰 wav 和前端，输入就是 AudioPreprocess 产出的特征。
 *
 * 解码循环的关键顺序（写错不会报错，只会多吐/少吐 token）：
 *   1. 特征缓冲 > T 才出块，取前 T 帧，时间轴前进 decode_chunk_len 帧；
 *   2. encoder 一次吃 T 帧，吐 nf 帧声学编码（运行时值，当前权重 12）；
 *   3. 对 nf 帧逐帧过 joiner，argmax 非 blank 才接进 recent_tokens_ 并刷新 decoder_out；
 *   4. blank 帧沿用旧 decoder_out，省掉绝大多数 decoder 前向。
 */

#include "audio_ai_process/asr_decode.h"
#include "audio_ai_process/text_utils.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace audio_ai {

// ==================== 构造 ====================

// 绑定三个会话，读取模型契约，建好状态模板与词表，最后 reset() 到初始态。
// 所有形状/尺寸都从模型本身读，换权重目录不用改代码。
AsrDecoder::AsrDecoder(Ort::Session& encoder, Ort::Session& decoder,
                       Ort::Session& joiner, const std::string& tokens_path,
                       int sample_rate)
    : encoder_(encoder),
      decoder_(decoder),
      joiner_(joiner),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      sample_rate_(sample_rate) {
    // —— 1. encoder 契约：模型类型、T、decode_chunk_len、n_mels ——
    // 直接从 encoder 的 custom metadata 读输入参数，不硬编码：
    // 当前使用的是 sherpa-onnx 的 zipformer2 流式 asr-encoder，只有 model_type
    // 对得上，下面的状态张量与贪心循环才成立。
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

    // encoder 第一个输入是 (batch, T, n_mels)，如 (1, 61, 80)；
    // 最后一维是 Mel 通道数，当前权重为静态的 80。
    {
        const auto shape = encoder_.GetInputTypeInfo(0)
                               .GetTensorTypeAndShapeInfo()
                               .GetShape();
        if (!shape.empty() && shape.back() > 0) {
            n_mels_ = static_cast<int>(shape.back());
        }
    }

    // encoder 输出最后一维 = joiner 的输入宽度（当前权重 512）。
    // 若导出时该维是动态的，再退回 joiner 第一个输入的最后一维；两者都拿不到才报错。
    {
        const auto shape = encoder_.GetOutputTypeInfo(0)
                               .GetTensorTypeAndShapeInfo()
                               .GetShape();
        if (!shape.empty() && shape.back() > 0) {
            encoder_dim_ = static_cast<int>(shape.back());
        }
    }

    // —— 2. decoder 契约：context_size / vocab_size / 输入类型 ——
    // decoder 内置 Embedding 字典，通过查表把 token id 转成隐藏层向量；
    // vocab_size 是内置词表大小，外部 tokens.txt 的行数与顺序必须与之严格对应。
    // context_size 是每次接龙喂给 decoder 的 token 个数（当前权重 2）。
    context_size_ = static_cast<int>(require_int(
        meta_value(decoder_, "context_size", allocator_), "context_size"));
    vocab_size_ = static_cast<int>(
        require_int(meta_value(decoder_, "vocab_size", allocator_), "vocab_size"));
    dec_input_type_ = decoder_.GetInputTypeInfo(0)
                          .GetTensorTypeAndShapeInfo()
                          .GetElementType();
    if (dec_input_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
        dec_input_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        throw std::runtime_error("decoder 输入不是 int64/int32");
    }

    // —— 3. 输入输出名字（存储在本对象里，指针指向这些 string）——
    // encoder 是多输入（特征 + 116 个状态）多输出（编码 + 新状态），
    // 逐个取名字存成 string，再建 const char* 数组供 Run 使用。
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

    // decoder 是单输入（最近 context_size 个 token）单输出（语言编码）。
    {
        auto name = decoder_.GetInputNameAllocated(0, allocator_);
        dec_in_name_ = name.get();
        auto out = decoder_.GetOutputNameAllocated(0, allocator_);
        dec_out_name_ = out.get();
    }
    // joiner 必须有两个输入：encoder 单帧声学编码 + decoder 输出。
    if (joiner_.GetInputCount() < 2) {
        throw std::runtime_error("joiner 输入少于 2 个");
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

    // —— 4. 状态张量模板：形状取自输入签名，动态维按 batch=1 填 1 ——
    // Zipformer2 是类 U-Net 的 transformer + cnn 混合网络：共 6 个 stack、
    // 19 个 block，时间压缩先压再恢复，不同 stack 的 T 帧长度不同。
    // encoder 规定了 512 帧（初始状态）的记忆特征，一帧 10 ms，总计约 5.12 秒
    // 历史；折到各 stack 的帧数是 256/128/64/32。
    //
    // 第二类输入（当前权重共 116 个状态）主要含三类：
    //   1. 2 个全局状态：processed_lens (1,) int64、embed_states (1,128,3,19) float；
    //   2. 76 个 KV cache 状态：19 block × 每层 4 个（key / value / value2 /
    //      non-linear attn），形状 (T=256/128/64/32, batch=1,
    //      key_dim=128/256 或 value_dim=48/96)；
    //   3. 38 个 CNN 边界状态：19 block × 每层 2 个（conv1 / conv2）。流式架构
    //      在边界不 padding，而是直接拿历史数据做因果卷积，因此输入特征左侧
    //      要拼接历史；历史长度 = kernel_size / 2（15 或 7），通道数
    //      192/256/512/768。
    //
    // 初值形状由 ONNX 输入签名推导：Python 版把字符串维（动态维）一律取 1，
    // 这里对 shape 里所有 < 0 的维做同样处理，换模型不用改代码。
    if (encoder_.GetInputCount() < 1) {
        throw std::runtime_error("encoder 没有输入");
    }
    // 输入 0 是特征 (1, T, n_mels)，第 1 个起全是状态张量，逐个读签名。
    for (size_t i = 1; i < encoder_.GetInputCount(); ++i) {
        // 形状 + 元素类型都来自 ONNX 输入签名，不写死。
        const auto type_info = encoder_.GetInputTypeInfo(i);
        const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = tensor_info.GetShape();
        // 动态维（字符串维）按 batch=1 填 1，与 Python 版一致。
        for (int64_t& dim : shape) {
            if (dim < 0) {
                dim = 1;  // Python 版：字符串维（动态维）一律取 1
            }
        }
        // 状态初值只支持这三种类型；其他类型说明模型不兼容，直接报错。
        const auto type = tensor_info.GetElementType();
        if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
            type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
            type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            throw std::runtime_error("encoder 状态张量类型未支持");
        }
        state_proto_.push_back({shape, type});
    }

    // —— 5. 词表与 blank ——
    // tokens.txt 的行数与顺序必须与 decoder 内置词表严格对应；
    // blank 的 id 从词表查，不硬编码。
    load_tokens(tokens_path);
    blank_id_ = static_cast<int>(find_blank());

    // —— 6. 初始状态：零状态 + context_size 个 blank + 第一次 decoder 前向 ——
    reset();
}

// 读取 custom metadata 的单个键；不存在返回空串。
std::string AsrDecoder::meta_value(
    const Ort::Session& session, const char* key,
    Ort::AllocatorWithDefaultOptions& allocator) {
    auto metadata = session.GetModelMetadata();
    auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
    // 键不存在返回空串，由 require_int 统一判断并抛出带键名的错误。
    return value ? std::string(value.get()) : std::string();
}

// 把 metadata 字符串转成整数：空值、非数字、带尾随字符都直接抛，
// 避免带着错参数继续构造。std::stoll 只要求前缀合法，所以必须检查 used。
int64_t AsrDecoder::require_int(const std::string& value, const char* key) {
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

// ==================== 状态重置 ====================

// 清空全部跨块状态，回到「刚开始听」：
//   状态张量按构造期保存的形状/类型重新分配并清零；recent_tokens_ 清空；
//   decoder_out 用全 blank 占位重算；特征缓冲、尾部文本与统计清零。
// 模型契约、词表都不动，reset 只影响流式状态。
void AsrDecoder::reset() {
    // 状态张量按构造期模板重新分配，并全部填 0（Python 的 np.zeros）。
    states_.clear();
    states_.reserve(state_proto_.size());
    for (const StateProto& proto : state_proto_) {
        // 按模板类型分别分配零张量。CreateTensor<T> 是模板，T 必须编译期
        // 确定，所以 float / int64 / int32 各写一遍。
        Ort::Value value(nullptr);
        if (proto.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            value = Ort::Value::CreateTensor<float>(
                allocator_, proto.shape.data(), proto.shape.size());
            std::fill_n(value.GetTensorMutableData<float>(),
                        static_cast<size_t>(std::accumulate(
                            proto.shape.begin(), proto.shape.end(), int64_t{1},
                            std::multiplies<int64_t>())),
                        0.0f);
        } else if (proto.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            value = Ort::Value::CreateTensor<int64_t>(
                allocator_, proto.shape.data(), proto.shape.size());
            std::fill_n(value.GetTensorMutableData<int64_t>(),
                        static_cast<size_t>(std::accumulate(
                            proto.shape.begin(), proto.shape.end(), int64_t{1},
                            std::multiplies<int64_t>())),
                        int64_t{0});
        } else {
            value = Ort::Value::CreateTensor<int32_t>(
                allocator_, proto.shape.data(), proto.shape.size());
            std::fill_n(value.GetTensorMutableData<int32_t>(),
                        static_cast<size_t>(std::accumulate(
                            proto.shape.begin(), proto.shape.end(), int64_t{1},
                            std::multiplies<int64_t>())),
                        int32_t{0});
        }
        states_.push_back(std::move(value));
    }

    // 最近 token 为空；run_decoder 会按 context_size 补满 blank 占位，得到
    // 「还没有听到任何字」时的接龙预测（结果与旧版 hyp_=[blk,blk] 等价）。
    recent_tokens_.clear();
    recent_text_.clear();
    decoder_out_ = run_decoder(recent_tokens_);

    // 特征缓冲、统计与尾部文本全部清零。
    feat_buf_.clear();
    n_buffered_ = 0;
    n_chunks_ = 0;
    n_enc_frames_ = 0;
}

// ==================== 三个会话的低层调用 ====================

// 喂 (1, T, n_mels) 特征块，返回 encoder_out；新状态就地写回 states_。
// 模型要求两类输入：第一项是本次音频数据，往后是上一次推理保留的特征状态；
// 输出第 0 项是编码结果，第 1 项起是更新后的状态，必须原样存回去 ——
// 这就是流式能记住前文的全部机制。
Ort::Value AsrDecoder::run_encoder(Ort::Value x) {
    // feed：模型要求的输入，第一项是本次音频数据，往后是上一次推理保留的
    // 特征状态记忆 states_。Run 需要连续的 Ort::Value 数组，而 Value 只能
    // 移动不能拷贝：先把 x 与旧状态整体移进 inputs。
    std::vector<Ort::Value> inputs;
    inputs.reserve(1 + states_.size());
    inputs.push_back(std::move(x));
    for (Ort::Value& state : states_) {
        inputs.push_back(std::move(state));
    }
    states_.clear();

    // 输入新音频 + 旧记忆，输出 高级时序声学表征 + 新记忆。
    auto outputs = encoder_.Run(
        Ort::RunOptions{nullptr}, enc_in_ptrs_.data(), inputs.data(),
        inputs.size(), enc_out_ptrs_.data(), enc_out_ptrs_.size());

    if (outputs.size() != enc_out_names_.size() || outputs.empty()) {
        throw std::runtime_error("encoder 输出个数与模型签名不一致");
    }
    // 新的状态记忆留在 states_ 内部跨块存活；输出第 0 项（声学表征）返回。
    states_.reserve(outputs.size() - 1);
    for (size_t i = 1; i < outputs.size(); ++i) {
        states_.push_back(std::move(outputs[i]));
    }
    return std::move(outputs[0]);
}

// 喂最近 context_size 个 token，返回 decoder_out（predictor 输出，纯 float 拷贝）。
// decoder 负责「接龙」：判断是否重复，并预测下一个最可能的字对应的特征。
// 输入 dtype 跟随模型签名（int64 或 int32），不假设一定是 int64。
// 只保留数值、不保留 Ort::Value：joiner 每次要把两个输入拼成连续数组，
// 存 float 缓冲比管理一个跨调用存活的 Value 更简单，512 个数的拷贝可以忽略。
//
// recent 只存已确认 token，不含旧版 hyp_ 开头的 blank 占位，因此这里负责补：
//   recent = []        -> [blk, blk]
//   recent = [t1]      -> [blk, t1]
//   recent = [t1, t2]  -> [t1, t2]
// 结果与旧版「hyp_ 开头两个 blank + 末尾切片」完全等价。
std::vector<float> AsrDecoder::run_decoder(const std::deque<int64_t>& recent) {
    // 先铺 context_size 个 blank，再把 recent 末尾最多 context_size 个 token
    // 覆盖到尾部；不足时开头保留 blank。
    std::vector<int64_t> last(static_cast<size_t>(context_size_),
                              static_cast<int64_t>(blank_id_));
    const size_t take =
        std::min(recent.size(), static_cast<size_t>(context_size_));
    std::copy(recent.end() - static_cast<ptrdiff_t>(take), recent.end(),
              last.end() - static_cast<ptrdiff_t>(take));

    // (1, context_size) 的 token id 张量，dtype 跟随 decoder 输入签名。
    const int64_t shape[2] = {1, static_cast<int64_t>(last.size())};
    std::vector<Ort::Value> inputs;
    if (dec_input_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        // 主流导出：int64 与 token id 元素类型相同，直接拷贝。
        Ort::Value y =
            Ort::Value::CreateTensor<int64_t>(allocator_, shape, 2);
        std::copy(last.begin(), last.end(), y.GetTensorMutableData<int64_t>());
        inputs.push_back(std::move(y));
    } else {
        // 少数导出用 int32：逐元素窄化转换。token id 远小于 2^31，安全。
        Ort::Value y =
            Ort::Value::CreateTensor<int32_t>(allocator_, shape, 2);
        int32_t* dst = y.GetTensorMutableData<int32_t>();
        for (size_t i = 0; i < last.size(); ++i) {
            dst[i] = static_cast<int32_t>(last[i]);
        }
        inputs.push_back(std::move(y));
    }

    // decoder 前向：判断是否重复，并预测下一个最可能的字对应的特征。
    const char* in_names[] = {dec_in_name_.c_str()};
    const char* out_names[] = {dec_out_name_.c_str()};
    auto outputs = decoder_.Run(Ort::RunOptions{nullptr}, in_names,
                                inputs.data(), inputs.size(), out_names, 1);

    // 只取数值，后面 joiner 每次都要把两个输入重新拼成连续数组。
    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    const size_t count = info.GetElementCount();
    const float* data = outputs[0].GetTensorData<float>();
    return std::vector<float>(data, data + count);
}

// encoder 的单帧 (1, encoder_dim) + decoder_out → 整个词表的打分 (1, vocab)。
// encoder 一次吃多帧，decoder / joiner 则是逐帧输入：这里只取 encoder 输出的
// 第 k 帧，与上一次出字时的 decoder_out 拼成 joiner 的两个输入。
Ort::Value AsrDecoder::run_joiner(const float* encoder_frame, size_t frame_dim,
                                  const std::vector<float>& decoder_out) {
    // joiner 输入 0：当前帧的声学表征 (1, encoder_dim)。
    const int64_t enc_shape[2] = {1, static_cast<int64_t>(frame_dim)};
    Ort::Value frame =
        Ort::Value::CreateTensor<float>(allocator_, enc_shape, 2);
    std::memcpy(frame.GetTensorMutableData<float>(), encoder_frame,
                frame_dim * sizeof(float));

    // joiner 输入 1：上一次出字时的 decoder 预测 (1, decoder_dim)。
    const int64_t dec_shape[2] = {
        1, static_cast<int64_t>(decoder_out.size())};
    Ort::Value dec =
        Ort::Value::CreateTensor<float>(allocator_, dec_shape, 2);
    std::memcpy(dec.GetTensorMutableData<float>(), decoder_out.data(),
                decoder_out.size() * sizeof(float));

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(frame));
    inputs.push_back(std::move(dec));

    // 联合网络输出整个词表的打分 (1, vocab_size)，调用方对每帧取 argmax。
    const char* in_names[] = {joi_in_name_0_.c_str(), joi_in_name_1_.c_str()};
    const char* out_names[] = {joi_out_name_.c_str()};
    auto outputs = joiner_.Run(Ort::RunOptions{nullptr}, in_names,
                               inputs.data(), inputs.size(), out_names, 1);
    return std::move(outputs[0]);
}

// ==================== 一体化接口 ====================

// 喂一块特征：追加进缓冲，按 `buffered > T` 排空能解的块。
// 每个块：取前 T 帧 → encoder 一次 → 时间轴前进 T_shift 帧 → 逐帧 joiner
// → 非 blank 才接进 recent_tokens_ 并刷新 decoder。
// 只返回「本次新增」的 token；整段历史由调用方累积。本轮有新 token 时
// 重算一次 recent_text_（只处理最近 kRecentTokens 个，代价与总时长无关）。
std::vector<int64_t> AsrDecoder::accept_features(const LogFbankFeatures& feats) {
    // 先校验形状：维度不对或 data 长度与 n_frames × n_mels 不符时越界读取
    // 不会报错，只会静默算出乱码，所以这里直接抛。
    if (feats.n_mels != n_mels_) {
        throw std::invalid_argument(
            "AsrDecoder: feats.n_mels 与模型输入维度不一致");
    }
    if (feats.data.size() !=
        static_cast<size_t>(feats.n_frames) * static_cast<size_t>(feats.n_mels)) {
        throw std::invalid_argument(
            "AsrDecoder: feats.data 长度与 n_frames * n_mels 不一致");
    }

    // 输入特征非空才加入缓冲区。缓冲区为空时 insert 等价于 Python 的
    // 「直接赋值」（首次输入 / 刚 reset），非空时等价于 np.concatenate。
    // 0 帧的输入跳过这里，仍会走到下面的排空逻辑。
    if (feats.n_frames > 0) {
        feat_buf_.insert(feat_buf_.end(), feats.data.begin(), feats.data.end());
        n_buffered_ += feats.n_frames;
    }

    // 本次调用新确认的 token；按时间顺序返回给调用方。
    std::vector<int64_t> new_ids;

    // 等待缓存帧数大于模型要求的输入帧数 T（当前 61）才出块。
    // 判据必须是严格大于：若写成 >=，批量条件恰好为 48k + T 时会多喂一次。
    while (n_buffered_ > input_frame_num_) {
        // 1) 每次从缓冲区最前面取 T 帧给 encoder，组成 (1, T, n_mels)。
        //    C++ 的 feat_buf_ 本身就是连续内存，对应 Python 的
        //    np.ascontiguousarray。
        const int64_t x_shape[3] = {1, input_frame_num_, n_mels_};
        Ort::Value x = Ort::Value::CreateTensor<float>(allocator_, x_shape, 3);
        const size_t x_count =
            static_cast<size_t>(input_frame_num_) * static_cast<size_t>(n_mels_);
        std::memcpy(x.GetTensorMutableData<float>(), feat_buf_.data(),
                    x_count * sizeof(float));

        // 2) encoder 网络推理：输入新的 T 帧音频，输出高级时序声学表征
        //    (1, nf, encoder_dim)。状态在 states_ 里累积（喂入旧记忆，吐出新记忆）。
        //    nf 是非重叠的 T_shift 帧经 4 倍时间压缩后的帧数（当前 48 → 12），
        //    必须取运行时 shape，写死 T 会错。
        Ort::Value enc_out = run_encoder(std::move(x));
        const auto enc_shape =
            enc_out.GetTensorTypeAndShapeInfo().GetShape();
        if (enc_shape.size() != 3) {
            throw std::runtime_error("encoder_out 不是三维张量");
        }
        const int nf = static_cast<int>(enc_shape[1]);
        const int dim = static_cast<int>(enc_shape[2]);
        n_enc_frames_ += nf;  // 已经编码的帧数
        ++n_chunks_;          // 累计推理输入的块数

        // 3) 时间轴前进 T_shift 帧，丢掉最老的帧；剩下的 T − T_shift 帧是
        //    前后两次输入共同的重叠帧，留给下一块做卷积上下文。
        const size_t drop =
            static_cast<size_t>(t_shift_) * static_cast<size_t>(n_mels_);
        feat_buf_.erase(feat_buf_.begin(), feat_buf_.begin() + drop);
        n_buffered_ -= t_shift_;

        // 4) 逐帧提取非重叠的音频压缩编码，送进预测网络 decoder 和联合网络
        //    joiner。encoder 是一块多帧，decoder / joiner 都是单帧输入，
        //    所以必须按帧顺序串行喂。
        const float* enc_data = enc_out.GetTensorData<float>();
        for (int k = 0; k < nf; ++k) {
            // joiner 根据「当前声学表征帧 + 上一次出字时 decoder 预测的下一个字」
            // 输出这一帧对应的文字 id（整个词表的打分取 argmax）。
            Ort::Value logits =
                run_joiner(enc_data + static_cast<size_t>(k) * dim, dim,
                           decoder_out_);
            const auto logit_shape =
                logits.GetTensorTypeAndShapeInfo().GetShape();
            const int64_t n_vocab = logit_shape.back();
            const float* scores = logits.GetTensorData<float>();

            // argmax：贪心搜索直接把概率最高的 token 作为本帧结果。
            int64_t best = 0;
            float best_score = scores[0];
            for (int64_t j = 1; j < n_vocab; ++j) {
                if (scores[j] > best_score) {
                    best_score = scores[j];
                    best = j;
                }
            }

            if (best != blank_id_) {
                // 预测字不是空白：接进 recent_tokens_（超过显示窗口就丢最老），
                // 并用最近 context_size 个 token 重算 decoder，判断是否重复并
                // 预测下一个字，继续向后接龙。
                // blank：什么都不做，沿用上一次的 decoder_out，省掉绝大多数
                // decoder 前向 —— 这是贪心搜索能跑得快的关键。
                recent_tokens_.push_back(best);
                if (recent_tokens_.size() > kRecentTokens) {
                    recent_tokens_.pop_front();
                }
                new_ids.push_back(best);
                decoder_out_ = run_decoder(recent_tokens_);
            }
        }
    }

    // 只有本轮吐出过新 token 才重算尾部文本：没有新字时与上次相同，不必
    // 重复做 token 查表和 detokenize。窗口固定为 kRecentTokens，代价有界。
    if (!new_ids.empty()) {
        const std::vector<int64_t> window(recent_tokens_.begin(),
                                          recent_tokens_.end());
        recent_text_ = text_of(window);
    }

    return new_ids;
}

// ==================== 词表与文本还原 ====================

// 读 tokens.txt，每行 `token id`。用最后一个空格切分（rpartition 口径）：
// BPE token 本身不含空格，遇到多余空白也不会切错 id。
void AsrDecoder::load_tokens(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("找不到词表：" + path);
    }

    id2token_.clear();
    std::string line;
    while (std::getline(in, line)) {
        // 兼容 Windows 换行；空行直接跳过。
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        // rpartition 口径：按最后一个空格切成 token 与 id 两段。
        const size_t pos = line.find_last_of(' ');
        if (pos == std::string::npos) {
            continue;
        }
        const std::string token = line.substr(0, pos);
        const std::string idx = line.substr(pos + 1);
        try {
            id2token_[std::stoll(idx)] = token;
        } catch (const std::exception&) {
            continue;  // 坏行跳过，与 Python 版只对格式异常做容错的口径一致
        }
    }

    if (id2token_.empty()) {
        throw std::runtime_error("词表为空：" + path);
    }
}

// blank 的 id 由模型词表决定：查 <blk> / <blank>，找不到才退回 0。
// 不要硬编码 0 —— 换个词表，blank 的 id 就可能变。
int64_t AsrDecoder::find_blank() const {
    // 遍历词表找 <blk> 或 <blank>，找到立刻返回它的 id。
    for (const auto& kv : id2token_) {
        if (kv.second == "<blk>" || kv.second == "<blank>") {
            return kv.first;
        }
    }
    // 找不到不致命：退回 0 并提示，避免整个解码流程直接失败。
    std::fprintf(stderr,
                 "警告：tokens.txt 里没找到 <blk>/<blank>，退回 blank_id=0\n");
    return 0;
}

// 把 BPE token 序列拼成可读文本（后处理）。
// 不能只做字符串 join，原因有两层：
//   1. ▁（U+2581）是 BPE 的词首标记，得还原成空格，否则会得到
//      「▁昨▁天▁是」；
//   2. 本模型把中文按单字切（▁礼▁拜▁二），单把 ▁ 换成空格会散成
//      「礼 拜 二」，所以还要去空格，但只能去特定位置：
//        · CJK 之间            去     「昨 天」→「昨天」
//        · 全角标点两侧        去     「Monday ，today」→「Monday，today」
//        · CJK 与 ASCII 之间   保留   「是 Monday」不能粘成「是Monday」
std::string AsrDecoder::detokenize(const std::vector<std::string>& tokens) const {
    // 先把 token 序列拼成完整字符串（对应 Python 的 "".join(tokens)）。
    std::string joined;
    for (const std::string& token : tokens) {
        joined += token;
    }

    // 按 UTF-8 解码成码点，再把 ▁ 统一替换成 ASCII 空格。
    std::vector<uint32_t> cps = utf8_decode(joined);
    for (uint32_t& cp : cps) {
        if (cp == 0x2581) {  // ▁
            cp = ' ';
        }
    }

    std::vector<uint32_t> out;
    out.reserve(cps.size());
    for (size_t i = 0; i < cps.size();) {
        if (is_space(cps[i])) {
            size_t j = i;
            while (j < cps.size() && is_space(cps[j])) {
                ++j;
            }

            const bool prev_cjk = !out.empty() && is_cjk(out.back());
            const bool prev_full = !out.empty() && is_fullwidth(out.back());
            const bool next_cjk = j < cps.size() && is_cjk(cps[j]);
            const bool next_full = j < cps.size() && is_fullwidth(cps[j]);

            // 对应 Python 的三条 re.sub：
            //   (?<=CJK)\s+(?=CJK) / \s+(?=FULL) / (?<=FULL)\s+
            if (!((prev_cjk && next_cjk) || next_full || prev_full)) {
                out.push_back(' ');
            }
            i = j;
        } else {
            out.push_back(cps[i]);
            ++i;
        }
    }

    // 去掉首尾空白（Python 的 text.strip()）。
    size_t begin = 0;
    while (begin < out.size() && is_space(out[begin])) {
        ++begin;
    }
    size_t end = out.size();
    while (end > begin && is_space(out[end - 1])) {
        --end;
    }

    // 去掉首尾空白后，把码点重新编码回 UTF-8 字符串。
    std::string text;
    text.reserve(joined.size());
    for (size_t i = begin; i < end; ++i) {
        utf8_append(text, out[i]);
    }
    return text;
}

// ==================== 对外文本接口 ====================

// 把 token id 序列逐个查表转成 token 字符串。查不到的 id 用 "?" 兜底：
// 词表外的未知 id 不应让整个解码崩溃。
std::vector<std::string> AsrDecoder::token_strings_of(
    const std::vector<int64_t>& ids) const {
    std::vector<std::string> token_strs;
    token_strs.reserve(ids.size());
    for (int64_t id : ids) {
        const auto it = id2token_.find(id);
        token_strs.push_back(it != id2token_.end() ? it->second : "?");
    }
    return token_strs;
}

// 对任意一段 token 序列做完整 detokenize：查表 → 拼字符串 → ▁ 与中英文
// 空格规则。段结算、最终文本、token 导出都走这里，保证与旧版 text_ 逐字节一致。
std::string AsrDecoder::text_of(const std::vector<int64_t>& ids) const {
    return detokenize(token_strings_of(ids));
}

// ==================== 只读信息 ====================

// 把构造期读到的关键尺寸打成一包，供入口打印与上层判断。
AsrDecoder::Info AsrDecoder::info() const {
    Info out;
    out.T = input_frame_num_;
    out.decode_chunk_len = t_shift_;
    out.context_size = context_size_;
    out.vocab_size = vocab_size_;
    out.blank_id = blank_id_;
    out.n_mels = n_mels_;
    out.n_states = states_.size();
    return out;
}

}  // namespace audio_ai

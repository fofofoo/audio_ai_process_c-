#pragma once

/**
 * @file asr_decode.h
 * @brief 流式 ASR 解码：LogFBank 特征 → 文字（C++ 版，对齐 src/asr_decode.py）。
 *
 * 职责边界：
 *   - 只负责「特征 → encoder → 逐帧 joiner 贪心搜索 → decoder 接龙 → 文本」；
 *   - 三个 Ort::Session 由调用方建好传进来：本类不定位权重、不决定跑 CPU 还是 CUDA；
 *   - 不碰 wav、不做预处理、不画图；输入是 AudioPreprocess 产出的特征。
 *
 * 三段模型的分工（streaming zipformer2 transducer）：
 *   encoder —— 吃 (1, T, n_mels) 特征块，输出声学编码 + 116 个更新后的状态；
 *   decoder —— 吃最近 context_size 个 token，输出语言编码（predictor）；
 *   joiner  —— encoder 的单帧 + decoder 输出 → 整个词表的打分，取 argmax。
 *
 * 流式状态（跨块存活，只有 reset() 清零）：
 *   1. encoder 的全部状态张量（形状/类型从 ONNX 输入签名推导，动态维填 1）；
 *   2. hyp：已确认的 token 序列（开头含 context_size 个 blank 占位）；
 *   3. decoder_out：最近一次 decoder 输出，blank 帧不刷新。
 *
 * 与 Python 版逐点对应的不变量：
 *   - 出块判据是 `buffered_frames > T`（严格大于），每块前进 decode_chunk_len 帧；
 *   - encoder 输出帧数取运行时 shape，不写死 T；
 *   - joiner argmax 是 blank 时什么都不做，沿用上一次 decoder_out；
 *   - blank_id 从 tokens.txt 的 <blk> / <blank> 查，不硬编码 0；
 *   - 尾部补 100 帧特征域 0 由调用方通过 AudioPreprocess::input_finished() 完成，
 *     本类只把它当普通特征吃掉。
 *
 * 线程模型：非线程安全，所有接口应在同一线程调用。
 */

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "audio_ai_process/audio_preprocess.h"

namespace audio_ai {

class AsrDecoder {
public:
    /// 入口打印用的关键尺寸。
    struct Info {
        int T = 0;                ///< encoder 输入帧数（当前权重 61）
        int decode_chunk_len = 0; ///< 每块时间轴前进帧数（当前权重 48）
        int context_size = 0;     ///< decoder 输入 token 个数（当前权重 2）
        int vocab_size = 0;       ///< 词表大小（当前权重 5000）
        int blank_id = 0;         ///< blank 的 token id（从 tokens.txt 查）
        int n_mels = 80;          ///< 特征维度
        size_t n_states = 0;      ///< encoder 状态张量个数（当前权重 116）
    };

    /**
     * @brief 绑定三个现成的会话并读取模型契约。
     * @param encoder     encoder 会话（调用方持有，生命周期必须长于本对象）
     * @param decoder     decoder 会话
     * @param joiner      joiner 会话
     * @param tokens_path tokens.txt 路径，每行格式 `token id`
     * @param sample_rate 采样率，仅记录到 info（模型固定 16 kHz）
     * @throws std::runtime_error 模型类型/元数据/词表不符合预期
     */
    AsrDecoder(Ort::Session& encoder, Ort::Session& decoder, Ort::Session& joiner,
               const std::string& tokens_path, int sample_rate = 16000);

    /// 清空全部跨块状态，回到「刚开始听」；模型契约与词表保持不变。
    void reset();

    /**
     * @brief 喂一块特征，把能解码的都解掉。
     * @param feats 行主序特征 (n_frames, n_mels)，允许 0 帧
     * @return 当前完整识别文本；没有新 token 时与上次相同
     */
    std::string accept_features(const LogFbankFeatures& feats);

    /// 当前识别文本。
    const std::string& text() const { return text_; }

    /// 已确认的 token id，**不含**开头那 context_size 个占位 blank。
    std::vector<int64_t> token_ids() const;

    Info info() const;

    int n_chunks() const { return n_chunks_; }        ///< 喂过几次 encoder
    int64_t n_enc_frames() const { return n_enc_frames_; }  ///< encoder 累计输出帧数
    int buffered_frames() const { return n_buffered_; }     ///< 当前待解码特征帧数

private:
    // —— 三个会话的低层调用 ——
    Ort::Value run_encoder(Ort::Value x);
    std::vector<float> run_decoder(const std::vector<int64_t>& hyp);
    Ort::Value run_joiner(const float* encoder_frame, size_t frame_dim,
                          const std::vector<float>& decoder_out);

    // —— 词表与文本还原 ——
    void load_tokens(const std::string& path);
    int64_t find_blank() const;
    std::string detokenize(const std::vector<std::string>& tokens) const;

    // —— 构造期：读 metadata / 输入签名 ——
    static std::string meta_value(const Ort::Session& session,
                                  const char* key,
                                  Ort::AllocatorWithDefaultOptions& allocator);
    /// 把 metadata 里的字符串转成整数；缺失或格式不合法的直接抛异常。
    /// 这是模型契约专用逻辑，留在类内；UTF-8 / 字符分类工具见 text_utils.h
    /// （该文件同时收纳路径等通用小工具）。
    static int64_t require_int(const std::string& value, const char* key);

private:
    // 三个会话按引用持有：生命周期由入口保证
    Ort::Session& encoder_;
    Ort::Session& decoder_;
    Ort::Session& joiner_;
    Ort::MemoryInfo memory_info_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;

    // 模型契约（构造期读取，运行期不变）
    int sample_rate_ = 16000;
    int n_mels_ = 80;
    int input_frame_num_ = 0;   ///< T
    int t_shift_ = 0;           ///< decode_chunk_len
    int context_size_ = 0;
    int vocab_size_ = 0;
    int blank_id_ = 0;
    int encoder_dim_ = 0;       ///< encoder_out 最后一维（joiner 输入宽度）

    // 输入输出名字与指针（指针指向上面的 string 存储，构造期一次建好）
    std::vector<std::string> enc_in_names_;
    std::vector<std::string> enc_out_names_;
    std::vector<const char*> enc_in_ptrs_;
    std::vector<const char*> enc_out_ptrs_;
    std::string dec_in_name_;
    std::string dec_out_name_;
    std::string joi_in_name_0_;
    std::string joi_in_name_1_;
    std::string joi_out_name_;
    ONNXTensorElementDataType dec_input_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;

    /// encoder 状态张量的形状与类型模板；动态维按 batch=1 取 1。
    struct StateProto {
        std::vector<int64_t> shape;
        ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    };
    std::vector<StateProto> state_proto_;

    // —— 流式状态 ——
    std::vector<Ort::Value> states_;
    std::vector<int64_t> hyp_;
    std::vector<float> decoder_out_;   ///< 最近一次 decoder 输出 (1, 512)
    std::vector<float> feat_buf_;   ///< 行主序待解码特征
    int n_buffered_ = 0;            ///< feat_buf_ 里的帧数
    int n_chunks_ = 0;
    int64_t n_enc_frames_ = 0;
    std::string text_;
    std::map<int64_t, std::string> id2token_;
};

}  // namespace audio_ai

#pragma once

/**
 * @file kws_decode.h
 * @brief 流式关键词唤醒：LogFBank 特征 → 命中关键词（C++ 版，对齐 src/kws_decode.py）。
 *
 * 职责边界：
 *   - 只负责「特征 → encoder → 逐帧束搜索（trie 约束）→ 触发判定」；
 *   - 三个 Ort::Session 由调用方建好传进来：本类不定位权重、不决定跑 CPU 还是 CUDA；
 *   - 不碰 wav、不做预处理、不画图；输入是 AudioPreprocess 产出的特征；
 *   - 关键词前缀树由 KwsContextGraph 独立实现，本类**按值持有**它，不复制树逻辑。
 *
 * 与 AsrDecoder 的分工：
 *   ASR —— 5000 词表上的贪心搜索，输出整句文本；
 *   KWS —— 263 个音素级单元上的束搜索，每条假设挂一个 trie 状态，只在走完整条
 *          关键词并攒够尾随 blank 帧后输出 KeywordHit。
 *
 * 流式状态（跨块存活，只有 reset() 清零）：
 *   1. encoder 的全部状态张量（形状/类型从 ONNX 输入签名推导，动态维填 1）；
 *   2. beam：≤ max_active_paths 条假设，含 token 历史、时间戳、声学概率、
 *      trie 状态与累计 log 概率；
 *   3. 未凑满一个 chunk 的特征尾巴；
 *   4. frame_offset：已解码的输出帧数，只用于时间戳。
 *
 * 与 Python 版逐点对应的不变量：
 *   - 出块判据是 `buffered_frames > T`（严格大于），每块前进 decode_chunk_len 帧；
 *   - blank_id 硬编码为 0，<unk> 也按 blank 处理；
 *   - 空白假设的 token 历史是 [-1]*(context_size-1) + [0]，-1 不 clamp；
 *   - 触发要求关键词后至少 num_trailing_blanks+1 个 blank 输出帧，且关键词各
 *     token 声学概率的算术平均 >= 该关键词的 ac_threshold；
 *   - 触发后只重置 beam，encoder 状态继续累积；
 *   - 连续静音超过 reset_after_silence 秒时软重置，保留未消费的特征尾巴。
 *
 * 线程模型：非线程安全，所有接口应在同一线程调用。
 */

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "audio_ai_process/audio_preprocess.h"
#include "audio_ai_process/kws_context_graph.h"

namespace audio_ai {

/**
 * @brief 一次关键词触发。
 *
 * tokens / timestamps 与关键词的 token 序列一一对应，timestamp 是**输出帧**
 * 换算后的秒数；start_time 取关键词首 token 的时间（Python 版口径，不是上游
 * C++ 「取结果那一刻」的时间）。
 */
struct KeywordHit {
    std::string keyword;              ///< 优先用 @原文，没有时用 token 拼接
    std::vector<std::string> tokens;  ///< 命中的 token 文本
    std::vector<float> timestamps;    ///< 每个 token 的时间（秒）
    float start_time = 0.0f;          ///< 首 token 时间（秒）
    float avg_prob = 0.0f;            ///< 关键词各 token 声学概率的算术平均
};

/**
 * @brief 被 ac_threshold 挡住的候选，调关键词 `#threshold` 的依据。
 *
 * 只在 debug=true 时记录；命中时 avg_prob 可以从 KeywordHit 拿到，真正缺的是
 * 「差一点」的分数。
 */
struct KwsBlockedEvent {
    std::string keyword;       ///< 关键词原文
    float avg_prob = 0.0f;     ///< 关键词各 token 声学概率的平均
    float threshold = 0.0f;    ///< 该关键词的 ac_threshold
    int level = 0;             ///< trie 深度（关键词 token 数）
    int64_t frame = 0;         ///< 当前块起点的输出帧号
};

/**
 * @brief 流式关键词唤醒器。
 *
 * 用法与 AsrDecoder 一致：
 *   KwsDecoder kws(encoder, decoder, joiner, tokens_path, keywords_path);
 *   std::vector<KeywordHit> hits = kws.accept_features(feats);
 *   kws.reset();
 *
 * 三个会话按引用持有，生命周期由调用方保证；本类不拥有它们。
 */
class KwsDecoder {
public:
    /// 入口打印用的关键尺寸。
    struct Info {
        int T = 0;                 ///< encoder 输入帧数（chunk-8 为 29）
        int decode_chunk_len = 0;  ///< 每块时间轴前进帧数（chunk-8 为 16）
        int context_size = 0;      ///< decoder 输入 token 个数（当前权重 2）
        int vocab_size = 0;        ///< 词表大小（当前权重 263）
        int blank_id = 0;          ///< blank 的 token id，固定为 0
        int unk_id = -1;           ///< <unk> 的 id，找不到为 -1
        int n_mels = 80;           ///< 特征维度
        size_t n_states = 0;       ///< encoder 状态张量个数（当前权重 38）
        int n_keywords = 0;        ///< 成功加载的关键词条数
        int max_active_paths = 4;  ///< 束宽
        float frame_shift_s = 0.04f;  ///< 每输出帧的时长（首次编码前为估计值）
    };

    /**
     * @brief 绑定三个现成的会话，读取模型契约，加载词表与关键词并建 trie。
     * @param encoder         encoder 会话（调用方持有，生命周期必须长于本对象）
     * @param decoder         decoder 会话
     * @param joiner          joiner 会话
     * @param tokens_path     tokens.txt 路径，每行格式 `token id`
     * @param keywords_path   关键词文件路径，格式 `token... [:boost] [#threshold] @原文`
     * @param max_active_paths 束宽，C++ 默认 4
     * @param num_trailing_blanks 触发前至少需要的 blank 输出帧数，默认 1
     * @param keywords_score     默认 boosting score（关键词行里 `:x` 优先）
     * @param keywords_threshold 默认触发阈值（关键词行里 `#x` 优先）
     * @param reset_after_silence 连续静音多少秒后软重置，默认 1.5
     * @param sample_rate     采样率，仅记录（模型固定 16 kHz）
     * @param debug           true 时记录被阈值挡住的候选，见 debug_events()
     * @throws std::runtime_error 模型类型/元数据/词表/关键词文件不符合预期
     */
    KwsDecoder(Ort::Session& encoder, Ort::Session& decoder, Ort::Session& joiner,
               const std::string& tokens_path, const std::string& keywords_path,
               int max_active_paths = 4, int num_trailing_blanks = 1,
               float keywords_score = 1.0f, float keywords_threshold = 0.25f,
               float reset_after_silence = 1.5f, int sample_rate = 16000,
               bool debug = false);

    // 三个会话与图都按引用/值长期持有，禁止拷贝。
    KwsDecoder(const KwsDecoder&) = delete;
    KwsDecoder& operator=(const KwsDecoder&) = delete;

    /// 清空全部跨块状态与统计，回到「刚开始监听」；模型契约与图保持不变。
    void reset();

    /**
     * @brief 喂一块特征，把能解码的都解掉。
     * @param feats 行主序特征 (n_frames, n_mels)，允许 0 帧
     * @return 本次新触发的关键词；没有命中返回空
     */
    std::vector<KeywordHit> accept_features(const LogFbankFeatures& feats);

    /// 关键词原文列表（没有 @原文 时用 token 拼接）。
    const std::vector<std::string>& keywords() const { return keywords_display_; }

    Info info() const;

    int n_chunks() const { return n_chunks_; }        ///< 喂过几次 encoder
    int64_t n_enc_frames() const { return n_enc_frames_; }  ///< encoder 累计输出帧数
    int n_hits() const { return n_hits_; }            ///< 累计触发次数
    int buffered_frames() const { return n_buffered_; }     ///< 当前待解码特征帧数
    int best_trie_level() const { return last_best_level_; }  ///< 最佳假设的 trie 深度

    /// 被阈值挡住的候选；只有 debug=true 时才累积，最多 2000 条。
    const std::vector<KwsBlockedEvent>& debug_events() const { return debug_events_; }

    /**
     * @brief 只读词表：每行按最后一个空格切成 token 与 id。
     * @param path tokens.txt 路径
     * @return {id: token}；空文件或坏 id 抛 std::runtime_error
     */
    static std::map<int32_t, std::string> load_tokens(const std::string& path);

    /**
     * @brief 只读关键词文件，解析成 KwsContextGraph 需要的 KeywordSpec。
     *
     * 对齐上游 `utils.cc: EncodeBase`：
     *   - 能查到 id 的词当作 token；
     *   - `:x` boosting score、`#x` 触发阈值、`@x` 原文；
     *   - 其余词一律报错并带行号（避免关键词悄悄少了几个 token）；
     *   - 整行 `#` 开头视为注释，没有 token 的行跳过。
     *
     * @param path     关键词文件路径
     * @param token2id tokens.txt 的逆表
     * @return 关键词列表；文件为空或没有有效关键词时抛 std::runtime_error
     */
    static std::vector<KeywordSpec> load_keywords(
        const std::string& path,
        const std::unordered_map<std::string, int32_t>& token2id);

private:
    /// 一条搜索路径：ys / timestamps / ys_probs 三者严格同步。
    struct Hypothesis {
        std::vector<int64_t> ys;          ///< 含开头 context_size 个 blank 占位
        std::vector<int64_t> timestamps;  ///< 输出帧号
        std::vector<float> ys_probs;      ///< 声学概率（非 log）
        float log_prob = 0.0f;            ///< 累计 log 概率
        const KwsContextState* context_state = nullptr;  ///< 指向 KwsContextGraph
        int num_trailing_blanks = 0;      ///< 末尾连续 blank 帧数
    };

    /// encoder 状态张量的形状与类型模板；动态维按 batch=1 取 1。
    struct StateProto {
        std::vector<int64_t> shape;
        ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    };

    // —— 状态重置 ——
    void reset_stream(bool keep_feat_buf);
    std::vector<Ort::Value> make_zero_states();

    // —— 三个会话的低层调用（与 AsrDecoder 同形，但 decoder / joiner 是 batch）——
    Ort::Value run_encoder(Ort::Value x);
    std::vector<float> run_decoder(const std::vector<int64_t>& ys,
                                   size_t n_hyps);
    std::vector<float> run_joiner(const float* encoder_frame, size_t n_hyps,
                                  size_t enc_dim,
                                  const std::vector<float>& decoder_out,
                                  size_t dec_dim);

    // —— 束搜索与触发 ——
    void decode_frames(const float* enc_data, int nf, int enc_dim,
                       std::vector<KeywordHit>* hits);
    static void merge_hypothesis(std::vector<Hypothesis>* hyps, Hypothesis hyp);
    void record_blocked(const KwsContextState* state, float avg_prob, int level);

    // —— 构造辅助 ——
    static std::unordered_map<std::string, int32_t> invert_token_map(
        const std::map<int32_t, std::string>& id2token);
    std::vector<int64_t> blank_ys() const;
    Hypothesis blank_hypothesis() const;

private:
    // 三个会话按引用持有：生命周期由入口保证
    Ort::Session& encoder_;
    Ort::Session& decoder_;
    Ort::Session& joiner_;
    Ort::MemoryInfo memory_info_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;

    // 词表、关键词与 trie。graph_ 依赖 keyword_specs_，声明顺序不能换。
    std::map<int32_t, std::string> id2token_;
    std::unordered_map<std::string, int32_t> token2id_;
    std::vector<KeywordSpec> keyword_specs_;
    KwsContextGraph graph_;

    // 模型契约与解码参数（构造期确定，运行期不变）
    int sample_rate_ = 16000;
    int n_mels_ = 80;
    int input_frame_num_ = 0;   ///< T
    int t_shift_ = 0;           ///< decode_chunk_len
    int context_size_ = 0;
    int vocab_size_ = 0;
    int blank_id_ = 0;
    int unk_id_ = -1;
    int encoder_dim_ = 0;       ///< encoder_out 最后一维
    int max_active_paths_ = 4;
    int num_trailing_blanks_ = 1;
    float keywords_score_ = 1.0f;
    float keywords_threshold_ = 0.25f;
    float reset_after_silence_ = 1.5f;
    bool debug_ = false;

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

    std::vector<StateProto> state_proto_;

    // —— 流式状态 ——
    std::vector<Ort::Value> states_;   ///< encoder 状态张量，跨块存活
    std::vector<Hypothesis> hyps_;     ///< 当前 beam；顺序 = 插入顺序
    std::vector<float> feat_buf_;      ///< 行主序待解码特征
    int n_buffered_ = 0;               ///< feat_buf_ 里的帧数
    int n_chunks_ = 0;
    int64_t n_enc_frames_ = 0;
    int n_hits_ = 0;
    int64_t frame_offset_ = 0;         ///< 已解码的输出帧数（时间戳基准）
    int best_trailing_blanks_ = 0;     ///< 上一块结束时最佳假设的尾随 blank 数
    int last_best_level_ = 0;          ///< 上一帧最佳假设的 trie 深度
    float frame_shift_s_ = 0.04f;      ///< 每个输出帧的秒数，按运行时 nf 校正

    // 展示与调试
    std::vector<std::string> keywords_display_;
    std::vector<KwsBlockedEvent> debug_events_;
    std::pair<std::string, int> debug_last_key_;  ///< (原文, 千分位分数) 去重
    bool debug_last_valid_ = false;
};

}  // namespace audio_ai

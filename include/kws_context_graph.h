#pragma once

/**
 * @file kws_context_graph.h
 * @brief 关键词前缀树：把若干条 token 序列组织成带 fail / output 弧的 trie。
 *
 * 职责边界：
 *   - 只负责「关键词集合 -> 可逐 token 步进的状态机」这一件事：建树、填
 *     fail/output 弧、按 token 走一步、判断是否走完某条关键词；
 *   - 不认识 tokens.txt / keywords.txt，也不碰 ORT：token id 由调用方查表后
 *     传进来，关键词文件解析和 onnx 推理分别在 B2 的加载器与解码器里；
 *   - 只依赖 C++ 标准库，无 ORT / OpenCV / HCNetSDK，可脱离权重单独编译与测试。
 *
 * 语义基准：
 *   逐行对照 sherpa-onnx 的 `csrc/context-graph.cc`（字段、分支、浮点比较全部
 *   保持一致），并以 `python_ref/src/kws_decode.py` 的 `_ContextGraph` 作为
 *   中间参照。上游唯一带断言的单测是 `context-graph-test.cc` 的 4 组用例，
 *   本模块的验收见 `tools/test_context_graph.cpp`。
 *
 * 关键口径（写错不会报错，只会静默改变加分）：
 *   - `KeywordSpec::score == 0` 表示「使用构造参数的 context_score」，不是真实
 *     的零分；`threshold == 0` 同理回落到 ac_threshold。这是上游 Build() 的
 *     `score == 0 ? context_score_ : score` 语义；
 *   - `output_score` 在 fill_fail_output() 里是 **+=**：节点自身分数在 build()
 *     时已写入，fail 链上 end 节点的奖励再叠加上去；
 *   - token 值 -1 只用于标记根节点，不参与任何数组下标运算，不需要 clamp；
 *   - strict_mode=false 只用于复刻上游单测的「命中即回根」分支，KWS 解码器
 *     始终使用默认的 strict_mode=true。
 *
 * 线程模型：构图完成后只读，可被多线程并发调用 forward_one_step / is_matched；
 * 构造过程非线程安全。图不可拷贝、不可移动：beam 里保存的是节点裸指针，
 * 地址必须与图同生命周期。
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace audio_ai {

/**
 * @brief 一条关键词的定义，由关键词文件解析器填写。
 *
 * 与上游 `ContextGraph` 的 6 个平行数组入参相比，这里把「同一条关键词」的
 * 4 项属性收进一个结构，避免下标错位。
 */
struct KeywordSpec {
    std::vector<int32_t> token_ids;  ///< token 序列（来自 tokens.txt）；不能为空
    float score = 0.0f;              ///< boosting score；0 = 用构造参数的 context_score
    float threshold = 0.0f;          ///< 触发阈值 [0,1]；0 = 用构造参数的 ac_threshold
    std::string phrase;              ///< `@` 后的原文，仅触发后展示，可为空
};

/**
 * @brief trie 节点，字段与上游 `ContextState` 一一对应。
 *
 * `next` 用节点式容器保存子节点，插入不会让已有节点地址失效，因此 beam 可以
 * 安全地长期持有 `const KwsContextState*`。
 */
struct KwsContextState {
    int32_t token = -1;          ///< 到达本节点消耗的 token；根节点为 -1
    float token_score = 0.0f;    ///< 这条边的 boosting score
    float node_score = 0.0f;     ///< 从根走到本节点的累计 score
    float output_score = 0.0f;   ///< 本节点或 fail 链上 end 节点的输出奖励
    int32_t level = 0;           ///< 本节点在所属关键词里的序号（1-based）
    float ac_threshold = 0.0f;   ///< 关键词触发阈值，仅 end 节点有效
    bool is_end = false;         ///< 是否是一条关键词的结尾
    std::string phrase;          ///< 关键词原文，仅 end 节点有效

    /// {token_id: 子节点}；本节点不拥有 fail / output 指向的节点
    std::unordered_map<int32_t, std::unique_ptr<KwsContextState>> next;
    const KwsContextState* fail = nullptr;    ///< Aho-Corasick 失配跳转
    const KwsContextState* output = nullptr;  ///< fail 链上最近的 end 节点

    KwsContextState() = default;

    /**
     * @brief 按上游构造函数语义初始化一个节点。
     * @param token        本节点对应的 token id（根节点为 -1）
     * @param token_score  入边的 boosting score
     * @param node_score   根到本节点的累计 score
     * @param output_score 输出奖励（建树时 end 节点先填 node_score，后续再叠加）
     * @param level        在关键词中的序号（1-based）
     * @param ac_threshold 触发阈值，仅 end 节点填写
     * @param is_end       是否关键词结尾
     * @param phrase       关键词原文，仅 end 节点填写
     */
    KwsContextState(int32_t token, float token_score, float node_score,
                    float output_score, int32_t level = 0,
                    float ac_threshold = 0.0f, bool is_end = false,
                    const std::string& phrase = std::string());

    // 节点的地址被父节点 next 里的 unique_ptr 和 beam 的裸指针同时引用，
    // 拷贝/移动会造成悬空指针，直接禁用。
    KwsContextState(const KwsContextState&) = delete;
    KwsContextState& operator=(const KwsContextState&) = delete;
};

/// forward_one_step() 的返回值，对应上游 std::tuple 的三个元素。
struct KwsStepResult {
    float context_score = 0.0f;                ///< 本步加分（strict 下含 output 奖励）
    const KwsContextState* state = nullptr;    ///< 走完这一步后的状态
    const KwsContextState* matched = nullptr;  ///< 本步命中的 end 节点，可为空
};

/**
 * @brief 关键词前缀树（Aho-Corasick 自动机），线程安全的只读查询。
 *
 * 构造时完成建树与 fail/output 弧填写；之后所有查询都是 const，不修改状态。
 * 调用方沿返回的 state 指针逐 token 前进，beam 里的每条假设各持一个指针。
 */
class KwsContextGraph {
public:
    /**
     * @brief 建树并填写 fail / output 弧。
     * @param keywords      关键词列表；空列表会构造出空树（num_keywords()==0），
     *                      是否视为配置错误由调用方决定
     * @param context_score 默认 boosting score，对应 keywords.txt 的 `:score`
     * @param ac_threshold  默认触发阈值，对应 keywords.txt 的 `#threshold`
     * @throw std::invalid_argument 某条关键词的 token_ids 为空时
     *
     * 单条关键词的 score/threshold 为 0 时回落到这两个默认值，是上游语义；
     * 不要把它理解成「不额外加分」或「阈值为 0」。
     */
    KwsContextGraph(const std::vector<KeywordSpec>& keywords,
                    float context_score = 1.0f,
                    float ac_threshold = 0.25f);

    // 图内的节点地址会被 beam 长期持有，禁止拷贝/移动。
    KwsContextGraph(const KwsContextGraph&) = delete;
    KwsContextGraph& operator=(const KwsContextGraph&) = delete;

    /**
     * @brief 沿 trie 走一步。
     * @param state       当前状态，必须以 root() 起始
     * @param token       下一个 token id
     * @param strict_mode true（默认）= 正常前缀匹配；false = 命中后立即回根，
     *                    只用于复刻上游非严格单测
     * @return 本步加分、下一步状态、本步命中的 end 节点（未命中为 nullptr）
     * @throw std::invalid_argument state 为空；std::logic_error fail 链损坏
     *
     * 失配时沿 fail 链回退，返回的 state 一定非空（最差回到根）。
     */
    KwsStepResult forward_one_step(const KwsContextState* state, int32_t token,
                                   bool strict_mode = true) const;

    /**
     * @brief 当前状态是否已经走完一条关键词。
     * @param state   当前状态
     * @param matched 输出参数：命中的 end 节点；不命中或 state 为空时置 nullptr。
     *                可为 nullptr，表示调用方只关心布尔结果
     * @return state 本身是 end，或 fail 链上最近节点是 end 时为 true
     */
    bool is_matched(const KwsContextState* state,
                    const KwsContextState** matched) const;

    /**
     * @brief 收尾：把当前路径的累计分退掉并回到根。
     * @param state 当前状态
     * @return {-state->node_score, root()}；只给上游单测对拍用，流式解码不用
     * @throw std::invalid_argument state 为空
     */
    std::pair<float, const KwsContextState*> finalize(
        const KwsContextState* state) const;

    /// 根节点；起点与 finalize() 的返回目标
    const KwsContextState* root() const { return root_.get(); }

    /// 成功插入的关键词条数（空 token_ids 会抛异常，不计入）
    size_t num_keywords() const { return num_keywords_; }

private:
    /// 把每条关键词按字符插入 trie，合并公共前缀并更新 is_end / output_score
    void build(const std::vector<KeywordSpec>& keywords);

    /// BFS 填写 fail 链与 output 弧；必须在 build() 之后调用
    void fill_fail_output();

private:
    float context_score_ = 1.0f;   ///< 默认 boosting score（来自构造参数）
    float ac_threshold_ = 0.25f;   ///< 默认触发阈值（来自构造参数）
    size_t num_keywords_ = 0;      ///< 已插入关键词数
    std::unique_ptr<KwsContextState> root_;  ///< 根节点，token == -1
};

}  // namespace audio_ai

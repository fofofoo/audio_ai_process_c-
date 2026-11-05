/**
 * @file kws_context_graph.cpp
 * @brief 关键词前缀树实现：建树、fail/output 弧、逐 token 步进。
 *
 * ============================================================
 * 这份代码是干什么的
 * ============================================================
 * 关键词唤醒解码器（B2 的 KwsDecoder）每帧要对 beam 里每条假设调一次
 * forward_one_step()：把一个 token 喂进 trie，拿回「这条路径的前缀还能
 * 不能接上某条关键词」以及当步的 boosting 加分。走完某条关键词后，
 * is_matched() 告诉解码器该路径已经命中，命中节点的 ac_threshold / phrase
 * 用于后续的触发判定和展示。
 *
 * 数据流：
 *     KeywordSpec[]（B2 从 keywords.txt 解析）
 *         -> KwsContextGraph（本文件，构建期一次）
 *         -> forward_one_step()（运行期每帧每假设一次）
 *
 * 本文件不碰 ORT、不读文件、不打日志，是纯内存算法。
 *
 * ============================================================
 * 与 Python 版的差异
 * ============================================================
 * 语义基准是 `python_ref/src/kws_decode.py` 的 `_ContextGraph`，它本身是上游
 * `sherpa-onnx/csrc/context-graph.cc` 的逐行移植。本实现与两处参考的差异：
 *
 *   1. 入参由「token_ids + 4 个平行数组」改为 `std::vector<KeywordSpec>`，
 *      消除下标错位；Build 内部仍按上游的 score/threshold 回落规则展开；
 *   2. 上游遇到 fail 链异常时调 SHERPA_ONNX_EXIT(-1) 直接退出进程；库代码
 *      不应这么做，这里改抛 std::logic_error（正常构图下不可达）；
 *   3. 空 token_ids 由「上游静默跳过」改为抛 std::invalid_argument：关键词
 *      来自文件解析，空序列是配置错误而不是合法的零长度关键词；
 *   4. 返回三元组改成 KwsStepResult 结构，字段名与 Python 版对应；
 *   5. `is_matched` 用「bool + 输出参数」，与本仓库其它库的成败表达一致。
 *
 * 其余分支——节点字段、公共前缀合并、output_score 的 +=、strict/non-strict
 * 的分数口径、浮点 `output_score != 0` 比较——全部照抄，不做"优化"。
 *
 * ============================================================
 * 调用顺序（对照各函数内的编号）
 * ============================================================
 *   构造：
 *     1.1 建根节点并让 fail 指向自己
 *     1.2 build()：逐条关键词插入 trie
 *     1.3 fill_fail_output()：BFS 填 fail / output 弧
 *   运行期：
 *     2.1 forward_one_step() 逐 token 前进（KWS 恒用 strict_mode=true）
 *     2.2 is_matched() 判断当前路径是否走完某条关键词
 *     3.1 finalize() 仅单测使用
 *
 * ============================================================
 * 用法
 * ============================================================
 *     std::vector<audio_ai::KeywordSpec> specs = ...;
 *     audio_ai::KwsContextGraph graph(specs, 1.0f, 0.25f);
 *     const KwsContextState* state = graph.root();
 *     auto step = graph.forward_one_step(state, token_id);
 *     state = step.state;
 *     const KwsContextState* matched = nullptr;
 *     if (graph.is_matched(state, &matched)) { ... }
 */

#include "audio_ai_process/kws_context_graph.h"

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>

namespace audio_ai {

// ============================================================
// 一、构造与建树
// ============================================================

KwsContextState::KwsContextState(int32_t token, float token_score,
                                 float node_score, float output_score,
                                 int32_t level, float ac_threshold,
                                 bool is_end, const std::string& phrase)
    : token(token),
      token_score(token_score),
      node_score(node_score),
      output_score(output_score),
      level(level),
      ac_threshold(ac_threshold),
      is_end(is_end),
      phrase(phrase) {}

// 构造：先建根节点（token=-1、fail 指向自己），再插关键词、最后补 fail 弧。
// 顺序不能换：fill_fail_output() 依赖 build() 建好的全部 next 边。
KwsContextGraph::KwsContextGraph(const std::vector<KeywordSpec>& keywords,
                                 float context_score, float ac_threshold)
    : context_score_(context_score), ac_threshold_(ac_threshold) {
    // 1.1 根节点的 node_score 固定为 0；fail 自指让失配回退的循环有终止条件。
    root_ = std::make_unique<KwsContextState>(-1, 0.0f, 0.0f, 0.0f);
    root_->fail = root_.get();

    // 1.2 逐条插入；1.3 所有边齐了之后才能算 fail/output。
    build(keywords);
    fill_fail_output();
}

// 把每条关键词插入 trie，合并公共前缀。参数 keywords 来自调用方（B2 的文件
// 解析器）；空 token_ids 抛 std::invalid_argument，异常信息带下标便于定位。
// 单条 score/threshold 为 0 时回落到 context_score_ / ac_threshold_（上游语义）。
void KwsContextGraph::build(const std::vector<KeywordSpec>& keywords) {
    for (size_t i = 0; i < keywords.size(); ++i) {
        const KeywordSpec& spec = keywords[i];
        if (spec.token_ids.empty()) {
            throw std::invalid_argument(
                "KwsContextGraph: 第 " + std::to_string(i) +
                " 条关键词的 token_ids 为空");
        }

        // score == 0 是「用默认值」的哨兵，不能当成零加分；threshold 同理。
        const float score =
            spec.score == 0.0f ? context_score_ : spec.score;
        const float threshold =
            spec.threshold == 0.0f ? ac_threshold_ : spec.threshold;

        KwsContextState* node = root_.get();
        for (size_t j = 0; j < spec.token_ids.size(); ++j) {
            const int32_t token = spec.token_ids[j];
            const bool is_last = (j + 1 == spec.token_ids.size());

            auto it = node->next.find(token);
            if (it == node->next.end()) {
                // 新边：output_score 在结尾节点上先填 node_score，
                // 非结尾节点留 0，等 fill_fail_output() 再叠加 suffix 奖励。
                const float node_score = node->node_score + score;
                auto child = std::make_unique<KwsContextState>(
                    token, score, node_score, is_last ? node_score : 0.0f,
                    static_cast<int32_t>(j) + 1, is_last ? threshold : 0.0f,
                    is_last, is_last ? spec.phrase : std::string());
                it = node->next.emplace(token, std::move(child)).first;
            } else {
                // 公共前缀：边分取较大值；若本关键词共享这条边，累计分随之更新。
                // 一条关键词是另一条的前缀时（SHE / SHELL），is_end 保持 true，
                // output_score 要按新的 node_score 重算，不能清零。
                KwsContextState* child = it->second.get();
                const float token_score = std::max(score, child->token_score);
                child->token_score = token_score;
                child->node_score = node->node_score + token_score;
                const bool is_end = is_last || child->is_end;
                child->output_score = is_end ? child->node_score : 0.0f;
                child->is_end = is_end;

                // phrase / ac_threshold 只属于「在这条路径结束」的关键词；
                // 中间经过的共享节点不覆盖，避免后插入的关键词污染前一条。
                if (is_last) {
                    child->phrase = spec.phrase;
                    child->ac_threshold = threshold;
                }
            }
            node = it->second.get();
        }
        ++num_keywords_;
    }
}

// BFS 填写 fail 链与 output 弧（Aho-Corasick 标准流程）。
// 前置条件：build() 已完成，root_->next 与所有子边都在。
// fail 指向「最长后缀」节点；output 指向 fail 链上最近的 end 节点，
// output_score 把该 end 节点的输出奖励叠加到当前节点（是 +=，不是 =）。
void KwsContextGraph::fill_fail_output() {
    std::queue<KwsContextState*> pending;

    // 根的直接子节点没有更短的非空后缀，fail 一律指向根。
    for (auto& kv : root_->next) {
        kv.second->fail = root_.get();
        pending.push(kv.second.get());
    }

    while (!pending.empty()) {
        KwsContextState* current = pending.front();
        pending.pop();

        for (auto& kv : current->next) {
            const int32_t token = kv.first;
            KwsContextState* child = kv.second.get();

            // 1) 沿父节点的 fail 链找能接上 token 的节点；到根即停。
            const KwsContextState* fail = current->fail;
            auto it = fail->next.find(token);
            if (it != fail->next.end()) {
                fail = it->second.get();
            } else {
                fail = fail->fail;
                while (fail->next.find(token) == fail->next.end()) {
                    fail = fail->fail;
                    if (fail->token == -1) break;  // 已回到根
                }
                auto found = fail->next.find(token);
                if (found != fail->next.end()) {
                    fail = found->second.get();
                }
            }
            child->fail = fail;

            // 2) fail 链上最近的 end 节点作为 output；根不是 end，走到根就收手。
            const KwsContextState* output = fail;
            while (!output->is_end) {
                output = output->fail;
                if (output->token == -1) {
                    output = nullptr;
                    break;
                }
            }
            child->output = output;

            // 3) 叠加后缀奖励。节点自身的分数在 build() 已写入，这里只能 +=。
            child->output_score +=
                (output == nullptr ? 0.0f : output->output_score);

            pending.push(child);
        }
    }
}

// ============================================================
// 二、运行期查询
// ============================================================

// 沿 trie 走一步。state 是当前假设持有的节点；token 是这一帧要尝试的 token。
// 返回本步加分与下一步状态；strict_mode=false 仅用于复刻上游单测。
KwsStepResult KwsContextGraph::forward_one_step(
    const KwsContextState* state, int32_t token, bool strict_mode) const {
    if (state == nullptr) {
        throw std::invalid_argument(
            "KwsContextGraph::forward_one_step: state 不能为空");
    }

    const KwsContextState* node = nullptr;
    float score = 0.0f;

    auto direct = state->next.find(token);
    if (direct != state->next.end()) {
        // 情况 A：当前节点直接有这条边，按边分加分。
        node = direct->second.get();
        score = node->token_score;
    } else {
        // 情况 B：失配，沿 fail 链找能接上 token 的最长后缀。
        // score 用「新节点累计分 - 旧节点累计分」，与上游一致。
        node = state->fail;
        while (node->next.find(token) == node->next.end()) {
            node = node->fail;
            if (node->token == -1) break;  // 回到根
        }
        auto found = node->next.find(token);
        if (found != node->next.end()) {
            node = found->second.get();
        }
        score = node->node_score - state->node_score;
    }

    // fail 链最终一定回到根，node 为空说明构图被破坏；库代码不直接退出进程。
    if (node == nullptr) {
        throw std::logic_error(
            "KwsContextGraph::forward_one_step: trie 状态损坏（fail 链未回到根）");
    }

    // 本步命中的 end 节点：自身是结尾，或 fail 链上最近的结尾节点。
    const KwsContextState* matched = nullptr;
    if (node->is_end) {
        matched = node;
    } else if (node->output != nullptr) {
        matched = node->output;
    }

    // 非 strict：命中关键词后立刻回根，把该关键词的累计分一次性结算。
    // 训练/解码不用这条分支，只用于上游单测；浮点比较照抄，不加 epsilon。
    if (!strict_mode && node->output_score != 0.0f) {
        const float output_score =
            node->is_end
                ? node->node_score
                : (node->output != nullptr ? node->output->node_score
                                           : node->node_score);
        return {score + output_score - node->node_score, root_.get(), matched};
    }

    // strict：状态继续前进，把 output 奖励叠进本步得分。
    return {score + node->output_score, node, matched};
}

// 判断当前状态是否走完一条关键词。state 为空按未命中处理（防御性）；
// matched 允许传 nullptr，此时只返回布尔结果。
bool KwsContextGraph::is_matched(const KwsContextState* state,
                                 const KwsContextState** matched) const {
    if (matched != nullptr) {
        *matched = nullptr;
    }
    if (state == nullptr) {
        return false;
    }
    if (state->is_end) {
        if (matched != nullptr) *matched = state;
        return true;
    }
    if (state->output != nullptr) {
        if (matched != nullptr) *matched = state->output;
        return true;
    }
    return false;
}

// 收尾：退掉当前路径的累计分并回到根。仅 tools/test_context_graph.cpp
// 对拍上游单测时使用；流式解码没有「收尾」这一步。
std::pair<float, const KwsContextState*> KwsContextGraph::finalize(
    const KwsContextState* state) const {
    if (state == nullptr) {
        throw std::invalid_argument(
            "KwsContextGraph::finalize: state 不能为空");
    }
    return {-state->node_score, root_.get()};
}

}  // namespace audio_ai

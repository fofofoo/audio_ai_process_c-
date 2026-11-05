/**
 * @file test_context_graph.cpp
 * @brief KwsContextGraph 单测：搬上游 4 组 gtest 期望值 + 字段自检。
 *
 * ============================================================
 * 这份代码是干什么的
 * ============================================================
 * 用 sherpa-onnx `csrc/context-graph-test.cc` 的 4 组用例（strict / 非 strict
 * × 默认分 / 自定义分）验证本仓库的关键词前缀树，再补一组字段自检，覆盖
 * 分数比对看不到的 phrase / ac_threshold / output 弧。
 *
 * 不加载 onnx、不需要权重、不依赖 ORT，纯 CPU + 标准库即可运行。
 *
 * 数据流：
 *     kContexts（9 条互为前后缀的字符串）
 *         -> build_graph() 按上游 TestHelper 口径建图
 *         -> 逐字符 forward_one_step() 累加分数
 *         -> finalize() 退掉路径分
 *         -> 与写死的期望值比对
 *
 * ============================================================
 * 与上游测试的差异
 * ============================================================
 *   1. 用 printf + 计数器代替 gtest，避免给工程引入测试框架依赖；
 *   2. 期望值与上游一致，但分数公式显式写成本文件的 build_graph()，
 *      便于看清 `score=0 -> 回落 context_score=1` 的口径；
 *   3. 新增「字段自检」：上游 4 组只比总分，phrase / ac_threshold /
 *      output 弧即使写错总分也可能碰巧对上；
 *   4. 新增「空 token_ids 抛 std::invalid_argument」检查（本仓库的
 *      刻意差异，见 kws_context_graph.cpp 文件头）。
 *
 * ============================================================
 * 调用顺序（对照 main 里的编号）
 * ============================================================
 *   1.1 逐组跑上游 4 组用例
 *   1.2 字段自检
 *   1.3 汇总并按失败数返回退出码
 *
 * ============================================================
 * 用法
 * ============================================================
 *     cmake --build build -j --target test_context_graph.out
 *     ./build/test_context_graph.out
 *
 * 全部通过时最后一行打印 OK 并返回 0；任一断言失败返回 1。
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "audio_ai_process/kws_context_graph.h"

namespace {

// ============================================================
// 一、用例数据（数值来自上游 context-graph-test.cc）
// ============================================================

/// 上游 TestHelper 里的 9 条关键词：互为前后缀，专门压 fail 链和 output 弧。
const char* const kContexts[] = {
    "S", "HE", "SHE", "SHELL", "HIS",
    "HERS", "HELLO", "THIS", "THEM",
};

struct Query {
    const char* text;  ///< 查询串，逐字符当作 token id
    float want;        ///< ForwardOneStep + Finalize 的累计期望分
};

struct Case {
    const char* name;          ///< 用例名（与上游 TEST 名一致）
    float base_score;          ///< 每条关键词的总分；0 表示回落到 context_score=1
    bool strict_mode;          ///< 传给 forward_one_step 的 strict_mode
    std::vector<Query> queries;
};

const Case kCases[] = {
    // TestBasic：base_score=0 -> 每条关键词的边分回落到 context_score=1
    {"TestBasic", 0.0f, true,
     {{"HEHERSHE", 14.0f}, {"HERSHE", 12.0f}, {"HISHE", 9.0f},
      {"SHED", 6.0f}, {"SHELF", 6.0f}, {"HELL", 2.0f},
      {"HELLO", 7.0f}, {"DHRHISQ", 4.0f}, {"THEN", 2.0f}}},

    // TestBasicNonStrict：同一组关键词，命中后立即回根结算
    {"TestBasicNonStrict", 0.0f, false,
     {{"HEHERSHE", 7.0f}, {"HERSHE", 5.0f}, {"HISHE", 5.0f},
      {"SHED", 3.0f}, {"SHELF", 3.0f}, {"HELL", 2.0f},
      {"HELLO", 2.0f}, {"DHRHISQ", 3.0f}, {"THEN", 2.0f}}},

    // TestCustomize：base_score=5，均摊到每个 token（HE -> 2.5、SHE -> 1.67）
    {"TestCustomize", 5.0f, true,
     {{"HEHERSHE", 35.84f}, {"HERSHE", 30.84f}, {"HISHE", 24.18f},
      {"SHED", 18.34f}, {"SHELF", 18.34f}, {"HELL", 5.0f},
      {"HELLO", 13.0f}, {"DHRHISQ", 10.84f}, {"THEN", 5.0f}}},

    // TestCustomizeNonStrict
    {"TestCustomizeNonStrict", 5.0f, false,
     {{"HEHERSHE", 20.0f}, {"HERSHE", 15.0f}, {"HISHE", 10.84f},
      {"SHED", 10.0f}, {"SHELF", 10.0f}, {"HELL", 5.0f},
      {"HELLO", 5.0f}, {"DHRHISQ", 5.84f}, {"THEN", 5.0f}}},
};

/// 浮点比较容差：期望值保留两位小数，1e-3 足够吸收 float 累加误差。
constexpr float kScoreTol = 1e-3f;

// ============================================================
// 二、断言与建图辅助
// ============================================================

int g_failures = 0;  ///< 全局失败计数；main 按它决定退出码

// 记录一次布尔断言失败；成功不打印，保持输出紧凑。
void check_true(bool cond, const std::string& what) {
    if (!cond) {
        std::printf("  FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

// 记录一次浮点断言失败，打印实得与期望，便于直接对照上游数值。
void check_near(float got, float want, const std::string& what) {
    if (std::fabs(got - want) > kScoreTol) {
        std::printf("  FAIL: %s 实得 %.4f，期望 %.4f\n", what.c_str(), got,
                    want);
        ++g_failures;
    }
}

// 按上游 TestHelper 的口径建图：每条关键词的边分 = 总分 / 字符数，
// 并用 std::round 保留两位（模拟上游 C++ 的 float 计算，不能用 double）。
// base_score=0 时每条边分是 0，构造时回落到 context_score=1。
audio_ai::KwsContextGraph build_graph(float base_score) {
    std::vector<audio_ai::KeywordSpec> specs;
    for (const char* word : kContexts) {
        const size_t len = std::strlen(word);
        audio_ai::KeywordSpec spec;
        for (size_t j = 0; j < len; ++j) {
            // 显式先转 unsigned char：避免 char 有符号时把高位字符变成负数。
            spec.token_ids.push_back(
                static_cast<int32_t>(static_cast<unsigned char>(word[j])));
        }
        spec.score =
            std::round(base_score / static_cast<float>(len) * 100.0f) / 100.0f;
        specs.push_back(std::move(spec));
    }
    // C++17 保证返回值直接构造，KwsContextGraph 不可拷贝也不影响。
    return audio_ai::KwsContextGraph(specs, 1.0f, 0.0f);
}

// ============================================================
// 三、跑一组上游用例
// ============================================================

// 逐条查询走 trie，累加每步分数与 finalize 分，和期望值比对。
// 结束时状态必须回到根；任何一条失败只计数，不中断其余查询。
void run_case(const Case& test_case) {
    audio_ai::KwsContextGraph graph = build_graph(test_case.base_score);
    const int failures_before = g_failures;

    for (const Query& query : test_case.queries) {
        const audio_ai::KwsContextState* state = graph.root();
        float total = 0.0f;

        for (const char* p = query.text; *p != '\0'; ++p) {
            const int32_t token =
                static_cast<int32_t>(static_cast<unsigned char>(*p));
            const audio_ai::KwsStepResult step =
                graph.forward_one_step(state, token, test_case.strict_mode);
            total += step.context_score;
            state = step.state;
        }

        const auto tail = graph.finalize(state);
        total += tail.first;

        const std::string prefix =
            std::string(test_case.name) + " / " + query.text;
        check_true(tail.second == graph.root(), prefix + "：Finalize 没回到根");
        check_true(graph.root()->token == -1, prefix + "：根节点 token 不是 -1");
        check_near(total, query.want, prefix);
    }

    std::printf("  %-24s %s（%zu 条查询）\n", test_case.name,
                g_failures == failures_before ? "OK" : "FAIL",
                test_case.queries.size());
}

// ============================================================
// 四、字段自检（上游 4 组分数覆盖不到的部分）
// ============================================================

// 检查 end 节点的 phrase / ac_threshold、fail/output 弧的指向与叠加，
// 以及 is_matched 的返回值。这些字段写错时总分可能仍然正确，必须单独断言。
void check_fields() {
    const int failures_before = g_failures;

    std::vector<audio_ai::KeywordSpec> specs;
    for (const char* word : kContexts) {
        const size_t len = std::strlen(word);
        audio_ai::KeywordSpec spec;
        for (size_t j = 0; j < len; ++j) {
            spec.token_ids.push_back(
                static_cast<int32_t>(static_cast<unsigned char>(word[j])));
        }
        // score 留 0 -> 回落到 context_score=1，便于心算 node_score。
        spec.phrase = word;
        if (std::strcmp(word, "SHE") == 0) {
            spec.threshold = 0.35f;  // 单独指定，验证逐关键词阈值
        }
        specs.push_back(std::move(spec));
    }

    audio_ai::KwsContextGraph graph(specs, 1.0f, 0.25f);
    const audio_ai::KwsContextState* root = graph.root();

    // --- S 节点：默认阈值 0.25，phrase = "S" ---
    const auto it_s = root->next.find('S');
    check_true(it_s != root->next.end(), "根节点缺少 S 边");
    if (it_s != root->next.end()) {
        const audio_ai::KwsContextState* node_s = it_s->second.get();
        check_true(node_s->is_end, "S 节点应为关键词结尾");
        check_true(node_s->phrase == "S", "S 节点 phrase 错误");
        check_true(node_s->ac_threshold == 0.25f,
                   "S 节点应回落到默认阈值 0.25");
    }

    // --- SHE 节点：单独的 0.35 阈值、output 指向 HE、output_score 叠加 ---
    const auto it_h = root->next.find('H');
    check_true(it_h != root->next.end(), "根节点缺少 H 边");
    if (it_h != root->next.end()) {
        const audio_ai::KwsContextState* node_he = it_h->second.get();
        const auto it_e = node_he->next.find('E');
        check_true(it_e != node_he->next.end(), "HE 节点缺少 E 边");
        if (it_e != node_he->next.end()) {
            const audio_ai::KwsContextState* node_he_end = it_e->second.get();

            // 走到 SHE：root -> S -> H -> E
            const auto it_she_h = it_s->second->next.find('H');
            check_true(it_she_h != it_s->second->next.end(),
                       "S 节点缺少 H 边");
            if (it_she_h != it_s->second->next.end()) {
                const auto it_she_e = it_she_h->second->next.find('E');
                check_true(it_she_e != it_she_h->second->next.end(),
                           "SH 节点缺少 E 边");
                if (it_she_e != it_she_h->second->next.end()) {
                    const audio_ai::KwsContextState* node_she =
                        it_she_e->second.get();

                    check_true(node_she->is_end, "SHE 节点应为关键词结尾");
                    check_true(node_she->phrase == "SHE", "SHE 节点 phrase 错误");
                    check_true(node_she->ac_threshold == 0.35f,
                               "SHE 节点应使用单独的阈值 0.35");
                    check_true(node_she->fail == node_he_end,
                               "SHE 的 fail 应指向 HE");
                    check_true(node_she->output == node_he_end,
                               "SHE 的 output 应指向 HE");
                    // SHE 自身 3 分 + 后缀 HE 的 2 分 = 5 分
                    check_true(node_she->output_score == 5.0f,
                               "SHE 的 output_score 应为 5（自身 3 + 后缀 HE 2）");

                    const audio_ai::KwsContextState* matched = nullptr;
                    check_true(graph.is_matched(node_she, &matched),
                               "is_matched(SHE) 应为 true");
                    check_true(matched == node_she,
                               "is_matched(SHE) 应返回 SHE 自身");
                }
            }
        }
    }

    // --- is_matched 在 end 节点上返回自身 ---
    if (it_s != root->next.end()) {
        const audio_ai::KwsContextState* matched = nullptr;
        check_true(graph.is_matched(it_s->second.get(), &matched),
                   "is_matched(S) 应为 true");
        check_true(matched == it_s->second.get(),
                   "is_matched(S) 应返回 S 自身");
    }

    // --- 空 token_ids 必须抛 std::invalid_argument（本仓库的刻意差异）---
    bool threw = false;
    try {
        std::vector<audio_ai::KeywordSpec> bad_specs(1);  // token_ids 为空
        audio_ai::KwsContextGraph bad(bad_specs, 1.0f, 0.25f);
        (void)bad;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check_true(threw, "空 token_ids 应抛 std::invalid_argument");

    std::printf("  %-24s %s\n", "字段自检",
                g_failures == failures_before ? "OK" : "FAIL");
}

}  // namespace

int main() {
    std::printf("============================================================\n");
    std::printf("ContextGraph 对拍（期望值来自上游 context-graph-test.cc）\n");
    std::printf("============================================================\n");

    // 1.1 上游 4 组用例
    for (const Case& test_case : kCases) {
        run_case(test_case);
    }

    // 1.2 字段自检
    check_fields();

    // 1.3 汇总
    std::printf("------------------------------------------------------------\n");
    if (g_failures == 0) {
        std::printf("OK\n");
        return 0;
    }
    std::printf("FAIL：%d 项断言未通过\n", g_failures);
    return 1;
}

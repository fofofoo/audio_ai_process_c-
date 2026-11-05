#pragma once

/**
 * @file kws_asr_session.h
 * @brief 双模式业务 Session：关键词唤醒（KWS）与完整识别（ASR）的切换中枢。
 *
 * ============================================================
 * 一、这份代码是干什么的
 * ============================================================
 * 平时只把特征喂给 KwsDecoder，命中唤醒词后切到 AsrDecoder 做完整识别，
 * 出现结束语或长时间没有新 token 再切回监听。同一时刻只有一个解码器吃特征，
 * AudioPreprocess 始终连续，特征时间轴不断。
 *
 * 数据流：
 *
 *   BlockWorker（Worker 线程，串行）
 *        │ AudioFeatureInput { feats, is_tail }
 *        ▼
 *   KwsAsrSession::on_block
 *        ├ KWS 模式：KwsDecoder::accept_features → 命中则 enter_asr()
 *        └ ASR 模式：AsrSession::on_block → 结束语 / 空闲超时 → back_to_kws()
 *        │
 *        ▼
 *   快照（模式 / 文本 / 计数 / 最近事件）——主线程随时读
 *
 * 职责边界：
 *   - 只做模式机、回填环、空闲计时、事件与统计；
 *   - 不建线程、不碰队列（BlockWorker 负责），不建 ONNX 会话（入口负责），
 *     不画图（主线程负责）；
 *   - 转写分段与落盘仍由 AsrSession 负责，本类只在切回时调用它的结算接口。
 *
 * ============================================================
 * 二、关键取舍
 * ============================================================
 *   - 三个解码相关对象按引用注入：入口决定模型路径与执行器，本类只管编排；
 *   - 回填 100 帧（约 1 秒）：唤醒后把最近特征重新喂给 ASR，避免命令词开头
 *     被 chunk 边界吃掉；回填含唤醒词本身，ASR 文本开头可能出现唤醒词，属正常；
 *   - 回填环按「帧」维护（不是按块），块长变化不影响回填长度；
 *   - 结束语判定用 UTF-8 后缀匹配，先剥掉尾部空白与中英文标点；
 *   - 空闲计时用 steady_clock 墙钟，只在 worker 线程读写；没有新音频时
 *     Worker 不执行 on_block，计时自然停住，与 Python 版行为一致；
 *   - 切回 KWS 时先结算 ASR 转写段、再清空本会话 token 历史、最后两个解码器
 *     reset()；转写文件与累计段数保留，下一次唤醒从空文本重新累积。
 *
 * ============================================================
 * 三、线程模型
 * ============================================================
 *   - on_block() / finish() / stats() 只在 BlockWorker 的后台线程调用
 *     （finish 在 finish_and_join() 之后由主线程调用）；
 *   - snapshot() 可从主线程随时调用，内部一把短锁，不等待推理；
 *   - 解码器全程只有 Worker 线程访问，因此不需要加锁。
 *
 * 详细设计见 doc/kws_plan_status.md 第三节与 doc/audio_ai_block_worker.md。
 */

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "audio_ai_process/asr_decode.h"
#include "audio_ai_process/asr_session.h"
#include "audio_ai_process/kws_decode.h"

namespace audio_ai {

/// 当前所处的模式。
enum class KwsAsrMode {
    Kws,  ///< 只监听唤醒词
    Asr,  ///< 只做完整识别
};

/// 模式名（ASCII，写窗口状态条 / 日志用）。
inline const char* mode_name(KwsAsrMode mode) {
    return mode == KwsAsrMode::Asr ? "ASR" : "KWS";
}

/// 双模式的可调参数；默认值即 Python 参考实现的口径。
struct KwsAsrSessionConfig {
    /// 初始模式；--initial-asr 时设为 Asr（调试纯识别行为）。
    KwsAsrMode initial_mode = KwsAsrMode::Kws;
    /// ASR 模式里连续多久没有新 token 就切回 KWS（秒）。
    double idle_timeout_s = 10.0;
    /// 唤醒时回填最近多少帧特征；0 = 不回填。
    int backfill_frames = 100;
    /// 识别文本（剥掉尾部标点后）以这些词结尾就立即切回 KWS。
    std::vector<std::string> end_phrases = {
        "再见", "拜拜", "结束", "退出", "没事了",
        "滚吧", "滚", "退下", "闭嘴", "退下吧",
    };
};

/// 主线程读的显示快照：只保留最新值，不排队、不累积。
struct KwsAsrSnapshot {
    KwsAsrMode mode = KwsAsrMode::Kws;  ///< 当前模式
    std::string text;                   ///< 当前 ASR 会话文本（KWS 模式为空）
    std::string last_text;              ///< 最近一次结束的 ASR 会话文本
    int wake_count = 0;                 ///< 累计唤醒次数
    int back_count = 0;                 ///< 累计切回 KWS 次数
    int64_t event_seq = 0;              ///< 事件序号，主线程据此判断有没有新事件
    std::string last_event;             ///< 最近事件文本（唤醒 / 结束语 / 超时）
};

/// 收尾统计；finish() 之后读取（此时累计值已并入当前会话）。
struct KwsAsrStats {
    int wake_count = 0;                  ///< 唤醒次数
    int back_count = 0;                  ///< 切回 KWS 次数
    int64_t asr_chunks = 0;              ///< ASR encoder 调用次数（跨会话累计）
    int64_t asr_frames = 0;              ///< ASR encoder 累计输出帧数
    int64_t asr_tokens = 0;              ///< ASR 累计 token 数（跨会话）
    int64_t kws_chunks = 0;              ///< KWS encoder 调用次数（跨会话）
    int64_t kws_frames = 0;              ///< KWS encoder 累计输出帧数
    int64_t kws_hits = 0;                ///< KWS 累计触发次数
    int64_t backfilled_frames = 0;       ///< 累计回填帧数
    int64_t tail_blocks_skipped = 0;     ///< KWS 模式下跳过的收尾块数
};

/**
 * @brief 双模式 Session：KWS 监听与 ASR 识别共用一个 Worker 线程。
 *
 * 用法（入口装配）：
 *   KwsAsrSession session(asr_decoder, kws_decoder, asr_session, cfg);
 *   BlockWorker<LogFbankFeatures> worker(
 *       [&session](const AudioFeatureInput& item) { session.on_block(item); });
 *
 * 生命周期：三个被引用的对象都必须长于本对象；Worker 必须声明在本对象之后，
 * 这样 C++ 逆序析构会先停线程，再销毁 Session 与解码器。
 */
class KwsAsrSession {
public:
    /**
     * @brief 绑定三个解码相关对象并初始化模式机。
     * @param asr_decoder ASR 解码器（本类在唤醒 / 切回时调用 reset()）
     * @param kws_decoder KWS 解码器（本类在切回时调用 reset()）
     * @param asr_session ASR 外部缓存（本类在 ASR 模式调用 on_block，
     *                    切回时结算段并清空本会话历史）
     * @param cfg         模式、超时、回填帧数、结束语
     */
    KwsAsrSession(AsrDecoder& asr_decoder, KwsDecoder& kws_decoder,
                  AsrSession& asr_session, KwsAsrSessionConfig cfg);

    KwsAsrSession(const KwsAsrSession&) = delete;
    KwsAsrSession& operator=(const KwsAsrSession&) = delete;

    /**
     * @brief 处理一个音频特征输入块（Worker 线程）。
     *
     * KWS 模式：把特征喂给 KwsDecoder；命中唤醒词则重置 ASR、回填最近特征、
     * 切到 ASR 模式并发布唤醒事件。ASR 模式：喂给 AsrSession；有新 token 就
     * 刷新空闲计时；解出结束语或空闲超时则结算、清空、切回 KWS。
     *
     * 收尾块（is_tail）只对 ASR 有意义：KWS 模式下直接跳过并计数，
     * 因为特征域的 0 是 log 能量 1，不是真静音，喂进束搜索没有意义。
     */
    void on_block(const AudioFeatureInput& item);

    /**
     * @brief 收尾：结算当前 ASR 段、关闭转写文件、返回最近一次 ASR 文本。
     *
     * 必须在 worker.finish_and_join() 之后调用（后台线程已退出）。
     * 当前在 ASR 模式时返回本会话完整文本；在 KWS 模式时返回上一次已结束
     * 会话的文本（与 Python 的 last_asr_text 口径一致）。
     */
    std::string finish();

    /// 主线程读显示快照（内部短锁，不等待推理）。
    KwsAsrSnapshot snapshot() const;

    /// 收尾统计；finish() 之后读取。
    KwsAsrStats stats() const { return stats_; }

private:
    // —— 模式切换 ——
    /// 唤醒：重置 ASR、清空历史、回填最近特征、切到 ASR 并发布事件。
    void enter_asr(const KeywordHit& hit, int n_hits);
    /// 切回：结算转写段、清空历史、两个解码器 reset、发布事件。
    /// @param event_text 事件文本（调用方已按「结束语 / 超时」格式化）
    /// @param text       本段 ASR 文本，存为 last_asr_text_
    void back_to_kws(const std::string& event_text, const std::string& text);
    /// 把当前会话的解码器计数并入累计值（reset() 会清零，必须先加）。
    void accumulate_counters();

    // —— 回填环（按帧存，块长无关） ——
    /// 把一块特征追加进回填环；n_mels 首次出现时初始化环容量。
    void append_backfill(const LogFbankFeatures& feats);
    /// 按时间顺序拼出当前回填环里的特征；为空时返回 0 帧。
    LogFbankFeatures backfill_features() const;

    // —— 结束语 ——
    /// 剥掉尾部空白与中英文标点后，判断是否以任一结束语结尾。
    bool is_end_phrase(const std::string& text) const;

    // —— 快照 ——
    /// 在锁内更新模式 / 文本 / 计数；事件由 publish_event() 单独写。
    void update_snapshot();
    /// 写入一条新事件并递增序号（主线程据此打印一次）。
    void publish_event(const std::string& text);

private:
    AsrDecoder& asr_decoder_;    ///< 生命周期由入口保证
    KwsDecoder& kws_decoder_;    ///< 生命周期由入口保证
    AsrSession& asr_session_;    ///< 生命周期由入口保证
    KwsAsrSessionConfig cfg_;

    KwsAsrMode mode_ = KwsAsrMode::Kws;

    // 回填环：ring_cap_frames_ 帧 × n_mels，head_ 是«下一帧写哪»的绝对位置。
    std::vector<float> ring_;
    size_t ring_cap_frames_ = 0;
    size_t ring_n_mels_ = 0;
    size_t ring_head_ = 0;
    size_t ring_frames_ = 0;

    std::chrono::steady_clock::time_point last_activity_;  ///< 最近一次新 token

    std::string last_asr_text_;  ///< 最近一次结束的 ASR 会话文本

    // 跨会话累计值（reset() 会清零解码器计数，所以另存一份）
    KwsAsrStats stats_;

    mutable std::mutex snapshot_mutex_;  ///< 保护 snap_
    KwsAsrSnapshot snap_;                ///< 主线程读的快照
};

}  // namespace audio_ai

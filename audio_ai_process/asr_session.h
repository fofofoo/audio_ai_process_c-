#pragma once

/**
 * @file asr_session.h
 * @brief ASR 的外部长期缓存：全量 token、按段结算、转写落盘与显示快照。
 *
 * ============================================================
 * 一、这份代码是干什么的
 * ============================================================
 * AsrDecoder 只保留「继续解码必需的固定大小状态」，对外只吐本次新增的
 * token。本类负责解码器之外的那一半：
 *
 *   - 累积全量 token（all_ids_），作为最终文本的唯一事实来源；
 *   - 累积当前段 token（segment_ids_），满 segment_tokens 个就结算；
 *   - 段结算时把该段文本追加一行到 txt 并 flush，异常退出最多丢尾段；
 *   - 保存解码器尾部窗口的显示快照，供主线程随时读取。
 *
 * 它不是线程安全的通用组件，但按固定线程模型使用是安全的：
 *   - on_block() 只在 BlockWorker 的后台线程调用（串行）；
 *   - latest_text() 可在主线程随时调用，内部用一把短锁保护快照；
 *   - finish() / all_token_ids() / n_segments() 必须在
 *     worker.finish_and_join() 之后调用，此时后台线程已退出。
 *
 * ============================================================
 * 二、分段与落盘的取舍
 * ============================================================
 *   - 只按 token 数分段（默认 64），不看时间、不看句末标点：规则确定，
 *     不受语速和静音影响；
 *   - 段边界不 reset 解码器：声学上下文连续，最终文本仍由全量 token 重算；
 *   - txt 一行一段，行内可能断在句子中间，这是简化换来的已知取舍；
 *   - 启动截断、之后每段追加并 flush，不做定时刷写、不做临时文件 rename；
 *   - 写失败只告警一次，不抛异常、不中断解码。
 *
 * 详细计划见 doc/asr_bounded_state_plan.md。
 */

#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "audio_ai_process/asr_decode.h"
#include "audio_ai_process/block_worker.h"

namespace audio_ai {

/// 音频特征输入队列里的一项：特征块 + 是否为收尾块。
/// 所有吃音频特征的能力（ASR / KWS / 双模式）共用这个类型。
using AudioFeatureInput = BlockWorker<LogFbankFeatures>::Item;

/// AsrSession 的可调参数；默认值即定稿值。
struct AsrSessionConfig {
    /// 当前段满多少个 token 就结算并追加一行。
    size_t segment_tokens = 64;
    /// 转写文件路径；启动时截断，之后按段追加。
    std::string transcript_path = "log/asr_last_text.txt";
};

class AsrSession {
public:
    /**
     * @brief 绑定解码器、准备转写文件。
     * @param decoder 生命周期必须长于本对象
     * @param cfg     分段长度与转写路径
     *
     * 构造时创建父目录并以截断方式打开文件；失败只在 stderr 告警一次，
     * 解码照常进行，transcript_ok() 返回 false。
     */
    AsrSession(AsrDecoder& decoder, AsrSessionConfig cfg);

    AsrSession(const AsrSession&) = delete;
    AsrSession& operator=(const AsrSession&) = delete;

    /**
     * @brief 处理一个音频特征输入块（后台线程）。
     *
     * 调用解码器拿到新增 token → 并入全量与当前段 → 段满则结算 →
     * 刷新显示快照。本函数不调用 run_encoder / run_decoder 等私有接口，
     * 所有模型调用都留在 AsrDecoder::accept_features() 内部。
     */
    void on_block(const AudioFeatureInput& item);

    /// 立即结算当前段（不足 segment_tokens 也结算）。
    /// 供双模式切换、暂停等「需要人为切断一段」的场景使用；
    /// 只能在 Worker 线程或 join 之后调用，不能与 on_block() 并发。
    void settle_segment();

    /// 当前会话累计的 token 数（不含已清空的历史）。
    /// 双模式用它判断「这一块有没有产出新 token」；只能 Worker 线程调用。
    size_t token_count() const { return all_ids_.size(); }

    /// 当前会话的全量文本（text_of(all_ids_)）。
    /// 双模式用它做结束语判定；只能 Worker 线程或 join 之后调用。
    std::string current_text() const;

    /// 清空本会话的 token 历史与显示快照，转写文件与段计数保留。
    /// 双模式切回 KWS 时调用：先把当前段 settle_segment() 落盘，再清历史，
    /// 下一次唤醒从空文本重新累积。只能 Worker 线程或 join 之后调用。
    void clear_history();

    /// 显示快照：解码器最近 kRecentTokens 个 token 的文本（有界）。
    std::string latest_text() const;

    /**
     * @brief 收尾：结算不足一段的尾段、关闭文件、返回全量文本。
     *
     * 必须在 worker.finish_and_join() 之后调用：此时后台线程已退出，
     * 不存在对 all_ids_ / segment_ids_ 的并发访问。
     */
    std::string finish();

    /// 全量已确认 token；同样只在 join 之后读取。
    const std::vector<int64_t>& all_token_ids() const { return all_ids_; }

    /// 已结算段数；只在 join 之后读取。
    int n_segments() const { return n_segments_; }

    /// 转写文件是否成功打开；写失败后也会变成 false。
    bool transcript_ok() const { return transcript_ok_; }

    /// 最近一次写入失败的原因（空串 = 一直正常）。
    const std::string& transcript_error() const { return transcript_error_; }

    const std::string& transcript_path() const { return cfg_.transcript_path; }

private:
    /// 结算当前段：非空才转文本、追加一行、flush、清空并计数。
    void close_segment();

    /// 把一行文本写进转写文件；换行替换成空格，保证一行一段。
    /// 失败只记录一次，不抛出、不中断解码。
    void append_transcript(const std::string& text);

private:
    AsrDecoder& decoder_;       ///< 解码器：生命周期由入口保证
    AsrSessionConfig cfg_;

    std::vector<int64_t> all_ids_;      ///< 全量 token（唯一事实来源）
    std::vector<int64_t> segment_ids_;  ///< 当前段 token；满一段就清空
    int n_segments_ = 0;                ///< 已结算段数

    std::ofstream transcript_;          ///< 转写文件；构造时 trunc 打开
    bool transcript_ok_ = false;        ///< 文件可用 / 写入未失败
    std::string transcript_error_;      ///< 第一条失败原因

    mutable std::mutex snapshot_mutex_; ///< 保护 snapshot_
    std::string snapshot_;              ///< 显示快照（解码器尾部窗口）
};

}  // namespace audio_ai

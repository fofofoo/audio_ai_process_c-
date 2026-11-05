/**
 * @file kws_asr_session.cpp
 * @brief KwsAsrSession 实现：模式机、回填环、空闲计时、事件与统计。
 *
 * 与 doc/kws_plan_status.md 第三节的流程一一对应：
 *
 *   on_block（KWS） → KwsDecoder → 命中 → enter_asr()
 *        ├ AsrDecoder::reset()
 *        ├ AsrSession::clear_history()
 *        ├ 回填最近 backfill_frames 帧
 *        └ 切到 ASR，发布「唤醒」事件
 *
 *   on_block（ASR） → AsrSession::on_block → 新 token 刷新计时
 *        └ 结束语 / 空闲超时 → back_to_kws()
 *             ├ settle_segment()（尾段落盘）
 *             ├ clear_history()（本会话 token 清空，文件与段数保留）
 *             ├ 两个解码器 reset()
 *             └ 切回 KWS，发布事件
 *
 * 本文件不做模型推理、不建线程、不碰队列：解码器调用都留在解码类内部，
 * 线程与背压由 BlockWorker 负责，入口只负责装配。
 */

#include "audio_ai_process/kws_asr_session.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <sstream>
#include <utility>

namespace audio_ai {

namespace {

// 结束语判定前要剥掉的尾部字符：ASCII 空白 / 标点 + 中文全角标点。
// 用完整 UTF-8 字节串匹配，避免按字节截断把汉字劈成两半。
const char* const kTrailingPunct[] = {
    " ",  "\t", "\r", "\n", ",",  ".",  "!",  "?",  ";",  ":",
    "\"", "'",  "，", "。", "！", "？", "、", "；", "：", "“",
    "”",  "‘",  "’",  "…",  "—",
};

// 判断 text 是否以 suffix 结尾（两者都是 UTF-8 字节串，直接做字节比较）。
bool ends_with(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// 反复剥掉尾部空白与中英文标点；每次只剥一个符号，直到不再匹配。
std::string strip_trailing_punct(const std::string& text) {
    std::string out = text;
    bool changed = true;
    while (changed && !out.empty()) {
        changed = false;
        for (const char* punct : kTrailingPunct) {
            const std::string p(punct);
            if (ends_with(out, p)) {
                out.erase(out.size() - p.size());
                changed = true;
                break;
            }
        }
    }
    return out;
}

// steady_clock 的「现在」，统一成一个短名字，避免每处写一长串。
std::chrono::steady_clock::time_point now() {
    return std::chrono::steady_clock::now();
}

}  // namespace

// 绑定三个对象、初始化模式与计时；回填环等到第一块特征出现 n_mels 再分配。
KwsAsrSession::KwsAsrSession(AsrDecoder& asr_decoder, KwsDecoder& kws_decoder,
                             AsrSession& asr_session, KwsAsrSessionConfig cfg)
    : asr_decoder_(asr_decoder),
      kws_decoder_(kws_decoder),
      asr_session_(asr_session),
      cfg_(std::move(cfg)),
      mode_(cfg_.initial_mode),
      last_activity_(now()) {
    // 负值按「不回填」处理，避免构造出无意义的大环。
    if (cfg_.backfill_frames < 0) {
        cfg_.backfill_frames = 0;
    }
    // 超时 <= 0 会让 ASR 模式几乎每块都切回，属于明显误配；回落到默认 10 s。
    if (cfg_.idle_timeout_s <= 0.0) {
        cfg_.idle_timeout_s = 10.0;
    }
    // 初始就处于 ASR 时，文本与计时都从零开始；KWS 模式没有可显示文本。
    if (mode_ == KwsAsrMode::Asr) {
        last_activity_ = now();
    }
    update_snapshot();
}

// 一块特征的总调度：两种模式互斥，收尾块在 KWS 模式下直接跳过。
void KwsAsrSession::on_block(const AudioFeatureInput& item) {
    // 1) 回填环始终跟随真实音频；收尾块是特征域的 0（log 能量 1），
    //    不是真静音，不能污染下一次唤醒的回填内容。
    if (!item.is_tail) {
        append_backfill(item.block);
    }

    // 2) KWS 模式：只喂唤醒器。
    if (mode_ == KwsAsrMode::Kws) {
        if (item.is_tail) {
            ++stats_.tail_blocks_skipped;
            return;
        }
        const std::vector<KeywordHit> hits =
            kws_decoder_.accept_features(item.block);
        if (!hits.empty()) {
            // 同一块可能触发多条，只按第一条唤醒（Python 同口径），
            // 但唤醒次数按命中条数累计。
            enter_asr(hits.front(), static_cast<int>(hits.size()));
        }
        return;
    }

    // 3) ASR 模式：只喂 ASR；有新 token 就刷新空闲计时。
    const size_t before = asr_session_.token_count();
    asr_session_.on_block(item);
    if (asr_session_.token_count() > before) {
        last_activity_ = now();
    }
    const std::string text = asr_session_.current_text();
    update_snapshot();

    // 4) 切回条件：结束语优先，其次空闲超时（与 Python 6.7 顺序一致）。
    if (is_end_phrase(text)) {
        std::ostringstream event;
        event << "[结束语] " << strip_trailing_punct(text);
        back_to_kws(event.str(), text);
        return;
    }
    const double idle_s =
        std::chrono::duration<double>(now() - last_activity_).count();
    if (idle_s > cfg_.idle_timeout_s) {
        std::ostringstream event;
        event << "[超时 " << static_cast<int>(idle_s) << "s] 切回关键词模式";
        back_to_kws(event.str(), text);
    }
}

// 唤醒：重置 ASR（状态 + 本会话 token），回填最近约 1 秒特征，再切模式。
// 回填包含唤醒词本身，ASR 文本开头可能出现唤醒词，这是可接受的折中。
void KwsAsrSession::enter_asr(const KeywordHit& hit, int n_hits) {
    asr_decoder_.reset();
    // 正常切回时历史已经清空；这里再清一次是防御，保证每次唤醒都从空文本
    // 重新累积，不会把上一段会话的字带进来。
    asr_session_.clear_history();

    mode_ = KwsAsrMode::Asr;
    last_activity_ = now();

    if (cfg_.backfill_frames > 0) {
        const LogFbankFeatures backfill = backfill_features();
        if (backfill.n_frames > 0) {
            AudioFeatureInput item;
            item.block = backfill;
            item.is_tail = false;
            asr_session_.on_block(item);
            stats_.backfilled_frames += backfill.n_frames;
        }
    }

    stats_.wake_count += n_hits;
    update_snapshot();

    std::ostringstream event;
    event << "[唤醒] " << hit.keyword << "  (avg_prob=" << hit.avg_prob
          << ", t=" << hit.start_time << "s)";
    publish_event(event.str());
}

// 切回 KWS：先结算当前转写段（落盘），再清本会话 token，最后两个解码器重置。
// 转写文件与累计段数保留，下一次唤醒从空文本重新累积。
void KwsAsrSession::back_to_kws(const std::string& event_text,
                                const std::string& text) {
    // 先加计数再 reset：reset() 会把解码器的 n_chunks / n_tokens 清零。
    stats_.asr_tokens += static_cast<int64_t>(asr_session_.token_count());
    accumulate_counters();

    asr_session_.settle_segment();
    asr_session_.clear_history();
    asr_decoder_.reset();
    kws_decoder_.reset();

    mode_ = KwsAsrMode::Kws;
    last_asr_text_ = text;
    last_activity_ = now();
    ++stats_.back_count;
    update_snapshot();
    publish_event(event_text);
}

// 把当前会话的解码器计数并入累计值；调用方保证随后会 reset()。
void KwsAsrSession::accumulate_counters() {
    stats_.asr_chunks += asr_decoder_.n_chunks();
    stats_.asr_frames += asr_decoder_.n_enc_frames();
    stats_.kws_chunks += kws_decoder_.n_chunks();
    stats_.kws_frames += kws_decoder_.n_enc_frames();
    stats_.kws_hits += kws_decoder_.n_hits();
}

// 按帧追加进环形缓冲：第 i 帧写到 (head_ % cap) 行，满了覆盖最旧一帧。
// 块长与出帧节奏不影响最终保留的帧数，回填长度始终等于 backfill_frames。
void KwsAsrSession::append_backfill(const LogFbankFeatures& feats) {
    if (cfg_.backfill_frames <= 0 || feats.n_frames <= 0 || feats.n_mels <= 0) {
        return;
    }
    const size_t cap = static_cast<size_t>(cfg_.backfill_frames);
    const size_t n_mels = static_cast<size_t>(feats.n_mels);

    // 首次出现（或 n_mels 变化）时按实际维度分配；模型固定 80，不会反复重建。
    if (ring_.empty() || ring_n_mels_ != n_mels) {
        ring_n_mels_ = n_mels;
        ring_cap_frames_ = cap;
        ring_.assign(cap * n_mels, 0.0f);
        ring_head_ = 0;
        ring_frames_ = 0;
    }

    for (int f = 0; f < feats.n_frames; ++f) {
        const float* src = feats.data.data() + static_cast<size_t>(f) * n_mels;
        float* dst = ring_.data() + (ring_head_ % ring_cap_frames_) * n_mels;
        std::copy(src, src + n_mels, dst);
        ++ring_head_;
        if (ring_frames_ < ring_cap_frames_) {
            ++ring_frames_;
        }
    }
}

// 按时间顺序（最旧 → 最新）拼出回填特征；环未满时从第 0 行开始。
LogFbankFeatures KwsAsrSession::backfill_features() const {
    LogFbankFeatures out;
    out.n_mels = static_cast<int>(ring_n_mels_);
    if (ring_frames_ == 0 || ring_n_mels_ == 0) {
        return out;
    }

    const size_t n_mels = ring_n_mels_;
    const size_t count = ring_frames_;
    const size_t start =
        (count == ring_cap_frames_) ? (ring_head_ % ring_cap_frames_) : 0;
    out.n_frames = static_cast<int>(count);
    out.data.resize(count * n_mels);

    // 环形缓冲可能跨过末尾，分两段拷贝：start..cap 与 0..剩余。
    const size_t first = std::min(count, ring_cap_frames_ - start);
    std::copy(ring_.begin() + static_cast<ptrdiff_t>(start * n_mels),
              ring_.begin() + static_cast<ptrdiff_t>((start + first) * n_mels),
              out.data.begin());
    if (first < count) {
        std::copy(ring_.begin(),
                  ring_.begin() + static_cast<ptrdiff_t>((count - first) * n_mels),
                  out.data.begin() + static_cast<ptrdiff_t>(first * n_mels));
    }
    return out;
}

// 结束语判定：先剥掉尾部空白与中英文标点，再做 UTF-8 后缀比较。
bool KwsAsrSession::is_end_phrase(const std::string& text) const {
    if (text.empty()) {
        return false;
    }
    const std::string stripped = strip_trailing_punct(text);
    for (const std::string& phrase : cfg_.end_phrases) {
        if (!phrase.empty() && ends_with(stripped, phrase)) {
            return true;
        }
    }
    return false;
}

// 收尾：当前在 ASR 就结算尾段并取本会话文本；在 KWS 就只关文件，
// 返回上一次已结束会话的文本（Python 的 last_asr_text 口径）。
std::string KwsAsrSession::finish() {
    if (mode_ == KwsAsrMode::Asr) {
        stats_.asr_tokens += static_cast<int64_t>(asr_session_.token_count());
        last_asr_text_ = asr_session_.finish();
    } else {
        // 历史已在切回时清空，finish() 只负责结算（空段直接返回）并关闭文件。
        asr_session_.finish();
    }
    accumulate_counters();
    update_snapshot();
    return last_asr_text_;
}

// 主线程读快照：只拷贝值，锁内不做 detokenize、不碰解码器。
KwsAsrSnapshot KwsAsrSession::snapshot() const {
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    return snap_;
}

// 刷新快照：detokenize 在锁外做，锁内只赋值；last_event 由 publish_event()
// 单独维护。主线程读快照因此不会等解码器算文本。
void KwsAsrSession::update_snapshot() {
    // KWS 模式没有当前 ASR 会话，文本置空；ASR 模式下按全量 token 重算。
    std::string text;
    if (mode_ == KwsAsrMode::Asr) {
        text = asr_session_.current_text();
    }
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    snap_.mode = mode_;
    snap_.text = std::move(text);
    snap_.last_text = last_asr_text_;
    snap_.wake_count = stats_.wake_count;
    snap_.back_count = stats_.back_count;
}

// 新事件：序号 +1，主线程发现序号变化就打印一次，避免后台线程直接写 stdout
// 与主循环的原地刷新互相踩行。
void KwsAsrSession::publish_event(const std::string& text) {
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    snap_.last_event = text;
    ++snap_.event_seq;
}

}  // namespace audio_ai

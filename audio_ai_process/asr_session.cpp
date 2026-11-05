/**
 * @file asr_session.cpp
 * @brief AsrSession 实现：全量 token 累积、按段结算、转写落盘、显示快照。
 *
 * 与 doc/asr_bounded_state_plan.md 第四节的流程一一对应：
 *   on_block → accept_features → all_ids / segment_ids → 段满结算 → 快照
 *   finish   → 结算尾段 → 关闭文件 → text_of(all_ids)
 *
 * 本文件不调用任何模型私有接口，也不重新实现 detokenize：文本转换一律
 * 走 AsrDecoder::text_of() / recent_text()，保证与离线入口逐字节一致。
 */

#include "audio_ai_process/asr_session.h"

#include <cstdio>
#include <filesystem>
#include <utility>

namespace audio_ai {

namespace {

// 转写文件一行一段：把文本里的换行替换成空格，避免一行被拆成多行。
std::string to_single_line(const std::string& text) {
    std::string out = text;
    for (char& c : out) {
        if (c == '\n' || c == '\r') {
            c = ' ';
        }
    }
    return out;
}

}  // namespace

// 绑定解码器、创建父目录、以截断方式打开转写文件。
// 目录或文件失败只告警一次并保持 transcript_ok_ = false，解码不受影响。
AsrSession::AsrSession(AsrDecoder& decoder, AsrSessionConfig cfg)
    : decoder_(decoder), cfg_(std::move(cfg)) {
    // 0 会让每块都触发结算，属于明显误配；回落到定稿默认值。
    if (cfg_.segment_tokens == 0) {
        cfg_.segment_tokens = 64;
    }

    const std::filesystem::path path(cfg_.transcript_path);
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            transcript_error_ = "创建目录失败：" + ec.message();
            std::fprintf(stderr, "[ASR] 无法创建转写目录 %s：%s\n",
                         parent.string().c_str(), ec.message().c_str());
            return;
        }
    }

    // 启动即截断：每次运行只保留本次会话的转写，避免旧内容混进来。
    transcript_.open(path.string(), std::ios::out | std::ios::trunc);
    if (!transcript_) {
        transcript_error_ = "打不开转写文件：" + cfg_.transcript_path;
        std::fprintf(stderr, "[ASR] 打不开转写文件 %s，本次不落盘\n",
                     cfg_.transcript_path.c_str());
        return;
    }
    transcript_ok_ = true;
}

// 处理一块特征：解码器只返回增量，本类负责累积、分段与刷新快照。
// 没有新 token 时直接返回：段长、快照都不会变，没必要重复写。
void AsrSession::on_block(const AudioFeatureInput& item) {
    const std::vector<int64_t> ids = decoder_.accept_features(item.block);
    if (ids.empty()) {
        return;
    }

    all_ids_.insert(all_ids_.end(), ids.begin(), ids.end());
    segment_ids_.insert(segment_ids_.end(), ids.begin(), ids.end());

    // 当前段满 64 个就结算；解码器不 reset，跨段声学上下文保持连续。
    if (segment_ids_.size() >= cfg_.segment_tokens) {
        close_segment();
    }

    // 显示快照取解码器的尾部窗口；段边界不 reset，窗口本身跨段滚动。
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    snapshot_ = decoder_.recent_text();
}

// 人为切断当前段：不足 segment_tokens 也结算。与 on_block 走同一套
// close_segment()，所以段计数、落盘、清空语义完全一致。
void AsrSession::settle_segment() { close_segment(); }

// 主线程随时可读；锁内只做一次字符串拷贝，不等待推理。
std::string AsrSession::latest_text() const {
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    return snapshot_;
}

// 收尾：结算尾段 → 关闭文件 → 对全量 token 算一次最终文本。
// 调用前必须已经 finish_and_join()，否则会与后台线程并发访问 token 数组。
std::string AsrSession::finish() {
    close_segment();
    if (transcript_.is_open()) {
        transcript_.flush();
        transcript_.close();
    }
    return decoder_.text_of(all_ids_);
}

// 段结算：空段直接返回；否则转文本、追加一行并 flush，然后清空当前段。
void AsrSession::close_segment() {
    if (segment_ids_.empty()) {
        return;
    }
    const std::string text = decoder_.text_of(segment_ids_);
    append_transcript(text);
    segment_ids_.clear();
    ++n_segments_;
}

// 追加一行并 flush：崩溃最多丢当前未结算的尾巴，已结算段都在文件里。
// 写失败只告警一次（之后 transcript_ok_ 为 false），不抛出、不中断解码。
void AsrSession::append_transcript(const std::string& text) {
    if (!transcript_ok_) {
        return;
    }
    transcript_ << to_single_line(text) << '\n';
    transcript_.flush();
    if (!transcript_) {
        transcript_ok_ = false;
        transcript_error_ = "写入转写文件失败：" + cfg_.transcript_path;
        std::fprintf(stderr, "[ASR] 转写写入失败，后续段落不再落盘：%s\n",
                     cfg_.transcript_path.c_str());
    }
}

}  // namespace audio_ai

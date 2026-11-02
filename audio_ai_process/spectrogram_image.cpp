/**
 * @file spectrogram_image.cpp
 * @brief SpectrogramImage 实现：LogFBank 特征 → BGR 滚动语谱图。
 *
 * 与 Python 版（src/spectrogram_image.py）的对应关系：
 *   accept_features() ↔ 同名方法；frame() ↔ 同名方法；
 *   calibrate() ↔ _calibrate()；header() ↔ _header()。
 *
 * 关键对齐点：
 *   1. 特征 (时间, Mel) → 图像 (Mel, 时间) 需要转置；
 *   2. 再上下翻转，让 Mel 0 落在最后一行（画面底部）；
 *   3. 滚动窗口右对齐：新帧永远贴右端，左边补黑；
 *   4. 色阶只标定一次，之后锁死，避免画面随音量呼吸。
 */

#include "spectrogram_image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace audio_ai {

namespace {

// 与 numpy percentile(..., method="linear") 一致的线性插值百分位。
// 输入必须已升序排列。
double percentile_linear(const std::vector<float>& sorted, double p) {
    if (sorted.empty()) {
        return 0.0;
    }
    const double index =
        static_cast<double>(sorted.size() - 1) * (p / 100.0);
    const size_t lo = static_cast<size_t>(std::floor(index));
    const size_t hi = static_cast<size_t>(std::ceil(index));
    const double frac = index - static_cast<double>(lo);
    return static_cast<double>(sorted[lo]) * (1.0 - frac) +
           static_cast<double>(sorted[hi]) * frac;
}

// 用一组特征的百分位估色阶 (lo, hi)。
// p_lo/p_hi 取 5/99.5：避开个别尖峰把上限拉飞。
// 动态范围低于 min_range 时认为「几乎没声音」，退回 fallback，
// 否则会把数值噪声放大成一片雪花。
std::pair<double, double> estimate_levels(const std::vector<float>& feats,
                                          double fallback_lo, double fallback_hi) {
    if (feats.empty()) {
        return {fallback_lo, fallback_hi};
    }

    std::vector<float> sorted = feats;
    std::sort(sorted.begin(), sorted.end());

    const double lo = percentile_linear(sorted, 5.0);
    const double hi = percentile_linear(sorted, 99.5);
    constexpr double kMinRange = 6.0;
    if (hi - lo < kMinRange) {
        return {fallback_lo, fallback_hi};
    }
    return {lo, hi};
}

}  // namespace

// ==================== 构造与重置 ====================

// 构造：保存画布参数，把标签压成 ASCII（cv::putText 不支持中文，
// 不处理会静默输出问号），清空特征缓存并按 calibrate_frames 决定初始标定状态。
SpectrogramImage::SpectrogramImage(int n_mels, int cols, double frame_shift,
                                   double level_lo, double level_hi,
                                   int calibrate_frames, cv::Size plot_size,
                                   int header_h, int footer_h, int colormap,
                                   std::string label)
    : n_mels_(n_mels),
      cols_(std::max(1, cols)),
      frame_shift_(frame_shift),
      plot_size_(plot_size),
      header_h_(header_h),
      footer_h_(footer_h),
      colormap_(colormap),
      label_(std::move(label)),
      calibrate_frames_(calibrate_frames),
      level_lo_(level_lo),
      level_hi_(level_hi),
      level0_lo_(level_lo),
      level0_hi_(level_hi),
      calibrated_(calibrate_frames <= 0) {
    // 非 ASCII 替换成 '?'，与 Python 的 encode("ascii", "replace") 一致。
    for (char& c : label_) {
        if (static_cast<unsigned char>(c) >= 0x80) {
            c = '?';
        }
    }
}

// 清空特征缓存，色阶回到构造时的值，重新进入「未标定」状态。
void SpectrogramImage::reset() {
    feat_buf_.clear();
    n_total_ = 0;
    level_lo_ = level0_lo_;
    level_hi_ = level0_hi_;
    calibrated_ = calibrate_frames_ <= 0;
}

// ==================== 对外接口 ====================

// 喂一块特征：追加进滚动缓存（只留最近 cols_ 帧），然后重画整幅画布。
// 0 帧也允许，此时只按现有缓存重画（刷新窗口用）。
// 前置条件：feats.data.size() == n_frames * n_mels，且 n_mels 与构造值一致。
cv::Mat SpectrogramImage::accept_features(const LogFbankFeatures& feats) {
    if (feats.n_mels != n_mels_) {
        throw std::invalid_argument(
            "SpectrogramImage: feats.n_mels must equal constructor n_mels");
    }
    if (feats.data.size() !=
        static_cast<size_t>(feats.n_frames) * static_cast<size_t>(feats.n_mels)) {
        throw std::invalid_argument(
            "SpectrogramImage: feats.data size does not match n_frames*n_mels");
    }

    if (feats.n_frames > 0) {
        feat_buf_.insert(feat_buf_.end(), feats.data.begin(), feats.data.end());
        n_total_ += feats.n_frames;

        // 只保留最近 cols_ 帧，旧帧立刻回收。
        const int buffered = buffered_frames();
        if (buffered > cols_) {
            const size_t drop =
                static_cast<size_t>(buffered - cols_) * static_cast<size_t>(n_mels_);
            feat_buf_.erase(feat_buf_.begin(), feat_buf_.begin() + drop);
        }
    }

    return frame();
}

// 按当前缓存渲染一整幅画布。
// 流程与 Python 一致：特征 → 灰度图（转置 + 翻转 + 右对齐补黑）→ 放大
// → viridis 伪彩 → 边框 → 顶部信息条 → 可选底部白条。
cv::Mat SpectrogramImage::frame() {
    calibrate();

    const int n_frames = buffered_frames();

    // 先建一张 cols 宽的黑色画布，把特征从右端贴入：
    // 左边空出来的是「还没到的时间」。
    cv::Mat gray(n_mels_, cols_, CV_8U, cv::Scalar(0));
    const double range = level_hi_ - level_lo_;
    const double scale = range > 0.0 ? 255.0 / range : 0.0;

    for (int i = 0; i < n_frames; ++i) {
        const int col = cols_ - n_frames + i;
        for (int m = 0; m < n_mels_; ++m) {
            const double value =
                static_cast<double>(feat_buf_[static_cast<size_t>(i) * n_mels_ + m]);
            double pixel = (value - level_lo_) * scale;
            pixel = std::clamp(pixel, 0.0, 255.0);

            // 行号翻转：Mel 0 落在最后一行（画面底部）。
            const int row = n_mels_ - 1 - m;
            gray.at<uchar>(row, col) = static_cast<uchar>(pixel);
        }
    }

    // 80 行太矮，先按显示尺寸放大，再上 viridis 伪彩。
    cv::Mat resized;
    cv::resize(gray, resized, plot_size_, 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat plot;
    cv::applyColorMap(resized, plot, colormap_);
    cv::rectangle(plot, cv::Point(0, 0),
                  cv::Point(plot_size_.width - 1, plot_size_.height - 1),
                  cv::Scalar(0, 0, 0), 1);

    // 顶部信息条、语谱图、可选底部白条自上而下拼接。
    std::vector<cv::Mat> parts;
    if (header_h_ > 0) {
        parts.push_back(header());
    }
    parts.push_back(plot);

    if (footer_h_ > 0) {
        cv::Mat footer(footer_h_, plot_size_.width, CV_8UC3,
                       cv::Scalar(255, 255, 255));
        cv::line(footer, cv::Point(0, 0),
                 cv::Point(plot_size_.width - 1, 0), cv::Scalar(0, 0, 0), 1);
        parts.push_back(footer);
    }

    cv::Mat canvas;
    cv::vconcat(parts, canvas);
    return canvas;
}

// ==================== 内部 ====================

// 攒够 calibrate_frames_ 帧后按实际数据标定一次色阶，之后锁死。
// 标定用的是滚动缓存（最多 cols_ 帧），与 Python 一致。
void SpectrogramImage::calibrate() {
    if (calibrated_ || n_total_ < calibrate_frames_) {
        return;
    }
    const auto levels = estimate_levels(feat_buf_, level_lo_, level_hi_);
    level_lo_ = levels.first;
    level_hi_ = levels.second;
    calibrated_ = true;
}

// 顶部白色信息条：两行 ASCII 文字 + 底部黑分隔线。
// 第一行是标签，第二行是时间 / 窗口 / Mel 数 / 当前色阶。
cv::Mat SpectrogramImage::header() const {
    const int width = plot_size_.width;
    const int height = header_h_;
    if (height <= 0) {
        return cv::Mat();
    }

    cv::Mat header(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    char line1[256];
    std::snprintf(line1, sizeof(line1), "%s",
                  label_.empty() ? "spectrogram" : label_.c_str());

    char line2[256];
    std::snprintf(line2, sizeof(line2),
                  "t=%5.2fs  win=%.2fs  n_mels=%d  level=[%.2f, %.2f]",
                  static_cast<double>(n_total_) * frame_shift_,
                  static_cast<double>(cols_) * frame_shift_, n_mels_,
                  level_lo_, level_hi_);

    cv::putText(header, line1, cv::Point(8, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    if (height >= 48) {
        cv::putText(header, line2, cv::Point(8, 42), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
    cv::line(header, cv::Point(0, height - 1), cv::Point(width - 1, height - 1),
             cv::Scalar(0, 0, 0), 1);
    return header;
}

// 当前缓存里的帧数。
int SpectrogramImage::buffered_frames() const {
    return n_mels_ > 0 ? static_cast<int>(feat_buf_.size() / n_mels_) : 0;
}

}  // namespace audio_ai

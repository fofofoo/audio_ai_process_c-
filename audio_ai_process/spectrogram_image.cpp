/**
 * @file spectrogram_image.cpp
 * @brief 语谱图：LogFBank 特征 → 可直接 imshow 的 BGR 画布。
 *
 * 模块边界（对应 Python 版 src/spectrogram_image.py）：
 *   - 只接收特征，不碰波形、不建 AudioPreprocess、不做解码；
 *   - 不负责写识别文字：顶部信息条和可选底部白条只提供背景，
 *     写什么、怎么写在入口里用 cv::putText 决定；
 *   - 特征缓存、色阶标定、画布拼接都在本类内部完成，入口每喂一块拿一整幅图。
 *
 * 典型用法（前端由入口自己装配）：
 *     SpectrogramImage view(80, 400, 0.01, ...);
 *     cv::Mat canvas = view.accept_features(front.accept_waveform(blk));
 *     cv::imshow("spectrogram", canvas);
 *
 * 坐标与朝向（最容易画反的地方）：
 *   特征矩阵形状是 (时间, Mel)，图像形状是 (行, 列) = (y, x)，要做两次变换：
 *     1. 转置：帧号变成列，Mel 通道变成行；
 *     2. 上下翻转：让 Mel 0（最低频）落在最后一行，也就是画面**底部**。
 *   OpenCV 的 row 0 在顶部；少了第 2 步会得到一张「高音在下」的反图，
 *   而且不会报错，只会让人对着颠倒的图分析半天。
 *   纵轴是 Mel 通道序号，不是线性 Hz：Mel 滤波器低频密、高频疏，
 *   所以本模块不画 Hz 刻度，避免给出「看起来精确、实际有偏」的刻度。
 *
 * 色阶：特征单位是 ln(能量)，静音帧停在地板 ln(float32_eps) ≈ -15.94。
 *   映射是一次线性拉伸 + 截断：
 *     pixel = clip((value - lo) / (hi - lo) * 255, 0, 255)
 *   前 calibrate_frames 帧（默认 50 ≈ 0.5 秒）按实际数据标定一次后锁死；
 *   否则安静段和说话段会互相挤压亮度，画面一直「呼吸」。
 *   calibrate_frames=0 表示完全不标定，一直用构造传入的固定色阶。
 *   注意 lo/hi 是 ln(能量)，不是 dB，不要和 10·log10 混用。
 *
 * 尾部补帧（调用方注意）：
 *   AudioPreprocess::input_finished() 默认在尾部补 100 帧特征域 0（对应能量 1），
 *   在图上是一块**比静音还亮**的色块（静音地板 -15.94，0 比它高），纯属视觉噪声。
 *   可视化路径构造前端时必须传 tail_pad_frames=0；收尾想保留最后一屏就调用
 *   frame()，不要为了出图去喂 input_finished() 补出来的 0 特征。
 *
 * 滚动缓存与画布组成：
 *   内部只保留最近 cols 帧（400 帧 × 10 ms = 4 秒）；
 *   新帧永远贴右端，左边空出来的是「还没到的时间」，补黑。
 *   每帧画布 = 顶部信息条（白色）+ 语谱图 + 可选底部白条。
 *
 * 与 Python 的对应关系：
 *   accept_features() / frame() / reset() / calibrate() / header() 一一对应；
 *   Python 的 features_to_image()（转置 + 翻转 + 线性拉伸）在 C++ 里
 *   直接内联到 frame() 的逐像素循环中，少一次中间矩阵分配。
 */

#include "spectrogram_image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace audio_ai {

// kDefaultLevelLo / kDefaultLevelHi 定义在 .h：下限取 kaldi 的静音地板
// ln(float32 eps) ≈ -15.94，这样「没声音」就是最暗；上限 8.0 是经验值，
// 只作为「还没标定 / 标定失败」时的兜底，不参与正常运行时的映射。

namespace {

// 与 numpy percentile(..., method="linear") 一致的线性插值百分位。
//   位置 index = (n - 1) * p / 100，落在两个相邻样本之间时按小数部分插值。
//   必须用 (n-1) 而不是 n：numpy 默认 linear 方法就是按这个口径定义分位数。
//   输入必须已升序排列；p 以百分数给出（5 表示 5%）。
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

// 用一组特征的百分位估色阶 (lo, hi)，单位是 ln(能量)。
//   - p5 / p99.5：避开个别尖峰把上限拉飞。不用 min/max 是因为一次爆音
//     就能让后面整幅图变暗；
//   - min_range = 6.0：动态范围低于 6（ln 能量差）时认为「几乎没声音」，
//     放弃这组数据退回 fallback；对着一小段静音强行拉伸，只会把数值噪声
//     放大成一片雪花；
//   - 空输入同样退回 fallback，保证调用方永远拿到一对可用色阶。
// 这里把 p / min_range 写成固定口径：C++ 版只有「启动标定一次」这一条路径，
// 没有 Python 静态图那种按单批数据自适应的需求。
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

// 构造：只保存画布参数并预置状态，不分配图像、不做标定。
//   - cols <= 0 时夹到 1：cols 同时是画布宽度和缓存上限，0 会得到空画布；
//   - label 非 ASCII 替换成 '?'：cv::putText 只支持 ASCII，中文会静默变问号；
//   - level0_lo_/hi_ 单独留一份：reset() 要回到**构造时**的兜底色阶，
//     而不是上一次标定后的值；
//   - calibrate_frames <= 0 表示永不标定，构造完就直接视为「已标定」；
//     否则先按未标定处理，等攒够帧数再标定。
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

// 清空滚动缓存和累计帧数，色阶恢复到构造值，重新进入「未标定」状态。
// 只重置运行期状态；构造参数不可变，也不需要重建任何滤波器/查表。
// 重置后下一帧画布会从黑底重新开始（左边全是「还没到的时间」）。
void SpectrogramImage::reset() {
    feat_buf_.clear();
    n_total_ = 0;
    level_lo_ = level0_lo_;
    level_hi_ = level0_hi_;
    calibrated_ = calibrate_frames_ <= 0;
}

// ==================== 对外接口 ====================

// 喂一块特征：追加进滚动缓存（只留最近 cols_ 帧），然后重画整幅画布。
//   - n_frames == 0 合法：不追加数据，只按现有缓存重画（入口没新音频时刷新窗口）；
//   - 先校验 n_mels 和 data 长度，形状错了直接抛异常，避免后续越界读；
//   - 每喂一块返回的是**整幅**画布，不是增量条带；调用方拿到即可 imshow / 存图 /
//     继续用 cv::putText 写文字；
//   - 裁剪用 erase 而不是保留视图，让旧帧占用的内存真正回收。
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

// 按当前缓存渲染一整幅画布（不接收新特征）。
// 流程与 Python 的 features_to_image() + frame() 一致：
//   1. 建 cols 宽的黑底灰度图，特征从右端贴入（右对齐滚动）；
//   2. 逐帧逐 Mel 做线性拉伸 + 截断，同时翻转行号让 Mel 0 落在底部；
//   3. 80 行太矮，先放大到 plot_size（先放大再上色，色带边缘更平滑）；
//   4. applyColorMap 转 viridis，再画 1 像素黑边框；
//   5. 顶部信息条 + 语谱图 + 可选底部白条纵向拼接。
// 收尾时想显示最后一屏就调用它；不要为了出图去喂 input_finished() 的补 0 特征。
cv::Mat SpectrogramImage::frame() {
    calibrate();

    const int n_frames = buffered_frames();

    // 先建一张 cols 宽的黑色画布，把特征从右端贴入：
    // 左边空出来的是「还没到的时间」。
    cv::Mat gray(n_mels_, cols_, CV_8U, cv::Scalar(0));

    // 线性映射到 0~255。range <= 0 属于调用方给了反的/相等的色阶，
    // 正常路径不会出现；这里退化成 scale=0，整幅图落黑，至少不会除零。
    const double range = level_hi_ - level_lo_;
    const double scale = range > 0.0 ? 255.0 / range : 0.0;

    for (int i = 0; i < n_frames; ++i) {
        // 右对齐：缓存的第 i 帧落在 cols_ - n_frames + i；
        // i = n_frames-1（最新帧）正好贴最右端。
        const int col = cols_ - n_frames + i;
        for (int m = 0; m < n_mels_; ++m) {
            const double value =
                static_cast<double>(feat_buf_[static_cast<size_t>(i) * n_mels_ + m]);
            // 线性拉伸后截断，不回绕：低于 lo 的保持纯黑，高于 hi 的保持纯白。
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

    // 底部白条只提供背景和顶边黑线，不放任何文字：
    // 写什么（识别结果、状态提示）由入口决定，本类不猜。
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
//   - 判据用累计帧数 n_total_ 而不是缓存长度 buffered_frames()：
//     缓存超过 cols_ 后会滚动裁剪，但它只影响「用哪些数据标定」，不该让
//     标定条件永远凑不够（例如 cols_ 小于 calibrate_frames_ 时）；
//   - 标定样本是当前滚动缓存（最多 cols_ 帧），与 Python 版一致；
//   - estimate_levels 在数据太平/为空时退回当前色阶，所以标定失败也不会
//     把画面弄坏，只是继续用兜底值；
//   - 只在第一次满足条件时执行，之后 calibrated_ 永久为 true：
//     每帧重新标定会让亮度和色带随内容来回跳，看起来像画面在「呼吸」。
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
//   - 第一行是标签（构造时传入，非 ASCII 已在构造里替换成 '?'）；
//   - 第二行是 t / 窗口长度 / Mel 数 / 当前色阶：
//       t  = 累计喂入帧数 × frame_shift_，是**音频时间**（文件里的位置），
//            不是墙钟时间；1x 实时播放时两者接近，--fast 回放时会超前；
//       win = cols_ × frame_shift_，即滚动窗口覆盖的音频时长；
//       level = 当前正在使用的 (lo, hi)，标定前后会变化，方便核对；
//   - header_h_ 小于 48 时只画第一行，避免文字挤在一起或画出边界；
//   - 末尾那条黑线把信息条和语谱图分开，与 Python 版一致。
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

// 当前缓存里的帧数（行主序：元素数 / n_mels_）。
// n_mels_ <= 0 时返回 0 而不是除零；正常路径构造参数就是正数。
int SpectrogramImage::buffered_frames() const {
    return n_mels_ > 0 ? static_cast<int>(feat_buf_.size() / n_mels_) : 0;
}

}  // namespace audio_ai

#pragma once

/**
 * @file spectrogram_image.h
 * @brief 语谱图：LogFBank 特征 → 可直接 imshow 的 BGR 画布（C++ 版）。
 *
 * 职责边界（对齐 src/spectrogram_image.py）：
 *   - 只接收特征，不碰波形、不建 AudioPreprocess、不做解码；
 *   - 内部维护最近 cols 帧的滚动缓存，每次喂特征返回一整幅 BGR 画布；
 *   - 不负责写识别文字：顶部信息条只画参数，其他文字由调用方用 cv::putText 叠加。
 *
 * 坐标与朝向：
 *   特征形状是 (时间, Mel)，图像形状是 (行, 列) = (y, x)。这里做两次变换：
 *   转置让时间变成列，上下翻转让 Mel 0（最低频）落在画面底部。
 *   OpenCV 的 row 0 在顶部，少了翻转会得到一张「高音在下」的反图，且不会报错。
 *
 * 色阶：
 *   特征单位是 ln(能量)，静音停在地板 −15.94。映射是一次线性拉伸 + 截断：
 *     pixel = clip((value − lo) / (hi − lo) × 255, 0, 255)
 *   前 calibrate_frames 帧用百分位标定一次后锁死，避免画面随音量「呼吸」；
 *   calibrate_frames=0 表示不标定，一直用构造传入的色阶。
 *
 * 画布组成（自上而下）：
 *   顶部信息条（header_h，白底 ASCII）+ 语谱图（plot_size，viridis）
 *   + 可选底部白条（footer_h，留给调用方写文字）。
 */

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "audio_preprocess.h"

namespace audio_ai {

// 默认色阶（对数能量域）：下限取 kaldi 静音地板 ln(float32 eps) ≈ −15.9423847，
// 上限是经验值，只作为「还没标定 / 标定失败」时的兜底。
constexpr double kDefaultLevelLo = -15.942384719848633;
constexpr double kDefaultLevelHi = 8.0;

/**
 * @brief 特征 → BGR 滚动语谱图画布。
 *
 * 线程模型：非线程安全，与前端一起放在同一线程（入口主循环）调用。
 */
class SpectrogramImage {
public:
    /**
     * @brief 构造画布参数；内部特征缓存为空。
     * @param n_mels            Mel 通道数，也是灰度图行数
     * @param cols              滚动窗口宽度（帧），同时决定画布固定宽度
     * @param frame_shift       一帧对应的时间（秒），用于顶部时间显示
     * @param level_lo/hi       兜底色阶 (lo, hi)，单位为 ln(能量)
     * @param calibrate_frames  收到多少帧后标定一次并锁死；0 = 不标定
     * @param plot_size         语谱图显示尺寸（宽, 高）；80 行太矮需要放大
     * @param header_h          顶部白色信息条高度；0 = 不画
     * @param footer_h          底部白色条高度；0 = 不预留
     * @param colormap          OpenCV colormap，默认 viridis
     * @param label             信息条第一行；非 ASCII 字符会被替换成 '?'
     */
    explicit SpectrogramImage(int n_mels = 80,
                              int cols = 400,
                              double frame_shift = 0.01,
                              double level_lo = kDefaultLevelLo,
                              double level_hi = kDefaultLevelHi,
                              int calibrate_frames = 50,
                              cv::Size plot_size = cv::Size(1200, 300),
                              int header_h = 52,
                              int footer_h = 0,
                              int colormap = cv::COLORMAP_VIRIDIS,
                              std::string label = "");

    /**
     * @brief 喂一块特征，返回**整幅 BGR 画布**（可直接 imshow）。
     * @param feats 特征块；n_frames 可以为 0，此时只重画当前缓存。
     * @return CV_8UC3 画布，尺寸 (header_h + plot_h + footer_h, plot_w)。
     * @throws std::invalid_argument feats.n_mels 与构造时的 n_mels 不一致。
     */
    cv::Mat accept_features(const LogFbankFeatures& feats);

    /**
     * @brief 按当前缓存渲染一帧画布（不接收新特征）。
     *
     * 收尾时想保留最后一屏就调用它；不要为了出图去喂 input_finished()
     * 补出来的 0 特征，那会在图上形成一块比静音还亮的假尾。
     */
    cv::Mat frame();

    /// 清空特征缓存，色阶回到构造时的值。
    void reset();

    /// 当前缓存内的特征（行主序，只读）；滚动模式下只含最近 cols 帧。
    const std::vector<float>& features() const { return feat_buf_; }

    /// 累计接收的帧数（不受滚动裁剪影响）。
    int64_t n_frames() const { return n_total_; }

    // —— 只读参数 ——
    int n_mels() const { return n_mels_; }
    int cols() const { return cols_; }
    double level_lo() const { return level_lo_; }
    double level_hi() const { return level_hi_; }

private:
    // 攒够 calibrate_frames_ 帧后标定一次色阶，之后不再改。
    void calibrate();

    // 顶部白色信息条（两行 ASCII + 底部黑分隔线）。
    cv::Mat header() const;

    // 特征缓存里当前有多少帧。
    int buffered_frames() const;

private:
    // 构造参数
    int n_mels_ = 80;
    int cols_ = 400;
    double frame_shift_ = 0.01;
    cv::Size plot_size_{1200, 300};
    int header_h_ = 52;
    int footer_h_ = 0;
    int colormap_ = cv::COLORMAP_VIRIDIS;
    std::string label_;
    int calibrate_frames_ = 50;

    // 色阶
    double level_lo_ = kDefaultLevelLo;
    double level_hi_ = kDefaultLevelHi;
    double level0_lo_ = kDefaultLevelLo;   ///< 构造时的兜底色阶，reset() 用
    double level0_hi_ = kDefaultLevelHi;
    bool calibrated_ = false;

    // 特征缓存：行主序，最多 cols_ 帧
    std::vector<float> feat_buf_;
    int64_t n_total_ = 0;
};

}  // namespace audio_ai

/**
 * @file audio_preprocess.cpp
 * @brief AudioPreprocess 实现：PCM 波形块 → (n, n_mels) LogFBank 特征。
 *
 * 与头文件的分工：接口契约、参数含义都在 .h，本文件只写实现层面的东西——
 * 分帧的绝对索引、补齐的边界条件、7 步流水线的逐行对应关系。
 *
 * 与 Python 版（src/audio_preprocess.py）的对应关系：
 *   accept_waveform() ↔ 同名方法；input_finished() ↔ 同名方法；
 *   can_emit/get_frame/pad_mode/left_pad/right_pad/trim/proc_frame
 *   一一对应，浮点运算顺序也保持一致，便于逐点对拍。
 *
 * FFT 选型：直接用 OpenCV 的 cv::dft（项目已依赖 OpenCV），
 * 不再引入 kissfft / pffft；512 点 CV_64F 的精度与 numpy rfft 同量级。
 */

#include "audio_preprocess.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <opencv2/core.hpp>

namespace audio_ai {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Hz → Mel（kaldi / HTK 口径）：1127·ln(1 + f/700)。
// 与 2595·log10(1 + f/700) 数学等价，但照抄 kaldi 的常数才能逐点对齐。
double hz_to_mel(double hz) {
    return 1127.0 * std::log(1.0 + hz / 700.0);
}

}  // namespace

// ==================== 构造与重置 ====================

// 构造：由采样率与毫秒参数算出帧长/帧移（截断，与 kaldi WindowSize 一致），
// 预计算 povey 窗与 Mel 滤波器组，然后把流式状态清零。
// 边界：n_fft 必须不小于帧长，否则 rfft 语义不成立，直接抛异常。
AudioPreprocess::AudioPreprocess(int sample_rate, int n_mels, double frame_len_ms,
                                 double frame_shift_ms, int n_fft, double preemph,
                                 double low_freq, double high_freq,
                                 int tail_pad_frames)
    : sample_rate_(sample_rate),
      n_mels_(n_mels),
      n_fft_(n_fft),
      preemph_(preemph),
      low_freq_(low_freq),
      high_freq_(high_freq),
      tail_pad_frames_(tail_pad_frames) {
    // kaldi 的 FrameExtractionOptions::WindowSize() 是截断而不是四舍五入：
    // 16000 * 0.001 * 25 = 400，16000 * 0.001 * 10 = 160。
    frame_size_ = static_cast<int>(sample_rate_ * 0.001 * frame_len_ms);
    frame_step_ = static_cast<int>(sample_rate_ * 0.001 * frame_shift_ms);
    n_freq_ = n_fft_ / 2 + 1;

    if (n_fft_ < frame_size_) {
        throw std::invalid_argument("AudioPreprocess: n_fft must be >= frame_size");
    }

    // 方案 B（同 Python）：第 m 帧起点 = m*step − 120，首帧等 17.5 ms，
    // 左补 120 点；与 kaldi snip_edges=False 的行为一致。
    first_ = -120;
    left_pad_ = -first_;

    window_ = build_window();
    build_mel_filterbank();
    reset();
}

// 重置：清空样本缓存与计数，回到「一个样本都没喂过」。
// 窗与 Mel 滤波器组只依赖构造参数，不在此重建。
void AudioPreprocess::reset() {
    samples_.clear();
    base_ = 0;
    n_samples_ = 0;
    next_frame_ = 0;
    finished_ = false;
    total_frames_ = 0;
}

// ==================== 对外接口 ====================

// 喂入一块波形：追加进样本缓存，然后 while 排空所有已经凑满的帧。
// 返回本次新产出的帧；块长任意，凑不够一帧时返回 n_frames == 0。
// 调用后缓存尾部最多留下不足一帧的样本（< 400 点），内存有界。
LogFbankFeatures AudioPreprocess::accept_waveform(const std::vector<float>& samples) {
    if (!samples.empty()) {
        samples_.insert(samples_.end(), samples.begin(), samples.end());
        n_samples_ += static_cast<int64_t>(samples.size());
    }

    LogFbankFeatures out;
    out.n_mels = n_mels_;

    // while 而不是 if：一次喂入多块音频时要把积压的帧全部排空。
    while (can_emit()) {
        const std::vector<double> frame = get_frame(next_frame_);
        const std::vector<double> mel = proc_frame(frame);
        out.data.reserve(out.data.size() + mel.size());
        for (double value : mel) {
            out.data.push_back(static_cast<float>(value));
        }
        ++next_frame_;
        trim();  // 丢弃后续帧再也读不到的旧样本
    }

    out.n_frames = static_cast<int>(out.data.size() / n_mels_);
    return out;
}

// 收尾：按批量公式 round(N/step) 补出剩余帧，再无条件补 tail_pad_frames 帧 0。
// 补的 0 是特征域的 0（log 域对应能量 1），不是静音值——这是给解码器凑 chunk 的，
// 可视化路径要把 tail_pad_frames 传 0，否则图上会多一块假亮尾。
// 重复调用（finished_ 已为 true）直接返回空特征。
LogFbankFeatures AudioPreprocess::input_finished() {
    LogFbankFeatures out;
    out.n_mels = n_mels_;

    if (finished_) {
        return out;
    }
    finished_ = true;

    // 批量版帧数公式：round(N / Frame_step_len)。
    total_frames_ = (n_samples_ + frame_step_ / 2) / frame_step_;

    while (next_frame_ < total_frames_) {
        const std::vector<double> frame = get_frame(next_frame_);
        const std::vector<double> mel = proc_frame(frame);
        for (double value : mel) {
            out.data.push_back(static_cast<float>(value));
        }
        ++next_frame_;
    }

    if (tail_pad_frames_ > 0) {
        out.data.insert(out.data.end(),
                        static_cast<size_t>(tail_pad_frames_) * n_mels_, 0.0f);
    }

    out.n_frames = static_cast<int>(out.data.size() / n_mels_);
    return out;
}

// ==================== 出帧 ====================

// 现在能不能把 next_frame_ 号帧定下来。
// 流未结束时只放 hi <= n 的帧（不右补）；收尾后帧数上界由批量公式给出，
// 允许右补齐。判据用绝对样本数 n_samples_，与块长无关。
bool AudioPreprocess::can_emit() const {
    if (finished_) {
        return next_frame_ < total_frames_;
    }
    const int64_t next_frame_end =
        next_frame_ * frame_step_ + first_ + frame_size_;
    return n_samples_ >= next_frame_end;
}

// 取第 m 帧的 frame_size_ 个原始样本，越界处按 pad_mode 补齐。
// 只补这一帧自己越界的部分；镜像长度固定，逐帧补与整段补结果相同。
// 正常帧直接从 samples_ 切片；m=0 或流末尾才走补齐分支。
std::vector<double> AudioPreprocess::get_frame(int64_t m) const {
    const int64_t lo = m * frame_step_ + first_;
    const int64_t hi = lo + frame_size_;
    const int64_t abs_start = std::max<int64_t>(lo, 0);
    const int64_t abs_end = std::min<int64_t>(hi, n_samples_);

    // 真实已有样本这一段；缓存只保存 [base_, base_+size)，所以先减 base_。
    std::vector<double> seg;
    if (abs_end > abs_start) {
        const int64_t from = abs_start - base_;
        const int64_t to = abs_end - base_;
        seg.assign(samples_.begin() + from, samples_.begin() + to);
    }

    const size_t nl = static_cast<size_t>(abs_start - lo);  // 左补点数
    const size_t nr = static_cast<size_t>(hi - abs_end);    // 右补点数
    if (nl == 0 && nr == 0) {
        return seg;
    }

    const std::string mode = pad_mode(n_samples_);
    std::vector<double> frame;
    frame.reserve(nl + seg.size() + nr);
    if (nl > 0) {
        const std::vector<double> pad = left_pad(nl, mode);
        frame.insert(frame.end(), pad.begin(), pad.end());
    }
    frame.insert(frame.end(), seg.begin(), seg.end());
    if (nr > 0) {
        const std::vector<double> pad = right_pad(nr, mode);
        frame.insert(frame.end(), pad.begin(), pad.end());
    }
    return frame;
}

// 决定补齐方式：symmetric（镜像时重复边缘样本）或退化 edge（复制边缘）。
// 退化只在整段样本数 n <= 140（< 9 ms）时触发，复刻 numpy 镜像补齐
// 「补齐宽度不能超过原数组长度」的限制，与 kaldi 的行为一致。
std::string AudioPreprocess::pad_mode(int64_t n) const {
    const int64_t nf = (n + frame_step_ / 2) / frame_step_;
    if (n <= 0 || nf <= 0) {
        return "edge";
    }
    const int64_t right =
        std::max<int64_t>(0, (nf - 1) * frame_step_ + first_ + frame_size_ - n);
    if (left_pad_ < n && right < n) {
        return "symmetric";
    }
    return "edge";
}

// 补在帧首的 k 个点：edge 复制首样本，symmetric 取开头 k 个样本倒序
// （sig[-j] = sig[j-1]）。只有第 0 帧会走到这里，此时 base_ 必为 0。
std::vector<double> AudioPreprocess::left_pad(size_t k, const std::string& mode) const {
    if (mode == "edge") {
        return std::vector<double>(k, samples_.front());
    }
    const int64_t from = 0 - base_;
    const int64_t to = static_cast<int64_t>(k) - base_;
    std::vector<double> pad(samples_.begin() + from, samples_.begin() + to);
    std::reverse(pad.begin(), pad.end());
    return pad;
}

// 补在帧尾的 k 个点：edge 复制末样本，symmetric 取末尾 k 个样本倒序
// （sig[n+j] = sig[n-1-j]）。只有流末尾的帧会走到这里。
std::vector<double> AudioPreprocess::right_pad(size_t k, const std::string& mode) const {
    if (mode == "edge") {
        return std::vector<double>(k, samples_.back());
    }
    const int64_t to = n_samples_ - base_;
    const int64_t from = n_samples_ - static_cast<int64_t>(k) - base_;
    std::vector<double> pad(samples_.begin() + from, samples_.begin() + to);
    std::reverse(pad.begin(), pad.end());
    return pad;
}

// 丢掉后续任何一帧都读不到的旧样本。
// 下一帧起点 = next_frame_*step + first_，它之前的样本永远不会再被读。
// 用 erase 而不是保存视图，保证内存真正回收；正常运行时缓存 < 400 点。
void AudioPreprocess::trim() {
    const int64_t keep_from =
        std::max<int64_t>(0, next_frame_ * frame_step_ + first_);
    const int64_t drop = keep_from - base_;
    if (drop <= 0) {
        return;
    }
    const int64_t count =
        std::min<int64_t>(drop, static_cast<int64_t>(samples_.size()));
    samples_.erase(samples_.begin(), samples_.begin() + count);
    base_ = keep_from;
}

// ==================== 帧内 7 步 ====================

// 一帧原始样本 → n_mels 维 LogFBank 特征（double）。
// 步骤与顺序严格照抄 Python _proc_frame：
//   1 去直流 → 2 倒序预加重 → 3 povey 窗 → 4 512 点 FFT → 5 功率谱
//   → 6 Mel 三角滤波 → 7 log(max(能量, float32_eps))
// 前置条件：frame.size() == frame_size_。
std::vector<double> AudioPreprocess::proc_frame(const std::vector<double>& frame) const {
    std::vector<double> x = frame;

    // 1) 去直流：减掉本帧均值，避免 0 Hz 直流偏置经 Mel 泄漏到低频通道。
    const double mean =
        std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
    for (double& value : x) {
        value -= mean;
    }

    // 2) 预加重：y[i] = x[i] − α·x[i−1]。
    // 必须倒序：正序会让 x[i-1] 已被改写，结果错误。
    // 帧首没有前一样本，约定用自身顶替：x[0] *= (1 − α)。
    for (size_t i = x.size() - 1; i > 0; --i) {
        x[i] -= preemph_ * x[i - 1];
    }
    x[0] *= (1.0 - preemph_);

    // 3) 加窗：povey 窗两端归零，减少帧边界不连续造成的频谱泄漏。
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] *= window_[i];
    }

    // 4) FFT：不足 n_fft 的部分补 0，实信号取正频率半边（n_freq_ 个 bin）。
    cv::Mat input(n_fft_, 1, CV_64F, cv::Scalar(0.0));
    for (size_t i = 0; i < x.size(); ++i) {
        input.at<double>(static_cast<int>(i), 0) = x[i];
    }
    cv::Mat spectrum;
    cv::dft(input, spectrum, cv::DFT_COMPLEX_OUTPUT);

    // 5) 功率谱：|X|² = re² + im²。
    std::vector<double> power(n_freq_, 0.0);
    for (int k = 0; k < n_freq_; ++k) {
        const cv::Vec2d value = spectrum.at<cv::Vec2d>(k, 0);
        power[k] = value[0] * value[0] + value[1] * value[1];
    }

    // 6)+7) Mel 滤波并取对数：每个三角只扫自己的非零 bin 区间，乘加后
    // 取 log；用 max 抬到地板而不是加 eps，静音帧才会停在 −15.94。
    std::vector<double> mel(n_mels_, 0.0);
    for (int m = 0; m < n_mels_; ++m) {
        const int lo = mel_bin_lo_[m];
        const int hi = mel_bin_hi_[m];
        double acc = 0.0;
        for (int k = lo; k < hi; ++k) {
            acc += mel_weights_[static_cast<size_t>(m) * n_freq_ + k] * power[k];
        }
        mel[m] = acc > kFloat32Eps ? std::log(acc) : std::log(kFloat32Eps);
    }
    return mel;
}

// ==================== 构造期：窗与 Mel 滤波器组 ====================

// povey 窗：(0.5 − 0.5·cos(2πn/(N−1)))^0.85。
// 分母必须是 N−1（对称窗，首尾都取到零点）；scipy 默认 periodic 用 N，
// N=400 时两者最大差 5.5e-3，足以把 log-mel 拉开可见差距。
std::vector<double> AudioPreprocess::build_window() const {
    std::vector<double> window(static_cast<size_t>(frame_size_), 0.0);
    const double a = 2.0 * kPi / static_cast<double>(frame_size_ - 1);
    for (int n = 0; n < frame_size_; ++n) {
        const double hann = 0.5 - 0.5 * std::cos(a * n);
        window[n] = std::pow(hann, 0.85);
    }
    return window;
}

// 构造 Mel 三角滤波器组（kaldi 口径）：
//   1. low/high 换到 Mel 刻度，在 Mel 域等分 n_mels+1 段，得到每个三角的
//      left / center / right；
//   2. 遍历每个整数 FFT bin，把中心频率换到 Mel；
//   3. 严格落在 (left, right) 内的 bin 才参与，按到 center 的距离在 Mel 域线性插值。
// 同时记录每个滤波器非零的 bin 区间 [lo, hi)，运行期只扫这一段。
void AudioPreprocess::build_mel_filterbank() {
    const double nyquist = 0.5 * sample_rate_;
    // high_freq > 0 直接用；0 表示 Nyquist；负数表示从 Nyquist 往回收。
    const double high =
        high_freq_ > 0.0 ? high_freq_ : nyquist + high_freq_;
    const double bin_width = sample_rate_ / static_cast<double>(n_fft_);

    const double mel_low = hz_to_mel(low_freq_);
    const double mel_high = hz_to_mel(high);
    const double mel_delta = (mel_high - mel_low) / (n_mels_ + 1);

    // 257 个 bin 的中心频率统一换成 Mel，避免 80 个滤波器重复计算。
    std::vector<double> bin_mels(static_cast<size_t>(n_freq_), 0.0);
    for (int k = 0; k < n_freq_; ++k) {
        bin_mels[k] = hz_to_mel(bin_width * k);
    }

    mel_weights_.assign(static_cast<size_t>(n_mels_) * n_freq_, 0.0);
    mel_bin_lo_.assign(n_mels_, 0);
    mel_bin_hi_.assign(n_mels_, 0);

    for (int m = 0; m < n_mels_; ++m) {
        const double left = mel_low + m * mel_delta;
        const double center = mel_low + (m + 1) * mel_delta;
        const double right = mel_low + (m + 2) * mel_delta;

        // bin_mels 单调递增，命中的 bin 一定是连续一段；用严格不等号，
        // 落在边界上的 bin 权重为 0（而不是插值出 0.5）。
        int lo = -1;
        int hi = -1;
        for (int k = 0; k < n_freq_; ++k) {
            if (bin_mels[k] > left && bin_mels[k] < right) {
                if (lo < 0) {
                    lo = k;
                }
                hi = k + 1;
            }
        }
        if (lo < 0) {
            // 三角窄到罩不住任何 bin 中心：整行全 0，区间记为空。
            continue;
        }

        for (int k = lo; k < hi; ++k) {
            const double bm = bin_mels[k];
            // 以顶点为界分两段线性插值：上升沿 0→1，下降沿 1→0。
            const double weight = bm <= center
                                      ? (bm - left) / (center - left)
                                      : (right - bm) / (right - center);
            mel_weights_[static_cast<size_t>(m) * n_freq_ + k] = weight;
        }
        mel_bin_lo_[m] = lo;
        mel_bin_hi_[m] = hi;
    }
}

}  // namespace audio_ai

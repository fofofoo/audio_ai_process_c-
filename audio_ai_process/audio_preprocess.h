#pragma once

/**
 * @file audio_preprocess.h
 * @brief 流式语音预处理：波形块 → LogFBank 特征（C++ 版，对齐 src/audio_preprocess.py）。
 *
 * 职责边界：
 *   - 只负责「PCM 波形 → (帧数, n_mels) 对数 Mel 特征」这一条链路；
 *   - 不碰设备、不碰 OpenCV、不做解码、不做重采样；
 *   - 输入块长任意，内部用绝对样本索引判断能出几帧，帧结果与块长无关。
 *
 * 七步流水线（逐帧执行，与 Python 版逐行对应）：
 *   去直流 → 逐帧倒序预加重 → povey 窗 → 512 点 FFT → 功率谱
 *   → 80 通道 Mel 三角滤波 → log(max(能量, float32_eps))
 *
 * 分帧口径（kaldi snip_edges=False）：
 *   第 m 帧覆盖绝对样本区间 [160m − 120, 160m + 280)，中心钉在 160m；
 *   越界部分按 symmetric 镜像补齐，整段样本数 n <= 140 时退化为 edge。
 *
 * 流式状态：
 *   - samples_ 只保存 [base_, base_ + samples_.size()) 这一段样本，
 *     base_ 是首样本在整条音频里的绝对序号；
 *   - next_frame_ 是下一帧的帧号，出帧后从缓存头部丢弃再也用不到的样本。
 *
 * 典型用法：
 *   AudioPreprocess front(16000, 80, 25.0, 10.0, 512, 0.97, 20.0, 0.0, 0);
 *   LogFbankFeatures feats = front.accept_waveform(chunk);   // 可能 0 帧
 *   feats = front.input_finished();                          // 收尾补尾帧
 *
 * 约束：
 *   1. 采样率、帧长/帧移、n_fft、Mel 参数都是构造参数，运行中不可改；
 *   2. 可视化路径要传 tail_pad_frames=0，解码路径保持默认 100；
 *   3. 特征按行主序存放：第 i 帧第 j 维在 data[i * n_mels + j]。
 */

#include <cstdint>
#include <string>
#include <vector>

namespace audio_ai {

// kaldi 给 Mel 能量取对数前的地板：float32 epsilon = 1.1920929e-07，
// log ≈ -15.9423847。静音帧停在这里，与 Python 版 FLOAT32_EPS 一致。
constexpr double kFloat32Eps = 1.1920928955078125e-07;

/**
 * @brief 一块 LogFBank 特征（行主序二维矩阵的扁平表示）。
 *
 * data.size() == n_frames * n_mels，第 i 帧第 j 维在 data[i * n_mels + j]。
 * 用扁平 vector 而不是 vector<vector>，便于直接交给 OpenCV / ONNX Runtime。
 */
struct LogFbankFeatures {
    int n_frames = 0;          ///< 本次返回的帧数
    int n_mels = 0;            ///< Mel 通道数
    std::vector<float> data;   ///< 行主序特征，长度 n_frames * n_mels
};

/**
 * @brief 流式 LogFBank 前端。
 *
 * 线程模型：非线程安全，所有接口应在同一线程调用（回调线程只负责把 PCM
 * 交给调用方，不要把本类直接放进设备回调里）。
 */
class AudioPreprocess {
public:
    /**
     * @brief 构造并计算窗函数与 Mel 滤波器组（各算一次，之后复用）。
     * @param sample_rate      采样率（Hz）
     * @param n_mels           Mel 通道数，也是输出特征维度
     * @param frame_len_ms     帧长（毫秒），默认 25
     * @param frame_shift_ms   帧移（毫秒），默认 10
     * @param n_fft            FFT 点数，需 >= frame_size 且为 2 的幂，默认 512
     * @param preemph          预加重系数，默认 0.97
     * @param low_freq         Mel 滤波器组下限（Hz），默认 20
     * @param high_freq        Mel 上限；>0 直接用，0 = Nyquist，<0 = Nyquist + 该值
     * @param tail_pad_frames  input_finished() 尾部补几帧特征域 0；可视化传 0
     */
    explicit AudioPreprocess(int sample_rate = 16000,
                             int n_mels = 80,
                             double frame_len_ms = 25.0,
                             double frame_shift_ms = 10.0,
                             int n_fft = 512,
                             double preemph = 0.97,
                             double low_freq = 20.0,
                             double high_freq = 0.0,
                             int tail_pad_frames = 100);

    /**
     * @brief 喂入一块波形，返回本次新凑出来的特征。
     * @param samples 任意长度的一维波形，量纲应与 Python 版一致（float [-1,1]）。
     * @return 本次产出的特征；还没凑够一帧时 n_frames == 0。
     */
    LogFbankFeatures accept_waveform(const std::vector<float>& samples);

    /**
     * @brief 宣告音频结束：补出尾部帧，再补 tail_pad_frames 帧特征域 0。
     *
     * 补的 0 是特征域的 0（log 域对应能量 1），不是静音值；重复调用返回空。
     * @return 剩余帧 + 尾部补帧；可视化路径 tail_pad_frames=0 时只补尾帧。
     */
    LogFbankFeatures input_finished();

    /// 清空全部状态，回到「一个样本都没喂过」；构造参数与滤波器组保持不变。
    void reset();

    // —— 只读参数（入口脚本打印配置用）——
    int sample_rate() const { return sample_rate_; }
    int n_mels() const { return n_mels_; }
    int frame_size() const { return frame_size_; }
    int frame_step() const { return frame_step_; }
    int n_fft() const { return n_fft_; }
    int64_t n_samples() const { return n_samples_; }   ///< 累计已喂入样本数
    int64_t n_frames() const { return next_frame_; }   ///< 累计已产出帧数

private:
    // —— 分帧与出帧 ——
    bool can_emit() const;
    std::vector<double> get_frame(int64_t frame_index) const;
    std::string pad_mode(int64_t n) const;
    std::vector<double> left_pad(size_t k, const std::string& mode) const;
    std::vector<double> right_pad(size_t k, const std::string& mode) const;
    void trim();

    // —— 帧内 7 步 ——
    std::vector<double> proc_frame(const std::vector<double>& frame) const;

    // —— 构造期：窗与 Mel 滤波器组 ——
    std::vector<double> build_window() const;
    void build_mel_filterbank();

private:
    // 构造参数（运行期不变）
    int sample_rate_ = 16000;
    int n_mels_ = 80;
    int n_fft_ = 512;
    double preemph_ = 0.97;
    double low_freq_ = 20.0;
    double high_freq_ = 0.0;
    int tail_pad_frames_ = 100;
    int frame_size_ = 400;     ///< 25 ms @16 kHz
    int frame_step_ = 160;     ///< 10 ms @16 kHz
    int n_freq_ = 257;         ///< n_fft / 2 + 1
    int first_ = -120;         ///< 第 m 帧起点 = m*step + first（与 Python 一致）
    int left_pad_ = 120;       ///< 第 0 帧左补齐点数 = -first

    // 预计算：povey 窗 (frame_size_)，Mel 权重 (n_mels_ * n_freq_) 与每行非零区间
    std::vector<double> window_;
    std::vector<double> mel_weights_;
    std::vector<int> mel_bin_lo_;
    std::vector<int> mel_bin_hi_;

    // 流式状态
    std::vector<double> samples_;   ///< 绝对区间 [base_, base_+size) 的样本
    int64_t base_ = 0;              ///< samples_[0] 的绝对样本序号
    int64_t n_samples_ = 0;         ///< 累计已喂入样本数
    int64_t next_frame_ = 0;        ///< 下一个要输出的帧号
    bool finished_ = false;         ///< input_finished() 是否已调用
    int64_t total_frames_ = 0;      ///< 收尾时按批量公式确定的帧数上界
};

}  // namespace audio_ai

/**
 * @file audio_resampler.cpp
 * @brief 整数倍降采样器的实现：Kaiser 窗低通 + 流式抽取。
 *
 * 这份代码是干什么的：
 *   把设备 PCM 原始采样率（本机实测 32 kHz）降到 ASR 前端要求的 16 kHz。
 *   采集客户端只负责「按需调用 process()」，滤波系数、历史状态、抽取相位
 *   全部封装在这里，调用方不需要知道 FIR 长度或群延迟。
 *
 * 设计取舍：
 *   - 选 FIR 而不是 IIR：线性相位、无反馈、块与块之间只需保存历史样本，
 *     不会因为参数错误而自激；代价是 191 抽头带来的约 3 ms 延迟。
 *   - 选整数倍抽取：32k→16k 恰好 2:1，不需要多相分数延迟；遇到 44.1k
 *     这类非整数比直接判 invalid，由上层拒绝启动，不偷偷近似。
 *   - 选 Kaiser 窗：阻带衰减由 β 单一参数控制，60 dB 足够压住 8 kHz 以上的
 *     折叠分量；Blackman/Hamming 在同等抽头数下过渡带更难压到 1 kHz 以内。
 *   - 用 double 累加：191 个 int16 乘加在 float 下误差约 1 LSB 量级，
 *     用 double 只是让「同输入必同输出」更容易推理，性能差异可忽略。
 *
 * 调用顺序（对照采集客户端 on_voice_data）：
 *   1) 设备回调给出 PCM16 字节 → 客户端转成 int16 样本数组
 *   2) process(samples, count) → 输出 int16 样本（32k→16k 时约为一半）
 *   3) 客户端把输出样本既写 wav、又放进取流缓冲
 *
 * 用法：
 *   AudioResampler rs(32000, 16000);
 *   if (!rs.valid()) { ... }                  // 参数不支持，调用方拒绝启动
 *   auto out = rs.process(pcm16.data(), pcm16.size());
 */

#include "audio_resampler.h"

#include <algorithm>
#include <cmath>

namespace hk_isapi {

namespace {

// 滤波器长度：191 抽头（奇数，保证有整数群延迟 (N-1)/2 = 95）。
// 在 32 kHz 输入下过渡带约 600 Hz，足够把 7.2 kHz 到 8 kHz 压下去。
constexpr size_t kFirTaps = 191;

// 截止频率相对「输出采样率」的比例：0.45 × 16000 = 7.2 kHz。
// 输出奈奎斯特是 8 kHz，留出 0.8 kHz 过渡带给有限长 FIR。
// 与 G.722.1 的 50–7000 Hz 通带基本重合，两条编码路径喂给 ASR 的带宽可比。
constexpr double kCutoffRatio = 0.45;

// Kaiser 窗形状参数：约 60 dB 阻带衰减（A=60 时 β≈0.1102×(60−8.7)≈5.65）。
constexpr double kKaiserBeta = 5.65;

constexpr double kPi = 3.141592653589793238462643383279;
constexpr double kTwoPi = 2.0 * kPi;

// 归一化直流增益：窗函数低通的原型系数之和应约为 1，
// 这里再除以实际和，消除离散化误差，保证静音/低频幅度不被整体缩放。
double sum_taps(const std::vector<double>& taps) {
    double sum = 0.0;
    for (double h : taps) {
        sum += h;
    }
    return sum;
}

}  // namespace

// 构造：校验整数倍关系并生成系数。参数不合法时只把 valid 置 false，
// 由调用方（采集客户端）拒绝启动并给出明确错误，构造函数本身不抛异常。
AudioResampler::AudioResampler(int in_rate, int out_rate)
    : m_in_rate(in_rate), m_out_rate(out_rate) {
    if (in_rate <= 0 || out_rate <= 0 || out_rate >= in_rate ||
        in_rate % out_rate != 0) {
        return;  // 只支持整数倍降采样；升采样/非整数比明确不支持
    }
    m_ratio = in_rate / out_rate;
    m_valid = true;
    build_coefficients();
    if (m_taps.empty()) {
        m_valid = false;  // 理论上不会发生，防御系数计算异常
        m_ratio = 0;
        return;
    }
    m_buf.assign(m_taps.size(), 0);
    reset();
}

// 生成 Kaiser 窗低通系数：
//   理想低通 h[m] = 2·fc·sinc(2·fc·m)，m 以输入样本为单位；
//   乘窗后归一化到直流增益 1。系数顺序 h[0] 作用于最新样本，
//   h[N−1] 作用于最旧样本，便于在环形缓冲里从当前写位置向前读。
void AudioResampler::build_coefficients() {
    // 截止频率换算成「每输入样本的周期数」：7.2 kHz / 32 kHz = 0.225。
    const double fc = kCutoffRatio * static_cast<double>(m_out_rate) /
                      static_cast<double>(m_in_rate);
    const double center = static_cast<double>(kFirTaps - 1) / 2.0;
    const double window_norm = bessel_i0(kKaiserBeta);

    m_taps.assign(kFirTaps, 0.0);
    for (size_t n = 0; n < kFirTaps; ++n) {
        const double m = static_cast<double>(n) - center;

        // 理想低通：m=0 处取极限 2·fc，其余为 sin(2π·fc·m)/(π·m)。
        double ideal = 0.0;
        if (std::fabs(m) < 1e-12) {
            ideal = 2.0 * fc;
        } else {
            ideal = std::sin(kTwoPi * fc * m) / (kPi * m);
        }

        // Kaiser 窗：w[n] = I0(β·sqrt(1−(2n/(N−1)−1)²)) / I0(β)。
        const double r = 2.0 * static_cast<double>(n) /
                             static_cast<double>(kFirTaps - 1) -
                         1.0;
        const double window = bessel_i0(kKaiserBeta * std::sqrt(std::max(0.0, 1.0 - r * r))) /
                              window_norm;

        m_taps[n] = ideal * window;
    }

    const double sum = sum_taps(m_taps);
    if (std::fabs(sum) > 1e-12) {
        for (double& h : m_taps) {
            h /= sum;
        }
    }
}

// 零阶第一类贝塞尔函数：用级数 I0(x) = Σ (x/2)^(2k) / (k!)² 计算。
// 这里只在构造期调用有限次，级数 50 项已远超 double 精度需求；
// 标准库 C++17 的 std::cyl_bessel_i 在部分编译器上不可用，自己实现更稳。
double AudioResampler::bessel_i0(double x) {
    const double half = x / 2.0;
    double sum = 1.0;
    double term = 1.0;
    for (int k = 1; k < 50; ++k) {
        term *= (half / k) * (half / k);
        sum += term;
        if (term < 1e-12 * sum) {
            break;
        }
    }
    return sum;
}

// 处理一段输入：逐样本写入环形缓冲，每 ratio 个输入算一个输出。
// 前置条件：valid() 为 true；data 可空仅当 count == 0。
// 边界：历史未填满时不产生输出（前 N−1 个输入只填状态）；
//       输出幅度超 int16 时饱和，不环绕。
std::vector<int16_t> AudioResampler::process(const int16_t* data, size_t count) {
    std::vector<int16_t> output;
    if (!m_valid || data == nullptr || count == 0) {
        return output;
    }

    const size_t taps = m_taps.size();
    output.reserve(count / static_cast<size_t>(m_ratio) + 1);

    for (size_t i = 0; i < count; ++i) {
        // 1) 写入当前位置；m_filled 记录有效历史长度，上限为 N。
        m_buf[m_pos] = data[i];
        if (m_filled < taps) {
            ++m_filled;
        }

        // 2) 历史满了且相位为 0 时输出一个点：
        //    y[t] = Σ h[k]·x[t−k]，从当前写位置向前读 k 个样本。
        //    环形缓冲回绕用「减到 0 就跳到末尾」的写法，避免取模开销。
        if (m_filled >= taps && m_phase == 0) {
            double acc = 0.0;
            size_t idx = m_pos;
            for (size_t k = 0; k < taps; ++k) {
                acc += m_taps[k] * static_cast<double>(m_buf[idx]);
                idx = (idx == 0) ? taps - 1 : idx - 1;
            }
            long value = std::lround(acc);
            value = std::max(-32768L, std::min(32767L, value));
            output.push_back(static_cast<int16_t>(value));
        }

        // 3) 推进抽取相位与写指针。相位在 ratio 个输入后回到 0，
        //    对应「每降采样 ratio 倍输出 1 个点」。
        m_phase = (m_phase + 1) % static_cast<size_t>(m_ratio);
        m_pos = (m_pos + 1 == taps) ? 0 : m_pos + 1;
    }
    return output;
}

// 清空历史与相位。系数不重建：同一实例可以反复用于多段音频
// （例如暂停后恢复），只要两段之间没有时间上的连续性要求。
void AudioResampler::reset() {
    std::fill(m_buf.begin(), m_buf.end(), 0);
    m_pos = 0;
    m_filled = 0;
    m_phase = 0;
}

}  // namespace hk_isapi

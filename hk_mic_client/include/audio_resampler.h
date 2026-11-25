#pragma once

/**
 * @file audio_resampler.h
 * @brief 流式整数倍降采样器：带抗混叠低通的抽取，当前用于 32 kHz PCM → 16 kHz。
 *
 * 职责边界：
 *   - 只做「PCM16 单声道、整数倍降采样」这一件事：in_rate / out_rate 必须
 *     同为整数且 in_rate % out_rate == 0，当前实测只有 32k→16k（2 倍抽取）。
 *   - 不做升采样、不做非整数比换算（如 44.1k→16k）：这类换算需要多相/有理
 *     重采样，参数与误差口径都不同，不在采集库的职责内。
 *   - 不依赖 ROS / OpenCV / ONNX Runtime / 海康 SDK，可单独编译与测试。
 *
 * 为什么必须带低通：
 *   - 32 kHz 音频的奈奎斯特频率是 16 kHz，直接每 2 点取 1 点会把 8–16 kHz
 *     的内容折叠回 0–8 kHz（混叠），语音里表现为沙沙声，ASR 特征会被污染。
 *   - 低通截止放在输出奈奎斯特之下（7.2 kHz），8 kHz 以上至少衰减 60 dB，
 *     折叠后的残留远低于语音能量。
 *   - 也不用「两点平均」：它在 8 kHz 处只有 -6 dB 左右，阻带衰减不够，
 *     而且会让 7 kHz 附近的语音明显变闷。
 *
 * 实现口径：
 *   - FIR 线性相位，Kaiser 窗设计，β≈5.65（约 60 dB 阻带），191 抽头；
 *     截止 7.2 kHz，过渡带约 0.6 kHz，群延迟 (N-1)/2 = 95 个输入样本
 *     ≈ 3 ms @32 kHz，对流式 ASR 可忽略。
 *   - 抽取在输出侧做：每 ratio 个输入样本算 1 个输出，卷积只算输出点，
 *     约 191 次乘加/输出样本（16k 输出 ≈ 3M MAC/s，CPU 占用可忽略）。
 *   - 内部用环形缓冲保存最近 N 个输入样本，跨 process() 调用保留状态；
 *     分块喂入与一次性喂入的输出逐点一致。
 *   - 启动阶段要先填满 191 个输入样本（约 6 ms）才产生第一个输出，
 *     因此一段音频的总输出数比「输入数 / ratio」少约 95 个点；
 *     稳态输出率仍是精确的 ratio 分频，实时流不受影响。
 *
 * 用法：
 *   AudioResampler rs(32000, 16000);          // 构造时算好系数
 *   if (!rs.valid()) { ... }                  // 参数不支持
 *   auto out = rs.process(pcm.data(), pcm.size());
 *   rs.reset();                               // 新一轮会话前清状态
 */

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hk_isapi {

/**
 * @brief 整数倍降采样器（PCM16 单声道，流式）。
 *
 * 线程模型：非线程安全；调用方在同一线程内顺序调用 process()，
 * 采集客户端把它放在回调锁内使用，因此天然串行。
 */
class AudioResampler {
public:
    /**
     * @brief 按输入/输出采样率构造并计算滤波器系数。
     * @param in_rate  输入采样率（Hz），必须 > 0
     * @param out_rate 输出采样率（Hz），必须 > 0 且能整除 in_rate
     * @note 参数不满足整数倍关系时 valid() 返回 false，不抛异常；
     *       调用方应据此拒绝启动，而不是继续用一个错误的采样率。
     */
    AudioResampler(int in_rate, int out_rate);

    /// 参数是否可用（整数倍、out < in、系数非空）
    bool valid() const { return m_valid; }

    int in_rate() const { return m_in_rate; }

    int out_rate() const { return m_out_rate; }

    /// 抽取倍数 in_rate / out_rate；参数无效时返回 0
    int ratio() const { return m_ratio; }

    /**
     * @brief 处理一段 PCM16 输入，返回本次新产生的降采样输出。
     * @param data  输入样本指针；count 为 0 时可为空
     * @param count 输入样本数（不是字节数）
     * @return 输出样本；不足一个输出周期时返回空向量
     * @note 状态（环形历史、抽取相位）跨调用保留；输出长度随输入长度
     *       累积变化，单次调用不保证是 count/ratio 的整数倍。
     */
    std::vector<int16_t> process(const int16_t* data, size_t count);

    /// 清空环形历史与抽取相位，回到「一个样本都没喂过」；系数保持不变。
    void reset();

private:
    /// 按 in_rate / out_rate 生成 Kaiser 窗低通系数并归一化到直流增益 1
    void build_coefficients();

    /// 零阶第一类贝塞尔函数，用于构造 Kaiser 窗（C++ 标准库无此函数）
    static double bessel_i0(double x);

private:
    int m_in_rate = 0;          ///< 输入采样率
    int m_out_rate = 0;         ///< 输出采样率
    int m_ratio = 0;            ///< 抽取倍数
    bool m_valid = false;       ///< 构造参数是否可用

    std::vector<double> m_taps; ///< FIR 系数，h[0] 作用于当前样本
    std::vector<int16_t> m_buf; ///< 环形历史缓冲，长度与 m_taps 相同
    size_t m_pos = 0;           ///< 环形缓冲的当前写入位置
    size_t m_filled = 0;        ///< 已填充样本数，上限为 m_buf.size()
    size_t m_phase = 0;         ///< 抽取相位：每 ratio 个输入产生 1 个输出
};

}  // namespace hk_isapi

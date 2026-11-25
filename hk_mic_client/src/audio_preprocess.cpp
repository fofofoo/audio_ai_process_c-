/**
 * @file audio_preprocess.cpp
 * @brief AudioPreprocess 实现：PCM 波形块 → (n, n_mels) LogFBank 特征。
 *
 * 与头文件的分工：接口契约、参数含义都在 .h，本文件只写实现层面的东西——
 * 分帧的绝对索引、补齐的边界条件、7 步流水线的逐行对应关系。
 *
 * 这份代码在做什么（对应 Python 模块 docstring 第 1 节）：
 *   麦克风一小块一小块送来的波形，转成模型能吃的 LogFBank 特征矩阵。
 *   每 10 ms 的音频块进来就尽量产出一帧 80 维特征；凑不满就攒着等下一块。
 *   整个过程只依赖已经到达的样本，不需要「未来样本」——这是流式（online）
 *   与批量（offline）唯一的本质差别。
 *   单帧因果延迟 = frame_size/2 = 200 个样本 = 12.5 ms：第 m 帧要等样本
 *   攒到 160m+280 才能算出来，而它的中心在 160m+80。
 *
 * 一、分帧：kaldi 的 FirstSampleOfFrame 语义
 *   帧长 400 样本（25 ms @16 kHz），帧移 160 样本（10 ms）。
 *   第 m 帧覆盖的绝对样本区间：
 *       lo = m·step + first_ = 160m − 120
 *       hi = lo + frame_size  = 160m + 280
 *   lo 可以是负数（第 0 帧就是 −120），hi 也可能越过已到达的样本数。
 *   kaldi snip_edges=False 的规矩是：越界部分用 symmetric 镜像补齐，
 *   而不是丢帧或补零——帧中心严格钉在 160m 上，每 10 ms 必出一帧。
 *   左补齐量恒为 120，且只有第 0 帧用得到（m=1 时 lo 已经是 40 > 0）。
 *
 * 二、跨块攒样本：为什么用绝对索引
 *   一帧要 400 点而一块通常只有 160 点，所以必须攒。samples_ 只保存
 *   绝对区间 [base_, base_ + samples_.size())，base_ 是 samples_[0] 在整条
 *   音频里的绝对序号。用绝对序号而不是「相对当前块」，跨块拼接才不会错位。
 *   攒够一帧立刻吐出，并把后续帧再也读不到的旧样本丢掉（trim），
 *   否则跑一晚上内存会一直涨。
 *
 * 三、一个不显然的退化分支
 *   numpy 的 symmetric 镜像要求补齐宽度小于原数组长度，太短的信号会直接报错；
 *   kaldi 在这种极短信号下退化成 edge（左右都复制边缘样本）。
 *   代入公式可算出该退化只在整段样本数 n <= 140（不到 9 ms）时触发，
 *   实时流里只可能出现在最开头那一两块。pad_mode() 复刻了这个判据。
 *
 * 四、七项必须对齐的 kaldi 口径（写错任何一项都会静默产生偏差）
 *   1. snip_edges=False 的分帧语义（见上）；
 *   2. povey 窗，且分母是 frame_size−1，不是 frame_size；
 *   3. 窗函数手算，不用 scipy（scipy 默认 periodic 约定与 kaldi 不同）；
 *   4. 逐帧预加重（不是对整段波形做），且帧内必须倒序执行；
 *   5. Mel 滤波器组遍历整数 FFT bin、在 Mel 域插值；
 *   6. 镜像补齐用 symmetric（重复边缘样本），不是 reflect；
 *   7. 取对数前的地板是 log(max(能量, float32_eps))，不是 log(能量 + eps)。
 *   进一步推导与实测依据见 python_ref/doc/sherpa-onnx-ASR预处理及后处理工作流报告.md。
 *
 * 与 Python 版（python_ref/src/audio_preprocess.py）的对应关系：
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
// 与 2595·log10(1 + f/700) 数学上完全等价（2595/ln10 ≈ 1127.01），
// 但这里照抄 kaldi 的 1127 取整常数与自然对数，才能逐点对齐它的滤波器组。
// 用 double 计算：Mel 值是后续插值的基准，精度降级会让权重出现末位差异。
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
    // 帧长决定每帧的样本数，帧移决定输出频率（10 ms 一帧）。
    frame_size_ = static_cast<int>(sample_rate_ * 0.001 * frame_len_ms);
    frame_step_ = static_cast<int>(sample_rate_ * 0.001 * frame_shift_ms);
    // 实信号 FFT 只需正频率半边：512 点 → 257 个 bin，最高到 Nyquist (fs/2)。
    n_freq_ = n_fft_ / 2 + 1;

    // dft 的定义要求 n_fft >= 信号长度；本项目固定 n_fft=512 > 400，
    // 不足部分补 0，相当于对整帧做零填充，不改变频谱包络。
    if (n_fft_ < frame_size_) {
        throw std::invalid_argument("AudioPreprocess: n_fft must be >= frame_size");
    }

    // 第 m 帧起点 = m·step + first_，首帧等待时间与镜像污染量由 first_ 决定。
    // 三种常见方案（只影响开头第一帧，对整体影响很小，三种都可实施）：
    //   方案 A：首帧 [0, 400]，       等 25 ms 输出，无镜像补齐，中心 200 + 160m；
    //   方案 B：首帧 [−120, 280]，    等 17.5 ms 输出，左补 120 点，中心 80 + 160m；
    //   方案 C：首帧 [−200, 200]，    等 12.5 ms 输出，左补 200 点，中心 0 + 160m。
    // 选方案 B：严格对齐 kaldi snip_edges=False，同时在首帧延迟（17.5 ms）
    // 与镜像补齐引入的假信号污染（120 点）之间取得平衡。
    first_ = -120;
    // 左补齐量（恒为 120，只有第 0 帧用得到）；pad_mode 的退化判据要用到。
    left_pad_ = -first_;

    // 窗与 Mel 滤波器组都只依赖构造参数，算一次、之后反复复用。
    window_ = build_window();
    build_mel_filterbank();
    reset();
}

// 重置：清空样本缓存与计数，回到「一个样本都没喂过」。
// 窗与 Mel 滤波器组只依赖构造参数，不在此重建。
void AudioPreprocess::reset() {
    samples_.clear();   // 缓存为空：绝对区间 [base_, base_+0)
    base_ = 0;          // samples_[0] 的绝对序号，重置回 0
    n_samples_ = 0;     // 累计已喂入样本总数（与缓存长度不同，缓存会 trim）
    next_frame_ = 0;    // 下一帧要输出的帧号
    finished_ = false;  // input_finished() 是否已调用
    total_frames_ = 0;  // 收尾时才按批量公式确定的帧数上界
}

// ==================== 对外接口 ====================

// 喂入一块波形：追加进样本缓存，然后 while 排空所有已经凑满的帧。
// 返回本次新产出的帧；块长任意，凑不够一帧时返回 n_frames == 0。
// 调用后缓存尾部最多留下不足一帧的样本（< 400 点），内存有界。
// 参数 samples 是任意长度的一维波形，量纲须与 Python 一致：float [-1,1]。
LogFbankFeatures AudioPreprocess::accept_waveform(const std::vector<float>& samples) {
    // 空块合法：不追加数据，但仍要走一遍 while（可能上一块已攒够帧），
    // 也保证调用方拿到一个 n_mels 正确的空特征结构。
    if (!samples.empty()) {
        // 每秒只拼 100 次左右、每次不到 1 KB，这点拷贝可以忽略。
        samples_.insert(samples_.end(), samples.begin(), samples.end());
        // 用 n_samples_ 记累计喂入量而不是 samples_.size()：缓存会被 trim 裁剪，
        // 绝对帧号必须基于「整条音频」的样本总数。
        n_samples_ += static_cast<int64_t>(samples.size());
    }

    LogFbankFeatures out;
    out.n_mels = n_mels_;

    // while 而不是 if：一次喂入多块音频时要把积压的帧全部排空。
    // 每轮：判据 → 取一帧原始样本 → 走完 7 步 → 追加到输出 → 帧号+1 → 回收旧样本。
    while (can_emit()) {
        const std::vector<double> frame = get_frame(next_frame_);
        const std::vector<double> mel = proc_frame(frame);
        // double → float32 只在这里转换一次，与 Python 的 .astype(np.float32) 对齐。
        out.data.reserve(out.data.size() + mel.size());
        for (double value : mel) {
            out.data.push_back(static_cast<float>(value));
        }
        ++next_frame_;  // next_frame_ 号帧已经定下来
        trim();         // 丢弃后续帧再也读不到的旧样本
    }

    out.n_frames = static_cast<int>(out.data.size() / n_mels_);
    return out;
}

// 收尾：按批量公式 round(N/step) 补出剩余帧，再无条件补 tail_pad_frames 帧 0。
// 补的 0 是特征域的 0（log 域对应能量 1），不是静音值——这是给解码器凑 chunk 的，
// 可视化路径要把 tail_pad_frames 传 0，否则图上会多一块假亮尾（比静音还亮）。
// 重复调用（finished_ 已为 true）直接返回空特征。
LogFbankFeatures AudioPreprocess::input_finished() {
    LogFbankFeatures out;
    out.n_mels = n_mels_;

    // 收尾只能做一次；第二次调用不再产出，避免重复补帧。
    if (finished_) {
        return out;
    }
    finished_ = true;

    // 样本总数此时已定，可用批量版帧数公式：round(N / frame_step)。
    // 整数写法等价于 Python 的 (N + step//2) // step，不引入浮点舍入。
    total_frames_ = (n_samples_ + frame_step_ / 2) / frame_step_;

    // 流式阶段只出「不越界」的帧；这里把其余尾帧全部补出来（可右补齐）。
    while (next_frame_ < total_frames_) {
        const std::vector<double> frame = get_frame(next_frame_);
        const std::vector<double> mel = proc_frame(frame);
        for (double value : mel) {
            out.data.push_back(static_cast<float>(value));
        }
        ++next_frame_;
    }

    // 无条件补 0，与官方 test_onnx.py 一致：
    // 整段不足一帧时 out 本是空的，这里会补成 (tail_pad_frames, n_mels) 全 0——
    // 这是刻意的，不要加「已有帧才补」的守卫，否则与批量版语义不一致。
    // 动机：解码循环末尾凑不满一个 chunk 的那段会被整段丢掉，尾字就识别不出来。
    if (tail_pad_frames_ > 0) {
        out.data.insert(out.data.end(),
                        static_cast<size_t>(tail_pad_frames_) * n_mels_, 0.0f);
    }

    out.n_frames = static_cast<int>(out.data.size() / n_mels_);
    return out;
}

// ==================== 出帧 ====================

// 现在能不能把 next_frame_ 号帧定下来。
// 流未结束时只放 hi <= n_samples_ 的帧（不需要右补齐）；收尾后帧数上界由
// 批量公式给出，允许右补齐。判据用绝对样本数 n_samples_，与块长无关。
bool AudioPreprocess::can_emit() const {
    // 收尾后不再看样本数，只看批量帧数上界（尾帧允许越界右补齐）。
    if (finished_) {
        return next_frame_ < total_frames_;
    }
    // 第 next_frame_ 帧的右端点（绝对样本序号）：
    //   lo + frame_size = next_frame_·step + first_ + frame_size。
    // 必须等样本攒到这里，这一帧的 400 点才「全部真实」，这也就是
    // 单帧因果延迟 12.5 ms 的来源（中心在 lo + 200，右端再多等 200 点）。
    const int64_t next_frame_end =
        next_frame_ * frame_step_ + first_ + frame_size_;
    return n_samples_ >= next_frame_end;
}

// 取第 m 帧的 frame_size_ 个原始样本，越界处按 pad_mode 补齐。
// 只补这一帧自己越界的部分；镜像长度固定、且补出的点只取决于贴边样本，
// 所以逐帧补与「整段补一次」得到的结果完全相同。
// 正常帧直接从 samples_ 切片；只有 m=0（左越界）或流末尾（右越界）走补齐分支。
std::vector<double> AudioPreprocess::get_frame(int64_t m) const {
    // 帧在整条音频里的绝对区间 [lo, hi)；lo 可为负，hi 可超过已到达样本数。
    const int64_t lo = m * frame_step_ + first_;
    const int64_t hi = lo + frame_size_;
    // 与真实样本相交的那一段：abs_start 取 max(lo,0)，abs_end 取 min(hi,n)。
    const int64_t abs_start = std::max<int64_t>(lo, 0);
    const int64_t abs_end = std::min<int64_t>(hi, n_samples_);

    // 真实已有样本这一段；缓存只保存 [base_, base_+size)，所以两位都要先减 base_。
    // abs_end <= abs_start 表示这一帧完全落在还没有样本的区域（极短音频），seg 为空。
    std::vector<double> seg;
    if (abs_end > abs_start) {
        const int64_t from = abs_start - base_;
        const int64_t to = abs_end - base_;
        seg.assign(samples_.begin() + from, samples_.begin() + to);
    }

    const size_t nl = static_cast<size_t>(abs_start - lo);  // 左补点数（只有 m=0 非 0）
    const size_t nr = static_cast<size_t>(hi - abs_end);    // 右补点数（只有末尾非 0）
    // 绝大多数帧走这里：不越界，直接就是一段切片。
    if (nl == 0 && nr == 0) {
        return seg;
    }

    // 补齐方式由「整段样本数 n」决定一次，与具体帧号无关。
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
//   判据 n <= 0 / nf <= 0：没有任何完整帧，直接退化；
//   right 是最后一个批量帧的右越界点数（最后一帧 hi 超出 n 的量），
//   只有流末尾的帧用得到；
//   symmetric 要求「左右补齐宽度都小于原数组长度 n」——这是 numpy 镜像
//   拷贝的限制，kaldi 同样遵守。left_pad_ 恒为 120，right < n 是主要条件，
//   代入公式可算出退化只在整段样本数 n <= 140（< 9 ms）时触发。
std::string AudioPreprocess::pad_mode(int64_t n) const {
    // 批量帧数（round(n/step)），用于定位最后一帧的右端点。
    const int64_t nf = (n + frame_step_ / 2) / frame_step_;
    if (n <= 0 || nf <= 0) {
        return "edge";
    }
    // 最后一帧的右越界点数；max(0, ...) 保证正常长度时为 0。
    const int64_t right =
        std::max<int64_t>(0, (nf - 1) * frame_step_ + first_ + frame_size_ - n);
    if (left_pad_ < n && right < n) {
        return "symmetric";
    }
    return "edge";
}

// 补在帧首的 k 个点，来源是信号开头。
//   edge：把首样本复制 k 次；
//   symmetric：sig[-j] = sig[j-1]，所以取前 k 个样本倒序。
// 只有第 0 帧会调用它，此时 trim 还没机会跑过，base_ 必为 0；
// 这里仍按 base_ 做相对索引，保持与 Python 一致的写法。
std::vector<double> AudioPreprocess::left_pad(size_t k, const std::string& mode) const {
    if (mode == "edge") {
        return std::vector<double>(k, samples_.front());
    }
    // 绝对索引 [0, k) → 缓存相对索引 [0-base_, k-base_)。
    const int64_t from = 0 - base_;
    const int64_t to = static_cast<int64_t>(k) - base_;
    std::vector<double> pad(samples_.begin() + from, samples_.begin() + to);
    std::reverse(pad.begin(), pad.end());
    return pad;
}

// 补在帧尾的 k 个点，来源是信号结尾。
//   edge：把末样本复制 k 次；
//   symmetric：sig[n+j] = sig[n-1-j]，所以取末尾 k 个样本倒序。
// 只有流末尾的帧会调用它；此时缓存尾部就是 n_samples_（trim 不会越过它）。
std::vector<double> AudioPreprocess::right_pad(size_t k, const std::string& mode) const {
    if (mode == "edge") {
        return std::vector<double>(k, samples_.back());
    }
    // 绝对索引 [n-k, n) → 缓存相对索引 [n-k-base_, n-base_)。
    const int64_t to = n_samples_ - base_;
    const int64_t from = n_samples_ - static_cast<int64_t>(k) - base_;
    std::vector<double> pad(samples_.begin() + from, samples_.begin() + to);
    std::reverse(pad.begin(), pad.end());
    return pad;
}

// 丢掉后续任何一帧都读不到的旧样本。
// 下一帧起点 = next_frame_·step + first_，它之前的样本永远不会再被读到，
// 因为 get_frame 只会取 [next_frame_·step + first_, ...) 及其右侧。
// 用 erase 而不是保存视图（Python 用 .copy() 同理），保证内存真正回收；
// 正常运行时缓存长度 < 一帧（400 点），不会无限增长。
void AudioPreprocess::trim() {
    // 下一帧的起点（绝对序号）；max(0,...) 处理第 0 帧起点为负的情况。
    const int64_t keep_from =
        std::max<int64_t>(0, next_frame_ * frame_step_ + first_);
    // 相对当前缓存头要丢弃的点数（keep_from − base_）。
    const int64_t drop = keep_from - base_;
    if (drop <= 0) {
        return;
    }
    // 防御性截断：drop 理论上不会超过缓存长度，min 保证迭代器不越界。
    const int64_t count =
        std::min<int64_t>(drop, static_cast<int64_t>(samples_.size()));
    samples_.erase(samples_.begin(), samples_.begin() + count);
    // 缓存头移动后，base_ 同步到新的绝对起点。
    base_ = keep_from;
}

// ==================== 帧内 7 步 ====================

// 一帧原始样本 → n_mels 维 LogFBank 特征（double）。
// 步骤与顺序严格照抄 Python _proc_frame：
//   1 去直流 → 2 倒序预加重 → 3 povey 窗 → 4 512 点 FFT → 5 功率谱
//   → 6 Mel 三角滤波 → 7 log(max(能量, float32_eps))
// 每一步都是一次显式运算，没有大矩阵批量，便于和 Python 逐行对照。
// 前置条件：frame.size() == frame_size_。
std::vector<double> AudioPreprocess::proc_frame(const std::vector<double>& frame) const {
    // 拷一份再改：调用方的帧数据不能被就地改写（与 Python 的 .copy() 一致）。
    std::vector<double> x = frame;

    // 1) 去直流：减掉本帧均值。
    // 麦克风有直流偏置，不去掉会在 0 Hz 处堆一个虚假的大峰值，
    // 经 Mel 滤波器泄漏到低频通道，凭空抬高最低几个通道的能量。
    const double mean =
        std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
    for (double& value : x) {
        value -= mean;
    }

    // 2) 预加重：y[i] = x[i] − α·x[i−1]。
    // 语音频谱整体按 -6 dB/倍频程衰减，预加重把高频抬回来，让各频段量级均衡。
    // 必须倒序：正序会让 x[i-1] 已被改写成新值，结果错误。
    // 帧首没有前一样本，kaldi 约定用自身顶替：x[0] − α·x[0] = x[0]·(1 − α)。
    for (size_t i = x.size() - 1; i > 0; --i) {
        x[i] -= preemph_ * x[i - 1];
    }
    x[0] *= (1.0 - preemph_);

    // 3) 加窗：FFT 隐含假设输入首尾相接。不加窗的话，帧两端的不连续会造成
    // 频谱泄漏（主频两侧拖出虚假谱线）。povey 窗两端严格归零，泄漏最小。
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] *= window_[i];
    }

    // 4) FFT：先把帧拷进长度 n_fft_ 的全零缓冲（不足部分补 0），再做复数 DFT，
    // 然后只取正频率半边。实数输入的负频率是共轭镜像，不必计算；
    // 采样率 16 kHz 时最高频率为 Nyquist = 8 kHz，对应第 256 个 bin。
    cv::Mat input(n_fft_, 1, CV_64F, cv::Scalar(0.0));
    for (size_t i = 0; i < x.size(); ++i) {
        input.at<double>(static_cast<int>(i), 0) = x[i];
    }
    cv::Mat spectrum;
    cv::dft(input, spectrum, cv::DFT_COMPLEX_OUTPUT);

    // 5) 功率谱：周期图定义 |X|² = re² + im²。
    // 平方后各频率分量的强弱被拉开，更贴近「能量」；不做归一化，
    // 与 numpy rfft + |X|² 的口径一致（kaldi 同样不除 N）。
    std::vector<double> power(n_freq_, 0.0);
    for (int k = 0; k < n_freq_; ++k) {
        const cv::Vec2d value = spectrum.at<cv::Vec2d>(k, 0);
        power[k] = value[0] * value[0] + value[1] * value[1];
    }

    // 6)+7) Mel 滤波并取对数：
    //   降维发生在这一步：257 维功率谱 × 80×257 三角权重 → 80 维。
    //   mel[m] = Σ_k weight[m, k] · power[k]，即把三角盖住的 bin 加权求和。
    //   8 kHz 内 257 维均匀频谱对语音冗余度很高，而人耳低频分辨力强、高频弱；
    //   Mel 三角低频密、高频疏，正好模拟这一点。
    //   权重表绝大多数是 0，所以预先记下每行非零区间 [lo, hi)，只扫这一段。
    //   取对数用 max 抬到地板而不是加 eps：静音帧才会停在 log(1.19e-7) = −15.94，
    //   而不是 log(能量 + 1e-10) 得到的 −23 一带。
    std::vector<double> mel(n_mels_, 0.0);
    for (int m = 0; m < n_mels_; ++m) {
        // 第 m 个三角的非零 bin 半开区间 [lo, hi)；空三角时 lo==hi==0。
        const int lo = mel_bin_lo_[m];
        const int hi = mel_bin_hi_[m];
        double acc = 0.0;
        // 权重 × 该 bin 的功率，逐 bin 累加成一个通道的总能量。
        for (int k = lo; k < hi; ++k) {
            acc += mel_weights_[static_cast<size_t>(m) * n_freq_ + k] * power[k];
        }
        // 7) 取对数并抬到地板；max 而不是加 eps，静音帧停在 −15.94。
        mel[m] = acc > kFloat32Eps ? std::log(acc) : std::log(kFloat32Eps);
    }
    return mel;
}

// ==================== 构造期：窗与 Mel 滤波器组 ====================

// povey 窗：(0.5 − 0.5·cos(2πn/(N−1)))^0.85，即汉宁窗整体取 0.85 次方。
// 手算而不用现成窗函数表，原因只有一个：分母。
// kaldi 的窗都用 N−1 作分母（对称窗，首尾都取到零点）；scipy 默认
// periodic=True 用 N 作分母，N=400 时两者最大差 5.5e-3，
// 足以把 log-mel 特征拉开可见差距。
// 0.85 次方让窗两端更陡、中心更平，是 kaldi 的固定口径，不是可调参数。
std::vector<double> AudioPreprocess::build_window() const {
    std::vector<double> window(static_cast<size_t>(frame_size_), 0.0);
    // 角频率步长 2π/(N−1)：n=0 与 n=N−1 处 cos 项恰好为 1，窗值归零。
    const double a = 2.0 * kPi / static_cast<double>(frame_size_ - 1);
    for (int n = 0; n < frame_size_; ++n) {
        const double hann = 0.5 - 0.5 * std::cos(a * n);
        window[n] = std::pow(hann, 0.85);
    }
    return window;
}

// 构造 Mel 三角滤波器组（kaldi 口径），形状 (n_mels, n_freq)。
// 与「先算浮点 bin 位置、再按 bin 序号插值」的常见写法不同，kaldi 是：
//   1. low/high 换到 Mel 刻度，在 Mel 域等分 n_mels+1 段，得到每个三角的
//      left / center / right 三个边界（都是 Mel 值）；
//   2. 遍历每一个**整数** FFT bin，把它的中心频率 k·(fs/n_fft) 换到 Mel；
//   3. 该 Mel 值严格落在 (left, right) 内才参与，按它到 center 的距离
//      在 **Mel 域**做线性插值（不是按 bin 序号插值）。
// 为什么必须这样：Hz→Mel 是非线性的，拿 Mel 值而不是 bin 序号插值，
// 才能得到 kaldi 的三角形形状；边界用严格不等号，落在边界上的 bin
// 直接不参与（权重 0），而不是插值出 0.5。
// 同时记录每个滤波器非零的 bin 区间 [lo, hi)，运行期只扫这一段。
void AudioPreprocess::build_mel_filterbank() {
    const double nyquist = 0.5 * sample_rate_;
    // 滤波器组上限：high_freq > 0 直接用；0 表示取到 Nyquist；
    // 负数表示从 Nyquist 往回收，如 -400 → 7600 Hz（kaldi C++ 的默认）。
    // 本项目默认 0（8000 Hz），与 sharpa 官方 test_onnx.py 一致。
    const double high =
        high_freq_ > 0.0 ? high_freq_ : nyquist + high_freq_;
    // 每个 FFT bin 的带宽 = fs/n_fft = 31.25 Hz；bin k 的中心频率 = k·bin_width。
    const double bin_width = sample_rate_ / static_cast<double>(n_fft_);

    // 三角滤波器的摆放全在 Mel 刻度上完成：低频密、高频疏。
    // 相邻等距点的 Mel 间距；一个三角横跨两段（left→center + center→right）。
    const double mel_low = hz_to_mel(low_freq_);
    const double mel_high = hz_to_mel(high);
    const double mel_delta = (mel_high - mel_low) / (n_mels_ + 1);

    // 257 个 bin 的中心频率统一换成 Mel，避免 80 个滤波器重复计算。
    // bin_mels 单调递增：低频段 Mel 值间距小，高频段间距大。
    std::vector<double> bin_mels(static_cast<size_t>(n_freq_), 0.0);
    for (int k = 0; k < n_freq_; ++k) {
        bin_mels[k] = hz_to_mel(bin_width * k);
    }

    // 权重表默认全 0；非零区间先记成空 [0, 0)。
    mel_weights_.assign(static_cast<size_t>(n_mels_) * n_freq_, 0.0);
    mel_bin_lo_.assign(n_mels_, 0);
    mel_bin_hi_.assign(n_mels_, 0);

    // 每个三角是一个带通滤波器：把落入其区间的各 bin 能量按三角权重
    // 加权求和，压缩成 1 个代表该频带总能量的标量。
    for (int m = 0; m < n_mels_; ++m) {
        // 第 m 个三角的三个端点（都是 Mel 值）：左脚 / 顶点 / 右脚。
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
        // 记成半开区间 [lo, hi)，供运行期只扫非零部分。
        mel_bin_lo_[m] = lo;
        mel_bin_hi_[m] = hi;
    }
}

}  // namespace audio_ai

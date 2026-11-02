/**
 * @file test_spectrogram_cv_wav.cpp
 * @brief wav 滚动语谱图 —— 文件版入口（对应 Python test_spectrogram_cv_mini.py）。
 *
 * 数据流：
 *   wav 文件
 *     → load_wav()（RIFF 解析，取第一声道，PCM16 / float32 → float [-1,1]）
 *     → AudioPreprocess（波形 → (n, 80) LogFBank）
 *     → SpectrogramImage（特征 → BGR 滚动画布）
 *     → cv::imshow
 *
 * 与 test_spectrogram_cv_mini.cpp（麦克风版）的区别：
 *   - 数据源从设备实时取流换成整段读入内存的 wav；
 *   - 主循环按 1x 节奏 sleep，模拟「边放边出图」；
 *     加 --fast 则不等待，尽快喂完并自动存图退出（批量检查用）；
 *   - 前端 / 可视化参数与 Python 版一致（80 Mel、400 列、前 50 帧标定色阶）。
 *
 * 按键：
 *   q / Esc   退出（退出时按已喂入的样本补完尾帧）
 *   s         把当前画面存成 results/spectrogram_wav.png
 *
 * 用法（项目根目录）：
 *   ./build/test_spectrogram_cv_wav.out                          # 默认 wav，1x 播放
 *   ./build/test_spectrogram_cv_wav.out results/xxx.wav          # 指定 wav，1x 播放
 *   ./build/test_spectrogram_cv_wav.out results/xxx.wav --fast   # 尽快喂完，存图退出
 */

#include "audio_ai_process/audio_preprocess.h"
#include "audio_ai_process/spectrogram_image.h"
#include "audio_ai_process/text_utils.h"

#include "tools/wav_reader.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace audio_ai;

// WAV 读取已抽到 tools/wav_reader.h（与 wav ASR 入口共用）；
// 路径工具（base_name）在 text_utils.h，入口只做参数与装配。

int main(int argc, char** argv) {
    // —— 一、运行参数（全部写死在 main，现场要改时一眼可见）——
    const std::string kDefaultWavPath =
        "results/hk_audio_20261102_085654.wav";  // 不传路径时读这条录音
    const int kNMels = 80;                       // Mel 通道数 = 图像行数
    const int kBlock = 160;                      // 每次喂 10ms（160 点 @16kHz）
    const int kCols = 400;                       // 滚动窗口宽度：400 帧 = 4 秒
    const int kCalibrateFrames = 50;             // 前 0.5 秒特征标定色阶
    const cv::Size kPlotSize(1200, 300);         // 语谱图显示尺寸（宽, 高）
    const int kHeaderH = 52;                     // 顶部信息条高度
    const int kFooterH = 0;                      // 本入口不写识别文字，不预留底部
    const std::string kSavePath = "results/spectrogram_wav.png";  // 结果图统一放 results/
    const std::string kWindow = "mini spectrogram (wav)";

    // 命令行：--fast 关闭 1x 节奏；其余参数当作 wav 路径（只取最后一个）。
    std::string wav_path = kDefaultWavPath;
    bool fast = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--fast") {
            fast = true;
        } else {
            wav_path = arg;
        }
    }

    // —— 二、读 wav（整段进内存；采样率不对时只提示，不偷偷重采样）——
    WavData wav;
    std::string error;
    if (!load_wav(wav_path, wav, error)) {
        std::cerr << "[Spectrogram] " << error << std::endl;
        return -1;
    }
    const int sample_rate = wav.sample_rate;
    if (sample_rate <= 0) {
        std::cerr << "[Spectrogram] wav 采样率非法" << std::endl;
        return -1;
    }
    const double duration = static_cast<double>(wav.samples.size()) / sample_rate;

    // —— 三、装配前端与可视化（各管一段，只靠特征相连）——
    // tail_pad_frames=0：可视化不补尾部 100 帧 0，否则图上会多一块假亮尾。
    AudioPreprocess front(sample_rate, kNMels,
                          25.0,   // 帧长 ms
                          10.0,   // 帧移 ms
                          512,    // FFT 点数
                          0.97,   // 预加重
                          20.0,   // Mel 下限 Hz
                          0.0,    // Mel 上限：0 = Nyquist
                          0);     // tail_pad_frames：可视化不补
    const double frame_shift =
        static_cast<double>(front.frame_step()) / sample_rate;

    SpectrogramImage view(kNMels, kCols, frame_shift, kDefaultLevelLo,
                          kDefaultLevelHi, kCalibrateFrames, kPlotSize,
                          kHeaderH, kFooterH, cv::COLORMAP_VIRIDIS,
                          base_name(wav_path));

    // —— 四、打印参数 ——
    std::cout << std::string(74, '=') << "\n"
              << "mini 滚动语谱图 —— wav -> LogFBank -> OpenCV 窗口\n"
              << std::string(74, '=') << "\n"
              << "wav   : " << wav_path << "\n"
              << "        " << duration << " 秒 / " << wav.samples.size()
              << " 样本 @ " << sample_rate << " Hz  " << wav.channels
              << " 声道  " << wav.bits_per_sample << " bit\n"
              << "前端  : 帧长 " << front.frame_size() << " 点("
              << front.frame_size() * 1000.0 / sample_rate << "ms)"
              << "  帧移 " << front.frame_step() << " 点("
              << frame_shift * 1000.0 << "ms)"
              << "  n_mels=" << kNMels << "\n"
              << "画布  : " << kCols << " 列 x " << kNMels << " 行 = "
              << kCols * frame_shift << " 秒窗口\n"
              << "节奏  : " << (fast ? "尽快喂完（--fast）" : "1x 实时模拟")
              << "\n"
              << "按键  : q / Esc 退出   s 保存当前画面\n"
              << std::string(74, '=') << std::endl;
    if (sample_rate != 16000) {
        std::cout << "注意：wav 是 " << sample_rate
                  << " Hz。本图按实际采样率画，仅用于观察；"
                     "喂 ASR 模型前要先重采样到 16000 Hz。\n"
                  << std::endl;
    }

    // —— 五、主循环：按 1x 节奏喂块 → 前端 → 滚动出图 → 显示 ——
    using Clock = std::chrono::steady_clock;
    const Clock::time_point t0 = Clock::now();
    size_t fed = 0;  // 已喂入的样本数，同时用于推算「这一块本该到达的时刻」
    bool quit = false;
    cv::Mat canvas;
    std::vector<float> chunk;
    chunk.reserve(kBlock);

    while (fed < wav.samples.size() && !quit) {
        // 5.1 1x 节奏：等到这一块音频本该到达的时刻再喂。
        //     每块都按绝对时间重算，偶尔睡过头也不会越拖越慢。
        if (!fast) {
            const auto due =
                t0 + std::chrono::duration_cast<Clock::duration>(
                         std::chrono::duration<double>(
                             static_cast<double>(fed) / sample_rate));
            std::this_thread::sleep_until(due);
        }

        // 5.2 喂一块（末尾可能不满 kBlock）：前端出特征，可视化返回整幅画布。
        const size_t n = std::min<size_t>(kBlock, wav.samples.size() - fed);
        chunk.assign(wav.samples.begin() + fed, wav.samples.begin() + fed + n);
        fed += n;
        canvas = view.accept_features(front.accept_waveform(chunk));

        // 5.3 显示 + 处理按键。
        cv::imshow(kWindow, canvas);
        const int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') {  // Esc / q
            quit = true;
        } else if (key == 's' && !canvas.empty()) {  // s：存图
            if (cv::imwrite(kSavePath, canvas)) {
                std::cout << "已保存 " << kSavePath << std::endl;
            } else {
                std::cerr << "保存失败：" << kSavePath << std::endl;
            }
        }
    }

    // —— 六、收尾：补完尾帧 → 打印统计 → 停住最后一屏 ——
    // tail_pad_frames=0，这里只补最后不足一帧的样本，不会多出假数据。
    canvas = view.accept_features(front.input_finished());
    cv::imshow(kWindow, canvas);

    std::cout << "\n结束：喂入 " << fed << " 样本，产出 " << view.n_frames()
              << " 帧（" << view.n_frames() * frame_shift << " 秒）" << std::endl;

    // --fast 是批量检查路径，没有人工按键：退出前把最后一屏落盘。
    if (fast && !canvas.empty()) {
        if (cv::imwrite(kSavePath, canvas)) {
            std::cout << "画面已保存：" << kSavePath << std::endl;
        } else {
            std::cerr << "保存失败：" << kSavePath << std::endl;
        }
    }

    if (!quit && !fast) {
        std::cout << "按任意键关闭窗口……" << std::endl;
        cv::waitKey(0);  // 停住最后一屏，方便慢慢看
    }
    cv::destroyAllWindows();
    return 0;
}

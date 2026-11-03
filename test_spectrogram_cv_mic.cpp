/**
 * @file test_spectrogram_cv_mic.cpp
 * @brief 实时滚动语谱图 —— 海康麦克风版（C++ 移植，对应 test_spectrogram_cv_mini.py）。
 *
 * 数据流：
 *   海康设备拾音器
 *     → HkSdkAudioClient（语音转发 + G.722.1 解码）
 *     → get_last_audio()（取走自上次调用以来累积的 PCM16）
 *     → int16 / 32768.0（转成 [-1,1]，与 Python/soundfile 的量纲一致）
 *     → AudioPreprocess（波形 → (n, 80) LogFBank）
 *     → SpectrogramImage（特征 → BGR 滚动画布）
 *     → cv::imshow
 *
 * 与 Python 入口的区别：
 *   - 音频源从「读 wav 后按 1x 节奏喂」换成「设备实时取流」；
 *   - 不再需要 soundfile / 实时 sleep：数据本来按真实时间到达；
 *   - 其余参数与 Python 版保持一致（80 Mel、400 列、前 50 帧标定色阶）。
 *
 * 按键：
 *   q / Esc   退出（退出时会停流并补完尾帧）
 *   s         把当前画面存成 results/spectrogram_mic.png
 *
 * 用法（项目根目录）：
 *   ./build/test_spectrogram_cv_mic.out          # 交互模式，q/Esc 退出
 *   ./build/test_spectrogram_cv_mic.out 10       # 自检模式，跑 10 秒后存图退出
 */

#include "audio_ai_process/audio_preprocess.h"
#include "audio_ai_process/spectrogram_image.h"
#include "hk_sdk_audio_client.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace audio_ai;
using namespace hk_isapi;

int main(int argc, char** argv) {
    // 自检模式：传入秒数 > 0 时，跑到时间就存图打印统计并退出；
    // 不传（或传 0）保持交互模式，由 q/Esc 结束。
    const double kRunSeconds = argc > 1 ? std::atof(argv[1]) : 0.0;

    // —— 一、运行参数（全部写死在 main，现场要改时一眼可见）——
    const std::string kDeviceIp = "10.168.2.65";  // 音频设备 IP
    const int kSdkPort = 8000;                    // HCNetSDK 私有协议端口
    const std::string kUsername = "admin";
    const std::string kPassword = "Password";
    const int kVoiceChannel = 1;                  // 语音通道号，本机实测为 1

    const int kNMels = 80;                        // Mel 通道数 = 图像行数
    const int kCols = 400;                        // 滚动窗口宽度：400 帧 = 4 秒
    const int kCalibrateFrames = 50;              // 前 0.5 秒特征标定色阶
    const cv::Size kPlotSize(1200, 300);          // 语谱图显示尺寸（宽, 高）
    const int kHeaderH = 52;                      // 顶部信息条高度
    const int kFooterH = 0;                       // 本入口不写识别文字，不预留底部
    const std::string kLabel = "hk mic";          // 信息条第一行（ASCII）
    const std::string kSavePath = "results/spectrogram_mic.png";  // 结果图统一放 results/
    const std::string kWindow = "mini spectrogram (hk mic)";

    // int16 → float 的归一化系数。Python 版麦克风是 float32 [-1,1]，
    // 直接拿 int16 当浮点喂前端会让 log 能量整体偏移 ln(32768²) ≈ 20.8。
    constexpr double kPcmScale = 1.0 / 32768.0;
    constexpr int kKeyWaitMs = 10;                // 无新数据时等待按键的节奏

    // —— 二、启动设备取流 ——
    AudioCaptureConfig capture_config;
    capture_config.ip = kDeviceIp;
    capture_config.sdk_port = kSdkPort;
    capture_config.username = kUsername;
    capture_config.password = kPassword;
    capture_config.voice_channel = kVoiceChannel;
    capture_config.verbose = true;

    // init() 只登录 + 查询编码；start() 才开流。本入口没有模型加载，
    // 两段紧挨着调用，但生命周期与实时 ASR 入口保持一致（exit 走 deinit）。
    HkSdkAudioClient client(capture_config);
    if (!client.init()) {
        std::cerr << "[Spectrogram] init failed: " << client.last_error()
                  << std::endl;
        return -1;
    }
    if (!client.start()) {
        std::cerr << "[Spectrogram] start failed: " << client.last_error()
                  << std::endl;
        client.deinit();
        return -1;
    }

    const int sample_rate = client.sample_rate();

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
                          kHeaderH, kFooterH, cv::COLORMAP_VIRIDIS, kLabel);

    // —— 四、打印参数 ——
    std::cout << std::string(74, '=') << "\n"
              << "mini 实时语谱图 —— 海康麦克风 -> LogFBank -> OpenCV 窗口\n"
              << std::string(74, '=') << "\n"
              << "设备  : " << kDeviceIp << ":" << kSdkPort
              << "  channel=" << kVoiceChannel << "\n"
              << "编码  : " << client.codec_name() << "  " << sample_rate
              << " Hz\n"
              << "前端  : 帧长 " << front.frame_size() << " 点("
              << front.frame_size() * 1000.0 / sample_rate << "ms)"
              << "  帧移 " << front.frame_step() << " 点("
              << frame_shift * 1000.0 << "ms)"
              << "  n_mels=" << kNMels << "\n"
              << "画布  : " << kCols << " 列 x " << kNMels << " 行 = "
              << kCols * frame_shift << " 秒窗口\n"
              << "按键  : q / Esc 退出   s 保存当前画面\n"
              << std::string(74, '=') << std::endl;
    if (sample_rate != 16000) {
        std::cout << "注意：设备采样率是 " << sample_rate
                  << " Hz。本图按实际采样率画；喂 ASR 模型前要先重采样到 "
                     "16000 Hz。\n"
                  << std::endl;
    }

    // —— 五、主循环：取流 → 前端 → 滚动出图 → 显示 ——
    bool quit = false;
    bool auto_exit = false;  // 自检模式到点退出，用于区分「按键退出」
    cv::Mat canvas;
    std::vector<int16_t> pcm;
    std::vector<float> chunk;
    const auto t_start = std::chrono::steady_clock::now();

    // 按键处理：q / Esc 退出，s 存图。返回 true 表示继续循环。
    auto handle_key = [&](int key) {
        if (key == 27 || key == 'q') {
            return false;
        }
        if (key == 's' && !canvas.empty()) {
            if (cv::imwrite(kSavePath, canvas)) {
                std::cout << "已保存 " << kSavePath << std::endl;
            } else {
                std::cerr << "保存失败：" << kSavePath << std::endl;
            }
        }
        return true;
    };

    while (!quit) {
        // 自检模式：只看墙钟时间，不依赖有没有数据，避免设备没出声就永远跑下去。
        if (kRunSeconds > 0.0) {
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              t_start)
                    .count();
            if (elapsed >= kRunSeconds) {
                quit = true;
                auto_exit = true;
                break;
            }
        }

        // 5.1 取音频：与是否录制无关，一次取走上次调用以来的全部 PCM16。
        if (!client.get_last_audio(pcm)) {
            quit = !handle_key(cv::waitKey(kKeyWaitMs) & 0xFF);
            continue;
        }

        // 5.2 int16 → float [-1,1]，与 Python 版 mic/wav 的量纲对齐。
        chunk.resize(pcm.size());
        for (size_t i = 0; i < pcm.size(); ++i) {
            chunk[i] = static_cast<float>(pcm[i] * kPcmScale);
        }

        // 5.3 前端出特征（块长任意，内部按绝对样本索引凑帧），可视化出画布。
        const LogFbankFeatures feats = front.accept_waveform(chunk);
        canvas = view.accept_features(feats);

        // 5.4 显示 + 处理按键。
        cv::imshow(kWindow, canvas);
        quit = !handle_key(cv::waitKey(1) & 0xFF);
    }

    // —— 六、收尾：停流 → 取残余样本 → 释放会话 → 补尾帧 → 停住最后一屏 ——
    client.stop();

    // stop() 之后缓冲区里可能还有没被取走的样本，先取完再收尾。
    if (client.get_last_audio(pcm) && !pcm.empty()) {
        chunk.resize(pcm.size());
        for (size_t i = 0; i < pcm.size(); ++i) {
            chunk[i] = static_cast<float>(pcm[i] * kPcmScale);
        }
        canvas = view.accept_features(front.accept_waveform(chunk));
    }

    // 残余样本取完后释放设备会话；deinit() 会清空取流缓冲，所以顺序不能反。
    client.deinit();

    // tail_pad_frames=0，这里只补最后不足一帧的样本，不会多出假数据。
    canvas = view.accept_features(front.input_finished());
    cv::imshow(kWindow, canvas);

    const AudioCaptureStats stats = client.get_stats();
    std::cout << "\n结束：喂入 " << front.n_samples() << " 样本，产出 "
              << view.n_frames() << " 帧（"
              << view.n_frames() * frame_shift << " 秒）\n"
              << "解码样本: " << stats.pcm_samples
              << "  取流丢弃: " << stats.stream_dropped_samples
              << "  解码错误: " << stats.decode_errors << std::endl;

    // 自检模式没有人工按 s，退出前把最后一屏落盘，方便肉眼确认内容。
    if (auto_exit && !canvas.empty()) {
        if (cv::imwrite(kSavePath, canvas)) {
            std::cout << "自检画面已保存：" << kSavePath << std::endl;
        } else {
            std::cerr << "保存失败：" << kSavePath << std::endl;
        }
    }

    if (!quit) {
        std::cout << "按任意键关闭窗口……" << std::endl;
        cv::waitKey(0);
    }
    cv::destroyAllWindows();
    return 0;
}

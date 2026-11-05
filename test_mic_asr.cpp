/**
 * @file test_mic_asr.cpp
 * @brief 实时麦克风语音识别入口（对应 Python test_mic_asr_mini.py）。
 *
 * ============================================================
 * 这份代码是干什么的
 * ============================================================
 * 把三个类按顺序接起来，对着海康设备拾音器说话，边说话边出字：
 *
 *   海康拾音器 → HkSdkAudioClient（语音转发 + G.722.1 解码）
 *              → get_last_audio()（取走自上次调用以来累积的 PCM16）
 *              → int16 / 32768.0（转 [-1,1]）
 *              → AudioPreprocess（波形 → (n, 80) LogFBank）
 *              → 同一份特征分两路：
 *                   BlockWorker（有界队列 + 后台线程）→ AsrSession
 *                   （AsrDecoder 三 ONNX 会话 + 贪心搜索；累积全量 token、
 *                   每 64 token 结算一段并追加 log/asr_last_text.txt）
 *                   → 显示快照
 *                   SpectrogramImage（滚动语谱图，主线程）→ BGR 画布
 *              → 主线程读 Session 快照 → 写 stdout + 画在窗口底部白条 → imshow
 *
 * 三个类只靠特征矩阵相连：入口只做「参数 + 数据源 + 装配」，流水线细节
 * 不写在这里。解码放在独立线程（BlockWorker + AsrSession），主循环只做取流、
 * 前端、渲染和显示，不再被推理卡住；代价是底部文字会比语谱图滞后若干块。
 * 数据本来按真实时间到达，不需要像 wav 入口那样模拟节奏；主循环拿不到新数据
 * 时只等 10 ms，把 CPU 让出来。
 *
 * ============================================================
 * 与 Python 版的差异
 * ============================================================
 *   - 音频源：Python 用 sounddevice（PortAudio，声卡不支持 16 kHz 时由
 *     PortAudio 内部重采样）；这里是海康设备语音转发，G.722.1 固定 16 kHz。
 *     设备实际采样率不是 16 kHz（切到 PCM 编码时会被钉在 32 kHz）时**直接
 *     报错退出**，不静默重采样 —— 与 wav 入口「宁可明确提示，也不悄悄喂错
 *     数据」一致。
 *   - 设备会话拆成两段：建 ONNX 会话之前先 init()（登录 + 查询编码 + 采样率
 *     校验），确认数据源可用；模型装配完成后再 start() 真正开流。这样既能在
 *     加载前发现设备不可用，又不会让 7~14 s 的模型加载把启动阶段的音频
 *     （取流缓冲只有 2.56 s）丢光。
 *   - 解码线程：Python 版在主循环里同步解码（推理、渲染、取流同一线程）；
 *     这里用 BlockWorker 把 encoder / decoder / joiner 放进后台线程，特征块
 *     经有界队列投递，队列满时丢最旧。理由是 Jetson CPU 上主循环吞吐约 0.99，
 *     恰好在实时线上，再加渲染或其它负载就会让取流缓冲（2.56 s）溢出丢块。
 *     拆开后主线程不被推理阻塞，但底部文字会滞后于语谱图，且过载时会丢弃
 *     最旧的音频块。AsrDecoder 明确非线程安全：全程只有 Worker 线程访问它，
 *     main 只在 join 之后读 n_chunks / all_token_ids 等统计。
 *   - 转写落盘：AsrSession 自己累积全量 token，每满 64 个结算一段，把该段
 *     文本追加一行到 log/asr_last_text.txt 并 flush；启动时截断。崩溃最多丢
 *     当前未结算的尾巴，正常退出由 finish() 结算尾段。Python 版没有落盘。
 *   - 窗口底部文字优先 FreeType 渲染中文（Python 的 cv2.putText 只支持
 *     ASCII，中文会静默变成 '?'），字体缺失时回退 cv2.putText。
 *   - 新增 --no-window / --cuda / --seconds 三个开关（Python 版配置全写死）。
 *   - Ctrl+C 用 SIGINT 置标志位、走正常收尾；Python 是捕获 KeyboardInterrupt。
 *
 * ============================================================
 * 调用顺序（对照 main 里的编号）
 * ============================================================
 *   一、参数与命令行         设备地址、模型路径、画布尺寸、字体候选、开关
 *   二、连接设备（Python 1） 2.1 配置  2.2 init  2.3 采样率校验
 *   三、加载模型与装配（Python 2）
 *       3.1 session 选项    3.2 可选 CUDA EP   3.3 三 session + AsrDecoder
 *       3.4 前端            3.5 语谱图         3.6 中文字体   3.7 start 开流
 *       3.8 装配 AsrSession + BlockWorker
 *   四、打印配置（Python 3）
 *   五、主循环（Python 4）
 *       5.0 退出条件  5.1 取音频  5.2 归一化  5.3 前端
 *       5.4 可视化（有新帧才重建画布）
 *       5.5 投递解码  5.6 读最新文本  5.7 画字显示  5.8 按键与节拍
 *   六、收尾与统计（Python 5）
 *       6.1 停流  6.2 残余样本  6.2.1 deinit  6.3 补尾解码 + join
 *       6.4 最后一屏  6.5 统计  6.6 关窗
 *
 * 结束（Ctrl+C、窗口里按 q/Esc、或 --seconds 到点）后必须走完收尾：把缓冲里
 * 残余的 PCM 解完，再让前端补出尾部帧、并在特征尾补 100 帧 0 一起解掉 ——
 * 少了这一步，最后几个字会被整段丢掉。那 100 帧 0 只喂解码器、不喂可视化，
 * 否则语谱图末尾会多出一块比静音还亮的假色块。
 *
 * ============================================================
 * 用法（项目根目录）
 * ============================================================
 *   ./build/test_mic_asr.out
 *       实时识别，窗口底部显示中文结果；q / Esc 退出，空格暂停/恢复语音转发
 *   ./build/test_mic_asr.out --no-window
 *       只打印文本（无显示器 / 远程调试用）
 *   ./build/test_mic_asr.out --seconds 10 --no-window
 *       自检：跑 10 秒后自动收尾退出
 *   ./build/test_mic_asr.out --cuda
 *       CUDA EP，失败自动回退 CPU
 */

#include "audio_ai_process/asr_decode.h"
#include "audio_ai_process/asr_session.h"
#include "audio_ai_process/audio_preprocess.h"
#include "audio_ai_process/block_worker.h"
#include "audio_ai_process/spectrogram_image.h"
#include "audio_ai_process/text_renderer.h"
#include "audio_ai_process/text_utils.h"
#include "hk_sdk_audio_client.h"

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/highgui.hpp>

using namespace audio_ai;
using namespace hk_isapi;

#ifndef HK_PROJECT_DIR
#define HK_PROJECT_DIR "."
#endif

namespace {

// Ctrl+C 只置标志位，不在信号处理函数里做任何收尾：停流、写完最后一屏、
// 打印统计都要在主循环退出后统一做，否则可能停在 SDK 回调半路。
volatile std::sig_atomic_t g_interrupted = 0;

void on_sigint(int) {
    g_interrupted = 1;
}

}  // namespace

int main(int argc, char** argv) {
    // —— 一、参数与命令行 ——
    // 1.1 设备参数：与 test_spectrogram_cv_mic.cpp 保持一致，换设备改这里。
    const std::string kDeviceIp = "10.168.2.65";  // 音频设备 IP
    const int kSdkPort = 8000;                    // HCNetSDK 私有协议端口
    const std::string kUsername = "admin";
    const std::string kPassword = "Password";
    const int kVoiceChannel = 1;                  // 语音通道号，本机实测为 1

    // 1.2 模型与词表：与 test_wav_asr.cpp 用同一套 int8 权重包。
    //     encoder / joiner 是 int8，decoder 保持 fp32（导出时就是这么切的）。
    const std::string project_dir = HK_PROJECT_DIR;
    const std::string asr_dir =
        project_dir +
        "/weight/"
        "sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-"
        "2026-06-05";
    const std::string encoder_path = join_path(asr_dir, "encoder.int8.onnx");
    const std::string decoder_path = join_path(asr_dir, "decoder.onnx");
    const std::string joiner_path = join_path(asr_dir, "joiner.int8.onnx");
    const std::string tokens_path = join_path(asr_dir, "tokens.txt");

    // 1.3 画布尺寸：与 Python 版一致（400 列 x 10 ms = 4 秒滚动窗口），
    //     底部 56 px 白条留给识别文字，顶部 52 px 写设备与模型信息。
    const int kSpecCols = 400;
    const int kFooterH = 56;
    const int kCalibrateFrames = 50;      // 前 0.5 秒标定色阶后锁死
    const cv::Size kPlotSize(1200, 300);  // 语谱图显示尺寸（宽，高）
    const int kHeaderH = 52;
    const std::string kLabel = "hk mic";  // 信息条标签（ASCII）
    // 窗口标题只能用 ASCII：部分 OpenCV 构建对非 ASCII 标题处理不一致。
    const std::string kWindow = "mini mic asr";

    // 1.4 中文字体：按顺序取第一个存在的文件。
    //     Noto Sans CJK 的 .ttc 里 0=JP、1=KR、2=SC、3=TC、4=HK，简中取 2；
    //     Droid / 文泉驿是单字面文件，取 0。FreeType 未编译进目标时整段无效，
    //     窗口文字自动回退 cv::putText（中文会变成 '?'）。
    const std::vector<FontCandidate> kCjkFontCandidates = {
        {"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 2},
        {"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf", 0},
        {"/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0},
        {"/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 0},
    };
    const int kCjkFontHeight = 24;  // 像素字高；底部白条高 56 px
    // 底部文字的左右边距（px）：可用宽度 = 画布宽 − 2 × 边距。
    // FreeType 路径按这个像素宽度从尾部截断，不再按字数估算。
    const int kStatusMargin = 8;
    // ASCII 回退路径的上限：cv::putText 只支持 ASCII，字宽约 12 px，
    // 50 个字符 ≈ 600 px 远小于画布宽度；FreeType 不可用时用它兜底。
    const size_t kMaxStatusChars = 50;

    // 1.5 数据与节奏常量。
    //     int16 → float 的归一化系数。直接拿 int16 当浮点喂前端会让 log
    //     能量整体偏移 ln(32768²) ≈ 20.8，与 Python 的 float [-1,1] 量纲对齐。
    constexpr double kPcmScale = 1.0 / 32768.0;
    // 没有新数据时的等待时间：既让出 CPU，又让 OpenCV 能泵事件（窗口不假死）。
    // 10 ms 与特征帧移一致，最长延迟一帧。
    constexpr int kIdleWaitMs = 10;

    // 1.6 命令行开关。设备地址、模型路径、画布尺寸仍写死在上方，
    //     命令行只控制「开不开窗口 / 用不用 GPU / 跑多久」。
    bool no_window = false;
    bool use_cuda = false;
    double run_seconds = 0.0;  // >0 时到点自动收尾，便于无人值守自检
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--no-window") {
            no_window = true;
        } else if (arg == "--cuda") {
            use_cuda = true;
        } else if (arg == "--seconds") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--seconds 需要一个秒数\n");
                return 2;
            }
            run_seconds = std::atof(argv[++i]);
            if (run_seconds <= 0.0) {
                std::fprintf(stderr, "--seconds 必须大于 0\n");
                return 2;
            }
        } else {
            std::fprintf(stderr, "未知参数：%s\n", arg.c_str());
            std::fprintf(stderr,
                         "用法：%s [--no-window] [--cuda] [--seconds 秒数]\n",
                         argv[0]);
            return 2;
        }
    }

    // Ctrl+C → 标志位 → 主循环正常退出并走收尾（对应 Python 的
    // KeyboardInterrupt 分支）。v2 之后 C++ 标准也保证 signal() 可用。
    std::signal(SIGINT, on_sigint);

    // —— 二、连接设备（对应 Python 第 1 节）——
    // 2.1 填配置：pcm_sample_rate 只在设备走 PCM 编码时作为兜底；
    //     G.722.1 的采样率由编码本身固定为 16 kHz，客户端会自行纠正。
    AudioCaptureConfig capture_config;
    capture_config.ip = kDeviceIp;
    capture_config.sdk_port = kSdkPort;
    capture_config.username = kUsername;
    capture_config.password = kPassword;
    capture_config.voice_channel = kVoiceChannel;
    capture_config.verbose = true;

    // 2.1.1 可选：启动前主动设置对讲编码（写设备持久配置，默认关闭）。
    //       打开后 init() 会先快照原值 → 设置 → 读回复核，不一致直接拒绝
    //       初始化；deinit() 默认把原配置写回。采样率跟随编码自动取：
    //       G.722.1 → 16 kHz，PCM → 32 kHz（本机实测 PCM 只能是 32 kHz）。
    capture_config.set_audio_format = false; // false=只读设备当前编码；true=启动前改成本文件的 audio_enc_type，退出时还原
    capture_config.audio_enc_type = 0; // 0=G.722.1(16k)，8=PCM(32k)
    capture_config.restore_audio_format = true; //< deinit() 时是否写回设置前的设备配置

    // 2.1.2 输出采样率钉在 16 kHz：预处理模块与模型只认 16 kHz。
    //       G.722.1 本来就是 16 kHz（直通，无额外开销）；切到 PCM 时设备
    //       被钉在 32 kHz，客户端内部做 32→16 抗混叠降采样（FIR 低通 +
    //       2 倍抽取），wav、取流缓冲、统计都按 16 kHz 输出。
    //       0 = 跟随设备；本入口不用 0，否则 PCM 下会因采样率不符拒绝启动。
    capture_config.output_sample_rate = 16000;

    // 2.2 建立设备会话：只登录 + 查询编码，不开流，也不写 wav 文件。
    //     开流要等到模型装配完（3.7）再做：Jetson 上 CPU 加载约 7 s、CUDA
    //     约 14 s，若此时设备已在转发，音频会持续压进取流缓冲（上限
    //     2.56 s）并在进入主循环前就被丢弃。先 init() 后 start() 能把这段
    //     空窗完全消掉，代价只是设备连接错误要等模型加载完才报出来。
    HkSdkAudioClient client(capture_config);
    if (!client.init()) {
        std::cerr << "[ASR] 设备初始化失败：" << client.last_error() << "\n";
        return 1;
    }

    // 2.3 采样率校验：模型按 16 kHz 训练，喂别的采样率不会报错、只会识别出
    //     乱七八糟的字。2.1.2 已要求把输出钉在 16 kHz：G.722.1 直通，
    //     PCM 由客户端做抗混叠降采样。这里再校验一次，防止配置被误改；
    //     不满足说明 output_sample_rate 与设备率无法匹配，直接拒绝启动。
    const int sample_rate = client.sample_rate();
    if (sample_rate != 16000) {
        std::cerr << "[ASR] 输出采样率是 " << sample_rate
                  << " Hz，而模型固定 16 kHz。\n"
                  << "      请检查 capture_config.output_sample_rate "
                     "（应为 16000）；\n"
                  << "      设备实际采样率 = " << client.device_sample_rate()
                  << " Hz。\n";
        client.deinit();
        return 1;
    }

    // —— 三、加载模型与装配（对应 Python 第 2 节）——
    // 3.1 session 选项：多个 chunk 之间本来就串行，intra / inter 各 1 线程，
    //     让 onnxruntime 再开线程池只会增加调度开销（与 Python 版一致）；
    //     日志降到 WARNING，避免加载时刷屏。
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test_mic_asr");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 3.2 可选 CUDA EP：注册失败只打印并回退 CPU，不中断识别。
    //     必须用 AppendExecutionProvider_CUDA；通用 AppendExecutionProvider
    //     只支持 QNN / SNPE / XNNPACK。
    //     cuda_active 记录「真的注册成功」，而不是「用户想要 CUDA」——
    //     banner 据此按实际状态打印，避免把回退误报成在用 GPU。
    bool cuda_active = false;
    if (use_cuda) {
        try {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            cuda_active = true;
            std::cout << "CUDA EP 已注册（device_id=0）\n";
        } catch (const std::exception& e) {
            std::cout << "CUDA EP 注册失败，回退 CPU：" << e.what() << "\n";
        }
    }

    // 3.3 建三个 session 并装配解码器。
    //     「模型从哪来、跑在哪块硬件上」是入口的决定，解码类只接收现成的
    //     session（与 Python build_ort_sessions 的分工一致）。
    Ort::Session encoder(env, encoder_path.c_str(), session_options);
    Ort::Session decoder_session(env, decoder_path.c_str(), session_options);
    Ort::Session joiner(env, joiner_path.c_str(), session_options);
    AsrDecoder decoder(encoder, decoder_session, joiner, tokens_path);
    const AsrDecoder::Info info = decoder.info();

    // 3.4 前端：tail_pad_frames 保持默认 100，收尾时补的特征 0 解码要用；
    //     那 100 帧 0 只喂解码器，不喂可视化（见 6.3）。n_mels 跟着模型走，
    //     避免两边维度对不上。采样率这里固定 16000，2.3 已校验设备一致。
    AudioPreprocess front(16000, info.n_mels,
                          25.0,   // 帧长 ms
                          10.0,   // 帧移 ms
                          512,    // FFT 点数
                          0.97,   // 预加重
                          20.0,   // Mel 下限 Hz
                          0.0,    // Mel 上限：0 = Nyquist
                          100);   // tail_pad_frames：解码路径补 100 帧

    // 3.5 语谱图：只接收特征，底部预留 56 px 白条，识别文字由入口画进去。
    SpectrogramImage view(info.n_mels, kSpecCols,
                          static_cast<double>(front.frame_step()) / 16000.0,
                          kDefaultLevelLo, kDefaultLevelHi, kCalibrateFrames,
                          kPlotSize, kHeaderH, kFooterH,
                          cv::COLORMAP_VIRIDIS, kLabel);

    // 3.6 中文字体：窗口模式才加载；失败只提示，draw_status_text 回退 ASCII。
    Utf8TextRenderer text_renderer;
    if (!no_window) {
        const FontCandidate font = find_existing_font(kCjkFontCandidates);
        const bool font_ok =
            !font.path.empty() && text_renderer.init(font.path, font.face_index);
        std::cout << "文字字体: "
                  << (font_ok ? font.path : std::string("未找到，回退 ASCII"))
                  << "\n";
    }

    // 3.7 模型与界面全部就绪后才真正开流：从这一刻起设备才开始转发音频，
    //     主循环紧接着就能取到数据，不会出现「模型加载 7~14 s、缓冲只有
    //     2.56 s」的启动丢块。start() 内部含 StartVoiceCom（同步阻塞约
    //     1.2 s）与上行保活线程；失败时上一行 init() 建立的会话还在，
    //     必须 deinit() 收尾，否则设备编码与登录态不会释放。
    if (!client.start()) {
        std::cerr << "[ASR] 语音转发启动失败：" << client.last_error() << "\n";
        client.deinit();
        return 1;
    }

    // 底部白条文字：优先 FreeType 渲染 UTF-8 中文，失败回退 ASCII putText。
    // Python 版这里用 cv2.putText（只支持 ASCII），C++ 补上中文渲染，
    // 位置与字号保持 (8, rows - 20) / 24 px 的观感。
    // 识别结果是累积假设，会越积越长；FreeType 路径按 frame.cols 算出的
    // 可用像素宽度从尾部截断（中英混排也准确），完整文本仍写在 stdout。
    auto draw_status_text = [&](cv::Mat& frame, const std::string& text) {
        const cv::Point org(kStatusMargin, frame.rows - 20);
        if (text_renderer.ready()) {
            // 画布宽度可能随窗口缩放变化，每帧按实际列数现算可用宽度。
            text_renderer.draw_tail(frame, text, org, kCjkFontHeight,
                                    frame.cols - 2 * kStatusMargin);
            return;
        }
        // 没有 FreeType 时只能回退 putText（中文变 '?'），按字数上限截。
        const std::string shown = utf8_tail(text, kMaxStatusChars);
        cv::putText(frame, shown, org, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    };

    // 3.8 装配 Session + Worker：从这里开始 encoder / decoder / joiner 只在
    //     后台线程里跑，主循环只负责取流、前端、渲染与文字展示。
    //     AsrSession 负责全量 token、64 token 分段与 log/asr_last_text.txt
    //     追加落盘；BlockWorker 只负责有界队列、串行与收尾。
    //     声明顺序：session 在 decoder 之后、worker 在 session 之后，C++ 逆序
    //     析构保证「先停线程，再销毁 Session 与模型会话」。
    AsrSessionConfig session_cfg;
    session_cfg.segment_tokens = 64;
    // 转写路径按工程根拼成绝对路径：不受运行目录影响。
    session_cfg.transcript_path = join_path(project_dir, "log/asr_last_text.txt");
    AsrSession session(decoder, session_cfg);
    BlockWorker<LogFbankFeatures> worker(
        [&session](const AudioFeatureInput& item) { session.on_block(item); },
        [](const LogFbankFeatures& f) {
            return f.n_frames <= 0 || f.data.empty();
        },
        50);

    // —— 四、打印配置（对应 Python 第 3 节）——
    std::cout << std::string(74, '=') << "\n"
              << "mini 实时 ASR —— 设备麦克风 -> 特征 -> 文字 + 语谱图\n"
              << std::string(74, '=') << "\n"
              << "设备  : " << kDeviceIp << ":" << kSdkPort
              << "  channel=" << kVoiceChannel << "\n"
              << "编码  : " << client.codec_name() << "  "
              << client.device_sample_rate() << " Hz"
              << (client.device_sample_rate() == sample_rate
                      ? std::string()
                      : " -> 降采样 " + std::to_string(sample_rate) + " Hz")
              << "\n"
              << "权重  : " << asr_dir << "\n"
              << "执行器: "
              << (use_cuda
                      ? (cuda_active ? "CUDA" : "CPU（CUDA 注册失败，已回退）")
                      : "CPU")
              << "\n"
              << "模型  : T=" << info.T
              << "  decode_chunk_len=" << info.decode_chunk_len
              << "  context_size=" << info.context_size
              << "  vocab=" << info.vocab_size
              << "  状态张量=" << info.n_states
              << "  n_mels=" << info.n_mels
              << "  blank_id=" << info.blank_id << "\n"
              << "前端  : 帧长 " << front.frame_size() << " 点("
              << front.frame_size() * 1000.0 / front.sample_rate() << "ms)"
              << "  帧移 " << front.frame_step() << " 点("
              << front.frame_step() * 1000.0 / front.sample_rate() << "ms)"
              << "\n"
              << "画布  : " << kSpecCols << " 列 x " << info.n_mels << " 行 = "
              << kSpecCols * front.frame_step() / front.sample_rate()
              << " 秒窗口\n"
              << std::string(74, '=') << "\n";
    if (run_seconds > 0.0) {
        std::cout << "自检模式：" << run_seconds << " 秒后自动收尾退出\n";
    }
    std::cout << "识别结果（Ctrl+C"
              << (no_window ? " 退出" : " 或窗口里按 q/Esc 退出")
              << "）：" << std::endl;
    if (!no_window) {
        std::cout << "窗口操作：q / Esc 退出，空格暂停/恢复语音转发" << std::endl;
    }

    // —— 五、主循环（对应 Python 第 4 节）——
    // 取一块音频 → 算特征 → 投给后台解码线程 → 读最新文字 → 画字 → 显示。
    // 解码不再阻塞主循环：即使一次 encoder 耗时超过一块音频，取流与渲染也
    // 照常进行（代价是底部文字比语谱图滞后，积压策略见 BlockWorker）。
    bool quit = false;
    bool quit_by_key = false;  // 区分「按键退出」和「Ctrl+C / 到点退出」
    bool auto_exit = false;    // --seconds 到点
    bool paused = false;       // 语音转发是否被空格暂停（与退出无关）
    cv::Mat canvas;
    std::vector<int16_t> pcm;
    std::vector<float> chunk;
    std::string shown;         // 已打印到 stdout 的文本
    double peak = 0.0;  // 峰值幅度（float 域），用于判断麦克风是否真采到声音
    const auto t_start = std::chrono::steady_clock::now();

    // 空格切换「暂停 / 恢复语音转发」：底层就是 client.stop() / client.start()，
    // 只停/起对讲线路，不重登录、不重建编码器；恢复后设备约 1.2 秒才送首包。
    // 界面不在这里重画：5.7 每轮都会重绘，下一轮自然显示新状态（最多晚 10 ms）。
    auto toggle_pause = [&]() {
        if (!paused) {
            if (client.stop()) {
                paused = true;
                std::cout << "\n[ASR] 已暂停语音转发，按空格恢复\n";
            } else {
                std::cout << "\n[ASR] 暂停失败：" << client.last_error() << "\n";
            }
        } else {
            std::cout << "\n[ASR] 正在恢复语音转发（首包约 1.2 秒后到达）...\n";
            if (client.start()) {
                paused = false;
            } else {
                std::cout << "[ASR] 恢复失败：" << client.last_error() << "\n";
            }
        }
    };

    while (!quit) {
        // 5.0 退出条件：Ctrl+C 置位，或 --seconds 到点。两者都不依赖有没有
        //     数据，避免设备没出声时永远跑下去。
        if (g_interrupted) {
            quit = true;
            break;
        }
        if (run_seconds > 0.0) {
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              t_start)
                    .count();
            if (elapsed >= run_seconds) {
                quit = true;
                auto_exit = true;
                break;
            }
        }

        // 5.1 取音频：一次性取走自上次调用以来累积的全部 PCM16，与是否
        //     录制无关；没有新数据时返回 false（此时缓冲已被清空）。
        //     没有数据不等于这一轮什么都不做：后台线程可能刚好解完上一块，
        //     文字仍要照常刷新，所以下面 5.6 / 5.7 放在 if 外面。
        const bool has_audio = client.get_last_audio(pcm) && !pcm.empty();
        // 5.2 int16 → float [-1,1]，与 Python mic/wav 的量纲对齐；顺手
        //     统计峰值幅度，收尾时用来提示「是不是根本没采到声音」。
        chunk.resize(pcm.size());
        for (size_t i = 0; i < pcm.size(); ++i) {
            const float value = static_cast<float>(pcm[i] * kPcmScale);
            chunk[i] = value;
            const float amplitude = value < 0.0f ? -value : value;
            if (amplitude > peak) {
                peak = amplitude;
            }
        }
        // 5.3 预处理：波形 → (n, 80) LogFBank。块长由设备回调节奏决定，
        //     不固定；前端按绝对样本索引凑帧，出帧结果与块长无关。
        const LogFbankFeatures feats = front.accept_waveform(chunk);
        // 5.4 可视化：有新帧才重建画布。设备约 40 ms 一批（通常 4 帧），
        //     所以画面按数据到达节奏前进 4 列，没有新帧时保持上一屏不动。
        //     不做「每轮都用 0 帧重画」：那只是把同一幅内容重复渲染，白烧 CPU。
        //     --no-window 时整段跳过。
        if (!no_window && has_audio && feats.n_frames > 0) {
            canvas = view.accept_features(feats);
        }
        if (has_audio) {
            // 5.5 投递解码：只把特征副本交给后台线程，主线程不等推理。
            //     本次 0 帧的纯空块由 push() 内部丢掉，不占队列、不唤醒线程。
            worker.push(feats);
        }
        

        // 5.6 读后台最新文本（解码器的尾部窗口）：只有文字变化时才刷新 stdout，
        //     补 8 个空格擦掉上一行多出来的尾巴。异步解码下文字会比语谱图
        //     滞后若干块，这是把推理挪出主线程的必然代价。
        const std::string text = session.latest_text();

        if (text != shown) {
            shown = text;
            std::cout << "\r" << shown << std::string(8, ' ') << std::flush;
        }

        if (!no_window) {
            // 5.7 识别结果画到底部白条，再显示：每轮都画、都 imshow。
            //     不按「画布或文字变过」跳过：部分窗口环境（WSLg / 远程 X）
            //     不保证自动重绘，跳过会表现为老字停在窗口里不动。暂停期间
            //     没有新帧，也靠这里每轮重画才有 [已暂停] 提示。
            //     画字前必须先清空底部白条：draw_tail 底层是 alpha 混合
            //     （dst = dst * (1 - alpha/255)），同一串字画在已经画过字的
            //     像素上会让边缘逐次变深，画布又不一定每轮重建，表现为文字
            //     一深一浅地闪。刷白 1200x55 区域约 200 KB，远比全量重建便宜。
            //     从白条第 2 行开始刷，保留首行那条黑分隔线（否则语谱图与
            //     白条会糊成一片）。
            //     画布还没生成（第一轮尚无音频）时跳过。
            if (!canvas.empty()) {
                canvas(cv::Rect(0, canvas.rows - kFooterH + 1, canvas.cols,
                                kFooterH - 1))
                    .setTo(cv::Scalar(255, 255, 255));
                draw_status_text(canvas, paused ? text + "  [已暂停]" : text);
                cv::imshow(kWindow, canvas);
            }
        }

        // 5.8 按键与节拍：有数据时按 1 ms 的节拍泵事件，保证 q/Esc 及时响应；
        //     没有数据时用 10 ms 节拍，窗口不假死；--no-window 下没有窗口可
        //     泵，直接 sleep 10 ms 让出 CPU（对应 Python 的 sleep）。
        //     空格暂停/恢复语音转发（暂停期间不再有新数据进来）。
        if (!no_window) {
            const int key = cv::waitKey(has_audio ? 1 : kIdleWaitMs) & 0xFF;
            if (key == 27 || key == 'q') {
                quit = true;
                quit_by_key = true;
            } else if (key == ' ') {
                toggle_pause();
            }
        } else if (!has_audio) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(kIdleWaitMs));
        }
    }

    // —— 六、收尾与统计（对应 Python 第 5 节）——
    // 6.1 关闭语音转发。
    client.stop();

    // 6.2 关流后缓冲区里可能还压着几块样本，先取走解完，否则最后一段话
    //     会因为没进前端而整段丢失。顺序不能反：deinit() 会清空取流缓冲。
    if (client.get_last_audio(pcm) && !pcm.empty()) {
        chunk.resize(pcm.size());
        for (size_t i = 0; i < pcm.size(); ++i) {
            const float value = static_cast<float>(pcm[i] * kPcmScale);
            chunk[i] = value;
            const float amplitude = value < 0.0f ? -value : value;
            if (amplitude > peak) {
                peak = amplitude;
            }
        }
        const LogFbankFeatures feats = front.accept_waveform(chunk);
        if (!no_window && feats.n_frames > 0) {
            // 与主循环一致：有新帧就贴一次整块。
            canvas = view.accept_features(feats);
        }
        // 与前文一致：只投递给后台线程，不在收尾阶段同步解码。
        worker.push(feats);
    }

    // 6.2.1 残余样本取完后释放设备会话：恢复设备编码（若本入口改过）、
    //       登出并 NET_DVR_Cleanup。放这里而不是程序末尾，是为了尽早释放
    //       设备占用；之后的解码、渲染与统计都不再需要设备连接。
    client.deinit();

    // 6.3 补尾解码：input_finished() 补出尾部帧，再在特征尾补 100 帧 0
    //     （log 域能量 1，不是静音值），使最后不满一个 chunk 的尾段也能被
    //     解码 —— 少了这一步最后几个字会被整段丢掉。返回值只喂解码器；
    //     喂可视化会多出一块比静音还亮的假色块。
    //     尾块走 push_final()（阻塞等空位、绝不丢弃），随后置结束标志并
    //     join：返回时队列已解空，AsrDecoder 不再被并发访问，主线程才能
    //     安全读最终文本与 n_chunks / all_token_ids 等统计。
    worker.push_final(front.input_finished());
    worker.finish_and_join();
    // join 之后 worker 线程已退出：这里结算尾段、关文件并对全量 token 算
    // 最终文本，没有并发（session.finish() 内含当前段结算）。
    const std::string final_text = session.finish();
    const auto end = std::chrono::steady_clock::now();
    const double wall_seconds =
        std::chrono::duration<double>(end - t_start).count();

    if (!no_window) {
        // 6.4 最后一屏：画布已随 5.4 / 6.2 更新到最新；一轮都没贴过时
        //     （极短会话）才用 view.frame() 按现有缓存补一张。
        //     不喂补 0 特征，避免多出假亮尾；完整识别结果写到底部白条
        //     （中文字体缺失时回退 ASCII）。这里也先刷白底条再画。
        if (canvas.empty()) {
            canvas = view.frame();
        }
        canvas(cv::Rect(0, canvas.rows - kFooterH + 1, canvas.cols,
                        kFooterH - 1))
            .setTo(cv::Scalar(255, 255, 255));
        draw_status_text(canvas, final_text);
        cv::imshow(kWindow, canvas);
    }

    // 6.5 统计：口径与 Python 5.3 一致 —— 墙钟时长、喂入样本 / 产出帧、
    //     encoder 调用次数与帧数、token 数、取流丢弃、峰值幅度。
    //     decoder 的统计现在由后台线程写；6.3 已 join，这里读没有并发。
    const AudioCaptureStats stats = client.get_stats();
    const std::vector<int64_t>& ids = session.all_token_ids();

    std::cout << "\n\n" << std::string(74, '=') << "\n"
              << "识别结果：" << final_text << "\n"
              << std::string(74, '=') << "\n"
              << "时长 " << wall_seconds << "s   实际转发 " << stats.forward_seconds
              << "s   喂入样本 " << front.n_samples() << "   产出帧 "
              << front.n_frames() << "\n"
              << "encoder 调用 " << decoder.n_chunks() << " 次 / 共 "
              << decoder.n_enc_frames() << " 帧   token " << ids.size()
              << " 个   转写段 " << session.n_segments() << " 段\n"
              << "转写  : " << session.transcript_path()
              << (session.transcript_ok() ? "（每段一行，已 flush）"
                                          : "（写入不可用）")
              << "\n"
              << "取流丢弃 " << static_cast<unsigned long long>(
                     stats.stream_dropped_samples)
              << "   解码丢弃 " << worker.dropped() << " 块（队列上限 "
              << worker.max_pending() << "）   峰值幅度 " << peak
              << "\n"
              << "上行保活 发送 " << static_cast<unsigned long long>(
                     stats.uplink_sent_frames)
              << " 帧 / 失败 " << static_cast<unsigned long long>(
                     stats.uplink_send_errors)
              << " 次（每 40 ms 回发一帧 G.722 静音，PCM 下行同样适用）\n"
              << "静音补零 " << static_cast<unsigned long long>(
                     stats.gap_padded_samples)
              << " 样本 / " << stats.gap_events
              << " 次（设备停顿按回调到达时刻补 0，对齐墙钟）\n";

    // 与 Python 5.4 相同的两条提示：丢块说明主循环跟不上取流速度；
    // 峰值为 0 说明设备根本没采到声音。后两条是本入口独有的：
    // 解码队列丢弃说明推理吞吐低于实时，异常说明后台线程已经停了。
    if (stats.stream_dropped_samples > 0) {
        std::cout << "提示：出现丢块，说明主循环跟不上采集速度（取流缓冲上限 "
                     "40960 样本 ≈ 2.56 s）。\n";
    }
    if (worker.dropped() > 0) {
        std::cout << "提示：后台解码跟不上采集，已丢弃最早的 "
                  << worker.dropped()
                  << " 块特征（显示文字会跳）；可改用 --cuda 或减少其它负载。\n";
    }
    // Handler 的异常以 exception_ptr 保存，这里展开成人能读的原因。
    const std::exception_ptr decode_error = worker.error();
    if (decode_error) {
        std::string reason = "未知异常";
        try {
            std::rethrow_exception(decode_error);
        } catch (const std::exception& e) {
            reason = e.what();
        } catch (...) {
        }
        std::cout << "警告：后台解码线程异常退出：" << reason
                  << "（其后的音频未参与识别）\n";
    }
    if (!session.transcript_ok()) {
        std::cout << "提示：转写文件不可用：" << session.transcript_error()
                  << "（识别本身不受影响）\n";
    }
    if (peak == 0.0) {
        std::cout << "提示：峰值幅度为 0，设备可能没采到声音，"
                     "请确认拾音器与语音通道配置。\n";
    }

    // 6.6 关窗：按键退出直接关；Ctrl+C 或到点退出则停住最后一屏，等用户
    //     看清结果再按键关闭（对应 Python 的 waitKey(0) 分支）。
    if (!no_window) {
        if (!canvas.empty() && (auto_exit || !quit_by_key)) {
            std::cout << "\n按任意键关闭语谱图窗口……\n";
            cv::waitKey(0);
        }
        cv::destroyAllWindows();
    }
    return 0;
}

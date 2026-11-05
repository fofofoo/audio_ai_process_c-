/**
 * @file test_mic_kws_asr.cpp
 * @brief 实时双模式入口：海康设备取流 → KWS 监听 → 唤醒后 ASR 识别（对应
 *        Python test_mic_kws_asr.py）。
 *
 * ============================================================
 * 这份代码是干什么的
 * ============================================================
 * 平时只跑关键词唤醒，命中唤醒词后切到完整识别，出现结束语或空闲超时再
 * 切回监听。两条解码分支共用同一份 LogFBank 特征与同一个后台线程：
 *
 *   海康拾音器 → HkSdkAudioClient（G.722.1，16 kHz）
 *              → get_last_audio() → int16 / 32768.0
 *              → AudioPreprocess（(n, 80) LogFBank）
 *              → 同一份特征分两路：
 *                   SpectrogramImage（主线程滚动语谱图）
 *                   BlockWorker（有界队列 + 后台线程）
 *                        → KwsAsrSession
 *                             ├ KWS 模式：KwsDecoder（trie 约束束搜索）
 *                             └ ASR 模式：AsrSession → AsrDecoder
 *                        → 快照（模式 / 文本 / 事件 / 计数）
 *              → 主线程读快照：stdout + 窗口底部两行状态
 *
 * 入口只做「参数 + 数据源 + 装配」：设备地址、两组模型路径、唤醒词、阈值、
 * 超时与回填帧数都直白写在 main，流水线细节在 Session 与解码类里。
 *
 * ============================================================
 * 与 Python 版 / 单模式入口的差异
 * ============================================================
 *   - 音频源是海康设备语音转发（G.722.1 固定 16 kHz），不是 sounddevice；
 *     输出采样率不是 16 kHz 时直接拒绝启动，不静默重采样。
 *   - 两个解码器与模式机放在同一个后台 Worker 线程：Python 在主循环里同步
 *     解码；Jetson CPU 上主循环吞吐约 0.99，推理留在主循环会让取流缓冲
 *     （2.56 s）溢出。这里主线程只做取流、前端、渲染与快照读取。
 *   - 回填、切回、结束语判定、空闲计时都收进 KwsAsrSession，入口不再维护
 *     模式变量；Python 版把这些状态摊在主循环里。
 *   - ASR 转写按 64 个 token 结算一段并追加 log/asr_last_text.txt（Python
 *     版没有落盘）；切回 KWS 时先结算尾段再清空本会话文本，转写文件保留。
 *   - 窗口底部优先 FreeType 渲染中文（Python 的 cv2.putText 只支持 ASCII）。
 *   - 新增 --no-window / --cuda / --seconds / --initial-asr / --keywords /
 *     --record 六个开关（Python 版配置全写死）。
 *   - Ctrl+C 用 SIGINT 置标志位、走正常收尾；Python 捕获 KeyboardInterrupt。
 *
 * ============================================================
 * 调用顺序（对照 main 里的编号）
 * ============================================================
 *   一、参数与命令行         设备、两组模型、关键词、画布、字体、开关
 *   二、连接设备（Python 1） 2.1 配置  2.2 init  2.3 采样率校验
 *   三、加载模型与装配（Python 2~4）
 *       3.1 session 选项    3.2 可选 CUDA EP  3.3 ASR 三 session
 *       3.4 KWS 三 session + 关键词  3.5 前端  3.6 语谱图  3.7 中文字体
 *       3.8 start 开流      3.9 AsrSession + KwsAsrSession + BlockWorker
 *   四、打印配置（Python 5）
 *   五、主循环（Python 6）
 *       5.0 退出条件  5.1 取音频  5.2 归一化  5.3 前端
 *       5.4 可视化  5.5 投递解码  5.6 读快照/事件/文本
 *       5.7 画两行状态  5.8 按键与节拍
 *   六、收尾与统计（Python 7）
 *       6.1 停流  6.2 残余样本  6.2.1 deinit
 *       6.3 尾块 + join + finish  6.4 最后一屏  6.5 统计  6.6 关窗
 *
 * KWS 模式不补特征域 0（那是 log 能量 1，不是真静音），收尾块由
 * KwsAsrSession 直接跳过；ASR 模式才把补 0 的尾块正常解码。
 *
 * ============================================================
 * 用法（项目根目录）
 * ============================================================
 *   ./build/test_mic_kws_asr.out
 *       默认 KWS 监听；窗口底部显示模式与识别结果，q / Esc 退出，
 *       空格暂停/恢复语音转发
 *   ./build/test_mic_kws_asr.out --no-window
 *       只打印文本与事件（无显示器 / 远程调试）
 *   ./build/test_mic_kws_asr.out --seconds 20 --no-window
 *       自检：跑 20 秒后自动收尾退出
 *   ./build/test_mic_kws_asr.out --cuda
 *       ASR 与 KWS 都使用 CUDA EP，失败自动回退 CPU
 *   ./build/test_mic_kws_asr.out --initial-asr
 *       初始进入 ASR 模式（调试纯识别行为，等价单模式入口）
 *   ./build/test_mic_kws_asr.out --keywords keywords/kws_keywords.txt
 *       覆盖默认唤醒词文件
 *   ./build/test_mic_kws_asr.out --no-window --record --seconds 45
 *       长跑同时把音频录到 results/kws_asr_<时间戳>.wav，便于事后回放复现
 */

#include "audio_ai_process/asr_decode.h"
#include "audio_ai_process/asr_session.h"
#include "audio_ai_process/audio_preprocess.h"
#include "audio_ai_process/block_worker.h"
#include "audio_ai_process/kws_asr_session.h"
#include "audio_ai_process/kws_decode.h"
#include "audio_ai_process/spectrogram_image.h"
#include "audio_ai_process/text_renderer.h"
#include "audio_ai_process/text_utils.h"
#include "hk_sdk_audio_client.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
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

// 生成带时间戳的录音路径，避免多次运行互相覆盖。
// 格式：results/kws_asr_20261105_153045.wav（本地时间）；采集库内部会把
// 相对路径转成绝对路径并建目录，这里只负责拼名字。
std::string make_record_path(const std::string& project_dir) {
    const std::time_t now_time =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm local_tm{};
    localtime_r(&now_time, &local_tm);
    std::ostringstream name;
    name << "kws_asr_" << std::put_time(&local_tm, "%Y%m%d_%H%M%S") << ".wav";
    return join_path(join_path(project_dir, "results"), name.str());
}

}  // namespace

int main(int argc, char** argv) {
    // —— 一、参数与命令行 ——
    // 1.1 设备参数：与 test_mic_asr.cpp 保持一致，换设备改这里。
    const std::string kDeviceIp = "10.168.2.65";  // 音频设备 IP
    const int kSdkPort = 8000;                    // HCNetSDK 私有协议端口
    const std::string kUsername = "admin";
    const std::string kPassword = "Password";
    const int kVoiceChannel = 1;                  // 语音通道号，本机实测为 1

    // 1.2 ASR 模型与词表：与 test_mic_asr.cpp / test_wav_asr.cpp 同一套 int8 包。
    //     encoder / joiner 是 int8，decoder 保持 fp32（导出时就是这么切的）。
    const std::string project_dir = HK_PROJECT_DIR;
    const std::string asr_dir =
        project_dir +
        "/weight/"
        "sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-"
        "2026-06-05";
    const std::string asr_encoder_path = join_path(asr_dir, "encoder.int8.onnx");
    const std::string asr_decoder_path = join_path(asr_dir, "decoder.onnx");
    const std::string asr_joiner_path = join_path(asr_dir, "joiner.int8.onnx");
    const std::string asr_tokens_path = join_path(asr_dir, "tokens.txt");

    // 1.3 KWS 模型与关键词：默认 chunk-8（160 ms 推进，唤醒延迟低于 chunk-16），
    //     encoder / joiner int8，decoder fp32。关键词文件在仓库根 keywords/，
    //     改唤醒词改它，不动 weight/ 里的原始文件。
    const std::string kws_dir =
        project_dir +
        "/weight/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20";
    const std::string kws_encoder_path =
        join_path(kws_dir, "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx");
    const std::string kws_decoder_path =
        join_path(kws_dir, "decoder-epoch-13-avg-2-chunk-8-left-64.onnx");
    const std::string kws_joiner_path =
        join_path(kws_dir, "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx");
    const std::string kws_tokens_path = join_path(kws_dir, "tokens.txt");
    const std::string kws_keywords_path =
        join_path(project_dir, "keywords/kws_keywords.txt");

    // 1.4 KWS 解码参数：与 test_wav_kws.cpp / Python 参考实现口径一致。
    //     关键词行里的 `:score #threshold` 会覆盖这两个默认值。
    const int kMaxActivePaths = 4;            // 束宽
    const int kNumTrailingBlanks = 1;         // 触发前至少 2 个 blank 输出帧
    const float kKeywordsScore = 1.0f;        // 默认 boosting score
    const float kKeywordsThreshold = 0.25f;   // 默认触发阈值
    const float kResetAfterSilence = 1.5f;    // 连续静音软重置（秒）
    const bool kKwsDebug = true;              // 记录被阈值挡住的候选，调参用

    // 1.5 双模式参数：与 Python 参考实现同值。
    const double kAsrIdleTimeout = 10.0;  // ASR 多久没有新 token 就切回（秒）
    const int kAsrBackfillFrames = 100;   // 唤醒时回填最近多少帧（1 帧 = 10 ms）

    // 1.6 画布尺寸：与 Python 版一致（400 列 x 10 ms = 4 秒滚动窗口）。
    //     底部 56 px 白条分两行：模式状态 + 识别文本；顶部 52 px 写设备信息。
    const int kSpecCols = 400;
    const int kFooterH = 56;
    const int kCalibrateFrames = 50;      // 前 0.5 秒标定色阶后锁死
    const cv::Size kPlotSize(1200, 300);  // 语谱图显示尺寸（宽，高）
    const int kHeaderH = 52;
    const std::string kLabel = "hk mic";  // 信息条标签（ASCII）
    // 窗口标题只能用 ASCII：部分 OpenCV 构建对非 ASCII 标题处理不一致。
    const std::string kWindow = "mini mic kws+asr";

    // 1.7 中文字体：按顺序取第一个存在的文件。
    //     Noto Sans CJK 的 .ttc 里 0=JP、1=KR、2=SC、3=TC、4=HK，简中取 2；
    //     Droid / 文泉驿是单字面文件，取 0。FreeType 未编译进目标时整段无效，
    //     窗口文字自动回退 cv::putText（中文会变成 '?'）。
    const std::vector<FontCandidate> kCjkFontCandidates = {
        {"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 2},
        {"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf", 0},
        {"/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0},
        {"/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 0},
    };
    const int kCjkFontHeight = 24;        // 识别文本字号（像素）
    const int kCjkFontHeightSmall = 20;   // 模式状态行字号（像素）
    // 底部文字的左右边距（px）：可用宽度 = 画布宽 − 2 × 边距。
    // FreeType 路径按这个像素宽度从尾部截断，不再按字数估算。
    const int kStatusMargin = 8;
    // ASCII 回退路径的上限：cv::putText 只支持 ASCII，字宽约 12 px，
    // 50 个字符 ≈ 600 px 远小于画布宽度；FreeType 不可用时用它兜底。
    const size_t kMaxStatusChars = 50;

    // 1.8 数据与节奏常量。
    //     int16 → float 的归一化系数。直接拿 int16 当浮点喂前端会让 log
    //     能量整体偏移 ln(32768²) ≈ 20.8，与 Python 的 float [-1,1] 量纲对齐。
    constexpr double kPcmScale = 1.0 / 32768.0;
    // 没有新数据时的等待时间：既让出 CPU，又让 OpenCV 能泵事件（窗口不假死）。
    // 10 ms 与特征帧移一致，最长延迟一帧。
    constexpr int kIdleWaitMs = 10;

    // 1.9 命令行开关。设备地址、模型路径、画布尺寸仍写死在上方，
    //     命令行只控制「开不开窗口 / 用不用 GPU / 跑多久 / 初始模式 / 唤醒词 / 录音」。
    bool no_window = false;
    bool use_cuda = false;
    bool initial_asr = false;
    bool record = false;       // 长跑时把取到的音频存 wav，便于回放复现
    double run_seconds = 0.0;  // >0 时到点自动收尾，便于无人值守自检
    std::string keywords_path = kws_keywords_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--no-window") {
            no_window = true;
        } else if (arg == "--cuda") {
            use_cuda = true;
        } else if (arg == "--initial-asr") {
            initial_asr = true;
        } else if (arg == "--record") {
            record = true;
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
        } else if (arg == "--keywords") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--keywords 需要一个文件路径\n");
                return 2;
            }
            keywords_path = argv[++i];
        } else {
            std::fprintf(stderr, "未知参数：%s\n", arg.c_str());
            std::fprintf(stderr,
                         "用法：%s [--no-window] [--cuda] [--seconds 秒数] "
                         "[--initial-asr] [--keywords 文件] [--record]\n",
                         argv[0]);
            return 2;
        }
    }
    // 录音路径在 start() 成功后才生成（含时间戳），失败时保持空表示未录音。
    std::string record_path;

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
    capture_config.set_audio_format = false;
    capture_config.audio_enc_type = 0;        // 0=G.722.1(16k)，8=PCM(32k)
    capture_config.restore_audio_format = true;

    // 2.1.2 输出采样率钉在 16 kHz：预处理模块与两组模型只认 16 kHz。
    //       G.722.1 本来就是 16 kHz（直通）；切到 PCM 时客户端内部做
    //       32→16 抗混叠降采样，wav、取流缓冲、统计都按 16 kHz 输出。
    capture_config.output_sample_rate = 16000;

    // 2.2 建立设备会话：只登录 + 查询编码，不开流，也不写 wav 文件。
    //     开流要等到六个模型都装配完（3.8）再做：Jetson 上两组模型加载
    //     更久，若此时设备已在转发，音频会持续压进取流缓冲（上限 2.56 s）
    //     并在进入主循环前就被丢弃。
    HkSdkAudioClient client(capture_config);
    if (!client.init()) {
        std::cerr << "[KWS+ASR] 设备初始化失败：" << client.last_error() << "\n";
        return 1;
    }

    // 2.3 采样率校验：两组模型都按 16 kHz 训练，喂别的采样率不会报错、
    //     只会识别出乱码。不满足说明 output_sample_rate 与设备率无法匹配，
    //     直接拒绝启动，不偷偷重采样。
    const int sample_rate = client.sample_rate();
    if (sample_rate != 16000) {
        std::cerr << "[KWS+ASR] 输出采样率是 " << sample_rate
                  << " Hz，而模型固定 16 kHz。\n"
                  << "          请检查 capture_config.output_sample_rate "
                     "（应为 16000）；\n"
                  << "          设备实际采样率 = " << client.device_sample_rate()
                  << " Hz。\n";
        client.deinit();
        return 1;
    }

    // —— 三、加载模型与装配（对应 Python 第 2~4 节）——
    // 3.1 session 选项：多个 chunk 之间本来就串行，intra / inter 各 1 线程；
    //     日志降到 WARNING，避免加载时刷屏。
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test_mic_kws_asr");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 3.2 可选 CUDA EP：注册失败只打印并回退 CPU，不中断识别。
    //     必须用 AppendExecutionProvider_CUDA；通用 AppendExecutionProvider
    //     只支持 QNN / SNPE / XNNPACK。两组 session 共用同一份选项，
    //     注册一次即可同时作用于 ASR 与 KWS。
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

    // 3.3 ASR：三个 session + AsrDecoder。
    //     「模型从哪来、跑在哪块硬件上」是入口的决定，解码类只接收现成的
    //     session（与 Python build_ort_sessions 的分工一致）。
    Ort::Session asr_encoder(env, asr_encoder_path.c_str(), session_options);
    Ort::Session asr_decoder_session(env, asr_decoder_path.c_str(),
                                     session_options);
    Ort::Session asr_joiner(env, asr_joiner_path.c_str(), session_options);
    AsrDecoder asr_decoder(asr_encoder, asr_decoder_session, asr_joiner,
                           asr_tokens_path);
    const AsrDecoder::Info asr_info = asr_decoder.info();

    // 3.4 KWS：三个 session + KwsDecoder（加载词表、唤醒词并建 trie）。
    //     debug=true 记录被 #threshold 挡住的候选，退出时汇总。
    Ort::Session kws_encoder(env, kws_encoder_path.c_str(), session_options);
    Ort::Session kws_decoder_session(env, kws_decoder_path.c_str(),
                                     session_options);
    Ort::Session kws_joiner(env, kws_joiner_path.c_str(), session_options);
    KwsDecoder kws_decoder(kws_encoder, kws_decoder_session, kws_joiner,
                           kws_tokens_path, keywords_path, kMaxActivePaths,
                           kNumTrailingBlanks, kKeywordsScore,
                           kKeywordsThreshold, kResetAfterSilence, 16000,
                           kKwsDebug);
    const KwsDecoder::Info kws_info = kws_decoder.info();

    // 两组模型共用同一份特征，n_mels 必须一致；不一致说明权重包配错，
    // 继续跑只会在 encoder 输入维度上抛异常，提前报清楚。
    if (kws_info.n_mels != asr_info.n_mels) {
        std::cerr << "[KWS+ASR] KWS 的 n_mels=" << kws_info.n_mels
                  << " 与 ASR 的 n_mels=" << asr_info.n_mels
                  << " 不一致，请检查两组权重。\n";
        client.deinit();
        return 1;
    }

    // 3.5 前端：tail_pad_frames 保持默认 100，收尾时补的特征 0 解码要用；
    //     那 100 帧 0 只喂解码器，不喂可视化（见 6.3）。n_mels 跟着模型走。
    AudioPreprocess front(16000, asr_info.n_mels,
                          25.0,   // 帧长 ms
                          10.0,   // 帧移 ms
                          512,    // FFT 点数
                          0.97,   // 预加重
                          20.0,   // Mel 下限 Hz
                          0.0,    // Mel 上限：0 = Nyquist
                          100);   // tail_pad_frames：解码路径补 100 帧

    // 3.6 语谱图：只接收特征，底部预留 56 px 白条，模式与文字由入口画进去。
    SpectrogramImage view(asr_info.n_mels, kSpecCols,
                          static_cast<double>(front.frame_step()) / 16000.0,
                          kDefaultLevelLo, kDefaultLevelHi, kCalibrateFrames,
                          kPlotSize, kHeaderH, kFooterH,
                          cv::COLORMAP_VIRIDIS, kLabel);

    // 3.7 中文字体：窗口模式才加载；失败只提示，draw_status_text 回退 ASCII。
    Utf8TextRenderer text_renderer;
    if (!no_window) {
        const FontCandidate font = find_existing_font(kCjkFontCandidates);
        const bool font_ok =
            !font.path.empty() && text_renderer.init(font.path, font.face_index);
        std::cout << "文字字体: "
                  << (font_ok ? font.path : std::string("未找到，回退 ASCII"))
                  << "\n";
    }

    // 3.8 模型与界面全部就绪后才真正开流：从这一刻起设备才开始转发音频，
    //     主循环紧接着就能取到数据，不会出现「加载十几秒、缓冲只有 2.56 s」
    //     的启动丢块。start() 内部含 StartVoiceCom（同步阻塞约 1.2 s）与
    //     上行保活线程；失败时上一行 init() 建立的会话还在，必须 deinit()。
    if (!client.start()) {
        std::cerr << "[KWS+ASR] 语音转发启动失败：" << client.last_error()
                  << "\n";
        client.deinit();
        return 1;
    }

    // 3.8.1 可选录音：开流成功后立刻开始写盘，退出时 client.stop() 会自动
    //       stop_record() 回填 wav 头并改名。录音失败只告警、不中断识别测试：
    //       录音是辅助手段，不能因为它写不了盘就放弃本轮真机验证。
    //       注意：窗口模式按空格暂停走 client.stop()，会把录音一并结束且恢复后
    //       不再续录——需要完整录音时用 --no-window 或不按暂停。
    if (record) {
        record_path = make_record_path(project_dir);
        if (!client.start_record(record_path)) {
            std::cerr << "[警告] 录音启动失败：" << client.last_error()
                      << "，继续识别测试\n";
            record_path.clear();
        }
    }

    // 3.9 装配会话与 Worker。
    //     AsrSession：全量 token、64 token 分段、log/asr_last_text.txt 落盘。
    //     KwsAsrSession：模式机、回填环、空闲计时、事件与统计。
    //     BlockWorker：唯一后台线程 + 有界队列，两个解码器都只在这个线程里跑。
    //     声明顺序：asr_decoder / kws_decoder → asr_session → session → worker，
    //     C++ 逆序析构保证「先停线程，再销毁 Session 与模型会话」。
    AsrSessionConfig asr_session_cfg;
    asr_session_cfg.segment_tokens = 64;
    // 转写路径按工程根拼成绝对路径：不受运行目录影响。
    asr_session_cfg.transcript_path =
        join_path(project_dir, "log/asr_last_text.txt");
    AsrSession asr_session(asr_decoder, asr_session_cfg);

    KwsAsrSessionConfig kws_asr_cfg;
    kws_asr_cfg.initial_mode = initial_asr ? KwsAsrMode::Asr : KwsAsrMode::Kws;
    kws_asr_cfg.idle_timeout_s = kAsrIdleTimeout;
    kws_asr_cfg.backfill_frames = kAsrBackfillFrames;
    // end_phrases 用结构体默认值（与 Python END_PHRASES 一致），下面打印出来。
    KwsAsrSession session(asr_decoder, kws_decoder, asr_session, kws_asr_cfg);

    BlockWorker<LogFbankFeatures> worker(
        [&session](const AudioFeatureInput& item) { session.on_block(item); },
        [](const LogFbankFeatures& f) {
            return f.n_frames <= 0 || f.data.empty();
        },
        50);

    // 底部文字：优先 FreeType 渲染 UTF-8 中文，失败回退 ASCII putText。
    // 两行分开画：第一行模式与计数（小字），第二行识别文本（大字）。
    // 文本是累积假设，会越积越长；FreeType 路径按 frame.cols 算出的可用
    // 像素宽度从尾部截断，完整文本仍写在 stdout。
    auto draw_status_line = [&](cv::Mat& frame, const std::string& text,
                                int baseline, int font_height) {
        const cv::Point org(kStatusMargin, frame.rows - baseline);
        if (text_renderer.ready()) {
            // 画布宽度可能随窗口缩放变化，每帧按实际列数现算可用宽度。
            text_renderer.draw_tail(frame, text, org, font_height,
                                    frame.cols - 2 * kStatusMargin);
            return;
        }
        // 没有 FreeType 时只能回退 putText（中文变 '?'），按字数上限截。
        const std::string shown = utf8_tail(text, kMaxStatusChars);
        cv::putText(frame, shown, org, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    };

    // —— 四、打印配置（对应 Python 第 5 节）——
    std::cout << std::string(74, '=') << "\n"
              << "mini 实时 KWS + ASR —— 唤醒词监听 -> 完整识别 + 语谱图\n"
              << std::string(74, '=') << "\n"
              << "设备  : " << kDeviceIp << ":" << kSdkPort
              << "  channel=" << kVoiceChannel << "\n"
              << "编码  : " << client.codec_name() << "  "
              << client.device_sample_rate() << " Hz"
              << (client.device_sample_rate() == sample_rate
                      ? std::string()
                      : " -> 降采样 " + std::to_string(sample_rate) + " Hz")
              << "\n"
              << "ASR   : " << asr_dir << "\n"
              << "KWS   : " << kws_dir << "\n"
              << "执行器: "
              << (use_cuda
                      ? (cuda_active ? "CUDA" : "CPU（CUDA 注册失败，已回退）")
                      : "CPU")
              << "\n"
              << "ASR 模型: T=" << asr_info.T
              << "  decode_chunk_len=" << asr_info.decode_chunk_len
              << "  context_size=" << asr_info.context_size
              << "  vocab=" << asr_info.vocab_size
              << "  状态张量=" << asr_info.n_states
              << "  n_mels=" << asr_info.n_mels
              << "  blank_id=" << asr_info.blank_id << "\n"
              << "KWS 模型: T=" << kws_info.T
              << "  decode_chunk_len=" << kws_info.decode_chunk_len
              << "  context_size=" << kws_info.context_size
              << "  vocab=" << kws_info.vocab_size
              << "  状态张量=" << kws_info.n_states
              << "  帧长=" << kws_info.frame_shift_s * 1e3 << "ms\n"
              << "前端  : 帧长 " << front.frame_size() << " 点("
              << front.frame_size() * 1000.0 / front.sample_rate() << "ms)"
              << "  帧移 " << front.frame_step() << " 点("
              << front.frame_step() * 1000.0 / front.sample_rate() << "ms)"
              << "\n"
              << "唤醒词: " << kws_info.n_keywords << " 条  ";
    for (size_t i = 0; i < kws_decoder.keywords().size(); ++i) {
        std::cout << (i == 0 ? "" : " / ") << kws_decoder.keywords()[i];
    }
    std::cout << "\n"
              << "切换  : 初始=" << mode_name(kws_asr_cfg.initial_mode)
              << "  空闲超时=" << kws_asr_cfg.idle_timeout_s << "s"
              << "  回填=" << kws_asr_cfg.backfill_frames << "帧"
              << "  结束语=";
    for (size_t i = 0; i < kws_asr_cfg.end_phrases.size(); ++i) {
        std::cout << (i == 0 ? "" : " / ") << kws_asr_cfg.end_phrases[i];
    }
    std::cout << "\n"
              << "录音  : "
              << (!record ? std::string("关闭")
                          : (record_path.empty() ? std::string("启用失败")
                                                 : record_path))
              << "\n"
              << "画布  : " << kSpecCols << " 列 x " << asr_info.n_mels
              << " 行 = " << kSpecCols * front.frame_step() / front.sample_rate()
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

    // —— 五、主循环（对应 Python 第 6 节）——
    // 取一块音频 → 算特征 → 投给后台 Session → 读快照 → 画字 → 显示。
    // 主线程从头到尾不调用模型；模式切换与解码都在 Worker 线程里完成。
    bool quit = false;
    bool quit_by_key = false;  // 区分「按键退出」和「Ctrl+C / 到点退出」
    bool auto_exit = false;    // --seconds 到点
    bool paused = false;       // 语音转发是否被空格暂停（与退出无关）
    cv::Mat canvas;
    std::vector<int16_t> pcm;
    std::vector<float> chunk;
    std::string shown;         // 已打印到 stdout 的文本
    int64_t shown_event_seq = 0;  // 已打印的事件序号
    double peak = 0.0;  // 峰值幅度（float 域），用于判断设备是否真采到声音
    const auto t_start = std::chrono::steady_clock::now();

    // 空格切换「暂停 / 恢复语音转发」：底层就是 client.stop() / client.start()，
    // 只停/起对讲线路，不重登录、不重建编码器；恢复后设备约 1.2 秒才送首包。
    // 暂停期间 Worker 没有新块可解，模式与文本保持不变；界面每轮重画，
    // 下一轮自然显示新状态（最多晚 10 ms）。
    auto toggle_pause = [&]() {
        if (!paused) {
            if (client.stop()) {
                paused = true;
                std::cout << "\n[KWS+ASR] 已暂停语音转发，按空格恢复\n";
            } else {
                std::cout << "\n[KWS+ASR] 暂停失败：" << client.last_error()
                          << "\n";
            }
        } else {
            std::cout << "\n[KWS+ASR] 正在恢复语音转发（首包约 1.2 秒后到达）...\n";
            if (client.start()) {
                paused = false;
            } else {
                std::cout << "[KWS+ASR] 恢复失败：" << client.last_error()
                          << "\n";
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
        //     没有数据不等于这一轮什么都不做：后台线程可能刚切了模式，
        //     状态与文字仍要照常刷新，所以下面 5.6 / 5.7 放在 if 外面。
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
        //     --no-window 时整段跳过。
        if (!no_window && has_audio && feats.n_frames > 0) {
            canvas = view.accept_features(feats);
        }
        if (has_audio) {
            // 5.5 投递：只把特征副本交给后台线程，主线程不等推理。
            //     本次 0 帧的纯空块由 push() 内部丢掉，不占队列、不唤醒线程。
            //     模式判断在 Worker 线程的 on_block 里做，入口不关心此刻是
            //     哪种模式，只负责把特征按到达顺序送进去。
            worker.push(feats);
        }

        // 5.6 读快照：事件（唤醒 / 结束语 / 超时）只打印一次；文本只在
        //     变化时刷新 stdout，补 8 个空格擦掉上一行多出来的尾巴。
        //     异步解码下文字会比语谱图滞后若干块，这是把推理挪出主线程的
        //     必然代价。
        const KwsAsrSnapshot snap = session.snapshot();
        if (snap.event_seq != shown_event_seq) {
            shown_event_seq = snap.event_seq;
            std::cout << "\n" << snap.last_event << "\n";
        }
        const std::string text = snap.text;
        if (text != shown) {
            shown = text;
            std::cout << "\r" << shown << std::string(8, ' ') << std::flush;
        }

        if (!no_window) {
            // 5.7 两行状态画到底部白条，再显示：每轮都画、都 imshow。
            //     不按「画布或文字变过」跳过：部分窗口环境（WSLg / 远程 X）
            //     不保证自动重绘，跳过会表现为老字停在窗口里不动。暂停期间
            //     没有新帧，也靠这里每轮重画才有 [已暂停] 提示。
            //     画字前必须先清空底部白条：draw_tail 底层是 alpha 混合，
            //     同一串字画在已经画过字的像素上会让边缘逐次变深，画布又
            //     不一定每轮重建，表现为文字一深一浅地闪。刷白 1200x55 区域
            //     约 200 KB，远比全量重建便宜。
            //     从白条第 2 行开始刷，保留首行那条黑分隔线（否则语谱图与
            //     白条会糊成一片）。画布还没生成（第一轮尚无音频）时跳过。
            if (!canvas.empty()) {
                canvas(cv::Rect(0, canvas.rows - kFooterH + 1, canvas.cols,
                                kFooterH - 1))
                    .setTo(cv::Scalar(255, 255, 255));
                // 第一行：模式 + 计数（小字）。ASR 模式下 text 是识别结果。
                std::ostringstream status;
                status << mode_name(snap.mode) << " | listening"
                       << "   唤醒 " << snap.wake_count << " / 切回 "
                       << snap.back_count;
                draw_status_line(canvas, status.str(), 34, kCjkFontHeightSmall);
                // 第二行：识别文本（大字）；KWS 模式为空，只留白。
                const std::string line2 =
                    paused ? text + "  [已暂停]" : text;
                draw_status_line(canvas, line2, 10, kCjkFontHeight);
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

    // —— 六、收尾与统计（对应 Python 第 7 节）——
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

    // 6.3 尾块 + join：前端补出尾部帧，再在特征尾补 100 帧 0（log 域能量 1，
    //     不是静音值），使最后不满一个 chunk 的尾段也能解码。尾块带 is_tail：
    //     KwsAsrSession 在 KWS 模式下直接跳过（特征域 0 对束搜索没有意义），
    //     在 ASR 模式下正常解码。尾块走 push_final()（阻塞等空位、绝不丢弃），
    //     随后置结束标志并 join：返回时队列已解空，解码器不再被并发访问，
    //     主线程才能安全读最终文本与统计。
    worker.push_final(front.input_finished());
    worker.finish_and_join();
    // join 之后 Worker 线程已退出：结算尾段、关文件并返回最近一次 ASR 文本。
    const std::string final_text = session.finish();
    const KwsAsrStats mode_stats = session.stats();
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
        const KwsAsrSnapshot snap = session.snapshot();
        std::ostringstream status;
        status << mode_name(snap.mode) << " | stopped   唤醒 "
               << snap.wake_count << " / 切回 " << snap.back_count;
        draw_status_line(canvas, status.str(), 34, kCjkFontHeightSmall);
        draw_status_line(canvas, final_text, 10, kCjkFontHeight);
        cv::imshow(kWindow, canvas);
    }

    // 6.5 统计：口径与 Python 7.3 一致 —— 墙钟时长、喂入样本 / 产出帧、
    //     两种解码器的 encoder 调用与帧数、token / 触发数、唤醒 / 切回次数、
    //     回填帧数、转写段数、取流与解码队列丢弃、峰值幅度。
    //     解码器统计现在由后台线程写；6.3 已 join，这里读没有并发。
    const AudioCaptureStats stats = client.get_stats();
    const std::vector<int64_t>& ids = asr_session.all_token_ids();

    std::cout << "\n\n" << std::string(74, '=') << "\n"
              << "识别结果：" << final_text << "\n"
              << std::string(74, '=') << "\n"
              << "时长 " << wall_seconds << "s   实际转发 " << stats.forward_seconds
              << "s   喂入样本 " << front.n_samples() << "   产出帧 "
              << front.n_frames() << "\n"
              << "ASR：encoder 调用 " << mode_stats.asr_chunks << " 次 / 共 "
              << mode_stats.asr_frames << " 帧   token " << mode_stats.asr_tokens
              << " 个   本段 token " << ids.size()
              << " 个   转写段 " << asr_session.n_segments() << " 段\n"
              << "KWS：encoder 调用 " << mode_stats.kws_chunks << " 次 / 共 "
              << mode_stats.kws_frames << " 帧   触发 " << mode_stats.kws_hits
              << " 次\n"
              << "唤醒 " << mode_stats.wake_count << " 次   切回关键词模式 "
              << mode_stats.back_count << " 次   回填 "
              << mode_stats.backfilled_frames << " 帧   尾块跳过 "
              << mode_stats.tail_blocks_skipped << " 次\n"
              << "转写  : " << asr_session.transcript_path()
              << (asr_session.transcript_ok() ? "（每段一行，已 flush）"
                                              : "（写入不可用）")
              << "\n"
              << "录音  : "
              << (!record ? std::string("未启用")
                          : (record_path.empty()
                                 ? std::string("启用失败（见启动日志）")
                                 : record_path + "（" +
                                       std::to_string(
                                           static_cast<unsigned long long>(
                                               stats.recorded_bytes)) +
                                       " 字节）"))
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

    // KWS 调参参考：被阈值挡住的候选按 avg_prob 降序。排在最前的是
    // 「差一点」的，离阈值越近说明 #threshold 越该往下调（与 Python 一致）。
    if (kKwsDebug && !kws_decoder.debug_events().empty()) {
        std::vector<KwsBlockedEvent> events = kws_decoder.debug_events();
        std::sort(events.begin(), events.end(),
                  [](const KwsBlockedEvent& a, const KwsBlockedEvent& b) {
                      return a.avg_prob > b.avg_prob;
                  });
        std::cout << "\n[KWS debug] 被阈值挡住 " << events.size()
                  << " 次（avg_prob 高 → 低）：\n";
        const size_t show = std::min<size_t>(events.size(), 20);
        for (size_t i = 0; i < show; ++i) {
            std::cout << "  " << events[i].keyword << "  avg_prob="
                      << events[i].avg_prob << "  threshold="
                      << events[i].threshold << "  level=" << events[i].level
                      << "\n";
        }
        if (events.size() > show) {
            std::cout << "  ... 其余 " << events.size() - show << " 条略\n";
        }
    }

    // 与 Python 7.4 相同的两条提示：丢块说明主循环跟不上取流速度；
    // 峰值为 0 说明设备根本没采到声音。后三条是本入口独有的诊断。
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
    if (!asr_session.transcript_ok()) {
        std::cout << "提示：转写文件不可用：" << asr_session.transcript_error()
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

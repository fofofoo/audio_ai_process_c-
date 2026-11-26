/**
 * @file kws_asr_processor_node.cpp
 * @brief 语音处理节点类实现：设备取流 → 前端 → 双模式解码 → 收尾统计。
 *
 * ============================================================
 * 这份代码是干什么的
 * ============================================================
 * 从 audio_ai_process/test_mic_kws_asr.cpp 移植：把入口里第二节到第六节的
 * 流水线搬进类，按 init() / start() / processOnce() / shutdown() 四段划分：
 *
 *   init()        一、连接设备（登录 + 查编码 + 采样率校验）
 *                 二、加载两组模型、前端、语谱图与三个会话
 *   start()       三、开语音转发 + 打印配置
 *   processOnce() 四、主循环一轮（取流 → 归一化 → 前端 → 投递 → 快照 → 画窗）
 *   shutdown()    五、收尾（停流 → 残余样本 → deinit → 尾块 → join → 统计）
 *
 * ============================================================
 * 与入口版本的差异
 * ============================================================
 *   - 退出条件从「Ctrl+C 标志位 / --seconds」改为 main 里的 ros::ok()：
 *     ROS 自己接管 SIGINT，本类不再装信号处理函数。
 *   - 不再写 results/ 录音、不再有 --record / --seconds 等调试开关；
 *     转写文件路径由 main 从 launch 参数传入。
 *   - 事件与文本不在类里直接发布，而是通过 KwsAsrProcessorStep 返回给
 *     main，保持「节点类不碰 publisher」的分层。
 *   - 窗口按键改为「返回 quit_requested」，由 main 调 ros::shutdown()。
 *
 * ============================================================
 * 调用顺序（对照 main）
 * ============================================================
 *   1) 构造（只保存配置）
 *   2) init()     1.x 连接设备 → 2.x 加载模型与会话
 *   3) start()    3.1 开流  3.2 打印配置
 *   4) processOnce()  4.1 取流 → 4.2 归一化 → 4.3 前端 → 4.4 可视化
 *                     4.5 投递 → 4.6 快照 → 4.7 画窗 → 4.8 按键
 *   5) shutdown() 5.1 停流 → 5.2 残余样本 → 5.3 deinit → 5.4 尾块 + join
 *                 5.5 结算 → 5.6 最后一屏 → 5.7 统计 → 5.8 关窗
 *
 * ============================================================
 * 用法
 * ============================================================
 *   roslaunch kws_asr_processor kws_asr_processor.launch
 */

#include "kws_asr_processor/kws_asr_processor_node.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <ros/ros.h>

#include "text_utils.h"

using namespace audio_ai;

namespace {

// ==================== 一、显示画布与窗口 ====================
// 与 test_mic_kws_asr.cpp 同口径：400 列 = 4 秒滚动窗口，
// 底部 56 px 白条分两行（模式 + 文本），顶部 52 px 写设备信息。
constexpr int kSpecCols = 400;
constexpr int kFooterH = 56;
constexpr int kCalibrateFrames = 50;  // 前 0.5 秒标定色阶后锁死
constexpr int kHeaderH = 52;
const cv::Size kPlotSize(1200, 300);  // 语谱图显示尺寸（宽，高）
const char kLabel[] = "hk mic";       // 信息条标签（只能是 ASCII）
const char kWindow[] = "kws_asr_processor kws+asr";  // 窗口标题（只能是 ASCII）

// ==================== 二、KWS 解码参数 ====================
// 与 test_wav_kws.cpp / test_mic_kws_asr.cpp 一致；关键词行里的
// `:score #threshold` 会覆盖后两个默认值。
constexpr int kMaxActivePaths = 4;         // 束宽
constexpr int kNumTrailingBlanks = 1;      // 触发前至少 2 个 blank 输出帧
constexpr float kKeywordsScore = 1.0f;     // 默认 boosting score
constexpr float kKeywordsThreshold = 0.25f;  // 默认触发阈值
constexpr float kResetAfterSilence = 1.5f;   // 连续静音软重置（秒）
constexpr bool kKwsDebug = true;             // 记录被阈值挡住的候选

// ==================== 三、前端与主环节奏 ====================
constexpr int kSampleRate = 16000;   // 模型与前端只认 16 kHz
constexpr int kTailPadFrames = 100;  // 收尾补 100 帧特征 0（只喂解码器）
// int16 → float [-1,1]。直接拿 int16 当浮点会让 log 能量整体偏移
// ln(32768²) ≈ 20.8，与 Python 参考实现的量纲对齐。
constexpr double kPcmScale = 1.0 / 32768.0;
constexpr int kIdleWaitMs = 10;  // 无新数据时等待一帧时长，让出 CPU

// ==================== 四、中文字体 ====================
// 按顺序取第一个存在的文件。Noto Sans CJK 的 .ttc 里 0=JP、1=KR、2=SC、
// 3=TC、4=HK，简中取 2；Droid / 文泉驿是单字面文件，取 0。
// FreeType 不可用时整段无效，窗口文字自动回退 cv::putText（中文变 '?'）。
const std::vector<FontCandidate> kCjkFontCandidates = {
    {"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 2},
    {"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf", 0},
    {"/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0},
    {"/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 0},
};
constexpr int kCjkFontHeight = 24;      // 识别文本字号（像素）
constexpr int kCjkFontHeightSmall = 20; // 模式状态行字号（像素）
// 底部文字左右边距：可用宽度 = 画布宽 − 2 × 边距。FreeType 路径按这个
// 像素宽度从尾部截断，不按字数估算（50 汉字 × 24 px 会超出 1200 px 画布）。
constexpr int kStatusMargin = 8;
// ASCII 回退路径的上限：cv::putText 只支持 ASCII，字宽约 12 px，
// 50 个字符 ≈ 600 px，远小于画布宽度。
constexpr size_t kMaxStatusChars = 50;

}  // namespace

// ==================== 构造 / 析构 ====================

// 只保存配置；设备与模型都要等 init()。
KwsAsrProcessorNode::KwsAsrProcessorNode(const KwsAsrProcessorConfig& config)
    : config_(config) {}

// 兜底收尾：main 正常退出时已经调过 shutdown()，这里靠 running_ 去重。
KwsAsrProcessorNode::~KwsAsrProcessorNode() {
    shutdown();
}

// ==================== 一、初始化：设备 + 模型 ====================

// 连接设备并加载全部模型与前端，成功后由 start() 开流。
// 失败时返回 false 并保证设备已 deinit，调用方直接退出即可。
bool KwsAsrProcessorNode::init() {
    // —— 1. 连接设备（对应入口第二节）——
    // 1.1 填配置：pcm_sample_rate 只在设备走 PCM 时兜底；G.722.1 的采样率
    //     由编码本身固定为 16 kHz，客户端会自行纠正。
    hk_isapi::AudioCaptureConfig capture_config;
    capture_config.ip = config_.device_ip;
    capture_config.sdk_port = config_.sdk_port;
    capture_config.username = config_.username;
    capture_config.password = config_.password;
    capture_config.voice_channel = config_.voice_channel;
    capture_config.verbose = true;

    // 不主动改设备编码：set_audio_format 写的是设备持久配置，
    // 会影响该设备上所有对讲客户端，本节点只查询、不修改。
    capture_config.set_audio_format = false;

    // 1.2 输出采样率钉在 16 kHz：前端与两组模型只认 16 kHz。
    //     G.722.1 本来就是 16 kHz（直通）；PCM 时由采集库内部做
    //     32→16 抗混叠降采样，wav、取流缓冲、统计都按输出率。
    capture_config.output_sample_rate = kSampleRate;

    // 1.3 建立会话：只登录 + 查询编码，不开流、不写文件。
    //     开流放到 start()，模型加载期间设备不推流，不会产生启动丢块。
    client_ = std::make_unique<hk_isapi::HkSdkAudioClient>(capture_config);
    if (!client_->init()) {
        ROS_ERROR("[kws_asr_processor] 设备初始化失败：%s",
                  client_->last_error().c_str());
        client_.reset();
        return false;
    }

    // 1.4 采样率校验：喂非 16 kHz 不会报错，只会识别出乱码。
    //     不满足说明 output_sample_rate 与设备率无法匹配，直接拒绝启动。
    sample_rate_ = client_->sample_rate();
    if (sample_rate_ != kSampleRate) {
        ROS_ERROR("[kws_asr_processor] 输出采样率是 %d Hz，而模型固定 16 kHz；"
                  "设备实际采样率 = %d Hz", sample_rate_, client_->device_sample_rate());
        client_->deinit();
        client_.reset();
        return false;
    }

    // —— 2. 加载模型与装配 ——
    if (!loadModels()) {
        client_->deinit();
        client_.reset();
        return false;
    }
    return true;
}

// 加载 ONNX 模型、前端与可视化，装配三个会话与后台 Worker。
// 调用前 client_ 必须已经 init() 成功；失败时已创建的对象由 unique_ptr 释放。
bool KwsAsrProcessorNode::loadModels() {
    // 2.1 session 选项：多个 chunk 之间本来就串行，intra / inter 各 1 线程；
    //     日志降到 WARNING，避免加载时刷屏。
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "kws_asr_processor");
    session_options_ = std::make_unique<Ort::SessionOptions>();
    session_options_->SetIntraOpNumThreads(1);
    session_options_->SetInterOpNumThreads(1);
    session_options_->SetGraphOptimizationLevel( GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 2.2 可选 CUDA EP：注册失败只告警并回退 CPU，不中断启动。
    //     必须用 AppendExecutionProvider_CUDA；通用 AppendExecutionProvider
    //     只支持 QNN / SNPE / XNNPACK。两组模型共用这一份选项。
    cuda_status_ = "CPU";
    if (config_.use_cuda) {
        try {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            session_options_->AppendExecutionProvider_CUDA(cuda_options);
            cuda_status_ = "CUDA";
            ROS_INFO("[kws_asr_processor] CUDA EP 已注册（device_id=0）");
        } catch (const std::exception& e) {
            cuda_status_ = "CPU（CUDA 注册失败，已回退）";
            ROS_WARN("[kws_asr_processor] CUDA EP 注册失败，回退 CPU：%s",
                     e.what());
        }
    }

    // 2.3 ASR：三个 session + AsrDecoder。
    //     「模型从哪来、跑在哪块硬件上」是 main 的决定，解码类只接收
    //     现成的 session（与 Python build_ort_sessions 的分工一致）。
    asr_encoder_ = std::make_unique<Ort::Session>(
        *env_, config_.asr_encoder_path.c_str(), *session_options_);
    asr_decoder_session_ = std::make_unique<Ort::Session>(
        *env_, config_.asr_decoder_path.c_str(), *session_options_);
    asr_joiner_ = std::make_unique<Ort::Session>(
        *env_, config_.asr_joiner_path.c_str(), *session_options_);
    asr_decoder_ = std::make_unique<AsrDecoder>(
        *asr_encoder_, *asr_decoder_session_, *asr_joiner_, config_.asr_tokens_path);
    asr_info_ = asr_decoder_->info();

    // 2.4 KWS：三个 session + KwsDecoder（加载词表、唤醒词并建 trie）。
    kws_encoder_ = std::make_unique<Ort::Session>(
        *env_, config_.kws_encoder_path.c_str(), *session_options_);
    kws_decoder_session_ = std::make_unique<Ort::Session>(
        *env_, config_.kws_decoder_path.c_str(), *session_options_);
    kws_joiner_ = std::make_unique<Ort::Session>(
        *env_, config_.kws_joiner_path.c_str(), *session_options_);
    kws_decoder_ = std::make_unique<KwsDecoder>(
        *kws_encoder_, *kws_decoder_session_, *kws_joiner_,
        config_.kws_tokens_path, config_.keywords_path, kMaxActivePaths,
        kNumTrailingBlanks, kKeywordsScore, kKeywordsThreshold,
        kResetAfterSilence, kSampleRate, kKwsDebug);
    kws_info_ = kws_decoder_->info();

    // 2.5 两组模型共用同一份特征，n_mels 必须一致；不一致说明权重包配错，
    //     继续跑只会在 encoder 输入维度上抛异常，提前报清楚。
    if (kws_info_.n_mels != asr_info_.n_mels) {
        ROS_ERROR("[kws_asr_processor] KWS 的 n_mels=%d 与 ASR 的 n_mels=%d "
                  "不一致，请检查两组权重",
                  kws_info_.n_mels, asr_info_.n_mels);
        return false;
    }

    // 2.6 前端：tail_pad_frames 保持 100，收尾时补的特征 0 解码要用；
    //     那 100 帧 0 只喂解码器，不喂可视化。n_mels 跟着模型走。
    front_ = std::make_unique<AudioPreprocess>(
        kSampleRate, asr_info_.n_mels,
        25.0,   // 帧长 ms
        10.0,   // 帧移 ms
        512,    // FFT 点数
        0.97,   // 预加重
        20.0,   // Mel 下限 Hz
        0.0,    // Mel 上限：0 = Nyquist
        kTailPadFrames);

    // 2.7 语谱图：只接收特征，底部预留 56 px 白条由本类画状态文字。
    view_ = std::make_unique<SpectrogramImage>(
        asr_info_.n_mels, kSpecCols,
        static_cast<double>(front_->frame_step()) / kSampleRate,
        kDefaultLevelLo, kDefaultLevelHi, kCalibrateFrames, kPlotSize,
        kHeaderH, kFooterH, cv::COLORMAP_VIRIDIS, kLabel);

    // 2.8 中文字体：窗口模式才加载；失败只提示，drawStatusLine 回退 ASCII。
    text_renderer_ = std::make_unique<Utf8TextRenderer>();
    if (config_.show_window) {
        const FontCandidate font = find_existing_font(kCjkFontCandidates);
        const bool font_ok =
            !font.path.empty() &&
            text_renderer_->init(font.path, font.face_index);
        ROS_INFO("[kws_asr_processor] 文字字体: %s",
                 font_ok ? font.path.c_str() : "未找到，回退 ASCII");
    }

    // 2.9 三个会话 + Worker。
    //     AsrSession：全量 token、64 token 分段、转写文件落盘。
    //     KwsAsrSession：模式机、回填环、空闲计时、事件与统计。
    //     BlockWorker：唯一后台线程 + 有界队列，两个解码器只在这个线程里跑。
    //     声明顺序：解码器 → asr_session → session → worker，
    //     C++ 逆序析构保证「先停线程，再销毁 Session 与模型会话」。
    AsrSessionConfig asr_session_cfg;
    asr_session_cfg.segment_tokens = 64;
    asr_session_cfg.transcript_path = config_.transcript_path;
    asr_session_ = std::make_unique<AsrSession>(*asr_decoder_, asr_session_cfg);

    KwsAsrSessionConfig kws_asr_cfg;
    kws_asr_cfg.initial_mode =  config_.initial_asr ? KwsAsrMode::Asr : KwsAsrMode::Kws;
    kws_asr_cfg.idle_timeout_s = config_.idle_timeout_s;
    kws_asr_cfg.backfill_frames = config_.backfill_frames;
    session_ = std::make_unique<KwsAsrSession>(*asr_decoder_, *kws_decoder_,
                                               *asr_session_, kws_asr_cfg);

    // Worker 用裸指针捕获 session_：worker_ 在 session_ 之后声明，
    // 析构时先停线程，session_ 生命周期覆盖整个 Worker 运行期。
    KwsAsrSession* session_ptr = session_.get();
    worker_ = std::make_unique<BlockWorker<LogFbankFeatures>>(
        [session_ptr](const AudioFeatureInput& item) {
            session_ptr->on_block(item);
        },
        [](const LogFbankFeatures& f) {
            return f.n_frames <= 0 || f.data.empty();
        },
        50);

    return true;
}

// 开语音转发并打印运行配置。失败时释放设备会话，调用方应退出。
bool KwsAsrProcessorNode::start() {
    if (!client_ || !worker_) {
        ROS_ERROR("[kws_asr_processor] start() 调用前必须先 init()");
        return false;
    }

    // 3.1 开流：StartVoiceCom 同步阻塞约 1.2 秒，失败时 init() 建立的
    //     会话还在，必须 deinit() 后再释放。
    if (!client_->start()) {
        ROS_ERROR("[kws_asr_processor] 语音转发启动失败：%s",
                  client_->last_error().c_str());
        client_->deinit();
        client_.reset();
        return false;
    }

    start_time_ = std::chrono::steady_clock::now();
    running_ = true;

    // 3.2 打印配置（设备 / 模型 / 唤醒词 / 切换规则）。
    printConfig();

    ROS_INFO("[kws_asr_processor] 已开始监听（Ctrl+C 退出）");
    return true;
}

// ==================== 二、主循环一轮 ====================

// 取一块音频 → 算特征 → 投给后台 Session → 读快照 →（可选）画窗与按键。
// 主线程从头到尾不调用模型；模式切换与解码都在 Worker 线程里完成。
KwsAsrProcessorStep KwsAsrProcessorNode::processOnce() {
    KwsAsrProcessorStep out;
    if (!running_) {
        return out;
    }

    // 4.1 取音频：一次性取走自上次调用以来累积的全部 PCM16；
    //     没有新数据返回 false（此时缓冲已被清空）。
    //     没有数据不等于什么都不做：后台线程可能刚切了模式，
    //     快照仍要照常读取（4.6 放在 if 外面）。
    const bool has_audio = client_->get_last_audio(pcm_) && !pcm_.empty();

    // 4.2 int16 → float [-1,1]，顺手统计峰值幅度，
    //     收尾时用来提示「是不是根本没采到声音」。
    chunk_.resize(pcm_.size());
    for (size_t i = 0; i < pcm_.size(); ++i) {
        const float value = static_cast<float>(pcm_[i] * kPcmScale);
        chunk_[i] = value;
        const float amplitude = value < 0.0f ? -value : value;
        if (amplitude > peak_) {
            peak_ = amplitude;
        }
    }

    // 4.3 预处理：波形 → (n, 80) LogFBank。块长由设备回调节奏决定，
    //     不固定；前端按绝对样本索引凑帧，出帧结果与块长无关。
    const LogFbankFeatures feats = front_->accept_waveform(chunk_);

    // 4.4 可视化：有新帧才重建画布（设备约 40 ms 一批，通常 4 帧）。
    if (config_.show_window && has_audio && feats.n_frames > 0) {
        canvas_ = view_->accept_features(feats);
    }

    // 4.5 投递：只把特征副本交给后台线程，主线程不等推理。
    //     本次 0 帧的纯空块由 push() 内部丢掉，不占队列、不唤醒线程。
    if (has_audio) {
        worker_->push(feats);
    }

    // 4.6 读快照：事件（唤醒 / 结束语 / 超时）只返回一次；
    //     文本只在变化时返回，main 收到后才发布话题。
    //     异步解码下文字会比语谱图滞后若干块，这是把推理挪出主线程的
    //     必然代价。
    snapshot_ = session_->snapshot();
    if (snapshot_.event_seq != shown_event_seq_) {
        shown_event_seq_ = snapshot_.event_seq;
        out.has_event = true;
        out.event = snapshot_.last_event;
        std::cout << "\n" << snapshot_.last_event << std::endl;
    }
    if (snapshot_.text != shown_) {
        shown_ = snapshot_.text;
        out.has_text = true;
        out.text = snapshot_.text;
    }

    // 4.7 画窗：每轮都画、都 imshow。部分窗口环境不保证自动重绘，
    //     跳过会表现为老字停在窗口里不动。
    if (config_.show_window) {
        drawFrame();
    }

    // 4.8 按键与节拍：有数据时按 1 ms 泵事件，保证 q/Esc 及时响应；
    //     没有数据时用 10 ms，窗口不假死；无窗时直接 sleep 10 ms 让出 CPU。
    if (config_.show_window) {
        const int key = cv::waitKey(has_audio ? 1 : kIdleWaitMs) & 0xFF;
        if (key == 27 || key == 'q') {
            out.quit_requested = true;
        } else if (key == ' ') {
            togglePause();
        }
    } else if (!has_audio) {
        std::this_thread::sleep_for(std::chrono::milliseconds(kIdleWaitMs));
    }

    return out;
}

// ==================== 三、收尾 ====================

// 停流 → 取完残余样本 → deinit → 尾块 + join → 结算 → 统计。
// 可重复调用：第二次起直接返回。
void KwsAsrProcessorNode::shutdown() {
    if (!running_ || !client_ || !worker_) {
        return;
    }
    running_ = false;  // 先置位，避免析构时再次进入

    // 5.1 关闭语音转发。
    client_->stop();

    // 5.2 关流后缓冲区里可能还压着几块样本，先取走解完，否则最后一段话
    //     会因为没进前端而整段丢失。顺序不能反：deinit() 会清空取流缓冲。
    std::vector<int16_t> rest;
    if (client_->get_last_audio(rest) && !rest.empty()) {
        chunk_.resize(rest.size());
        for (size_t i = 0; i < rest.size(); ++i) {
            const float value = static_cast<float>(rest[i] * kPcmScale);
            chunk_[i] = value;
            const float amplitude = value < 0.0f ? -value : value;
            if (amplitude > peak_) {
                peak_ = amplitude;
            }
        }
        const LogFbankFeatures feats = front_->accept_waveform(chunk_);
        if (config_.show_window && feats.n_frames > 0) {
            canvas_ = view_->accept_features(feats);
        }
        worker_->push(feats);
    }

    // 5.3 残余样本取完后释放设备会话：登出并 NET_DVR_Cleanup。
    //     放这里而不是程序末尾，是为了尽早释放设备占用。
    client_->deinit();

    // 5.4 尾块 + join：前端补出尾部帧，再在特征尾补 100 帧 0
    //     （log 域能量 1，不是静音值），使最后不满一个 chunk 的尾段也能解码。
    //     尾块带 is_tail：KwsAsrSession 在 KWS 模式下直接跳过（特征域 0
    //     对束搜索没有意义），ASR 模式下正常解码。
    //     push_final 阻塞等空位、绝不丢弃；join 返回时队列已解空，
    //     主线程才能安全读最终文本与统计。
    worker_->push_final(front_->input_finished());
    worker_->finish_and_join();

    // 5.5 结算：尾段落盘并返回最近一次 ASR 文本（Worker 已退出，无并发）。
    const std::string final_text = session_->finish();
    const double wall_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      start_time_)
            .count();

    // 5.6 最后一屏：画布已随 4.4 / 5.2 更新到最新；一轮都没贴过时
    //     （极短会话）才用 view_->frame() 按现有缓存补一张。
    //     不喂补 0 特征，避免多出假亮尾。
    if (config_.show_window) {
        if (canvas_.empty()) {
            canvas_ = view_->frame();
        }
        if (!canvas_.empty()) {
            canvas_(cv::Rect(0, canvas_.rows - kFooterH + 1, canvas_.cols,
                             kFooterH - 1))
                .setTo(cv::Scalar(255, 255, 255));
            std::ostringstream status;
            status << mode_name(snapshot_.mode) << " | stopped   唤醒 "
                   << snapshot_.wake_count << " / 切回 " << snapshot_.back_count;
            drawStatusLine(canvas_, status.str(), 34, kCjkFontHeightSmall);
            drawStatusLine(canvas_, final_text, 10, kCjkFontHeight);
            cv::imshow(kWindow, canvas_);
        }
    }

    // 5.7 统计（口径与入口 6.5 一致）。
    printStats(final_text, wall_seconds);

    // 5.8 关窗。ROS 节点没有「等按键看最后一屏」的交互需求，直接关闭。
    if (config_.show_window) {
        cv::destroyAllWindows();
    }
}

// ==================== 四、窗口与按键 ====================

// 空格切换「暂停 / 恢复语音转发」：底层就是 client.stop() / client.start()，
// 只停/起对讲线路，不重登录、不重建编码器；恢复后设备约 1.2 秒才送首包。
void KwsAsrProcessorNode::togglePause() {
    if (!paused_) {
        if (client_->stop()) {
            paused_ = true;
            ROS_INFO("[kws_asr_processor] 已暂停语音转发，按空格恢复");
        } else {
            ROS_WARN("[kws_asr_processor] 暂停失败：%s",
                     client_->last_error().c_str());
        }
    } else {
        ROS_INFO("[kws_asr_processor] 正在恢复语音转发（首包约 1.2 秒后到达）...");
        if (client_->start()) {
            paused_ = false;
        } else {
            ROS_WARN("[kws_asr_processor] 恢复失败：%s",
                     client_->last_error().c_str());
        }
    }
}

// 每轮重画底部白条与两行状态，再 imshow。
void KwsAsrProcessorNode::drawFrame() {
    if (canvas_.empty()) {
        return;
    }
    // 画字前必须先刷白底条：draw_tail 底层是 alpha 混合，同一串字重复画
    // 在同一像素上会让边缘逐次变深，画布又不一定每轮重建，表现为文字
    // 一深一浅地闪。从白条第 2 行开始刷，保留首行黑分隔线。
    canvas_(cv::Rect(0, canvas_.rows - kFooterH + 1, canvas_.cols, kFooterH - 1))
        .setTo(cv::Scalar(255, 255, 255));
    // 第一行：模式 + 计数（小字）。
    std::ostringstream status;
    status << mode_name(snapshot_.mode) << " | listening"
           << "   唤醒 " << snapshot_.wake_count << " / 切回 "
           << snapshot_.back_count;
    drawStatusLine(canvas_, status.str(), 34, kCjkFontHeightSmall);
    // 第二行：识别文本（大字）；KWS 模式为空，只留白。
    const std::string line2 =
        paused_ ? snapshot_.text + "  [已暂停]" : snapshot_.text;
    drawStatusLine(canvas_, line2, 10, kCjkFontHeight);
    cv::imshow(kWindow, canvas_);
}

// 底部状态文字：优先 FreeType 渲染 UTF-8 中文，失败回退 ASCII putText。
void KwsAsrProcessorNode::drawStatusLine(cv::Mat& frame,
                                          const std::string& text,
                                          int baseline, int font_height) {
    const cv::Point org(kStatusMargin, frame.rows - baseline);
    if (text_renderer_ && text_renderer_->ready()) {
        // 画布宽度可能随窗口缩放变化，每帧按实际列数现算可用宽度。
        text_renderer_->draw_tail(frame, text, org, font_height,
                                  frame.cols - 2 * kStatusMargin);
        return;
    }
    // 没有 FreeType 时只能回退 putText（中文变 '?'），按字数上限截。
    const std::string shown = utf8_tail(text, kMaxStatusChars);
    cv::putText(frame, shown, org, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

// ==================== 五、配置与统计打印 ====================

// 打印设备、模型、唤醒词与切换规则，格式与入口第四节一致。
void KwsAsrProcessorNode::printConfig() {
    std::cout << std::string(74, '=') << "\n"
              << "kws_asr_processor —— KWS 唤醒 -> ASR 识别"
              << (config_.show_window ? " + 语谱图" : "") << "\n"
              << std::string(74, '=') << "\n"
              << "设备  : " << config_.device_ip << ":" << config_.sdk_port
              << "  channel=" << config_.voice_channel << "\n"
              << "编码  : " << client_->codec_name() << "  "
              << client_->device_sample_rate() << " Hz"
              << (client_->device_sample_rate() == sample_rate_
                      ? std::string()
                      : " -> 降采样 " + std::to_string(sample_rate_) + " Hz")
              << "\n"
              << "ASR   : " << config_.asr_encoder_path << "\n"
              << "KWS   : " << config_.kws_encoder_path << "\n"
              << "执行器: " << cuda_status_ << "\n"
              << "ASR 模型: T=" << asr_info_.T
              << "  decode_chunk_len=" << asr_info_.decode_chunk_len
              << "  context_size=" << asr_info_.context_size
              << "  vocab=" << asr_info_.vocab_size
              << "  状态张量=" << asr_info_.n_states
              << "  n_mels=" << asr_info_.n_mels
              << "  blank_id=" << asr_info_.blank_id << "\n"
              << "KWS 模型: T=" << kws_info_.T
              << "  decode_chunk_len=" << kws_info_.decode_chunk_len
              << "  context_size=" << kws_info_.context_size
              << "  vocab=" << kws_info_.vocab_size
              << "  状态张量=" << kws_info_.n_states
              << "  帧长=" << kws_info_.frame_shift_s * 1e3 << "ms\n"
              << "前端  : 帧长 " << front_->frame_size() << " 点("
              << front_->frame_size() * 1000.0 / front_->sample_rate()
              << "ms)  帧移 " << front_->frame_step() << " 点("
              << front_->frame_step() * 1000.0 / front_->sample_rate()
              << "ms)\n"
              << "唤醒词: " << kws_info_.n_keywords << " 条  ";
    for (size_t i = 0; i < kws_decoder_->keywords().size(); ++i) {
        std::cout << (i == 0 ? "" : " / ") << kws_decoder_->keywords()[i];
    }
    std::cout << "\n"
              << "切换  : 初始="
              << (config_.initial_asr ? "ASR" : "KWS")
              << "  空闲超时=" << config_.idle_timeout_s << "s"
              << "  回填=" << config_.backfill_frames << "帧\n"
              << "转写  : " << config_.transcript_path << "\n"
              << "画布  : " << kSpecCols << " 列 x " << asr_info_.n_mels
              << " 行 = "
              << kSpecCols * front_->frame_step() / front_->sample_rate()
              << " 秒窗口\n"
              << std::string(74, '=') << "\n";
}

// 收尾统计：口径与入口 6.5 一致。调用时 Worker 已 join，读解码统计无并发。
void KwsAsrProcessorNode::printStats(const std::string& final_text,
                                      double wall_seconds) {
    const hk_isapi::AudioCaptureStats stats = client_->get_stats();
    const std::vector<int64_t>& ids = asr_session_->all_token_ids();
    const KwsAsrStats mode_stats = session_->stats();

    std::cout << "\n\n" << std::string(74, '=') << "\n"
              << "识别结果：" << final_text << "\n"
              << std::string(74, '=') << "\n"
              << "时长 " << wall_seconds << "s   实际转发 "
              << stats.forward_seconds << "s   喂入样本 " << front_->n_samples()
              << "   产出帧 " << front_->n_frames() << "\n"
              << "ASR：encoder 调用 " << mode_stats.asr_chunks << " 次 / 共 "
              << mode_stats.asr_frames << " 帧   token " << mode_stats.asr_tokens
              << " 个   本段 token " << ids.size()
              << " 个   转写段 " << asr_session_->n_segments() << " 段\n"
              << "KWS：encoder 调用 " << mode_stats.kws_chunks << " 次 / 共 "
              << mode_stats.kws_frames << " 帧   触发 " << mode_stats.kws_hits
              << " 次\n"
              << "唤醒 " << mode_stats.wake_count << " 次   切回关键词模式 "
              << mode_stats.back_count << " 次   回填 "
              << mode_stats.backfilled_frames << " 帧   尾块跳过 "
              << mode_stats.tail_blocks_skipped << " 次\n"
              << "转写  : " << asr_session_->transcript_path()
              << (asr_session_->transcript_ok() ? "（每段一行，已 flush）"
                                                : "（写入不可用）")
              << "\n"
              << "取流丢弃 "
              << static_cast<unsigned long long>(stats.stream_dropped_samples)
              << "   解码丢弃 " << worker_->dropped() << " 块（队列上限 "
              << worker_->max_pending() << "）   峰值幅度 " << peak_ << "\n"
              << "上行保活 发送 "
              << static_cast<unsigned long long>(stats.uplink_sent_frames)
              << " 帧 / 失败 "
              << static_cast<unsigned long long>(stats.uplink_send_errors)
              << " 次\n"
              << "静音补零 "
              << static_cast<unsigned long long>(stats.gap_padded_samples)
              << " 样本 / " << stats.gap_events << " 次\n";

    // KWS 调参参考：被阈值挡住的候选按 avg_prob 降序，排最前的是
    // 「差一点」的，离阈值越近说明 #threshold 越该往下调。
    if (kKwsDebug && !kws_decoder_->debug_events().empty()) {
        std::vector<KwsBlockedEvent> events = kws_decoder_->debug_events();
        std::sort(events.begin(), events.end(),
                  [](const KwsBlockedEvent& a, const KwsBlockedEvent& b) {
                      return a.avg_prob > b.avg_prob;
                  });
        std::cout << "\n[KWS debug] 被阈值挡住 " << events.size()
                  << " 次（avg_prob 高 → 低）：\n";
        const size_t show = std::min<size_t>(events.size(), 20);
        for (size_t i = 0; i < show; ++i) {
            std::cout << "  " << events[i].keyword
                      << "  avg_prob=" << events[i].avg_prob
                      << "  threshold=" << events[i].threshold
                      << "  level=" << events[i].level << "\n";
        }
        if (events.size() > show) {
            std::cout << "  ... 其余 " << events.size() - show << " 条略\n";
        }
    }

    // 诊断提示：丢块说明主循环或后台解码跟不上；峰值为 0 说明没采到声音。
    if (stats.stream_dropped_samples > 0) {
        std::cout << "提示：出现丢块，说明主循环跟不上采集速度（取流缓冲上限 "
                     "40960 样本 ≈ 2.56 s）。\n";
    }
    if (worker_->dropped() > 0) {
        std::cout << "提示：后台解码跟不上采集，已丢弃最早的 "
                  << worker_->dropped()
                  << " 块特征（显示文字会跳）；可启用 use_cuda 或减少其它负载。\n";
    }
    // Handler 的异常以 exception_ptr 保存，这里展开成人能读的原因。
    const std::exception_ptr decode_error = worker_->error();
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
    if (!asr_session_->transcript_ok()) {
        std::cout << "提示：转写文件不可用：" << asr_session_->transcript_error()
                  << "（识别本身不受影响）\n";
    }
    if (peak_ == 0.0) {
        std::cout << "提示：峰值幅度为 0，设备可能没采到声音，"
                     "请确认拾音器与语音通道配置。\n";
    }
}

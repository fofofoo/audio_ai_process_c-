/**
 * @file hk_sdk_audio_client.cpp
 * @brief HkSdkAudioClient 实现：设备端拾音器 → 编码音频 → PCM16 → wav。
 *
 * 与头文件的分工：职责边界、公共接口的完整 Doxygen 都写在 .h，本文件只补充
 * 实现层面的东西——线程模型、加锁范围、切帧规则、wav 收尾的边界条件。
 *
 * 线程模型：
 *   - init() / start() / stop() / deinit() / get_stats() / output_path()
 *     由调用线程（main）执行；
 *   - on_voice_data() 运行在 SDK 内部回调线程，会被连续调用，且不保证与
 *     主线程错开。所有对 wav 文件、m_encoded_buf、m_stats、m_data_bytes 的
 *     读写都在 m_mutex 内完成，回调里只做「追加 → 切帧 → 解码 → fwrite」，
 *     不做重活、不等待主线程，避免拖慢 SDK 回调导致丢包。
 *
 * 会话顺序（init() → start() → stop() → deinit()）：
 *   init()  ：init_sdk → login_device → query_audio_format → InitG722Decoder
 *   start() ：InitG722Decoder（重建） → StartVoiceCom_MR_V30 → 上行保活
 *   stop()  ：停保活 → StopVoiceCom → 收尾 wav → 残包计数
 *   deinit()：stop → 恢复设备编码 → 登出 → 释放句柄 → NET_DVR_Cleanup
 * init()/start() 都不碰文件；写盘由 start_record(path) 打开 wav、
 * stop_record() 收尾。任意一步失败都直接返回 false，已申请的资源留给
 * deinit() 兜底，不做中途回滚（重复调用 init() / start() 是幂等的）。
 *
 * 约束提醒：本类沿用实例级 NET_DVR_Init/Cleanup，与可见光/热成像客户端
 * 混跑时先析构的一才会把 SDK 清掉，见 .h 文件约束 1。
 */

#include "hk_sdk_audio_client.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <system_error>

namespace hk_isapi {

namespace {

// —— G.722.1 解码缓冲 ——
// 一帧编码 80 B 对应 40 ms，解码后是 640 个 PCM16 采样点 = 1280 字节（16 kHz 单声道）。
// 这里按 4 帧（160 ms）预留，给「一次回调里塞了多帧」的固件留余量；
// 实际写出长度仍以 SDK 回填的 out_frame_size 为准，不能假设它一定等于 1280。

constexpr size_t kPcmBytesPerFrame = 1280;
constexpr size_t kPcmBufferSize = kPcmBytesPerFrame * 4;

// wav 头固定 44 字节（PCM 且不带扩展块）：open_wav 先写占位，
// close_wav 再回填 RIFF 与 data 两个长度字段。
constexpr size_t kWavHeaderSize = 44;

// 停止后再次起流的重试节奏：实测 StopVoiceCom 后约 200 ms 内设备会拒绝
// 重新起流（error 29 NET_DVR_DVROPRATEFAILED），间隔 200 ms 起流稳定；
// 5 次最多覆盖约 1 秒，足够吸收设备侧释放会话的抖动。
constexpr int kResumeRetryCount = 5;
constexpr int kResumeRetryMs = 200;

}  // namespace

// ==================== 构造与状态访问 ====================

// 构造：只保存配置并预分配解码输出缓冲，不碰 SDK、不连设备。
// 网络与设备资源一律等到 init() 再申请，方便构造后仍可改配置再启动。
// m_pcm_buf 在这里一次性分配好，回调线程里不再扩容（否则要在锁内分配内存）。
HkSdkAudioClient::HkSdkAudioClient(const AudioCaptureConfig& config)
    : m_config(config) {
    m_pcm_buf.resize(kPcmBufferSize);
}

// 默认构造：使用 AudioCaptureConfig 的默认值（演示设备地址/端口）。
HkSdkAudioClient::HkSdkAudioClient() : HkSdkAudioClient(AudioCaptureConfig{}) {}

// 析构：与 deinit() 完全等价，作为调用方忘记显式释放时的兜底。
// deinit() 是幂等的，重复调用只剩空操作。
HkSdkAudioClient::~HkSdkAudioClient() {
    deinit();
}

// 初始化会话：SDK 初始化 → 登录 → 可选设置编码 → 查询编码 → 建降采样器与
// 解码器。只确认设备可用，不调用 StartVoiceCom、不产生音频、不打开 wav。
// 已初始化时直接返回 true（可安全重复调用）。
// 失败时返回 false，原因写进 last_error；已登录/已初始化的资源不在这里回收，
// 由 deinit() 统一清理，下次 init() 会复用它们。
bool HkSdkAudioClient::init() {
    if (m_session_ready.load()) {
        return true;
    }

    // 零、上一轮统计与缓冲清零，避免重复 init() 时统计累加。
    m_last_error.clear();
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stats = AudioCaptureStats{};                     // 统计清零（包数/字节数/解码帧/错误/墙钟）
        m_encoded_buf.clear();                             // 清空编码缓冲，残留半帧会切错帧边界
        m_stream_buf.clear();                              // 清空取流缓冲，避免上一会话的样本混进新流
        m_data_bytes = 0;                                  // 已写入的 PCM 字节数，用于回填 wav 头长度
        m_codec = AudioCodec::kG7221;                      // 设备编码默认值，稍后由设备实际值覆盖
        m_device_sample_rate = m_config.pcm_sample_rate;   // 设备采样率兜底，稍后以设备实际值为准
        m_output_sample_rate = m_config.pcm_sample_rate;   // 输出采样率，setup_output_resampler 再校正
        m_resampler.reset();                               // 新会话不继承上一段的重采样历史
        m_recording = false;                               // 新会话从「不录制」开始，需显式 start_record()
    }

    // 一、初始化sdk，和 SDK/设备建立关系
    // 以下四步任一失败都直接返回：init 失败 → 无法 login；login 失败 →
    // 后续 NET_DVR_* 全部无意义；格式不支持 → 写了 wav 也会标错采样率。
    // 1.初始化 SDK，init_sdk() — 无需输入
    if (!m_sdk_initialized && !init_sdk()) {
        return false;
    }
    // 2.登录设备，login_device() — 输入设备ip/端口号/用户名/密码（成员变量，不显式输入）
    if (m_user_id < 0 && !login_device()) {
        return false;
    }
    // 2.5 可选：按配置主动设置对讲编码。必须在查询之前，这样第 3 步读到的
    //     就是设置后实际生效的值；apply 内部会先快照原配置，供 deinit() 恢复。
    if (m_config.set_audio_format && !apply_audio_format()) {
        return false;
    }
    // 3. 查询设备音频编码，query_audio_format()
    if (!query_audio_format()) {
        // 设置过编码但后续不可用（例如设备拒绝、查询失败）：先恢复原配置，
        // 否则下次 init() 会把「已经改过的值」当成原值快照。
        restore_audio_format();
        return false;
    }
    // 3.5 可选：按配置建立降采样器。必须在解码器/起流之前完成，
    //     否则首包到达时 wav 头与输出采样率还没确定，会写出错头。
    if (!setup_output_resampler()) {
        restore_audio_format();
        return false;
    }

    // 二、初始化hk_sdk的 G.722.1 解码器
    // 按实例复用：重复 init() 时不再重建，deinit() 里统一释放。即便当前编码是 PCM，也照样创建——万一运行中设备切换编码，
    if (m_decoder == nullptr) {
        m_decoder = NET_DVR_InitG722Decoder();
        if (m_decoder == nullptr) {
            set_error("NET_DVR_InitG722Decoder failed, error=" +
                      std::to_string(NET_DVR_GetLastError()));
            restore_audio_format();
            return false;
        }
    }

    // 三、不打开录音文件
    // start() 只负责把语音转发拉起来；文件由 start_record(path) 打开，
    // 因此首包到达时若尚未开始录制，回调只会解码后丢弃（见 on_voice_data）。

    // 会话层资源都已就绪：后续 start()/stop() 只动对讲线路，不再重跑这些步骤。
    m_session_ready.store(true);
    m_start_time = std::chrono::steady_clock::now();

    // 四、刻意不调用 StartVoiceCom：init() 只确认设备可用，真正拉流由 start()
    //     在调用方准备好消费音频（例如模型加载完成）之后发起。这样设备不会
    //     在上层还没就绪时一直推流，把取流缓冲塞满并产生无谓的丢样本。
    if (m_config.verbose) {
        std::cout << "[Audio] session ready, channel=" << m_config.voice_channel
                  << ", codec=" << codec_name() << std::endl;
    }
    return true;
}

// 启动语音转发：重建解码器 → StartVoiceCom（含 200 ms × 5 重试）→ 起保活。
// 前置条件：init() 已成功（m_session_ready）。
// 返回 true 只代表设备接受了起流请求；首个音频回调约 1.2 秒后到达。
bool HkSdkAudioClient::start() {
    if (m_running.load()) {
        return true;
    }
    if (!m_session_ready.load()) {
        set_error("start(): init() has not succeeded");
        return false;
    }

    // 停止期间设备可能重新开始编码，旧解码器的自适应预测状态可能与码流对不上；
    // 重建一次成本很低，换来确定性。此刻没有回调（线路已停），可安全释放。
    if (m_decoder != nullptr) {
        NET_DVR_ReleaseG722Decoder(m_decoder);
        m_decoder = nullptr;
    }
    m_decoder = NET_DVR_InitG722Decoder();
    if (m_decoder == nullptr) {
        set_error("NET_DVR_InitG722Decoder failed, error=" +
                  std::to_string(NET_DVR_GetLastError()));
        return false;
    }

    // 起流失败（error 29）的重试在 start_voice_forward() 内部完成，
    // 这里失败说明 5 次都没成功，last_error 已是最后一次的错误码。
    if (!start_voice_forward()) {
        return false;
    }

    if (m_config.verbose) {
        std::cout << "[Audio] voice com started, channel=" << m_config.voice_channel
                  << ", recording=off" << std::endl;
    }
    return true;
}

// 起流并启动上行保活：由 start() 调用。
// 前置条件：SDK 已初始化、已登录、编码已确认、解码器已建。
// 失败时撤销运行标志并写 last_error，句柄保持 -1；成功时开始累计转发时长。
bool HkSdkAudioClient::start_voice_forward() {
    // 新会话重置静音补零时间轴：暂停/重开之间的空档是调用方主动造成的，
    // 不是设备静音，不能补成 0（否则恢复后的 wav 会被灌入一大段静音）。
    // 必须早于 StartVoiceCom：SDK 可能在本函数返回前就触发第一个回调。
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        reset_silence_gap();
    }

    // 启动设备端语音对讲（MR = Multi-Route 多路语音，V30 为版本号）。
    // 它向固件申请开一条语音对讲会话：已登录用户 X 请求在通道 1 上做语音对讲，
    // 让设备把一直在采集、编码的音频推给本客户端；它既不启动采集，也不修改
    // 设备配置，只是让 SDK 把「设备编码后的音频」异步推给 voice_callback。
    // 编码格式、采样率、码率、时长、文件格式由另外的接口在固件侧配置。
    // 返回句柄 < 0 表示失败，成功时该句柄仅供 NET_DVR_StopVoiceCom() 使用。
    //
    // 需要重试：设备在上一会话刚结束时（StopVoiceCom 后约 200 ms 内，或上一个
    // 进程刚退出）会返回 error 29 NET_DVR_DVROPRATEFAILED。实测间隔 200 ms
    // 再起就能成功，所以这里只对 29 重试；网络、权限等其它错误立即返回，
    // 避免无谓等待。
    DWORD last_error_code = 0;
    for (int attempt = 0; attempt < kResumeRetryCount; ++attempt) {
        // 置位运行标志必须早于 StartVoiceCom：SDK 可能在本函数返回前就异步
        // 触发首个回调，on_voice_data 里会检查该标志，否则首包会被丢掉。
        m_running.store(true);
        m_voice_handle = NET_DVR_StartVoiceCom_MR_V30(
            m_user_id,                                  // 登录设备时拿到的用户 ID
            static_cast<DWORD>(m_config.voice_channel), // 语音通道号，本机实测为 1
            &HkSdkAudioClient::voice_callback,          // C 风格回调，SDK 用它把音频推回来
            this                                        // 回调时原样回传，用来还原实例
        );
        if (m_voice_handle >= 0) {
            break;
        }
        last_error_code = NET_DVR_GetLastError();
        m_running.store(false);
        // 只有「设备操作失败」值得等一个节拍再试；最后一次失败也不再等。
        if (last_error_code != NET_DVR_DVROPRATEFAILED ||
            attempt + 1 == kResumeRetryCount) {
            break;
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(kResumeRetryMs));
    }
    if (m_voice_handle < 0) {
        set_error("NET_DVR_StartVoiceCom_MR_V30 failed, error=" +
                  std::to_string(last_error_code));
        return false;
    }

    m_last_error.clear();
    m_forward_start = std::chrono::steady_clock::now();

    // 上行保活：手册要求语音转发双向（除了收设备音频，也要用
    // NET_DVR_VoiceComSendData 发数据回去）。实测只收不发时，设备在约
    // 10.5 秒后停止回调，窗口和识别随之冻结。两种下行编码都回发
    // 80 B G.722 静音——PCM 下裸 PCM 静音会让设备直接停流，原因见
    // init_uplink_keepalive() 的说明。
    if (m_config.uplink_keepalive) {
        if (init_uplink_keepalive()) {
            start_uplink_keepalive();
            if (m_config.verbose) {
                std::cout << "[Audio] uplink keepalive: 每 40 ms 回发一帧 "
                             "G.722 静音（"
                          << m_silence_frame.size() << " B，PCM 下行同样适用）"
                          << std::endl;
            }
        } else if (m_config.verbose) {
            std::cout << "[Audio] uplink keepalive 初始化失败，设备可能在约 10 "
                         "秒后停止语音转发"
                      << std::endl;
        }
    }
    return true;
}

// 把本段转发时长累加进统计：调用方必须已持有 m_mutex。
// 只在真正转发过时累加，避免重复调用把时长叠加多次。
void HkSdkAudioClient::add_forward_seconds_locked() {
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - m_forward_start).count();
    if (seconds > 0.0) {
        m_stats.forward_seconds += seconds;
    }
}

// 开启录制：在已启动的语音转发上打开 wav，并置位录制标志。
// 前置条件：start() 已成功（m_running），且当前未在录制。
// 文件打开放在锁内，保证回调线程要么看到「未录制 + 无文件」，要么看到
// 「已录制 + 文件已就绪」，不会出现写了半截才打开文件的情况。
// 每段录制独立：recorded_bytes 在这里清零，便于调用方按段统计时长。
bool HkSdkAudioClient::start_record(const std::string& wav_path) {
    if (!m_running.load()) {
        set_error("start_record failed: voice com not started");
        return false;
    }
    if (wav_path.empty()) {
        set_error("start_record failed: empty wav path");
        return false;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_recording.load()) {
        set_error("start_record failed: already recording");
        return false;
    }

    // open_wav 内部会解析绝对路径、建目录、写 44 字节占位头；
    // 失败时 m_wav 仍为 nullptr，录制标志保持 false。
    if (!open_wav(wav_path)) {
        return false;
    }

    m_stats.recorded_bytes = 0;
    m_recording.store(true);

    if (m_config.verbose) {
        std::cout << "[Audio] recording started, output=" << m_final_path << std::endl;
    }
    return true;
}

// 结束录制：先落旗再关文件。
// 顺序不能反——回调线程在 m_mutex 内先检查 m_recording 再写文件，
// 先置 false 能让后续回调跳过写入；已进入写入的回调持锁期间本函数等待，
// 因此 close_wav 时不可能还有 fwrite 在飞。
// 语音转发（m_running / m_voice_handle）保持不变，可再次 start_record()。
void HkSdkAudioClient::stop_record() {
    m_recording.store(false);

    std::string finished_path;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_wav == nullptr) {
            return;
        }
        finished_path = m_final_path;
        close_wav();
    }

    if (m_config.verbose) {
        std::cout << "[Audio] recording stopped, output=" << finished_path << std::endl;
    }
}

// 停止语音转发：先停流（阻断新回调），再收尾录制与残包。
// 顺序不能反——必须先 StopVoiceCom 保证回调线程不再进入写文件路径，
// 否则 fclose 之后仍可能有回调触发 fwrite（悬垂 FILE*）。
// 只拆对讲线路，保留 SDK/登录/解码器/取流缓冲，之后可以再次 start()。
// 可重复调用；未在转发时直接返回 true（幂等）。
bool HkSdkAudioClient::stop() {
    // exchange 保证并发调用时只有一方真正执行拆除；已停止则幂等成功。
    if (!m_running.exchange(false)) {
        return true;
    }

    // 先停上行保活线程，再关语音转发：反过来会让线程往正在销毁的句柄发数据。
    stop_uplink_keepalive();

    // 让设备关掉这条语音对讲线路，不再转发回调
    if (m_voice_handle >= 0) {
        NET_DVR_StopVoiceCom(m_voice_handle);
        m_voice_handle = -1;
    }
    m_voice_handle_atomic.store(-1);

    // 录制中停止时先收尾文件：此时回调已停止，可安全 fclose + 改名。
    // 再次 start() 后需要重新 start_record()。
    stop_record();

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        add_forward_seconds_locked();

        // 线路断开时缓冲里可能留着半帧，留着会污染下次起流后的第一帧。
        if (!m_encoded_buf.empty()) {
            m_stats.dropped_partial_bytes += m_encoded_buf.size();
            m_encoded_buf.clear();
        }

        // 本段转发结束，清掉时间轴状态；下次 start() 会从首个回调重新起表。
        reset_silence_gap();
    }

    if (m_config.verbose) {
        std::cout << "[Audio] audio forwarding stopped" << std::endl;
    }
    return true;
}

// 结束会话：先 stop()，再恢复设备编码、结算墙钟、登出并释放 SDK 资源。
// 与析构函数等价、幂等；调用方想「先读统计再释放」时显式调用它。
void HkSdkAudioClient::deinit() {
    // 一、还在转发就先停：停保活 → 停流 → 收尾 wav → 结算本段转发时长。
    stop();

    // 二、结算会话墙钟并清空取流缓冲：此时回调已停，可以安全清空。
    if (m_session_ready.exchange(false)) {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto now = std::chrono::steady_clock::now();
        m_stats.wall_seconds =
            std::chrono::duration<double>(now - m_start_time).count();
        m_stream_buf.clear();
    }

    // 三、恢复设备编码：本次 init() 主动改过才写回。必须在登出之前，
    //     恢复请求要用当前登录会话发出。
    if (m_config.restore_audio_format) {
        restore_audio_format();
    } else {
        m_audio_format_saved = false;
        m_audio_format_changed = false;
    }

    // 四、按反序释放句柄：登出 → G722 解码器 → G722 编码器 → NET_DVR_Cleanup。
    if (m_user_id >= 0) {
        NET_DVR_Logout(m_user_id);
        m_user_id = -1;
    }
    if (m_decoder != nullptr) {
        NET_DVR_ReleaseG722Decoder(m_decoder);
        m_decoder = nullptr;
    }
    // 保活用的 G.722 编码器与解码器一样按实例复用，这里统一释放。
    if (m_encoder != nullptr) {
        NET_DVR_ReleaseG722Encoder(m_encoder);
        m_encoder = nullptr;
    }
    if (m_sdk_initialized) {
        NET_DVR_Cleanup();
        m_sdk_initialized = false;
    }
}

// ==================== 上行保活 ====================

// 准备上行保活帧：用 SDK 的 G.722 编码器把 640 点静音编成 80 B。
//
// 为什么 PCM 下行也用 G.722 静音（实测结论，不是想当然）：
//   - 下行切到 PCM 后，回发裸 PCM 静音（20 ms 1280 B 和 40 ms 2560 B 都试过）
//     会让设备在收到第一帧后立即停止回调，packets 停在 0~3；
//   - 同一次会话里改回 G.722 80 B 静音，PCM 下行持续正常（25 秒实测
//     471 个回调包、约 21.7 秒音频），说明上行对讲编码没有跟着下行切换。
//   - 手册《NET_DVR_COMPRESSION_AUDIO》明确写「配置后需要重启设备才生效」，
//     而本客户端是运行期 SetDVRConfig、不重启设备，所以上行仍是 G.722.1。
//   如果以后设备重启后真把上行也切成 PCM，可关掉 uplink_keepalive 或在此
//   按 m_codec 分支回发裸 PCM；在此之前固定 G.722 是唯一验证通过的做法。
//
// 前置条件：语音转发已启动。编码器按实例复用（重复 start 不重建）；
// 静音帧只在这里写一次，保活线程只读，因此两者不需要额外加锁。
bool HkSdkAudioClient::init_uplink_keepalive() {
    if (m_encoder == nullptr) {
        NET_DVR_AUDIOENC_INFO enc_info{};
        enc_info.in_frame_size =
            static_cast<DWORD>(kPcmBytesPerFrame);  // 640 点 PCM16 = 1280 B
        m_encoder = NET_DVR_InitG722Encoder(&enc_info);
        if (m_encoder == nullptr) {
            set_error("NET_DVR_InitG722Encoder failed, error=" +
                      std::to_string(NET_DVR_GetLastError()));
            return false;
        }
    }

    // 设备只校验「上行数据与当前编码一致」，内容可以是静音：把 640 个 0
    // 采样点编成一帧 G.722（默认 80 B）。长度取配置里的 encoded_frame_bytes，
    // 改码率时不用改代码。
    m_silence_frame.assign(static_cast<size_t>(m_config.encoded_frame_bytes), 0);
    std::vector<unsigned char> pcm(kPcmBytesPerFrame, 0);
    NET_DVR_AUDIOENC_PROCESS_PARAM param{};
    param.in_buf = pcm.data();
    param.out_buf = m_silence_frame.data();
    param.out_frame_size = static_cast<DWORD>(m_silence_frame.size());
    if (!NET_DVR_EncodeG722Frame(m_encoder, &param)) {
        set_error("NET_DVR_EncodeG722Frame failed, error=" +
                  std::to_string(NET_DVR_GetLastError()));
        return false;
    }
    return true;
}

// 启动保活线程。句柄先写进原子量再起线程，线程不会读到旧值或半初始化状态。
void HkSdkAudioClient::start_uplink_keepalive() {
    m_voice_handle_atomic.store(m_voice_handle);
    m_keepalive_running.store(true);
    m_keepalive_thread = std::thread(&HkSdkAudioClient::uplink_keepalive_loop, this);
}

// 停保活线程并清空句柄。可重复调用；join 最多等循环里的一小片 sleep（5 ms）。
void HkSdkAudioClient::stop_uplink_keepalive() {
    m_keepalive_running.store(false);
    if (m_keepalive_thread.joinable()) {
        m_keepalive_thread.join();
    }
    m_voice_handle_atomic.store(-1);
}

// 保活线程主体：每 40 ms 发一帧静音（G.722.1 为 80 B 编码帧，
// PCM 为 40 ms 裸 PCM；两者时长一致，线程只负责按节奏发送）。
// 发送失败只计数，不打印、不退出：偶发网络抖动不应该把线程打死，
// 持续失败可以从 stats.uplink_send_errors 看出来。
void HkSdkAudioClient::uplink_keepalive_loop() {
    while (m_keepalive_running.load()) {
        const LONG handle = m_voice_handle_atomic.load();
        if (handle >= 0 && !m_silence_frame.empty()) {
            const bool ok = NET_DVR_VoiceComSendData(
                handle, reinterpret_cast<char*>(m_silence_frame.data()),
                static_cast<DWORD>(m_silence_frame.size()));
            std::lock_guard<std::mutex> lock(m_mutex);
            if (ok) {
                ++m_stats.uplink_sent_frames;
            } else {
                ++m_stats.uplink_send_errors;
            }
        }
        // 分 8 片睡，stop() 的 join 不用等满一整帧。
        for (int i = 0; i < 8 && m_keepalive_running.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
}

// 是否处于采集状态。注意它只反映本类 flag，不代表链路真的在出音频：
// 设备静音时 SDK 可能长期不下发回调，此时 is_running() 仍为 true。
bool HkSdkAudioClient::is_running() const {
    return m_running.load();
}

// 是否正在写 wav。它和 is_running() 相互独立：语音转发启动后、
// start_record() 之前，is_running() == true 而 is_recording() == false。
bool HkSdkAudioClient::is_recording() const {
    return m_recording.load();
}

// 取走自上次调用以来的全部解码样本，语义对齐 Python 版 get_last_audio()：
// 一次性取空缓冲 → 拼成连续一维 int16 波形 → 清空。没有新数据时返回 false。
// 放在锁内复制，避免与回调线程追加样本交错；stop() 之后仍可取走残余样本，
// 只有 deinit() 才会清空缓冲。
bool HkSdkAudioClient::get_last_audio(std::vector<int16_t>& samples) {
    samples.clear();

    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_stream_buf.empty()) {
        return false;
    }

    samples.assign(m_stream_buf.begin(), m_stream_buf.end());
    m_stats.stream_samples_taken += samples.size();
    m_stream_buf.clear();
    return true;
}

// 取统计快照。运行中把 wall_seconds 换成「运行至今」的实时值，
// 停止后则保留 stop() 时定格的最终值，调用方拿到的语义一致。
AudioCaptureStats HkSdkAudioClient::get_stats() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    AudioCaptureStats stats = m_stats;
    if (m_running.load()) {
        const auto now = std::chrono::steady_clock::now();
        stats.wall_seconds =
            std::chrono::duration<double>(now - m_start_time).count();
    }
    return stats;
}

// 最近一次录制的正式文件路径：start_record() 解析出绝对路径，stop_record()
// 改名后仍保留在 m_final_path 里。从未录制时返回空字符串。
// 注意它读的是非原子字符串，调用时机应与 stop_record()/stop() 错开或由
// 单一控制线程使用（与 get_stats() 一样按「控制线程访问」约定）。
std::string HkSdkAudioClient::output_path() const {
    return m_final_path;
}

// 最近一次错误文本；只有 init()/start() 失败或 set_error() 被调用时才会更新，
// 运行中的解码/写文件错误只计数 + 首次打印，不会改这里。
std::string HkSdkAudioClient::last_error() const {
    return m_last_error;
}

// 当前编码的可读名，等价于查询结果里 type 的语义（0 / 8）。
std::string HkSdkAudioClient::codec_name() const {
    return m_codec == AudioCodec::kPcm ? "PCM" : "G.722.1";
}

// 输出采样率：wav 头、get_last_audio()、统计统一按它。
// 未配置降采样时等于设备率（G.722.1 16000 / PCM 本机实测 32000）；
// 配置 output_sample_rate=16000 后这里返回 16000，调用方无需再判断。
int HkSdkAudioClient::sample_rate() const {
    return m_output_sample_rate;
}

// 设备原始采样率（降采样前）。日志与「降采样前后样本数对账」用；
// 未 init() 时是 pcm_sample_rate 兜底值，不代表真实设备配置。
int HkSdkAudioClient::device_sample_rate() const {
    return m_device_sample_rate;
}

// ==================== SDK 生命周期 ====================

// SDK 初始化：进程级调用一次，但本类按实例记录标志（见 .h 约束 1）。
// 连接超时 3 s、重试 1 次：设备在局域网内，超时给太长会让 Ctrl+C 收尾变慢。
// 这里不调用 NET_DVR_SetReconnect——语音转发异常时 SDK 不会自动重建，
// 需要上层重新 init()/start()（当前 demo 不实现 watchdog）。
bool HkSdkAudioClient::init_sdk() {
    if (!NET_DVR_Init()) {
        set_error("NET_DVR_Init failed, error=" +
                  std::to_string(NET_DVR_GetLastError()));
        return false;
    }
    NET_DVR_SetConnectTime(3000, 1);
    m_sdk_initialized = true;
    return true;
}

// 登录设备（HCNetSDK 私有协议，默认 8000 端口），成功后保存 userId。
// 字符串字段用 strncpy 且保留 1 字节给 '\0'，配置里的 IP/账号超长会被截断，
// 不会越界。失败时 m_user_id 保持 -1，下次 init() 会重新尝试登录。
bool HkSdkAudioClient::login_device() {
    NET_DVR_USER_LOGIN_INFO login_info{};
    NET_DVR_DEVICEINFO_V40 device_info{};

    std::strncpy(login_info.sDeviceAddress, m_config.ip.c_str(),
                 sizeof(login_info.sDeviceAddress) - 1);
    login_info.wPort = static_cast<WORD>(m_config.sdk_port);
    std::strncpy(login_info.sUserName, m_config.username.c_str(),
                 sizeof(login_info.sUserName) - 1);
    std::strncpy(login_info.sPassword, m_config.password.c_str(),
                 sizeof(login_info.sPassword) - 1);

    m_user_id = NET_DVR_Login_V40(&login_info, &device_info);
    if (m_user_id < 0) {
        set_error("NET_DVR_Login_V40 failed, error=" +
                  std::to_string(NET_DVR_GetLastError()) + ", ip=" + m_config.ip);
        return false;
    }

    if (m_config.verbose) {
        std::cout << "[Audio] login success, userId=" << m_user_id
                  << ", ip=" << m_config.ip << std::endl;
    }
    return true;
}

// 读取设备当前音频编码（不写配置，避免像热成像那样留下持久改动）。
// 关键点：编码类型必须在 init() 阶段确定，因为 wav 头的采样率、以及
// 「走解码器还是裸 PCM 直写」两条路径都由它决定；中途切换编码会导致 wav
// 内容与头不一致，所以这里只接受 G.722.1 与 PCM 两种，其余直接拒绝启动。
// 返回 false 时 m_codec / m_device_sample_rate 保持 init() 里的默认值。
bool HkSdkAudioClient::query_audio_format() {
    NET_DVR_COMPRESSION_AUDIO compress{};
    if (!NET_DVR_GetCurrentAudioCompress(m_user_id, &compress)) {
        set_error("NET_DVR_GetCurrentAudioCompress failed, error=" +
                  std::to_string(NET_DVR_GetLastError()));
        return false;
    }

    const unsigned type = static_cast<unsigned>(compress.byAudioEncType);
    const unsigned rate_enum = static_cast<unsigned>(compress.byAudioSamplingRate);

    // byAudioSamplingRate：0-默认，1-16k，2-32k，3-48k，4-44.1k，5-8k。
    // 0（默认）没有对应的固定值，只能用调用方配置的兜底采样率；
    // 未列出的枚举返回 -1，由上层判定为不支持，避免猜出一个错误采样率。
    auto rate_from_enum = [&](int fallback) -> int {
        switch (rate_enum) {
            case 0: return fallback;
            case 1: return 16000;
            case 2: return 32000;
            case 3: return 48000;
            case 4: return 44100;
            case 5: return 8000;
            default: return -1;
        }
    };

    if (type == 0) {
        // G.722.1 标准就是 16 kHz，设备回 0（默认）或 1（16k）都视为 16 kHz；
        // 回其他值说明实际不是 G.722.1，wav 头会写错，直接拒绝。
        if (rate_enum != 0 && rate_enum != 1) {
            set_error("unsupported G.722.1 sample rate enum=" +
                      std::to_string(rate_enum) + " (expected 16 kHz)");
            return false;
        }
        m_codec = AudioCodec::kG7221;
        m_device_sample_rate = 16000;
        m_output_sample_rate = 16000;
    } else if (type == 8) {
        // PCM 是裸 PCM16，不需要解码；采样率按设备配置写进 wav 头。
        // 注意：本机实测设备会把 PCM 强制钉在 32 kHz，SetDVRConfig 请求 16 kHz
        // 不生效，所以这里的值必须来自设备返回，不能直接用 m_config。
        const int rate = rate_from_enum(m_config.pcm_sample_rate);
        if (rate <= 0) {
            set_error("unsupported PCM sample rate enum=" + std::to_string(rate_enum));
            return false;
        }
        m_codec = AudioCodec::kPcm;
        m_device_sample_rate = rate;
        m_output_sample_rate = rate;
    } else {
        set_error("unsupported audio encoding type=" + std::to_string(type) +
                  " (supported: G.722.1/type 0, PCM/type 8)");
        return false;
    }

    if (m_config.verbose) {
        std::cout << "[Audio] codec: " << codec_name()
                  << " (type=" << type << "), sampleRateEnum=" << rate_enum
                  << " -> device " << m_device_sample_rate << " Hz"
                  << ", bitRateEnum=" << static_cast<unsigned>(compress.byAudioBitRate)
                  << std::endl;
    }
    return true;
}

// 按配置决定输出采样率，必要时建立降采样器。
// 前置条件：query_audio_format() 已确定 m_device_sample_rate。
// 规则：0 → 直通；等于设备率 → 直通；高于设备率或不能整除 → 拒绝启动
// （只做整数倍降采样，不偷偷近似）。成功后 m_output_sample_rate 是
// wav 头、取流缓冲、统计统一使用的值。
bool HkSdkAudioClient::setup_output_resampler() {
    const int target = m_config.output_sample_rate;

    // 0 = 跟随设备：保持旧行为，G.722.1 输出 16k、PCM 输出设备实际的 32k。
    if (target <= 0) {
        m_output_sample_rate = m_device_sample_rate;
        m_resampler.reset();
        return true;
    }

    // 目标与设备一致：不需要滤波，直接直通（仍视作显式配置成功）。
    if (target == m_device_sample_rate) {
        m_output_sample_rate = target;
        m_resampler.reset();
        return true;
    }

    // 只做降采样：升采样会让语音变「糊」，且本工程没有对应需求。
    if (target > m_device_sample_rate) {
        set_error("output_sample_rate=" + std::to_string(target) +
                  " is higher than device rate " +
                  std::to_string(m_device_sample_rate) +
                  " Hz; upsampling is not supported");
        return false;
    }

    // 非整数比（如 44.1k→16k）需要多相/有理重采样，本类不实现；
    // 直接拒绝比悄悄用一个近似比例更安全。
    if (m_device_sample_rate % target != 0) {
        set_error("output_sample_rate=" + std::to_string(target) +
                  " cannot divide device rate " +
                  std::to_string(m_device_sample_rate) +
                  " Hz; only integer-ratio downsampling is supported");
        return false;
    }

    m_resampler =
        std::make_unique<AudioResampler>(m_device_sample_rate, target);
    if (!m_resampler->valid()) {
        m_resampler.reset();
        set_error("create resampler failed: " +
                  std::to_string(m_device_sample_rate) + " -> " +
                  std::to_string(target) + " Hz");
        return false;
    }
    m_output_sample_rate = target;

    if (m_config.verbose) {
        std::cout << "[Audio] resample: " << m_device_sample_rate << " -> "
                  << m_output_sample_rate << " Hz, ratio="
                  << m_resampler->ratio()
                  << " (anti-alias FIR, cutoff="
                  << m_output_sample_rate * 0.45 << " Hz)" << std::endl;
    }
    return true;
}

// 主动设置语音对讲编码：先快照设备原配置，再写目标值并读回复核。
// 仅在 init() 登录后、StartVoiceCom 前调用；以下情况返回 false 并写 last_error：
// 原配置读不到、SetDVRConfig 失败、或读回值与请求不一致（设备可能拒绝某组合，
// 例如实测 PCM 16 kHz 被钉在 32 kHz）。失败时尽量写回快照，避免设备停在半改状态。
bool HkSdkAudioClient::apply_audio_format() {
    // 1) 快照原配置：优先读设备保存值（1058）；部分设备不支持该命令，
    //    退回读当前生效值。两接口输出同一个 NET_DVR_COMPRESSION_AUDIO，
    //    恢复时按快照原样写回。
    NET_DVR_COMPRESSION_AUDIO saved{};
    DWORD returned = 0;
    if (!NET_DVR_GetDVRConfig(m_user_id, NET_DVR_GET_COMPRESSCFG_AUD, 0xFFFFFFFF,
                              &saved, sizeof(saved), &returned)) {
        saved = NET_DVR_COMPRESSION_AUDIO{};
        if (!NET_DVR_GetCurrentAudioCompress(m_user_id, &saved)) {
            set_error("save current audio format failed, error=" +
                      std::to_string(NET_DVR_GetLastError()));
            return false;
        }
    }
    m_saved_audio_format = saved;
    m_audio_format_saved = true;
    m_audio_format_changed = false;

    // 采样率不对外暴露，跟随编码取固定值：G.722.1 标准 16 kHz；
    // PCM 本机实测只能 32 kHz（请求 16 kHz 会被设备忽略）。
    // 其它编码没有解码/写盘路径，直接拒绝，避免写出内容与头不一致的 wav。
    BYTE target_rate_enum = 0;
    if (m_config.audio_enc_type == 0) {
        target_rate_enum = 1;  // 16 kHz
    } else if (m_config.audio_enc_type == 8) {
        target_rate_enum = 2;  // 32 kHz
    } else {
        set_error("unsupported target audio enc type=" +
                  std::to_string(m_config.audio_enc_type) +
                  " (supported: G.722.1/0, PCM/8)");
        return false;
    }

    // 2) 只改编码类型与采样率，其余字段（码率 / byres / bySupport）沿用快照，
    //    避免把设备不认识的字段写坏。lChannel 按手册对 1059 固定为无效值。
    NET_DVR_COMPRESSION_AUDIO target = saved;
    target.byAudioEncType = static_cast<BYTE>(m_config.audio_enc_type);
    target.byAudioSamplingRate = target_rate_enum;
    if (!NET_DVR_SetDVRConfig(m_user_id, NET_DVR_SET_COMPRESSCFG_AUD, 0xFFFFFFFF,
                              &target, sizeof(target))) {
        set_error("NET_DVR_SetDVRConfig(NET_DVR_SET_COMPRESSCFG_AUD) failed, error=" +
                  std::to_string(NET_DVR_GetLastError()));
        return false;
    }
    m_audio_format_changed = true;

    // 3) 读回复核：设备可能「返回成功但没接受请求」，wav 头与 ASR 输入都依赖
    //    实际采样率，所以必须用读回值而不是请求值。编码类型必须一致；采样率
    //    G.722.1 沿用 query_audio_format 的口径（0=默认、1=16k 都算 16 kHz），
    //    PCM 必须是 32k，否则说明设备没接受请求。
    NET_DVR_COMPRESSION_AUDIO actual{};
    if (!NET_DVR_GetCurrentAudioCompress(m_user_id, &actual)) {
        set_error("verify audio format failed, error=" +
                  std::to_string(NET_DVR_GetLastError()));
        restore_audio_format();
        return false;
    }
    const bool enc_ok = actual.byAudioEncType == target.byAudioEncType;
    const bool rate_ok =
        (m_config.audio_enc_type == 0)
            ? (actual.byAudioSamplingRate == 0 ||
               actual.byAudioSamplingRate == 1)
            : (actual.byAudioSamplingRate == 2);
    if (!enc_ok || !rate_ok) {
        set_error("audio format not applied: requested type=" +
                  std::to_string(m_config.audio_enc_type) + " rate=" +
                  std::to_string(target_rate_enum) +
                  ", actual type=" + std::to_string(actual.byAudioEncType) +
                  " rate=" + std::to_string(actual.byAudioSamplingRate));
        restore_audio_format();
        return false;
    }

    if (m_config.verbose) {
        std::cout << "[Audio] audio format set: type="
                  << static_cast<unsigned>(actual.byAudioEncType)
                  << ", sampleRateEnum="
                  << static_cast<unsigned>(actual.byAudioSamplingRate)
                  << " (original saved for restore)" << std::endl;
    }
    return true;
}

// 把 apply_audio_format() 保存的原配置写回设备。尽力而为：失败只记录
// last_error，不抛异常、不阻断 stop() 收尾；重复调用无副作用。
// 登录会话已失效时无法写回，直接清标志，避免下次误用旧快照。
void HkSdkAudioClient::restore_audio_format() {
    if (!m_audio_format_saved || !m_audio_format_changed) {
        return;
    }
    if (m_user_id < 0) {
        m_audio_format_saved = false;
        m_audio_format_changed = false;
        return;
    }
    if (!NET_DVR_SetDVRConfig(m_user_id, NET_DVR_SET_COMPRESSCFG_AUD, 0xFFFFFFFF,
                              &m_saved_audio_format,
                              sizeof(m_saved_audio_format))) {
        set_error("restore audio format failed, error=" +
                  std::to_string(NET_DVR_GetLastError()));
    } else if (m_config.verbose) {
        std::cout << "[Audio] audio format restored" << std::endl;
    }
    m_audio_format_saved = false;
    m_audio_format_changed = false;
}

// ==================== 回调与解码 ====================

// 1.hk_sdk触发的回调函数体
// SDK 回调的静态跳板：SDK 只收 C 风格函数指针，这里通过 pUser 找回实例。
// pUser 在 StartVoiceCom_MR_V30 时传入 this；SDK 可能在 stop() 返回后仍有
// 已排队的回调，因此 on_voice_data 内部会检查 m_running 再访问文件。
void CALLBACK HkSdkAudioClient::voice_callback(LONG /*lVoiceComHandle*/,
                                               char* pRecvDataBuffer, DWORD dwBufSize,
                                               BYTE byAudioFlag, void* pUser) {
    auto* self = static_cast<HkSdkAudioClient*>(pUser);
    if (self != nullptr) {
        self->on_voice_data(pRecvDataBuffer, dwBufSize, byAudioFlag);
    }
}

// 2.回调线程主体：过滤非设备音频 → 加锁 → 按编码分流。
// byAudioFlag 是 SDK 的音频来源标志，只有 1 表示「设备端编码后的音频」；
// 其他值可能是本地麦克风上行的控制数据，混进来会把噪声写进 wav。
// 本函数在锁内完成全部处理，不允许在此阻塞（例如打印过多或文件系统卡顿），
// 否则会反压 SDK 回调并造成丢包。注意：回调一次给的不一定刚好一帧，
// 编码路径先把字节攒进 m_encoded_buf，再由 decode_pending_locked 切帧。
void HkSdkAudioClient::on_voice_data(const char* buf, DWORD size, BYTE flag) {
    // byAudioFlag == 1 才是设备编码后的音频；其他标志是本地麦克风/控制数据
    if (flag != 1 || buf == nullptr || size == 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    // stop() 可能已经关闭文件；这里先看标志，避免对已 fclose 的 FILE* 写入。
    if (!m_running.load()) {
        return;
    }

    ++m_stats.packets;
    m_stats.encoded_bytes += size;

    // 0.静音/断线补零：先按到达时刻把虚拟时间轴推进到 now，若期间设备没有
    //   推音频，就在这里补等长 PCM16 0。必须放在交付本包音频之前——静音属于
    //   本包音频之前的空档，补在后面会让 ASR 的时间顺序颠倒。
    //   本函数其余部分会累加 m_gap_fed_samples，供下一次比较。
    update_silence_gap_locked(std::chrono::steady_clock::now());

    // 1.hk设备是pcm编码格式：原始 PCM16 直接交付（必要时先降采样）
    if (m_codec == AudioCodec::kPcm) {
        // 设备给的是 PCM16 裸数据；配置了 output_sample_rate 时在这里
        // 统一降采样，wav / 取流缓冲 / 统计都拿到同一份输出样本。
        // 未录制时 write_pcm_locked 因 m_wav == nullptr 自动变成空操作。
        m_stats.input_samples += size / 2;
        m_gap_fed_samples += static_cast<int64_t>(size / 2);
        deliver_pcm_locked(reinterpret_cast<const unsigned char*>(buf), size);
        return;
    }

    // 2.hk设备是G.722.1编码格式，先存入解码器的缓存 m_encoded_buf 里面
    // G.722.1：码流帧边界与回调边界无关，先全部追加再按固定帧长切。
    // 即便当前没在录制也继续解码：解码器是有状态的，跳过帧会让重新开始
    // 录制后的前几帧因状态失配而失真；解码结果写盘与否由 write_pcm_locked 判断。
    m_encoded_buf.insert(m_encoded_buf.end(),
                         reinterpret_cast<const unsigned char*>(buf),
                         reinterpret_cast<const unsigned char*>(buf) + size);
    // 把 m_encoded_buf 里已经攒够整帧的字节全部解码掉，解到剩余不足一帧为止。
    // 调用sdk的DecodeG722Frame（80 B → 1280 B）耗时微秒级，因此不单开解码线程，直接在回调中完成解码
    decode_pending_locked();
}

// 3.hk_sdk回调数据为编码压缩数据时，回调中的同步解码流程
// 从 m_encoded_buf 头部按 encoded_frame_bytes（默认 80 B）逐帧解码，
// 直到剩余不足一帧。调用前必须已持有 m_mutex，函数名后缀 locked 即此意。
// 失败策略：解码失败只计一次日志然后丢一帧继续，避免同一坏帧原地死循环；
// 若 proc_data_size 异常（0 或超过剩余长度），按消费一整帧处理，防止缓冲不前进。
// 写文件失败不影响继续解码：音频本身仍然有效，只是落盘有损。
void HkSdkAudioClient::decode_pending_locked() {
    const size_t frame_bytes = static_cast<size_t>(m_config.encoded_frame_bytes);
    if (frame_bytes == 0) {
        return;
    }

    // 循环而不是一次，是因为一次回调可能给了好几帧（实测平均 168 字节 ≈ 2 帧）
    while (m_encoded_buf.size() >= frame_bytes) {

        // 1.构造单帧音频结构体
        // 每次新建 param：in_data_size 必须是一帧，不能把整个缓冲传给解码器，否则多帧会被当成一帧解析，后续帧全部错位。
        NET_DVR_AUDIODEC_PROCESS_PARAM param{};
        param.in_buf = m_encoded_buf.data();
        param.in_data_size = static_cast<DWORD>(frame_bytes);
        param.out_buf = m_pcm_buf.data();
        param.out_frame_size = static_cast<DWORD>(m_pcm_buf.size());
        param.g711_type = 1;  // 与已验证探针保持一致；G.722.1 解码不使用该字段

        // 2.使用sdk的NET_DVR_DecodeG722Frame解码
        // param里面填入里要解码的数据信息param.in_buf，解码后的数据存在param.out_buf中
        if (!NET_DVR_DecodeG722Frame(m_decoder, &param)) 
        {
            // 单帧异常处理，跳过
            ++m_stats.decode_errors;
            if (m_stats.decode_errors == 1) {
                std::cerr << "[Audio] NET_DVR_DecodeG722Frame failed, error="
                          << NET_DVR_GetLastError() << std::endl;
            }
            // 丢掉这一帧继续，避免同一坏帧无限重试
            m_encoded_buf.erase(m_encoded_buf.begin(),
                                m_encoded_buf.begin() + frame_bytes);
            continue;
        }
        
        // 3. write_pcm_locked 将当前帧数据写入输出缓存，同时移除输入缓存m_encoded_buf中已经处理的部分
        // 3.1 防御性检查sdk消费数据和实际输入数据是否一致，假如实际消费大于规定的in_data_size，以实际消费的大小为准
        // proc_data_size 是本次sdk实际消费的编码字节数,大小为80b，和我们输入时规定的 in_data_size 一致
        size_t consumed = static_cast<size_t>(param.proc_data_size);
        if (consumed == 0 || consumed > m_encoded_buf.size()) {
            consumed = frame_bytes;
        }
        // 3.2 将解码结果写入输出缓存 m_pcm_buf
        if (param.out_frame_size > 0) {
            // 解码统计与是否录制无关：未录制时也累计，便于判断链路是否在出音频。
            // out_frame_size 是字节数，PCM16 单声道下 /2 才是采样点数。
            // 交付统一走 deliver_pcm_locked：G.722.1 输出本来就是 16 kHz，
            // 没有降采样器时直通，不产生额外拷贝以外的开销。
            m_stats.input_samples += param.out_frame_size / 2;
            m_gap_fed_samples += static_cast<int64_t>(param.out_frame_size / 2);
            deliver_pcm_locked(m_pcm_buf.data(), param.out_frame_size);
        }
        ++m_stats.decoded_frames;

        // 3.3 从 输入缓存 m_encoded_buf 头部，移除已消费部分；
        m_encoded_buf.erase(m_encoded_buf.begin(), m_encoded_buf.begin() + consumed);
    }
}

// ==================== 静音 / 断线补零 ====================

// 重置虚拟时间轴：下一次音频回调到达时重新起表（首次回调不补零）。
// 调用方必须持有 m_mutex；start_voice_forward() 与 stop() 都已在锁内调用。
void HkSdkAudioClient::reset_silence_gap() {
    m_gap_clock_started = false;
    m_gap_timeline_samples = 0;
    m_gap_fed_samples = 0;
    m_gap_fed_at_check = 0;
    m_gap_last = 0;
    m_gap_pending = false;
}

// 每次音频回调进来先调用：按墙钟推进「应到样本」，差值超过阈值就补 PCM16 0。
//
// 为什么需要补：设备静音时可能完全不回调，链路短暂停流时也会断包。若把前后
// 音频直接拼接，wav 与 ASR 的时间轴会比墙钟短，且越安静短得越多。这里维护
// 两条计数器（都在设备采样率域）：
//   m_gap_timeline_samples —— 从首次回调起，按墙钟「应到」的样本数；
//   m_gap_fed_samples      —— 实际交付的真实音频样本数（不含本函数补的 0）。
// 每次回调把 timeline 推进 dt × 设备率，差值超过阈值就说明中间有设备没推的
// 空档，补等长 0；补完把 timeline 对齐到 fed，避免同一段空档被重复补。
//
// 为什么不能「一超阈值就补」：网络或进程卡顿后，SDK 会把积压音频晚一些补发。
// 这些音频只是晚了、并没有丢，若按墙钟差补 0 再叠加补发音频，wav 会比墙钟
// 还长（实测冻结进程 2 秒，简单补零多出约 1.5 秒）。因此补零要过三道确认：
//   1) 差值首次超阈值只挂起，不立即补；
//   2) 上一次检查以来单次交付超过 200 ms（正常 G.722.1 约 84 ms、PCM 约 47 ms）
//      视为积压回放，继续等；
//   3) 差值比上次检查时小，说明还在追赶，也继续等。
// 只有差值稳定在一个较大值时才补，补的是「确认丢失」的那部分。
//
// 为什么用到达时刻：SDK 回调签名只有缓冲区和标志，没有设备时间戳。
// 为什么首次回调不补：StartVoiceCom 本身阻塞约 1.2 秒，设备起流前的等待
// 不是「丢掉的音频」，补了会在 wav 开头凭空造出一段静音。
// 阈值的作用：正常回调间隔约 40~90 ms，网络抖动会让 dt 个别偏大；小于
// threshold 的差值视为抖动，宁可少补也不要在语音中间插入碎静音。
void HkSdkAudioClient::update_silence_gap_locked(
    const std::chrono::steady_clock::time_point& now) {
    if (!m_config.pad_silence_gaps || m_device_sample_rate <= 0) {
        return;
    }

    if (!m_gap_clock_started) {
        m_gap_clock_started = true;
        m_last_voice_time = now;
        m_gap_timeline_samples = 0;
        m_gap_fed_samples = 0;
        m_gap_fed_at_check = m_gap_fed_samples;
        m_gap_last = 0;
        m_gap_pending = false;
        return;
    }

    const double dt =
        std::chrono::duration<double>(now - m_last_voice_time).count();
    m_last_voice_time = now;
    if (dt <= 0.0) {
        return;
    }
    m_gap_timeline_samples += static_cast<int64_t>(
        dt * static_cast<double>(m_device_sample_rate) + 0.5);

    // 上一次检查到现在，本次回调之外实际交付了多少音频（设备原始采样率）。
    // 单次超过 200 ms 说明 SDK 在补发积压（正常回调 G.722.1 约 84 ms、
    // PCM 约 47 ms），这时时间轴上的差值会随积压回放很快缩小，不能当成静音。
    const int64_t actual_prev = m_gap_fed_samples - m_gap_fed_at_check;
    m_gap_fed_at_check = m_gap_fed_samples;

    const int64_t gap = m_gap_timeline_samples - m_gap_fed_samples;
    const int64_t threshold = static_cast<int64_t>(
        m_config.silence_gap_threshold_seconds *
        static_cast<double>(m_device_sample_rate));
    if (gap <= threshold) {
        m_gap_last = 0;
        m_gap_pending = false;
        return;
    }

    // 1) 本回调之前刚发生过积压回放：先让它把缓冲追完，追完再判断残留。
    // 2) 差值比上次小：同样说明正在追赶，继续观察。
    // 3) 第一次发现空档：挂起，等下一次确认不是一次性抖动或回放。
    // 三个条件都排除后才补零，避免把「晚到的音频」再补一份 0（会把 wav
    // 拉得比墙钟还长，实测冻结进程 2 秒可多补约 1.5 秒）。
    const bool catchup_burst = actual_prev >
        static_cast<int64_t>(0.2 * static_cast<double>(m_device_sample_rate));
    const bool shrinking = m_gap_last > gap;
    m_gap_last = gap;
    if (catchup_burst || shrinking || !m_gap_pending) {
        m_gap_pending = true;
        return;
    }
    m_gap_pending = false;
    m_gap_last = 0;

    // 单次停顿封顶：设备断线几分钟后恢复时，一次性补几分钟静音会撑大 wav、
    // 也会让 ASR 长时间无输出；超过上限的部分丢弃，时间轴直接对齐到实际位置。
    int64_t pad = gap;
    if (m_config.silence_gap_max_seconds > 0.0) {
        const int64_t max_pad = static_cast<int64_t>(
            m_config.silence_gap_max_seconds *
            static_cast<double>(m_device_sample_rate));
        if (pad > max_pad) {
            pad = max_pad;
        }
    }

    // 分块补 0：与解码单元同量级（640 点 = 40 ms @16 kHz / 20 ms @32 kHz），
    // 避免为一次长停顿分配整段大缓冲；走 deliver_pcm_locked 意味着降采样、
    // wav 与取流缓冲看到的 0 和真实音频完全同一条路径。
    constexpr size_t kGapChunkSamples = 640;
    if (m_gap_zero_buf.size() < kGapChunkSamples * 2) {
        m_gap_zero_buf.assign(kGapChunkSamples * 2, 0);
    }
    int64_t remaining = pad;
    while (remaining > 0) {
        const size_t chunk = static_cast<size_t>(std::min<int64_t>(
            remaining, static_cast<int64_t>(kGapChunkSamples)));
        deliver_pcm_locked(m_gap_zero_buf.data(), chunk * 2);
        remaining -= static_cast<int64_t>(chunk);
    }
    m_gap_fed_samples += pad;
    m_stats.gap_padded_samples += static_cast<uint64_t>(pad);
    ++m_stats.gap_events;

    // 补完（或截断）后把时间轴对齐，避免剩余差值在后续回调里继续补 0。
    m_gap_timeline_samples = m_gap_fed_samples;
}

// 把一段 PCM16 字节交付给「wav + 取流缓冲 + 统计」，必要时先降采样。
// 前置条件：调用方持有 m_mutex。无降采样器时按原始字节直通，不做任何转换，
// 保持旧路径行为；有降采样器时转 int16 → FIR 抽取 → 再按小端写回字节，
// 让 wav、get_last_audio() 与统计看到完全相同的输出样本。
// input_samples 由调用方在进来前累加（设备原始点数），这里只累加输出点数。
void HkSdkAudioClient::deliver_pcm_locked(const unsigned char* data,
                                          size_t bytes) {
    if (data == nullptr || bytes < 2) {
        return;
    }

    // 1) 无降采样器：直通。奇数字节按原样写盘，取流侧丢弃尾字节。
    if (!m_resampler) {
        m_stats.pcm_samples += bytes / 2;
        write_pcm_locked(data, bytes);
        append_stream_locked(data, bytes);
        return;
    }

    // 2) 小端字节 → int16 样本（与 append_stream_locked 的解码口径一致）。
    const size_t count = bytes / 2;
    std::vector<int16_t> input(count);
    for (size_t i = 0; i < count; ++i) {
        const uint16_t raw = static_cast<uint16_t>(data[2 * i]) |
                             (static_cast<uint16_t>(data[2 * i + 1]) << 8);
        std::memcpy(&input[i], &raw, sizeof(int16_t));
    }

    // 3) 滤波抽取。历史未填满或不足一个输出周期时返回空，
    //    此时不写文件、不进缓冲、不计统计，样本留在重采样器内部状态里。
    const std::vector<int16_t> output =
        m_resampler->process(input.data(), input.size());
    if (output.empty()) {
        return;
    }

    // 4) 输出样本 → 小端字节，复用同一套写盘/缓冲路径。
    //    负数转 uint16 是模 2^16 的位模式转换，正好是补码表示。
    std::vector<unsigned char> out_bytes(output.size() * 2);
    for (size_t i = 0; i < output.size(); ++i) {
        const uint16_t raw = static_cast<uint16_t>(output[i]);
        out_bytes[2 * i] = static_cast<unsigned char>(raw & 0xFF);
        out_bytes[2 * i + 1] = static_cast<unsigned char>((raw >> 8) & 0xFF);
    }
    m_stats.pcm_samples += output.size();
    write_pcm_locked(out_bytes.data(), out_bytes.size());
    append_stream_locked(out_bytes.data(), out_bytes.size());
}

// 4. 回调中最后处理解码完成数据的方法，写入wav文件或丢弃
// 往 wav 追加 PCM16 数据，并累加 m_data_bytes / recorded_bytes。
// 调用前必须持有 m_mutex。
// 三种情况视为成功且不写：未在录制（m_wav == nullptr）、空数据、空指针，
// 这样上层无需到处判断「是否在录文件」——录制开关只体现为 m_wav 是否打开。
// 写失败只计一次日志并返回 false，不改变两个字节计数，后续帧继续尝试写。
bool HkSdkAudioClient::write_pcm_locked(const unsigned char* data, size_t size) {
    if (m_wav == nullptr || data == nullptr || size == 0) {
        return true;
    }

    const size_t written = std::fwrite(data, 1, size, m_wav);
    if (written != size) {
        ++m_stats.write_errors;
        if (m_stats.write_errors == 1) {
            std::cerr << "[Audio] fwrite failed, errno=" << std::ferror(m_wav)
                      << std::endl;
        }
        return false;
    }

    m_data_bytes += size;              // 用于 close_wav 回填 wav 头
    m_stats.recorded_bytes += size;    // 本次录制实际落盘字节，供调用方统计时长
    return true;
}

// ==================== 取流缓冲 ====================

// 把一段 PCM 字节追加到取流缓冲，供 get_last_audio() 一次性取走。
// 调用前必须持有 m_mutex；与录制无关，调用方（回调）总是传入解码后的 PCM16。
// 字节按小端解释（与 wav 规范一致），逐样本合成 int16；
// 不足一个样本的奇数字节直接丢弃（PCM 路径下属于异常半样本）。
// 缓冲超过 max_stream_samples 时从队头淘汰最旧样本，保证内存有界；
// 被淘汰的样本计入 stream_dropped_samples，持续增长说明调用方取流太慢。
void HkSdkAudioClient::append_stream_locked(const unsigned char* data, size_t bytes) {
    const int max_samples = m_config.max_stream_samples;
    if (data == nullptr || max_samples <= 0) {
        return;  // 未启用取流缓存
    }

    const size_t count = bytes / 2;
    for (size_t i = 0; i < count; ++i) {
        // 显式按小端拼 16 位，再按位模式拷贝成有符号样本，
        // 避免依赖主机字节序（部分 ARM 板可能为大端）。
        const uint16_t raw = static_cast<uint16_t>(data[2 * i]) |
                             (static_cast<uint16_t>(data[2 * i + 1]) << 8);
        int16_t sample = 0;
        std::memcpy(&sample, &raw, sizeof(sample));
        m_stream_buf.push_back(sample);
    }

    const size_t limit = static_cast<size_t>(max_samples);
    if (m_stream_buf.size() > limit) {
        const size_t overflow = m_stream_buf.size() - limit;
        for (size_t i = 0; i < overflow; ++i) {
            m_stream_buf.pop_front();
        }
        m_stats.stream_dropped_samples += overflow;
    }
}

// ==================== wav 写入 ====================

// 打开输出文件并先写 44 字节占位头；返回 false 时调用方不应置位录制标志。
// 前置条件：init() 已确定 m_output_sample_rate；wav_path 非空（start_record 校验）。
// 采用「先写 .part、收尾再改名」的策略：录制中途崩溃/强杀不会留下一个
// 长度字段错误但看起来合法的 wav。
bool HkSdkAudioClient::open_wav(const std::string& wav_path) {
    // 统一转成绝对路径，保证从任意工作目录运行都写到同一位置
    // （调用方传入的可能是相对路径，如 results/xxx.wav）。
    std::error_code ec;
    const std::filesystem::path final_path =
        std::filesystem::absolute(wav_path, ec);
    if (ec) {
        set_error("resolve output path failed: " + ec.message());
        return false;
    }

    // 路径先放局部量，只有文件完整打开成功才写回成员；
    // 这样本次打开失败不会把上一次成功录制的 output_path() 冲掉。
    const std::string final_str = final_path.string();
    const std::string part_str = final_str + ".part";

    // 目录不存在则逐级创建；空 parent（输出在当前目录）不需要创建。
    const std::filesystem::path parent = final_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            set_error("create output directory failed: " + ec.message());
            return false;
        }
    }

    FILE* file = std::fopen(part_str.c_str(), "wb");
    if (file == nullptr) {
        set_error("open wav failed: " + part_str);
        return false;
    }

    // 先写 44 字节占位头，stop_record() 时回填 RIFF/data 长度
    const uint32_t sample_rate = static_cast<uint32_t>(m_output_sample_rate);
    write_u32(file, 0x46464952);  // "RIFF"
    write_u32(file, 36);          // 占位：36 + data
    write_u32(file, 0x45564157);  // "WAVE"
    write_u32(file, 0x20746d66);  // "fmt "
    write_u32(file, 16);          // fmt chunk size
    write_u16(file, 1);           // PCM
    write_u16(file, 1);           // mono
    write_u32(file, sample_rate);
    write_u32(file, sample_rate * 2);  // byte rate = rate * channels * bytes/sample
    write_u16(file, 2);                // block align
    write_u16(file, 16);               // bits per sample
    write_u32(file, 0x61746164);       // "data"
    write_u32(file, 0);                // 占位：data size

    // 用 ftell 校验头长度：若磁盘写失败，fwrite 返回值会少于预期，
    // 这里能尽早发现，而不是等到收尾回填长度时才报错。
    if (std::ftell(file) != static_cast<long>(kWavHeaderSize)) {
        set_error("write wav header failed: " + part_str);
        std::fclose(file);
        std::error_code remove_ec;
        std::filesystem::remove(part_str, remove_ec);  // 头都写不全，不留半成品
        return false;
    }

    // 到这里文件才真正可用：写回成员并置零本段统计。
    m_wav = file;
    m_final_path = final_str;
    m_part_path = part_str;
    // m_data_bytes 只统计 PCM 负载，不含 44 字节头，close_wav 回填时用它。
    m_data_bytes = 0;
    return true;
}

// 收尾 wav：回填 RIFF/data 长度 → fflush → fclose → .part 改名为 .wav。
// 前置条件：调用方已持有 m_mutex，且 m_recording 已置 false（stop_record 先落旗）。
// 满足这两点时回调线程不会再进入 fwrite；语音转发本身可以仍在运行，
// 后续音频继续解码但 m_wav == nullptr，写入自动变成空操作。
// 改名失败不返回错误（本函数为 void）：保留 .part 让数据不丢，只在 stderr 提示。
// 可重复调用，m_wav 为空时直接返回。
void HkSdkAudioClient::close_wav() {
    if (m_wav == nullptr) {
        return;
    }

    // 回填长度：RIFF chunk size = 36 + data，data chunk size = PCM 字节数
    const uint32_t data_size =
        m_data_bytes > 0xFFFFFFFFull ? 0xFFFFFFFFu
                                     : static_cast<uint32_t>(m_data_bytes);
    std::fseek(m_wav, 4, SEEK_SET);
    write_u32(m_wav, 36u + data_size);
    std::fseek(m_wav, 40, SEEK_SET);
    write_u32(m_wav, data_size);
    std::fflush(m_wav);
    std::fclose(m_wav);
    m_wav = nullptr;
    m_data_bytes = 0;

    // 防御性检查：正常路径下两者在 open_wav 成功时已同时赋值；
    // 若被外部清空则跳过改名，只保留已写出的 .part 内容。
    if (m_part_path.empty() || m_final_path.empty()) {
        return;
    }

    // 同目录 rename 在 Linux 上是原子操作，读者不会看到半成品。
    std::error_code ec;
    std::filesystem::rename(m_part_path, m_final_path, ec);
    if (ec) {
        std::cerr << "[Audio] rename wav failed: " << ec.message()
                  << ", partial file kept at " << m_part_path << std::endl;
    }
}

// 按小端写 2 字节。wav 规范固定小端，不能直接 fwrite 主机字节序的整数，
// 否则在大端平台（如部分 ARM 板）写出的头会被解析成错误采样率。
void HkSdkAudioClient::write_u16(FILE* f, uint16_t value) {
    const unsigned char bytes[2] = {
        static_cast<unsigned char>(value & 0xFF),
        static_cast<unsigned char>((value >> 8) & 0xFF),
    };
    std::fwrite(bytes, 1, sizeof(bytes), f);
}

// 按小端写 4 字节，用于 RIFF/fmt/data 的大小、采样率、byte rate 字段。
// 不检查 fwrite 返回值：头长度由 open_wav 末尾的 ftell 统一校验。
void HkSdkAudioClient::write_u32(FILE* f, uint32_t value) {
    const unsigned char bytes[4] = {
        static_cast<unsigned char>(value & 0xFF),
        static_cast<unsigned char>((value >> 8) & 0xFF),
        static_cast<unsigned char>((value >> 16) & 0xFF),
        static_cast<unsigned char>((value >> 24) & 0xFF),
    };
    std::fwrite(bytes, 1, sizeof(bytes), f);
}

// 记录并打印错误：写 m_last_error（供 last_error() 查询）同时输出到 stderr。
// 不走 m_config.verbose 开关——错误信息始终要可见，verbose 只控制正常的
// 登录/编码格式等过程日志（见 init()/login_device()/query_audio_format()）。
void HkSdkAudioClient::set_error(const std::string& message) {
    m_last_error = message;
    std::cerr << "[Audio] " << message << std::endl;
}

}  // namespace hk_isapi

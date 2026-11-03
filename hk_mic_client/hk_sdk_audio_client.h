#pragma once

/**
 * @file hk_sdk_audio_client.h
 * @brief 海康 HCNetSDK 设备端音频采集客户端（独立于可见光/热成像客户端）。
 *
 * 职责边界：
 *   - 只负责「设备端拾音器 → 编码音频 → PCM → wav」这一条链路：
 *     登录设备、发起语音转发、接收回调、G.722.1 解码、可选降采样、写 wav、统计。
 *   - 不做语音识别、不做升采样/非整数比重采样、不做静音检测；
 *     识别由上层 ASR 模块完成。
 *   - 降采样默认关闭（output_sample_rate = 0）：只有调用方明确要求把
 *     输出钉到 16 kHz 时才启用，避免悄悄改变录音内容。
 *
 * 为什么用 NET_DVR_StartVoiceCom_MR_V30：
 *   - 该接口的回调 byAudioFlag == 1 时给的是**设备编码后的音频**，
 *     不需要本地声卡，适合无桌面/服务器环境。
 *   - NET_DVR_StartVoiceCom_V30 依赖本地声卡；NET_DVR_ClientAudioStart_V30
 *     的广播模式在 Linux 下不可用。
 *
 * 当前编码支持范围：
 *   - G.722.1（byAudioEncType == 0）：16 kHz、单声道、80 字节/40 ms，
 *     用 SDK 的 G722 解码器还原成 PCM16。
 *   - PCM（byAudioEncType == 8）：设备直接下发 PCM16 裸数据，按设备实际返回的
 *     采样率处理。本机实测切到 PCM 后运行态被固定为 32 kHz（请求 16 kHz
 *     不生效）；把 output_sample_rate 设为 16000 时，客户端内部用带抗混叠
 *     低通的 2 倍抽取降到 16 kHz，wav、取流缓冲、统计三者都是降采样后的数据。
 *   - 其他编码（G.711/AAC/MP2L2 等）会明确报错并拒绝录制，避免写出采样率错误的 wav。
 *
 * 数据流：
 *   init()：SDK 初始化 → 登录 → 查编码（只确认设备可用，不拉流）
 *   start()：NET_DVR_StartVoiceCom_MR_V30（只拉流，不写文件）
 *   stop()：停保活 → StopVoiceCom（拆线路，会话保留，可再次 start()）
 *   deinit()：恢复编码 → 登出 → 释放句柄 → NET_DVR_Cleanup
 *     → voice_callback(byAudioFlag == 1)
 *     ├─ 未录制：G.722.1 仍继续解码（保持解码器状态连续），结果直接丢弃
 *     ├─ G.722.1：编码字节缓冲（按 80 字节切帧）
 *     │            → NET_DVR_DecodeG722Frame
 *     │            → PCM16 16 kHz 单声道
 *     └─ PCM：PCM16 裸数据（设备实测 32 kHz）
 *     → 可选降采样：output_sample_rate 非 0 时降到目标率
 *     → <wav_path>.wav.part 追加写（录制中）
 *     → 同时追加进取流缓冲（无论是否在录制），供 get_last_audio() 取走
 *   stop_record() 时回填 RIFF/data 长度并改名为 .wav，语音转发继续保持运行
 *
 * 约束：
 *   1. 本类沿用现有客户端的「实例级 NET_DVR_Init/Cleanup」写法。
 *      同一进程同时创建音频客户端和视频/热成像客户端时，先析构的那个调用
 *      NET_DVR_Cleanup 会让另一个失效；混跑前需改成进程级引用计数（本轮不解决）。
 *   2. 设备静音时可能不下发音频包，因此 wav 的音频时长可能小于墙钟时长；
 *      本类不补静音，只统计两者，便于调用方判断。
 *   3. 回调线程里直接解码并写文件，写文件失败只计数，不中断取流。
 *   4. init() 只登录并确认编码，start() 只建立语音转发，两者都不产生文件；
 *      真正写盘由 start_record(path) 开启、stop_record() 结束。两段录制之间
 *      （或从未录制时）不落任何文件。
 *   5. 取流缓冲独立于录制：只要语音转发在运行就一直累积（受 max_stream_samples
 *      上限约束），get_last_audio() 取走后清空；录制开关不影响取流内容。
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "HCNetSDK.h"
#include "audio_resampler.h"

namespace hk_isapi {

    
/**
 * @brief 设备实际下发的音频编码。
 *
 * - kG7221：编码帧，需要 SDK 的 G722 解码器；
 * - kPcm：已经是对应采样率的 PCM16 裸数据，直接写 wav，不需要解码。
 *
 * AAC/MP2L2 等编码网络 SDK 没有提供解码接口，本客户端暂不支持。
 */
enum class AudioCodec {
    kG7221,
    kPcm,
};

/**
 * @brief 音频采集配置（由 main 逐项填写，类内不藏业务参数）。
 */
struct AudioCaptureConfig {
    std::string ip = "192.168.1.65";      ///< 设备 IP
    int sdk_port = 8000;                  ///< HCNetSDK 私有协议端口
    std::string username = "admin";       ///< 登录用户名
    std::string password = "Password";    ///< 登录密码
    int voice_channel = 1;                ///< 语音通道号，本机实测 1
    int pcm_sample_rate = 16000;          ///< rateEnum=0(默认) 时的兜底采样率；G.722.1 固定 16 kHz
    int encoded_frame_bytes = 80;         ///< G.722.1 单帧编码字节数：80 B / 40 ms

    // —— 可选输出采样率（降采样）——
    // 0 = 跟随设备（默认，行为与旧版完全一致）；16000 = 把输出钉在 16 kHz。
    // 只在设备率高于目标率且能整除时启用（本机 PCM 32k→16k）；升采样、
    // 非整数比（如 44.1k→16k）直接拒绝启动，不做近似。
    // 降采样带 Kaiser 窗低通（7.2 kHz 截止、8 kHz 以上 ≥60 dB），
    // 避免高频折叠进语音频带；wav 头、取流缓冲、统计都用输出采样率。
    int output_sample_rate = 0;

    // —— 启动前主动设置语音对讲编码（可选）——
    // 走 NET_DVR_SetDVRConfig(NET_DVR_SET_COMPRESSCFG_AUD, 1059) + NET_DVR_COMPRESSION_AUDIO。
    // 注意：该接口写的是**设备持久配置**，会影响该设备上所有对讲客户端；
    // 默认关闭，保持「只查询、不修改设备」的原有行为。
    bool set_audio_format = false;        ///< 是否在 init() 登录后主动设置编码
    int audio_enc_type = 0;               ///< 目标编码：0=G.722.1（自动 16k），8=PCM（自动 32k）；
                                          ///< 采样率不单独配置，跟随编码取固定值
    bool restore_audio_format = true;     ///< deinit() 时是否写回设置前的设备配置

    int max_stream_samples = 40960;       ///< 取流缓冲上限（样本数）：40960 ≈ 2.56 s @16 kHz，
                                          ///< 与 Python mic 的 256×160 块对齐；满了丢最旧样本；
                                          ///< <=0 表示关闭取流缓存（get_last_audio 恒返回 false）
    bool uplink_keepalive = true;         ///< 每 40 ms 回发一帧 80 B G.722 静音
                                          ///< （PCM 下行也用 G.722 上行，见 .cpp 实测说明）。
                                          ///< 手册要求语音转发双向；实测只收不发时设备
                                          ///< 约 10 秒后停掉回调，置 false 会复现该断流
    bool verbose = true;                  ///< 是否打印登录、编码格式、错误信息
};

/**
 * @brief 采集统计（stop() 后读取；运行中也可读）。
 */
struct AudioCaptureStats {
    uint64_t packets = 0;             ///< 收到的语音回调包数
    uint64_t encoded_bytes = 0;       ///< 收到的编码字节数
    uint64_t decoded_frames = 0;      ///< 成功解码的 80 字节帧数
    uint64_t input_samples = 0;       ///< 设备原始采样点数（降采样前；G.722.1 与 pcm_samples 相同）
    uint64_t pcm_samples = 0;         ///< 输出 PCM 采样点数（降采样后；未录制时也累计）
    uint64_t recorded_bytes = 0;      ///< 最近一次录制实际写入的 PCM 字节数，start_record() 时清零
    uint64_t stream_dropped_samples = 0; ///< 取流缓冲满被淘汰的样本数；持续增长说明取流太慢
    uint64_t stream_samples_taken = 0;   ///< 已被 get_last_audio() 取走的累计样本数
    uint64_t decode_errors = 0;       ///< DecodeG722Frame 失败次数
    uint64_t write_errors = 0;        ///< fwrite 失败次数
    uint64_t uplink_sent_frames = 0;  ///< 上行保活成功发送的静音帧数（固定 80 B G.722/帧）
    uint64_t uplink_send_errors = 0;  ///< 上行保活发送失败次数；持续增长说明会话已失效
    uint64_t dropped_partial_bytes = 0; ///< stop() 时不足一帧而丢弃的字节数
    double wall_seconds = 0.0;        ///< 从 init() 到 deinit() 的墙钟时长（含未转发时间）
    double forward_seconds = 0.0;     ///< 累计实际转发时长：每次 stop() 累加，不含停止时间
};

/**
 * @brief 设备端音频采集客户端。
 *
 * 典型用法：
 *   AudioCaptureConfig cfg;
 *   cfg.ip = "10.168.2.65";
 *   HkSdkAudioClient client(cfg);
 *   client.init();                                 // 登录并确认编码，不拉流
 *   ...                                            // 可以在这段时间加载模型/初始化上层
 *   client.start();                                // 建立语音转发，不写文件
 *   client.start_record("/path/to/hk_audio.wav");  // 开始写盘
 *   ...                                            // 录制任意时长
 *   client.stop_record();                          // 收尾写 wav 头并改名
 *   client.stop();                                 // 断开语音转发，保留会话
 *   client.deinit();                               // 登出并释放 SDK 资源
 */
class HkSdkAudioClient {
public:
    explicit HkSdkAudioClient(const AudioCaptureConfig& config);
    HkSdkAudioClient();
    ~HkSdkAudioClient();

    HkSdkAudioClient(const HkSdkAudioClient&) = delete;
    HkSdkAudioClient& operator=(const HkSdkAudioClient&) = delete;

    /**
     * @brief 初始化会话：SDK 初始化、登录设备、确认编码、建降采样器与解码器。
     *
     * 只确认设备可用，不调用 StartVoiceCom、不产生音频、不打开 wav。
     * 典型用法是把它放在耗时的上层初始化（例如加载 ASR 模型）之前，
     * 设备不可用时能尽早报错；等上层就绪后再 start() 拉流，避免设备
     * 在没人消费的时候一直推流把取流缓冲塞满。
     * @return 成功返回 true；失败时 last_error() 给出原因。
     * @note 可重复调用；成功后重复调用直接返回 true。
     */
    bool init();

    /**
     * @brief 启动语音转发：StartVoiceCom + 上行保活。
     *
     * 不重新登录、不重查编码；为保证与设备当前编码流匹配，每次都会重建
     * G.722 解码器。设备在 StopVoiceCom 后约 200 ms 才允许重新起流，
     * 因此 StartVoiceCom 失败时按 200 ms 间隔重试，最多 5 次。
     * @return 起流成功或本来就在转发返回 true；未 init() 或重试全部失败返回 false。
     * @note 调用会同步阻塞约 1.2 秒（StartVoiceCom 本身的开销），之后首个
     *       音频回调约 1.2 秒后到达，不是丢数据。
     */
    bool start();

    /**
     * @brief 停止语音转发：只关闭对讲线路，保留 SDK、登录会话与解流缓冲。
     *
     * 停保活线程 → StopVoiceCom → 自动 stop_record()（录制中时）→
     * 残包计数 → 累加实际转发时长。与 deinit() 的区别是「之后还能 start()」。
     * 取流缓冲里已缓存的样本保留，调用方仍可用 get_last_audio() 取完。
     * @return 停止成功或本来就没在转发返回 true；写错误原因见 last_error()。
     * @note stop() 后 is_running() 返回 false；设备编码配置此时不恢复，
     *       要等 deinit() 才写回原值。
     */
    bool stop();

    /**
     * @brief 结束会话：停在转发 → 恢复设备编码 → 释放解码器 → 登出 → 清理 SDK。
     *
     * 与析构函数做的事相同，显式调用可以把「释放资源」和「打印统计」的
     * 时机分开：先 stop() 取完残余、读 get_stats()，再 deinit()。
     * @note 可重复调用；未 init() 时为空操作。析构函数会兜底再调一次。
     */
    void deinit();

    /**
     * @brief 开启录制：打开 wav 文件并开始追加 PCM 数据。
     * @param wav_path 输出路径（绝对或相对路径均可，父目录不存在会自动创建）。
     * @return 成功返回 true；以下情况返回 false 并写入 last_error()：
     *         wav_path 为空、尚未 start()、已经在录制、文件无法打开。
     * @note 必须已 start()；可重复「start_record → stop_record」多段录制，
     *       每次传入不同路径即可得到多个独立 wav。
     */
    bool start_record(const std::string& wav_path);

    /**
     * @brief 结束录制：回填 RIFF/data 长度，把 .wav.part 改名为 .wav。
     *
     * 语音转发保持运行，后续音频仍会解码但不再写盘，可再次 start_record()。
     * @note 可重复调用；未在录制时为空操作。
     */
    void stop_record();

    /// 当前是否正在转发（stop() 后返回 false；不代表一定有音频，设备静音时也可能没数据）
    bool is_running() const;

    /// 当前是否正在写 wav
    bool is_recording() const;

    /**
     * @brief 取走自上次调用以来累积的全部解码音频，拼成一维 PCM16 波形。
     *
     * 与音频是否正在录制无关：只要语音转发在运行，解码结果就会进入取流缓冲。
     * 接口语义对齐 Python 版 mic_client.get_last_audio()：
     *   - 一次性取走并清空缓冲，波形连续、不重复、不丢样本；
     *   - 自上次调用以来没有新数据时返回 false，samples 被清空；
     *   - 缓冲超过 max_stream_samples 时丢弃最旧样本（计入 stream_dropped_samples）。
     * @param[out] samples 单声道 PCM16 样本，采样率见 sample_rate()。
     * @return 取到新数据返回 true；无新数据返回 false。
     * @note 线程安全，可与回调线程并发调用；取走的是缓冲区当前内容，
     *       因此调用频率应高于缓冲上限对应的时长（默认 2.56 s），否则会丢旧数据。
     */
    bool get_last_audio(std::vector<int16_t>& samples);

    /// 获取统计快照（线程安全）
    AudioCaptureStats get_stats() const;

    /// 最近一次录制的正式文件路径（.part 改名后）；从未录制时返回空字符串
    std::string output_path() const;

    /// 最近一次错误描述
    std::string last_error() const;

    /// 当前设备编码的可读名称（G.722.1 / PCM）
    std::string codec_name() const;

    /**
     * @brief 输出采样率（wav 头、get_last_audio()、统计都按它）。
     *
     * output_sample_rate 为 0 时等于设备采样率；配置了降采样时是目标率
     * （例如设备 32 kHz、配置 16000 → 返回 16000）。
     */
    int sample_rate() const;

    /// 设备实际采样率（降采样前的原始率），用于日志与统计；未 init() 时为兜底值
    int device_sample_rate() const;

private:
    // —— SDK 生命周期 ——
    bool init_sdk();
    bool login_device();
    bool query_audio_format();
    bool setup_output_resampler();
    bool open_wav(const std::string& wav_path);
    void close_wav();

    // —— 回调与解码 ——
    static void CALLBACK voice_callback(LONG lVoiceComHandle, char* pRecvDataBuffer,
                                        DWORD dwBufSize, BYTE byAudioFlag, void* pUser);
    void on_voice_data(const char* buf, DWORD size, BYTE flag);
    void decode_pending_locked();
    void deliver_pcm_locked(const unsigned char* data, size_t bytes);
    void append_stream_locked(const unsigned char* data, size_t bytes);

    // —— 语音转发起停（init 建会话，start / stop 反复开关对讲线路）——
    bool start_voice_forward();
    void add_forward_seconds_locked();

    // —— 语音对讲编码设置（可选，写设备持久配置）——
    bool apply_audio_format();
    void restore_audio_format();

    // —— 上行保活（两种下行编码都回发 G.722 80 B 静音）——
    bool init_uplink_keepalive();
    void start_uplink_keepalive();
    void stop_uplink_keepalive();
    void uplink_keepalive_loop();

    // —— wav 写入 ——
    bool write_pcm_locked(const unsigned char* data, size_t size);
    static void write_u16(FILE* f, uint16_t value);
    static void write_u32(FILE* f, uint32_t value);

    void set_error(const std::string& message);

private:
    AudioCaptureConfig m_config;

    // SDK 句柄
    LONG m_user_id = -1;
    LONG m_voice_handle = -1;
    void* m_decoder = nullptr;
    bool m_sdk_initialized = false;

    // 采集状态
    std::atomic<bool> m_session_ready{false}; ///< SDK/登录/编码/解码器已就绪（init 成功后置位）
    std::atomic<bool> m_running{false};       ///< 是否正在转发（start 置 true，stop 置 false）
    std::atomic<bool> m_recording{false};   ///< 是否正在写 wav（start_record/stop_record 控制）
    mutable std::mutex m_mutex;
    std::vector<unsigned char> m_encoded_buf;   ///< 编码字节缓冲，按 80 字节切帧
    std::vector<unsigned char> m_pcm_buf;       ///< 解码输出缓冲
    std::deque<int16_t> m_stream_buf;           ///< 取流缓冲，回调写尾、get_last_audio 取走整段
    AudioCaptureStats m_stats;

    AudioCodec m_codec = AudioCodec::kG7221;    ///< 设备当前编码
    int m_device_sample_rate = 16000;           ///< 设备实际采样率（降采样前）
    int m_output_sample_rate = 16000;           ///< 输出采样率：wav 头 / 取流 / 统计
    // 整数倍降采样器：仅当 output_sample_rate 非 0 且低于设备率时创建；
    // 为空表示直通（G.722.1 路径与未配置降采样的 PCM 路径都走这里）。
    std::unique_ptr<AudioResampler> m_resampler;

    // 主动设置编码时的原值快照：apply 前保存，deinit()/init 失败时写回，
    // 避免把设备留在「被本进程改过」的持久状态。
    bool m_audio_format_saved = false;          ///< 是否已成功读到原配置
    bool m_audio_format_changed = false;        ///< 是否已成功写入目标配置
    NET_DVR_COMPRESSION_AUDIO m_saved_audio_format{}; ///< apply 前的配置快照

    // 上行保活：句柄用原子量给保活线程读；stop() 先停线程再关语音转发
    void* m_encoder = nullptr;                   ///< G.722 编码器（start 里建静音帧用）
    std::vector<unsigned char> m_silence_frame;  ///< 80 B G.722 静音帧，线程只读
    std::atomic<LONG> m_voice_handle_atomic{-1}; ///< 保活线程看到的语音转发句柄
    std::atomic<bool> m_keepalive_running{false};
    std::thread m_keepalive_thread;

    // wav 文件
    FILE* m_wav = nullptr;
    std::string m_part_path;      ///< 正在写的 .wav.part
    std::string m_final_path;     ///< 收尾后改名为 .wav
    uint64_t m_data_bytes = 0;    ///< 已写入的 PCM 字节数

    // 计时
    std::chrono::steady_clock::time_point m_start_time;   ///< 本次 init() 的墙钟起点
    std::chrono::steady_clock::time_point m_forward_start; ///< 本段转发的起点，stop() 时结算

    std::string m_last_error;
};

}  // namespace hk_isapi

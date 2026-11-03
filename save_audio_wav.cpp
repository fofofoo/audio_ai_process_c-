/**
 * @file save_audio_wav.cpp
 * @brief 设备端音频录制 demo：启动海康语音转发，录制 10 秒并保存为 wav。
 *
 * 职责边界：
 *   - 只做「配置参数 → init/start → 录制固定时长 → 停止落盘 → deinit → 打印统计」；
 *   - 登录、编码查询、G.722.1 解码、wav 头写入全部由 HkSdkAudioClient 完成；
 *   - 不做重采样、不做语音识别，产出的 wav 供 Python 侧直接读取。
 *
 * 输出位置：
 *   <工程根>/results/hk_audio_<YYYYmmdd_HHMMSS>.wav
 *   文件名带时间戳，重复运行不会覆盖上一段录音；目录不存在时由客户端自动创建。
 *
 * 注意：
 *   - 录的是设备端拾音器音频，不是本机麦克风；
 *   - 设备静音时可能不下发音频包，wav 实际时长可能小于 10 秒墙钟时长，
 *     程序结束时会同时打印墙钟时长与按采样点数折算的音频时长，便于判断。
 */

#include "hk_sdk_audio_client.h"

#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace hk_isapi;

namespace {

// 工程根目录：CMake 通过 HK_PROJECT_DIR 编译宏注入（指向 CMakeLists 所在目录），
// 未定义时退回当前工作目录，保证单独手工编译也能跑。
#ifdef HK_PROJECT_DIR
constexpr const char* kProjectDir = HK_PROJECT_DIR;
#else
constexpr const char* kProjectDir = ".";
#endif

// 生成带时间戳的 wav 文件名，避免多次运行互相覆盖。
// 格式：hk_audio_20261102_153045.wav（本地时间）。
std::string make_wav_filename() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    std::tm local_tm{};
    localtime_r(&now_time, &local_tm);

    std::ostringstream name;
    name << "hk_audio_" << std::put_time(&local_tm, "%Y%m%d_%H%M%S") << ".wav";
    return name.str();
}

}  // namespace

int main() {
    // —— 一、录制参数（全部写死在 main，现场需要改时一眼可见）——
    const int kRecordSeconds = 10;             // 录制时长：墙钟 10 秒
    const std::string kDeviceIp = "10.168.2.65";  // 音频设备 IP（与可见光 10.168.2.64 不是同一台）
    const int kSdkPort = 8000;                 // HCNetSDK 私有协议端口
    const std::string kUsername = "admin";
    const std::string kPassword = "Password";
    const int kVoiceChannel = 1;               // 语音通道号，本机实测为 1

    // —— 二、填写采集配置 ——
    // 输出路径：<工程根>/results/<时间戳>.wav，客户端内部会转绝对路径并建目录。
    const std::filesystem::path output_path =
        std::filesystem::path(kProjectDir) / "results" / make_wav_filename();

    AudioCaptureConfig config;
    config.ip = kDeviceIp;
    config.sdk_port = kSdkPort;
    config.username = kUsername;
    config.password = kPassword;
    config.voice_channel = kVoiceChannel;
    config.verbose = true;
    // 可选：启动前主动设置对讲编码（写设备持久配置，默认关闭）。
    // 打开后 init() 先快照原值再设置并读回复核，deinit() 默认写回原配置。
    // 采样率跟随编码自动取：G.722.1 → 16 kHz，PCM → 32 kHz。
    config.set_audio_format = false;
    config.audio_enc_type = 0; // 0=G.722.1(16k)，8=PCM(32k)
    config.restore_audio_format = true;
    // 输出采样率：0=跟随设备（默认，PCM 下仍是 32 kHz）；16000=降采样到 16 kHz。
    // 录制 demo 保持设备原始采样率；需要 16 kHz wav 时改成 16000。
    config.output_sample_rate = 0;

    std::cout << "[Record] target: " << output_path.string() << "\n"
              << "[Record] device: " << kDeviceIp << ":" << kSdkPort
              << ", channel=" << kVoiceChannel
              << ", duration=" << kRecordSeconds << "s" << std::endl;

    // —— 三、启动采集 ——
    // init() 只登录 + 查询编码，start() 才建立语音转发；本入口没有模型加载，
    // 两段紧挨着调用，但仍按同一套生命周期写，与实时 ASR 入口保持一致。
    // 两者都不产生文件，写盘必须显式 start_record()。
    HkSdkAudioClient client(config);
    if (!client.init()) {
        // 失败原因已由客户端打印到 stderr，这里只补上下文。
        std::cerr << "[Record] init failed: " << client.last_error() << std::endl;
        return -1;
    }
    if (!client.start()) {
        std::cerr << "[Record] start failed: " << client.last_error() << std::endl;
        client.deinit();
        return -1;
    }

    std::cout << "[Record] codec=" << client.codec_name()
              << ", sample_rate=" << client.sample_rate() << " Hz" << std::endl;

    if (!client.start_record(output_path.string())) {
        std::cerr << "[Record] start_record failed: " << client.last_error() << std::endl;
        client.deinit();
        return -1;
    }

    // —— 四、录制固定时长 ——
    // 每秒打印一次进度，让使用者知道没有卡住；
    // 退出条件只依赖墙钟，因为 is_running() 只反映本地标志，
    // 设备静音或网络中断时它仍然为 true，不能用来判断「录到了音频」。
    for (int elapsed = 1; elapsed <= kRecordSeconds; ++elapsed) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        const AudioCaptureStats stats = client.get_stats();

        // 4.1 取音频：与录制落盘无关，取走自上次调用以来累积的全部 PCM16 样本，
        // 对齐 Python 版 mic.get_last_audio() 的用法；没有新数据时 samples 为空。
        std::vector<int16_t> samples;
        const bool has_audio = client.get_last_audio(samples);

        std::cout << "[Record] " << elapsed << "/" << kRecordSeconds
                  << " s, packets=" << stats.packets
                  << ", audio=" << std::fixed << std::setprecision(2)
                  << (client.sample_rate() > 0
                          ? static_cast<double>(stats.recorded_bytes) / 2 /
                                client.sample_rate()
                          : 0.0)
                  << " s, stream=" << (has_audio ? samples.size() : 0) << " smp"
                  << std::endl;
    }

    // —— 五、结束录制并落盘 ——
    // stop_record() 只收尾文件，语音转发仍在运行；
    // stop() 再断开转发（内部会兜底调用 stop_record()）；
    // deinit() 最后释放会话（恢复设备编码、登出、NET_DVR_Cleanup），
    // 同时结算 wall_seconds —— 下面第六节要打印它，所以必须放在前面。
    client.stop_record();
    client.stop();
    client.deinit();

    // —— 六、打印结果 ——
    const AudioCaptureStats stats = client.get_stats();
    const std::string final_path = client.output_path();
    // 录制时长按实际落盘字节折算：recorded_bytes / 2 字节每样本 / 采样率。
    const double audio_seconds =
        client.sample_rate() > 0
            ? static_cast<double>(stats.recorded_bytes) / 2 / client.sample_rate()
            : 0.0;

    std::cout << "\n[Record] ===== summary =====\n"
              << "[Record] file        : " << final_path << "\n"
              << "[Record] codec       : " << client.codec_name() << "\n"
              << "[Record] sample_rate : " << client.sample_rate() << " Hz\n"
              << "[Record] wall        : " << std::fixed << std::setprecision(2)
              << stats.wall_seconds << " s\n"
              << "[Record] audio       : " << audio_seconds << " s\n"
              << "[Record] packets     : " << stats.packets << "\n"
              << "[Record] recorded    : " << stats.recorded_bytes << " B\n"
              << "[Record] pcm_samples : " << stats.pcm_samples << "\n"
              << "[Record] stream_taken: " << stats.stream_samples_taken << " smp\n"
              << "[Record] stream_drop : " << stats.stream_dropped_samples << " smp\n"
              << "[Record] decode_err  : " << stats.decode_errors << "\n"
              << "[Record] write_err   : " << stats.write_errors << "\n"
              << "[Record] dropped     : " << stats.dropped_partial_bytes << " B"
              << std::endl;

    // 落盘字节为 0 说明整段没录到音频（设备静音、通道号错误或未授权），
    // 此时 wav 只有 44 字节合法头，文件存在但不含音频。
    if (stats.recorded_bytes == 0) {
        std::cerr << "[Record] warning: no audio captured, wav contains header only"
                  << std::endl;
        return -2;
    }

    return 0;
}

#pragma once

/**
 * @file wav_reader.h
 * @brief 最小 RIFF/WAVE 读取工具：wav 文件 → 单声道 float [-1,1]。
 *
 * 从 test_spectrogram_cv_wav.cpp 的匿名 namespace 抽出来，供 wav 语谱图入口
 * 与 wav ASR 入口共用，避免两份解析逻辑随修改走偏。解析行为完全照搬原实现：
 *   - 支持 PCM16（format=1）与 IEEE float32（format=3），
 *     包括 WAVE_FORMAT_EXTENSIBLE（0xFFFE）包装；
 *   - 多声道只取第 0 声道；
 *   - 不做重采样：采样率原样返回，由调用方决定是否继续。
 *
 * 头文件内实现（inline）：两个入口各编译一份，不新增 .cpp 与链接依赖。
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace audio_ai {

namespace wav_detail {

// 小端读取：wav 的块长度、采样率等字段都按小端存放。
// 不直接 memcpy 到整型，避免依赖主机字节序（ARM 上需要）。
inline uint16_t read_le16(const unsigned char* p) {
    return static_cast<uint16_t>(p[0]) |
           static_cast<uint16_t>(static_cast<uint16_t>(p[1]) << 8);
}

inline uint32_t read_le32(const unsigned char* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

}  // namespace wav_detail

// 读入的 wav 内容：样本已统一成单声道 float [-1,1]。
struct WavData {
    std::vector<float> samples;
    int sample_rate = 0;
    int channels = 0;          // 文件原始声道数；samples 只保留第一声道
    int bits_per_sample = 0;   // 16 = PCM16，32 = float32
};

// 解析 wav 文件并转成 float 单声道。
// 支持：PCM 16-bit、IEEE float 32-bit（含 WAVE_FORMAT_EXTENSIBLE 包装）；
// 多声道只取第 0 声道，与 Python 版 soundfile 的 data[:, 0] 一致。
// 失败时写 error 并返回 false，不修改 out 之外的任何状态。
inline bool load_wav(const std::string& path, WavData& out, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "打不开 wav：" + path;
        return false;
    }

    in.seekg(0, std::ios::end);
    const std::streamoff file_size = in.tellg();
    if (file_size < 12) {
        error = "wav 文件太短：" + path;
        return false;
    }
    in.seekg(0, std::ios::beg);

    std::vector<unsigned char> bytes(static_cast<size_t>(file_size));
    in.read(reinterpret_cast<char*>(bytes.data()), file_size);
    if (!in) {
        error = "读取 wav 失败：" + path;
        return false;
    }

    if (std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
        std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
        error = "不是 RIFF/WAVE 文件：" + path;
        return false;
    }

    // 逐块扫描，只关心 fmt 与 data。块长度按 RIFF 规定做 2 字节对齐。
    uint16_t format_tag = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits = 0;
    bool have_fmt = false;
    bool have_data = false;
    size_t data_offset = 0;
    size_t data_size = 0;

    size_t pos = 12;
    while (pos + 8 <= bytes.size()) {
        const std::string id(reinterpret_cast<const char*>(bytes.data() + pos), 4);
        const uint32_t chunk_size = wav_detail::read_le32(bytes.data() + pos + 4);
        pos += 8;

        // 文件被截断时按实际剩余字节算，避免越界；后续校验会兜底。
        const size_t avail =
            std::min<size_t>(chunk_size, bytes.size() - pos);

        if (id == "fmt ") {
            if (avail < 16) {
                error = "fmt 块不完整：" + path;
                return false;
            }
            format_tag = wav_detail::read_le16(bytes.data() + pos);
            channels = wav_detail::read_le16(bytes.data() + pos + 2);
            sample_rate = wav_detail::read_le32(bytes.data() + pos + 4);
            bits = wav_detail::read_le16(bytes.data() + pos + 14);

            // WAVE_FORMAT_EXTENSIBLE：真正格式藏在 SubFormat GUID 的前 2 字节。
            // fmt 块偏移 24 处开始是 GUID，这里只取前两字节的格式码。
            if (format_tag == 0xFFFE && avail >= 40) {
                format_tag = wav_detail::read_le16(bytes.data() + pos + 24);
            }
            have_fmt = true;
        } else if (id == "data") {
            data_offset = pos;
            data_size = avail;
            have_data = true;
        }

        if (have_fmt && have_data) {
            break;
        }
        pos += static_cast<size_t>(chunk_size) + (chunk_size & 1u);
    }

    if (!have_fmt) {
        error = "wav 缺少 fmt 块：" + path;
        return false;
    }
    if (!have_data) {
        error = "wav 缺少 data 块：" + path;
        return false;
    }
    if (channels == 0 || sample_rate == 0 || bits == 0) {
        error = "wav 头字段非法：" + path;
        return false;
    }

    const bool is_pcm16 = (format_tag == 1 && bits == 16);
    const bool is_float32 = (format_tag == 3 && bits == 32);
    if (!is_pcm16 && !is_float32) {
        error = "只支持 PCM16 / float32，当前 format=" +
                std::to_string(format_tag) + " bits=" + std::to_string(bits) +
                "：" + path;
        return false;
    }

    const size_t bytes_per_sample = bits / 8;
    const size_t frame_bytes = bytes_per_sample * channels;
    const size_t n_samples = data_size / frame_bytes;
    out.samples.resize(n_samples);
    out.sample_rate = static_cast<int>(sample_rate);
    out.channels = channels;
    out.bits_per_sample = bits;

    // 多声道只取第 0 声道；帧内偏移 0 就是第一声道。
    const unsigned char* p = bytes.data() + data_offset;
    for (size_t i = 0; i < n_samples; ++i) {
        if (is_pcm16) {
            const int16_t v = static_cast<int16_t>(
                wav_detail::read_le16(p + i * frame_bytes));
            // /32768：与 Python/soundfile 读 int16 得到 float [-1,1] 的量纲一致。
            out.samples[i] = static_cast<float>(v) / 32768.0f;
        } else {
            const unsigned char* q = p + i * frame_bytes;
            const uint32_t raw = wav_detail::read_le32(q);
            float v = 0.0f;
            std::memcpy(&v, &raw, sizeof(v));
            out.samples[i] = v;
        }
    }
    return true;
}

}  // namespace audio_ai

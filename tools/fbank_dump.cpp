/**
 * @file fbank_dump.cpp
 * @brief 前端对拍工具：raw float32 PCM -> AudioPreprocess -> raw float32 特征。
 *
 * 只做「喂数据 + 倒特征」，不读 wav、不画图，避免把容器解析/显示差异
 * 混进数值比较。输入输出都是无头部的裸 float32（小端），由
 * tools/compare_fbank.py 负责生成输入、解析输出。
 *
 * 用法：
 *   ./build/fbank_dump.out <in.f32> <out.f32> [chunk=160] [sr=16000] [n_mels=80]
 *
 * 参数含义：
 *   in.f32    原始波形，float32 [-1,1]，单声道
 *   out.f32   特征，行主序 (n_frames, n_mels) float32
 *   chunk     每次 accept_waveform 的样本数；必须与 Python 侧一致
 *   sr        采样率（Hz）
 *   n_mels    Mel 通道数
 */

#include "audio_ai_process/audio_preprocess.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace audio_ai;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "用法: " << argv[0]
                  << " <in.f32> <out.f32> [chunk=160] [sr=16000] [n_mels=80]"
                  << std::endl;
        return 2;
    }
    const std::string in_path = argv[1];
    const std::string out_path = argv[2];
    const int chunk = argc > 3 ? std::stoi(argv[3]) : 160;
    const int sample_rate = argc > 4 ? std::stoi(argv[4]) : 16000;
    const int n_mels = argc > 5 ? std::stoi(argv[5]) : 80;
    if (chunk <= 0 || sample_rate <= 0 || n_mels <= 0) {
        std::cerr << "chunk / sr / n_mels 必须为正数" << std::endl;
        return 2;
    }

    // 读裸 float32：文件长度必须是 4 的倍数。
    std::ifstream in(in_path, std::ios::binary);
    if (!in) {
        std::cerr << "打不开输入：" << in_path << std::endl;
        return 1;
    }
    in.seekg(0, std::ios::end);
    const std::streamoff bytes = in.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
        std::cerr << "输入不是 float32 数组：" << in_path << std::endl;
        return 1;
    }
    in.seekg(0, std::ios::beg);
    std::vector<float> samples(static_cast<size_t>(bytes) / sizeof(float));
    in.read(reinterpret_cast<char*>(samples.data()), bytes);
    if (!in) {
        std::cerr << "读取输入失败：" << in_path << std::endl;
        return 1;
    }

    // 参数与 Python 参考实现 / 可视化入口保持一致，tail_pad_frames=0。
    AudioPreprocess front(sample_rate, n_mels, 25.0, 10.0, 512, 0.97, 20.0, 0.0,
                          0);

    // 按固定块长喂，块边界与 Python 脚本完全相同。
    std::vector<float> feats;
    for (size_t off = 0; off < samples.size();
         off += static_cast<size_t>(chunk)) {
        const size_t n =
            std::min<size_t>(static_cast<size_t>(chunk), samples.size() - off);
        const std::vector<float> blk(samples.begin() + off,
                                     samples.begin() + off + n);
        const LogFbankFeatures out = front.accept_waveform(blk);
        feats.insert(feats.end(), out.data.begin(), out.data.end());
    }
    const LogFbankFeatures tail = front.input_finished();
    feats.insert(feats.end(), tail.data.begin(), tail.data.end());

    std::ofstream os(out_path, std::ios::binary);
    if (!os) {
        std::cerr << "打不开输出：" << out_path << std::endl;
        return 1;
    }
    os.write(reinterpret_cast<const char*>(feats.data()),
             static_cast<std::streamsize>(feats.size() * sizeof(float)));
    if (!os) {
        std::cerr << "写输出失败：" << out_path << std::endl;
        return 1;
    }

    std::cout << "C++ 前端：样本 " << samples.size() << "，帧 "
              << feats.size() / static_cast<size_t>(n_mels) << "，块长 " << chunk
              << std::endl;
    return 0;
}

/**
 * @file test_wav_kws.cpp
 * @brief 离线 wav 关键词唤醒入口（对应 Python test_wav_kws_mini.py）。
 *
 * ============================================================
 * 这份代码是干什么的
 * ============================================================
 * 把 audio_ai_process/ 下的前端与 KWS 解码器按顺序接起来，读 wav 文件从头
 * 解到尾，打印命中的关键词、首 token 时间与平均声学概率：
 *
 *   wav 文件 → load_wav（取第一声道，float [-1,1]）
 *            → AudioPreprocess（波形 → (n, 80) LogFBank）
 *            → KwsDecoder（三个 ONNX 会话 + trie 约束的束搜索）
 *            → 命中的关键词列表
 *
 * 三个类只靠特征矩阵相连：入口只做「参数 + 数据源 + 装配」，流水线细节不写
 * 在这里。与 test_wav_asr.cpp 不同，本入口**不开窗口、不画语谱图**：KWS 的
 * 输出是离散触发事件，没有逐帧文本可滚动显示。
 *
 * ============================================================
 * 为什么要在 wav 末尾补真静音
 * ============================================================
 * 触发条件要求关键词后面至少出现 2 个 blank 输出帧（80 ms）。wav 里关键词
 * 后面可能紧跟着别的词，没有静音就永远不触发。真机麦克风是连续流，这个问题
 * 被掩盖了；离线验收必须在末尾补一段**样本域**的静音（0 值，不是特征域 0），
 * 默认补 0.5 秒，可用 --tail-silence 调整。
 *
 * ============================================================
 * 与 Python test_wav_kws_mini.py 的差异
 * ============================================================
 *   - 支持一次传多个文件或目录，逐个解完再汇总（同 test_wav_asr.cpp）；
 *   - --dump-hits 把命中写成 TSV，供后续与 Python 版对拍；
 *   - --score / --threshold / --block / --tail-silence 可覆盖默认值；
 *   - --cuda 尝试 CUDA EP，失败自动回退 CPU；
 *   - --debug 记录被 ac_threshold 挡住的候选（调 #threshold 的依据）。
 *
 * ============================================================
 * 调用顺序（对照 main 里的编号）
 * ============================================================
 *   1) 参数                模型路径、块长、末尾静音、期望表（全部写死）
 *   2) 命令行与输入         解析开关，把目录展开成 .wav 列表
 *   3) ONNX Runtime        环境 / session 选项 / 可选 CUDA EP
 *      3.1 session 选项     3.2 可选 CUDA EP
 *      3.3 建三个 session + 装配 KwsDecoder        3.4 打印模型契约
 *      3.5 打开命中导出文件
 *   4) 逐个 wav            读文件 → 补静音 → 前端 → 束搜索 → 期望检查
 *   5) 流式不变性          换块长重跑第一条 wav，命中结果必须一致
 *   6) 汇总                所有文件的命中、encoder 调用次数、通过与否
 *
 * ============================================================
 * 用法（项目根目录）
 * ============================================================
 *   ./build/test_wav_kws.out
 *       跑权重包自带的 test_wavs/ 全部 9 条 wav，并检查期望集合
 *   ./build/test_wav_kws.out path/to/a.wav --block 1600
 *       指定文件 / 目录，换喂入块长
 *   ./build/test_wav_kws.out --dump-hits /tmp/kws_hits.tsv
 *       导出命中，格式：文件名 <TAB> 关键词 <TAB> 首token时间 <TAB> avg_prob
 *   ./build/test_wav_kws.out --score 2.0 --threshold 0.3 --debug
 *       覆盖 boosting score / 触发阈值，并打印被阈值挡住的候选
 */

#include "audio_ai_process/audio_preprocess.h"
#include "audio_ai_process/kws_decode.h"
#include "audio_ai_process/text_utils.h"
#include "tools/wav_reader.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace audio_ai;

#ifndef HK_PROJECT_DIR
#define HK_PROJECT_DIR "."
#endif

namespace {

// ============================================================
// 一、命令行数值解析
// ============================================================

// 解析整数参数；空串、非数字、带尾随字符都返回 false，由调用方打印用法。
bool parse_int_arg(const char* text, int* out) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    try {
        size_t used = 0;
        const long long value = std::stoll(text, &used);
        if (used != std::string(text).size() ||
            value < std::numeric_limits<int>::min() ||
            value > std::numeric_limits<int>::max()) {
            return false;
        }
        *out = static_cast<int>(value);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// 解析浮点参数；语义与 parse_int_arg 相同。
bool parse_float_arg(const char* text, float* out) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    try {
        size_t used = 0;
        const float value = std::stof(text, &used);
        if (used != std::string(text).size()) {
            return false;
        }
        *out = value;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// 解析双精度参数（--tail-silence 用）。
bool parse_double_arg(const char* text, double* out) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    try {
        size_t used = 0;
        const double value = std::stod(text, &used);
        if (used != std::string(text).size()) {
            return false;
        }
        *out = value;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// ============================================================
// 二、单条 wav 的解码
// ============================================================

/// 一次解码的完整结果（主流程与不变性复测共用）。
struct RunResult {
    std::vector<KeywordHit> hits;
    int chunks = 0;          // encoder 调用次数
    int64_t enc_frames = 0;  // encoder 输出帧总数
    int64_t samples = 0;     // 前端喂入样本数（含末尾补的静音）
    int64_t frames = 0;      // 前端产出特征帧数
};

// 把一条 wav 按 block 样本切块喂完（末尾补 tail_silence_s 秒真静音），
// 返回命中的关键词与统计。调用前必须已经 decoder.reset()。
RunResult run_one(KwsDecoder& decoder, const std::vector<float>& samples,
                  int sample_rate, int n_mels, int block,
                  double tail_silence_s) {
    // KWS 不使用 input_finished()：不补特征域 0，只在样本域补真静音。
    AudioPreprocess front(sample_rate, n_mels, 25.0, 10.0, 512, 0.97, 20.0,
                          0.0, 0);

    std::vector<float> padded = samples;
    if (tail_silence_s > 0.0) {
        const size_t n_tail =
            static_cast<size_t>(tail_silence_s * sample_rate);
        padded.resize(padded.size() + n_tail, 0.0f);
    }

    RunResult result;
    for (size_t fed = 0; fed < padded.size();) {
        const size_t n = std::min<size_t>(
            static_cast<size_t>(block), padded.size() - fed);
        const std::vector<float> chunk(padded.begin() + fed,
                                       padded.begin() + fed + n);
        fed += n;

        const LogFbankFeatures feats = front.accept_waveform(chunk);
        std::vector<KeywordHit> part = decoder.accept_features(feats);
        result.hits.insert(result.hits.end(), part.begin(), part.end());
    }

    result.chunks = decoder.n_chunks();
    result.enc_frames = decoder.n_enc_frames();
    result.samples = front.n_samples();
    result.frames = front.n_frames();
    return result;
}

// 命中签名：关键词 + 首 token 时间（精确到 10 ms）。用于块长不变性比较。
std::vector<std::string> signature(const std::vector<KeywordHit>& hits) {
    std::vector<std::string> out;
    out.reserve(hits.size());
    for (const KeywordHit& hit : hits) {
        out.push_back(hit.keyword + "@" +
                      std::to_string(
                          std::lround(hit.start_time * 100.0f)));
    }
    return out;
}

// 把签名列表打成一行的形式，便于两条结果直接对照。
std::string describe_signature(const std::vector<std::string>& sig) {
    std::string out = "[";
    for (size_t i = 0; i < sig.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += sig[i];
    }
    out += "]";
    return out;
}

// ============================================================
// 三、期望命中集合
// ============================================================

// 每条 wav 期望命中的关键词（原文来自 keywords.txt 的 @ 后缀）。
// zh_0~2 是负样本（不许误触发），zh_3~6 各含两个关键词。
const std::map<std::string, std::set<std::string>> kExpected = {
    {"en_0.wav", {"LIGHT_UP"}},
    {"en_1.wav", {"LOVELY_CHILD"}},
    {"zh_0.wav", {}},
    {"zh_1.wav", {}},
    {"zh_2.wav", {}},
    {"zh_3.wav", {"文森特卡索", "法国"}},
    {"zh_4.wav", {"蒋友伯", "女儿"}},
    {"zh_5.wav", {"周望军", "落实"}},
    {"zh_6.wav", {"朱丽楠", "见面会"}},
};

// 核对一条 wav 的命中集合；不在 kExpected 里的自定义文件直接视为通过。
// 返回 false 时已经打印缺少 / 多出的关键词。
bool check_expected(const std::string& name,
                    const std::vector<KeywordHit>& hits) {
    const auto it = kExpected.find(name);
    if (it == kExpected.end()) {
        return true;
    }

    std::set<std::string> got;
    for (const KeywordHit& hit : hits) {
        got.insert(hit.keyword);
    }
    if (got == it->second) {
        return true;
    }

    for (const std::string& want : it->second) {
        if (got.find(want) == got.end()) {
            std::printf("          缺少期望的关键词：%s\n", want.c_str());
        }
    }
    for (const std::string& extra : got) {
        if (it->second.find(extra) == it->second.end()) {
            std::printf("          出现期望之外的关键词：%s\n",
                        extra.c_str());
        }
    }
    std::printf("          FAIL\n");
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    // —— 一、参数（模型路径、块长、末尾静音、期望表全部写死在 main）——
    // 1.1 KWS 权重：默认 chunk-8（每 160 ms 推进一次，唤醒延迟低于 chunk-16）。
    //     encoder / joiner 是 int8，decoder 保持 fp32；换权重只改 kws_dir。
    const std::string project_dir = HK_PROJECT_DIR;
    const std::string kws_dir =
        project_dir +
        "/weight/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20";
    const std::string encoder_path = join_path(
        kws_dir, "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx");
    const std::string decoder_path = join_path(
        kws_dir, "decoder-epoch-13-avg-2-chunk-8-left-64.onnx");
    const std::string joiner_path = join_path(
        kws_dir, "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx");
    const std::string tokens_path = join_path(kws_dir, "tokens.txt");
    const std::string default_wav_dir = join_path(kws_dir, "test_wavs");
    const std::string default_keywords =
        join_path(default_wav_dir, "keywords.txt");

    // 1.2 默认解码参数：与 Python test_wav_kws_mini.py 口径一致。
    const int kDefaultBlock = 160;          // 160 样本 = 10 ms @16 kHz
    const double kDefaultTailSilence = 0.5; // 末尾补真静音（秒）
    const float kDefaultScore = 1.0f;       // 关键词 boosting score
    const float kDefaultThreshold = 0.25f;  // 触发阈值
    const int kMaxActivePaths = 4;          // 束宽，C++ 默认 4
    const int kNumTrailingBlanks = 1;       // 触发前至少 2 个 blank 帧
    const float kResetAfterSilence = 1.5f;  // 静音软重置（秒）

    // —— 二、命令行与输入展开 ——
    // 2.1 解析开关；其余位置参数当作 wav 文件或目录（可以传多个）。
    std::vector<std::string> inputs;
    std::string keywords_path = default_keywords;
    std::string dump_hits_path;
    float keywords_score = kDefaultScore;
    float keywords_threshold = kDefaultThreshold;
    int block = kDefaultBlock;
    double tail_silence = kDefaultTailSilence;
    bool use_cuda = false;
    bool debug = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--keywords") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--keywords 需要一个文件路径\n");
                return 2;
            }
            keywords_path = argv[++i];
        } else if (arg == "--score") {
            if (i + 1 >= argc ||
                !parse_float_arg(argv[i + 1], &keywords_score)) {
                std::fprintf(stderr, "--score 需要合法浮点数\n");
                return 2;
            }
            ++i;
        } else if (arg == "--threshold") {
            if (i + 1 >= argc ||
                !parse_float_arg(argv[i + 1], &keywords_threshold)) {
                std::fprintf(stderr, "--threshold 需要合法浮点数\n");
                return 2;
            }
            ++i;
        } else if (arg == "--block") {
            if (i + 1 >= argc || !parse_int_arg(argv[i + 1], &block) ||
                block <= 0) {
                std::fprintf(stderr, "--block 需要正整数\n");
                return 2;
            }
            ++i;
        } else if (arg == "--tail-silence") {
            if (i + 1 >= argc ||
                !parse_double_arg(argv[i + 1], &tail_silence) ||
                tail_silence < 0.0) {
                std::fprintf(stderr, "--tail-silence 需要非负浮点数\n");
                return 2;
            }
            ++i;
        } else if (arg == "--dump-hits") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--dump-hits 需要一个文件路径\n");
                return 2;
            }
            dump_hits_path = argv[++i];
        } else if (arg == "--cuda") {
            use_cuda = true;
        } else if (arg == "--debug") {
            debug = true;
        } else if (!arg.empty() && arg[0] == '-') {
            std::fprintf(stderr, "未知参数：%s\n", arg.c_str());
            std::fprintf(
                stderr,
                "用法：%s [--keywords 文件] [--score 1.0] [--threshold 0.25] "
                "[--block 160] [--tail-silence 0.5] [--dump-hits 文件] "
                "[--cuda] [--debug] [wav 或目录 ...]\n",
                argv[0]);
            return 2;
        } else {
            inputs.push_back(arg);
        }
    }
    if (inputs.empty()) {
        inputs.push_back(default_wav_dir);  // 不收参数时跑权重包自带的 test_wavs/
    }

    // 2.2 目录展开成 .wav 列表（text_utils::list_files，按路径字典序），
    //     普通文件原样加入；目录里没有 wav 时不算错误，最后统一检查。
    std::vector<std::string> wav_paths;
    for (const std::string& path : inputs) {
        if (is_directory(path)) {
            const std::vector<std::string> found = list_files(path, ".wav");
            wav_paths.insert(wav_paths.end(), found.begin(), found.end());
        } else {
            wav_paths.push_back(path);
        }
    }
    if (wav_paths.empty()) {
        std::fprintf(stderr, "没有找到 wav 输入：%s\n", inputs.front().c_str());
        return 1;
    }

    // —— 三、ONNX Runtime 环境与会话 ——
    // 3.1 session 选项：多个 chunk 之间本来就串行，intra / inter 各 1 线程；
    //     日志降到 WARNING，避免加载时刷屏。
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test_wav_kws");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 3.2 可选 CUDA EP：注册失败只打印并回退 CPU，不中断解码。
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

    std::cout << std::string(74, '=') << "\n"
              << "mini 离线 wav KWS —— 文件 -> 特征 -> 命中的关键词\n"
              << std::string(74, '=') << "\n"
              << "权重  : " << kws_dir << "\n"
              << "onnx  : encoder=" << base_name(encoder_path)
              << "  decoder=" << base_name(decoder_path)
              << "  joiner=" << base_name(joiner_path) << "\n"
              << "输入  : " << wav_paths.size() << " 个 wav\n"
              << "执行器: "
              << (use_cuda
                      ? (cuda_active ? "CUDA" : "CPU（CUDA 注册失败，已回退）")
                      : "CPU")
              << "\n\n";

    // 3.3 建三个 session 并装配 KwsDecoder。
    //     「模型从哪来、跑在哪块硬件上」是入口的决定，解码类只接收现成的
    //     session（与 Python build_ort_sessions 的分工一致）。
    Ort::Session encoder(env, encoder_path.c_str(), session_options);
    Ort::Session decoder_session(env, decoder_path.c_str(), session_options);
    Ort::Session joiner(env, joiner_path.c_str(), session_options);
    KwsDecoder decoder(encoder, decoder_session, joiner, tokens_path,
                       keywords_path, kMaxActivePaths, kNumTrailingBlanks,
                       keywords_score, keywords_threshold, kResetAfterSilence,
                       16000, debug);
    const KwsDecoder::Info info = decoder.info();

    // 3.4 打印模型契约与关键词列表。
    std::cout << "模型  : T=" << info.T
              << "  decode_chunk_len=" << info.decode_chunk_len
              << "  context_size=" << info.context_size
              << "  vocab=" << info.vocab_size
              << "  状态张量=" << info.n_states
              << "  n_mels=" << info.n_mels
              << "  blank_id=" << info.blank_id
              << "  unk_id=" << info.unk_id
              << "  帧长=" << std::lround(info.frame_shift_s * 1e3)
              << "ms\n"
              << "关键词: " << info.n_keywords << " 条  ";
    for (size_t i = 0; i < decoder.keywords().size(); ++i) {
        if (i > 0) {
            std::cout << " / ";
        }
        std::cout << decoder.keywords()[i];
    }
    std::cout << "\n\n";

    // 3.5 命中导出文件：每行「文件名 <TAB> 关键词 <TAB> 首token时间 <TAB>
    //     avg_prob」；没有命中的文件不写行，B3 对拍时按空集处理。
    std::ofstream dump_out;
    if (!dump_hits_path.empty()) {
        dump_out.open(dump_hits_path, std::ios::out | std::ios::trunc);
        if (!dump_out) {
            std::fprintf(stderr, "打不开命中导出文件：%s\n",
                         dump_hits_path.c_str());
            return 1;
        }
    }

    // —— 四、逐个 wav 解码 ——
    struct Result {
        std::string name;
        std::string path;
        double duration = 0.0;
        RunResult run;
        bool ok = true;
    };
    std::vector<Result> results;
    bool all_ok = true;

    std::cout << std::string(74, '-') << "\n";
    for (const std::string& wav_path : wav_paths) {
        // 4.1 读 wav：多声道只取第一声道。采样率不对时只提示、不悄悄重采样
        //     —— 喂进错误的数据结果只会更糟（同 Python load_wav）。
        WavData wav;
        std::string error;
        if (!load_wav(wav_path, wav, error)) {
            std::cerr << "[KWS] " << error << "\n";
            all_ok = false;
            continue;
        }
        if (wav.sample_rate != 16000) {
            std::printf("%s: 采样率 %d != 16000，跳过\n",
                        base_name(wav_path).c_str(), wav.sample_rate);
            all_ok = false;
            continue;
        }

        const std::string name = base_name(wav_path);
        const double duration =
            static_cast<double>(wav.samples.size()) / wav.sample_rate;

        // 4.2 每个 wav 必须 reset()：beam、encoder 状态、特征尾巴都跨块存活，
        //     不重置会把上一条音频的上下文带进来。
        decoder.reset();
        Result result;
        result.name = name;
        result.path = wav_path;
        result.duration = duration;
        result.run = run_one(decoder, wav.samples, wav.sample_rate,
                             info.n_mels, block, tail_silence);

        // 4.3 打印命中：关键词@首token时间(平均概率)；没有命中就写「未命中」。
        std::printf("%-9s %5.2fs  ", name.c_str(), duration);
        if (result.run.hits.empty()) {
            std::printf("未命中");
        } else {
            for (size_t i = 0; i < result.run.hits.size(); ++i) {
                const KeywordHit& hit = result.run.hits[i];
                if (i > 0) {
                    std::printf("  ");
                }
                std::printf("%s@%.2fs(p=%.2f)", hit.keyword.c_str(),
                            hit.start_time, hit.avg_prob);
            }
        }
        std::printf("\n");

        // 4.4 期望集合检查：负样本期望空集，多出一个词也算 FAIL。
        result.ok = check_expected(name, result.run.hits);
        if (!result.ok) {
            all_ok = false;
        }

        // 4.5 命中导出。
        if (dump_out) {
            for (const KeywordHit& hit : result.run.hits) {
                dump_out << name << '\t' << hit.keyword << '\t'
                         << std::fixed << std::setprecision(3)
                         << hit.start_time << '\t' << hit.avg_prob << '\n';
            }
        }

        results.push_back(std::move(result));
    }

    // —— 五、流式不变性：换块长重跑第一条 wav ——
    // 块长只影响喂入节奏，不该影响解码结果（前端出帧与块长无关，解码按 T
    // 出块）。主流程用的块长不是 160 时，换成 160 对照。
    if (!results.empty()) {
        const Result& first = results.front();
        const int alt_block = (block == 160 ? 1600 : 160);

        WavData wav;
        std::string error;
        std::vector<KeywordHit> alt_hits;
        if (load_wav(first.path, wav, error) && wav.sample_rate == 16000) {
            decoder.reset();
            alt_hits = run_one(decoder, wav.samples, wav.sample_rate,
                               info.n_mels, alt_block, tail_silence)
                           .hits;
        } else {
            std::fprintf(stderr, "不变性复测读文件失败：%s\n", error.c_str());
        }

        const std::vector<std::string> a = signature(first.run.hits);
        const std::vector<std::string> b = signature(alt_hits);
        const bool same = a == b;
        std::printf("%s\n", std::string(74, '-').c_str());
        std::printf("流式不变性 %s：块 %d -> %s   块 %d -> %s   %s\n",
                    first.name.c_str(), block,
                    describe_signature(a).c_str(), alt_block,
                    describe_signature(b).c_str(), same ? "OK" : "FAIL");
        if (!same) {
            all_ok = false;
        }
    }

    // —— 六、汇总 ——
    // 统计口径与 Python 版一致：encoder 调用次数 / 输出帧数 / 累计触发次数。
    int total_chunks = 0;
    int64_t total_frames = 0;
    size_t total_hits = 0;
    for (const Result& result : results) {
        total_chunks += result.run.chunks;
        total_frames += result.run.enc_frames;
        total_hits += result.run.hits.size();
    }

    std::printf("%s\n", std::string(74, '-').c_str());
    std::printf("encoder 调用 %d 次 / 共 %lld 帧   累计触发 %zu 次\n",
                total_chunks, static_cast<long long>(total_frames),
                total_hits);
    std::printf("%s\n", all_ok && !results.empty() ? "OK" : "FAIL");
    return all_ok && !results.empty() ? 0 : 1;
}

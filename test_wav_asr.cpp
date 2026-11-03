/**
 * @file test_wav_asr.cpp
 * @brief 离线 wav 语音识别入口（对应 Python test_wav_asr_mini.py）。
 *
 * ============================================================
 * 这份代码是干什么的
 * ============================================================
 * 把 audio_ai_process/ 下的三个类按顺序接起来，读 wav 文件从头解到尾：
 *
 *   wav 文件 → load_wav（取第一声道，float [-1,1]）
 *            → AudioPreprocess（波形 → (n, 80) LogFBank，尾部补 100 帧 0）
 *            → AsrDecoder（三个 ONNX 会话 + 贪心搜索）→ 文本
 *            → （可选）SpectrogramImage 滚动语谱图 + 底部识别文字
 *
 * 这三个类只靠特征矩阵相连：入口只做「参数 + 数据源 + 装配」，流水线细节
 * 不写在这里。与 Python 版一样不模拟实时节奏（不按音频时长 sleep）：
 * wav 是离线场景，尽快跑完更有用；想看完整滚动过程用谱图入口。
 *
 * 与 Python test_wav_asr_mini.py 的差异：
 *   - 支持一次传多个文件或目录，逐个解完再汇总；
 *   - --no-window 只跑识别、不开窗口（批量对照 / 自动检查用）；
 *   - --cuda 尝试 CUDA EP，失败自动回退 CPU；
 *   - 底部文字优先 FreeType 渲染中文（Python 的 cv2.putText 只支持 ASCII，
 *     中文会静默变成 '?'），字体缺失时回退 cv2.putText；
 *   - 路径拼接 / 目录展开用 text_utils.h 的公共实现。
 *
 * ============================================================
 * 调用顺序（对照 main 里的编号）
 * ============================================================
 *   1) 参数                模型路径、块长、画布尺寸、字体候选（全部写死）
 *   2) 命令行与输入         解析开关，把目录展开成 .wav 列表
 *   3) ONNX Runtime        环境 / session 选项 / 可选 CUDA EP
 *      3.1 session 选项     3.2 可选 CUDA EP
 *      3.3 建三个 session + 装配 AsrDecoder        3.4 打印模型契约
 *      3.5 装配中文字体     3.6 打开 token 导出文件
 *   4) 逐个 wav            读文件 → 前端 → 解码 →（可选）语谱图 + 文字
 *      4.1 读文件  4.2 重置三类状态  4.3 打印文件信息  4.4 主循环
 *      4.5 收尾补 100 帧 0  4.6 打印统计
 *   5) 汇总                所有文件的文本、token 数、RTF
 *
 * 结束（喂完 wav 或窗口里按 q/Esc）后每个文件都要走一次收尾：补完尾部帧、
 * 并在特征尾补 100 帧 0，再把剩下的一起解掉 —— 否则最后几个字会丢。那 100
 * 帧 0 只喂解码器、不喂可视化，否则语谱图末尾会多出一块比静音还亮的假色块。
 *
 * ============================================================
 * 用法（项目根目录）
 * ============================================================
 *   ./build/test_wav_asr.out --no-window
 *       跑默认目录 test_wavs/ 下的全部 wav，只打印文本
 *   ./build/test_wav_asr.out weight/.../test_wavs/1.wav
 *       识别单个文件，带语谱图窗口
 *   ./build/test_wav_asr.out --no-window --cuda --dump-tokens /tmp/tok.txt
 *       CUDA EP + 导出 token id 供 Python 参考实现逐 token 对拍
 *
 * 按键（窗口模式）：q / Esc 停止后续文件，最终仍需按键关闭窗口。
 */

#include "audio_ai_process/asr_decode.h"
#include "audio_ai_process/audio_preprocess.h"
#include "audio_ai_process/spectrogram_image.h"
#include "audio_ai_process/text_renderer.h"
#include "audio_ai_process/text_utils.h"
#include "tools/wav_reader.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>

using namespace audio_ai;

#ifndef HK_PROJECT_DIR
#define HK_PROJECT_DIR "."
#endif

namespace {

// 路径拼接 / 文件名 / 目录展开见 text_utils.h（各入口共用）。

// 把 token id 写成一行 CSV，供 tools/compare_asr.py 与 Python 参考实现比较。
// 格式：文件名 <TAB> id,id,id...；每个 wav 一行。没开 --dump-tokens 时
// out 是空流，这里直接返回，调用方不用额外判断。
void dump_token_line(std::ofstream& out, const std::string& wav_name,
                     const std::vector<int64_t>& ids) {
    if (!out) {
        return;
    }
    out << wav_name << '\t';
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) {
            out << ',';
        }
        out << ids[i];
    }
    out << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    // —— 一、参数（模型路径、块长、画布尺寸全部写死在 main）——
    // 1.1 模型与词表：与 Python test_wav_asr_mini.py 用同一套 int8 权重包。
    //     encoder / joiner 是 int8，decoder 保持 fp32（导出时就是这么切的），
    //     正式模型目录名带 int8 后缀，换权重时只改 asr_dir 这一处。
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
    const std::string default_input = join_path(asr_dir, "test_wavs");

    // 1.2 喂入节奏与画布尺寸
    const int kBlock = 160;        // 每块 160 样本 = 10 ms @16 kHz。
                                   // 与麦克风路径的默认 block 相同，
                                   // 同段音频两条路径的结果可以直接对照。
    const int kSpecCols = 400;     // 400 列 x 10 ms = 4 秒滚动窗口
    const int kFooterH = 56;       // 底部白条，写识别文字
    const int kCalibrateFrames = 50;      // 前 50 帧（0.5 秒）标定色阶后锁死
    const cv::Size kPlotSize(1200, 300);  // 语谱图显示尺寸（宽，高）
    const int kHeaderH = 52;              // 顶部信息条
    // 窗口标题只能用 ASCII：部分 OpenCV 构建对非 ASCII 标题处理不一致。
    const std::string kWindow = "mini wav asr";

    // 1.3 中文字体：按顺序取第一个存在的文件。
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
    // 50 个字符 ≈ 600 px 远小于画布宽度；完整文本仍由 stdout / token 导出保留。
    const size_t kMaxStatusChars = 50;

    // —— 二、命令行与输入展开 ——
    // 2.1 解析开关；其余位置参数当作 wav 文件或目录（可以传多个）。
    std::vector<std::string> inputs;
    std::string dump_tokens_path;
    bool no_window = false;
    bool use_cuda = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--no-window") {
            no_window = true;
        } else if (arg == "--cuda") {
            use_cuda = true;
        } else if (arg == "--dump-tokens") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--dump-tokens 需要一个文件路径\n");
                return 2;
            }
            dump_tokens_path = argv[++i];
        } else if (!arg.empty() && arg[0] == '-') {
            std::fprintf(stderr, "未知参数：%s\n", arg.c_str());
            std::fprintf(stderr,
                         "用法：%s [--no-window] [--cuda] [--dump-tokens 文件] "
                         "[wav 或目录 ...]\n",
                         argv[0]);
            return 2;
        } else {
            inputs.push_back(arg);
        }
    }
    if (inputs.empty()) {
        inputs.push_back(default_input);  // 不收参数时跑权重包自带的 test_wavs/
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
    // 3.1 session 选项：多个 chunk 之间本来就串行，intra / inter 各 1 线程，
    //     让 onnxruntime 再开线程池只会增加调度开销（与 Python 版一致）；
    //     日志降到 WARNING，避免加载时刷屏。
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test_wav_asr");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 3.2 可选 CUDA EP：注册失败只打印并回退 CPU，不中断识别。
    //     必须用 AppendExecutionProvider_CUDA；通用 AppendExecutionProvider
    //     只支持 QNN / SNPE / XNNPACK。
    // cuda_active 记录「真的注册成功」，banner 据此打印实际执行器，
    // 避免把注册失败后的 CPU 回退误显示成在用 GPU。
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
              << "mini 离线 wav ASR —— 文件 -> 特征 -> 文字\n"
              << std::string(74, '=') << "\n"
              << "权重  : " << asr_dir << "\n"
              << "输入  : " << wav_paths.size() << " 个 wav\n"
              << "执行器: "
              << (use_cuda
                      ? (cuda_active ? "CUDA" : "CPU（CUDA 注册失败，已回退）")
                      : "CPU")
              << "\n\n";

    // 3.3 建三个 session 并装配 AsrDecoder。
    //     「模型从哪来、跑在哪块硬件上」是入口的决定，解码类只接收现成的
    //     session（与 Python build_ort_sessions 的分工一致）。
    Ort::Session encoder(env, encoder_path.c_str(), session_options);
    Ort::Session decoder_session(env, decoder_path.c_str(), session_options);
    Ort::Session joiner(env, joiner_path.c_str(), session_options);
    AsrDecoder decoder(encoder, decoder_session, joiner, tokens_path);
    const AsrDecoder::Info info = decoder.info();

    // 3.4 打印模型契约：T / decode_chunk_len / context_size / 词表 / 状态张量。
    //     n_mels 跟着模型走，后面建前端时直接用它，避免两边维度对不上。
    std::cout << "模型  : T=" << info.T
              << "  decode_chunk_len=" << info.decode_chunk_len
              << "  context_size=" << info.context_size
              << "  vocab=" << info.vocab_size
              << "  状态张量=" << info.n_states
              << "  n_mels=" << info.n_mels
              << "  blank_id=" << info.blank_id << "\n";

    // 3.5 中文字体：窗口模式才加载；失败只提示，draw_status_text 回退 ASCII。
    Utf8TextRenderer text_renderer;
    if (!no_window) {
        const FontCandidate font = find_existing_font(kCjkFontCandidates);
        const bool font_ok = !font.path.empty() &&
                             text_renderer.init(font.path, font.face_index);
        std::cout << "文字字体: "
                  << (font_ok ? font.path : std::string("未找到，回退 ASCII"))
                  << "\n";
    }
    std::cout << "\n";

    // 底部白条文字：优先 FreeType 渲染 UTF-8 中文，失败回退 ASCII putText。
    // Python 版这里用 cv2.putText（只支持 ASCII），C++ 补上中文渲染，
    // 位置与字号保持 (8, rows - 20) / 24 px 的观感。
    // 识别结果是累积假设，会越积越长；FreeType 路径按 frame.cols 算出的
    // 可用像素宽度从尾部截断（中英混排也准确），完整文本写 stdout 不受影响。
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

    // 3.6 token 导出文件（--dump-tokens）：每行「文件名 <TAB> id,id,...」，
    //     供 tools/compare_asr.py 与 Python 参考实现逐 token 对拍。
    std::ofstream dump_out;
    if (!dump_tokens_path.empty()) {
        dump_out.open(dump_tokens_path, std::ios::out | std::ios::trunc);
        if (!dump_out) {
            std::fprintf(stderr, "打不开 token 导出文件：%s\n",
                         dump_tokens_path.c_str());
            return 1;
        }
    }

    // —— 四、逐个 wav 解码 ——
    // 单个文件的统计结果；多文件时在第 5 节一次性汇总。
    struct Result {
        std::string name;        // 文件名（汇总行标签）
        std::string text;        // 完整识别结果
        double duration = 0.0;   // 音频时长（秒）
        double elapsed = 0.0;    // 解码耗时（秒）
        int64_t tokens = 0;      // 输出 token 数
        int64_t samples = 0;     // 前端喂入样本数
        int64_t frames = 0;      // 前端产出特征帧数
        int chunks = 0;          // encoder 调用次数
        int64_t enc_frames = 0;  // encoder 输出帧总数
    };
    std::vector<Result> results;
    bool quit = false;

    for (const std::string& wav_path : wav_paths) {
        if (quit) {
            break;
        }

        // 4.1 读 wav：多声道只取第一声道。采样率不对时只提示、不悄悄重采样
        //     —— 喂进错误的数据结果只会更糟，宁可明确报出来（同 Python load_wav）。
        WavData wav;
        std::string error;
        if (!load_wav(wav_path, wav, error)) {
            std::cerr << "[ASR] " << error << "\n";
            continue;
        }
        if (wav.sample_rate != 16000) {
            std::cout << "注意：wav 是 " << wav.sample_rate
                      << " Hz，模型按 16 kHz 训练，结果会不对。\n";
        }

        const double duration =
            static_cast<double>(wav.samples.size()) / wav.sample_rate;

        // 4.2 每个 wav 重置前端、解码器与画布，互不影响。
        //     解码器内部 116 个状态 + hyp + decoder_out 跨块存活，换文件必须
        //     reset()，否则上一段的上下文会带进来。
        //     前端 tail_pad_frames=100：收尾解码要用；那 100 帧特征域 0
        //     只喂解码器，不喂可视化（见 4.5）。
        AudioPreprocess front(wav.sample_rate, info.n_mels,
                              25.0, 10.0, 512, 0.97, 20.0, 0.0,
                              100);
        decoder.reset();

        // 语谱图和 AsrDecoder 一样只接收特征；前 50 帧标定色阶，底部预留
        // kFooterH 像素白条，识别文字由入口画进去。
        SpectrogramImage view(info.n_mels, kSpecCols,
                              static_cast<double>(front.frame_step()) /
                                  wav.sample_rate,
                              kDefaultLevelLo, kDefaultLevelHi,
                              kCalibrateFrames, kPlotSize, kHeaderH, kFooterH,
                              cv::COLORMAP_VIRIDIS, base_name(wav_path));

        // 4.3 打印本文件信息；识别文字随后在主循环里原地刷新。
        std::cout << "------------------------------------------------------------\n"
                  << "wav   : " << wav_path << "\n"
                  << "        " << duration << " 秒 / " << wav.samples.size()
                  << " 样本 @ " << wav.sample_rate << " Hz\n";
        std::cout.flush();

        const auto begin = std::chrono::steady_clock::now();
        std::string shown;
        cv::Mat canvas;

        // 4.4 主循环：切块 → 预处理 → 可视化 → 推理 → 打印 → 刷新窗口
        //     （对应 Python 第 4 节；C++ 这里外面还多了一层 wav 文件循环）。
        for (size_t fed = 0; fed < wav.samples.size() && !quit;) {
            // 4.4.1 取一块（末尾那块可能不满）
            const size_t n =
                std::min<size_t>(kBlock, wav.samples.size() - fed);
            std::vector<float> block(wav.samples.begin() + fed,
                                     wav.samples.begin() + fed + n);
            fed += n;

            // 4.4.2 预处理：波形 → (n, 80) LogFBank。流式出帧，返回帧数
            //       与块长无关，最后几帧留到下一块补齐。
            const LogFbankFeatures feats = front.accept_waveform(block);

            // 4.4.3 可视化：返回整幅伪彩画布；--no-window 时跳过，省渲染。
            if (!no_window) {
                canvas = view.accept_features(feats);
            }
            // 4.4.4 推理 + 后处理：贪心搜索和跨块状态都在 AsrDecoder 内部。
            const std::string text = decoder.accept_features(feats);

            // 4.4.5 打印：只在文字变化时刷新 stdout（窗口每帧都重画）；
            //       补 8 个空格擦掉上一行多出来的尾巴。
            if (text != shown) {
                shown = text;
                std::cout << "\r" << text << std::string(8, ' ')
                          << std::flush;
            }

            if (!no_window) {
                // 4.4.6 识别结果画到底部白条，再显示 + 处理按键。
                //       与 Python 版一致：每个块都重画、都 imshow，而不是只在
                //       文字变化时刷新 —— 这样语谱图和文字始终是同一帧。
                //       Python 用 cv2.putText（只支持 ASCII），这里优先用
                //       FreeType 画中文，缺少字体时回退同一套 putText。
                draw_status_text(canvas, text);
                cv::imshow(kWindow, canvas);
                const int key = cv::waitKey(1) & 0xFF;
                if (key == 27 || key == 'q') {
                    quit = true;  // 停止后续文件，当前文件的收尾照常走完
                }
            }
        }

        // 4.5 收尾：让前端补出尾部帧，并在特征尾补 100 帧 0（log 域能量 1，
        //     不是静音值），使最后不满一个 chunk 的尾段也能被解码 —— 少了
        //     这一步，最后几个字会被整段丢掉。返回值尾部带着 100 帧 0，
        //     只喂解码器；喂可视化会多一块假亮尾。
        const std::string final_text =
            decoder.accept_features(front.input_finished());
        const auto end = std::chrono::steady_clock::now();
        const double elapsed =
            std::chrono::duration<double>(end - begin).count();

        if (!no_window) {
            // 4.5.1 最后一屏按当前缓存重画（不喂补 0 特征），避免多出假亮尾；
            //       完整识别结果写到底部白条（中文字体缺失时回退 ASCII）。
            canvas = view.frame();
            draw_status_text(canvas, final_text);
            cv::imshow(kWindow, canvas);
        }

        // 4.6 token 导出 + 统计：音频时长、耗时、RTF、喂入样本 / 产出帧、
        //     encoder 调用次数与 token 数（口径与 Python 5.2 一致）。
        const std::vector<int64_t> ids = decoder.token_ids();
        dump_token_line(dump_out, base_name(wav_path), ids);

        std::cout << "\n识别结果：" << final_text << "\n"
                  << "音频 " << duration << "s   耗时 " << elapsed << "s   RTF "
                  << (duration > 0.0 ? elapsed / duration : 0.0)
                  << "（< 1 即快于实时）\n"
                  << "喂入样本 " << front.n_samples() << "   产出帧 "
                  << front.n_frames() << "\n"
                  << "encoder 调用 " << decoder.n_chunks() << " 次 / 共 "
                  << decoder.n_enc_frames() << " 帧   token " << ids.size()
                  << " 个\n\n";

        results.push_back({base_name(wav_path), final_text, duration, elapsed,
                           static_cast<int64_t>(ids.size()),
                           front.n_samples(), front.n_frames(),
                           decoder.n_chunks(), decoder.n_enc_frames()});
    }

    // —— 五、汇总（多文件时一次性打印，便于和 Python / 对拍脚本对照）——
    std::cout << std::string(74, '=') << "\n汇总\n"
              << std::string(74, '=') << "\n";
    for (const Result& r : results) {
        std::cout << r.name << "  " << r.duration << "s  " << r.tokens
                  << " token  RTF " << (r.duration > 0.0 ? r.elapsed / r.duration : 0.0)
                  << "\n  " << r.text << "\n";
    }

    if (!no_window) {
        std::cout << "\n按任意键关闭语谱图窗口……\n";
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    return results.empty() ? 1 : 0;
}

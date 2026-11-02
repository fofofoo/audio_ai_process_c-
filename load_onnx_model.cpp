/**
 * @file load_onnx_model.cpp
 * @brief 流式 Zipformer2 transducer 三模型加载自检（encoder / decoder / joiner）。
 *
 * 职责边界：
 *   - 只做「能不能加载 + 接口长什么样」：建 Ort::Session、打印输入输出与
 *     custom metadata、校验 sherpa-onnx 约定的关键字段；
 *   - 不做推理、不读 wav、不实现贪心搜索和解码；数值链路在后续入口里接；
 *   - 不改任何模型文件，失败只报告，不尝试修复。
 *
 * 模型契约（与 weight/<模型目录>/test_onnx.py 一致）：
 *   encoder : 第 0 个输入是 (1, T, 80) 的 float 特征块，其余输入是流式状态；
 *             输出第 0 个是 encoder_out，其余是更新后的状态；
 *             metadata 必须含 model_type="zipformer2"、decode_chunk_len、T。
 *   decoder : 输入是 (1, context_size) 的 token id，输出 decoder_out；
 *             metadata 必须含 context_size、vocab_size。
 *   joiner  : 输入 encoder_out + decoder_out，输出 (1, 1, vocab_size) logits。
 *
 * 失败分级：
 *   三个模型各自独立加载，单个失败记 ERROR 后继续下一个；
 *   最后统一汇总，任意一个失败或关键 metadata 缺失就返回非 0。
 *
 * 用法：
 *   ./build/load_onnx_model.out                 # 默认非 int8 模型，CPU
 *   ./build/load_onnx_model.out --cuda          # 尝试 CUDA EP（失败自动回退 CPU）
 *   ./build/load_onnx_model.out <模型目录>       # 切到 int8 目录等
 *
 * 注意：CUDA EP 需要 cuDNN 9 在 LD_LIBRARY_PATH 里，具体见 onnxruntime/README.md。
 */

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

// —— 默认模型目录 ——
// 由 CMake 注入工程根路径，保证从任意工作目录启动都能找到模型；
// 单独编译本文件时才退回当前目录。
#ifndef HK_PROJECT_DIR
#define HK_PROJECT_DIR "."
#endif

// 与 python_ref 使用同一套 sherpa-onnx 导出模型；换模型时只改这一行或走 argv。
const char* kModelDirSuffix =
    "/weight/"
    "sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-2026-06-05";

// 三个模型在目录里的文件名。int8 版本 encoder/joiner 带 .int8 后缀，
// decoder 保持原名，所以这里同时给出两种候选，由 find_model_file() 择优。
struct ModelSpec {
    const char* role;        // 打印用角色名
    const char* base_name;   // 不带扩展名、不带 .int8 的基础名
};

const ModelSpec kModelSpecs[] = {
    {"encoder", "encoder"},
    {"decoder", "decoder"},
    {"joiner", "joiner"},
};

// 单个模型的加载结果。session 为空表示加载失败。
struct ModelInfo {
    std::string role;
    std::string path;
    long long size_bytes = -1;
    double load_ms = 0.0;
    bool loaded = false;
    std::unique_ptr<Ort::Session> session;
    std::map<std::string, std::string> meta;
};

std::string join_path(const std::string& dir, const std::string& name) {
    if (dir.empty()) {
        return name;
    }
    if (dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

bool file_exists(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return in.good();
}

// 文件字节数；打不开返回 -1。只用 ifstream，避免额外引入 filesystem 库。
long long file_size_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        return -1;
    }
    return static_cast<long long>(in.tellg());
}

// 在模型目录里找 <base>.onnx，找不到再退回 <base>.int8.onnx。
// 返回空串表示两种都没有。
std::string find_model_file(const std::string& dir, const std::string& base) {
    const std::string full = join_path(dir, base + ".onnx");
    if (file_exists(full)) {
        return full;
    }
    const std::string int8 = join_path(dir, base + ".int8.onnx");
    if (file_exists(int8)) {
        return int8;
    }
    return std::string();
}

// ONNX 张量元素类型 → 可读名字。
// ORT 的枚举值在头文件里，这里只覆盖语音模型常见的几种；
// 其他类型打印数值，避免漏掉时无法排查。
const char* element_type_name(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return "float64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return "int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return "bool";
        default:
            return "other";
    }
}

// shape 转字符串。动态维在 ORT 里是 -1，符号维是负的枚举值，照实打印；
// 不臆测成具体数字，后续推理前必须按 metadata 里的 T / context_size 填。
std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
}

// 打印一个会话的输入/输出：名字、形状、元素类型、动态维数量。
// 非张量输入（序列/映射等）只打印 ONNX 类型，避免 GetTensorTypeAndShapeInfo 抛异常。
void print_session_io(const Ort::Session& session) {
    // 名字由 ORT 分配，AllocatedStringPtr 超出作用域自动归还，不能只存裸指针。
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t n_inputs = session.GetInputCount();
    const size_t n_outputs = session.GetOutputCount();

    std::printf("  输入 %zu 个：\n", n_inputs);
    for (size_t i = 0; i < n_inputs; ++i) {
        auto name = session.GetInputNameAllocated(i, allocator);
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
            const auto info = type_info.GetTensorTypeAndShapeInfo();
            const auto shape = info.GetShape();
            std::printf("    [%zu] %-24s %-8s shape=%s\n", i, name.get(),
                        element_type_name(info.GetElementType()),
                        shape_to_string(shape).c_str());
        } else {
            std::printf("    [%zu] %-24s (非张量)\n", i, name.get());
        }
    }

    std::printf("  输出 %zu 个：\n", n_outputs);
    for (size_t i = 0; i < n_outputs; ++i) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
            const auto info = type_info.GetTensorTypeAndShapeInfo();
            const auto shape = info.GetShape();
            std::printf("    [%zu] %-24s %-8s shape=%s\n", i, name.get(),
                        element_type_name(info.GetElementType()),
                        shape_to_string(shape).c_str());
        } else {
            std::printf("    [%zu] %-24s (非张量)\n", i, name.get());
        }
    }
}

// 抽取 custom metadata 为 map，方便按 key 查询与统一打印。
// 顺序按 map 的字典序，输出稳定，便于多次运行对比。
std::map<std::string, std::string> read_custom_metadata(
    const Ort::ModelMetadata& metadata, Ort::AllocatorWithDefaultOptions& allocator) {
    std::map<std::string, std::string> result;
    auto keys = metadata.GetCustomMetadataMapKeysAllocated(allocator);
    for (const auto& key : keys) {
        auto value = metadata.LookupCustomMetadataMapAllocated(key.get(), allocator);
        result.emplace(key.get(), value ? value.get() : "");
    }
    return result;
}

void print_metadata(const std::map<std::string, std::string>& meta) {
    if (meta.empty()) {
        std::printf("  custom metadata：无\n");
        return;
    }
    std::printf("  custom metadata %zu 项：\n", meta.size());
    for (const auto& kv : meta) {
        std::printf("    %-28s = %s\n", kv.first.c_str(), kv.second.c_str());
    }
}

// 从 metadata 取整数字段；缺失或解析失败返回 false，不抛异常。
bool meta_int(const std::map<std::string, std::string>& meta, const char* key,
              long long* out) {
    const auto it = meta.find(key);
    if (it == meta.end()) {
        return false;
    }
    try {
        size_t used = 0;
        const long long value = std::stoll(it->second, &used);
        if (used != it->second.size()) {
            return false;
        }
        *out = value;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// 统计 tokens.txt 行数：每行一个 token，末尾空行不计。
long long count_token_lines(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return -1;
    }
    long long count = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            ++count;
        }
    }
    return count;
}

}  // namespace

int main(int argc, char** argv) {
    // —— 一、参数 ——
    // 参数直白写在 main：默认模型目录、CUDA 开关、可选的目录覆盖。
    const std::string default_model_dir =
        std::string(HK_PROJECT_DIR) + kModelDirSuffix;
    std::string model_dir = default_model_dir;
    bool use_cuda = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--cuda") {
            use_cuda = true;
        } else if (!arg.empty() && arg[0] == '-') {
            std::fprintf(stderr, "未知参数：%s\n", arg.c_str());
            std::fprintf(stderr,
                         "用法：%s [--cuda] [模型目录]\n", argv[0]);
            return 2;
        } else {
            model_dir = arg;
        }
    }

    std::printf("============================================================\n");
    std::printf("ONNX Runtime 模型加载自检\n");
    std::printf("============================================================\n");
    std::printf("ORT 版本     : %s\n", Ort::GetVersionString().c_str());
    std::printf("模型目录     : %s\n", model_dir.c_str());
    std::printf("执行提供者   : %s\n", use_cuda ? "CUDA（失败回退 CPU）" : "CPU");

    // 可用 provider 列表反映的是「库编译进来了哪些」，不等于「本次会话真正用上」；
    // CUDA 能否创建还要看 cuDNN 9 是否可加载，所以下面逐模型加载时以异常为准。
    const std::vector<std::string> providers = Ort::GetAvailableProviders();
    std::printf("可用 provider：");
    for (size_t i = 0; i < providers.size(); ++i) {
        std::printf("%s%s", i == 0 ? "" : ", ", providers[i].c_str());
    }
    std::printf("\n\n");

    // —— 二、创建环境与会话选项 ——
    // 线程数与 Python 参考实现保持一致（各 1），后续对拍时不受线程调度影响。
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "load_onnx_model");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (use_cuda) {
        try {
            // 用 CUDA 专用接口而不是通用的 AppendExecutionProvider("CUDA")：
            // 后者在当前版本只支持 QNN / SNPE / XNNPACK。device_id=0 为默认卡。
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            std::printf("CUDA EP 已注册（device_id=0）\n\n");
        } catch (const std::exception& e) {
            // 注册失败不影响加载自检：继续用 CPU，把原因打出来供排查。
            // 常见原因：没设 LD_LIBRARY_PATH 指向 cuDNN 9，或 cuDNN 版本不匹配。
            std::printf("CUDA EP 注册失败，回退 CPU：%s\n\n", e.what());
        }
    }

    // —— 三、逐个加载模型 ——
    std::vector<ModelInfo> models;
    models.reserve(3);

    for (const ModelSpec& spec : kModelSpecs) {
        ModelInfo info;
        info.role = spec.role;
        info.path = find_model_file(model_dir, spec.base_name);

        std::printf("------------------------------------------------------------\n");
        std::printf("[%s]\n", spec.role);

        if (info.path.empty()) {
            std::printf("  ERROR：目录里找不到 %s.onnx / %s.int8.onnx\n",
                        spec.base_name, spec.base_name);
            models.push_back(std::move(info));
            continue;
        }

        info.size_bytes = file_size_bytes(info.path);
        std::printf("  文件：%s\n", info.path.c_str());
        std::printf("  大小：%.1f MB\n",
                    static_cast<double>(info.size_bytes) / (1024.0 * 1024.0));

        try {
            const auto begin = std::chrono::steady_clock::now();
            info.session = std::make_unique<Ort::Session>(env, info.path.c_str(),
                                                          session_options);
            const auto end = std::chrono::steady_clock::now();
            info.load_ms =
                std::chrono::duration<double, std::milli>(end - begin).count();
            info.loaded = true;

            std::printf("  加载成功：%.0f ms\n", info.load_ms);
            print_session_io(*info.session);

            Ort::AllocatorWithDefaultOptions allocator;
            info.meta = read_custom_metadata(info.session->GetModelMetadata(),
                                             allocator);
            print_metadata(info.meta);
        } catch (const std::exception& e) {
            std::printf("  ERROR：加载失败：%s\n", e.what());
        }

        models.push_back(std::move(info));
    }

    // —— 四、契约校验与汇总 ——
    // 只校验后续推理真正依赖的字段：模型类型、chunk 帧数、帧移、上下文长度、词表大小。
    std::printf("\n============================================================\n");
    std::printf("契约校验\n");
    std::printf("============================================================\n");

    bool all_ok = true;
    for (const ModelInfo& model : models) {
        if (!model.loaded) {
            std::printf("  %-8s : FAIL（未加载）\n", model.role.c_str());
            all_ok = false;
            continue;
        }
        std::printf("  %-8s : OK  load=%.0f ms\n", model.role.c_str(),
                    model.load_ms);
    }

    // encoder：模型类型 + 流式分块参数
    const ModelInfo* encoder = nullptr;
    const ModelInfo* decoder = nullptr;
    for (const ModelInfo& model : models) {
        if (model.role == "encoder") encoder = &model;
        if (model.role == "decoder") decoder = &model;
    }

    long long chunk_len = -1;
    long long chunk_frames = -1;
    long long context_size = -1;
    long long vocab_size = -1;

    if (encoder != nullptr && encoder->loaded) {
        const auto it = encoder->meta.find("model_type");
        if (it == encoder->meta.end()) {
            std::printf("  encoder  : FAIL 缺少 metadata model_type\n");
            all_ok = false;
        } else if (it->second != "zipformer2") {
            std::printf("  encoder  : FAIL model_type=%s，期望 zipformer2\n",
                        it->second.c_str());
            all_ok = false;
        } else {
            std::printf("  encoder  : model_type=zipformer2\n");
        }

        if (!meta_int(encoder->meta, "decode_chunk_len", &chunk_len)) {
            std::printf("  encoder  : FAIL 缺少或无法解析 decode_chunk_len\n");
            all_ok = false;
        }
        if (!meta_int(encoder->meta, "T", &chunk_frames)) {
            std::printf("  encoder  : FAIL 缺少或无法解析 T\n");
            all_ok = false;
        }
        const size_t n_states =
            encoder->session->GetInputCount() > 0
                ? encoder->session->GetInputCount() - 1
                : 0;
        std::printf("  状态张量 : %zu 个（encoder 输入数 - 1）\n", n_states);
    }

    if (decoder != nullptr && decoder->loaded) {
        if (!meta_int(decoder->meta, "context_size", &context_size)) {
            std::printf("  decoder  : FAIL 缺少或无法解析 context_size\n");
            all_ok = false;
        }
        if (!meta_int(decoder->meta, "vocab_size", &vocab_size)) {
            std::printf("  decoder  : FAIL 缺少或无法解析 vocab_size\n");
            all_ok = false;
        }
    }

    // tokens.txt 行数应等于词表大小；多一行/少一行都说明模型目录不配套。
    const std::string tokens_path = join_path(model_dir, "tokens.txt");
    const long long token_lines = count_token_lines(tokens_path);
    if (token_lines < 0) {
        std::printf("  tokens   : WARN 找不到 %s\n", tokens_path.c_str());
    } else if (vocab_size >= 0 && token_lines != vocab_size) {
        std::printf("  tokens   : FAIL tokens.txt %lld 行，vocab_size=%lld\n",
                    token_lines, vocab_size);
        all_ok = false;
    } else {
        std::printf("  tokens   : %lld 行\n", token_lines);
    }

    std::printf("\n------------------------------------------------------------\n");
    std::printf("流式分块参数\n");
    std::printf("------------------------------------------------------------\n");
    std::printf("  T (每块帧数)          : %lld\n", chunk_frames);
    std::printf("  decode_chunk_len(帧)  : %lld（每块前进的帧数）\n", chunk_len);
    std::printf("  音频时长/块           : %.2f 秒（T × 10 ms）\n",
                chunk_frames > 0 ? static_cast<double>(chunk_frames) * 0.01 : 0.0);
    std::printf("  块间重叠              : %lld 帧（%.0f ms）\n",
                chunk_frames > 0 && chunk_len > 0 ? chunk_frames - chunk_len : 0,
                chunk_frames > 0 && chunk_len > 0
                    ? static_cast<double>(chunk_frames - chunk_len) * 10.0
                    : 0.0);
    std::printf("  context_size          : %lld\n", context_size);
    std::printf("  vocab_size            : %lld\n", vocab_size);

    std::printf("\n结果：%s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}

#pragma once

/**
 * @file text_utils.h
 * @brief 通用小工具：UTF-8 编解码、字符分类、系统路径处理。
 *
 * 职责边界：
 *   - 只收纳跨入口复用的小工具函数，不承载业务逻辑，也不认识 ASR / 词表；
 *   - 不依赖 ROS / OpenCV / ONNX Runtime，实现只用标准库与 POSIX
 *     （dirent / sys/stat），因此任意目标都可以直接链接；
 *   - 全部是无状态纯函数，可在任意线程调用。
 *
 * 来源与口径：
 *   文本部分原先是 asr_decode.cpp 匿名 namespace 里的本地工具，配合
 *   detokenize() 实现 Python 版的三条去空格规则；路径部分原先是各入口
 *   匿名 namespace 里重复的实现（join_path 在 ASR / 模型加载入口各一份，
 *   base_name 在 ASR / 谱图入口各一份）。集中到一处避免口径漂移。
 *
 * 字符区间与 Python 参考实现保持一致：
 *   CJK       —— 统一表意文字 U+4E00–U+9FFF + 扩展 A U+3400–U+4DBF；
 *   全角      —— CJK 标点 U+3000–U+303F + 全角形 U+FF00–U+FFEF。
 */

#include <cstdint>
#include <string>
#include <vector>

namespace audio_ai {

/// 是否 CJK 字符：统一表意文字 + 扩展 A。
bool is_cjk(uint32_t cp);

/// 是否全角字符：CJK 标点 + 全角形。
bool is_fullwidth(uint32_t cp);

/// 是否空白：ASCII space/tab/LF/CR + 全角空格 U+3000（对齐 Python 的 \s）。
bool is_space(uint32_t cp);

/**
 * @brief UTF-8 字符串 → Unicode 码点序列。
 *
 * 非法字节写 U+FFFD 并前进 1 字节，保证不会死循环；模型词表是合法 UTF-8，
 * 这条路径只做防御。
 */
std::vector<uint32_t> utf8_decode(const std::string& text);

/// 把单个 Unicode 码点按 UTF-8 编码追加到 out 末尾（1~4 字节）。
void utf8_append(std::string& out, uint32_t cp);

// —— 系统路径 ——
// 各入口拼模型路径、取文件名、展开目录时共用，口径见实现文件。

/// 拼接目录与文件名：dir 为空直接返回 name，仅在缺少 '/' 时补一个。
std::string join_path(const std::string& dir, const std::string& name);

/// 取路径最后一段（兼容 Unix 的 '/' 与 Windows 的 '\'）。
std::string base_name(const std::string& path);

/// 路径是否为目录；不存在、无权限或本身是文件都返回 false。
bool is_directory(const std::string& path);

/// 列出目录下指定后缀的文件（含目录前缀），按路径字典序排序。
/// suffix 需带点（如 ".wav"），只匹配长度大于后缀的文件名。
std::vector<std::string> list_files(const std::string& dir,
                                    const std::string& suffix);

}  // namespace audio_ai

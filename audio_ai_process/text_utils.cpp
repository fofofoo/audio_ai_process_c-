/**
 * @file text_utils.cpp
 * @brief text_utils.h 的实现：UTF-8 编解码 + 字符分类 + 系统路径处理。
 *
 * 文本口径全部照抄 python_ref/src/asr_decode.py 的 detokenize 规则，
 * 改动字符区间会导致中英文空格还原结果与 Python 版不一致；
 * 路径函数是各入口原先的匿名 namespace 实现，行为保持不变。
 * 本文件只依赖标准库与 POSIX（dirent / sys/stat），不引入 ROS / OpenCV / ORT。
 */

#include "audio_ai_process/text_utils.h"

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>

namespace audio_ai {

// 是否 CJK：统一表意文字 + 扩展 A。
bool is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||   // 一-鿿
           (cp >= 0x3400 && cp <= 0x4DBF);     // 㐀-䶿
}

// 是否全角：CJK 标点 + 全角形。
bool is_fullwidth(uint32_t cp) {
    return (cp >= 0x3000 && cp <= 0x303F) ||   // 　-〿
           (cp >= 0xFF00 && cp <= 0xFFEF);     // ＀-￯
}

// Python 的 \s 在这里只可能由 ▁ 替换出的 ASCII 空格产生；
// U+3000（全角空格）也按空白处理，与 Python 的 unicode \s 对齐。
bool is_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x3000;
}

// 逐字节扫描：按首字节的高位判断 1~4 字节序列，拼出码点后前进对应长度。
// 任何不完整/非法序列都记 U+FFFD 并只前进 1 字节，保证必然推进。
std::vector<uint32_t> utf8_decode(const std::string& text) {
    std::vector<uint32_t> out;
    out.reserve(text.size());
    const size_t n = text.size();
    for (size_t i = 0; i < n;) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        uint32_t cp = 0xFFFD;
        size_t len = 1;
        if (c < 0x80) {
            // 0xxxxxxx：ASCII 单字节。
            cp = c;
        } else if ((c >> 5) == 0x6 && i + 1 < n) {
            // 110xxxxx 10xxxxxx：2 字节。
            cp = (static_cast<uint32_t>(c & 0x1F) << 6) |
                 (static_cast<unsigned char>(text[i + 1]) & 0x3F);
            len = 2;
        } else if ((c >> 4) == 0xE && i + 2 < n) {
            // 1110xxxx 10xxxxxx 10xxxxxx：3 字节（CJK 走这里）。
            cp = (static_cast<uint32_t>(c & 0x0F) << 12) |
                 ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
                 (static_cast<unsigned char>(text[i + 2]) & 0x3F);
            len = 3;
        } else if ((c >> 3) == 0x1E && i + 3 < n) {
            // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx：4 字节。
            cp = (static_cast<uint32_t>(c & 0x07) << 18) |
                 ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 12) |
                 ((static_cast<unsigned char>(text[i + 2]) & 0x3F) << 6) |
                 (static_cast<unsigned char>(text[i + 3]) & 0x3F);
            len = 4;
        }
        out.push_back(cp);
        i += len;
    }
    return out;
}

// 从尾部逐字节回溯，遇到「非续字节」（10xxxxxx 之外）记一个字符边界。
// 数够 max_chars 个字符就从该边界截取；不是按字节截断，中文不会被切碎。
// 非法 UTF-8 也只会让边界判断偏保守，不会越界或死循环。
std::string utf8_tail(const std::string& text, size_t max_chars) {
    if (max_chars == 0) return std::string();
    size_t count = 0;
    size_t i = text.size();
    while (i > 0) {
        --i;
        if ((static_cast<unsigned char>(text[i]) & 0xC0) != 0x80) {
            ++count;
            if (count == max_chars) return text.substr(i);
        }
    }
    return text;  // 字符数不足 max_chars，整串返回
}

// 按码点范围选择 1~4 字节模板依次写出，与 utf8_decode 互为逆过程。
void utf8_append(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        // 0xxxxxxx
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        // 110xxxxx 10xxxxxx
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        // 1110xxxx 10xxxxxx 10xxxxxx
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// 拼接目录与文件名：dir 为空直接返回 name；末尾已有 '/' 就不再补。
// 只处理 Unix 风格分隔符，Windows 路径请先自行转换。
std::string join_path(const std::string& dir, const std::string& name) {
    if (dir.empty()) {
        return name;
    }
    if (dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

// 取路径最后一段：同时认 Unix 的 '/' 与 Windows 的 '\'。
// 没有分隔符说明传入的本身就是文件名。
std::string base_name(const std::string& path) {
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

// 用 stat 判断目录。失败（不存在、无权限）按「不是目录」处理，
// 调用方通常据此回退到「按单个文件处理」。
bool is_directory(const std::string& path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// 遍历目录只取指定后缀的普通文件名（不递归）。
// 名字长度必须大于后缀，避免把文件名本身恰好等于 ".wav" 的条目算进来；
// 结果按完整路径字典序排序，保证 0.wav、1.wav、10.wav 顺序稳定。
std::vector<std::string> list_files(const std::string& dir,
                                    const std::string& suffix) {
    std::vector<std::string> files;
    DIR* dp = ::opendir(dir.c_str());
    if (dp == nullptr) {
        return files;
    }
    while (dirent* entry = ::readdir(dp)) {
        const std::string name = entry->d_name;
        if (name.size() > suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) ==
                0) {
            files.push_back(join_path(dir, name));
        }
    }
    ::closedir(dp);
    std::sort(files.begin(), files.end());
    return files;
}

}  // namespace audio_ai

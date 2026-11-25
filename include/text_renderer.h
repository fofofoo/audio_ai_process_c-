#pragma once

/**
 * @file text_renderer.h
 * @brief 把 UTF-8 文本画到 OpenCV BGR 画布上（用于显示中文识别结果）。
 *
 * 职责边界：
 *   - 只做「文本 → 像素」，不认识 ASR、词表、消息或窗口；
 *   - 依赖 OpenCV core；FreeType 是可选依赖，CMake 找到时才编译真实实现；
 *   - 同一实例内部持有字体句柄，非线程安全，不要跨线程共用。
 *
 * 背景：cv::putText 只支持 ASCII，识别结果里的中文会被静默替换成 '?'。
 * 本机 OpenCV 来自 /usr/local 且不带 opencv_contrib 的 freetype 模块，
 * 因此直接用系统 FreeType 渲染，不混链发行版 libopencv_freetype。
 *
 * 与 text_utils.h 的分工：那边是纯文本（UTF-8 编解码、字符分类），不碰
 * 图形库；这边负责把文本变成像素。渲染器内部用 text_utils 的 utf8_decode。
 */

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace audio_ai {

/// 字体候选：文件路径 + .ttc 内的字面索引（简体中文通常 2，单字面字体 0）。
struct FontCandidate {
    std::string path;
    int face_index = 0;
};

/**
 * @brief 依次检查候选字体，返回第一个存在的文件。
 *
 * 这里只判断文件能否打开，不验证字体格式是否有效；真正加载由
 * Utf8TextRenderer::init() 完成。全部不存在时返回 path 为空的候选。
 */
FontCandidate find_existing_font(const std::vector<FontCandidate>& candidates);

/**
 * @brief UTF-8 文本渲染器：RAII 持有字体，把黑字按 alpha 混合到 BGR 画布。
 *
 * FreeType 不可用（构建时未找到）或字体加载失败时，ready() 返回 false，
 * draw() 不做任何事，调用方应回退 cv::putText。
 */
class Utf8TextRenderer {
public:
    Utf8TextRenderer();
    ~Utf8TextRenderer();
    Utf8TextRenderer(const Utf8TextRenderer&) = delete;
    Utf8TextRenderer& operator=(const Utf8TextRenderer&) = delete;

    /**
     * @brief 加载字体文件。
     * @param font_path   .ttf / .otf / .ttc 路径
     * @param face_index  .ttc 内的字面索引（Noto Sans CJK：2 = 简体中文）
     * @return 是否加载成功；重复调用会先释放旧字体，失败后 ready() 为 false
     */
    bool init(const std::string& font_path, int face_index);

    /// 字体是否可用；false 时 draw() 为空操作。
    bool ready() const;

    /**
     * @brief 在 org（基线左端点，与 cv::putText 语义一致）绘制一条 UTF-8 文本。
     *
     * pixel_height 是字号像素高度；逐字按 FreeType 灰度位图做 alpha 混合，
     * 超出画布的部分自动裁剪。字体不可用时直接返回。
     */
    void draw(cv::Mat& canvas, const std::string& text, const cv::Point& org,
              int pixel_height) const;

    /**
     * @brief 从尾部按像素宽度截断后绘制（用于累积文本的滑动显示）。
     *
     * 从 text 末尾往前累加字形的 advance，超过 max_width 就停，只画能放下
     * 的那段尾巴。按字形推进宽度而不是码点数计算，中英混排同样准确；
     * 截断点落在码点边界上，不会切碎中文。
     *
     * @param max_width 可用像素宽度（应为画布宽减去左右边距）
     *
     * 字体不可用、text 为空或 max_width <= 0 时不做任何事；
     * 末尾单个字形比 max_width 还宽时同样不画，避免画出界。
     */
    void draw_tail(cv::Mat& canvas, const std::string& text,
                   const cv::Point& org, int pixel_height,
                   int max_width) const;

private:
    struct Impl;  // 定义在 .cpp，FreeType 类型不进入头文件
    Impl* impl_ = nullptr;
};

}  // namespace audio_ai

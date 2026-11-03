/**
 * @file text_renderer.cpp
 * @brief text_renderer.h 的实现：FreeType 逐字渲染 UTF-8 文本。
 *
 * 构建口径：
 *   CMake 找到 FreeType 时定义 HK_HAVE_FREETYPE，这里编译真实实现；
 *   否则只编译空实现（ready() 恒 false、draw() 无操作），保证链接通过，
 *   调用方靠 ready() 判断是否回退 cv::putText。
 *
 * 数据流：
 *   UTF-8 字符串 → utf8_decode（text_utils）→ FT_Load_Char(FT_LOAD_RENDER)
 *                → 8-bit 灰度位图 → 按 alpha 混合到 BGR 画布
 */

#include "audio_ai_process/text_renderer.h"

#include "audio_ai_process/text_utils.h"

#include <fstream>

#ifdef HK_HAVE_FREETYPE
#include <ft2build.h>
#include FT_FREETYPE_H
#endif

#include <algorithm>

namespace audio_ai {

// —— 一、字体发现 ——
// 只判断文件能否打开：候选表由调用方（入口 main）按平台给出。
FontCandidate find_existing_font(const std::vector<FontCandidate>& candidates) {
    for (const FontCandidate& candidate : candidates) {
        std::ifstream in(candidate.path, std::ios::binary);
        if (in) {
            return candidate;
        }
    }
    return FontCandidate{};
}

// —— 二、渲染器状态 ——
// 真实实现持有一对 FreeType 句柄；未启用 FreeType 时是空结构，行为一致。
#ifdef HK_HAVE_FREETYPE
struct Utf8TextRenderer::Impl {
    FT_Library library = nullptr;
    FT_Face face = nullptr;

    ~Impl() {
        if (face != nullptr) {
            FT_Done_Face(face);
        }
        if (library != nullptr) {
            FT_Done_FreeType(library);
        }
    }
};
#else
struct Utf8TextRenderer::Impl {};
#endif

Utf8TextRenderer::Utf8TextRenderer() : impl_(new Impl()) {}

Utf8TextRenderer::~Utf8TextRenderer() { delete impl_; }

// 加载字体：先释放旧字体，再初始化库与字面。
// 失败时把对应句柄清空，ready() 返回 false，调用方回退 ASCII。
bool Utf8TextRenderer::init(const std::string& font_path, int face_index) {
#ifdef HK_HAVE_FREETYPE
    if (impl_->face != nullptr) {
        FT_Done_Face(impl_->face);
        impl_->face = nullptr;
    }
    // FreeType 库实例只建一次，重复 init 换字体时复用。
    if (impl_->library == nullptr && FT_Init_FreeType(&impl_->library) != 0) {
        impl_->library = nullptr;
        return false;
    }
    if (FT_New_Face(impl_->library, font_path.c_str(), face_index,
                    &impl_->face) != 0) {
        impl_->face = nullptr;
        return false;
    }
    return true;
#else
    (void)font_path;
    (void)face_index;
    return false;
#endif
}

bool Utf8TextRenderer::ready() const {
#ifdef HK_HAVE_FREETYPE
    return impl_->face != nullptr;
#else
    return false;
#endif
}

#ifdef HK_HAVE_FREETYPE
namespace {

// 把 8-bit 灰度位图按 alpha 混合到 BGR 画布上（黑色前景、白底）。
// 越界部分自动裁掉，不要求调用方保证文字不出画布。
void blit_gray(cv::Mat& canvas, const FT_Bitmap& bitmap, int x0, int y0) {
    if (bitmap.pixel_mode != FT_PIXEL_MODE_GRAY || bitmap.width == 0 ||
        bitmap.rows == 0) {
        return;
    }
    const int x_begin = std::max(0, x0);
    const int y_begin = std::max(0, y0);
    const int x_end =
        std::min(canvas.cols, x0 + static_cast<int>(bitmap.width));
    const int y_end =
        std::min(canvas.rows, y0 + static_cast<int>(bitmap.rows));
    for (int y = y_begin; y < y_end; ++y) {
        const unsigned char* src =
            bitmap.buffer + static_cast<size_t>(y - y0) * bitmap.pitch;
        cv::Vec3b* dst = canvas.ptr<cv::Vec3b>(y);
        for (int x = x_begin; x < x_end; ++x) {
            const int alpha = src[x - x0];
            if (alpha == 0) {
                continue;
            }
            // 黑字覆盖白底：dst = dst * (1 - alpha/255)。
            const int keep = 255 - alpha;
            cv::Vec3b& p = dst[x];
            p[0] = static_cast<unsigned char>((p[0] * keep) / 255);
            p[1] = static_cast<unsigned char>((p[1] * keep) / 255);
            p[2] = static_cast<unsigned char>((p[2] * keep) / 255);
        }
    }
}

// 逐码点渲染到画布：调用方已设置字号。
// 参数：face 已加载的字面；canvas BGR 画布；cps 码点序列；org 基线左端点。
// 前置条件：face 非空、字号已用 FT_Set_Pixel_Sizes 设好。
// 返回值：无。越界部分由 blit_gray 裁掉，pen_x 按 advance（26.6 定点）前进。
void render_codepoints(FT_Face face, cv::Mat& canvas,
                       const std::vector<uint32_t>& cps,
                       const cv::Point& org) {
    int pen_x = org.x;
    for (uint32_t cp : cps) {
        if (FT_Load_Char(face, cp, FT_LOAD_RENDER) != 0) {
            continue;
        }
        const FT_GlyphSlot slot = face->glyph;
        blit_gray(canvas, slot->bitmap, pen_x + slot->bitmap_left,
                  org.y - slot->bitmap_top);
        pen_x += static_cast<int>(slot->advance.x >> 6);
    }
}

}  // namespace
#endif

// 逐码点渲染：设置字号 → 取灰度位图 → 混合 → 按 advance 前进。
// FT_LOAD_RENDER 让 FreeType 直接输出位图，省一次手动渲染。
void Utf8TextRenderer::draw(cv::Mat& canvas, const std::string& text,
                            const cv::Point& org, int pixel_height) const {
#ifdef HK_HAVE_FREETYPE
    if (impl_->face == nullptr || text.empty()) {
        return;
    }
    if (FT_Set_Pixel_Sizes(impl_->face, 0,
                           static_cast<FT_UInt>(pixel_height)) != 0) {
        return;
    }

    render_codepoints(impl_->face, canvas, utf8_decode(text), org);
#else
    (void)canvas;
    (void)text;
    (void)org;
    (void)pixel_height;
#endif
}

// 按像素宽度从尾部截断后绘制：从最后一个码点往前累加 advance，
// 直到再加一个就超过 max_width；截断点天然落在码点边界上。
// 这样累积的识别文本始终显示最新的一段，且不会被画布右边缘裁掉。
void Utf8TextRenderer::draw_tail(cv::Mat& canvas, const std::string& text,
                                 const cv::Point& org, int pixel_height,
                                 int max_width) const {
#ifdef HK_HAVE_FREETYPE
    if (impl_->face == nullptr || text.empty() || max_width <= 0) {
        return;
    }
    if (FT_Set_Pixel_Sizes(impl_->face, 0,
                           static_cast<FT_UInt>(pixel_height)) != 0) {
        return;
    }

    const std::vector<uint32_t> cps = utf8_decode(text);
    size_t begin = cps.size();
    int width = 0;
    while (begin > 0) {
        // FT_LOAD_DEFAULT 只取 advance，不生成位图，比 FT_LOAD_RENDER 便宜。
        if (FT_Load_Char(impl_->face, cps[begin - 1], FT_LOAD_DEFAULT) != 0) {
            --begin;
            continue;
        }
        const int advance =
            static_cast<int>(impl_->face->glyph->advance.x >> 6);
        if (width + advance > max_width) {
            break;
        }
        width += advance;
        --begin;
    }
    // begin == cps.size() 表示一个字都放不下（窗口过窄），整段不画。
    if (begin >= cps.size()) {
        return;
    }
    render_codepoints(impl_->face, canvas,
                      std::vector<uint32_t>(cps.begin() + begin, cps.end()),
                      org);
#else
    (void)canvas;
    (void)text;
    (void)org;
    (void)pixel_height;
    (void)max_width;
#endif
}

}  // namespace audio_ai

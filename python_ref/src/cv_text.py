# -*- coding: utf-8 -*-
"""在 OpenCV 画布上绘制单行文本的辅助函数。

============================================================
为什么需要它
============================================================
cv2.putText 既不换行也不裁剪：文本一长，超出画布右侧的部分直接画不出来，
而且不报错。语谱图画布宽度固定（默认 400 列，显示约 1200 px），实时识别
说上十几个字就会撞到这条边。

这里按「像素宽度」量出能放下的最长**尾部**子串 —— 新的留着，旧的裁掉，
相当于一个按像素滚动的窗口。

为什么从尾部裁、不从头部裁：
    实时场景里用户最关心「刚说的是什么」。从左裁，看到的反而是最早的
    几个字，越说越看不到新内容。

============================================================
边界与约定
============================================================
- 只负责「放得下」，不负责渲染。cv2.putText 只支持 ASCII，中文会画成
  '?'，可读输出仍以控制台为准。
- 测量与绘制必须用同一组 font_face / font_scale / thickness，否则算出来的
  宽度和实际笔画对不上。调用方把同一个值两边都传一次即可。
- 截断点若落在空格上，顺手去掉，免得行首空一格。

依赖：只用 OpenCV。
"""

import cv2


def fit_text_tail(text, max_width,
                  font_face=cv2.FONT_HERSHEY_SIMPLEX,
                  font_scale=0.7, thickness=1):
    """返回 text 的尾部子串，绘制宽度不超过 max_width 像素。

    参数：
        text       —— 完整文本，可以为空
        max_width  —— 可用的水平像素数（画布宽度减去左右边距）
        其余参数   —— 与 cv2.putText 保持一致，测量和绘制要用同一组

    返回：
        能完整放下的最长后缀；连最后一个字符都放不下时返回空串。
    """
    if not text or max_width <= 0:
        return ""

    # 先整体量一次：放得下就不折腾
    (width, _), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
    if width <= max_width:
        return text

    # 从尾部往前扩，找到还能放下的最长后缀。
    # 线性扫描足够：一屏文本通常不超过一百个字符，getTextSize 也很轻。
    start = len(text)
    for i in range(len(text) - 1, -1, -1):
        (width, _), _ = cv2.getTextSize(
            text[i:], font_face, font_scale, thickness)
        if width > max_width:
            break
        start = i

    # 截断点落在空格上时去掉，避免行首空一格
    return text[start:].lstrip()

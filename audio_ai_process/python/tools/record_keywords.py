# -*- coding: utf-8 -*-
"""录制唤醒词样本，供调 #threshold / :score 使用。

============================================================
这份脚本是干什么的
============================================================
真机联调时每次唤醒只吐一行 avg_prob，样本量太少，没法判断参数该往哪调。
这个脚本把「说唤醒词」这件事单独拎出来：按回车录一条，存成 16 kHz 单声道
wav，攒够数量后交给 tools/eval_keywords.py 批量评估。

    recordings/
        小智同学/
            01.wav  02.wav ...
        小智小智/
            01.wav ...
        meta.txt            ← 录制时间、设备、参数，便于回溯

唤醒词清单直接读 keywords/kws_keywords_raw.txt，不用另外维护一份。

============================================================
为什么是「按回车一条一条录」
============================================================
能量门限自动切分看起来更省事，但阈值要另外标定，录到一半失手就全乱。
逐条录制节奏可控、失败可重录，样本量大起来之后这点麻烦可以忽略。

每条样本按两次回车：

    第一次  开始等起音
    第二次  说完了按回车，立刻停止录制

另有两条兜底：连续静音 2 秒自动停（忘了按回车时），或最长 6 秒。

用法::

    python -I tools/record_keywords.py                  # 默认每词 10 遍
    python -I tools/record_keywords.py --repeats 5
    python -I tools/record_keywords.py --seconds 6      # 放宽最长时长上限
    python -I tools/record_keywords.py --only 小智同学   # 只录某个词
    python -I tools/record_keywords.py --device 61      # 指定输入设备

录音要点：自然语速、正常音量、离麦 20~30 cm，别刻意拖长或加重。
录得太「标准」反而和真实使用场景不一致，调出来的阈值会偏保守。
"""

import argparse
import os
import sys
import time
import wave

import numpy as np

# -I 隔离模式下脚本目录不进 sys.path，项目根要手动插
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.mic_client import MicClient

# ---- 路径与默认参数 ----
RAW_TXT = os.path.join(_ROOT, "keywords", "kws_keywords_raw.txt")
REC_DIR = os.path.join(_ROOT, "recordings")

SAMPLE_RATE = 16000          # KWS 模型固定 16 kHz
REC_SECONDS = 6.0            # 最长录多久（保护上限）
REC_TAIL_SILENCE = 2.0       # 兜底：连续静音多久自动停（主控是用户按回车）
REC_STOP_PAD = 0.4           # 按下回车后多录一点静音尾巴，给 KWS 留 blank
REC_START_TIMEOUT = 4.0      # 等起音最多等多久，超时算这一遍作废
N_REPEATS = 10                # 每个词录几遍
MIC_DEVICE = 3              # 与 test_mic_kws_asr.py 保持一致
# 录音缓冲要够大：用户按回车前可能发呆十几秒，deque 满了会丢块。
# 1024 块 x 10 ms = 10.24 s；仍按「取走前清空」处理，不影响正片。
MAX_BLOCKS = 2048


def setup_stdout_utf8():
    """把 stdout/stderr 切成 UTF-8（Windows 控制台默认 GBK）。"""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:      # 极老的 Python 才没有 reconfigure
            pass


def load_phrases(path):
    """从 raw 文件取原文列（`:` / `#` / `@` 之前的部分），跳过注释行。

    与 tools/gen_keywords.py 的 parse_raw 同一条规则，但这里只关心原文，
    不校验参数 —— 录制和生成是两件事。
    """
    phrases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = []
            for w in line.split():
                if w[0] in ":#@":
                    break
                parts.append(w)
            phrase = " ".join(parts).strip()
            if phrase:
                phrases.append(phrase)
    if not phrases:
        raise SystemExit("raw 文件里没有有效关键词：%s" % path)
    return phrases


def _rms(blk):
    """一块波形的均方根能量。"""
    return float(np.sqrt(np.mean(np.asarray(blk, dtype=np.float64) ** 2)))


def _enter_pressed():
    """非阻塞检查是否按了回车（Windows 控制台）。

    用 msvcrt 直接读控制台缓冲，不碰 stdin —— 录音循环是阻塞的，
    input() 没法在「边说边读」时用。非 Windows 直接返回 False，
    退化成静音自动停。
    """
    try:
        import msvcrt
    except ImportError:
        return False
    hit = False
    while msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            hit = True
    return hit


def record_once(mic, max_seconds, timeout=REC_START_TIMEOUT,
                tail_silence=REC_TAIL_SILENCE):
    """等用户开口，录到用户按回车为止，返回一维 float32 波形（没开口返回 None）。

    固定时长录音有两个坑：
        ① 按回车到真正开口之间有 0.5~1.2 秒反应时间，会被算进窗口；
        ② 不同人、不同语速说同一句话耗时差很多，定死了就会切断慢的。

    所以分三段：

        ① 先量 0.25 秒噪声底，定一个起音阈值；
        ② 等能量超过阈值（最长 timeout 秒），期间保留最近 0.3 秒前导，
           免得把起音的第一个辅音切掉；
        ③ 从起音点开始录，**用户说完按回车即停**，再多带 REC_STOP_PAD 秒
           静音尾巴。连续静音 tail_silence 秒自动停是兜底（忘了按回车），
           max_seconds 是最后一道保险。

    停止时机由人判断，不会因为句间换气被切；起音点由能量判断，
    录音长度和「多快反应过来」无关。
    """
    mic.get_last_audio()        # 丢掉按回车之前积压的数据
    time.sleep(0.10)

    # ---- ① 噪声底：0.25 秒 ----
    noise = []
    t_end = time.perf_counter() + 0.25
    while time.perf_counter() < t_end:
        blk = mic.get_last_audio()
        if blk is not None:
            noise.append(blk)
        time.sleep(0.002)
    floor = _rms(np.concatenate(noise)) if noise else 0.0
    # 3 倍底噪起步，另设 0.008 的绝对地板，防止安静环境下把呼吸声当起音
    threshold = max(0.008, floor * 3.0)

    # ---- ② 等起音 ----
    pre = []
    started = False
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        blk = mic.get_last_audio()
        if blk is None:
            time.sleep(0.002)
            continue
        pre.append(blk)
        pre = pre[-30:]         # 只留最近约 0.3 秒作为前导
        if _rms(blk) > threshold:
            started = True
            break
        time.sleep(0.002)

    if not started:
        return None

    # ---- ③ 从起音点录到用户按回车 ----
    # 主控是人：说完按回车，立刻停，再多带 REC_STOP_PAD 秒静音。
    # 兜底是连续 tail_silence 秒静音（能量低于阈值一半，避免把气声、
    # 句间换气算成说话），最后还有 max_seconds 上限。
    while _enter_pressed():
        pass                    # 吃掉按「开始」时残留的回车
    chunks = list(pre)
    silence_s = 0.0
    stop_at = None              # 按了回车之后的收尾时刻
    t0 = time.perf_counter()
    while True:
        now = time.perf_counter()
        if stop_at is not None and now >= stop_at:
            break
        if now - t0 >= max_seconds:
            break
        if stop_at is None and _enter_pressed():
            stop_at = now + REC_STOP_PAD
            print("\n  [回车] 收到，收尾中……")

        blk = mic.get_last_audio()
        if blk is None:
            time.sleep(0.002)
            continue
        chunks.append(blk)
        if _rms(blk) < threshold * 0.5:
            silence_s += len(blk) / float(mic.sample_rate)
            if silence_s >= tail_silence:
                break
        else:
            silence_s = 0.0
        time.sleep(0.002)

    if not chunks:
        return None
    return np.concatenate(chunks)


def save_wav(path, samples):
    """float32 [-1, 1] -> 16 bit PCM 单声道 wav。"""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def main():
    setup_stdout_utf8()

    ap = argparse.ArgumentParser(description="录制唤醒词样本")
    ap.add_argument("--repeats", type=int, default=N_REPEATS,
                    help="每个词录几遍（默认 %d）" % N_REPEATS)
    ap.add_argument("--seconds", type=float, default=REC_SECONDS,
                    help="最长录制时长（默认 %.1f 秒）；说完按回车即停"
                         % REC_SECONDS)
    ap.add_argument("--device", default=MIC_DEVICE,
                    help="输入设备编号或名字（默认 %s）" % MIC_DEVICE)
    ap.add_argument("--out", default=REC_DIR,
                    help="输出根目录（默认 recordings/）")
    ap.add_argument("--only", default=None,
                    help="只录指定的词（必须与 raw 里的原文完全一致）")
    args = ap.parse_args()

    phrases = load_phrases(RAW_TXT)
    if args.only:
        if args.only not in phrases:
            raise SystemExit("raw 里没有这个词：%r\n可用：%s"
                             % (args.only, " / ".join(phrases)))
        phrases = [args.only]

    print("=" * 68)
    print("录制唤醒词：%d 个词 x %d 遍（说完按回车停，最长 %.1f 秒）"
          % (len(phrases), args.repeats, args.seconds))
    print("设备：%s    输出：%s" % (args.device, args.out))
    print("=" * 68)
    print("录音要点：自然语速、正常音量、离麦 20~30 cm。")
    print("每遍两次回车：第一次开始说，说完了再按一次回车结束。\n")

    mic = MicClient(sample_rate=SAMPLE_RATE, device=args.device,
                    max_blocks=MAX_BLOCKS)
    mic.start()

    saved = []          # [(phrase, path, peak)]
    try:
        for pi, phrase in enumerate(phrases, 1):
            out_dir = os.path.join(args.out, phrase)
            os.makedirs(out_dir, exist_ok=True)

            print("-" * 68)
            print("[%d/%d] %s   输出目录：%s"
                  % (pi, len(phrases), phrase, out_dir))
            print("-" * 68)

            for r in range(1, args.repeats + 1):
                input("  按回车开始，然后直接说……")
                print("  >>> 请说：%s —— 说完再按一次回车" % phrase)

                audio = record_once(mic, args.seconds)
                if audio is None:
                    print("  !! %.0f 秒内没听到起音，这一遍作废，重来一次\n"
                          % REC_START_TIMEOUT)
                    continue

                dur = len(audio) / float(SAMPLE_RATE)
                peak = float(np.abs(audio).max())
                # 峰值过低基本等于没说话；高于 1.0 是削波（float 输入一般到不了）
                warn = ""
                if peak < 0.01:
                    warn = "  <- 峰值过低，可能没录到声音"
                elif peak >= 0.99:
                    warn = "  <- 削波，离麦远一点"
                if dur >= args.seconds - 0.05:
                    warn += "  <- 到上限了，可能没按回车"
                print("  [录到 %.2f 秒]  峰值 %.3f%s" % (dur, peak, warn))

                path = os.path.join(out_dir, "%02d.wav" % r)
                save_wav(path, audio)
                saved.append((phrase, path, peak))
                print("  [完成] %s\n" % path)
    except KeyboardInterrupt:
        print("\n\n已中断，已保存的样本保留。")
    finally:
        mic.stop()

    # ---- 写 meta.txt，方便回溯这批样本是怎么来的 ----
    if saved:
        os.makedirs(args.out, exist_ok=True)
        meta_path = os.path.join(args.out, "meta.txt")
        with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("录制时间: %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
            f.write("设备: %s\n" % args.device)
            f.write("采样率: %d Hz  单声道  16 bit\n" % SAMPLE_RATE)
            f.write("停顿结束: %.2f s  最长: %.2f s  遍数: %d\n"
                    % (REC_TAIL_SILENCE, args.seconds, args.repeats))
            f.write("\n")
            for phrase, path, peak in saved:
                f.write("%s\t%s\tpeak=%.3f\n"
                        % (phrase, os.path.basename(path), peak))

    print("=" * 68)
    print("共保存 %d 条，峰值过低 %d 条。"
          % (len(saved), sum(1 for _, _, p in saved if p < 0.01)))
    if saved:
        print("下一步：python -I tools/eval_keywords.py")
    print("=" * 68)


if __name__ == "__main__":
    main()

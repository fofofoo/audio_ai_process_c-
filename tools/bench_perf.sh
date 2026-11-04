#!/usr/bin/env bash
# ============================================================================
# bench_perf.sh —— ASR 实时性 / 吞吐基准脚本
#
# 这份脚本是干什么的：
#   把「离线 wav 识别」与「设备实时识别」两个入口自己打印的统计行抓下来，
#   换算成同一套指标（RTF / 主循环吞吐 / 丢块时长 / 启动积压），横向比较执行器
#   （CPU / CUDA）与渲染开销（有窗 / 无窗）对实时性的影响。
#   脚本只读不写：不改源码、不产出结果文件，日志失败时才保留。
#
# 设计取舍 / 关键约束：
#   - 纯 bash + awk，不依赖 Python / numpy；指标全部由入口打印的行换算，
#     不在脚本里重新计时或重实现解码，避免两边口径漂移。
#   - 离线模式用 test_wav_asr.out --no-window，按 kBlock=160 尽快喂完，
#     测的是「前端 + 三模型推理」的纯吞吐，不受设备和渲染干扰。
#   - 实时模式用 test_mic_asr.out --seconds N，测的是端到端追赶能力：
#     采集 + 前端 + 推理 +（可选）渲染，能否跟上设备的 16 kHz 实时流。
#   - CUDA 注册失败时入口自己会回退 CPU 并打印原因；脚本只如实转述 banner
#     里的执行器，不擅自判定，也不把回退当成成功。
#
# 调用顺序（对照下面编号）：
#   一、参数解析与前置检查
#   二、运行一轮并解析统计
#   三、打印对比表与结论提示
#
# 用法：
#   ./tools/bench_perf.sh                        # 离线，CPU，默认权重目录 4 条 wav
#   ./tools/bench_perf.sh --cuda                 # 离线，CPU + CUDA 对比
#   ./tools/bench_perf.sh --wav results/x.wav    # 离线，指定 wav 文件或目录
#   ./tools/bench_perf.sh --realtime 20          # 实时，设备在线，各跑 20 秒（无窗）
#   ./tools/bench_perf.sh --cuda --realtime 20   # 实时，CPU / CUDA 对比
#   ./tools/bench_perf.sh --realtime 20 --window # 实时 + 窗口，测渲染开销
#
# 指标口径（与入口统计行一一对应）：
#   RTF        = 解码耗时 / 音频时长（< 1 快于实时，越小越好）
#   主循环吞吐 = 喂入样本 / 16000 / 墙钟秒数（稳态应 ≈ 1；> 1 表示在消化积压）
#   丢块时长   = 取流丢弃样本 / 16000 秒（> 0 说明取流缓冲已经溢出）
#   解码丢弃   = 后台解码队列因满而丢弃的块数（上限 50 块，每块约 480 ms 音频）。
#                取流没丢但这项 > 0，说明推理/渲染慢于实时、丢的是识别内容；
#                它衡量「后台解码跟不上」，与取流丢块衡量的「主循环/采集跟不上」
#                是两种不同的落后，不能互相替代。
#   启动积压   = 实际转发秒数 − 墙钟秒数。入口现在先 init() 登录、模型加载
#                完才 start() 开流，这项应接近 0；若明显大于 0，说明开流早于
#                主循环，取流缓冲只有 2.56 s，超出的部分会丢——丢块要按这个
#                口径归因，不能直接当成「主循环慢」。
# ============================================================================
set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

WAV_ASR="./build/test_wav_asr.out"
MIC_ASR="./build/test_mic_asr.out"
SAMPLE_RATE=16000   # 两个入口都要求 16 kHz，丢块 / 喂入样本按它换算秒数
# 取流缓冲上限：max_stream_samples = 40960 样本 @ 16 kHz ≈ 2.56 s。
# 启动积压超过它，超出的部分在进入主循环之前就已经被丢了。
STREAM_BUFFER_SECONDS=2.56

# ---------------------------------------------------------------------------
# 一、参数解析与前置检查
# ---------------------------------------------------------------------------
MODE="offline"
USE_CUDA=0
REALTIME_SECONDS=""
WITH_WINDOW=0
WAV_INPUT=""

usage() {
    cat <<'EOF'
用法：
  ./tools/bench_perf.sh [--cuda] [--wav <文件或目录>]
  ./tools/bench_perf.sh [--cuda] --realtime <秒数> [--window]

选项：
  --cuda            额外跑一遍 CUDA EP，与 CPU 对比
  --wav <路径>      离线模式指定输入（默认走入口自带的默认权重目录）
  --realtime <秒>   实时模式：设备必须在线，按指定秒数自动收尾
  --window          实时模式带窗口（不加则 --no-window），用于对比渲染开销
  -h, --help        显示本帮助
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)     USE_CUDA=1; shift ;;
        --wav)      WAV_INPUT="${2:-}"; shift 2 ;;
        --window)   WITH_WINDOW=1; shift ;;
        --realtime)
            MODE="realtime"
            REALTIME_SECONDS="${2:-}"
            shift 2
            ;;
        -h|--help)  usage; exit 0 ;;
        *)          echo "未知参数：$1" >&2; usage; exit 2 ;;
    esac
done

if [[ "$MODE" == "realtime" ]]; then
    # 秒数必须是正数：入口的 --seconds 只接受 > 0，这里提前拦下，
    # 免得设备连上以后才发现参数错、白等一次登录。
    if ! [[ "$REALTIME_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] \
        || ! awk -v s="$REALTIME_SECONDS" 'BEGIN { exit !(s > 0) }'; then
        echo "--realtime 需要一个大于 0 的秒数" >&2
        exit 2
    fi
fi

if [[ "$MODE" == "offline" && ! -x "$WAV_ASR" ]]; then
    echo "找不到 $WAV_ASR，先执行：cmake --build build -j" >&2
    exit 1
fi
if [[ "$MODE" == "realtime" && ! -x "$MIC_ASR" ]]; then
    echo "找不到 $MIC_ASR，先执行：cmake --build build -j" >&2
    exit 1
fi

# 汇总行统一用「|」分隔，三、里按模式打印成表格。
ROWS=()

# 从 log 里取 banner 的执行器，保证打印的是入口真实状态（含回退）。
parse_executor() {
    grep -m1 '^执行器:' "$1" | sed 's/^执行器: *//' || true
}

# ---------------------------------------------------------------------------
# 二、运行一轮并解析统计
# ---------------------------------------------------------------------------

# 离线一轮：解析每个 wav 的「音频 / 耗时 / RTF」，累加得到总体 RTF。
# 参数：$1 标签  $2 是否 CUDA（1/0）
# 返回：0 成功；1 失败（打印退出码与日志头部）
run_offline() {
    local label="$1" cuda="$2"
    local log; log="$(mktemp)"
    local -a cmd=("$WAV_ASR" --no-window)
    [[ "$cuda" == "1" ]] && cmd+=("--cuda")
    [[ -n "$WAV_INPUT" ]] && cmd+=("$WAV_INPUT")

    # 先跑再取退出码：用 if ! cmd 的形式拿到的 $? 是取反后的 0，会掩盖真实原因。
    "${cmd[@]}" >"$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[离线 $label] 运行失败（退出码 $rc），日志保留在：$log" >&2
        sed -n '1,20p' "$log" >&2
        return 1
    fi

    # 每识别完一条 wav，入口打印：
    #   音频 3.24s   耗时 0.85s   RTF 0.26（< 1 即快于实时）
    # 去掉单位与中文括号，只留数值；最后按总音频 / 总耗时算总体 RTF。
    local stats
    stats="$(awk '
        /^音频 / {
            d = $2; sub(/s.*$/, "", d)
            e = $4; sub(/s.*$/, "", e)
            sum_d += d; sum_e += e; n++
        }
        END { if (n > 0) printf "%.4f %.4f %d", sum_d, sum_e, n }
    ' "$log")"

    if [[ -z "$stats" ]]; then
        echo "[离线 $label] 没解析到统计行，日志保留在：$log" >&2
        return 1
    fi

    local sum_d sum_e files
    read -r sum_d sum_e files <<<"$stats"
    ROWS+=("$label|$(parse_executor "$log")|$sum_d|$sum_e|$files")
    rm -f "$log"
    return 0
}

# 实时一轮：解析「实际转发 / 喂入样本 / 取流丢弃 / encoder 调用」。
# 参数：$1 标签  $2 是否 CUDA（1/0）  $3 是否窗口（1/0）
# 返回：0 成功；1 失败
run_realtime() {
    local label="$1" cuda="$2" window="$3"
    local log; log="$(mktemp)"
    local -a cmd=("$MIC_ASR" --seconds "$REALTIME_SECONDS")
    [[ "$window" == "1" ]] || cmd+=("--no-window")
    [[ "$cuda" == "1" ]] && cmd+=("--cuda")

    "${cmd[@]}" >"$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[实时 $label] 运行失败（退出码 $rc），日志保留在：$log" >&2
        sed -n '1,25p' "$log" >&2
        return 1
    fi

    # 收尾统计数行（格式见 test_mic_asr.cpp 6.5）：
    #   时长 31.24s   实际转发 38.18s   喂入样本 536320   产出帧 3352
    #   encoder 调用 71 次 / 共 852 帧   token 53 个
    #   取流丢弃 73600   解码丢弃 3 块（队列上限 50）   峰值幅度 1
    # 解码丢弃与取流丢弃打印在同一行、字段位置随版本可能变化，按字段名取，
    # 不按固定列号；缺失时按 0 处理（旧日志没有这段）。
    local stats
    stats="$(awk '
        BEGIN { dropped = 0; decdrop = 0 }
        /^时长 / {
            wall = $2; sub(/s.*$/, "", wall)
            fwd = $4; sub(/s.*$/, "", fwd)
            fed = $6
        }
        /^encoder 调用 / { chunks = $3 }
        /^取流丢弃 / {
            dropped = $2
            for (i = 1; i <= NF; ++i)
                if ($i == "解码丢弃") decdrop = $(i + 1)
        }
        END {
            if (fwd != "")
                printf "%.4f %.4f %d %d %d %d", wall, fwd, fed, dropped, chunks, decdrop
        }
    ' "$log")"

    if [[ -z "$stats" ]]; then
        echo "[实时 $label] 没解析到统计行，日志保留在：$log" >&2
        return 1
    fi

    local wall fwd fed dropped chunks decdrop
    read -r wall fwd fed dropped chunks decdrop <<<"$stats"
    ROWS+=("$label|$(parse_executor "$log")|$wall|$fwd|$fed|$dropped|$chunks|$decdrop")
    rm -f "$log"
    return 0
}

echo "工程目录：$PROJECT_DIR"
echo "模式    ：$MODE$([[ "$USE_CUDA" == "1" ]] && echo "（CPU + CUDA 对比）" || echo "（仅 CPU）")"
echo

if [[ "$MODE" == "offline" ]]; then
    run_offline "CPU" 0 || exit 1
    if [[ "$USE_CUDA" == "1" ]]; then
        run_offline "CUDA" 1 || exit 1
    fi
else
    if [[ "$WITH_WINDOW" == "1" ]]; then
        run_realtime "CPU 有窗" 0 1 || exit 1
        [[ "$USE_CUDA" == "1" ]] && { run_realtime "CUDA 有窗" 1 1 || exit 1; }
    else
        run_realtime "CPU 无窗" 0 0 || exit 1
        [[ "$USE_CUDA" == "1" ]] && { run_realtime "CUDA 无窗" 1 0 || exit 1; }
    fi
fi

# ---------------------------------------------------------------------------
# 三、打印对比表与结论提示
# ---------------------------------------------------------------------------
echo "$(printf '=%.0s' {1..74})"

if [[ "$MODE" == "offline" ]]; then
    printf '%-4s %-26s %6s %9s %9s %8s  %s\n' \
        "运行" "执行器" "文件数" "音频(s)" "解码(s)" "RTF" "相对实时"
    for row in "${ROWS[@]}"; do
        IFS='|' read -r label exec sum_d sum_e files <<<"$row"
        awk -v label="$label" -v exec="$exec" -v f="$files" \
            -v d="$sum_d" -v e="$sum_e" '
            BEGIN {
                rtf = (d > 0) ? e / d : 0
                speed = (rtf > 0) ? 1 / rtf : 0
                printf "%-4s %-26s %6d %9.2f %9.2f %8.4f  %.2f 倍\n",
                       label, exec, f, d, e, rtf, speed
            }'
    done

    # 两条都跑过时给一句直接结论：RTF 越小越好。
    # CUDA 注册失败时入口会回退 CPU，此时第二行和第一行本质同构，
    # 比较出来的「快 1.00 倍」只会误导，改成明确提示。
    IFS='|' read -r _ exec_second _ <<<"${ROWS[1]:-}"
    if [[ ${#ROWS[@]} -ge 2 && "$exec_second" == *回退* ]]; then
        echo "注意：CUDA 未生效（$exec_second），第二行实际仍跑 CPU，性能不可比。"
    elif [[ ${#ROWS[@]} -ge 2 ]]; then
        read -r name0 rtf0 <<<"$(awk -F'|' '
            { rtf = ($3 > 0) ? $4 / $3 : 0; printf "%s %.4f", $1, rtf }' \
            <<<"${ROWS[0]}")"
        read -r name1 rtf1 <<<"$(awk -F'|' '
            { rtf = ($3 > 0) ? $4 / $3 : 0; printf "%s %.4f", $1, rtf }' \
            <<<"${ROWS[1]}")"
        awk -v a="$name0" -v ra="$rtf0" -v b="$name1" -v rb="$rtf1" '
            BEGIN {
                if (rb <= 0 || ra <= 0) exit
                if (ra < rb)
                    printf "结论：%s 更快（RTF %.4f vs %.4f，约 %.2f 倍）\n",
                           a, ra, rb, rb / ra
                else
                    printf "结论：%s 更快（RTF %.4f vs %.4f，约 %.2f 倍）\n",
                           b, rb, ra, ra / rb
            }'
    fi
else
    printf '%-10s %-26s %8s %8s %10s %7s %9s %11s %8s %10s\n' \
        "运行" "执行器" "墙钟(s)" "转发(s)" "处理音频(s)" "吞吐" \
        "丢块(s)" "启动积压(s)" "encoder" "解码丢弃"
    for row in "${ROWS[@]}"; do
        IFS='|' read -r label exec wall fwd fed dropped chunks decdrop <<<"$row"
        awk -v label="$label" -v exec="$exec" -v wall="$wall" -v fwd="$fwd" \
            -v fed="$fed" -v dropped="$dropped" -v chunks="$chunks" \
            -v decdrop="$decdrop" -v sr="$SAMPLE_RATE" '
            BEGIN {
                processed = fed / sr
                rate = (wall > 0) ? processed / wall : 0
                lost = dropped / sr
                gap = fwd - wall
                printf "%-10s %-26s %8.2f %8.2f %10.2f %7.2f %9.2f %11.2f %8d %10d\n",
                       label, exec, wall, fwd, processed, rate, lost, gap, chunks, decdrop
            }'
    done

    # 丢块归因，避免把「开流早于主循环时到达的音频」误判成主循环慢：
    #   启动阶段的预期丢块 = max(0, 启动积压 − 2.56 s)：
    #   缓冲能留下 2.56 s，剩下的在进入主循环前就已被淘汰。
    #   入口已改为模型加载后才 start()，正常情况下启动积压与这项预期都应为 0。
    #   1) 吞吐 < 0.95 且没有启动积压 → 主循环持续跟不上；
    #   2) 丢块明显多于启动阶段的预期值 → 启动丢一部分，之后还在持续丢；
    #   3) 丢块 ≈ 启动阶段的预期值 → 主循环跟得上，丢块只发生在启动阶段；
    #   4) 其余 → 未出现积压。
    # 解码已挪到后台线程，取流不丢不代表识别完整：后台队列（上限 50 块）
    # 满时会丢最旧的块，单独提示，不能混进取流丢块的口径里。
    for row in "${ROWS[@]}"; do
        IFS='|' read -r label exec wall fwd fed dropped chunks decdrop <<<"$row"
        # 格式串写成一行：Jetson 自带的 mawk 对跨行相邻字符串拼接解析不稳，
        # 会报 "syntax error at or near ,"（gawk 则能过），这里不依赖该用法。
        awk -v label="$label" -v wall="$wall" -v fwd="$fwd" -v fed="$fed" \
            -v dropped="$dropped" -v decdrop="$decdrop" -v sr="$SAMPLE_RATE" \
            -v cap="$STREAM_BUFFER_SECONDS" '
            BEGIN {
                rate = (wall > 0) ? (fed / sr) / wall : 0
                lost = dropped / sr
                gap = fwd - wall
                expect = (gap > cap) ? gap - cap : 0
                if (decdrop > 0)
                    printf "提示：%s 后台解码队列丢弃 %d 块（上限 50，约 %.1f s 音频未参与识别），推理吞吐低于采集。\n", label, decdrop, decdrop * 0.48
                if (rate < 0.95 && gap < 1.0)
                    printf "提示：%s 主循环吞吐 %.2f < 1，持续跟不上采集；解码已在后台，优先减少渲染/前端开销并核对解码队列。\n", label, rate
                else if (lost > expect + 1.0)
                    printf "提示：%s 主循环吞吐 %.2f；启动阶段预期丢 %.2f s，实际丢 %.2f s，多出的 %.2f s 是主循环期间持续丢的。\n", label, rate, expect, lost, lost - expect
                else if (lost > 0 && gap > 0)
                    printf "提示：%s 主循环吞吐 %.2f，跟得上；丢块 %.2f s 来自启动阶段（开流早于主循环，缓冲上限 %.2f s）。\n", label, rate, lost, cap
                else
                    printf "提示：%s 主循环吞吐 %.2f，未出现积压。\n", label, rate
            }'
    done
fi
echo "$(printf '=%.0s' {1..74})"

# KWS 待机 CPU 占用实测计划

- 状态：计划待执行（2026-11-06 起草，计划执行人：本人，时间：下周）
- 目标：给出「KWS 待机态 vs ASR 识别态」的 CPU 占用对比数据，填掉周报里的
  `kws 的 cpu 占用 xxx 相比 asr xxxx`
- 相关文档：
  - [KWS 接入计划与进度](kws_plan_status.md)
  - [BlockWorker 异步处理机制](audio_ai_block_worker.md)
  - [ASR 有界状态与转写落盘](asr_bounded_state_plan.md)

---

## 一、要回答的三个问题

1. **KWS 待机时占多少 CPU**：设备持续有声音、但没有任何唤醒词时，单进程的
   CPU 占用（以「单核 = 100%」计）。
2. **ASR 识别时占多少 CPU**：同一段音频、同一前端节奏下，切到 ASR 模式的占用。
3. **比值与渲染占比**：ASR / KWS 的 CPU 倍数；以及窗口渲染（语谱图 +
   FreeType）在这两项里各占多少。

只有第 3 项能把「双模型同时驻留」的成本拆开，避免把渲染开销误算成解码开销。

---

## 二、度量口径（必须统一，否则数字不可比）

| 项 | 口径 | 说明 |
|---|---|---|
| CPU 指标 | `/proc/<pid>/stat` 的 `utime + stime` 差分 | 单核满载 = 100%；多核机器不满一核时也是几十个百分点 |
| 采样窗口 | 开流后 **预热 10 s** 再采样 **20 s** | 预热期排除模型加载、色阶标定、首包抖动 |
| 重复次数 | 每个场景 **3 次**，取**中位数** | 单次受系统抖动影响大，报中位数并记录三次原始值 |
| 音频源 | **同一段 60 s 音频**（外放视频 / 有声书） | KWS 与 ASR 必须听同一段，前端与取流成本才一致 |
| 窗口 | 有窗 / 无窗各测一遍 | 差值即渲染成本 |
| 平台 | Jetson（主） + x86（次） | 周报要的是 Jetson 数据；x86 用来交叉验证趋势 |
| 执行器 | 只用 CPU | CUDA 在 Jetson 上更慢，且与结论无关 |

**基线要一起测**：只开语谱图、不接任何解码器的入口（`test_spectrogram_cv_mic.out`）
给出「取流 + 前端 + 渲染」的地板值。`KWS − 基线` 才是 KWS 解码本身的成本，
`ASR − 基线` 同理。没有基线时，两边的渲染开销会被重复计入。

**注意语谱图入口没有 `--no-window`**，只能拿到「前端 + 渲染」的合计值（记为 A1）。
因此**渲染成本改用同一入口关窗 / 开窗之差来反推**（B1 − B0，C1 − C0 交叉验证），
再拿 A1 减去它近似出「纯前端地板」F。这一步是近似，写结论时要标明。

---

## 三、测试矩阵

| # | 场景 | 入口与参数 | 期望观察 |
|---|---|---|---|
| A1 | 纯前端基线（只能带窗） | `test_spectrogram_cv_mic.out 40` | 取流 + 前端 + 渲染的合计值；40 s 后自动存图退出 |
| B0 | KWS 待机（无窗） | `test_mic_kws_asr.out --no-window --seconds 40` | 唤醒 0 次；KWS encoder 持续调用 |
| B1 | KWS 待机（有窗） | `test_mic_kws_asr.out --seconds 40` | 同上，CPU 更高 |
| C0 | ASR 识别（无窗） | `test_mic_kws_asr.out --initial-asr --no-window --seconds 40` | ASR encoder 持续调用、token 持续增长 |
| C1 | ASR 识别（有窗） | `test_mic_kws_asr.out --initial-asr --seconds 40` | 同上 |

注意：

- B0 / B1 用的音频里**不能含唤醒词**（否则会切进 ASR，测的就不是待机）。
  执行前先看 `keywords/kws_keywords.txt`，确认所选视频没有相近发音；
  或临时用 `--keywords` 指向一个只含生僻词的临时文件。
- C0 / C1 必须让音频**持续有人声**，且**中间不能有 10 秒以上静音**：ASR 在静音时
  每帧都是 blank、不刷新 decoder，CPU 偏低；更糟的是空闲满 10 s 会**自动切回 KWS**，
  后半段测的就不是识别态了。跑完核对统计行的「ASR：encoder 调用」次数在涨、
  「切回关键词模式」为 0。
- 每个场景结束都记录统计行里的「唤醒 / 切回 / 取流丢弃 / 解码丢弃」，
  丢块不为 0 的那次结果作废重测。

---

## 四、执行步骤

### 4.1 准备

```bash
cd /home/liyun/code/AI/audio_ai_process
cmake --build build -j                     # 确认二进制是最新的
pgrep -af "test_mic|test_spectrogram_cv_mic|save_audio_wav"   # 确认没人占用设备
DISPLAY=:0 xrandr >/dev/null 2>&1 && echo "DISPLAY 可用"      # 有窗场景用
```

设备语音通道同一时刻只能被一个进程占用；上一轮进程没退干净时
`StartVoiceCom` 会报 error 29。**不要 kill 旁人启动的进程**，先确认是谁的。

### 4.2 采样函数（复制到当前 shell）

```bash
# 用法：cpu_sample <pid> <秒数>
# 输出：进程在该窗口内的 CPU 占用（单核 = 100%）
cpu_sample() {
    local pid="$1" secs="$2" hz a b
    hz=$(getconf CLK_TCK)
    read -r -a s < <(awk '{print $14, $15}' "/proc/$pid/stat")
    a=$(( s[0] + s[1] ))
    sleep "$secs"
    read -r -a s < <(awk '{print $14, $15}' "/proc/$pid/stat")
    b=$(( s[0] + s[1] ))
    awk -v d=$(( b - a )) -v hz="$hz" -v s="$secs" \
        'BEGIN { printf "%.2f\n", d / hz / s * 100 }'
}
```

说明：`/proc/<pid>/stat` 的 14、15 字段是 utime / stime（单位 jiffies）。
进程名里有空格时会错位，但本工程二进制名不含空格，可以放心用。
装了 `sysstat` 的话也可以用 `pidstat -p <pid> 20 1` 交叉核对。

### 4.3 单场景流程（以 B0 为例）

```bash
# 1) 启动入口，日志单独存
./build/test_mic_kws_asr.out --no-window --seconds 40 > /tmp/cpu_b0_run1.log 2>&1 &

# 2) 等它打印「识别结果（Ctrl+C」再开始采样（此时已开流）
until grep -q "识别结果（Ctrl+C" /tmp/cpu_b0_run1.log; do sleep 0.5; done

# 3) 预热 10 s（让色阶/首包/模型缓存稳定）
sleep 10

# 4) 采样 20 s
PID=$(pgrep -f "test_mic_kws_asr.out --no-window --seconds 40" | head -1)
cpu_sample "$PID" 20

# 5) 等它自然结束（--seconds 40），核对统计行
grep -E "^时长|^ASR：|^KWS：|^唤醒|^取流丢弃" /tmp/cpu_b0_run1.log
```

每个场景重复 3 次，把 3 个 CPU 值和中位数填进第五节表格。
如果某次出现唤醒 / 丢块，该次作废，换音频或重来。

### 4.4 有窗场景

把 `--no-window` 去掉即可，其余不变。窗口必须在**真的显示出来**的显示器上
（WSLg 下用 `DISPLAY=:0`），最小化窗口会改变渲染开销，不要最小化。
有窗场景到点后停在最后一屏、**统计行要按键之后才打印**，所以 4.3 第 5 步的
`grep` 之前先在窗口按任意键（或直接用 `q` 手动结束，采样已拿到就不影响结果）。

### 4.5 纯前端基线 A1

```bash
./build/test_spectrogram_cv_mic.out 40 > /tmp/cpu_a1_run1.log 2>&1 &
sleep 3                                     # 它是轮询型入口，没有「识别结果」横幅
PID=$(pgrep -f "test_spectrogram_cv_mic.out 40" | head -1)
sleep 7                                     # 省掉预热：总共等约 10 s 再采样
cpu_sample "$PID" 20
wait                                        # 40 s 后自动存图退出
ls -l results/spectrogram_mic.png
```

---

## 五、记录表（执行时填写）

### 5.1 x86（WSL2，交叉验证）

| # | 场景 | 平台 | 窗口 | 第 1 次 | 第 2 次 | 第 3 次 | 中位数 | 唤醒 | 丢块 |
|---|---|---|---|---|---|---|---|---|---|
| A1 | 纯前端 | x86 | 有（仅此一档） | | | | | — | |
| B0 | KWS 待机 | x86 | 无 | | | | | | |
| B1 | KWS 待机 | x86 | 有 | | | | | | |
| C0 | ASR 识别 | x86 | 无 | | | | | | |
| C1 | ASR 识别 | x86 | 有 | | | | | | |

### 5.2 Jetson Orin NX（周报主数据）

| # | 场景 | 平台 | 窗口 | 第 1 次 | 第 2 次 | 第 3 次 | 中位数 | 唤醒 | 丢块 |
|---|---|---|---|---|---|---|---|---|---|
| A1 | 纯前端 | Jetson | 有（仅此一档） | | | | | — | |
| B0 | KWS 待机 | Jetson | 无 | | | | | | |
| B1 | KWS 待机 | Jetson | 有 | | | | | | |
| C0 | ASR 识别 | Jetson | 无 | | | | | | |
| C1 | ASR 识别 | Jetson | 有 | | | | | | |

### 5.3 派生结论（写周报用）

| 指标 | 计算方式 | 值 |
|---|---|---|
| 渲染成本（主口径） | B1 − B0（用 C1 − C0 交叉验证） | |
| 纯前端地板 F（近似） | A1 − (B1 − B0) | |
| 解码成本（KWS） | B0 − F | |
| 解码成本（ASR） | C0 − F | |
| ASR / KWS 解码倍数 | (C0 − F) / (B0 − F) | |
| 整机占用比（周报口径） | C0 / B0（前端费用两边相同，被自然约掉一部分） | |
| 待机整机占比 | B1 /（核数 × 100） | |

---

## 六、预期与判定

- **预期**：KWS 待机明显低于 ASR。依据：KWS 模型 3M 参数、38 个状态、每块 4 帧；
  ASR 116 个状态、每块 12 帧且 encoder 更大。但**这只是先验，不作为结论**，
  以实测为准。
- **判定「可长时间待机」**：B1（有窗待机）中位数 < 单核 100% 的 30%，
  且 10 分钟长跑 RSS 增长 < 5 MB。
- **异常排查**：
  - B0 与 C0 差距不到 1.2 倍 → 先怀疑前端占了大头，看 A1 减去渲染后（F）是否已经很高；
  - C0 明显低于预期 → 检查音频是不是静音太多（blank 帧不跑 decoder）；
  - 三次方差 > 30% → 机器上有别的负载，换时段重测。

---

## 七、执行后要同步的地方

1. 把 5.3 的结论写进 `doc/kws_plan_status.md` 3.8 的「第二轮真机」表；
2. 在 `README.md` 的稳定性基线表补一行「KWS 待机 vs ASR 识别 CPU」；
3. 如果数据稳定，再把采样逻辑并入 `tools/bench_perf.sh` 的双模式分支
   （目前脚本只解析入口统计行，不采样 CPU），让以后一条命令就能复现；
4. 周报里填 `kws 的 cpu 占用 xxx 相比 asr xxxx`。

---

## 八、已知陷阱

| 现象 | 原因 | 处理 |
|---|---|---|
| `StartVoiceCom failed, error=29` | 设备语音通道被别的客户端占用，或距上次 `stop()` 不到约 200 ms | 先 `pgrep` 找占用进程，确认是谁的；不要擅自 kill 用户自己启动的进程 |
| 有窗场景 CPU 异常高 | 窗口被最小化 / 不在可见桌面 / 软件渲染回退 | 确认 `DISPLAY=:0`、窗口可见；WSLg 与 Jetson 桌面各测各的，不混比 |
| ASR 场景 CPU 偏低 | 音频里大段静音，blank 帧不刷新 decoder | 用持续说话的音频，并核对统计行 token 数在增长 |
| KWS 场景中途唤醒 | 音频里有「小智」「同学」等相近发音 | 换音频或临时换关键词文件，保证 B 组 0 次唤醒 |
| 采样值忽高忽低 | 采样窗口与模型加载/GC 重叠，或机器有其他任务 | 预热 10 s 后再采样；三次取中位数 |
| 窗口关不掉 | `--seconds` 到点后停在最后一屏等按键 | 在窗口按任意键；无窗场景没有这个问题 |
| A1 没法关窗 | `test_spectrogram_cv_mic.out` 只有「带窗」一档 | 只能测合计值；渲染成本靠 B1 − B0 反推，别直接拿 A1 当纯前端 |
| 跑完后 `results/` 多出 png | `test_spectrogram_cv_mic.out 40` 到点自动存 `spectrogram_mic.png` | 属正常运行产物，不删不影响结论 |

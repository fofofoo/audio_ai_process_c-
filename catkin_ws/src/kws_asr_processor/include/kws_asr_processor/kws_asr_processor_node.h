#pragma once

/**
 * @file kws_asr_processor_node.h
 * @brief 语音处理节点类：海康设备取流 + KWS/ASR 双模式解码的流水线封装。
 *
 * ============================================================
 * 这份代码是干什么的
 * ============================================================
 * 从 audio_ai_process/test_mic_kws_asr.cpp 移植而来：把该入口里「设备连接 →
 * 前端 → 双模式解码 → 收尾统计」整条流水线收进一个类，main 只负责读 launch
 * 参数、发布话题与主循环节奏。
 *
 *   HkSdkAudioClient（G.722.1，钉 16 kHz）
 *     → get_last_audio() → int16 / 32768.0
 *     → AudioPreprocess（(n, 80) LogFBank）
 *     → 同一份特征分两路：
 *          SpectrogramImage（show_window=true 时滚动语谱图）
 *          BlockWorker（有界队列 + 后台线程）
 *               → KwsAsrSession
 *                    ├ KWS 模式：KwsDecoder（trie 约束束搜索）
 *                    └ ASR 模式：AsrSession → AsrDecoder
 *               → 快照（模式 / 文本 / 计数 / 最近事件）
 *     → processOnce() 返回本轮新增的事件与文本，main 发布成 ROS 话题
 *
 * 本类不持有 publisher、不读参数服务器、不调用 ROS 的节奏函数；
 * 话题名与发布时机都在 main，符合「节点类只做数据流水线」的约定。
 *
 * ============================================================
 * 设计取舍 / 关键约束
 * ============================================================
 * - 音频源是海康设备语音转发（固定 16 kHz），不是 ROS 音频话题；
 *   输出采样率不是 16 kHz 时 init() 直接失败，不做静默重采样。
 * - 两个解码器与模式机跑在同一个后台 Worker 线程：主线程只做取流、前端、
 *   渲染与快照读取，避免推理拖慢取流（取流缓冲上限约 2.56 s）。
 * - 设备会话拆两段：init() 只登录 + 查编码（确认数据源可用），
 *   start() 才开流；模型加载期间设备不推流，不会产生启动丢块。
 * - 模型、会话、Worker 全部用 unique_ptr 持有并在 init() 里按依赖顺序创建，
 *   成员声明顺序与创建顺序一致：C++ 逆序析构保证「先停 Worker 线程，
 *   再销毁解码器与 Ort::Session」。
 * - 收尾顺序固定（见 shutdown()）：停流 → 取完残余样本 → deinit →
 *   补尾块 → join → finish → 打印统计。顺序颠倒会丢最后一段话或与
 *   Worker 线程并发读写解码器。
 *
 * ============================================================
 * 调用顺序（对照 main）
 * ============================================================
 *   1) main 读 launch 参数，组装 KwsAsrProcessorConfig
 *   2) main 构造节点 → init()：连接设备、加载模型、装配前端与 Session
 *   3) main 调 start()：开语音转发、打印配置
 *   4) main 主循环：processOnce() → 发布事件 / 文本 → ros::Rate 节拍
 *   5) main 退出前调 shutdown()：收尾与统计；析构时兜底再调一次
 *
 * ============================================================
 * 用法
 * ============================================================
 *   roslaunch kws_asr_processor kws_asr_processor.launch
 */
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "asr_decode.h"
#include "asr_session.h"
#include "audio_preprocess.h"
#include "block_worker.h"
#include "hk_sdk_audio_client.h"
#include "kws_asr_session.h"
#include "kws_decode.h"
#include "spectrogram_image.h"
#include "text_renderer.h"

/**
 * @brief 节点运行配置：main 从 launch 参数组装后传入。
 *
 * 模型路径全部是「main 拼好的完整路径」（目录来自 launch，文件名写死 main），
 * 节点类不关心权重包的文件命名，只负责加载。
 */
struct KwsAsrProcessorConfig {
    // —— 设备连接（launch 参数）——
    std::string device_ip = "10.168.2.65";  ///< 音频设备 IP
    int sdk_port = 8000;                    ///< HCNetSDK 私有协议端口
    std::string username = "admin";         ///< 登录用户名
    std::string password = "Password";      ///< 登录密码
    int voice_channel = 1;                  ///< 语音通道号，本机实测为 1

    // —— ASR 模型与词表（main 拼好）——
    std::string asr_encoder_path;  ///< encoder.int8.onnx
    std::string asr_decoder_path;  ///< decoder.onnx（fp32）
    std::string asr_joiner_path;   ///< joiner.int8.onnx
    std::string asr_tokens_path;   ///< tokens.txt，5000 行

    // —— KWS 模型、词表与唤醒词（main 拼好）——
    std::string kws_encoder_path;  ///< encoder-...-chunk-8...int8.onnx
    std::string kws_decoder_path;  ///< decoder-...-chunk-8...onnx
    std::string kws_joiner_path;   ///< joiner-...-chunk-8...int8.onnx
    std::string kws_tokens_path;   ///< tokens.txt，263 个音素 token
    std::string keywords_path;     ///< 关键词文件（token:score #threshold @原文）

    // —— 双模式行为（launch 参数）——
    bool initial_asr = false;       ///< true = 从 ASR 模式起步（调试纯识别）
    double idle_timeout_s = 10.0;   ///< ASR 空闲多久无新 token 就切回 KWS
    int backfill_frames = 100;      ///< 唤醒时回填最近多少帧（10 ms/帧）

    // —— 运行开关（launch 参数）——
    bool use_cuda = false;          ///< ASR / KWS 都尝试 CUDA EP，失败回退 CPU
    bool show_window = false;       ///< 是否显示语谱图窗口（服务器上保持 false）
    std::string transcript_path;    ///< ASR 分段转写落盘路径（每段一行）
};

/**
 * @brief 主循环一轮的产出：main 据此发布话题。
 *
 * 事件与文本都只在「变化时」置位，main 不需要自己保存上一次的值。
 */
struct KwsAsrProcessorStep {
    bool has_event = false;       ///< 本轮出现新事件（唤醒 / 结束语 / 超时）
    std::string event;            ///< 事件文本（UTF-8，可直接发布）
    bool has_text = false;        ///< 本轮 ASR 会话文本有变化
    std::string text;             ///< 当前 ASR 会话文本（UTF-8，KWS 模式为空）
    bool quit_requested = false;  ///< 窗口里按了 q / Esc，main 应结束主循环
};

/**
 * @brief 语音处理节点：设备取流、双模式解码与统计的流水线封装。
 */
class KwsAsrProcessorNode {
public:
    /**
     * @brief 保存配置；不连接设备、不加载模型。
     * @param config 运行配置，main 已按 launch 参数拼好全部路径
     */
    explicit KwsAsrProcessorNode(const KwsAsrProcessorConfig& config);

    /**
     * @brief 兜底收尾：main 未调用 shutdown() 时补一次（内部靠 running_ 去重）。
     */
    ~KwsAsrProcessorNode();

    // 持有设备句柄、Ort::Session 与后台线程，禁止拷贝
    KwsAsrProcessorNode(const KwsAsrProcessorNode&) = delete;
    KwsAsrProcessorNode& operator=(const KwsAsrProcessorNode&) = delete;

    /**
     * @brief 连接设备并加载全部模型与前端，不开流。
     *
     * 步骤：设备 init → 采样率校验（必须 16 kHz）→ Ort 环境与选项 →
     * ASR 三 session → KWS 三 session → 前端 → 语谱图 → 中文字体 →
     * AsrSession / KwsAsrSession / BlockWorker 装配。
     *
     * @return true 全部就绪，可由 start() 开流；false 已打印原因（设备已 deinit）
     */
    bool init();

    /**
     * @brief 开语音转发并打印运行配置。
     * @return true 开流成功；false 已 deinit，调用方应退出
     */
    bool start();

    /**
     * @brief 主循环一轮：取音频 → 前端 → 投递 → 读快照 → （可选）画窗与按键。
     * @return 本轮新增事件 / 文本变化 / 窗口退出请求
     */
    KwsAsrProcessorStep processOnce();

    /**
     * @brief 收尾：停流、取完残余样本、补尾块、join、结算并打印统计。
     *
     * 可重复调用，第二次起直接返回；析构时兜底调用一次。
     */
    void shutdown();

private:
    /**
     * @brief 加载 ONNX 模型与前端/可视化组件，装配三个会话与 Worker。
     * @return true 全部成功
     */
    bool loadModels();

    /**
     * @brief 空格暂停 / 恢复语音转发：只调 stop() / start()，不重跑 init()。
     */
    void togglePause();

    /**
     * @brief 把一行状态文字画到画布底部（FreeType 中文，失败回退 ASCII）。
     * @param frame       目标画布
     * @param text        UTF-8 文本
     * @param baseline    基线距画布底部的像素数
     * @param font_height 字号（像素）
     */
    void drawStatusLine(cv::Mat& frame, const std::string& text, int baseline,
                        int font_height);

    /// 刷新并显示语谱图窗口（show_window 时每轮调用；画布未生成时跳过）。
    void drawFrame();

    /// 打印设备 / 模型 / 唤醒词 / 切换规则等启动配置（init + start 后调用）。
    void printConfig();

    /// 打印收尾统计（shutdown 里调用，此时 Worker 已 join）。
    void printStats(const std::string& final_text, double wall_seconds);

    /// 运行配置（main 传入，只读）
    KwsAsrProcessorConfig config_;

    // —— 依赖顺序：以下成员声明顺序即创建顺序，析构时逆序销毁 ——
    // 设备会话（采集库）：init() 创建并 init()，start() 开流
    std::unique_ptr<hk_isapi::HkSdkAudioClient> client_;
    // ORT 环境与选项：必须活到所有 Ort::Session 销毁之后
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    // 两组三模型 session：解码器持有其引用，必须后于解码器销毁
    std::unique_ptr<Ort::Session> asr_encoder_;
    std::unique_ptr<Ort::Session> asr_decoder_session_;
    std::unique_ptr<Ort::Session> asr_joiner_;
    std::unique_ptr<Ort::Session> kws_encoder_;
    std::unique_ptr<Ort::Session> kws_decoder_session_;
    std::unique_ptr<Ort::Session> kws_joiner_;
    // 解码器：非线程安全，全程只由 Worker 线程访问
    std::unique_ptr<audio_ai::AsrDecoder> asr_decoder_;
    std::unique_ptr<audio_ai::KwsDecoder> kws_decoder_;
    // ASR 长期缓存（全量 token / 分段转写）
    std::unique_ptr<audio_ai::AsrSession> asr_session_;
    // 双模式模式机（持有上面三个对象的引用）
    std::unique_ptr<audio_ai::KwsAsrSession> session_;
    // 后台线程与有界队列：必须最先停（最后销毁）
    std::unique_ptr<audio_ai::BlockWorker<audio_ai::LogFbankFeatures>> worker_;
    // 前端与可视化（无相互依赖，放在 Worker 之后销毁即可）
    std::unique_ptr<audio_ai::AudioPreprocess> front_;
    std::unique_ptr<audio_ai::SpectrogramImage> view_;
    std::unique_ptr<audio_ai::Utf8TextRenderer> text_renderer_;

    // —— 启动信息（打印配置用）——
    audio_ai::AsrDecoder::Info asr_info_;  ///< ASR metadata：T / 帧数 / 状态数等
    audio_ai::KwsDecoder::Info kws_info_;  ///< KWS metadata：帧长 / 关键词数等
    int sample_rate_ = 0;                  ///< 采集输出采样率（必须 16000）
    std::string cuda_status_;              ///< CUDA EP 注册结果文本（打印用）

    // —— 主循环状态 ——
    std::vector<int16_t> pcm_;       ///< 本轮取走的 PCM16
    std::vector<float> chunk_;       ///< 归一化后的 float [-1,1]
    cv::Mat canvas_;                 ///< 最新一帧画布（show_window）
    std::string shown_;              ///< 上次已返回给 main 的文本（去重用）
    int64_t shown_event_seq_ = 0;    ///< 已处理的事件序号
    double peak_ = 0.0;              ///< 峰值幅度，收尾时判断是否真采到声音
    bool paused_ = false;            ///< 语音转发是否被空格暂停
    bool running_ = false;           ///< start() 成功且尚未 shutdown()
    audio_ai::KwsAsrSnapshot snapshot_;  ///< 最近一次快照（主线程读）

    // —— 计时（统计墙钟用）——
    std::chrono::steady_clock::time_point start_time_;  ///< start() 成功时刻
};

/**
 * @file main.cpp
 * @brief 语音处理节点入口：读 launch 参数、发布话题、驱动主循环。
 *
 * ============================================================
 * 这份代码是干什么的
 * ============================================================
 * main 直白定义设备 / 模型 / 话题参数，构造 KwsAsrProcessorNode，
 * 然后进入显式 while (ros::ok()) 轮询：
 *
 *   1) 读 launch 参数（设备地址、模型目录、开关、超时）
 *   2) 拼出两组模型的完整路径（目录来自 launch，文件名写在这里）
 *   3) advertise 两个 std_msgs/String 话题（话题名是 main 字面量）
 *   4) 构造节点 → init() 连接设备并加载模型 → start() 开流
 *   5) 主循环：processOnce() 一轮 → 有新事件 / 文本就发布 → 10 ms 节拍
 *   6) 退出前 shutdown()：停流、补尾块、打印统计
 *
 * 节点是「主动轮询型」：没有订阅者，数据由海康设备推送、主循环主动取。
 * 因此 main 保留显式 while 循环，按大步骤调用节点类；话题名、publish
 * 与节奏都在 main，节点类不碰 ROS 通讯。
 *
 * ============================================================
 * 话题
 * ============================================================
 *   /audio_ai/asr_text   std_msgs/String
 *       当前 ASR 会话文本，只在变化时发布；KWS 模式切回后为空串。
 *   /audio_ai/kws_event  std_msgs/String
 *       唤醒 / 结束语 / 超时切回等事件文本，每个事件发布一次。
 *
 * ============================================================
 * 用法
 * ============================================================
 *   roslaunch kws_asr_processor kws_asr_processor.launch
 *   roslaunch kws_asr_processor kws_asr_processor.launch use_cuda:=true
 *   rosrun kws_asr_processor kws_asr_processor_node   # 用代码内默认参数
 */
#include <locale.h>

#include <string>

#include <ros/ros.h>
#include <std_msgs/String.h>

#include "kws_asr_processor/kws_asr_processor_node.h"
#include "text_utils.h"

using namespace audio_ai;

int main(int argc, char** argv) {
    // 终端按 UTF-8 输出中文日志；缺少这一行时 roslaunch 环境下中文可能变成 "?"。
    setlocale(LC_ALL, "");

    // —— 一、初始化 ROS 节点（只用主句柄 nh） ——
    ros::init(argc, argv, "kws_asr_processor_node");
    ros::NodeHandle nh;

    // —— 二、读 launch 参数，组装运行配置 ——
    // 全部参数放在 <node> 外的全局命名空间，main 用 nh 读取（与工作区其它节点一致）。
    // 每行第三个实参是 config 里的代码内默认值：launch / rosrun 都未提供时生效，
    // 与 .launch 里的 <arg> 默认值保持一致（唯一例外是 show_window，launch 默认为 true）。
    KwsAsrProcessorConfig config;
    // 设备连接
    nh.param<std::string>("ip", config.device_ip, config.device_ip);              // 音频设备 IP（默认 10.168.2.65）
    nh.param("sdk_port", config.sdk_port, config.sdk_port);                       // HCNetSDK 私有协议端口（默认 8000，不是 HTTP 端口）
    nh.param<std::string>("username", config.username, config.username);          // 设备登录用户名（默认 admin）
    nh.param<std::string>("password", config.password, config.password);          // 设备登录密码（默认 Password）
    nh.param("voice_channel", config.voice_channel, config.voice_channel);        // 语音通道号，从 1 起（本机实测为 1）
    // 运行开关
    nh.param("use_cuda", config.use_cuda, config.use_cuda);                       // true = ASR / KWS 都尝试 CUDA EP，注册失败自动回退 CPU
    nh.param("show_window", config.show_window, config.show_window);              // true = 显示滚动语谱图窗口；机器人 / 服务器上保持 false
    // 双模式行为
    nh.param("initial_asr", config.initial_asr, config.initial_asr);              // true = 从 ASR 模式起步（调试纯识别，跳过唤醒）
    nh.param("idle_timeout", config.idle_timeout_s, config.idle_timeout_s);       // ASR 模式连续多少秒没有新 token 就切回 KWS（默认 10.0）
    nh.param("backfill_frames", config.backfill_frames, config.backfill_frames);  // 唤醒时回填最近多少帧，10 ms/帧（默认 100 帧 = 1 s）

    // 2.1 模型目录：默认取 audio_ai_process 仓库 weight/ 下的权重包。
    //     audio_ai_root 是仓库根（库与权重都在它下面），换机器时在 launch 里改。
    std::string audio_ai_root;
    nh.param<std::string>("audio_ai_root", audio_ai_root,"/home/liyun/code/AI/audio_ai_process");

    std::string asr_model_dir;
    nh.param<std::string>( "asr_model_dir", asr_model_dir, join_path(join_path(audio_ai_root, "weight"), "sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05"));

    std::string kws_model_dir;
    nh.param<std::string>("kws_model_dir", kws_model_dir, join_path(join_path(audio_ai_root, "weight"), "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"));

    // 2.2 模型文件名：与权重包绑定，直接写在这里；换权重包时目录与文件名一起改。
    //     ASR：encoder / joiner 是 int8，decoder 保持 fp32（导出时就是这么切的）；
    //     KWS：默认 chunk-8（160 ms 推进，唤醒延迟低于 chunk-16）。
    const std::string kAsrEncoderFile = "encoder.int8.onnx";
    const std::string kAsrDecoderFile = "decoder.onnx";
    const std::string kAsrJoinerFile = "joiner.int8.onnx";
    const std::string kAsrTokensFile = "tokens.txt";
    const std::string kKwsEncoderFile = "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx";
    const std::string kKwsDecoderFile = "decoder-epoch-13-avg-2-chunk-8-left-64.onnx";
    const std::string kKwsJoinerFile = "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx";
    const std::string kKwsTokensFile = "tokens.txt";
    // asr模型地址
    config.asr_encoder_path = join_path(asr_model_dir, kAsrEncoderFile);
    config.asr_decoder_path = join_path(asr_model_dir, kAsrDecoderFile);
    config.asr_joiner_path = join_path(asr_model_dir, kAsrJoinerFile);
    config.asr_tokens_path = join_path(asr_model_dir, kAsrTokensFile);
    // kws模型地址
    config.kws_encoder_path = join_path(kws_model_dir, kKwsEncoderFile);
    config.kws_decoder_path = join_path(kws_model_dir, kKwsDecoderFile);
    config.kws_joiner_path = join_path(kws_model_dir, kKwsJoinerFile);
    config.kws_tokens_path = join_path(kws_model_dir, kKwsTokensFile);

    // 2.3 唤醒词与转写文件：唤醒词改文件不动代码；
    //     转写每 64 个 token 结算一行，父目录由 AsrSession 自动创建。
    nh.param<std::string>("keywords", config.keywords_path, join_path(audio_ai_root, "audio_ai_process/keywords/kws_keywords.txt"));
    nh.param<std::string>("transcript_path", config.transcript_path, join_path(join_path(audio_ai_root, "log"), "asr_last_text.txt"));

    // —— 三、发布话题（话题名是 main 字面量，现场不改） ——
    // 只发布，不订阅：音频源是设备语音转发，不走 ROS 话题。
    const std::string kAsrTextTopic = "/audio_ai/asr_text";
    const std::string kKwsEventTopic = "/audio_ai/kws_event";
    ros::Publisher asr_text_pub = nh.advertise<std_msgs::String>(kAsrTextTopic, 10);
    ros::Publisher kws_event_pub = nh.advertise<std_msgs::String>(kKwsEventTopic, 10);

    // —— 四、构造节点并启动 ——
    // init() 只登录设备 + 加载模型，start() 才开流：
    // 模型加载期间设备不推流，不会产生启动丢块。
    KwsAsrProcessorNode node(config);
    if (!node.init()) {
        ROS_ERROR("[kws_asr_processor] 初始化失败，节点退出");
        return 1;
    }
    if (!node.start()) {
        ROS_ERROR("[kws_asr_processor] 启动失败，节点退出");
        return 1;
    }

    // —— 五、主循环 ——
    // 10 ms 一拍，与特征帧移一致（对应入口的 kIdleWaitMs）。
    // 事件先于文本发布：唤醒 / 切回是文本语义变化的前提。
    ros::Rate rate(100);
    while (ros::ok()) {
        const KwsAsrProcessorStep step = node.processOnce();

        if (step.has_event) {
            std_msgs::String msg;
            msg.data = step.event;
            kws_event_pub.publish(msg);
        }
        if (step.has_text) {
            std_msgs::String msg;
            msg.data = step.text;
            asr_text_pub.publish(msg);
        }

        // 窗口里按 q / Esc：走正常收尾，不用 ros::shutdown 的强制路径。
        if (step.quit_requested) {
            ROS_INFO("[kws_asr_processor] 窗口按键退出");
            break;
        }

        // 无订阅者，spinOnce 只为处理 ROS 内部事件（Ctrl+C / 参数更新）。
        ros::spinOnce();
        rate.sleep();
    }

    // —— 六、收尾 ——
    // stop → 取残余样本 → deinit → 补尾块 → join → 结算 → 打印统计。
    node.shutdown();
    ROS_INFO("[kws_asr_processor] 节点已退出");
    return 0;
}

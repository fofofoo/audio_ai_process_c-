#pragma once

/**
 * @file block_worker.h
 * @brief 通用异步处理管家：一个有界输入队列 + 一个串行消费线程。
 *
 * ============================================================
 * 一、这份代码是干什么的
 * ============================================================
 * 主线程把「一块输入」投进队列后立刻返回，后台唯一线程按顺序取出、
 * 交给 Handler 处理。它解决的是「采集/渲染不能被推理卡住」的问题：
 * 队列有界，跟不上时丢最旧的块，保证结果始终贴近当前输入；收尾块单独
 * 走阻塞路径，绝不丢。
 *
 * 本类只知道「块」和「怎么处理一块」，不知道块里是音频特征还是别的，
 * 也不知道处理结果是什么 —— 结果由 Handler 所属的 Session 类自己保存，
 * 主线程通过 Session 的读取接口拿快照。设计说明见
 * doc/audio_ai_block_worker.md。
 *
 * ============================================================
 * 二、关键约定
 * ============================================================
 *   1. 一个 Worker 只开一个线程，Handler 严格串行，被它调用的解码器因此
 *      不需要加锁；
 *   2. push() 非阻塞，队列满时丢最旧并计入 dropped()；
 *   3. push_final() 是尾块专用：队列满时阻塞等空位，绝不丢；必须在
 *      finish_and_join() 之前调用；
 *   4. Skip 返回 true 的块（如 0 帧特征）不入队、不占容量、不计丢弃；
 *   5. Handler 抛出的异常被捕获并记录，随后停止消费；error() 可查；
 *   6. finish_and_join() 幂等，先把已入队的块处理完再退出；析构兜底 join。
 *
 * ============================================================
 * 三、线程与生命周期
 * ============================================================
 * 线程在构造时启动；Handler、Skip 按值保存，调用方必须保证它们引用的
 * 对象（解码器、Session）活得比 Worker 长。C++ 按声明逆序析构，入口里
 * 把 Worker 声明在 Session 与模型会话之后即可满足。
 *
 * 持锁期间绝不调用 Handler：队列锁只保护入队/出队，处理在锁外进行，
 * 主线程投递不会被推理拖住，两把锁也不会嵌套。
 */

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace audio_ai {

template <typename Block>
class BlockWorker {
public:
    /// 队里的一项：输入块 + 是否为收尾块。
    /// is_tail 只由 Worker 标记，语义由 Handler 解释（例如 KWS 模式跳过）。
    struct Item {
        Block block;
        bool is_tail = false;
    };

    using Handler = std::function<void(const Item&)>;
    using Skip = std::function<bool(const Block&)>;

    /**
     * @brief 绑定处理函数并启动后台线程。
     * @param handler     每块的处理逻辑，在后台线程串行执行
     * @param skip        返回 true 的块直接丢弃（空块过滤）；可为空
     * @param max_pending 队列上限（块），至少为 1
     */
    explicit BlockWorker(Handler handler, Skip skip = {},
                         size_t max_pending = 50)
        : handler_(std::move(handler)),
          skip_(std::move(skip)),
          max_pending_(max_pending == 0 ? 1 : max_pending) {
        thread_ = std::thread([this] { run(); });
    }

    /// 兜底 join：漏调 finish_and_join() 也不会让 std::thread 在 joinable
    /// 状态下析构而 terminate。
    ~BlockWorker() { finish_and_join(); }

    BlockWorker(const BlockWorker&) = delete;
    BlockWorker& operator=(const BlockWorker&) = delete;

    /**
     * @brief 非阻塞投递一块输入。
     * @return true 已入队；false 被 Skip 过滤或 Worker 已结束。
     *
     * 队列满时丢最旧的一块再入队：容量代表「可容忍的结果延迟」，不是
     * 内存上限。丢弃数进 dropped()，用于判断后台是否跟得上。
     */
    bool push(Block block) {
        // 1. 空块等无意义输入直接过滤：不占容量、不计丢弃。
        if (skip_ && skip_(block)) {
            return false;
        }

        // 2. 已结束（或 Handler 异常退出）后不再接收。
        std::lock_guard<std::mutex> lock(mutex_);
        if (finished_) {
            return false;
        }

        // 3. 背压：满时丢最旧，保证新块能进来。
        while (queue_.size() >= max_pending_) {
            queue_.pop_front();
            ++dropped_;
        }
        queue_.push_back(Item{std::move(block), false});
        cv_.notify_one();
        return true;
    }

    /**
     * @brief 投递收尾块：队列满时阻塞等空位，绝不丢。
     *
     * 它承载最后几个字 / 最终状态，丢了无法补回。此时主循环通常已经
     * 退出、不要求实时，等是安全的。必须在 finish_and_join() 之前调用。
     */
    void push_final(Block block) {
        if (skip_ && skip_(block)) {
            return;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
            return finished_ || queue_.size() < max_pending_;
        });
        if (finished_) {
            return;
        }
        queue_.push_back(Item{std::move(block), true});
        cv_.notify_one();
    }

    /// 置结束标志并等后台线程把队列解空后退出。幂等。
    /// 返回后 Handler 不再被调用，调用方可以安全读取 Session 的最终结果。
    void finish_and_join() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            finished_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    /// 因队列满而丢弃的块数。
    size_t dropped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_;
    }

    /// 队列上限（块）。
    size_t max_pending() const { return max_pending_; }

    /// Handler 抛出的第一个异常；一直正常时为空。
    std::exception_ptr error() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return error_;
    }

private:
    /// 后台线程主循环：等队列非空 → 取一块 → 锁外处理 → 记录异常。
    /// finished_ 且队列排空后退出。
    void run() {
        for (;;) {
            Item item;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                // 1. 空队列时睡在条件变量上，不轮询、不占 CPU。
                cv_.wait(lock, [this] {
                    return finished_ || !queue_.empty();
                });

                // 2. 已结束且排空：所有块都处理完了，线程退出。
                if (queue_.empty()) {
                    return;
                }

                // 3. 取走队首再放锁：处理期间主线程仍可继续投递。
                item = std::move(queue_.front());
                queue_.pop_front();

                // 4. 腾出空位，叫醒可能阻塞在 push_final() 的主线程。
                cv_.notify_all();
            }

            // 5. 锁外处理，持锁不解码；异常留在线程内，不穿出线程。
            try {
                handler_(item);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                error_ = std::current_exception();
                finished_ = true;
                queue_.clear();
                cv_.notify_all();
                return;
            }
        }
    }

private:
    Handler handler_;                 ///< 每块的处理逻辑（串行调用）
    Skip skip_;                       ///< 空块 / 无意义块过滤；可为空
    size_t max_pending_ = 50;         ///< 队列上限（块）

    mutable std::mutex mutex_;        ///< 保护 queue_ / finished_ / dropped_ / error_
    std::condition_variable cv_;      ///< 队列非空或结束状态变化时唤醒
    std::deque<Item> queue_;          ///< 待处理输入块（按到达顺序）
    bool finished_ = false;           ///< 收尾或异常后置位
    size_t dropped_ = 0;              ///< 队列满时丢最旧的累计块数
    std::exception_ptr error_;        ///< Handler 抛出的第一个异常

    std::thread thread_;              ///< 最后声明：构造时启动、析构时 join
};

}  // namespace audio_ai

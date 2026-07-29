#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace kimrt::llm {

namespace detail {
    struct AdmissionState;
}

/*
 *Admission 只管理Serving Runtime 自己能够明确约束的资源;
 *
 * 1.活动请求数量;
 * 2.已预留的输入token数量
 * 3.已预留的最大输出token数量
 * 4.request_id唯一性。
 * 
 * 它不负责 TensorRT-LLM 内部的 KV Cache 分配与连续批处理调度。
 */

struct AdmissionConfig {
    std::size_t max_active_requests{8};
    std::size_t max_total_input_tokens{4096};
    std::size_t max_reserved_output_tokens{256};
};

struct AdmissionRequest {
    std::uint64_t request_id{0};
    std::size_t input_tokens{0};
    std::size_t reserved_output_tokens{0};
};

struct AdmissionSnapshot {
    bool accepting{false};

    std::size_t active_requests{0};
    std::size_t reserved_input_tokens{0};
    std::size_t reserved_output_tokens{0};

    std::size_t max_active_requests{0};
    std::size_t max_total_input_tokens{0};
    std::size_t max_reserved_output_tokens{0};
};

enum class AdmissionCode: uint8_t {
    Admitted,

    InvalidRequest,
    NotAccepting,
    DuplicateRequestId,

    ActiveRequestLimit,
    InputTokenLimit,
    OutputTokenLimit,
};

/*
   * 一次成功 Admission 对应一个 Lease。
   *
   * Lease 是 move-only 的。它被显式 release() 或析构时，自动归还：
   *
   * - active request slot；
   * - input token budget；
   * - reserved output token budget；
   * - request_id 占用。
   *
   * Lease 持有 AdmissionState 的 shared_ptr，因此即使
   * AdmissionController 先析构，Lease 仍然可以安全释放资源。
   */

class AdmissionLease final {
public:
    AdmissionLease() noexcept = default;
    ~AdmissionLease();

    AdmissionLease(AdmissionLease const & ) = delete;
    AdmissionLease& operator = (AdmissionLease const&) = delete;

    AdmissionLease(AdmissionLease&& other)noexcept;
    AdmissionLease& operator=(AdmissionLease && other)noexcept;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint64_t requestId() const noexcept;
    [[nodiscard]] std::size_t inputTokens() const noexcept;
    [[nodiscard]] std::size_t reservedOutputTokens() const noexcept;

    // 幂等操作；重复调用不会重复归还资源。
    void release() noexcept;

private:
    friend class AdmissionController;

    AdmissionLease(
    std::shared_ptr<detail::AdmissionState> state,
    AdmissionRequest request) noexcept;

    std::shared_ptr<detail::AdmissionState> state_;
    AdmissionRequest request_;
    bool released_{true};

};

struct AdmissionDecision {
    AdmissionCode code{AdmissionCode::InvalidRequest};
    AdmissionLease lease;

    [[nodiscard]] bool admitted() const noexcept {
      return code == AdmissionCode::Admitted && lease.valid();
    }

    explicit operator bool() const noexcept {
      return admitted();
    }
  };

class AdmissionController final {
public:
    explicit AdmissionController(AdmissionConfig config);
    ~AdmissionController();

    AdmissionController(AdmissionController const &) = delete;
    AdmissionController &operator=(AdmissionController const &) = delete;

    AdmissionController(AdmissionController &&) = delete;
    AdmissionController &operator=(AdmissionController &&) = delete;

    /*
     * 新建的 AdmissionController 默认不接收请求。
     *
     * GenerationRuntime::start() 在 Backend 启动成功后调用 open()。
     * 只有当前不存在活动 Lease 时才能重新打开。
     */
    [[nodiscard]] bool open() noexcept;

    /*
     * 立即拒绝后续 Admission，但不撤销已经发出的 Lease。
     *
     * GenerationRuntime::stop() 应先 close()，再取消和收敛活动请求。
     */
    void close() noexcept;

    [[nodiscard]] bool accepting() const noexcept;

    /*
     * 原子完成以下检查与资源预留：
     *
     * 1. 参数合法；
     * 2. Controller 正在接收请求；
     * 3. request_id 不重复；
     * 4. active request 数量未超限；
     * 5. input token 总预算未超限；
     * 6. reserved output token 总预算未超限。
     *
     * 成功时返回持有全部资源的 move-only AdmissionLease。
     */
    [[nodiscard]] AdmissionDecision tryAcquire(
        AdmissionRequest request);

    [[nodiscard]] AdmissionSnapshot snapshot() const noexcept;

private:
    std::shared_ptr<detail::AdmissionState> state_;
  };

}
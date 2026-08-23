#include "runtime/admission_controller.h"

#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace kimrt::llm::detail {

struct AdmissionState {
    explicit AdmissionState(AdmissionConfig admission_config)
    :config_(admission_config){}

    AdmissionConfig config_;
    //这里先用一把大锁，是否改为原子变量待定
    mutable std::mutex mtx_;
    bool accepting{false};

    std::unordered_set<std::uint64_t>active_request_ids;
    std::size_t reserved_input_tokens{0};
    std::size_t reserved_output_tokens{0};
};

}//namespace kimrt::llm::detail

namespace kimrt::llm {
    AdmissionLease::AdmissionLease(
      std::shared_ptr<detail::AdmissionState> state,
      AdmissionRequest request) noexcept
      : state_(std::move(state)),
        request_(request),
        released_(false) {}

    AdmissionLease::~AdmissionLease(){
        release();
    }

    AdmissionLease::AdmissionLease(AdmissionLease&& other) noexcept
        :state_(std::move(other.state_)),
         request_(std::exchange(other.request_,{})),
         released_(std::exchange(other.released_,true)){}
    
    AdmissionLease& AdmissionLease::operator = (AdmissionLease&& other)noexcept {
        if(this == &other){
            return *this;
        }

        //这里先归还当前Lease持有的资源，再接管other
        release();

        state_ = std::move(other.state_);
        request_ = std::exchange(other.request_,{});
        released_ = std::exchange(other.released_,true);

        return *this;
    }

    bool AdmissionLease::valid() const noexcept {
        return state_ != nullptr && !released_;
    }

    std::uint64_t AdmissionLease::requestId() const noexcept {
        return request_.request_id;
    }

    std::size_t AdmissionLease::inputTokens() const noexcept {
        return request_.input_tokens;
    }

    std::size_t AdmissionLease::reservedOutputTokens() const noexcept {
        return request_.reserved_output_tokens;
    }

    void AdmissionLease::release() noexcept {
        if(!valid()){ 
            return;
        }

        auto state = std::move(state_);
        const AdmissionRequest request = 
            std::exchange(request_,{});
        released_ = true;

        std::lock_guard<std::mutex>lock(state->mtx_);

        const std::size_t erased =
        state->active_request_ids.erase(request.request_id);

        if (erased == 0) {
            // 正常情况下不会发生；避免异常状态下重复扣减预算。
            return;
        }

        state->reserved_input_tokens -= request.input_tokens;
        state->reserved_output_tokens -=
        request.reserved_output_tokens;
    }

    AdmissionController::AdmissionController(
        AdmissionConfig config
    ){
        if (config.max_active_requests == 0 ||
          config.max_total_input_tokens == 0 ||
          config.max_reserved_output_tokens == 0) {
          throw std::invalid_argument(
              "admission capacity limits must be greater than zero");
        }
        state_ =
          std::make_shared<detail::AdmissionState>(config);
    }

    AdmissionController::~AdmissionController() {
        close();
    }
    
    bool AdmissionController::open() noexcept {
        std::lock_guard<std::mutex>lock(state_->mtx_);

        // 只有所有 Lease 均已归还时才允许重新开放。
        if (!state_->active_request_ids.empty() ||
            state_->reserved_input_tokens != 0 ||
            state_->reserved_output_tokens != 0) {
            return false;
        }

        state_->accepting = true;
        return true;
    }

    void AdmissionController::close() noexcept {
        std::lock_guard<std::mutex> lock(state_->mtx_);
        state_->accepting = false;
    }

    bool AdmissionController::accepting() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mtx_);
        return state_->accepting;
    }

    AdmissionDecision AdmissionController::tryAcquire(
        AdmissionRequest request
    ){
        /*
       * 静态参数检查不依赖共享状态，可以在加锁前完成。
       * 当前约定 request_id、输入 Token 和输出预算都必须大于 0。
       */
      if (request.request_id == 0 ||
          request.input_tokens == 0 ||
          request.reserved_output_tokens == 0) {
          return {
              AdmissionCode::InvalidRequest,
              {},
          };
        }

         /*
       * 从状态检查到资源预留全程持有同一把锁，
       * 保证多个调用者并发 tryAcquire() 时不会超额预留。
       */
      std::lock_guard<std::mutex> lock(state_->mtx_);

      if (!state_->accepting) {
          return {
              AdmissionCode::NotAccepting,
              {},
          };
      }

      if (state_->active_request_ids.find(request.request_id) !=
          state_->active_request_ids.end()) {
          return {
              AdmissionCode::DuplicateRequestId,
              {},
          };
      }

      if (state_->active_request_ids.size() >=
          state_->config_.max_active_requests) {
          return {
              AdmissionCode::ActiveRequestLimit,
              {},
          };
        }

        /*
       * 使用 request > limit - current，避免 current + request
       * 在 size_t 上发生整数溢出。
       */
      if (request.input_tokens >
          state_->config_.max_total_input_tokens -
              state_->reserved_input_tokens) {
          return {
              AdmissionCode::InputTokenLimit,
              {},
          };
        }

      if (request.reserved_output_tokens >
          state_->config_.max_reserved_output_tokens -
              state_->reserved_output_tokens) {
          return {
              AdmissionCode::OutputTokenLimit,
              {},
          };
        }

        // 所有检查通过后，在同一临界区内完成资源预留。
        state_->active_request_ids.insert(request.request_id);
        state_->reserved_input_tokens += request.input_tokens;
        state_->reserved_output_tokens +=
            request.reserved_output_tokens;

        return {
            AdmissionCode::Admitted,
            AdmissionLease(state_, request),
        };
    }

    AdmissionSnapshot AdmissionController::snapshot() const noexcept {
      std::lock_guard<std::mutex> lock(state_->mtx_);

      return {
          state_->accepting,
          state_->active_request_ids.size(),
          state_->reserved_input_tokens,
          state_->reserved_output_tokens,
          state_->config_.max_active_requests,
          state_->config_.max_total_input_tokens,
          state_->config_.max_reserved_output_tokens,
      };
    }
}
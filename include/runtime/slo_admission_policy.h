#pragma once

#include "runtime/admission_controller.h"

#include <cstddef>
#include <string>
#include <vector>

namespace kimrt::llm {

struct SloProfileEntry {
    std::size_t max_input_tokens{0};
    std::size_t active_requests{0};
    double predicted_ttft_p95_ms{0.0};
};

struct SloAdmissionPolicyConfig {
    std::string model_id;
    std::string revision;
    std::string engine_fingerprint;
    double ttft_slo_ms{0.0};
    double safety_margin_ms{0.0};
    std::vector<SloProfileEntry> entries;
};

struct SloAdmissionDecision {
    bool admitted{false};
    double predicted_ttft_p95_ms{0.0};
};

// 基于离线画像的软门禁。它只允许提前拒绝请求，不能绕过硬 Admission。
class SloAdmissionPolicy final {
public:
    explicit SloAdmissionPolicy(SloAdmissionPolicyConfig config);

    [[nodiscard]] SloAdmissionDecision evaluate(
        std::size_t input_tokens,
        AdmissionSnapshot const& admission) const noexcept;

    [[nodiscard]] SloAdmissionPolicyConfig const& config() const noexcept;

private:
    SloAdmissionPolicyConfig config_;
};

} // namespace kimrt::llm

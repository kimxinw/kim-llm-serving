#include "runtime/slo_admission_policy.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace kimrt::llm {

SloAdmissionPolicy::SloAdmissionPolicy(SloAdmissionPolicyConfig config)
    : config_(std::move(config)) {
    if (config_.model_id.empty() || config_.revision.empty() ||
        config_.engine_fingerprint.empty() ||
        !std::isfinite(config_.ttft_slo_ms) ||
        config_.ttft_slo_ms <= 0.0 ||
        !std::isfinite(config_.safety_margin_ms) ||
        config_.safety_margin_ms < 0.0 ||
        config_.entries.empty()) {
        throw std::invalid_argument("invalid SLO Admission policy configuration");
    }

    for (auto const& entry : config_.entries) {
        if (entry.max_input_tokens == 0 || entry.active_requests == 0 ||
            !std::isfinite(entry.predicted_ttft_p95_ms) ||
            entry.predicted_ttft_p95_ms <= 0.0) {
            throw std::invalid_argument("invalid SLO profile entry");
        }
    }

    std::sort(
        config_.entries.begin(),
        config_.entries.end(),
        [](SloProfileEntry const& left, SloProfileEntry const& right) {
            return std::tie(left.max_input_tokens, left.active_requests) <
                std::tie(right.max_input_tokens, right.active_requests);
        });
    auto const duplicate = std::adjacent_find(
        config_.entries.begin(),
        config_.entries.end(),
        [](SloProfileEntry const& left, SloProfileEntry const& right) {
            return left.max_input_tokens == right.max_input_tokens &&
                left.active_requests == right.active_requests;
        });
    if (duplicate != config_.entries.end()) {
        throw std::invalid_argument("duplicate SLO profile bucket");
    }
}

SloAdmissionDecision SloAdmissionPolicy::evaluate(
    std::size_t input_tokens,
    AdmissionSnapshot const& admission) const noexcept {
    if (input_tokens == 0 || !admission.accepting) {
        return {};
    }
    if (admission.active_requests >= admission.max_active_requests) {
        // 交给硬 Admission 返回准确的 active-request limit 原因。
        return {true, 0.0};
    }

    auto const prospective_active = admission.active_requests + 1;
    auto const entry = std::find_if(
        config_.entries.begin(),
        config_.entries.end(),
        [input_tokens, prospective_active](SloProfileEntry const& candidate) {
            return input_tokens <= candidate.max_input_tokens &&
                prospective_active <= candidate.active_requests;
        });
    if (entry == config_.entries.end()) {
        return {};
    }

    auto const predicted = entry->predicted_ttft_p95_ms;
    return {
        predicted + config_.safety_margin_ms <= config_.ttft_slo_ms,
        predicted,
    };
}

SloAdmissionPolicyConfig const& SloAdmissionPolicy::config() const noexcept {
    return config_;
}

} // namespace kimrt::llm

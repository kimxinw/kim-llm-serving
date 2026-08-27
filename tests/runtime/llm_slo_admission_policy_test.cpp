#include "runtime/slo_admission_policy.h"

#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {

using kimrt::llm::AdmissionSnapshot;
using kimrt::llm::SloAdmissionPolicy;
using kimrt::llm::SloAdmissionPolicyConfig;
using kimrt::llm::SloProfileEntry;

bool expect(bool condition, std::string_view message, int& failures) {
    if (condition) {
        return true;
    }
    ++failures;
    std::cerr << "[FAIL] " << message << '\n';
    return false;
}

AdmissionSnapshot snapshot(std::size_t active, bool accepting = true) {
    return {
        accepting,
        active,
        active * 16,
        active * 32,
        8,
        1024,
        256,
    };
}

bool constructionThrows(SloAdmissionPolicyConfig config) {
    try {
        SloAdmissionPolicy policy(std::move(config));
        (void)policy;
    } catch (std::invalid_argument const&) {
        return true;
    }
    return false;
}

void testValidation(int& failures) {
    expect(
        constructionThrows({"model", "revision", "engine", 0.0, 0.0, {{32, 1, 10.0}}}),
        "zero TTFT SLO must be rejected",
        failures);
    expect(
        constructionThrows({"model", "revision", "engine", 50.0, -1.0, {{32, 1, 10.0}}}),
        "negative safety margin must be rejected",
        failures);
    expect(
        constructionThrows({"model", "revision", "engine", 50.0, 1.0, {}}),
        "empty profile must be rejected",
        failures);
    expect(
        constructionThrows({
            "model", "revision", "engine", 50.0,
            1.0,
            {{32, 1, 10.0}, {32, 1, 11.0}},
        }),
        "duplicate profile buckets must be rejected",
        failures);
}

void testConservativeBucketSelection(int& failures) {
    SloAdmissionPolicy policy({
        "model",
        "revision",
        "engine",
        50.0,
        5.0,
        {
            {128, 4, 48.0},
            {32, 2, 30.0},
            {128, 2, 40.0},
            {32, 4, 44.0},
        },
    });

    auto short_low_load = policy.evaluate(17, snapshot(1));
    expect(
        short_low_load.admitted &&
            short_low_load.predicted_ttft_p95_ms == 30.0,
        "policy must choose the smallest covering input and active bucket",
        failures);

    auto short_higher_load = policy.evaluate(17, snapshot(2));
    expect(
        short_higher_load.admitted &&
            short_higher_load.predicted_ttft_p95_ms == 44.0,
        "policy must move to the next active bucket",
        failures);

    auto long_higher_load = policy.evaluate(100, snapshot(2));
    expect(
        !long_higher_load.admitted &&
            long_higher_load.predicted_ttft_p95_ms == 48.0,
        "safety margin must reject a predicted SLO miss",
        failures);

    expect(
        !policy.evaluate(129, snapshot(0)).admitted,
        "an uncovered input bucket must be conservatively rejected",
        failures);
    expect(
        !policy.evaluate(17, snapshot(4)).admitted,
        "an uncovered active bucket must be conservatively rejected",
        failures);
    expect(
        !policy.evaluate(17, snapshot(0, false)).admitted,
        "a closed hard Admission must not pass the soft policy",
        failures);
}

void testHardLimitRemainsAuthoritative(int& failures) {
    SloAdmissionPolicy policy({
        "model", "revision", "engine", 50.0, 5.0, {{32, 8, 20.0}}
    });
    auto saturated = snapshot(8);
    auto decision = policy.evaluate(17, saturated);
    expect(
        decision.admitted && decision.predicted_ttft_p95_ms == 0.0,
        "hard active limit must retain its own rejection reason",
        failures);
}

} // namespace

int main() {
    int failures{0};
    testValidation(failures);
    testConservativeBucketSelection(failures);
    testHardLimitRemainsAuthoritative(failures);
    if (failures == 0) {
        std::cout << "[PASS] LLM SLO Admission Policy contract\n";
    }
    return failures == 0 ? 0 : 1;
}

#pragma once
#include <chrono>
#include <cstdint>
#include <string>

namespace kimrt {
    struct RequestContext{
        std::uint64_t request_id{0};
        int priority{0};
        std::chrono::steady_clock::time_point deadline {
            std::chrono::steady_clock::time_point::max()
        };

        std::string trace_id;
        [[nodiscard]] bool hasDeadline() const noexcept {
            return deadline != std::chrono::steady_clock::time_point::max();
        }
    };

}//namesapce kimrt
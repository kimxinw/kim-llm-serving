#pragma once
#include "core/request_context.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace kimrt::llm {
    using TokenId = std::int32_t;

    struct SamplingParams {
        float temperature{1.0F};
        std::int32_t top_k{1};
        float top_p{1.0F};
        std::uint64_t random_seed{0};
    };

    struct GenerationRequest {
        RequestContext context;
        std::string model_name;
        std::vector<TokenId> input_token_ids;

        std::size_t max_new_tokens{32};
        bool streaming{false};
        SamplingParams sampling;

        std::optional<TokenId>end_id;
        std::optional<TokenId>pad_id;
        std::vector<TokenId> stop_token_ids;
    };

    enum class FinishReason : std::uint8_t{
        Eos,
        Length,
        Stop,
        Cancelled,
        Timeout,
    };

    struct TokenChunk {
        std::uint64_t request_id{0};
        std::vector<TokenId> token_ids;
    };
}
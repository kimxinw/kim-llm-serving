#pragma once
#include "core/request_context.h"
#include "core/status.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

/*
*edited on 2026/07/22
*删除了GenerationRequest的model_name，模型路由以后由ModelBundle负责
*stop_token_ids改为stop_sequences，现在可表达{{1,2},{3,4,5}}这样的多token停止序列
*/

namespace kimrt::llm {
    using TokenId = std::int32_t;
    using StopSequence = std::vector<TokenId>;

    struct SamplingParams {
        float temperature{1.0F};
        std::int32_t top_k{1};
        float top_p{1.0F};
        std::uint64_t random_seed{0};
    };

    struct GenerationRequest {
        RequestContext context;
        std::vector<TokenId> input_token_ids;

        std::size_t max_new_tokens{32};
        bool streaming{false};
        SamplingParams sampling;

        std::optional<TokenId>end_id;
        std::optional<TokenId>pad_id;
        std::vector<StopSequence> stop_sequences;
    };

    enum class FinishReason : std::uint8_t{
        Eos,
        Length,
        Stop,
        Cancelled,
        Timeout,
        Backpressure,
    };

    struct TokenDelta {
        std::uint64_t request_id{0};
        std::uint64_t sequence_no{0};
        std::vector<TokenId> token_ids;
    };

    struct GenerationUsage {
        std::size_t prompt_tokens{0};
        std::size_t completion_tokens{0};
    };

    struct TerminalEvent {
        std::uint64_t request_id{0};
        Status status;
        std::optional<FinishReason> finish_reason;
        GenerationUsage usage;
    };

    using GenerationEvent = std::variant<TokenDelta,TerminalEvent>;

}//namespcace kimrt::llm
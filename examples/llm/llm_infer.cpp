#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <vector>

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

namespace fs = std::filesystem;
namespace tle = tensorrt_llm::executor;

namespace
{

constexpr std::size_t kMaxInputTokens = 512;
constexpr std::size_t kMaxSequenceTokens = 544;
constexpr tle::SizeType32 kMaxOutputTokens = 32;
constexpr tle::SizeType32 kBeamWidth = 1;

bool parseInt32(std::string_view text, std::int32_t& value)
{
    if (text.empty())
    {
        return false;
    }

    auto const* begin = text.data();
    auto const* end = text.data() + text.size();

    auto const [ptr, error] = std::from_chars(begin, end, value);

    return error == std::errc{} && ptr == end;
}

void printTokens(
    std::string_view label,
    std::vector<tle::TokenIdType> const& tokens)
{
    std::cout << label << "=[";

    for (std::size_t index = 0; index < tokens.size(); ++index)
    {
        if (index != 0)
        {
            std::cout << ", ";
        }

        std::cout << tokens[index];
    }

    std::cout << "]\n";
}

std::string_view finishReasonName(tle::FinishReason reason)
{
    switch (reason)
    {
    case tle::FinishReason::kNOT_FINISHED:
        return "not_finished";
    case tle::FinishReason::kEND_ID:
        return "end_id";
    case tle::FinishReason::kSTOP_WORDS:
        return "stop_words";
    case tle::FinishReason::kLENGTH:
        return "length";
    case tle::FinishReason::kTIMED_OUT:
        return "timed_out";
    case tle::FinishReason::kCANCELLED:
        return "cancelled";
    }

    return "unknown";
}

void printUsage(char const* program)
{
    std::cerr
        << "Usage: " << program
        << " <engine_dir> <max_new_tokens> <token_id>...\n"
        << "Example: " << program
        << " /path/to/engine 5 1 2 3 4\n";
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        printUsage(argv[0]);
        return 2;
    }

    fs::path const engineDir{argv[1]};

    if (!fs::is_directory(engineDir))
    {
        std::cerr
            << "Invalid Engine directory: "
            << engineDir << '\n';
        return 2;
    }

    if (!fs::is_regular_file(engineDir / "config.json"))
    {
        std::cerr
            << "Engine config.json does not exist: "
            << engineDir / "config.json" << '\n';
        return 2;
    }

    tle::SizeType32 maxNewTokens{0};

    if (!parseInt32(argv[2], maxNewTokens)
        || maxNewTokens <= 0
        || maxNewTokens > kMaxOutputTokens)
    {
        std::cerr
            << "max_new_tokens must be an integer in [1, "
            << kMaxOutputTokens << "]\n";
        return 2;
    }

    tle::VecTokens inputTokens;
    inputTokens.reserve(static_cast<std::size_t>(argc - 3));

    for (int index = 3; index < argc; ++index)
    {
        tle::TokenIdType tokenId{0};

        if (!parseInt32(argv[index], tokenId) || tokenId < 0)
        {
            std::cerr
                << "Invalid token id: "
                << argv[index] << '\n';
            return 2;
        }

        inputTokens.push_back(tokenId);
    }

    if (inputTokens.empty())
    {
        std::cerr << "At least one input token is required\n";
        return 2;
    }

    if (inputTokens.size() > kMaxInputTokens)
    {
        std::cerr
            << "Input token count exceeds Engine limit "
            << kMaxInputTokens << '\n';
        return 2;
    }

    auto const totalTokens
        = inputTokens.size()
        + static_cast<std::size_t>(maxNewTokens);

    if (totalTokens > kMaxSequenceTokens)
    {
        std::cerr
            << "input_tokens + max_new_tokens exceeds Engine limit "
            << kMaxSequenceTokens << '\n';
        return 2;
    }

    try
    {
        initTrtLlmPlugins();

        auto const engineLoadStart
            = std::chrono::steady_clock::now();

        tle::ExecutorConfig executorConfig{kBeamWidth};

        tle::Executor executor{
            engineDir,
            tle::ModelType::kDECODER_ONLY,
            executorConfig,
        };

        auto const engineLoadEnd
            = std::chrono::steady_clock::now();

        /*
        * TensorRT-LLM 0.16 requires temperature > 0 when explicitly set.
        * top_k=1 already gives deterministic greedy generation, so the
        * temperature field is left at its valid default.
        */
        tle::SamplingConfig samplingConfig{kBeamWidth};
        samplingConfig.setTopK(1);
        samplingConfig.setSeed(0);

        tle::Request request{
            inputTokens,
            maxNewTokens,
            false,
            samplingConfig,
        };

        auto const requestStart
            = std::chrono::steady_clock::now();

        auto const requestId = executor.enqueueRequest(request);

        auto const responses = executor.awaitResponses(
            requestId,
            std::chrono::milliseconds{180000});

        auto const requestEnd
            = std::chrono::steady_clock::now();

        bool foundFinalResponse{false};
        tle::VecTokens outputTokens;
        std::optional<tle::FinishReason> finishReason;

        for (auto const& response : responses)
        {
            if (response.getRequestId() != requestId)
            {
                throw std::runtime_error(
                    "Executor returned an unexpected request id");
            }

            if (response.hasError())
            {
                throw std::runtime_error(response.getErrorMsg());
            }

            auto const& result = response.getResult();

            if (!result.isFinal)
            {
                continue;
            }

            if (result.outputTokenIds.empty())
            {
                throw std::runtime_error(
                    "Final response contains no output beam");
            }

            outputTokens = result.outputTokenIds.front();

            if (!result.finishReasons.empty())
            {
                finishReason = result.finishReasons.front();
            }

            foundFinalResponse = true;
        }

        if (!foundFinalResponse)
        {
            throw std::runtime_error(
                "No final response received before timeout");
        }

        auto const engineLoadMs
            = std::chrono::duration<double, std::milli>(
                engineLoadEnd - engineLoadStart)
                .count();

        auto const requestLatencyMs
            = std::chrono::duration<double, std::milli>(
                requestEnd - requestStart)
                .count();

        std::cout << "request_id=" << requestId << '\n';
        printTokens("input_tokens", inputTokens);
        printTokens("output_tokens", outputTokens);

        std::cout
            << "finish_reason="
            << (finishReason.has_value()
                    ? finishReasonName(*finishReason)
                    : "unknown")
            << '\n';

        std::cout
            << "engine_load_ms=" << engineLoadMs << '\n'
            << "request_latency_ms=" << requestLatencyMs << '\n';

        return 0;
    }
    catch (std::exception const& exception)
    {
        std::cerr
            << "LLM inference failed: "
            << exception.what() << '\n';
        return 3;
    }
}
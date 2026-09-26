#include "backends/kimkv/kimkv_backend.h"

#include "kim-kv/engine/engine_kv.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace kimrt::llm;
using namespace kimkvcache;

int failures = 0;

void expect(bool condition, std::string const& message) {
    if (!condition) {
        std::cerr << "[FAILED] " << message << '\n';
        ++failures;
    }
}

class FakeKvBackend final : public EngineKvBackend {
public:
    EngineKvBackendKind kind() const noexcept override {
        return EngineKvBackendKind::Heterogeneous;
    }
    EngineKvConfig config() const noexcept override { return {}; }
    EngineKvStatus createRequest(RequestId id) override {
        return requests.insert(id).second ? EngineKvStatus{} :
            EngineKvStatus{EngineKvError::RequestAlreadyExists};
    }
    EngineKvStatus forkRequest(RequestId, RequestId) override {
        return {EngineKvError::InvalidState};
    }
    EngineKvStatus releaseRequest(RequestId id) override {
        return requests.erase(id) == 1 ? EngineKvStatus{} :
            EngineKvStatus{EngineKvError::RequestNotFound};
    }
    TokenReserveResult reserveToken(ReserveTokenRequest const&) override {
        TokenReserveResult result;
        result.status = {EngineKvError::InvalidState};
        return result;
    }
    EngineKvBackendSnapshot snapshot() const override {
        return {requests.size(), 0, 0};
    }
    bool checkInvariants() const override { return true; }
private:
    std::unordered_set<RequestId> requests;
};

class FakeModelRunner final : public GenerationModelRunner {
public:
    bool slow{false};
    bool fail{false};
    TinyLlamaConfig generationConfig() const noexcept override {
        return {8, 16, 2, 4, 2, 2, 256, 256, 1, 255,
            1.0e-5F, 10000.0F, false};
    }
    GenerationStepResult generationForwardToken(
        RequestId, std::uint32_t token, std::uint32_t) override {
        if (fail) {
            throw std::runtime_error("injected model failure");
        }
        if (slow) {
            std::this_thread::sleep_for(std::chrono::milliseconds{2});
        }
        return {true, static_cast<std::uint32_t>((token + 1) % 251), {}};
    }
};

std::unique_ptr<KimKvBackend> makeBackend(
    bool slow = false, bool fail = false) {
    KimKvBackendConfig config;
    config.eos_token_id = 255;
    config.max_input_tokens = 32;
    config.max_output_tokens = 128;
    config.max_sequence_tokens = 256;
    config.max_active_requests = 2;
    config.max_batched_tokens = 4;
    config.prefill_chunk_size = 4;
    config.model_max_batch_tokens = 4;
    config.max_kv_tokens = 512;
    KimKvBackendResources resources;
    resources.kv = std::make_unique<FakeKvBackend>();
    auto runner = std::make_unique<FakeModelRunner>();
    runner->slow = slow;
    runner->fail = fail;
    resources.runner = std::move(runner);
    return std::make_unique<KimKvBackend>(
        std::move(config), std::move(resources));
}

kimrt::llm::GenerationRequest request(
    std::uint64_t id, std::size_t output_tokens) {
    kimrt::llm::GenerationRequest result;
    result.context.request_id = id;
    result.input_token_ids = {3, 5};
    result.max_new_tokens = output_tokens;
    result.end_id = 255;
    result.streaming = true;
    return result;
}

TerminalEvent collect(std::shared_ptr<GenerationMailbox> const& mailbox,
    std::vector<TokenId>& tokens, std::size_t starting_sequence = 0) {
    for (std::size_t attempt = 0; attempt < 300; ++attempt) {
        GenerationEvent event;
        auto const waited = mailbox->waitPop(event,
            std::chrono::milliseconds{100});
        if (waited != MailboxWaitResult::Event) {
            continue;
        }
        if (auto const* delta = std::get_if<TokenDelta>(&event)) {
            expect(delta->sequence_no == starting_sequence + tokens.size(),
                "delta sequence is contiguous");
            tokens.insert(tokens.end(), delta->token_ids.begin(),
                delta->token_ids.end());
        } else {
            return std::get<TerminalEvent>(std::move(event));
        }
    }
    expect(false, "terminal event arrived before timeout");
    return {};
}

void testGreedyAndRetirement() {
    auto backend = makeBackend();
    expect(backend->start().ok(), "backend starts with CPU model");
    for (std::uint32_t iteration = 0; iteration < 40; ++iteration) {
        auto mailbox = std::make_shared<GenerationMailbox>(
            GenerationMailboxConfig{16, 16});
        expect(backend->submit(request(10, 3), mailbox).ok(),
            "request id can be reused after terminal retirement");
        std::vector<TokenId> tokens;
        auto terminal = collect(mailbox, tokens);
        expect(terminal.status.ok() &&
            terminal.finish_reason == FinishReason::Length,
            "greedy request completes with length reason");
        expect(tokens == std::vector<TokenId>({6, 7, 8}),
            "tokens are forwarded in generation order");
        expect(terminal.usage.prompt_tokens == 2 &&
            terminal.usage.completion_tokens == 3,
            "terminal usage reflects delivered tokens");
    }
    backend->stop();
}

void testBackpressureAndUnsupportedParameters() {
    auto backend = makeBackend();
    expect(backend->start().ok(), "backpressure backend starts");
    auto mailbox = std::make_shared<GenerationMailbox>(
        GenerationMailboxConfig{1, 1});
    expect(backend->submit(request(20, 8), mailbox).ok(),
        "backpressure request accepted");
    std::this_thread::sleep_for(std::chrono::milliseconds{50});
    std::vector<TokenId> tokens;
    auto terminal = collect(mailbox, tokens);
    expect(terminal.status.code == kimrt::StatusCode::QueueFull &&
        terminal.finish_reason == FinishReason::Backpressure,
        "mailbox pressure produces explicit backpressure terminal");
    expect(tokens.size() == terminal.usage.completion_tokens,
        "rejected token is excluded from usage");

    auto unsupported = request(21, 2);
    unsupported.sampling.top_k = 2;
    auto const result = backend->submit(
        std::move(unsupported),
        std::make_shared<GenerationMailbox>(GenerationMailboxConfig{2, 2}));
    expect(result.code == kimrt::StatusCode::InvalidInput,
        "unsupported sampling is rejected before acceptance");
    backend->stop();
}

void testCancellationAndStop() {
    auto backend = makeBackend(true);
    expect(backend->start().ok(), "cancellation backend starts");
    auto mailbox = std::make_shared<GenerationMailbox>(
        GenerationMailboxConfig{128, 128});
    expect(backend->submit(request(30, 100), mailbox).ok(),
        "long request accepted");
    GenerationEvent first;
    expect(mailbox->waitPop(first, std::chrono::seconds{2})
        == MailboxWaitResult::Event &&
        std::holds_alternative<TokenDelta>(first),
        "first streaming token arrives before cancellation");
    backend->cancel(30);
    std::vector<TokenId> remaining;
    auto terminal = collect(mailbox, remaining, 1);
    expect(terminal.status.code == kimrt::StatusCode::Cancelled &&
        terminal.finish_reason == FinishReason::Cancelled,
        "cancel reaches one terminal");
    backend->stop();
}

void testUnexpectedSchedulerFailure() {
    auto backend = makeBackend(false, true);
    expect(backend->start().ok(), "failure backend starts");
    auto mailbox = std::make_shared<GenerationMailbox>(
        GenerationMailboxConfig{4, 4});
    expect(backend->submit(request(40, 2), mailbox).ok(),
        "request is accepted before model failure");
    std::vector<TokenId> tokens;
    auto terminal = collect(mailbox, tokens);
    expect(terminal.status.code == kimrt::StatusCode::InternalError,
        "unexpected model exception produces a terminal");
    auto next = backend->submit(request(41, 2),
        std::make_shared<GenerationMailbox>(GenerationMailboxConfig{4, 4}));
    expect(next.code == kimrt::StatusCode::NotReady,
        "failed scheduler rejects later requests without hanging");
    backend->stop();
}

} // namespace

int main() {
    testGreedyAndRetirement();
    testBackpressureAndUnsupportedParameters();
    testCancellationAndStop();
    testUnexpectedSchedulerFailure();
    if (failures == 0) {
        std::cout << "KimKV backend CPU contract passed\n";
    }
    return failures == 0 ? 0 : 1;
}

#pragma once

#include "runtime/generation_backend.h"

#include "kim-kv/engine/generation.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace kimrt::llm {

struct KimKvBackendConfig {
    std::filesystem::path weights_manifest;
    std::filesystem::path weights_data;
    std::uint32_t eos_token_id{0};
    std::uint32_t max_input_tokens{0};
    std::uint32_t max_output_tokens{0};
    std::uint32_t max_sequence_tokens{0};
    std::uint32_t max_active_requests{0};
    std::uint32_t max_batched_tokens{32};
    std::uint32_t prefill_chunk_size{16};
    std::uint32_t model_max_batch_tokens{32};
    std::uint64_t max_kv_tokens{0};
    std::uint32_t micro_page_capacity{0};
    std::uint32_t extent_page_capacity{0};
};

// Destruction order is runner, KV backend, then its optional CUDA stream.
struct KimKvBackendResources {
    std::shared_ptr<void> stream_owner;
    std::unique_ptr<kimkvcache::EngineKvBackend> kv;
    std::unique_ptr<kimkvcache::GenerationModelRunner> runner;
};

class KimKvBackend final : public GenerationBackend {
public:
    KimKvBackend(KimKvBackendConfig config, KimKvBackendResources resources);
    ~KimKvBackend() override;

    KimKvBackend(KimKvBackend const&) = delete;
    KimKvBackend& operator=(KimKvBackend const&) = delete;

    Status start() override;
    Status submit(
        GenerationRequest request,
        std::shared_ptr<GenerationMailbox> mailbox) override;
    void cancel(std::uint64_t request_id) override;
    void stop() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Creates and validates the CUDA model resources before Worker readiness.
[[nodiscard]] std::unique_ptr<GenerationBackend> createCudaKimKvBackend(
    KimKvBackendConfig config);

} // namespace kimrt::llm

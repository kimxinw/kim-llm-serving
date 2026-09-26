#include "backends/kimkv/kimkv_backend.h"

#include "kim-kv/cuda/cuda_engine_kv_backend.h"
#include "kim-kv/cuda/cuda_model_runner.h"
#include "kim-kv/model/weight_manifest.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace kimrt::llm {

std::unique_ptr<GenerationBackend> createCudaKimKvBackend(
    KimKvBackendConfig config) {
    if (config.weights_manifest.empty() || config.weights_data.empty() ||
        config.micro_page_capacity == 0 || config.extent_page_capacity == 0 ||
        config.model_max_batch_tokens == 0 || config.max_kv_tokens == 0) {
        throw std::invalid_argument("KimKV CUDA weights and capacities are required");
    }
    auto loaded = kimkvcache::loadWeightManifest(
        config.weights_manifest.string());
    if (!loaded.ok()) {
        throw std::runtime_error("KimKV weight manifest: " +
            loaded.status.detail);
    }
    auto const model = loaded.manifest.config;
    if (loaded.manifest.data_type != "fp16" ||
        config.eos_token_id != model.eos_token_id ||
        config.max_sequence_tokens > model.max_position_embeddings ||
        config.max_kv_tokens >
            static_cast<std::uint64_t>(config.micro_page_capacity) * 8U +
                static_cast<std::uint64_t>(config.extent_page_capacity) * 64U) {
        throw std::invalid_argument(
            "KimKV service limits do not match the FP16 model or KV capacity");
    }

    cudaStream_t stream = nullptr;
    auto const created_stream = cudaStreamCreate(&stream);
    if (created_stream != cudaSuccess) {
        throw std::runtime_error(std::string{"cudaStreamCreate: "} +
            cudaGetErrorString(created_stream));
    }
    KimKvBackendResources resources;
    resources.stream_owner = std::shared_ptr<void>(stream, [](void* value) {
        static_cast<void>(cudaStreamDestroy(
            reinterpret_cast<cudaStream_t>(value)));
    });

    auto const attention_workspace =
        static_cast<std::size_t>(model.attention_head_count) *
        model.max_position_embeddings * sizeof(float);
    kimkvcache::EngineKvConfig kv_config{
        kimkvcache::KvLayout{
            model.layer_count, model.kv_head_count, model.head_dimension,
        },
        model.attention_head_count,
        attention_workspace,
    };
    resources.kv = kimkvcache::createHeterogeneousCudaEngineKvBackend(
        kv_config, config.micro_page_capacity, config.extent_page_capacity);
    if (!resources.kv) {
        throw std::runtime_error("failed to create KimKV CUDA KV backend");
    }
    auto created_runner = kimkvcache::createCudaTinyLlamaModelRunner(
        config.weights_manifest.string(), config.weights_data.string(),
        *resources.kv, reinterpret_cast<kimkvcache::EngineStream>(stream),
        kimkvcache::CudaModelRunnerOptions{config.model_max_batch_tokens});
    if (!created_runner.ok()) {
        throw std::runtime_error("failed to load KimKV CUDA model: " +
            created_runner.status.detail);
    }
    resources.runner = std::move(created_runner.runner);
    return std::make_unique<KimKvBackend>(
        std::move(config), std::move(resources));
}

} // namespace kimrt::llm

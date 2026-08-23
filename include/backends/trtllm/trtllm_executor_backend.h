#pragma once

#include "runtime/generation_backend.h"

#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace kimrt::llm {
    struct TrtLlmBackendConfig {
        std::filesystem::path engine_dir;
        std::size_t max_input_tokens{512};
        std::size_t max_sequence_tokens{544};
        std::size_t max_output_tokens{32};
        // Executor 内部 iteration stats 缓冲区的容量上限。
        // TensorRT-LLM 默认 kDefaultIterStatsMaxIterations = 1000
        // (executor.h:1273)，写满后静默淘汰最旧条目、不报错也不计数。
        // 这里显式放大只是安全网；真正的保证来自调用方在测量期间周期 drain。

        //关于 iteration :
        //在 LLM 推理中，一次迭代通常指生成一个 Token 的过程。
        //TensorRT-LLM 的执行器（Executor）会循环执行这些迭代，直到完成所有请求
        std::int32_t iter_stats_max_iterations{100000};
    };

    class TrtLlmExecutorBackend final: public GenerationBackend{
    public:
        explicit TrtLlmExecutorBackend(TrtLlmBackendConfig config);
        ~TrtLlmExecutorBackend()override;

        TrtLlmExecutorBackend(const TrtLlmExecutorBackend&) = delete;
        TrtLlmExecutorBackend& operator = (const TrtLlmExecutorBackend&) = delete;

        TrtLlmExecutorBackend(TrtLlmExecutorBackend&&) = delete;
        TrtLlmExecutorBackend& operator = (TrtLlmExecutorBackend&&) = delete;

        Status start()override;
        Status submit(GenerationRequest request,std::shared_ptr<GenerationMailbox>mailbox)override;
        
        void cancel(std::uint64_t request_id) override;

        void stop() override;

        [[nodiscard]]std::vector<std::string> drainIterationStatsJson();
    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

}//namespace kimrt::llm
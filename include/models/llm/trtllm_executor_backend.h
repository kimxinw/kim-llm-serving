#pragma once

#include "models/llm/generation_backend.h"

#include <cstddef>
#include <filesystem>
#include <memory>

namespace kimrt::llm {
    struct TrtLlmBackendConfig {
        std::filesystem::path engine_dir;
        std::size_t max_input_tokens{512};
        std::size_t max_sequence_tokens{544};
        std::size_t max_output_tokens{32};
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
    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

}//namespace kimrt::llm
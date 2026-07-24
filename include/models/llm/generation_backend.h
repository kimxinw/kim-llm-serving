#pragma once

#include "core/status.h"
#include "models/llm/generation_mailbox.h"
#include "models/llm/generation_types.h"

#include <cstdint>
#include <memory>

namespace kimrt::llm{
class GenerationBackend {
public:
    virtual ~GenerationBackend() = default;
    virtual Status start() = 0;
    virtual Status submit(
            GenerationRequest request,
            std::shared_ptr<GenerationMailbox> mailbox
    ) = 0;

    virtual void cancel(std::uint64_t request_id) = 0;
    virtual void stop() = 0;
};

}//namespace kimrt::llm
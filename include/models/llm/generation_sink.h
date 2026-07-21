#pragma once

#include "core/status.h"
#include "models/llm/generation_types.h"

namespace kimrt::llm{

class GenerationSink {
public:
    virtual ~GenerationSink() = default;
    virtual void onTokens(TokenChunk chunk) = 0;
    virtual void onComplete(FinishReason reason) = 0;
    virtual void onError(Status status) = 0;
};

}//namespace kimrt::llm
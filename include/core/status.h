#pragma once

#include <cstdint>
#include <string>
#include <utility>

namespace kimrt {
    enum class StatusCode: uint8_t{
        Ok,
        InvalidInput,
        InvalidShape,
        QueueFull,
        EngineNotFound,
        ContextUnavailable,
        CudaError,
        TensorRTError,
        InternalError,
        Timeout,
        Cancelled,
    };

    struct Status{
        StatusCode code{StatusCode::Ok};
        std::string message;

        static Status error(StatusCode code,std::string message){
            return Status{code ,std::move(message)};
        }

        static Status success(){
            return {};
        }

        bool ok() const noexcept {
            return code == StatusCode::Ok;
        }

        explicit operator bool() const noexcept {
            return ok();
        }
    };

}//namespace kimrt
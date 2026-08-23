#include "runtime/generation_types.h"

#include <cstdint>
#include <iostream>
#include <string_view>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {
    using kimrt::Status;
    using kimrt::StatusCode;
    using kimrt::llm::FinishReason;
    using kimrt::llm::GenerationEvent;
    using kimrt::llm::GenerationRequest;
    using kimrt::llm::GenerationUsage;
    using kimrt::llm::StopSequence;
    using kimrt::llm::TerminalEvent;
    using kimrt::llm::TokenDelta;
    using kimrt::llm::TokenId;

    bool expect (bool condition,std::string_view message,int& failures){
        if(condition){
            return true;
        }

        ++failures;
        std::cerr<<"[FAIL]"<<message<<"\n";
        return false;
    }

    int run(){
        static_assert(std::is_same_v<TokenId,std::int32_t>);
        static_assert(std::is_same_v<StopSequence,std::vector<TokenId>>);

        int failures{0};

        GenerationRequest request;
        request.context.request_id = 17;
        request.input_token_ids = {1,2,3,4};
        request.stop_sequences = {
            {10},
            {20,21},
            {30,31,32}
        };

        expect(request.stop_sequences.size() == 3,
                "request must preserve every stop sequence",failures);
        expect(request.stop_sequences[1] == StopSequence({20,21}),
                "multi-token stop sequence must preserve token order",failures);
        
        TokenDelta delta;
        delta.request_id = request.context.request_id;
        delta.sequence_no = 0;
        delta.token_ids = {100,101};

        GenerationEvent event(std::move(delta));

        expect(std::holds_alternative<TokenDelta>(event),
        "GenerationEvent must hold TokenDelta", failures);

        auto const *storedDelta = std::get_if<TokenDelta>(&event);
        expect(storedDelta != nullptr,
            "TokenDelta must be retrievable from GenerationEvent", failures);

        if (storedDelta != nullptr) {
        expect(storedDelta->request_id == 17,
                "TokenDelta must preserve request id", failures);
        expect(storedDelta->sequence_no == 0,
                "first TokenDelta sequence number must be zero", failures);
        expect(storedDelta->token_ids == std::vector<TokenId>({100, 101}),
                "TokenDelta must preserve token order", failures);
        }

        TerminalEvent terminal;
        terminal.request_id = request.context.request_id;
        terminal.status = Status::success();
        terminal.finish_reason = FinishReason::Length;
        terminal.usage = GenerationUsage{
            request.input_token_ids.size(),
            2,
        };

        event = std::move(terminal);

        expect(std::holds_alternative<TerminalEvent>(event),
            "GenerationEvent must hold TerminalEvent", failures);

        auto const *storedTerminal = std::get_if<TerminalEvent>(&event);
        expect(storedTerminal != nullptr,
            "TerminalEvent must be retrievable from GenerationEvent", failures);

        if (storedTerminal != nullptr) {
        expect(storedTerminal->request_id == 17,
                "TerminalEvent must preserve request id", failures);
        expect(storedTerminal->status.ok(),
                "successful TerminalEvent must preserve success status", failures);
        expect(storedTerminal->finish_reason == FinishReason::Length,
                "TerminalEvent must preserve finish reason", failures);
        expect(storedTerminal->usage.prompt_tokens == 4,
                "TerminalEvent must preserve prompt usage", failures);
        expect(storedTerminal->usage.completion_tokens == 2,
                "TerminalEvent must preserve completion usage", failures);
        }

        TerminalEvent errorTerminal;
        errorTerminal.request_id = 18;
        errorTerminal.status =
            Status::error(StatusCode::InternalError, "synthetic failure");
        errorTerminal.finish_reason = std::nullopt;

        event = std::move(errorTerminal);
        storedTerminal = std::get_if<TerminalEvent>(&event);

        expect(storedTerminal != nullptr,
            "error terminal must use the same TerminalEvent type", failures);

        if (storedTerminal != nullptr) {
        expect(!storedTerminal->status.ok(),
                "error terminal must preserve error status", failures);
        expect(!storedTerminal->finish_reason.has_value(),
                "internal error may omit finish reason", failures);
        }

        TerminalEvent backpressureTerminal;
        backpressureTerminal.request_id = 19;
        backpressureTerminal.status =
            Status::error(StatusCode::QueueFull, "output budget exhausted");
        backpressureTerminal.finish_reason = FinishReason::Backpressure;

        event = std::move(backpressureTerminal);
        storedTerminal = std::get_if<TerminalEvent>(&event);

        expect(storedTerminal != nullptr &&
                storedTerminal->finish_reason == FinishReason::Backpressure,
            "Backpressure must be expressible as a terminal finish reason",
            failures);

        if (failures == 0) {
        std::cout << "[PASS] LLM generation type contract\n";
        }

        return failures;
    }

}//namespace

int main(){
    return run() == 0 ? 0:1;
}
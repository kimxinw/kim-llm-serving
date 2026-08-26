#include "backends/trtllm/trtllm_executor_backend.h"
#include "ipc/ipc_protocol.h"
#include "runtime/generation_runtime.h"
#include "worker/worker_server.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <time.h>
#include <type_traits>
#include <unistd.h>
#include <utility>

namespace {

using Json = nlohmann::json;
using kimrt::Status;
using kimrt::llm::AdmissionConfig;
using kimrt::llm::GenerationMailboxConfig;
using kimrt::llm::GenerationRuntime;
using kimrt::llm::GenerationRuntimeConfig;
using kimrt::llm::RuntimeBridgeConfig;
using kimrt::llm::SloAdmissionPolicyConfig;
using kimrt::llm::SloProfileEntry;
using kimrt::llm::TrtLlmBackendConfig;
using kimrt::llm::TrtLlmExecutorBackend;
using kimrt::llm::WorkerServer;
using kimrt::llm::WorkerServerConfig;
using kimrt::llm::ipc::ModelManifest;
using kimrt::llm::ipc::WorkerLimits;

struct WorkerApplicationConfig {
    std::filesystem::path engine_dir;
    std::string socket_path;
    ModelManifest manifest;
    WorkerLimits limits;
    std::optional<SloAdmissionPolicyConfig> slo_policy;
};

template <typename Value>
[[nodiscard]] Value requireRepresentable(
    std::uint64_t input,
    char const* description) {
    if (input > static_cast<std::uint64_t>(
                    std::numeric_limits<Value>::max())) {
        throw std::runtime_error(
            std::string{description} +
            " exceeds the local platform range");
    }
    return static_cast<Value>(input);
}

class WorkerConfigurationLoader final {
public:
    explicit WorkerConfigurationLoader(std::filesystem::path path)
        : path_(std::move(path)) {}

    [[nodiscard]] WorkerApplicationConfig load() const {
        std::ifstream input(path_);
        if (!input) {
            throw std::runtime_error(
                "failed to open worker configuration: " + path_.string());
        }

        Json root;
        try {
            input >> root;
        } catch (Json::exception const& exception) {
            throw std::runtime_error(
                std::string{"failed to parse worker configuration: "} +
                exception.what());
        }

        if (root.contains("slo_policy")) {
            requireExactFields(
                root,
                {"engine_dir", "socket_path", "manifest", "limits",
                 "slo_policy"},
                "worker configuration");
        } else {
            requireExactFields(
                root,
                {"engine_dir", "socket_path", "manifest", "limits"},
                "worker configuration");
        }

        WorkerApplicationConfig config;
        config.engine_dir = requireString(root, "engine_dir");
        config.socket_path = requireString(root, "socket_path");
        config.manifest = parseManifest(root.at("manifest"));
        config.limits = parseLimits(root.at("limits"));
        if (root.contains("slo_policy")) {
            config.slo_policy = parseSloPolicy(root.at("slo_policy"));
            if (config.slo_policy->model_id != config.manifest.model_id ||
                config.slo_policy->revision != config.manifest.revision ||
                config.slo_policy->engine_fingerprint !=
                    config.manifest.engine_fingerprint) {
                throw std::runtime_error(
                    "slo_policy identity does not match the Worker manifest");
            }
        }
        if (config.engine_dir.empty() || config.socket_path.empty()) {
            throw std::runtime_error(
                "engine_dir and socket_path must not be empty");
        }
        auto const encoded = kimrt::llm::ipc::encodePayload(
            kimrt::llm::ipc::Message{kimrt::llm::ipc::HelloAck{
                kimrt::llm::ipc::kProtocolVersion,
                1,
                config.manifest,
                config.limits,
            }});
        if (!encoded.ok()) {
            throw std::runtime_error(
                "invalid Worker protocol configuration: " +
                encoded.status.message);
        }
        return config;
    }

private:
    static void requireExactFields(
        Json const& object,
        std::initializer_list<std::string_view> fields,
        std::string_view description) {
        if (!object.is_object() || object.size() != fields.size()) {
            throw std::runtime_error(
                std::string{description} +
                " contains missing or unexpected fields");
        }
        for (auto const field : fields) {
            if (!object.contains(std::string{field})) {
                throw std::runtime_error(
                    std::string{description} +
                    " is missing field " + std::string{field});
            }
        }
    }

    static std::string requireString(
        Json const& object,
        char const* key) {
        auto const& value = object.at(key);
        if (!value.is_string()) {
            throw std::runtime_error(
                std::string{key} + " must be a string");
        }
        auto result = value.get<std::string>();
        if (result.empty()) {
            throw std::runtime_error(
                std::string{key} + " must not be empty");
        }
        return result;
    }

    template <typename Integer>
    static Integer requireInteger(
        Json const& object,
        char const* key) {
        auto const& value = object.at(key);
        if (!value.is_number_integer() && !value.is_number_unsigned()) {
            throw std::runtime_error(
                std::string{key} + " must be an integer");
        }

        if constexpr (std::is_signed_v<Integer>) {
            if (value.is_number_unsigned()) {
                auto const input = value.get<std::uint64_t>();
                if (input > static_cast<std::uint64_t>(
                                std::numeric_limits<Integer>::max())) {
                    throw std::runtime_error(
                        std::string{key} +
                        " is outside the supported range");
                }
                return static_cast<Integer>(input);
            }
            auto const input = value.get<std::int64_t>();
            if (input < static_cast<std::int64_t>(
                            std::numeric_limits<Integer>::min()) ||
                input > static_cast<std::int64_t>(
                            std::numeric_limits<Integer>::max())) {
                throw std::runtime_error(
                    std::string{key} +
                    " is outside the supported range");
            }
            return static_cast<Integer>(input);
        } else {
            if (!value.is_number_unsigned()) {
                auto const input = value.get<std::int64_t>();
                if (input < 0) {
                    throw std::runtime_error(
                        std::string{key} +
                        " is outside the supported range");
                }
            }
            auto const input = value.get<std::uint64_t>();
            if (input > static_cast<std::uint64_t>(
                            std::numeric_limits<Integer>::max())) {
                throw std::runtime_error(
                    std::string{key} +
                    " is outside the supported range");
            }
            return static_cast<Integer>(input);
        }
    }

    static double requireNumber(
        Json const& object,
        char const* key,
        bool allow_zero = false) {
        auto const& value = object.at(key);
        if (!value.is_number()) {
            throw std::runtime_error(std::string{key} + " must be numeric");
        }
        auto const result = value.get<double>();
        if (!std::isfinite(result) ||
            (allow_zero ? result < 0.0 : result <= 0.0)) {
            throw std::runtime_error(
                std::string{key} + " is outside the supported range");
        }
        return result;
    }

    static ModelManifest parseManifest(Json const& object) {
        requireExactFields(
            object,
            {"model_id", "revision", "tokenizer_fingerprint",
             "chat_template_fingerprint", "engine_fingerprint",
             "eos_token_id", "pad_token_id", "max_input_tokens",
             "max_output_tokens", "max_sequence_tokens", "precision",
             "max_batch_size"},
            "manifest");

        ModelManifest manifest;
        manifest.model_id = requireString(object, "model_id");
        manifest.revision = requireString(object, "revision");
        manifest.tokenizer_fingerprint =
            requireString(object, "tokenizer_fingerprint");
        manifest.chat_template_fingerprint =
            requireString(object, "chat_template_fingerprint");
        manifest.engine_fingerprint =
            requireString(object, "engine_fingerprint");
        manifest.eos_token_id =
            requireInteger<std::int32_t>(object, "eos_token_id");
        manifest.pad_token_id =
            requireInteger<std::int32_t>(object, "pad_token_id");
        manifest.max_input_tokens =
            requireInteger<std::uint32_t>(object, "max_input_tokens");
        manifest.max_output_tokens =
            requireInteger<std::uint32_t>(object, "max_output_tokens");
        manifest.max_sequence_tokens =
            requireInteger<std::uint32_t>(object, "max_sequence_tokens");
        manifest.precision = requireString(object, "precision");
        manifest.max_batch_size =
            requireInteger<std::uint32_t>(object, "max_batch_size");
        return manifest;
    }

    static WorkerLimits parseLimits(Json const& object) {
        requireExactFields(
            object,
            {"max_active_requests", "max_total_input_tokens",
             "max_reserved_output_tokens", "max_frame_payload_bytes",
             "max_session_egress_frames", "max_session_egress_bytes",
             "max_request_egress_frames", "max_request_egress_bytes"},
            "limits");

        WorkerLimits limits;
        limits.max_active_requests =
            requireInteger<std::uint32_t>(object, "max_active_requests");
        limits.max_total_input_tokens = requireInteger<std::uint64_t>(
            object,
            "max_total_input_tokens");
        limits.max_reserved_output_tokens = requireInteger<std::uint64_t>(
            object,
            "max_reserved_output_tokens");
        limits.max_frame_payload_bytes = requireInteger<std::uint32_t>(
            object,
            "max_frame_payload_bytes");
        limits.max_session_egress_frames = requireInteger<std::uint32_t>(
            object,
            "max_session_egress_frames");
        limits.max_session_egress_bytes = requireInteger<std::uint64_t>(
            object,
            "max_session_egress_bytes");
        limits.max_request_egress_frames = requireInteger<std::uint32_t>(
            object,
            "max_request_egress_frames");
        limits.max_request_egress_bytes = requireInteger<std::uint64_t>(
            object,
            "max_request_egress_bytes");
        return limits;
    }

    static SloAdmissionPolicyConfig parseSloPolicy(Json const& object) {
        requireExactFields(
            object,
            {"model_id", "revision", "engine_fingerprint", "ttft_slo_ms",
             "safety_margin_ms", "entries"},
            "slo_policy");
        auto const& entries = object.at("entries");
        if (!entries.is_array() || entries.empty()) {
            throw std::runtime_error("slo_policy.entries must be a non-empty array");
        }

        SloAdmissionPolicyConfig policy;
        policy.model_id = requireString(object, "model_id");
        policy.revision = requireString(object, "revision");
        policy.engine_fingerprint =
            requireString(object, "engine_fingerprint");
        policy.ttft_slo_ms = requireNumber(object, "ttft_slo_ms");
        policy.safety_margin_ms =
            requireNumber(object, "safety_margin_ms", true);
        policy.entries.reserve(entries.size());
        for (auto const& entry : entries) {
            requireExactFields(
                entry,
                {"max_input_tokens", "active_requests",
                 "predicted_ttft_p95_ms"},
                "slo_policy entry");
            policy.entries.push_back(SloProfileEntry{
                requireInteger<std::size_t>(entry, "max_input_tokens"),
                requireInteger<std::size_t>(entry, "active_requests"),
                requireNumber(entry, "predicted_ttft_p95_ms"),
            });
        }
        return policy;
    }

    std::filesystem::path path_;
};

class WorkerApplication final {
public:
    explicit WorkerApplication(WorkerApplicationConfig config)
        : config_(std::move(config)) {}

    [[nodiscard]] int run() {
        blockTerminationSignals();

        auto backend_config = makeBackendConfig();
        auto runtime_config = makeRuntimeConfig();
        auto server_config = makeServerConfig();
        GenerationRuntime runtime(
            std::make_unique<TrtLlmExecutorBackend>(
                std::move(backend_config)),
            std::move(runtime_config));

        auto status = runtime.start();
        if (!status.ok()) {
            reportFailure("failed to start GenerationRuntime", status);
            return 3;
        }

        WorkerServer server(runtime, std::move(server_config));
        status = server.start();
        if (!status.ok()) {
            reportFailure("failed to start WorkerServer", status);
            static_cast<void>(runtime.stop());
            return 4;
        }

        std::cout << "llm_worker ready on " << config_.socket_path << '\n';
        status = waitForShutdown(server);

        auto const server_stop = server.stop();
        auto const runtime_stop = runtime.stop();

        if (!status.ok()) {
            reportFailure("llm_worker terminated unexpectedly", status);
            return 5;
        }
        if (!server_stop.ok()) {
            reportFailure("WorkerServer stop failed", server_stop);
            return 6;
        }
        if (!runtime_stop.ok()) {
            reportFailure("GenerationRuntime stop failed", runtime_stop);
            return 7;
        }
        return 0;
    }

private:
    void blockTerminationSignals() {
        ::sigemptyset(&termination_signals_);
        ::sigaddset(&termination_signals_, SIGINT);
        ::sigaddset(&termination_signals_, SIGTERM);
        int const result = ::pthread_sigmask(
            SIG_BLOCK,
            &termination_signals_,
            nullptr);
        if (result != 0) {
            throw std::runtime_error(
                "failed to block termination signals");
        }
    }

    [[nodiscard]] TrtLlmBackendConfig makeBackendConfig() const {
        TrtLlmBackendConfig config;
        config.engine_dir = config_.engine_dir;
        config.max_input_tokens = config_.manifest.max_input_tokens;
        config.max_output_tokens = config_.manifest.max_output_tokens;
        config.max_sequence_tokens = config_.manifest.max_sequence_tokens;
        return config;
    }

    [[nodiscard]] GenerationRuntimeConfig makeRuntimeConfig() const {
        GenerationRuntimeConfig config;
        config.admission = AdmissionConfig{
            config_.limits.max_active_requests,
            requireRepresentable<std::size_t>(
                config_.limits.max_total_input_tokens,
                "max_total_input_tokens"),
            requireRepresentable<std::size_t>(
                config_.limits.max_reserved_output_tokens,
                "max_reserved_output_tokens"),
        };
        config.mailbox = GenerationMailboxConfig{
            16,
            config_.manifest.max_output_tokens,
        };
        config.slo_policy = config_.slo_policy;
        return config;
    }

    [[nodiscard]] WorkerServerConfig makeServerConfig() const {
        if (config_.limits.max_session_egress_frames <=
                config_.limits.max_request_egress_frames ||
            config_.limits.max_session_egress_bytes <=
                config_.limits.max_request_egress_bytes) {
            throw std::runtime_error(
                "Worker limits must reserve Session capacity for control messages");
        }
        auto const frame_reserve =
            config_.limits.max_session_egress_frames -
            config_.limits.max_request_egress_frames;
        auto const byte_reserve =
            config_.limits.max_session_egress_bytes -
            config_.limits.max_request_egress_bytes;
        WorkerServerConfig server;
        server.socket_path = config_.socket_path;
        server.listen_backlog = 8;
        server.supervisor_poll_interval = std::chrono::milliseconds{50};
        server.manifest = config_.manifest;
        server.session.frame_codec.max_payload_bytes =
            config_.limits.max_frame_payload_bytes;
        server.session.max_egress_frames =
            config_.limits.max_session_egress_frames;
        server.session.max_egress_bytes =
            config_.limits.max_session_egress_bytes;
        server.session.read_buffer_bytes = 16U * 1024U;

        RuntimeBridgeConfig bridge;
        bridge.worker_epoch = generateWorkerEpoch();
        bridge.limits = config_.limits;
        bridge.control_reserve_frames =
            std::min<std::uint32_t>(16, frame_reserve);
        bridge.control_reserve_bytes =
            std::min<std::uint64_t>(64U * 1024U, byte_reserve);
        bridge.terminal_reserve_bytes = std::min<std::uint64_t>(
            2048,
            config_.limits.max_request_egress_bytes);
        server.bridge = std::move(bridge);
        return server;
    }

    [[nodiscard]] std::uint64_t generateWorkerEpoch() const noexcept {
        auto const now = static_cast<std::uint64_t>(
            std::chrono::system_clock::now().time_since_epoch().count());
        auto const process = static_cast<std::uint64_t>(::getpid());
        auto const epoch = now ^ (process << 32U);
        return epoch == 0 ? 1 : epoch;
    }

    [[nodiscard]] Status waitForShutdown(WorkerServer& server) const {
        while (server.running()) {
            timespec timeout{};
            timeout.tv_nsec = 200L * 1000L * 1000L;
            int const signal = ::sigtimedwait(
                &termination_signals_,
                nullptr,
                &timeout);
            if (signal == SIGINT || signal == SIGTERM) {
                return Status::success();
            }
            if (signal < 0 && errno == EAGAIN) {
                continue;
            }
            if (signal < 0 && errno == EINTR) {
                continue;
            }
            if (signal < 0) {
                return Status::error(
                    kimrt::StatusCode::InternalError,
                    "failed while waiting for termination signal");
            }
        }
        return server.terminalStatus();
    }

    static void reportFailure(
        std::string_view context,
        Status const& status) {
        std::cerr << context << ": " << status.message << '\n';
    }

    WorkerApplicationConfig config_;
    sigset_t termination_signals_{};
};

} // namespace

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <worker-config.json>\n";
        return 2;
    }

    try {
        WorkerConfigurationLoader loader(argv[1]);
        WorkerApplication application(loader.load());
        return application.run();
    } catch (std::exception const& exception) {
        std::cerr << "llm_worker configuration failed: "
                  << exception.what() << '\n';
        return 2;
    }
}

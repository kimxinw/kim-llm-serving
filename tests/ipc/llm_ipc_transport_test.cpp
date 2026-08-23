#include "ipc/ipc_transport.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace kimrt;
using namespace kimrt::llm::ipc;
using namespace std::chrono_literals;

void expect(bool condition, std::string const& message, int& failures) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << message << '\n';
    }
}

class TemporarySocketDirectory final {
public:
    TemporarySocketDirectory() {
        char path[] = "/tmp/kimrt-ipc-XXXXXX";
        char* const created = ::mkdtemp(path);
        if (created == nullptr) {
            throw std::runtime_error("failed to create temporary directory");
        }
        directory_ = created;
        socket_path_ = directory_ + "/worker.sock";
    }

    ~TemporarySocketDirectory() {
        ::unlink(socket_path_.c_str());
        ::rmdir(directory_.c_str());
    }

    TemporarySocketDirectory(TemporarySocketDirectory const&) = delete;
    TemporarySocketDirectory& operator=(
        TemporarySocketDirectory const&) = delete;

    [[nodiscard]] std::string const& directory() const noexcept {
        return directory_;
    }

    [[nodiscard]] std::string const& socketPath() const noexcept {
        return socket_path_;
    }

private:
    std::string directory_;
    std::string socket_path_;
};

[[nodiscard]] ModelManifest makeManifest() {
    return ModelManifest{
        "tinyllama",
        "revision-1",
        "tokenizer-sha",
        "template-sha",
        "engine-sha",
        2,
        0,
        128,
        64,
        192,
        "fp16",
        8};
}

[[nodiscard]] WorkerLimits makeLimits() {
    return WorkerLimits{
        8,
        1024,
        512,
        kDefaultMaxFramePayloadBytes,
        1024,
        4U * 1024U * 1024U,
        128,
        2U * 1024U * 1024U};
}

[[nodiscard]] Hello makeHello() {
    return Hello{kProtocolVersion, "tinyllama", "revision-1"};
}

[[nodiscard]] HelloAck makeHelloAck() {
    return HelloAck{
        kProtocolVersion,
        77,
        makeManifest(),
        makeLimits()};
}

[[nodiscard]] IpcSessionConfig makeSessionConfig() {
    IpcSessionConfig config;
    config.max_egress_frames = 1024;
    config.max_egress_bytes = 4U * 1024U * 1024U;
    config.read_buffer_bytes = 4096;
    return config;
}

struct ConnectionPair {
    UdsConnection client;
    UdsConnection server;
};

[[nodiscard]] ConnectionPair connectPair(UdsListener& listener) {
    UdsAcceptResult accepted;
    std::thread accept_thread([&] {
        accepted = listener.accept();
    });
    auto connected = UdsConnection::connect(listener.socketPath());
    accept_thread.join();
    if (!connected.ok()) {
        throw std::runtime_error(connected.status.message);
    }
    if (!accepted.ok()) {
        throw std::runtime_error(accepted.status.message);
    }
    return {
        std::move(connected.connection),
        std::move(accepted.connection)};
}

void writeRaw(
    int file_descriptor,
    std::vector<std::uint8_t> const& bytes,
    std::size_t offset,
    std::size_t size) {
    std::size_t written{0};
    while (written < size) {
        ssize_t const result = ::send(
            file_descriptor,
            bytes.data() + offset + written,
            size - written,
            MSG_NOSIGNAL);
        if (result > 0) {
            written += static_cast<std::size_t>(result);
            continue;
        }
        if (result < 0 && errno == EINTR) {
            continue;
        }
        throw std::runtime_error("failed to write raw UDS bytes");
    }
}

[[nodiscard]] Message readRawMessage(UdsConnection& connection) {
    FrameDecoder decoder;
    std::vector<char> buffer(4096);
    while (true) {
        auto read_result =
            connection.readSome(buffer.data(), buffer.size());
        if (!read_result.ok() || read_result.peer_closed) {
            throw std::runtime_error("failed to read raw IPC message");
        }
        auto frames = decoder.feed(std::string_view(
            buffer.data(), read_result.bytes_read));
        if (!frames.ok()) {
            throw std::runtime_error(frames.status.message);
        }
        if (!frames.payloads.empty()) {
            auto message = decodePayload(frames.payloads.front());
            if (!message.ok()) {
                throw std::runtime_error(message.status.message);
            }
            return std::move(*message.message);
        }
    }
}

[[nodiscard]] std::size_t countOpenFileDescriptors() {
    DIR* const directory = ::opendir("/proc/self/fd");
    if (directory == nullptr) {
        throw std::runtime_error("failed to inspect /proc/self/fd");
    }
    std::size_t count{0};
    while (::readdir(directory) != nullptr) {
        ++count;
    }
    ::closedir(directory);
    return count;
}

void testListenerOwnershipAndPathSafety(int& failures) {
    TemporarySocketDirectory temporary;
    {
        auto result = UdsListener::listen(temporary.socketPath());
        expect(result.ok(), "listener must bind a real UDS path", failures);

        struct stat socket_status {};
        bool const inspected =
            ::lstat(temporary.socketPath().c_str(), &socket_status) == 0;
        expect(inspected, "bound UDS path must exist", failures);
        if (inspected) {
            expect(
                S_ISSOCK(socket_status.st_mode),
                "bound UDS path must be a socket",
                failures);
            expect(
                (socket_status.st_mode & 0777) == 0600,
                "UDS path permissions must be owner-only",
                failures);
        }
    }
    expect(
        ::access(temporary.socketPath().c_str(), F_OK) != 0,
        "listener destruction must remove its socket path",
        failures);

    std::string const regular_path = temporary.directory() + "/regular";
    {
        std::ofstream regular_file(regular_path);
        regular_file << "keep";
    }
    auto refused = UdsListener::listen(regular_path);
    expect(
        !refused.ok() && refused.status.code == StatusCode::AlreadyExists,
        "listener must refuse an existing non-socket path",
        failures);
    expect(
        ::access(regular_path.c_str(), F_OK) == 0,
        "listener must not delete an existing regular file",
        failures);
    ::unlink(regular_path.c_str());
}

void testHandshakeAndOrderedMessages(int& failures) {
    TemporarySocketDirectory temporary;
    auto listening = UdsListener::listen(temporary.socketPath());
    expect(listening.ok(), "session test listener must start", failures);
    if (!listening.ok()) {
        return;
    }
    auto connections = connectPair(listening.listener);

    std::mutex received_mutex;
    std::condition_variable received_condition;
    std::vector<Message> server_messages;
    std::vector<Message> client_messages;

    IpcSession server(
        std::move(connections.server),
        makeSessionConfig(),
        makeHelloAck());
    IpcSession client(
        std::move(connections.client),
        makeSessionConfig(),
        makeHello());

    auto status = server.start([&](Message message) {
        {
            std::lock_guard<std::mutex> lock(received_mutex);
            server_messages.push_back(std::move(message));
        }
        received_condition.notify_all();
    });
    expect(status.ok(), "server session must start", failures);
    status = client.start([&](Message message) {
        {
            std::lock_guard<std::mutex> lock(received_mutex);
            client_messages.push_back(std::move(message));
        }
        received_condition.notify_all();
    });
    expect(status.ok(), "client session must start", failures);
    expect(
        server.waitUntilReady(1s).ok(),
        "server handshake must complete",
        failures);
    expect(
        client.waitUntilReady(1s).ok(),
        "client handshake must complete",
        failures);

    Submit submit;
    submit.worker_epoch = 77;
    submit.request_id = 9;
    submit.input_token_ids = {1, 2, 3};
    submit.max_new_tokens = 4;
    status = client.send(Message{submit});
    expect(status.ok(), "client must send Submit after handshake", failures);

    status = server.send(Message{Accepted{kProtocolVersion, 77, 9}});
    expect(status.ok(), "server must send Accepted", failures);
    status = server.send(Message{TokenDelta{
        kProtocolVersion, 77, 9, 0, {10, 11}}});
    expect(status.ok(), "server must send TokenDelta", failures);
    status = server.send(Message{Terminal{
        kProtocolVersion,
        77,
        9,
        Status::success(),
        FinishReason::Eos,
        Usage{3, 2}}});
    expect(status.ok(), "server must send Terminal", failures);

    {
        std::unique_lock<std::mutex> lock(received_mutex);
        bool const completed = received_condition.wait_for(
            lock,
            1s,
            [&] {
                return server_messages.size() == 1 &&
                    client_messages.size() == 3;
            });
        expect(completed, "all session messages must arrive", failures);
    }
    {
        std::lock_guard<std::mutex> lock(received_mutex);
        expect(
            server_messages.size() == 1 &&
                std::holds_alternative<Submit>(server_messages[0]),
            "server must receive Submit",
            failures);
        expect(
            client_messages.size() == 3 &&
                std::holds_alternative<Accepted>(client_messages[0]) &&
                std::holds_alternative<TokenDelta>(client_messages[1]) &&
                std::holds_alternative<Terminal>(client_messages[2]),
            "single writer must preserve Accepted/Delta/Terminal order",
            failures);
    }

    status = server.send(Message{submit});
    expect(
        !status.ok() && status.code == StatusCode::InvalidInput,
        "server must reject a client-direction message",
        failures);

    expect(client.stop().ok(), "client stop must succeed", failures);
    auto const disconnected = server.waitUntilClosed(1s);
    expect(
        !disconnected.ok() &&
            disconnected.code == StatusCode::Unavailable,
        "peer disconnect must close the server session",
        failures);
    static_cast<void>(server.stop());
}

void testFragmentedHandshakeAndPartialDisconnect(int& failures) {
    TemporarySocketDirectory temporary;
    auto listening = UdsListener::listen(temporary.socketPath());
    expect(listening.ok(), "fragment test listener must start", failures);
    if (!listening.ok()) {
        return;
    }

    {
        auto connections = connectPair(listening.listener);
        IpcSession server(
            std::move(connections.server),
            makeSessionConfig(),
            makeHelloAck());
        expect(server.start().ok(), "fragment server must start", failures);

        auto payload = encodePayload(Message{makeHello()});
        auto frame = encodeFrame(payload.payload);
        expect(frame.ok(), "Hello frame must encode", failures);
        if (frame.ok()) {
            writeRaw(
                connections.client.nativeHandle(),
                frame.bytes,
                0,
                2);
            std::this_thread::sleep_for(20ms);
            expect(
                server.state() == IpcSessionState::Handshaking,
                "partial prefix must not complete the handshake",
                failures);
            writeRaw(
                connections.client.nativeHandle(),
                frame.bytes,
                2,
                frame.bytes.size() - 2);
            expect(
                server.waitUntilReady(1s).ok(),
                "fragmented Hello must complete over real UDS",
                failures);
            auto ack = readRawMessage(connections.client);
            expect(
                std::holds_alternative<HelloAck>(ack),
                "server must reply with HelloAck",
                failures);
        }
        connections.client.shutdown();
        connections.client.close();
        static_cast<void>(server.waitUntilClosed(1s));
        static_cast<void>(server.stop());
    }

    {
        auto connections = connectPair(listening.listener);
        IpcSession server(
            std::move(connections.server),
            makeSessionConfig(),
            makeHelloAck());
        expect(
            server.start().ok(),
            "partial-disconnect server must start",
            failures);
        std::vector<std::uint8_t> partial_prefix{0, 0};
        writeRaw(
            connections.client.nativeHandle(),
            partial_prefix,
            0,
            partial_prefix.size());
        connections.client.shutdown();
        connections.client.close();
        auto const closed = server.waitUntilClosed(1s);
        expect(
            !closed.ok() && closed.code == StatusCode::InvalidInput,
            "disconnect with a partial frame must be a protocol error",
            failures);
        static_cast<void>(server.stop());
    }
}

void testHandshakeRejection(int& failures) {
    TemporarySocketDirectory temporary;
    auto listening = UdsListener::listen(temporary.socketPath());
    expect(listening.ok(), "rejection test listener must start", failures);
    if (!listening.ok()) {
        return;
    }

    auto connections = connectPair(listening.listener);
    IpcSession server(
        std::move(connections.server),
        makeSessionConfig(),
        makeHelloAck());
    expect(server.start().ok(), "rejection server must start", failures);

    std::string const invalid_hello =
        R"({"type":"hello","protocol_version":2,"model_id":"tinyllama","revision":"revision-1"})";
    auto frame = encodeFrame(invalid_hello);
    expect(frame.ok(), "invalid-version frame must still frame", failures);
    if (frame.ok()) {
        writeRaw(
            connections.client.nativeHandle(),
            frame.bytes,
            0,
            frame.bytes.size());
    }
    auto const rejected = server.waitUntilClosed(1s);
    expect(
        !rejected.ok() && rejected.code == StatusCode::InvalidInput,
        "unsupported protocol version must fail the handshake",
        failures);
    static_cast<void>(server.stop());

    auto mismatched_connections = connectPair(listening.listener);
    IpcSession mismatched_server(
        std::move(mismatched_connections.server),
        makeSessionConfig(),
        makeHelloAck());
    Hello mismatched_hello = makeHello();
    mismatched_hello.revision = "wrong-revision";
    IpcSession mismatched_client(
        std::move(mismatched_connections.client),
        makeSessionConfig(),
        std::move(mismatched_hello));
    expect(
        mismatched_server.start().ok() && mismatched_client.start().ok(),
        "manifest-mismatch sessions must start",
        failures);
    auto const mismatch = mismatched_server.waitUntilClosed(1s);
    expect(
        !mismatch.ok() && mismatch.code == StatusCode::InvalidInput,
        "model revision mismatch must fail the handshake",
        failures);
    static_cast<void>(mismatched_client.waitUntilClosed(1s));
    static_cast<void>(mismatched_client.stop());
    static_cast<void>(mismatched_server.stop());
}

void testBoundariesAndResourceCleanup(int& failures) {
    std::size_t const descriptors_before = countOpenFileDescriptors();
    {
        TemporarySocketDirectory temporary;
        auto listening = UdsListener::listen(temporary.socketPath());
        expect(listening.ok(), "cleanup test listener must start", failures);
        if (!listening.ok()) {
            return;
        }

        for (int iteration = 0; iteration < 20; ++iteration) {
            auto connections = connectPair(listening.listener);
            IpcSession server(
                std::move(connections.server),
                makeSessionConfig(),
                makeHelloAck());
            IpcSession client(
                std::move(connections.client),
                makeSessionConfig(),
                makeHello());
            expect(server.start().ok(), "cleanup server must start", failures);
            expect(client.start().ok(), "cleanup client must start", failures);
            expect(
                server.waitUntilReady(1s).ok() &&
                    client.waitUntilReady(1s).ok(),
                "cleanup handshake must complete",
                failures);
            expect(client.stop().ok(), "cleanup client must stop", failures);
            static_cast<void>(server.waitUntilClosed(1s));
            static_cast<void>(server.stop());
        }
    }
    std::size_t const descriptors_after = countOpenFileDescriptors();
    expect(
        descriptors_after == descriptors_before,
        "repeated session shutdown must not leak file descriptors",
        failures);

    TemporarySocketDirectory temporary;
    auto listening = UdsListener::listen(temporary.socketPath());
    if (!listening.ok()) {
        expect(false, "boundary test listener must start", failures);
        return;
    }
    auto connections = connectPair(listening.listener);
    auto invalid_config = makeSessionConfig();
    invalid_config.max_egress_bytes = 8;
    IpcSession invalid_session(
        std::move(connections.client),
        invalid_config,
        makeHello());
    auto const invalid_start = invalid_session.start();
    expect(
        !invalid_start.ok() &&
            invalid_start.code == StatusCode::InvalidInput,
        "session must reject an egress limit smaller than one frame",
        failures);
    connections.server.close();

    auto bounded_connections = connectPair(listening.listener);
    int send_buffer_bytes = 4096;
    expect(
        ::setsockopt(
            bounded_connections.server.nativeHandle(),
            SOL_SOCKET,
            SO_SNDBUF,
            &send_buffer_bytes,
            sizeof(send_buffer_bytes)) == 0,
        "bounded test must reduce the server send buffer",
        failures);

    IpcSessionConfig bounded_config;
    bounded_config.frame_codec.max_payload_bytes = 128U * 1024U;
    bounded_config.max_egress_frames = 1;
    bounded_config.max_egress_bytes = 128U * 1024U + 4U;
    bounded_config.read_buffer_bytes = 4096;
    auto bounded_ack = makeHelloAck();
    bounded_ack.limits.max_frame_payload_bytes =
        bounded_config.frame_codec.max_payload_bytes;
    bounded_ack.limits.max_session_egress_frames =
        bounded_config.max_egress_frames;
    bounded_ack.limits.max_session_egress_bytes =
        bounded_config.max_egress_bytes;
    bounded_ack.limits.max_request_egress_frames = 1;
    bounded_ack.limits.max_request_egress_bytes =
        bounded_config.max_egress_bytes;

    IpcSession bounded_server(
        std::move(bounded_connections.server),
        bounded_config,
        std::move(bounded_ack));
    expect(
        bounded_server.start().ok(),
        "bounded-egress server must start",
        failures);
    auto hello_payload = encodePayload(Message{makeHello()});
    auto hello_frame = encodeFrame(
        hello_payload.payload,
        bounded_config.frame_codec);
    writeRaw(
        bounded_connections.client.nativeHandle(),
        hello_frame.bytes,
        0,
        hello_frame.bytes.size());
    static_cast<void>(readRawMessage(bounded_connections.client));
    expect(
        bounded_server.waitUntilReady(1s).ok(),
        "bounded-egress handshake must complete",
        failures);

    bool queue_full{false};
    std::string large_error(96U * 1024U, 'x');
    for (std::uint64_t request_id = 1;
         request_id <= 100 && !queue_full;
         ++request_id) {
        auto const send_status = bounded_server.send(Message{Rejected{
            kProtocolVersion,
            77,
            request_id,
            Status::error(
                StatusCode::ResourceExhausted,
                large_error)}});
        if (send_status.code == StatusCode::QueueFull) {
            queue_full = true;
        } else {
            expect(
                send_status.ok(),
                "bounded egress send must either enqueue or report QueueFull",
                failures);
        }
    }
    expect(
        queue_full,
        "a non-reading peer must saturate the bounded egress queue",
        failures);

    auto const stop_started = std::chrono::steady_clock::now();
    static_cast<void>(bounded_server.stop());
    auto const stop_elapsed =
        std::chrono::steady_clock::now() - stop_started;
    expect(
        stop_elapsed < 1s,
        "shutdown must wake a writer blocked by a non-reading peer",
        failures);
    bounded_connections.client.close();
}

} // namespace

int main() {
    int failures{0};
    try {
        testListenerOwnershipAndPathSafety(failures);
        testHandshakeAndOrderedMessages(failures);
        testFragmentedHandshakeAndPartialDisconnect(failures);
        testHandshakeRejection(failures);
        testBoundariesAndResourceCleanup(failures);
    } catch (std::exception const& exception) {
        ++failures;
        std::cerr << "FAILED: unexpected exception: "
                  << exception.what() << '\n';
    }

    if (failures == 0) {
        std::cout << "llm_ipc_transport_contract: PASS\n";
        return 0;
    }
    std::cerr << "llm_ipc_transport_contract: "
              << failures << " failure(s)\n";
    return 1;
}

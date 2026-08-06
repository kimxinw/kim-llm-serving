#pragma once

  #include "core/status.h"

  #include <array>
  #include <cstddef>
  #include <cstdint>
  #include <optional>
  #include <string>
  #include <string_view>
  #include <variant>
  #include <vector>

  namespace kimrt::llm::ipc {

  inline constexpr std::uint32_t kProtocolVersion{1};
  inline constexpr std::uint32_t kDefaultMaxFramePayloadBytes{
      1024U * 1024U
  };

  struct FrameCodecConfig {
      std::uint32_t max_payload_bytes{kDefaultMaxFramePayloadBytes};
  };

  struct FrameEncodeResult {
      Status status;
      std::vector<std::uint8_t> bytes;

      [[nodiscard]] bool ok() const noexcept;
      explicit operator bool() const noexcept;
  };

  [[nodiscard]] FrameEncodeResult encodeFrame(
      std::string_view payload,
      FrameCodecConfig config = {});

  enum class FrameDecodeState : std::uint8_t {
      NeedMore,
      FramesReady,
      Error,
  };

  struct FrameDecodeResult {
      FrameDecodeState state{FrameDecodeState::NeedMore};
      Status status;
      std::vector<std::string> payloads;

      [[nodiscard]] bool ok() const noexcept;
      explicit operator bool() const noexcept;
  };

  class FrameDecoder final {
  public:
      explicit FrameDecoder(FrameCodecConfig config = {});

      [[nodiscard]] FrameDecodeResult feed(std::string_view bytes);

      void reset() noexcept;

      [[nodiscard]] bool failed() const noexcept;
      [[nodiscard]] bool hasPartialFrame() const noexcept;

  private:
      [[nodiscard]] FrameDecodeResult fail(
          StatusCode code,
          std::string message);

      FrameCodecConfig config_;

      std::array<std::uint8_t, 4> prefix_{};
      std::size_t prefix_size_{0};

      std::uint32_t expected_payload_size_{0};
      std::string payload_;

      bool failed_{false};
      Status failure_;
  };

  struct Hello {
      std::uint32_t protocol_version{kProtocolVersion};
  };

  struct HelloAck {
      std::uint32_t protocol_version{kProtocolVersion};
      std::uint64_t worker_epoch{0};
  };

  using HandshakeMessage = std::variant<Hello, HelloAck>;

  struct PayloadEncodeResult {
      Status status;
      std::string payload;

      [[nodiscard]] bool ok() const noexcept;
      explicit operator bool() const noexcept;
  };

  struct HandshakeDecodeResult {
      Status status;
      std::optional<HandshakeMessage> message;

      [[nodiscard]] bool ok() const noexcept;
      explicit operator bool() const noexcept;
  };

  [[nodiscard]] PayloadEncodeResult encodeHelloPayload(
      Hello const& hello);

  [[nodiscard]] PayloadEncodeResult encodeHelloAckPayload(
      HelloAck const& hello_ack);

  [[nodiscard]] HandshakeDecodeResult decodeHandshakePayload(
      std::string_view payload);

  } // namespace kimrt::llm::ipc
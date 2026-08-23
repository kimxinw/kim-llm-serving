#include "backends/trtllm/trtllm_executor_backend.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <variant>
#include <vector>
#include <sstream>

namespace {
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  using kimrt::llm::GenerationEvent;
  using kimrt::llm::GenerationMailbox;
  using kimrt::llm::GenerationMailboxConfig;
  using kimrt::llm::GenerationRequest;
  using kimrt::llm::MailboxWaitResult;
  using kimrt::llm::TerminalEvent;
  using kimrt::llm::TokenDelta;
  using kimrt::llm::TokenId;
  using kimrt::llm::TrtLlmBackendConfig;
  using kimrt::llm::TrtLlmExecutorBackend;

  constexpr std::size_t kMaxInputTokens{512};
  constexpr std::size_t kMaxSequenceTokens{544};
  constexpr std::size_t kMaxOutputTokens{32};
  constexpr auto kRequestTimeout = std::chrono::seconds{180};

  constexpr auto kStatsSampleInterval = std::chrono::milliseconds{100};

  // measured 期间必须持续 drain。Executor 的 iteration stats 是容量有限的
  // 缓冲区，只在结束时取一次会静默丢掉最早的迭代，而留下的恰好是请求收尾、
  // in-flight 数量往 0 掉的尾段——正好会低估 active/scheduled 和 KV 占用。
  // c1 × 200 请求 × OSL32 约 6600 次迭代，远超默认的 1000。
  class IterationStatsSampler final {
  public:
    IterationStatsSampler(TrtLlmExecutorBackend &backend,
                          std::chrono::milliseconds interval)
        : backend_(backend), interval_(interval), worker_([this] { run(); }) {}

    ~IterationStatsSampler() { (void)stop(); }

    IterationStatsSampler(IterationStatsSampler const &) = delete;
    IterationStatsSampler &operator=(IterationStatsSampler const &) = delete;
    IterationStatsSampler(IterationStatsSampler &&) = delete;
    IterationStatsSampler &operator=(IterationStatsSampler &&) = delete;

    // 停止采样，补最后一次 drain，返回按时间有序的全量结果。
    std::vector<std::string> stop() {
      if (stopRequested_.exchange(true)) {
        return {};
      }

      {
        std::lock_guard lock(mutex_);
        stopCondition_.notify_all();
      }

      if (worker_.joinable()) {
        worker_.join();
      }

      // worker 已 join，此后 stats_ 无并发访问。
      append(backend_.drainIterationStatsJson());

      return std::move(stats_);
    }

  private:
    void run() {
      while (true) {
        {
          std::unique_lock lock(mutex_);
          stopCondition_.wait_for(lock, interval_, [this] {
            return stopRequested_.load(std::memory_order_acquire);
          });
        }

        append(backend_.drainIterationStatsJson());

        if (stopRequested_.load(std::memory_order_acquire)) {
          return;
        }
      }
    }

    // 只由 worker 线程调用，或在 join 之后由 stop() 调用。
    void append(std::vector<std::string> &&batch) {
      stats_.insert(stats_.end(),
                    std::make_move_iterator(batch.begin()),
                    std::make_move_iterator(batch.end()));
    }

    TrtLlmExecutorBackend &backend_;
    std::chrono::milliseconds interval_;

    std::mutex mutex_;
    std::condition_variable stopCondition_;
    std::atomic<bool> stopRequested_{false};

    std::vector<std::string> stats_;
    std::thread worker_;
  };

  struct Options {
    std::filesystem::path engineDir;
    std::size_t concurrency{0};
    std::size_t warmupRequests{0};
    std::size_t measuredRequests{0};
    std::size_t maxNewTokens{0};

    std::filesystem::path summaryPath;
    std::filesystem::path requestCsvPath;
    std::filesystem::path iterationStatsPath;

    std::vector<TokenId> inputTokens;
  };

  struct RequestResult {
    std::uint64_t requestId{0};
    bool success{false};

    std::size_t promptTokens{0};
    std::size_t completionTokens{0};

    double ttftMs{-1.0};
    double tpotMs{-1.0};
    double e2eMs{-1.0};

    std::string error;
  };

  struct RunResult {
    std::vector<RequestResult> requests;
    double durationSeconds{0.0};
  };

  class BackendStopGuard final {
  public:
    explicit BackendStopGuard(TrtLlmExecutorBackend& backend) noexcept 
      :backend_(backend){}
    
    ~BackendStopGuard()noexcept {
      try {
        backend_.stop();
      }catch(...){
        //析构函数不能向外抛出异常
      }
    }

    BackendStopGuard(const BackendStopGuard&) = delete;
    BackendStopGuard& operator=(const BackendStopGuard&) = delete;

  private:
    TrtLlmExecutorBackend &backend_;
  };

  [[nodiscard]] bool allRequestsSuccessful(RunResult const &  run){
    return std::all_of(
      run.requests.begin(),
      run.requests.end(),
      [](RequestResult const &request){
        return request.success;
      }
    );
  }

  [[nodiscard]] std::string firstFailureMessage(
    std::string_view phase,
    RunResult const & run
  ){
    auto const failure = std::find_if(
      run.requests.begin(),
      run.requests.end(),
      [](RequestResult const & request){
        return !request.success;
      }
    );

    if(failure == run.requests.end()){
      return {};
    }

    std::ostringstream oss;
    oss<<phase
       <<" request "
       <<failure->requestId
       <<" failed";

    if(!failure->error.empty()){
    oss<<": "<<failure->error;
    }

    return oss.str();
  }

  bool parseSize(std::string_view text, std::size_t &value) {
    std::uint64_t parsed{0};

    auto const [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), parsed);

    if (text.empty() || error != std::errc{} ||
        end != text.data() + text.size() ||
        parsed > std::numeric_limits<std::size_t>::max()) {
      return false;
    }

    value = static_cast<std::size_t>(parsed);
    return true;
  }

  bool parseToken(std::string_view text, TokenId &value) {
    auto const [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);

    return !text.empty() && error == std::errc{} &&
           end == text.data() + text.size() && value >= 0;
  }

  std::optional<Options> parseOptions(int argc, char **argv) {
    if (argc < 10) {
      std::cerr
          << "Usage: " << argv[0]
          << " <engine_dir> <concurrency> <warmup_requests>"
          << " <measured_requests> <max_new_tokens>"
          << " <summary.json> <requests.csv> <iteration_stats.jsonl>"
          << " <token_id>...\n";
      return std::nullopt;
    }

    Options options;
    options.engineDir = argv[1];
    options.summaryPath = argv[6];
    options.requestCsvPath = argv[7];
    options.iterationStatsPath = argv[8];

    if (!parseSize(argv[2], options.concurrency) ||
        !parseSize(argv[3], options.warmupRequests) ||
        !parseSize(argv[4], options.measuredRequests) ||
        !parseSize(argv[5], options.maxNewTokens)) {
      std::cerr << "invalid numeric argument\n";
      return std::nullopt;
    }

    if (options.concurrency == 0 || options.concurrency > 8) {
      std::cerr << "concurrency must be in [1, 8]\n";
      return std::nullopt;
    }

    if (options.measuredRequests == 0) {
      std::cerr << "measured_requests must be greater than zero\n";
      return std::nullopt;
    }

    if (options.maxNewTokens == 0 ||
        options.maxNewTokens > kMaxOutputTokens) {
      std::cerr << "max_new_tokens must be in [1, 32]\n";
      return std::nullopt;
    }

    for (int index = 9; index < argc; ++index) {
      TokenId token{0};

      if (!parseToken(argv[index], token)) {
        std::cerr << "invalid token id: " << argv[index] << '\n';
        return std::nullopt;
      }

      options.inputTokens.push_back(token);
    }

    if (options.inputTokens.empty() ||
        options.inputTokens.size() > kMaxInputTokens ||
        options.inputTokens.size() + options.maxNewTokens >
            kMaxSequenceTokens) {
      std::cerr << "input token count exceeds Engine limits\n";
      return std::nullopt;
    }

    return options;
  }

  void ensureParentDirectory(std::filesystem::path const &path) {
    auto const parent = path.parent_path();

    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }
  }

  RequestResult runOneRequest(
      TrtLlmExecutorBackend &backend,
      std::uint64_t requestId,
      std::vector<TokenId> const &inputTokens,
      std::size_t maxNewTokens) {
    RequestResult result;
    result.requestId = requestId;

    auto mailbox = std::make_shared<GenerationMailbox>(
        GenerationMailboxConfig{
            maxNewTokens + 1,
            maxNewTokens + 1,
        });

    GenerationRequest request;
    request.context.request_id = requestId;
    request.input_token_ids = inputTokens;
    request.max_new_tokens = maxNewTokens;
    request.streaming = true;
    request.sampling.top_k = 1;
    request.sampling.top_p = 1.0F;
    request.sampling.temperature = 1.0F;
    request.sampling.random_seed = 0;

    auto const begin = Clock::now();
    auto const deadline = begin + kRequestTimeout;
    request.context.deadline = deadline;

    auto const submitStatus =
        backend.submit(std::move(request), mailbox);

    if (!submitStatus) {
      result.error = "submit failed: " + submitStatus.message;
      result.e2eMs =
          std::chrono::duration<double, std::milli>(
              Clock::now() - begin)
              .count();
      return result;
    }

    std::optional<TimePoint> firstTokenTime;
    std::optional<TimePoint> lastTokenTime;

    std::uint64_t expectedSequenceNo{0};
    std::size_t observedTokens{0};

    while (true) {
      auto const now = Clock::now();

      if (now >= deadline) {
        backend.cancel(requestId);
        result.error = "request timed out";
        break;
      }

      GenerationEvent event;
      auto const waitResult = mailbox->waitPop(
          event,
          std::chrono::duration_cast<std::chrono::milliseconds>(
              deadline - now));

      if (waitResult == MailboxWaitResult::Timeout) {
        backend.cancel(requestId);
        result.error = "mailbox wait timed out";
        break;
      }

      if (waitResult == MailboxWaitResult::Closed) {
        backend.cancel(requestId);
        result.error = "mailbox closed before Terminal";
        break;
      }

      if (auto *delta = std::get_if<TokenDelta>(&event)) {
        auto const receiveTime = Clock::now();

        if (delta->request_id != requestId) {
          backend.cancel(requestId);
          result.error = "unexpected request id in TokenDelta";
          break;
        }

        if (delta->sequence_no != expectedSequenceNo) {
          backend.cancel(requestId);
          result.error = "non-contiguous TokenDelta sequence";
          break;
        }

        ++expectedSequenceNo;

        if (!firstTokenTime) {
          firstTokenTime = receiveTime;
        }

        lastTokenTime = receiveTime;
        observedTokens += delta->token_ids.size();
        continue;
      }

      auto *terminal = std::get_if<TerminalEvent>(&event);

      if (!terminal) {
        backend.cancel(requestId);
        result.error = "unknown generation event";
        break;
      }

      if (terminal->request_id != requestId) {
        backend.cancel(requestId);
        result.error = "unexpected request id in Terminal";
        break;
      }

      result.promptTokens = terminal->usage.prompt_tokens;
      result.completionTokens = terminal->usage.completion_tokens;

      if (!terminal->status) {
        result.error = terminal->status.message;
        break;
      }

      if (observedTokens != terminal->usage.completion_tokens) {
        result.error = "TokenDelta count differs from Terminal usage";
        break;
      }

      result.success = true;
      break;
    }

    auto const end = Clock::now();

    result.e2eMs =
        std::chrono::duration<double, std::milli>(end - begin).count();

    if (firstTokenTime) {
      result.ttftMs =
          std::chrono::duration<double, std::milli>(
              *firstTokenTime - begin)
              .count();
    }

    if (firstTokenTime && lastTokenTime && observedTokens > 1) {
      result.tpotMs =
          std::chrono::duration<double, std::milli>(
              *lastTokenTime - *firstTokenTime)
              .count() /
          static_cast<double>(observedTokens - 1);
    }

    return result;
  }

  RunResult runClosedLoop(
      TrtLlmExecutorBackend &backend,
      std::size_t concurrency,
      std::size_t totalRequests,
      std::uint64_t firstRequestId,
      std::vector<TokenId> const &inputTokens,
      std::size_t maxNewTokens) {
    RunResult result;
    result.requests.resize(totalRequests);

    if (totalRequests == 0) {
      return result;
    }

    auto const workerCount = std::min(concurrency, totalRequests);

    std::atomic<std::size_t> nextIndex{0};
    std::atomic<bool> startGate{false};

    std::vector<std::thread> workers;
    workers.reserve(workerCount);

    for (std::size_t worker = 0; worker < workerCount; ++worker) {
      workers.emplace_back([&] {
        while (!startGate.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }

        while (true) {
          auto const index =
              nextIndex.fetch_add(1, std::memory_order_relaxed);

          if (index >= totalRequests) {
            break;
          }

          result.requests[index] = runOneRequest(
              backend,
              firstRequestId + index,
              inputTokens,
              maxNewTokens);
        }
      });
    }

    auto const begin = Clock::now();
    startGate.store(true, std::memory_order_release);

    for (auto &worker : workers) {
      worker.join();
    }

    auto const end = Clock::now();

    result.durationSeconds =
        std::chrono::duration<double>(end - begin).count();

    return result;
  }

  double percentile(std::vector<double> values, double quantile) {
    if (values.empty()) {
      return -1.0;
    }

    std::sort(values.begin(), values.end());

    auto const rank = static_cast<std::size_t>(
        std::ceil(quantile * static_cast<double>(values.size())));

    auto const index =
        std::min(values.size() - 1, std::max<std::size_t>(1, rank) - 1);

    return values[index];
  }

  std::string distributionJson(std::vector<double> const &values) {
    if (values.empty()) {
      return "null";
    }

    std::ostringstream output;
    output << std::fixed << std::setprecision(3)
           << "{\"p50\":" << percentile(values, 0.50)
           << ",\"p95\":" << percentile(values, 0.95)
           << ",\"p99\":" << percentile(values, 0.99)
           << '}';

    return output.str();
  }

  std::string csvEscape(std::string const &value) {
    std::string escaped{"\""};

    for (char character : value) {
      if (character == '"') {
        escaped += "\"\"";
      } else {
        escaped += character;
      }
    }

    escaped += '"';
    return escaped;
  }

  void writeRequestCsv(
      std::filesystem::path const &path,
      std::vector<RequestResult> const &requests) {
    ensureParentDirectory(path);

    std::ofstream output{path};

    if (!output) {
      throw std::runtime_error("failed to open request CSV");
    }

    output
        << "request_id,success,prompt_tokens,completion_tokens,"
        << "ttft_ms,tpot_ms,e2e_ms,error\n";

    output << std::fixed << std::setprecision(3);

    for (auto const &request : requests) {
      output << request.requestId << ','
             << (request.success ? 1 : 0) << ','
             << request.promptTokens << ','
             << request.completionTokens << ',';

      if (request.ttftMs >= 0.0) {
        output << request.ttftMs;
      }

      output << ',';

      if (request.tpotMs >= 0.0) {
        output << request.tpotMs;
      }

      output << ','
             << request.e2eMs << ','
             << csvEscape(request.error) << '\n';
    }
  }

  void writeIterationStats(
      std::filesystem::path const &path,
      std::vector<std::string> const &stats) {
    ensureParentDirectory(path);

    std::ofstream output{path};

    if (!output) {
      throw std::runtime_error("failed to open iteration stats file");
    }

    for (auto const &stat : stats) {
      output << stat << '\n';
    }
  }

  void writeSummary(
      Options const &options,
      RunResult const &run,
      double loadMs) {
    std::vector<double> ttftValues;
    std::vector<double> tpotValues;
    std::vector<double> e2eValues;

    std::size_t successfulRequests{0};
    std::size_t completionTokens{0};

    for (auto const &request : run.requests) {
      if (!request.success) {
        continue;
      }

      ++successfulRequests;
      completionTokens += request.completionTokens;
      e2eValues.push_back(request.e2eMs);

      if (request.ttftMs >= 0.0) {
        ttftValues.push_back(request.ttftMs);
      }

      if (request.tpotMs >= 0.0) {
        tpotValues.push_back(request.tpotMs);
      }
    }

    auto const requestThroughput =
        run.durationSeconds > 0.0
            ? static_cast<double>(successfulRequests) / run.durationSeconds
            : 0.0;

    auto const outputTokenThroughput =
        run.durationSeconds > 0.0
            ? static_cast<double>(completionTokens) / run.durationSeconds
            : 0.0;

    ensureParentDirectory(options.summaryPath);
    std::ofstream output{options.summaryPath};

    if (!output) {
      throw std::runtime_error("failed to open summary JSON");
    }

    output << std::fixed << std::setprecision(6);
    output << "{\n";
    output << "  \"schema_version\": 1,\n";
    output << "  \"engine_dir\": "
           << csvEscape(options.engineDir.string()) << ",\n";
    output << "  \"concurrency\": " << options.concurrency << ",\n";
    output << "  \"warmup_requests\": " << options.warmupRequests << ",\n";
    output << "  \"measured_requests\": "
           << options.measuredRequests << ",\n";
    output << "  \"successful_requests\": "
           << successfulRequests << ",\n";
    output << "  \"failed_requests\": "
           << options.measuredRequests - successfulRequests << ",\n";
    output << "  \"input_tokens\": " << options.inputTokens.size() << ",\n";
    output << "  \"max_new_tokens\": " << options.maxNewTokens << ",\n";
    output << "  \"load_ms\": " << loadMs << ",\n";
    output << "  \"duration_seconds\": "
           << run.durationSeconds << ",\n";
    output << "  \"request_throughput_rps\": "
           << requestThroughput << ",\n";
    output << "  \"output_token_throughput_tps\": "
           << outputTokenThroughput << ",\n";
    output << "  \"ttft_ms\": " << distributionJson(ttftValues) << ",\n";
    output << "  \"tpot_ms\": " << distributionJson(tpotValues) << ",\n";
    output << "  \"e2e_ms\": " << distributionJson(e2eValues) << '\n';
    output << "}\n";
  }

}//namespace

int main(int argc,char** argv){
    auto options = parseOptions(argc,argv);

    if(!options){
        return 2;
    }

    if(!std::filesystem::is_directory(options->engineDir)){
        std::cerr<<"invalid Engine directory\n";
        return 2;
    }

    try {
        TrtLlmBackendConfig config;
        config.engine_dir = options->engineDir;
        config.max_input_tokens = kMaxInputTokens;
        config.max_sequence_tokens = kMaxSequenceTokens;
        config.max_output_tokens = kMaxOutputTokens;

        TrtLlmExecutorBackend backend{std::move(config)};

        auto const loadBegin = Clock::now();
        auto const startStatus = backend.start();
        auto const loadEnd = Clock::now();

        auto const loadMs = std::chrono::duration<double,std::milli>(
            loadEnd - loadBegin
        ).count();


        if (!startStatus) {
          throw std::runtime_error(
              "backend start failed: " + startStatus.message);
        }
        //从 Backend 成功启动开始，后面的任意异常路径都会自动stop.
        BackendStopGuard backendStopGuard{backend};

        auto warmup = runClosedLoop(
            backend,
            options->concurrency,
            options->warmupRequests,
            1,
            options->inputTokens,
            options->maxNewTokens
        );

        if(!allRequestsSuccessful(warmup)){
          throw std::runtime_error(
            firstFailureMessage("warmup",warmup)
          );
        }

        // 清除 Engine 加载和 warmup 阶段产生的 iteration stats，
        // 确保正式结果只包含 measured 阶段。
        (void)backend.drainIterationStatsJson();

        IterationStatsSampler statsSampler{backend, kStatsSampleInterval};

        auto measured = runClosedLoop(
            backend,
            options->concurrency,
            options->measuredRequests,
            1 + options->warmupRequests,
            options->inputTokens,
            options->maxNewTokens
        );

        auto iterationStats = statsSampler.stop();

        // 文件写入不属于推理测量区间。
        backend.stop();

        writeRequestCsv(options->requestCsvPath, measured.requests);
        writeIterationStats(options->iterationStatsPath, iterationStats);
        writeSummary(*options, measured, loadMs);

        return allRequestsSuccessful(measured) ? 0 : 4;
    }catch(std::exception const & e){
        std::cerr<<"benchmark failed: "<<e.what()<<'\n';
        return 3; 
    }
}
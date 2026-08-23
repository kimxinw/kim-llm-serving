#include "runtime/admission_controller.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {
    using kimrt::llm::AdmissionCode;
    using kimrt::llm::AdmissionConfig;
    using kimrt::llm::AdmissionController;
    using kimrt::llm::AdmissionLease;
    using kimrt::llm::AdmissionRequest;

    bool expect(bool condition,std::string_view message,int& failures){
        if(condition){
            return true;
        }

        ++failures;
        std::cerr<<"[FAIL] "<<message<<'\n';
        return false;
    }

    bool configThrows(AdmissionConfig config){
        try {
            AdmissionController controller(config);
            (void)controller;
        }catch(std::invalid_argument const &){
            return true;
        }

        return false;
    }

    void testConfigValidation(int& failures){
         expect(
            configThrows({0, 10, 10}),
            "zero active-request capacity must be rejected",
            failures
        );

        expect(
            configThrows({1, 0, 10}),
            "zero input-token capacity must be rejected",
            failures
        );

        expect(
            configThrows({1, 10, 0}),
            "zero output-token capacity must be rejected",
            failures
        );
    }

    void testOpenCloseAndRequestValidation(int &failures) {
        AdmissionController controller({2, 16, 16});

        expect(
            !controller.accepting(),
            "new controller must not accept requests",
            failures);

        auto closed = controller.tryAcquire({1, 1, 1});
        expect(
            closed.code == AdmissionCode::NotAccepting,
            "valid request must be rejected while controller is closed",
            failures);

        expect(
            controller.tryAcquire({0, 1, 1}).code ==
                AdmissionCode::InvalidRequest,
            "zero request id must be rejected",
            failures);

        expect(
            controller.tryAcquire({1, 0, 1}).code ==
                AdmissionCode::InvalidRequest,
            "zero input tokens must be rejected",
            failures);

        expect(
            controller.tryAcquire({1, 1, 0}).code ==
                AdmissionCode::InvalidRequest,
            "zero reserved output tokens must be rejected",
            failures);

        expect(controller.open(), "empty controller must open", failures);
        expect(controller.accepting(), "open controller must accept", failures);

        auto admitted = controller.tryAcquire({1, 4, 5});
        expect(admitted.admitted(), "valid request must be admitted", failures);

        controller.close();

        expect(
            !controller.accepting(),
            "close must stop accepting requests",
            failures
        );

        expect(
            controller.tryAcquire({2, 1, 1}).code ==
                AdmissionCode::NotAccepting,
            "close must reject later requests",
            failures
        );

        auto closed_snapshot = controller.snapshot();
        expect(
            closed_snapshot.active_requests == 1 &&
                closed_snapshot.reserved_input_tokens == 4 &&
                closed_snapshot.reserved_output_tokens == 5,
            "close must not revoke an existing lease",
            failures
        );

        expect(
            !controller.open(),
            "controller must not reopen while a lease is active",
            failures
        );

        admitted.lease.release();

        expect(
            controller.open(),
            "controller must reopen after all leases are released",
            failures
        );
    }

    void testActiveRequestLimit(int &failures) {
        AdmissionController controller({2, 100, 100});
        expect(controller.open(), "active-limit controller must open", failures);

        auto first = controller.tryAcquire({1, 1, 1});
        auto second = controller.tryAcquire({2, 1, 1});
        auto rejected = controller.tryAcquire({3, 1, 1});

        expect(first.admitted(), "first request must be admitted", failures);
        expect(second.admitted(), "second request must be admitted", failures);
        expect(
            rejected.code == AdmissionCode::ActiveRequestLimit,
            "third request must hit active-request limit",
            failures);

        auto snapshot = controller.snapshot();
        expect(
            snapshot.active_requests == 2 &&
                snapshot.reserved_input_tokens == 2 &&
                snapshot.reserved_output_tokens == 2,
            "active-limit rejection must not change reservations",
            failures);
    }

    void testInputTokenLimit(int &failures) {
    AdmissionController controller({4, 10, 100});
    expect(controller.open(), "input-limit controller must open", failures);

    auto first = controller.tryAcquire({1, 6, 1});
    auto rejected = controller.tryAcquire({2, 5, 1});

    expect(first.admitted(), "first input reservation must succeed", failures);
    expect(
        rejected.code == AdmissionCode::InputTokenLimit,
        "input-token overcommit must be rejected",
        failures);

    auto after_rejection = controller.snapshot();
    expect(
        after_rejection.active_requests == 1 &&
            after_rejection.reserved_input_tokens == 6 &&
            after_rejection.reserved_output_tokens == 1,
        "input-limit rejection must not consume resources",
        failures);

    auto exact_boundary = controller.tryAcquire({3, 4, 1});
    expect(
        exact_boundary.admitted(),
        "input reservation at exact capacity must succeed",
        failures);

    auto full = controller.snapshot();
    expect(
        full.active_requests == 2 &&
            full.reserved_input_tokens == 10,
        "input reservations must reach exact configured capacity",
        failures);
  }

  void testOutputTokenLimit(int &failures) {
    AdmissionController controller({4, 100, 10});
    expect(controller.open(), "output-limit controller must open", failures);

    auto first = controller.tryAcquire({1, 1, 6});
    auto rejected = controller.tryAcquire({2, 1, 5});

    expect(first.admitted(), "first output reservation must succeed", failures);
    expect(
        rejected.code == AdmissionCode::OutputTokenLimit,
        "output-token overcommit must be rejected",
        failures);

    auto after_rejection = controller.snapshot();
    expect(
        after_rejection.active_requests == 1 &&
            after_rejection.reserved_input_tokens == 1 &&
            after_rejection.reserved_output_tokens == 6,
        "output-limit rejection must not consume resources",
        failures);

    auto exact_boundary = controller.tryAcquire({3, 1, 4});
    expect(
        exact_boundary.admitted(),
        "output reservation at exact capacity must succeed",
        failures);

    auto full = controller.snapshot();
    expect(
        full.active_requests == 2 &&
            full.reserved_output_tokens == 10,
        "output reservations must reach exact configured capacity",
        failures);
  }

  void testDuplicateIdAndRelease(int &failures) {
    AdmissionController controller({2, 16, 16});
    expect(controller.open(), "duplicate-id controller must open", failures);

    auto first = controller.tryAcquire({42, 3, 4});
    auto duplicate = controller.tryAcquire({42, 1, 1});

    expect(first.admitted(), "first request id must be admitted", failures);
    expect(
        duplicate.code == AdmissionCode::DuplicateRequestId,
        "duplicate active request id must be rejected",
        failures);

    auto before_release = controller.snapshot();
    expect(
        before_release.active_requests == 1 &&
            before_release.reserved_input_tokens == 3 &&
            before_release.reserved_output_tokens == 4,
        "duplicate rejection must not change reservations",
        failures);

    first.lease.release();
    first.lease.release();

    auto after_release = controller.snapshot();
    expect(
        after_release.active_requests == 0 &&
            after_release.reserved_input_tokens == 0 &&
            after_release.reserved_output_tokens == 0,
        "release must be idempotent and return all resources",
        failures);

    auto reused = controller.tryAcquire({42, 2, 2});
    expect(
        reused.admitted(),
        "request id must be reusable after lease release",
        failures);
  }

  void testLeaseMoveSemantics(int &failures) {
    AdmissionController controller({3, 32, 32});
    expect(controller.open(), "move-semantics controller must open", failures);

    auto first = controller.tryAcquire({1, 4, 5});
    auto second = controller.tryAcquire({2, 2, 3});

    expect(
        first.admitted() && second.admitted(),
        "leases used by move test must be admitted",
        failures);

    AdmissionLease moved(std::move(first.lease));

    expect(moved.valid(), "move-constructed lease must be valid", failures);
    expect(
        !first.lease.valid(),
        "move source must become invalid",
        failures);

    expect(
        moved.requestId() == 1 &&
            moved.inputTokens() == 4 &&
            moved.reservedOutputTokens() == 5,
        "move-constructed lease must preserve request metadata",
        failures);

    AdmissionLease assigned;
    assigned = std::move(moved);

    expect(
        assigned.valid() && !moved.valid(),
        "move assignment must transfer lease ownership",
        failures);

    /*
     * assigned 当前持有 request 1。
     * move-assign request 2 时，request 1 的资源必须先被归还。
     */
    assigned = std::move(second.lease);

    expect(
        assigned.valid() && assigned.requestId() == 2,
        "move assignment must take ownership of the new lease",
        failures);

    expect(
        !second.lease.valid(),
        "second move source must become invalid",
        failures);

    auto after_assignment = controller.snapshot();
    expect(
        after_assignment.active_requests == 1 &&
            after_assignment.reserved_input_tokens == 2 &&
            after_assignment.reserved_output_tokens == 3,
        "move assignment must release the previously held lease",
        failures);

    auto reused = controller.tryAcquire({1, 1, 1});
    expect(
        reused.admitted(),
        "request id released by move assignment must be reusable",
        failures);

    assigned.release();
    reused.lease.release();

    auto final_snapshot = controller.snapshot();
    expect(
        final_snapshot.active_requests == 0 &&
            final_snapshot.reserved_input_tokens == 0 &&
            final_snapshot.reserved_output_tokens == 0,
        "all moved leases must return resources exactly once",
        failures);
  }

  void testLeaseOutlivesController(int &failures) {
    AdmissionLease lease;

    {
      AdmissionController controller({1, 8, 8});
      expect(
          controller.open(),
          "controller used by lifetime test must open",
          failures);

      auto decision = controller.tryAcquire({77, 3, 4});
      expect(
          decision.admitted(),
          "lifetime-test request must be admitted",
          failures);

      lease = std::move(decision.lease);
    }

    expect(
        lease.valid(),
        "lease must remain valid after controller destruction",
        failures);

    expect(
        lease.requestId() == 77 &&
            lease.inputTokens() == 3 &&
            lease.reservedOutputTokens() == 4,
        "lease metadata must survive controller destruction",
        failures);

    lease.release();

    expect(
        !lease.valid(),
        "lease must safely release after controller destruction",
        failures);
  }

  void testConcurrentCapacity(int &failures) {
    constexpr std::size_t kCapacity = 8;
    constexpr std::size_t kContenders = 32;

    AdmissionController controller({
        kCapacity,
        kCapacity,
        kCapacity,
    });

    expect(controller.open(), "concurrency controller must open", failures);

    std::atomic<std::size_t> ready{0};
    std::atomic<std::size_t> attempted{0};
    std::atomic<std::size_t> admitted{0};
    std::atomic<bool> start{false};
    std::atomic<bool> release{false};

    std::vector<std::thread> threads;
    threads.reserve(kContenders);

    for (std::size_t i = 0; i < kContenders; ++i) {
      threads.emplace_back([&, i] {
        ready.fetch_add(1);

        while (!start.load()) {
          std::this_thread::yield();
        }

        auto decision = controller.tryAcquire(AdmissionRequest{
            static_cast<std::uint64_t>(i + 1),
            1,
            1,
        });

        if (decision.admitted()) {
          admitted.fetch_add(1);
        }

        attempted.fetch_add(1);

        if (decision.admitted()) {
          while (!release.load()) {
            std::this_thread::yield();
          }
        }
      });
    }

    while (ready.load() != kContenders) {
      std::this_thread::yield();
    }

    start.store(true);

    while (attempted.load() != kContenders) {
      std::this_thread::yield();
    }

    expect(
        admitted.load() == kCapacity,
        "concurrent acquisition must not exceed active capacity",
        failures);

    auto saturated = controller.snapshot();
    expect(
        saturated.active_requests == kCapacity &&
            saturated.reserved_input_tokens == kCapacity &&
            saturated.reserved_output_tokens == kCapacity,
        "concurrent reservations must exactly match configured capacity",
        failures);

    release.store(true);

    for (auto &thread : threads) {
      thread.join();
    }

    auto released = controller.snapshot();
    expect(
        released.active_requests == 0 &&
            released.reserved_input_tokens == 0 &&
            released.reserved_output_tokens == 0,
        "all concurrent leases must return resources",
        failures);
  }

}//namespace

int main() {
    int failures{0};

    testConfigValidation(failures);
    testOpenCloseAndRequestValidation(failures);
    testActiveRequestLimit(failures);
    testInputTokenLimit(failures);
    testOutputTokenLimit(failures);
    testDuplicateIdAndRelease(failures);
    testLeaseMoveSemantics(failures);
    testLeaseOutlivesController(failures);
    testConcurrentCapacity(failures);

    if (failures == 0) {
      std::cout << "[PASS] LLM AdmissionController contract\n";
    }

    return failures == 0 ? 0 : 1;
}
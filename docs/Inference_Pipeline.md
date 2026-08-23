先抓住一条主线：一次推理请求会依次经过下面这些模块。

  IPC Submit
  → IpcSession 解帧
  → RuntimeBridge 转换请求
  → GenerationRuntime 申请资源
  → TensorRT-LLM Backend 执行
  → RequestState 整理 Token/终态
  → GenerationMailbox 暂存事件
  → RuntimeBridge RequestPump 转发
  → SessionEgress 公平排队
  → IpcSession 写回 Gateway

  需要先说明：目前 A6-3 还没有真正的 Python Gateway 和独立 llm_worker。所以当前“请求入口”是已经连接好的
  Client IpcSession；A6-4 才会把它装进两个独立进程。

  模块化重构后，本文中的文件名按以下职责目录解析：公共类型位于 `include/common/`，生成生命周期位于
  `include/runtime/` 与 `src/runtime/`，协议和 UDS Session 位于 `include/ipc/` 与 `src/ipc/`，
  Runtime Bridge/Egress 位于 `include/worker/` 与 `src/worker/`，TensorRT-LLM 实现位于
  `include/backends/trtllm/` 与 `src/backends/trtllm/`。测试在 `tests/` 下使用相同分类。

  ———

  ## 一、请求涉及的模块

   顺序    模块                   文件                    一句话职责
  ━━━━━━  ━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      1    IPC Protocol           ipc_protocol.h/.cpp     定义 Submit、Accepted、Delta、Terminal 等网络消息
  ──────  ─────────────────────  ──────────────────────  ───────────────────────────────────────────────────
      2    IpcSession             ipc_transport.h/.cpp    收发字节、解帧、握手、校验消息方向
  ──────  ─────────────────────  ──────────────────────  ───────────────────────────────────────────────────
      3    RuntimeBridge          runtime_bridge.h/.cp    把 IPC 请求接到 GenerationRuntime
                                  p
  ──────  ─────────────────────  ──────────────────────  ───────────────────────────────────────────────────
      4    GenerationRuntime      generation_runtime.h    申请 Admission 资源并提交 Backend
                                  /.cpp
  ──────  ─────────────────────  ──────────────────────  ───────────────────────────────────────────────────
      5    AdmissionController    admission_controller    限制并发请求数和 Token 预算
                                  .h/.cpp
  ──────  ─────────────────────  ──────────────────────  ───────────────────────────────────────────────────
      6    TRT-LLM Backend        trtllm_executor_back    转成 Executor Request 并提交 GPU
                                  end.cpp
  ──────  ─────────────────────  ──────────────────────  ───────────────────────────────────────────────────
      7    RequestState           request_state.cpp       保证 Token 顺序和唯一 Terminal
  ──────  ─────────────────────  ──────────────────────  ───────────────────────────────────────────────────
      8    GenerationMailbox      generation_mailbox.c    Backend 与 Bridge 之间的有界事件队列
                                  pp
  ──────  ─────────────────────  ──────────────────────  ───────────────────────────────────────────────────
      9    SessionEgress          session_egress.cpp      请求间公平调度、Terminal 容量预留
  ──────  ─────────────────────  ──────────────────────  ───────────────────────────────────────────────────
     10    IpcSession writer      ipc_transport.cpp       编码成帧并写回 UDS

  下面以 request_id=100 的请求为例，完整走一遍。

  ———

  # 第一阶段：请求进入 Worker

  ## 1. Gateway 发送 Submit

  IPC 层的请求结构定义在 ipc_protocol.h:137-151：

  struct Submit {
      std::uint64_t worker_epoch;
      std::uint64_t request_id;
      std::int32_t priority;
      std::optional<std::uint64_t> timeout_ms;
      std::string trace_id;
      std::vector<std::int32_t> input_token_ids;
      std::uint32_t max_new_tokens;
      bool streaming;
      SamplingParameters sampling;
      ...
  };

  例如：

  request_id      = 100
  worker_epoch    = 77
  input_token_ids = [1, 2, 3]
  max_new_tokens  = 8
  streaming       = true
  timeout_ms      = 5000

  这里传的是 Token ID，不是文本。未来 Gateway 负责：

  用户文本
  → Tokenizer
  → input_token_ids
  → Submit

  A6-3 测试中的构造代码在 llm_runtime_bridge_test.cpp:204-221。

  ———

  ## 2. IpcSession 将 Submit 写入 UDS

  客户端调用：

  client_session.send(Message{submit});

  进入 ipc_transport.cpp:516-520：

  Status IpcSession::send(Message message) {
      if (!outboundMessageAllowed(message)) {
          return invalid(...);
      }
      return enqueueMessage(std::move(message), true);
  }

  接下来 enqueueMessage() 在 ipc_transport.cpp:686-729 中完成：

  Submit 对象
  → encodePayload()
  → JSON Payload
  → encodeFrame()
  → 4 字节长度前缀 + Payload
  → IpcSession 有界发送队列

  这里不会由调用线程直接写 Socket，而是唤醒 IpcSession 自己的 writer 线程。

  真正执行 Socket 写入的是 ipc_transport.cpp:902-945：

  frame = std::move(egress_.front());
  connection_.writeAll(frame.bytes);

  所以客户端业务线程不会被 Socket 写操作长时间阻塞。

  ———

  ## 3. Worker 侧 IpcSession 解帧

  Worker 的 IpcSession reader 运行在 ipc_transport.cpp:810-900。

  主要过程：

  readSome()
  → FrameDecoder::feed()
  → 恢复完整帧
  → decodePayload()
  → 得到 Message{Submit}
  → dispatchReadyMessage()

  关键代码：

  auto decoded_frames = decoder.feed(...);
  auto decoded_message = decodePayload(payload);
  status = dispatchReadyMessage(
      std::move(*decoded_message.message));

  dispatchReadyMessage() 位于 ipc_transport.cpp:776-808，会检查：

  - Session 已完成握手；
  - 当前角色允许接收 Submit；
  - 已经注册业务处理器。

  然后调用 RuntimeBridge 注册的回调：

  handler(std::move(message));

  RuntimeBridge 在启动时注册回调，见 runtime_bridge.cpp:181-183：

  status = session.start([this](ipc::Message message) {
      handleMessage(std::move(message));
  });

  至此，请求从“网络消息”进入了 Worker 业务层。

  ———

  # 第二阶段：RuntimeBridge 接收请求

  ## 4. RuntimeBridge 识别 Submit

  入口在 runtime_bridge.cpp:300-318：

  void handleMessage(ipc::Message message) {
      std::visit(
          [this](auto typed_message) {
              if constexpr (std::is_same_v<T, ipc::Submit>) {
                  handleSubmit(std::move(typed_message));
              } else if constexpr (...) {
                  ...
              }
          },
          std::move(message));
  }

  当前 Worker 侧只允许三类业务消息：

   消息      处理函数
  ━━━━━━━━  ━━━━━━━━━━━━━━━━
   Submit    handleSubmit()
  ────────  ────────────────
   Cancel    handleCancel()
  ────────  ────────────────
   Health    handleHealth()

  如果 Gateway 发来 TokenDelta 之类方向错误的消息，IpcSession 或 Bridge 会把 Session 判定为失败。

  ———

  ## 5. IPC Submit 转换成 GenerationRequest

  协议层和 Runtime 层使用不同的数据结构：

  ipc::Submit
        ↓ convertSubmit()
  GenerationRequest

  转换代码在 runtime_bridge.cpp:326-385。

  主要字段映射如下：

   IPC 字段           Runtime 字段
  ━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   request_id         request.context.request_id
  ─────────────────  ────────────────────────────
   priority           request.context.priority
  ─────────────────  ────────────────────────────
   trace_id           request.context.trace_id
  ─────────────────  ────────────────────────────
   timeout_ms         steady_clock deadline
  ─────────────────  ────────────────────────────
   input_token_ids    input_token_ids
  ─────────────────  ────────────────────────────
   max_new_tokens     max_new_tokens
  ─────────────────  ────────────────────────────
   sampling           sampling
  ─────────────────  ────────────────────────────
   end_id/pad_id      end_id/pad_id
  ─────────────────  ────────────────────────────
   stop_sequences     stop_sequences

  超时不是继续保存为“5000 毫秒”，而是转换为绝对 deadline：

  output.context.deadline =
      std::chrono::steady_clock::now() + timeout;

  这样后续 Backend 可以直接判断请求是否已经超时。

  这里还会校验 worker_epoch：

  if (input.worker_epoch != config.worker_epoch) {
      return InvalidInput;
  }

  它防止 Worker 重启后，旧连接上的请求污染新的 Worker 生命周期。

  转换失败时直接发送：

  Rejected(request_id=100)

  此时请求从未进入活动生命周期，所以后面不能再发送 Terminal。

  ———

  # 第三阶段：GenerationRuntime 申请资源

  ## 6. RuntimeBridge 调用 GenerationRuntime

  代码在 runtime_bridge.cpp:396：

  auto submission = runtime.submit(std::move(request));

  GenerationRuntime::submit() 位于 generation_runtime.cpp:174-271。

  内部有三个重要步骤。

  ### 6.1 基础校验

  generation_runtime.cpp:17-47 检查：

  - request ID 不能为 0；
    -输入 Token 不能为空；

  - max_new_tokens 不能为 0；
  - deadline 不能已经过期。

  ### 6.2 Admission 资源申请

  代码在 generation_runtime.cpp:197-207：

  auto decision = admission_.tryAcquire({
      request.context.request_id,
      request.input_token_ids.size(),
      request.max_new_tokens,
  });

  它尝试申请三种资源：

  active_requests        += 1
  reserved_input_tokens  += input_token数量
  reserved_output_tokens += max_new_tokens

  具体实现在 admission_controller.cpp:140-217。

  以当前请求为例：

  request_id             = 100
  input_tokens           = 3
  reserved_output_tokens = 8

  Admission 会依次检查：

   检查                       失败结果
  ━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━
   当前是否接受请求           NotAccepting
  ─────────────────────────  ────────────────────
   request ID 是否重复        DuplicateRequestId
  ─────────────────────────  ────────────────────
   活动请求是否满             ActiveRequestLimit
  ─────────────────────────  ────────────────────
   输入 Token 总预算是否满    InputTokenLimit
  ─────────────────────────  ────────────────────
   输出 Token 预留是否满      OutputTokenLimit

  所有检查和资源预留都在同一把锁内完成，所以多个请求并发进入时不会超额申请。

  如果资源不足：

  GenerationRuntime 返回拒绝
  → RuntimeBridge 发送 Rejected
  → 没有 Accepted
  → 没有 Terminal

  ———

  ## 7. 创建 GenerationMailbox

  Admission 成功后，Runtime 创建 Mailbox，见 generation_runtime.cpp:210-230：

  mailbox = std::make_shared<GenerationMailbox>(
      mailboxConfig_,
      std::move(decision.lease));

  这里最关键的关系是：

  Backend ──写事件──> GenerationMailbox ──读事件──> RuntimeBridge

  而且 AdmissionLease 被移动进 Mailbox：

  Admission 资源
      ↓
  AdmissionLease
      ↓
  GenerationMailbox 持有

  这意味着只要请求还没产生 Terminal，资源就不会被释放。

  ———

  # 第四阶段：提交 TensorRT-LLM

  ## 8. Runtime 调用 Backend

  代码在 generation_runtime.cpp:233-269：

  backendStatus = backend_->submit(
      std::move(request),
      mailbox);

  Backend 抽象接口非常简单，见 generation_backend.h:11-21：

  virtual Status submit(
      GenerationRequest request,
      std::shared_ptr<GenerationMailbox> mailbox) = 0;

  GenerationRuntime 不知道底层是：

  - TensorRT-LLM；
  - Fake Backend；
  - 以后其他推理引擎。

  它只要求 Backend 最终把结果写入 Mailbox。

  ———

  ## 9. 转成 TensorRT-LLM Request

  真实 Backend 的入口在 trtllm_executor_backend.cpp:156-225。

  首先再次进行 Engine 级校验：

  trtllm_executor_backend.cpp:399-440

  包括：

  - 输入长度是否超过 Engine；
  - max_new_tokens 是否超过上限；
  - 输入加输出是否超过最大序列长度；
  - sampling 参数是否合法；
  - stop sequence 是否合法；
  - deadline 是否已经过期。

  随后转换为 TensorRT-LLM 请求，见 trtllm_executor_backend.cpp:443-479：

  tle::Request result{
      input_token_ids,
      max_new_tokens,
      streaming,
      sampling,
      outputConfig
  };

  然后提交给 Executor：

  executorId = executor_->enqueueRequest(*executorRequest);

  对应 trtllm_executor_backend.cpp:182-185。

  这里同时存在两个 ID：

   ID             用途
  ━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   external ID    Gateway 看到的 request_id=100
  ─────────────  ────────────────────────────────
   executor ID    TensorRT-LLM 内部生成的请求 ID

  Backend 用两张表维护映射，见 trtllm_executor_backend.cpp:770-771：

  executor_id → RequestState
  external_id → executor_id

  之后收到 TensorRT-LLM Response 时，就能从 executor ID 找回 Gateway 的 request ID。

  ### 当前一个容易忽略的限制

  真实 Backend 在 trtllm_executor_backend.cpp:405-408 明确要求：

  if (request.context.priority != 0) {
      return InvalidInput;
  }

  也就是说协议虽然有 priority 字段，但当前真实 Engine 只支持 priority=0。

  A6-3 测试使用 priority=3，只是为了确认 RuntimeBridge 没有丢失字段；Fake Backend 不会拒绝它。真实请求目前必
  须使用默认优先级。

  ———

  # 第五阶段：为什么这时才能返回 Accepted

  ## 10. Runtime 提交成功

  只有以下步骤全部成功后，GenerationRuntime::submit() 才返回成功：

  请求校验成功
  → Admission 资源申请成功
  → Mailbox 创建成功
  → TensorRT-LLM enqueueRequest 成功
  → Backend RequestState 注册成功

  返回值包含：

  GenerationSubmission {
      Status::success(),
      mailbox
  }

  因此 RuntimeBridge 此时才知道：

  > 请求确实进入了 Backend，并且资源已经预留。

  ———

  ## 11. RuntimeBridge 注册 Egress 和请求归属

  回到 runtime_bridge.cpp:404-427：

  egress.registerRequest(request_id);
  requests.emplace(request_id, record);

  这里又注册了两种状态。

   状态                           为什么需要
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   SessionEgress request queue    给 Token/Terminal 分配有界输出空间
  ─────────────────────────────  ────────────────────────────────────
   RuntimeBridge requests map     记录这个 Session 拥有哪些活动请求

  请求归属记录中保存：

  request_id
  mailbox
  RequestPump thread
  cancel_requested
  backpressure_requested
  terminal_enqueued

  如果 Gateway 断连，Bridge 就通过这个表找到所有属于该 Session 的请求并取消。

  ———

  ## 12. 创建 RequestPump，但暂时不让它读取

  代码在 runtime_bridge.cpp:442-446：

  record->pump = std::thread(
      &Impl::requestPumpLoop,
      this,
      record);

  但 Pump 启动后会先等待门禁：

  record->waitUntilForwardingAllowed();

  也就是说此时：

  Backend 可能已经开始生成
  Mailbox 也可能已经有 Token
  但 RequestPump 暂时不能转发

  ———

  ## 13. Accepted 先入队，再放开 Pump

  代码在 runtime_bridge.cpp:467-476：

  egress.enqueueControl(Accepted{...});
  record->allowForwarding();

  所以严格顺序是：

  Accepted 进入发送队列
  → RequestPump 才能读取 Mailbox
  → TokenDelta/Terminal 才可能进入发送队列

  这就是 Accepted 顺序保证的核心代码。

  ———

  # 第六阶段：GPU 产生 Token

  ## 14. Backend ResponsePump 等待 TensorRT-LLM 结果

  Backend 启动时会创建一个 ResponsePump：

  trtllm_executor_backend.cpp:137-144

  responsePump_ = std::thread(&Impl::pumpResponses, this);

  它在 trtllm_executor_backend.cpp:482-516 中循环：

  auto responses =
      executor->awaitResponses(kPollInterval);

  for (auto const& response : responses) {
      handleResponse(response);
  }

  GPU 推理完成一部分后，TensorRT-LLM 返回 Response。

  ———

  ## 15. RequestState 将 Response 转成 TokenDelta

  Response 处理位于 trtllm_executor_backend.cpp:518-627。

  先通过 Executor ID 找到对应的 RequestState：

  auto request =
      findRequest(response.getRequestId());

  然后取出 Token：

  auto const& responseTokens =
      result.outputTokenIds.front();

  交给 RequestState：

  request->tryCommitDelta(
      std::move(deltaTokens));

  RequestState::tryCommitDelta() 位于 request_state.cpp:89-131，负责：

  - 检查是否已经 Terminal；
  - 检查是否正在 Cancel；
  - 生成连续的 sequence_no；
  - 更新 completion token 计数；
  - 把 Delta 放入 Mailbox。

  例如：

  TokenDelta {
      request_id = 100
      sequence_no = 0
      token_ids = [10, 11]
  }

  下一批就是：

  sequence_no = 1

  只有 Delta 真正进入 Mailbox 后，sequence 和 usage 才会增加。

  ———

  # 第七阶段：Mailbox 接收 Token 和 Terminal

  ## 16. Delta 进入 Mailbox

  GenerationMailbox::tryPushDelta() 位于 generation_mailbox.cpp:28-60。

  Mailbox 有两个容量限制：

  最大排队 Delta 数
  最大排队 Token 数

  如果 Mailbox 满了，返回 false。

  Backend 随后会：

  把请求标记为 Cancelling
  → 提交 Backpressure Terminal
  → 取消 TensorRT-LLM 请求

  对应 trtllm_executor_backend.cpp:568-578。

  所以这是第一层背压：

  Backend → Mailbox 背压

  ———

  ## 17. 最终 Response 生成 TerminalEvent

  如果 TensorRT-LLM 表示生成结束，Backend 在 trtllm_executor_backend.cpp:598-626：

  读取 finish reason
  → finalizeRequest()
  → RequestState::finalize()

  RequestState::finalize() 位于 request_state.cpp:133-163。

  它生成：

  TerminalEvent {
      request_id,
      status,
      finish_reason,
      usage
  }

  并写入 Mailbox 的独立 Terminal 槽。

  phase_ == Terminal 后，再次 finalize 只会返回 AlreadyCommitted，不会产生第二个 Terminal。

  ———

  ## 18. Terminal 进入 Mailbox 后立即释放 Admission

  关键代码在 generation_mailbox.cpp:62-86：

  terminal_.emplace(std::move(terminal));
  terminalCommitted_ = true;

  admissionLease_.release();

  资源释放发生在这里，而不是等待 Gateway 收到 Terminal。

  因此：

  GPU 已经结束生成
  → Terminal 进入 Mailbox
  → active_requests 减 1
  → 输入 Token 预算归还
  → 输出 Token 预算归还

  即使 Gateway 很慢，GPU Admission 资源也已经可以服务新请求。

  ———

  # 第八阶段：RequestPump 将结果送入 IPC

  ## 19. 每请求 Pump 从 Mailbox 取事件

  requestPumpLoop() 位于 runtime_bridge.cpp:578-679。

  它循环执行：

  record->mailbox->waitPop(
      event,
      mailbox_wait_timeout);

  Mailbox 的弹出顺序在 generation_mailbox.cpp:93-125：

  先取所有已排队 Delta
  → 再取 Terminal
  → Terminal 取走后关闭 Mailbox

  所以同一请求内不会出现：

  Terminal → TokenDelta

  ———

  ## 20. Runtime Delta 转成 IPC Delta

  代码在 runtime_bridge.cpp:601-613：

  ipc::TokenDelta wire;
  wire.worker_epoch = config.worker_epoch;
  wire.request_id = delta->request_id;
  wire.sequence_no = delta->sequence_no;
  wire.token_ids = std::move(delta->token_ids);

  egress.enqueueDelta(request_id, Message{wire});

  这里完成第二次类型转换：

  kimrt::llm::TokenDelta
  → kimrt::llm::ipc::TokenDelta

  区别是 IPC 版本额外携带：

  - protocol version；
  - worker epoch；
    -网络稳定字段。

  ———

  # 第九阶段：SessionEgress 公平排队

  ## 21. 为什么不能直接 session.send()

  假设 A 请求生成很快：

  A1 A2 A3 A4 A5 A6...

  B 请求很短：

  B1 Terminal

  如果所有 Pump 直接写同一个共享 FIFO，B 的 Terminal 可能长期排在 A 后面。

  所以 RequestPump 不直接发送，而是进入 SessionEgress。

  ———

  ## 22. SessionEgress 的内部队列

  定义在 session_egress.cpp:73-153：

  control queue
      Accepted / Rejected / Stats

  request queues
      request 100 → Delta0, Delta1, Terminal
      request 101 → Delta0, Terminal

  round_robin
      100 → 101 → 100 → 101

  waitPop() 在 session_egress.cpp:404-462 中：

  1. 先取控制消息；
  2. 再从一个请求取一条消息；
  3. 请求还有消息就放到 round-robin 尾部；
  4. 下次轮到另一个请求。

  因此输出会类似：

  Accepted(100)
  Accepted(101)
  Delta(100, 0)
  Delta(101, 0)
  Delta(100, 1)
  Terminal(101)
  Terminal(100)

  ———

  ## 23. Terminal 为什么不会被 Delta 挤掉

  注册请求时：

  egress.registerRequest(request_id);

  SessionEgress 会提前保留：

  1 个 Terminal frame
  terminal_reserve_bytes

  对应 session_egress.cpp:162-188。

  普通 Delta 入队检查位于 session_egress.cpp:300-318：

  queue.messages.size() + 2 >
      max_request_frames

  这里的两个位置分别是：

  当前准备加入的 Delta
  未来必须加入的 Terminal

  如果只剩一个位置，Delta 会被拒绝，但 Terminal 仍然可以进入。

  这是第二层背压：

  Mailbox → IPC Egress 背压

  当这里发生背压时，RuntimeBridge 会取消这个请求，并最终向 Gateway 返回：

  Terminal {
      status = QueueFull
      finish_reason = Backpressure
  }

  其他请求不受影响。

  ———

  # 第十阶段：结果写回 Gateway

  ## 24. RuntimeBridge writer 取出消息

  Bridge writer 位于 runtime_bridge.cpp:691-739：

  egress.waitPop(message, timeout);
  session.send(message);

  如果 IpcSession 自己的队列暂时满了，Bridge 会短暂重试：

  每隔 session_send_retry 重试
  最多等待 session_stall_timeout

  超过最大停滞时间，说明连接已经无法正常消费，Bridge 会关闭 Session 并取消活动请求。

  ———

  ## 25. IpcSession 编码并写入 UDS

  session.send() 再进入 ipc_transport.cpp:686-729：

  IPC TokenDelta/Terminal
  → JSON
  → 长度前缀 Frame
  → IpcSession 有界队列

  最后 IpcSession writer 在 ipc_transport.cpp:902-945：

  connection_.writeAll(frame.bytes);

  Gateway 收到后解码，按 request ID 将消息交给相应的 HTTP/SSE 请求。

  这一部分 Gateway 逻辑目前还没有实现，属于 A6-4。

  ———

  # 第十一阶段：Terminal 后各层如何清理

  Terminal 会依次触发不同层的清理：

   时间点                         清理内容                                对应代码
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Terminal 进入 Mailbox          释放 AdmissionLease                     generation_mailbox.cpp:62-86
  ─────────────────────────────  ──────────────────────────────────────  ───────────────────────────────────
   Backend finalize 成功          删除 Executor/External ID 映射          trtllm_executor_backend.cpp:690-
                                                                          707
  ─────────────────────────────  ──────────────────────────────────────  ───────────────────────────────────
   Terminal 进入 SessionEgress    标记 Bridge 请求已终态                  runtime_bridge.cpp:656-667
  ─────────────────────────────  ──────────────────────────────────────  ───────────────────────────────────
   Terminal 从 Egress 弹出        删除请求输出队列                        session_egress.cpp:445-453
  ─────────────────────────────  ──────────────────────────────────────  ───────────────────────────────────
   Bridge 后续回收                join RequestPump、删除 RequestRecord    runtime_bridge.cpp:770-809
  ─────────────────────────────  ──────────────────────────────────────  ───────────────────────────────────
   Gateway 收到 Terminal          结束 SSE/HTTP 请求                      A6-4 尚未实现

  这些状态表看起来重复，其实分别管理不同资源：

   状态表                    管什么
  ━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━
   Admission active IDs      Token 和并发预算
  ────────────────────────  ─────────────────────────
   Backend ID maps           TensorRT-LLM 请求
  ────────────────────────  ─────────────────────────
   RuntimeBridge requests    当前 Session 拥有的请求
  ────────────────────────  ─────────────────────────
   SessionEgress requests    尚未发送完的消息

  ———

  # 取消请求的流程

  Gateway 发送：

  Cancel(request_id=100)

  经过 IpcSession 后进入 runtime_bridge.cpp:495-510。

  Bridge 查找请求归属，然后调用：

  cancelRecord(record, true);

  内部使用：

  if (record->cancel_requested.exchange(true)) {
      return;
  }
  runtime.cancel(request_id);

  因此重复 Cancel 只会有一次真正到达 Backend。

  真实 Backend 在 trtllm_executor_backend.cpp:227-251：

  RequestState → Cancelling
  → executor.cancelRequest()
  → 丢弃后续迟到 Token
  → 等待最终 Response
  → 生成 Cancelled Terminal

  完整顺序：

  Cancel
  → Backend cancel
  → RequestState Cancelling
  → 不再交付迟到 Token
  → Terminal(Cancelled)
  → Mailbox
  → SessionEgress
  → Gateway

  ———

  # Gateway 断连的流程

  Bridge monitor 在 runtime_bridge.cpp:741-768 等待 Session 关闭。

  发现断连后：

  Session Unavailable
  → RuntimeBridge beginShutdown
  → 停止输出队列
  → 遍历当前 Session 的 requests
  → 逐个 runtime.cancel()
  → Backend 提交 Terminal
  → Admission 资源归零

  由于 Gateway 已经断开，Worker 侧不再尝试发送 Terminal。

  未来 A6-4 的 GenerationClient 需要在 Gateway 侧为所有已 Accepted、未收到 Terminal 的请求合成：

  Unavailable Terminal

  ———

  # 测试里完整请求是怎么跑的

  A6-3 的 BridgeFixture 位于 llm_runtime_bridge_test.cpp:417-469，它实际创建了：

  真实 UDS Listener
  真实 Client IpcSession
  真实 Server IpcSession
  真实 GenerationRuntime
  真实 AdmissionController
  真实 GenerationMailbox
  Fake Backend
  真实 RuntimeBridge
  真实 SessionEgress

  正常请求测试位于 llm_runtime_bridge_test.cpp:671-770：

  1. Client 发送 Submit(100)
  2. Fake Backend 收到 GenerationRequest
  3. Fake Backend 写入 TokenDelta
  4. Fake Backend 写入 TerminalEvent
  5. Client 收到 3 条消息

  断言消息顺序必须是：

  messages[0] = Accepted
  messages[1] = TokenDelta
  messages[2] = Terminal

  然后继续验证 Admission：

  active_requests        == 0
  reserved_input_tokens  == 0
  reserved_output_tokens == 0

  所以 A6-3 测试验证的是：

  > 从真实 UDS 请求进入，到 Runtime、Mailbox、公平 Egress，再回到真实 UDS Client 的完整 Worker 侧链路。

  唯一被替换的是 GPU/TensorRT-LLM Backend，这部分真实 Engine 验证属于 A6-5。

  ———

  ## 最后用一句话记住每层

   模块                   可以把它理解成
  ━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   IpcSession             收发室：负责字节和连接
  ─────────────────────  ──────────────────────────────────────────────────
   RuntimeBridge          接线员：把 IPC 接到 Runtime
  ─────────────────────  ──────────────────────────────────────────────────
   AdmissionController    售票口：容量不足就拒绝
  ─────────────────────  ──────────────────────────────────────────────────
   GenerationRuntime      总调度入口：申请资源、创建 Mailbox、提交 Backend
  ─────────────────────  ──────────────────────────────────────────────────
   TRT-LLM Backend        真正执行推理
  ─────────────────────  ──────────────────────────────────────────────────
   RequestState           请求状态机：Token 顺序和唯一 Terminal
  ─────────────────────  ──────────────────────────────────────────────────
   GenerationMailbox      每请求结果邮箱
  ─────────────────────  ──────────────────────────────────────────────────
   SessionEgress          公平发件队列
  ─────────────────────  ──────────────────────────────────────────────────
   IpcSession writer      最终把帧写回 Gateway

  整条主链可以压缩成：

  Submit
  → 解帧
  → 转换
  → Admission
  → Executor enqueue
  → Accepted
  → GPU Response
  → RequestState
  → Mailbox
  → RequestPump
  → Fair Egress
  → UDS
  → TokenDelta / Terminal

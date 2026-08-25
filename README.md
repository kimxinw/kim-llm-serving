# kim-llm-serving

基于 C++17 与 TensorRT-LLM Executor API 的大模型推理服务后端。项目当前聚焦请求生命周期、流式事件交付、取消、超时、背压和可验证性；连续批处理与 KV Cache 等模型执行能力由 TensorRT-LLM 提供。

## 当前能力

| 领域 | 状态 |
|---|---|
| TensorRT-LLM C++ Executor 接入 | 已完成 |
| Token ID 离线推理 | 已完成 |
| `GenerationBackend` 抽象 | 已完成 |
| 有界 `GenerationMailbox` | 已完成 |
| 流式 `TokenDelta` 与唯一 `TerminalEvent` | 已完成 |
| 取消、Deadline 与停止收敛 | 已完成基础链路 |
| CPU / 无 GPU 契约测试 | `15/15 PASS`（2026-08-25） |
| Direct Backend 真实 Engine 生命周期测试 | 已完成基线 |
| IPC v1 codec 与真实 UDS Session | 已完成 A6-2（`b625df0`） |
| Runtime Bridge 与终态感知公平 Egress | 已完成 A6-3（`050a263`） |
| 单活 WorkerServer、独立 `llm_worker` 与 Python `GenerationClient` | 已完成基础版（`1676515`） |
| Python → UDS → Worker → TinyLlama Engine | A6-4 已完成：流式/非流式 Token IDs 与 Direct 路径一致，Worker 断连后未完成请求收敛为 `Unavailable`（`2589c4e`） |
| Admission、全局输出预算、Direct Benchmark | 已完成基础版 |
| 固定 workload 的 Direct / IPC 增量开销 | A6-5 已完成（`3334046`） |
| CPU-only configure/build/test | 已完成模块边界 |
| Tokenizer、OpenAI API、SSE | 最小 Gateway 已完成（`0581c0a`） |
| HTTP/SSE Benchmark Harness | closed/open-loop、有界 `max_inflight`、SLO goodput、拒绝分类、慢客户端分组与证据门禁已完成（`c37cc81`）；Gateway 正常停止误判已修复（`c3f24ab`） |
| 可观测性与正式性能矩阵 | Gateway/SSE/Worker egress 高水位及 Admission 快照已接入；真实 HTTP 分层数据与正式过载矩阵未完成 |

## 架构边界

```text
OpenAI Client
  -> FastAPI Gateway / Hugging Face Tokenizer
  -> Python GenerationClient
  -> UDS
  -> WorkerServer / IpcSession
  -> IPC Message
  -> RuntimeBridge / SessionEgress
  -> GenerationRuntime
      -> AdmissionController
      -> RequestState + GenerationMailbox
      -> GenerationBackend / TrtLlmExecutorBackend
          -> TensorRT-LLM Executor
              -> GPU
```

本项目实现 TensorRT-LLM 之上的 Serving Runtime，不宣称重新实现 Attention Kernel、Paged KV Cache 或连续批处理引擎。

## 目录

```text
include/common/           通用状态与请求上下文
include/runtime/          生成类型、Backend 契约、生命周期与资源控制
include/ipc/              IPC v1 协议、UDS Transport 与 Session
include/worker/           Runtime Bridge 与公平 Egress
include/backends/trtllm/  TensorRT-LLM Backend 公共接口
src/                      与 include 镜像的 runtime/ipc/worker/backends 实现
tests/                    与生产模块镜像的 CPU 契约及 GPU 门禁测试
apps/                     `llm_worker`、离线推理等可执行程序入口
clients/python/           Python GenerationClient 与 IPC 协议实现
gateway/python/           OpenAI-compatible HTTP/SSE Gateway、Tokenizer 与运行时适配
benchmark/                Direct、IPC 与 HTTP/SSE Benchmark
docs/                     Pipeline、开发计划与历史任务
```

## 构建

CPU 契约不需要 CUDA、TensorRT、TensorRT-LLM、MPI 或 Conda 环境：

```bash
cmake -S . -B build-cpu \
  -DKIM_LLM_ENABLE_TRTLLM=OFF
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
```

GPU 路径的验证基线使用 TensorRT-LLM `0.16.0`、TensorRT `10.7.0.23`、C++17 与 `_GLIBCXX_USE_CXX11_ABI=0`。构建前需要激活对应 Conda 环境，并显式提供依赖路径：

```bash
cmake -S . -B build-llm \
  -DKIM_LLM_ENABLE_TRTLLM=ON \
  -DTRTLLM_SOURCE_DIR=/path/to/TensorRT-LLM \
  -DTRTLLM_LIB_DIR=/path/to/tensorrt_llm/libs \
  -DTENSORRT_ROOT=/path/to/TensorRT-10.7.0.23 \
  -DKIMRT_LLM_TEST_ENGINE_DIR=/path/to/engine

cmake --build build-llm --parallel
ctest --test-dir build-llm --output-on-failure
ctest --test-dir build-llm -L gpu --output-on-failure -V
```

`KIMRT_LLM_TEST_ENGINE_DIR` 为空时，只注册无 GPU 测试；非空时额外注册 Direct Backend 与 Python/Worker 跨进程真实 Engine 测试。GPU 测试统一标记为 `gpu;integration`，并通过资源锁避免并行争抢 GPU 0。

## 最小 Gateway

提交 `0581c0a` 完成固定单模型 Gateway：启动时严格校验 Tokenizer、Chat Template、
特殊 Token 与 Worker `ModelManifest`，提供 `/v1/chat/completions` 非流式及 SSE、
`/healthz`、`/readyz`、`/metrics` 和 `/v1/models`。每请求 SSE 队列和 Gateway
待处理请求数均有硬上限；HTTP 断连或慢消费者溢出会触发幂等 Cancel，并在后台继续
排空该请求直到唯一 Terminal，避免提前释放 Worker 生命周期。

```bash
PYTHONPATH=clients/python:gateway/python \
  python -m kim_llm_gateway inspect-tokenizer \
  --tokenizer-path /path/to/tokenizer

PYTHONPATH=clients/python:gateway/python \
  python -m kim_llm_gateway serve \
  --config configs/gateway.example.json
```

2026-08-24 真实 TinyLlama 链路已验证非流式和 SSE 输出一致；非流式响应的
`prompt_tokens/completion_tokens` 为 `17/8`，HTTP 流式客户端主动断连后
`http_disconnect_cancels_total=1`、活动请求回到 `0`，Worker 与 Gateway 均可优雅停止。

## HTTP/SSE Benchmark Harness

提交 `c37cc81` 新增隔离运行的 HTTP/SSE Benchmark：每轮自动启动并停止独立
Worker/Gateway，使用固定 Chat Template 记录实际输入 Token IDs，支持有界
closed-loop/open-loop、固定或 Poisson 到达、TTFT/E2E SLO goodput、拒绝原因、
慢客户端分组以及 Gateway/Worker 资源归零检查。`max_inflight` 是客户端硬边界，
容量耗尽记录为 `client_overflow`，不会形成无界任务队列。

Gateway `/metrics` 同步补充 active/SSE buffer 高水位、Worker Admission 当前值和
Session egress 当前值/高水位；metrics 抓取不会为不可用 Worker 隐式重连。正式运行
默认要求 Git 无已跟踪改动且证据路径不存在，避免把开发态或被覆盖的结果当作结论。

截至 2026-08-25，Harness 代码和 CPU 契约已经完成，当前无 GPU 回归
`14/14 PASS`。提交 `c3f24ab` 修复 Harness 主动发送 `SIGTERM` 后将 Gateway
预期返回码 `-15` 误判为失败的问题，并补充主动停止与意外退出的区分测试。修复后的
真实 Engine C1 开发冒烟已稳定生成 summary、CSV 和 Worker/Gateway 日志，5 个测量
请求全部成功且 `resources_released=true`。该轮运行时工作区仍为 dirty，只作为修复
验证；干净提交上的固定 workload 分层数据、慢客户端隔离数据和正式 open-loop
过载矩阵仍未完成，因此不声明 HTTP 开销或 SLO goodput 收益。

## Direct / IPC / HTTP 分层基线

`benchmark/run_direct_ipc_http_benchmark.py` 使用同一 Chat Template 和 Tokenizer
生成固定输入 Token IDs，并按三阶轮换顺序执行 Direct、IPC 和 HTTP/SSE。每轮要求
Direct/IPC 输出 Token IDs 完全一致，同时要求 HTTP 文本逐请求等于相同 Token IDs
经固定 Tokenizer 解码后的结果，且 completion token 数、成功请求数和资源归零状态
全部一致。最终报告给出 TTFT、TPOT、E2E、请求吞吐和输出 Token 吞吐的三路径均值、
样本标准差以及 `IPC - Direct`、`HTTP - IPC` 配对增量。

设计参考 NVIDIA TensorRT-LLM Triton Backend 的
[`benchmark_core_model.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/triton_backend/tools/inflight_batcher_llm/benchmark_core_model.py)：
采用“核心推理路径与端到端服务路径分层测量”和机器可读 workload 元数据；本项目调整为
三路径轮换配对实验，并增加输出等价性、协议正确性和资源释放门禁，不采用 Triton/gRPC
传输及其请求 schema。

```bash
python3 benchmark/run_direct_ipc_http_benchmark.py \
  --direct-benchmark build-llm/llm_direct_benchmark \
  --worker build-llm/llm_worker \
  --engine-dir /path/to/tinyllama-engine \
  --tokenizer-path /path/to/TinyLlama-1.1B-Chat-v1.0 \
  --output-dir benchmark/results/direct-ipc-http-c1 \
  --repetitions 3 \
  --concurrency 1 \
  --warmup-requests 5 \
  --measured-requests 200 \
  --max-new-tokens 32 \
  --prompt Hello
```

正式运行要求 Git 无已跟踪改动、结果目录不存在且至少完成三次重复；只有完整三轮
轮换才能标记 `minimum_repetitions_met=true` 和
`balanced_execution_order_cycle=true`。单轮 `--allow-dirty` 仅用于开发冒烟。
新增三路径比较契约后，当前无 GPU 回归为 `15/15 PASS`。

## Direct / IPC 增量开销

`llm_direct_benchmark` 与 `benchmark/llm_ipc_benchmark.py` 使用相同的
closed-loop workload、采样参数和请求计时边界。总控脚本默认交错执行 3 轮
Direct/IPC 配对实验，逐请求校验输出 Token IDs 完全一致，并报告 TTFT、
TPOT、E2E、request throughput 和 output token throughput 的 IPC 增量及样本标准差。
这里的增量包含 Python `GenerationClient`、JSON 编解码、UDS、`WorkerServer`、
`GenerationRuntime` 与 Bridge/Egress，而不是只测 UDS 系统调用的微基准。

```bash
python3 benchmark/run_direct_ipc_benchmark.py \
  --direct-benchmark build-llm/llm_direct_benchmark \
  --worker build-llm/llm_worker \
  --engine-dir /path/to/tinyllama-engine \
  --output-dir benchmark/results/direct-ipc-c1 \
  --repetitions 3 \
  --concurrency 1 \
  --warmup-requests 5 \
  --measured-requests 50 \
  --max-new-tokens 32 \
  --input-token-ids 1 2 3 4
```

正式证据默认要求 Git 工作区无已跟踪改动，结果目录也必须不存在，避免覆盖旧实验。
脚本在相邻轮次交换 Direct/IPC 执行顺序以降低温度和时间漂移影响；最终报告为
`<output-dir>/direct-ipc-comparison.json`，每轮原始 summary、逐请求 CSV、
Direct iteration stats、Worker 日志和命令日志均保留在对应 `run-XX/` 目录。
开发期单轮冒烟可使用 `--repetitions 1 --allow-dirty`，但不得作为正式性能结论。

截至提交 `3334046`，A6-5 已使用 C1/ISL4/OSL32 固定 workload 完成
3 轮交错配对实验，每条路径每轮测量 200 个请求。Direct 与 IPC 的逐请求
Token IDs 完全一致，失败、Rejected、Backpressure 和 Cancel 均为 0，资源最终归零。

| 指标 | IPC 相对 Direct 的配对均值 |
|---|---:|
| TTFT P50 | `+0.661 ms`（`+8.13%`） |
| TPOT P50 | `+0.025 ms`（`+0.33%`） |
| E2E P50 | `+1.482 ms`（`+0.61%`） |
| request throughput | `-0.66%` |
| output token throughput | `-0.66%` |

结果说明当前进程与协议隔离成本主要体现为亚毫秒级首 Token 固定开销，
稳态 Decode 基本不变；该结论只适用于上述固定模型、Engine、硬件和 workload，
不替代后续 HTTP 路径及正式过载矩阵。

## 路线图

当前收尾顺序和验收条件见 [求职.md](求职.md)；长期路线与历史设计参考见 [docs/LLM_INFERENCE_PLAN.md](docs/LLM_INFERENCE_PLAN.md)。

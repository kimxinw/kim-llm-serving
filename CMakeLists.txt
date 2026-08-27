cmake_minimum_required(VERSION 3.18)

project(kim_llm_serving LANGUAGES CXX)

include(CTest)
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(
    KIM_LLM_ENABLE_TRTLLM
    "Build the TensorRT-LLM backend, GPU applications, and Engine tests"
    ON
)

set(KIM_LLM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Threads REQUIRED)

find_package(nlohmann_json 3.11.3 QUIET)
if(NOT nlohmann_json_FOUND)
    set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

function(kim_llm_enable_warnings target_name)
    target_compile_options(
        ${target_name}
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )
endfunction()

function(
    kim_llm_add_contract_test
    target_name
    test_name
    source_file
    timeout_seconds
)
    add_executable(
        ${target_name}
        "${KIM_LLM_ROOT}/${source_file}"
    )
    kim_llm_enable_warnings(${target_name})
    target_link_libraries(${target_name} PRIVATE ${ARGN})
    add_test(NAME ${test_name} COMMAND ${target_name})
    set_tests_properties(
        ${test_name}
        PROPERTIES TIMEOUT ${timeout_seconds}
    )
endfunction()

# Shared value types and public include/ABI contract. This target deliberately
# has no compiled sources and no dependency on a runtime, transport, or backend.
add_library(kim_llm_common INTERFACE)
add_library(kim_llm::common ALIAS kim_llm_common)

target_include_directories(
    kim_llm_common
    INTERFACE
        "${KIM_LLM_ROOT}/include"
)

target_compile_definitions(
    kim_llm_common
    INTERFACE
        _GLIBCXX_USE_CXX11_ABI=0
)

# Pure CPU request lifecycle, resource admission, mailbox, and backend contract.
add_library(
    kim_llm_runtime
    STATIC
        "${KIM_LLM_ROOT}/src/runtime/admission_controller.cpp"
        "${KIM_LLM_ROOT}/src/runtime/generation_mailbox.cpp"
        "${KIM_LLM_ROOT}/src/runtime/generation_runtime.cpp"
        "${KIM_LLM_ROOT}/src/runtime/request_state.cpp"
        "${KIM_LLM_ROOT}/src/runtime/slo_admission_policy.cpp"
)
add_library(kim_llm::runtime ALIAS kim_llm_runtime)

kim_llm_enable_warnings(kim_llm_runtime)

target_link_libraries(
    kim_llm_runtime
    PUBLIC
        kim_llm::common
    PRIVATE
        Threads::Threads
)

# Versioned wire messages and strict codec. No socket or GPU dependency.
add_library(
    kim_llm_ipc_protocol
    STATIC
        "${KIM_LLM_ROOT}/src/ipc/ipc_protocol.cpp"
)
add_library(kim_llm::ipc_protocol ALIAS kim_llm_ipc_protocol)

kim_llm_enable_warnings(kim_llm_ipc_protocol)

target_link_libraries(
    kim_llm_ipc_protocol
    PUBLIC
        kim_llm::common
    PRIVATE
        nlohmann_json::nlohmann_json
)

# UDS connection and session lifecycle. It depends only on the protocol and
# the platform thread library, so real-socket tests remain CPU-only.
add_library(
    kim_llm_ipc_transport
    STATIC
        "${KIM_LLM_ROOT}/src/ipc/ipc_transport.cpp"
)
add_library(kim_llm::ipc_transport ALIAS kim_llm_ipc_transport)

kim_llm_enable_warnings(kim_llm_ipc_transport)

target_link_libraries(
    kim_llm_ipc_transport
    PUBLIC
        kim_llm::ipc_protocol
    PRIVATE
        Threads::Threads
)

# Worker-side Session supervision and orchestration between the CPU runtime
# and IPC. The llm_worker executable composes this target with one backend.
add_library(
    kim_llm_worker_core
    STATIC
        "${KIM_LLM_ROOT}/src/worker/runtime_bridge.cpp"
        "${KIM_LLM_ROOT}/src/worker/session_egress.cpp"
        "${KIM_LLM_ROOT}/src/worker/worker_server.cpp"
)
add_library(kim_llm::worker_core ALIAS kim_llm_worker_core)

kim_llm_enable_warnings(kim_llm_worker_core)

target_link_libraries(
    kim_llm_worker_core
    PUBLIC
        kim_llm::runtime
        kim_llm::ipc_transport
    PRIVATE
        Threads::Threads
)

if(KIM_LLM_ENABLE_TRTLLM)
    set(
        TRTLLM_SOURCE_DIR
        ""
        CACHE PATH
        "TensorRT-LLM v0.16.0 source directory"
    )
    set(
        TRTLLM_LIB_DIR
        ""
        CACHE PATH
        "Directory containing TensorRT-LLM wheel shared libraries"
    )
    set(
        TENSORRT_ROOT
        ""
        CACHE PATH
        "TensorRT 10.7.0.23 SDK root"
    )

    foreach(required_var IN ITEMS
        TRTLLM_SOURCE_DIR
        TRTLLM_LIB_DIR
        TENSORRT_ROOT
    )
        if("${${required_var}}" STREQUAL "")
            message(FATAL_ERROR "${required_var} must be provided")
        endif()
    endforeach()

    if(NOT DEFINED ENV{CONDA_PREFIX})
        message(FATAL_ERROR "Activate conda trtllm016 first")
    endif()

    find_package(CUDAToolkit REQUIRED)
    find_package(MPI REQUIRED COMPONENTS CXX)

    find_path(
        TRT_INCLUDE_DIR
        NvInfer.h
        PATHS
            "${TENSORRT_ROOT}/include"
            "${TENSORRT_ROOT}/targets/x86_64-linux-gnu/include"
        NO_DEFAULT_PATH
    )

    find_library(
        TRT_NVINFER_LIBRARY
        NAMES nvinfer
        PATHS
            "${TENSORRT_ROOT}/lib"
            "${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib"
        NO_DEFAULT_PATH
    )

    find_library(
        TRTLLM_LIBRARY
        NAMES tensorrt_llm
        PATHS "${TRTLLM_LIB_DIR}"
        NO_DEFAULT_PATH
    )

    find_library(
        TRTLLM_PLUGIN_LIBRARY
        NAMES nvinfer_plugin_tensorrt_llm
        PATHS "${TRTLLM_LIB_DIR}"
        NO_DEFAULT_PATH
    )

    foreach(required_path IN ITEMS
        TRT_INCLUDE_DIR
        TRT_NVINFER_LIBRARY
        TRTLLM_LIBRARY
        TRTLLM_PLUGIN_LIBRARY
    )
        if(NOT ${required_path})
            message(FATAL_ERROR "${required_path} was not found")
        endif()
    endforeach()

    get_filename_component(
        TRT_LIBRARY_DIR
        "${TRT_NVINFER_LIBRARY}"
        DIRECTORY
    )

    add_library(
        kim_llm_trtllm_backend
        STATIC
            "${KIM_LLM_ROOT}/src/backends/trtllm/trtllm_executor_backend.cpp"
    )
    add_library(kim_llm::trtllm_backend ALIAS kim_llm_trtllm_backend)

    kim_llm_enable_warnings(kim_llm_trtllm_backend)

    target_include_directories(
        kim_llm_trtllm_backend
        PRIVATE
            "${TRTLLM_SOURCE_DIR}/cpp/include"
            "${TRT_INCLUDE_DIR}"
            "${CUDAToolkit_INCLUDE_DIRS}"
            "$ENV{CONDA_PREFIX}/targets/x86_64-linux/include"
    )

    target_compile_definitions(
        kim_llm_trtllm_backend
        PRIVATE
            ENABLE_MULTI_DEVICE=1
            ENABLE_BF16
            ENABLE_FP8
    )

    target_link_libraries(
        kim_llm_trtllm_backend
        PUBLIC
            kim_llm::runtime
        PRIVATE
            "${TRTLLM_PLUGIN_LIBRARY}"
            "${TRTLLM_LIBRARY}"
            "${TRT_NVINFER_LIBRARY}"
            CUDA::cuda_driver
            CUDA::cudart_static
            CUDA::nvml
            MPI::MPI_CXX
            Threads::Threads
            "${CMAKE_DL_LIBS}"
    )

    add_executable(
        llm_worker
        "${KIM_LLM_ROOT}/apps/llm_worker.cpp"
    )
    kim_llm_enable_warnings(llm_worker)
    target_link_options(llm_worker PRIVATE -Wl,--allow-shlib-undefined)
    target_link_libraries(
        llm_worker
        PRIVATE
            kim_llm::worker_core
            kim_llm::trtllm_backend
            nlohmann_json::nlohmann_json
    )

    add_executable(
        llm_infer
        "${KIM_LLM_ROOT}/apps/llm_infer.cpp"
    )
    kim_llm_enable_warnings(llm_infer)
    target_link_options(llm_infer PRIVATE -Wl,--allow-shlib-undefined)
    target_link_libraries(llm_infer PRIVATE kim_llm::trtllm_backend)

    add_executable(
        llm_direct_benchmark
        "${KIM_LLM_ROOT}/benchmark/llm_direct_benchmark.cpp"
    )
    kim_llm_enable_warnings(llm_direct_benchmark)
    target_link_options(
        llm_direct_benchmark
        PRIVATE
            -Wl,--allow-shlib-undefined
    )
    target_link_libraries(
        llm_direct_benchmark
        PRIVATE
            kim_llm::trtllm_backend
    )
endif()

if(BUILD_TESTING)
    kim_llm_add_contract_test(
        llm_ipc_protocol_test
        llm_ipc_protocol_contract
        tests/ipc/llm_ipc_protocol_test.cpp
        10
        kim_llm::ipc_protocol
    )

    kim_llm_add_contract_test(
        llm_ipc_transport_test
        llm_ipc_transport_contract
        tests/ipc/llm_ipc_transport_test.cpp
        15
        kim_llm::ipc_transport
    )

    kim_llm_add_contract_test(
        llm_generation_runtime_test
        llm_generation_runtime_contract
        tests/runtime/llm_generation_runtime_test.cpp
        10
        kim_llm::runtime
    )

    kim_llm_add_contract_test(
        llm_runtime_bridge_test
        llm_runtime_bridge_contract
        tests/worker/llm_runtime_bridge_test.cpp
        20
        kim_llm::worker_core
    )

    kim_llm_add_contract_test(
        llm_worker_server_test
        llm_worker_server_contract
        tests/worker/llm_worker_server_test.cpp
        20
        kim_llm::worker_core
    )

    kim_llm_add_contract_test(
        llm_admission_controller_test
        llm_admission_controller_contract
        tests/runtime/llm_admission_controller_test.cpp
        10
        kim_llm::runtime
    )

    kim_llm_add_contract_test(
        llm_slo_admission_policy_test
        llm_slo_admission_policy_contract
        tests/runtime/llm_slo_admission_policy_test.cpp
        10
        kim_llm::runtime
    )

    kim_llm_add_contract_test(
        llm_generation_mailbox_test
        llm_generation_mailbox_contract
        tests/runtime/llm_generation_mailbox_test.cpp
        10
        kim_llm::runtime
    )

    kim_llm_add_contract_test(
        llm_request_state_test
        llm_request_state_contract
        tests/runtime/llm_request_state_test.cpp
        10
        kim_llm::runtime
    )

    kim_llm_add_contract_test(
        llm_generation_types_test
        llm_generation_types_contract
        tests/runtime/llm_generation_types_test.cpp
        10
        kim_llm::runtime
    )

    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME llm_python_generation_client_contract
            COMMAND
                "${Python3_EXECUTABLE}"
                "${KIM_LLM_ROOT}/tests/clients/python/test_generation_client.py"
        )
        set_tests_properties(
            llm_python_generation_client_contract
            PROPERTIES
                TIMEOUT 20
                ENVIRONMENT "PYTHONPATH=${KIM_LLM_ROOT}/clients/python"
        )
        add_test(
            NAME llm_python_direct_ipc_benchmark_contract
            COMMAND
                "${Python3_EXECUTABLE}"
                "${KIM_LLM_ROOT}/tests/benchmark/test_direct_ipc_benchmark.py"
        )
        set_tests_properties(
            llm_python_direct_ipc_benchmark_contract
            PROPERTIES
                TIMEOUT 20
                ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
        )
        add_test(
            NAME llm_python_direct_ipc_http_benchmark_contract
            COMMAND
                "${Python3_EXECUTABLE}"
                "${KIM_LLM_ROOT}/tests/benchmark/test_direct_ipc_http_benchmark.py"
        )
        set_tests_properties(
            llm_python_direct_ipc_http_benchmark_contract
            PROPERTIES
                TIMEOUT 20
                ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
        )
        add_test(
            NAME llm_python_http_benchmark_contract
            COMMAND
                "${Python3_EXECUTABLE}"
                "${KIM_LLM_ROOT}/tests/benchmark/test_http_benchmark.py"
        )
        set_tests_properties(
            llm_python_http_benchmark_contract
            PROPERTIES
                TIMEOUT 20
                ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
        )
        add_test(
            NAME llm_python_gateway_core_contract
            COMMAND
                "${Python3_EXECUTABLE}"
                "${KIM_LLM_ROOT}/tests/gateway/python/test_gateway_core.py"
        )
        set_tests_properties(
            llm_python_gateway_core_contract
            PROPERTIES
                TIMEOUT 20
                ENVIRONMENT
                    "PYTHONPATH=${KIM_LLM_ROOT}/clients/python:${KIM_LLM_ROOT}/gateway/python;PYTHONDONTWRITEBYTECODE=1"
        )
        add_test(
            NAME llm_python_gateway_api_contract
            COMMAND
                "${Python3_EXECUTABLE}"
                "${KIM_LLM_ROOT}/tests/gateway/python/test_gateway_api.py"
        )
        set_tests_properties(
            llm_python_gateway_api_contract
            PROPERTIES
                TIMEOUT 20
                ENVIRONMENT
                    "PYTHONPATH=${KIM_LLM_ROOT}/clients/python:${KIM_LLM_ROOT}/gateway/python:${KIM_LLM_ROOT}/tests/gateway/python;PYTHONDONTWRITEBYTECODE=1"
        )
        add_test(
            NAME llm_python_project_entrypoint_contract
            COMMAND
                "${Python3_EXECUTABLE}"
                "${KIM_LLM_ROOT}/tests/tools/test_project_entrypoint.py"
        )
        set_tests_properties(
            llm_python_project_entrypoint_contract
            PROPERTIES
                TIMEOUT 20
                ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
        )
    else()
        message(
            STATUS
            "Python3 interpreter was not found; Python client contract test is not registered"
        )
    endif()

    if(KIM_LLM_ENABLE_TRTLLM)
        set(
            KIMRT_LLM_TEST_ENGINE_DIR
            ""
            CACHE PATH
            "TinyLlama Engine directory used by LLM Backend lifecycle tests"
        )

        add_executable(
            llm_backend_lifecycle_test
            "${KIM_LLM_ROOT}/tests/backends/trtllm/llm_backend_lifecycle_test.cpp"
        )
        kim_llm_enable_warnings(llm_backend_lifecycle_test)
        target_link_options(
            llm_backend_lifecycle_test
            PRIVATE
                -Wl,--allow-shlib-undefined
        )
        target_link_libraries(
            llm_backend_lifecycle_test
            PRIVATE
                kim_llm::trtllm_backend
        )

        if(NOT "${KIMRT_LLM_TEST_ENGINE_DIR}" STREQUAL "")
            add_test(
                NAME llm_backend_token_and_lifecycle_baseline
                COMMAND
                    llm_backend_lifecycle_test
                    "${KIMRT_LLM_TEST_ENGINE_DIR}"
            )
            set_tests_properties(
                llm_backend_token_and_lifecycle_baseline
                PROPERTIES
                    TIMEOUT 300
                    ENVIRONMENT "CUDA_VISIBLE_DEVICES=0"
                    LABELS "gpu;integration"
                    RESOURCE_LOCK kim_llm_gpu_0
            )

            if(Python3_Interpreter_FOUND)
                add_test(
                    NAME llm_python_real_worker_engine_e2e
                    COMMAND
                        "${Python3_EXECUTABLE}"
                        "${KIM_LLM_ROOT}/tests/integration/test_real_worker_engine.py"
                        --worker "$<TARGET_FILE:llm_worker>"
                        --engine-dir "${KIMRT_LLM_TEST_ENGINE_DIR}"
                )
                set_tests_properties(
                    llm_python_real_worker_engine_e2e
                    PROPERTIES
                        TIMEOUT 360
                        ENVIRONMENT
                            "CUDA_VISIBLE_DEVICES=0;PYTHONPATH=${KIM_LLM_ROOT}/clients/python;PYTHONDONTWRITEBYTECODE=1"
                        LABELS "gpu;integration"
                        RESOURCE_LOCK kim_llm_gpu_0
                )
            else()
                message(
                    STATUS
                    "Python3 interpreter was not found; real Worker/Engine IPC test is not registered"
                )
            endif()
        else()
            message(
                STATUS
                "KIMRT_LLM_TEST_ENGINE_DIR is empty; Engine test is not registered"
            )
        endif()
    else()
        message(
            STATUS
            "KIM_LLM_ENABLE_TRTLLM=OFF; GPU applications and Engine tests are disabled"
        )
    endif()
endif()

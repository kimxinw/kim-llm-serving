cmake_minimum_required(VERSION 3.18)

project(kimrt_llm_l1 LANGUAGES CXX)

include(CTest)

get_filename_component(
    KIMRT_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../.."
    ABSOLUTE
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Release)
endif()

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
find_package(Threads REQUIRED)

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
    kimrt_llm
    STATIC
    "${KIMRT_ROOT}/src/models/llm/trtllm_executor_backend.cpp"
    "${KIMRT_ROOT}/src/core/serial_executor.cpp"
)

target_include_directories(
    kimrt_llm
    PUBLIC
        "${KIMRT_ROOT}/include"
    PRIVATE
        "${TRTLLM_SOURCE_DIR}/cpp/include"
        "${TRT_INCLUDE_DIR}"
        "${CUDAToolkit_INCLUDE_DIRS}"
        "$ENV{CONDA_PREFIX}/targets/x86_64-linux/include"
)

target_compile_definitions(
    kimrt_llm
    PUBLIC
        _GLIBCXX_USE_CXX11_ABI=0
    PRIVATE
        ENABLE_MULTI_DEVICE=1
        ENABLE_BF16
        ENABLE_FP8
)

target_compile_options(
    kimrt_llm
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_link_libraries(
    kimrt_llm
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
    llm_infer
    llm_infer.cpp
)

target_compile_options(
    llm_infer
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_link_options(
    llm_infer
    PRIVATE
        -Wl,--allow-shlib-undefined
)

target_link_libraries(
    llm_infer
    PRIVATE
        kimrt_llm
)

if(BUILD_TESTING)
    add_executable(
        llm_generation_types_test
        "${KIMRT_ROOT}/tests/llm_generation_types_test.cpp"
    )
    target_include_directories(
        llm_generation_types_test
        PRIVATE
            "${KIMRT_ROOT}/include"
    )

    target_compile_definitions(
        llm_generation_types_test
        PRIVATE
            _GLIBCXX_USE_CXX11_ABI=0
    )

    target_compile_options(
        llm_generation_types_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    add_test(
        NAME llm_generation_types_contract
        COMMAND llm_generation_types_test
    )

    set_tests_properties(
        llm_generation_types_contract
        PROPERTIES
            TIMEOUT 10
    )

    set(
        KIMRT_LLM_TEST_ENGINE_DIR
        ""
        CACHE PATH
        "TinyLlama Engine directory used by LLM Backend lifecycle tests"
    )

    add_executable(
        llm_backend_lifecycle_test
        "${KIMRT_ROOT}/tests/llm_backend_lifecycle_test.cpp"
    )

    target_compile_options(
        llm_backend_lifecycle_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    target_link_options(
        llm_backend_lifecycle_test
        PRIVATE
            -Wl,--allow-shlib-undefined
    )

    target_link_libraries(
        llm_backend_lifecycle_test
        PRIVATE
            kimrt_llm
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
        )
    else()
        message(
            STATUS
            "KIMRT_LLM_TEST_ENGINE_DIR is empty; LLM runtime test is not registered"
        )
    endif()
endif()

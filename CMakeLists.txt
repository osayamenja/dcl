cmake_minimum_required(VERSION 4.0)
project(dcl CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(LIB_FLAGS "-Wall -Wextra -Wno-unknown-pragmas -Wnull-dereference -Wnarrowing -Wno-unused-but-set-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LIB_FLAGS}")

set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin -compress-all")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --display_error_number")
set(CMAKE_CUDA_FLAGS "-Xfatbin -compress-all -Xcudafe --display_error_number -Xcompiler \"${LIB_FLAGS}\"")

include(CheckCompilerFlag)
find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(NVSHMEM REQUIRED HINTS "$ENV{NVSHMEM_HOME}/lib/cmake/nvshmem")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
string(SUBSTRING "${CMAKE_CUDA_ARCHITECTURES_NATIVE}" 0 2 COMPUTE_CAPABILITY) # xx-real -> xx
math(EXPR GPU_ARCH "${COMPUTE_CAPABILITY} * 10" OUTPUT_FORMAT DECIMAL)
message(STATUS "GPU Compute Capability: ${COMPUTE_CAPABILITY}")

function(dcl_target_common tgt)
    target_link_libraries(${tgt} PRIVATE
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::nvtx3
            MPI::MPI_CXX
    )
    if(${tgt} STREQUAL "nvsh")
        target_link_libraries(${tgt} PRIVATE nvshmem::nvshmem)
        set_target_properties(${tgt} PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RESOLVE_DEVICE_SYMBOLS   ON
        )
    endif ()

    # Modern arch setting on the target itself
    set_property(TARGET ${tgt} PROPERTY CUDA_ARCHITECTURES "native")

    target_compile_options(${tgt} PRIVATE
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g>
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-O0;-g;-G;--generate-line-info;-Xptxas=-v>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--generate-line-info;-Xptxas=-v>
    )
endfunction()

# Playground for experimentation
add_executable(play
        main.cu
)
dcl_target_common(play)

# Benchmarks
# 1.) Prefill and decode
add_executable(bench_prefill_decode
        benchmark/pd.cu
)
dcl_target_common(bench_prefill_decode)
# 2.) copy engine via cudaMemcpyPeerAsync
add_executable(ce
        benchmark/copy_engine.cu
)
dcl_target_common(ce)
# 3.) ld,st latency
add_executable(lat
        benchmark/instruction_latency.cu
)
dcl_target_common(lat)
# 4.)
add_executable(nvsh
        benchmark/nvsh.cu
)
dcl_target_common(nvsh)
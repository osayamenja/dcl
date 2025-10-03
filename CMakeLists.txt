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

#set_target_properties(dcl PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)

include(CheckCompilerFlag)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
string(SUBSTRING "${CMAKE_CUDA_ARCHITECTURES_NATIVE}" 0 2 COMPUTE_CAPABILITY) # xx-real -> xx
math(EXPR GPU_ARCH "${COMPUTE_CAPABILITY} * 10" OUTPUT_FORMAT DECIMAL)
message(STATUS "GPU Compute Capability: ${COMPUTE_CAPABILITY}")

add_executable(dcl main.cu)
target_link_libraries(dcl PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvtx3)

find_package(MPI REQUIRED COMPONENTS CXX)
target_include_directories(dcl PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(dcl PRIVATE ${MPI_CXX_LIBRARIES})
target_compile_options(dcl PRIVATE ${MPI_CXX_COMPILE_FLAGS})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(dcl PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O0;-g;>
            $<$<COMPILE_LANGUAGE:CUDA>:-O0; -g; -G>
            $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info; -Xptxas -v>
    )
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(dcl PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -v>
    )
endif ()

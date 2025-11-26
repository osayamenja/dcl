// nvshmem_put_device_min.cu
// Minimal device-side NVSHMEM put benchmark.
// Used for investigating emitted instructions

#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>
#include <cuda/memory>
#include <cuda/utility>
#include <cuda/std/atomic>
#include <cstdio>
#include <iostream>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} } while(0)
#define B_TO_GB (1000 * 1000 * 1000)
using ll_t = long long int;
using ull_t = unsigned long long int;

/// Copy
template<int Size>
__device__ __forceinline__
void cp_async_global_to_shared(void* __restrict__ const& smem_ptr, const void* __restrict__ const& gmem_ptr) {
    static_assert(Size == 4 || Size == 8 || Size == 16,
                  "cp.async only supports Size in {4, 8, 16}");
    uint32_t sp = __cvta_generic_to_shared(smem_ptr);
#if __CUDA_ARCH__ >= 900
    // Hopper, Blackwell, etc. (SM90+)
    asm volatile(
        "cp.async.ca.shared.global.L2::256B [%0], [%1], %2;\n"
        :
        : "r"(sp), "l"(gmem_ptr), "n"(Size)
    );
#else
    // SM 800
    asm volatile(
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
        :
        : "r"(sp), "l"(gmem_ptr), "n"(Size)
    );
#endif
}
__device__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template<int N>
__device__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
template<>
__device__ void cp_async_wait<0>() { asm volatile("cp.async.wait_all;\n" ::); }
__device__ __forceinline__
void kxAssume(const bool& exp) {
#if defined(__CUDA_ARCH__)
    __builtin_assume(exp);
#endif
}

#define SC(T, v) static_cast<T>(v)
template<int threads, int pipeStages = 2, int stageExtent = 2, int Alignment = 16>
__device__ __forceinline__
void kxPut(void* __restrict__ const& dst, const void* __restrict__ const& src,
    cuda::std::byte* __restrict__ const& workspace, const size_t& partition) {
    using VT = uint4;
    static_assert(cuda::is_power_of_two(pipeStages) && pipeStages >= 1 && pipeStages <= 8);
    auto* __restrict__ vW = reinterpret_cast<VT*>(workspace);
    //assert(__isShared(workspace));
    const int vP = static_cast<int>(partition / Alignment);
    auto* __restrict__ vD = static_cast<VT*>(dst);
    const auto* __restrict__ vS = static_cast<const VT*>(src);
    if (partition <= threads * Alignment * pipeStages * stageExtent) {
        // use direct loads as pipelining is not necessary
        // may be worth unrolling this loop
        for (int i = SC(int, threadIdx.x); i < vP; i += threads) {
            vD[i] = vS[i];
        }
    }
    else {
        //assert(partition % (threads * Alignment * stageExtent) == 0);
        //assert(stages > pipeStages);
        const int stages = static_cast<int>(partition / (threads * Alignment * stageExtent));
        cuda::static_for<pipeStages>([&vW, &vS](auto i)
        {
            cuda::static_for<stageExtent>([&i, &vW, &vS](auto j)
            {
                const int slot = ((i * stageExtent + j) * threads) + threadIdx.x;
                // async gmem -> smem
                cp_async_global_to_shared<Alignment>(vW + slot, vS + slot);
            });
            cp_async_commit();
        });
        VT reginald[stageExtent];
        for (int i = pipeStages; i < stages; ++i) {
            cp_async_wait<pipeStages - 1>();
            const int stage_out = i - pipeStages;
            const int cs = stage_out % pipeStages;
            cuda::static_for<stageExtent>([&i, &cs, &vW, &reginald, &vS](auto j)
            {
                const int csW = (cs * stageExtent + j) * threads + threadIdx.x;
                const long int slot = (i * stageExtent + j) * threads + threadIdx.x;
                // smem -> rmem
                reginald[j] = vW[csW];
                // async gmem -> smem prefetch
                cp_async_global_to_shared<Alignment>(vW + csW, vS + slot);
            });
            cuda::static_for<stageExtent>([&stage_out, &reginald, &vD](auto j)
            {
                const long int slot = (stage_out * stageExtent + j) * threads + threadIdx.x;
                // rmem -> gmem
                vD[slot] = reginald[j];
            });
            // commit async transfers from this stage
            cp_async_commit();
        }
        // tail
        cuda::static_for<pipeStages>([&vW, &reginald, &vS, &vD, &stages](auto i)
        {
            const int stage = (stages - pipeStages) + i;
            const int cs = stage % pipeStages;
            cp_async_wait<pipeStages - 1 - i>();
            cuda::static_for<stageExtent>([&i, &cs, &vW, &reginald, &vS, &stages](auto j)
            {
                const int csW = (cs * stageExtent + j) * threads + threadIdx.x;
                // smem -> rmem
                reginald[j] = vW[csW];
            });
            cuda::static_for<stageExtent>([&stage, &reginald, &vD](auto j)
            {
                const long int slot = (stage * stageExtent + j) * threads + threadIdx.x;
                // rmem -> gmem
                vD[slot] = reginald[j];
            });
        });
    }
}

__device__ __forceinline__
void kxFlush() {
    __threadfence_system();
}

template<int threads, int pipeStages = 2, int stageExtent = 2, int Alignment = 16>
__global__ void bw_v3(cuda::std::byte* __restrict__ dst, const cuda::std::byte* __restrict__ src,
                                       const size_t __grid_constant__ partition, // bytes
                                       const int __grid_constant__ peer,
                                       const int __grid_constant__ iters,
                                       int* __restrict__ checkpoint) {
    //assert(partition % 16 == 0);
    kxAssume(partition % Alignment == 0);
    //assert(cuda::is_aligned(dst, 16));
    //assert(cuda::is_aligned(src, 16));
    __shared__ __align__(Alignment) cuda::std::byte workspace[threads * Alignment * pipeStages * stageExtent];
    auto* __restrict__ aD = static_cast<cuda::std::byte*>(__builtin_assume_aligned(dst + blockIdx.x * partition, Alignment));
    auto* __restrict__ aS = static_cast<const cuda::std::byte*>(__builtin_assume_aligned(src + blockIdx.x * partition, Alignment));
    auto gridSync = [&checkpoint](const int& epoch)
    {
        __threadfence();
        const auto expected = (epoch + 1) * gridDim.x;
        bool checkAgain = atomicAdd(checkpoint, 1) + 1 < expected;
        while (checkAgain) {
            checkAgain = atomicAdd(checkpoint, 0) < expected;
        }
    };
    // here we unroll and use LDGSTS
    // gmem -> smem
    for (int i = 0; i < iters; ++i) {
        kxPut<threads, pipeStages, stageExtent, Alignment>(aD, aS, workspace, partition);
        __syncthreads();
        gridSync(i);
    }
    __syncthreads();
    if (!threadIdx.x) {
       kxFlush(); // equivalent to __threadfence_system();
    }
}

__global__
void bw_v2(cuda::std::byte* __restrict__ dst, const cuda::std::byte* __restrict__ src,
                                       const size_t __grid_constant__ partition, // bytes
                                       const int __grid_constant__ peer,
                                       const int __grid_constant__ iters,
                                       int* __restrict__ checkpoint)
{
    const auto sOff = blockIdx.x * partition;
    auto gridSync = [&checkpoint](const int& epoch)
    {
        __threadfence();
        const auto expected = (epoch + 1) * gridDim.x;
        bool checkAgain = atomicAdd(checkpoint, 1) + 1 < expected;
        while (checkAgain) {
            checkAgain = atomicAdd(checkpoint, 0) < expected;
        }
    };
    for (int i = 0; i < iters; ++i) {
        nvshmemx_putmem_nbi_block(dst + sOff, src + sOff, partition, peer);
        if (!threadIdx.x) {
            gridSync(i);
        }
    }
    __syncthreads();
    if (!threadIdx.x) {
        nvshmem_quiet();
    }
}

// Parse sizes like 4096, 4K, 16M, 1G
auto parseSize(const std::string& s) {
    char unit = 0;
    double val = 0.0;
    if (sscanf(s.c_str(), "%lf%c", &val, &unit) >= 1) {
        size_t mult = 1;
        switch (unit) {
            case 'k': case 'K': mult = 1024ull; break;
            case 'm': case 'M': mult = 1024ull * 1024ull; break;
            case 'g': case 'G': mult = 1024ull * 1024ull * 1024ull; break;
            default: mult = 1; break;
        }
        if (unit == 0 || (unit != 'K' && unit != 'k' && unit != 'M' && unit != 'm' && unit != 'G' && unit != 'g')) {
            // no unit, already parsed in val
            return static_cast<size_t>(val);
        }
        return static_cast<size_t>(val * mult);
    }
    std::cerr << "Invalid size: " << s << std::endl;
    std::exit(EXIT_FAILURE);
}

struct Args {
    int threads = 256;
    int ctas = 32;
    size_t minBytes = 1 << 10;   // 1 KiB
    size_t maxBytes = 1 << 30;   // 1 GiB
    int iters = 32;
    int warmup = 32;
    int step = 2;
};

Args parseArgs(const int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto needVal = [&](const char* name){
            if (i+1 >= argc) { std::cerr << name << " requires a value\n"; std::exit(EXIT_FAILURE);} };
        if (key == "--threads") { needVal("--threads"); a.threads = std::stoi(argv[++i]); }
        else if (key == "--ctas") { needVal("--ctas"); a.ctas = std::stoi(argv[++i]); }
        else if (key == "--min") { needVal("--min"); a.minBytes = parseSize(argv[++i]); }
        else if (key == "--max") { needVal("--max"); a.maxBytes = parseSize(argv[++i]); }
        else if (key == "--iters") { needVal("--iters"); a.iters = std::stoi(argv[++i]); }
        else if (key == "--warmup") { needVal("--warmup"); a.warmup = std::stoi(argv[++i]); }
        else if (key == "--step") { needVal("--step"); a.step = std::max(1, std::stoi(argv[++i])); }
        else if (key == "-h" || key == "--help") {
            std::cout << "Usage: " << argv[0] <<
                " [--min BYTES] [--max BYTES] [--iters N] [--warmup N] [--threads N] [--ctas N] [--step N]\n";
            std::cout << "  Sizes accept suffixes K/M/G (base 1024).\n";
            std::exit(EXIT_SUCCESS);
        } else {
            std::cerr << "Unknown arg: " << key << " (use --help)\n";
            std::exit(EXIT_FAILURE);
        }
    }
    if (a.minBytes == 0 || a.maxBytes == 0 || a.minBytes > a.maxBytes) {
        std::cerr << "Invalid min/max bytes" << std::endl; std::exit(EXIT_FAILURE);
    }
    return a;
}

void kx_test() {
    CUDA_CHECK(cudaSetDevice(0));
    using Element = cuda::std::byte;
    Element* src = nullptr;
    Element* dst = nullptr;
    int* checkPoint = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    constexpr auto size = 32 * 1024;
    CUDA_CHECK(cudaMallocAsync(&src, size, stream));
    CUDA_CHECK(cudaMallocAsync(&dst, size, stream));
    CUDA_CHECK(cudaMallocAsync(&checkPoint, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(checkPoint, 0, sizeof(int), stream));
    auto* h = static_cast<Element*>(std::malloc(size));
    auto* fh = reinterpret_cast<int*>(h);
    constexpr auto v = -98;
    constexpr int elements = size / sizeof(int);
    std::ranges::fill(fh, fh + elements, v);
    CUDA_CHECK(cudaMemcpyAsync(src, fh, size, cudaMemcpyHostToDevice, stream));
    constexpr auto threads = 256;
    constexpr auto Alignment = 16;
    constexpr auto stages = 2;
    constexpr auto stageExtent = 2;
    constexpr auto blocks = 2;
    static_assert(blocks <= (size / (threads * Alignment * stageExtent * stages)));
    constexpr auto partition = size / blocks;
    bw_v3<threads, stages, stageExtent, Alignment><<<blocks, threads, 0, stream>>>
    (dst, src, partition, 1, 1, checkPoint);
    CUDA_CHECK(cudaMemcpyAsync(fh, dst, size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    bool correct = true;
    for (int i = 0; i < elements; ++i) {
        if (fh[i] != v) {
            correct = false;
            break;
        }
    }
    printf("kx passed? %s\n", correct ? "yes" : "no");
    CUDA_CHECK(cudaFreeAsync(src, stream));
    CUDA_CHECK(cudaFreeAsync(dst, stream));
    CUDA_CHECK(cudaFreeAsync(checkPoint, stream));
    std::free(h);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

#define MAX_ALIGNMENT 16
void bench(int argc, char** argv) {
    const auto args = parseArgs(argc, argv);

    // Choose CUDA device for this PE via init_attr (recommended).
    nvshmem_init();  // minimal init

    const int mype = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();
    if (npes < 2) {
        if (mype == 0) fprintf(stderr, "Need >= 2 PEs\n");
        nvshmem_finalize(); return;
    }

    const int dev = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp deviceProp{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    // Symmetric buffers (max size)
    using Element = cuda::std::byte;
    auto* src = static_cast<Element*>(nvshmem_malloc(args.maxBytes));
    auto* dst = static_cast<Element*>(nvshmem_malloc(args.maxBytes));
    if (!src || !dst) {
        fprintf(stderr, "PE %d: nvshmem_malloc failed\n", mype);
        nvshmem_finalize(); return;
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemsetAsync(src, 0xAB, args.maxBytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dst, 0x00, args.maxBytes, stream));

    const int peer = (mype + 1) % npes;

    if (mype == 0) {
        printf("# device-side NVSHMEM put benchmark (block API)\n");
        printf("# npes=%d, device=%s, iters=%d\n", npes, deviceProp.name, args.iters);
        printf("bytes,iters,total_us,avg_us,GBps\n");
    }

    // Device buffers for timing
    int* checkpoint = nullptr;
    CUDA_CHECK(cudaMallocAsync(&checkpoint, sizeof(int), stream));
    // Benchmark over sizes
    float milliseconds;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    if (!mype) {
        for (auto nBytes = args.minBytes; nBytes <= args.maxBytes; nBytes *= args.step) {
            // Sync all PEs and streams before measuring this size
            nvshmemx_barrier_all_on_stream(stream);
            CUDA_CHECK(cudaMemsetAsync(checkpoint, 0, sizeof(int), stream));

            // Warmup a few iterations (device-side)
            const long int partition = static_cast<long int>(nBytes / sizeof(Element)) / args.ctas;
            bw_v2<<<args.ctas,args.threads, 0, stream>>>(dst, src, partition, peer, args.warmup, checkpoint);
            CUDA_CHECK(cudaMemsetAsync(checkpoint, 0, sizeof(int), stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Timed loop
            CUDA_CHECK(cudaEventRecord(start));
            bw_v2<<<args.ctas, args.threads, 0, stream>>>(dst, src, partition, peer, args.iters, checkpoint);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

            //const auto total_us = to_us_from_cycles(cycles, ghz);
            const auto avg_us = (static_cast<double>(milliseconds) / args.iters) * 1000.0;
            const auto bytes_GB = static_cast<double>(nBytes) / B_TO_GB;
            const auto GBps= bytes_GB / (avg_us / 1e6);

            if (mype == 0) {
                printf("%zu,%d,%.2f,%.2f\n",
                       nBytes, args.iters, avg_us, GBps);
                fflush(stdout);
            }
        }
    }
    else {
        for (auto nBytes = args.minBytes; nBytes <= args.maxBytes; nBytes *= args.step) {
            nvshmemx_barrier_all_on_stream(stream);
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvshmem_free(src);
    nvshmem_free(dst);
    CUDA_CHECK(cudaStreamDestroy(stream));
    nvshmem_finalize();
}

template<int threads = 256, int stages = 2, int stageExtent = 2, int Alignment = 16>
void kxBench(int argc, char** argv) {
    const auto args = parseArgs(argc, argv);

    // Choose CUDA device for this PE via init_attr (recommended).
    nvshmem_init();  // minimal init

    const int mype = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();
    if (npes < 2) {
        if (mype == 0) fprintf(stderr, "Need >= 2 PEs\n");
        nvshmem_finalize(); return;
    }

    const int dev = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp deviceProp{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    // Symmetric buffers (max size)
    using Element = cuda::std::byte;
    auto* src = static_cast<Element*>(nvshmem_malloc(args.maxBytes));
    auto* dst = static_cast<Element*>(nvshmem_malloc(args.maxBytes));
    if (!src || !dst) {
        fprintf(stderr, "PE %d: nvshmem_malloc failed\n", mype);
        nvshmem_finalize(); return;
    }
    if (args.minBytes % MAX_ALIGNMENT != 0) {
        if (!mype) {
            fprintf(stderr, "min Bytes should be a multiple of 16\n");
        }
        nvshmem_finalize(); return;
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemsetAsync(src, 0xAB, args.maxBytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dst, 0x00, args.maxBytes, stream));

    const int peer = (mype + 1) % npes;

    if (mype == 0) {
        printf("# device-side NVSHMEM put benchmark (block API)\n");
        printf("# npes=%d, device=%s, iters=%d\n", npes, deviceProp.name, args.iters);
        printf("bytes,iters,total_us,avg_us,GBps\n");
    }

    // Device buffers for timing
    int* checkpoint = nullptr;
    CUDA_CHECK(cudaMallocAsync(&checkpoint, sizeof(int), stream));
    // Benchmark over sizes
    float milliseconds;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    if (!mype) {
        for (auto nBytes = args.minBytes; nBytes <= args.maxBytes; nBytes *= args.step) {
            // Sync all PEs and streams before measuring this size
            nvshmemx_barrier_all_on_stream(stream);
            CUDA_CHECK(cudaMemsetAsync(checkpoint, 0, sizeof(int), stream));
            const int ec = static_cast<int>(args.minBytes / MAX_ALIGNMENT);
            const auto blocks = min(ec, args.ctas);

            // Warmup a few iterations (device-side)
            const long int partition = static_cast<long int>(nBytes / sizeof(Element)) / blocks;
            bw_v3<threads, stages, stageExtent, Alignment><<<blocks, threads, 0, stream>>>
            (dst, src, partition, 1, args.iters, checkpoint);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            CUDA_CHECK(cudaMemsetAsync(checkpoint, 0, sizeof(int), stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            // Timed loop
            CUDA_CHECK(cudaEventRecord(start));
            bw_v3<threads, stages, stageExtent, Alignment><<<blocks, threads, 0, stream>>>
            (dst, src, partition, 1, args.iters, checkpoint);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

            //const auto total_us = to_us_from_cycles(cycles, ghz);
            const auto avg_us = (static_cast<double>(milliseconds) / args.iters) * 1000.0;
            const auto bytes_GB = static_cast<double>(nBytes) / B_TO_GB;
            const auto GBps= bytes_GB / (avg_us / 1e6);

            if (mype == 0) {
                printf("%zu,%d,%.2f,%.2f\n",
                       nBytes, args.iters, avg_us, GBps);
                fflush(stdout);
            }
        }
    }
    else {
        for (auto nBytes = args.minBytes; nBytes <= args.maxBytes; nBytes *= args.step) {
            nvshmemx_barrier_all_on_stream(stream);
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvshmem_free(src);
    nvshmem_free(dst);
    CUDA_CHECK(cudaStreamDestroy(stream));
    nvshmem_finalize();
}
int main(int argc, char** argv) {
    kxBench(argc, argv);
    return 0;
}
// nvshmem_put_device_min.cu
// Minimal device-side NVSHMEM put benchmark (block collective API).
// Used for investigating emitted instructions
// Measures per-iteration latency of: put_nbi_block + quiet
// Prints: bytes,iters,total_us,avg_us,GBps,cycles_per_iter
//

#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} } while(0)

using ll_t = long long int;
using ull_t = unsigned long long int;
__global__ void bw(void* __restrict__ dst, const void* __restrict__ src,
                                       const size_t __grid_constant__ nbytes,
                                       const int __grid_constant__ peer,
                                       const int __grid_constant__ iters,
                                       uint64_t* __restrict__ cycles_out)
{
    uint64_t start = 0, stop = 0;
    if (threadIdx.x == 0) {
        start = clock64();
    }

    for (int i = 0; i < iters; ++i) {
        // Non-blocking put (block collective); all threads participate
        nvshmemx_putmem_nbi_block(dst, src, nbytes, peer);

        // Ensure completion of outstanding puts before next iter
        nvshmem_quiet();  // device-side quiet
        __syncthreads();   // serialize iterations for clean timing
    }

    if (threadIdx.x == 0) {
        stop = clock64();
        *cycles_out = (stop - start);
    }
}

__device__ inline auto read_globaltimer() {
    #if __CUDA_ARCH__ >= 700
    ll_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
    #else
    return clock64();  // fallback if you don’t care about cross-SM epoch
    #endif
}

__global__
void bw_v2(double* __restrict__ dst, const double* __restrict__ src,
                                       const size_t __grid_constant__ partition,
                                       const int __grid_constant__ peer,
                                       const int __grid_constant__ iters,
                                       int* __restrict__ checkpoint)
{
    const auto sOff = blockIdx.x * partition;
    for (int i = 0; i < iters; i++) {
        nvshmemx_double_put_nbi_block(dst + sOff, src + sOff, partition, peer);
    }
    __syncthreads();
    if (!threadIdx.x) {
        /*__threadfence();
        auto allDone = atomicAdd(checkpoint, 1) + 1 == nB;
        while (!allDone) {
            allDone = atomicAdd(checkpoint, 0) == nB;
        }*/
        // wait until everyone is done
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
    size_t minBytes = 1 << 29;   // 1 KiB
    size_t maxBytes = 1 << 30;   // 1 GiB
    int iters = 10;
    int warmup = 5;
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

int main(int argc, char** argv) {
    const auto args = parseArgs(argc, argv);

    // Choose CUDA device for this PE via init_attr (recommended).
    nvshmem_init();  // minimal init

    const int mype = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();
    if (npes < 2) {
        if (mype == 0) fprintf(stderr, "Need >= 2 PEs\n");
        nvshmem_finalize(); return 1;
    }

    const int dev = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp deviceProp{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    // Symmetric buffers (max size)
    void* src_dst = nvshmem_malloc(args.maxBytes * 2);
    using Element = double;
    auto* src = static_cast<Element*>(src_dst);
    auto* dst = src + (args.maxBytes / sizeof(Element));
    if (!src || !dst) {
        fprintf(stderr, "PE %d: nvshmem_malloc failed\n", mype);
        nvshmem_finalize(); return 1;
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
            CUDA_CHECK(cudaStreamSynchronize(stream));

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
            const auto bytes_GB = static_cast<double>(nBytes) / (1024 * 1024 * 1024);
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
    nvshmem_free(src_dst);
    CUDA_CHECK(cudaStreamDestroy(stream));
    nvshmem_finalize();
    return 0;
}
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

template<bool iterSync = true, bool nbi = true>
__global__
void bw_v2(cuda::std::byte* __restrict__ dst, const cuda::std::byte* __restrict__ src,
                                       const size_t __grid_constant__ nBytes,
                                       const int __grid_constant__ peer,
                                       const int __grid_constant__ iters,
                                       uint64_t* __restrict__ cycles_out,
                                       unsigned long long int* __restrict__ checkpoint)
{
    uint64_t start = 0;
    const int tid = static_cast<int>(threadIdx.x);
    const int bid = static_cast<int>(blockIdx.x);
    const int nB = static_cast<int>(gridDim.x);
    const auto partition = nBytes / nB;
    const int residue = static_cast<int>(nBytes % nB);
    const auto slice = partition + (bid < residue);
    const auto sOff =  partition + min(bid, residue);
    if (!tid) {
        start = clock64();
    }

    for (int i = 0; i < iters; i++) {
        if constexpr (nbi) {
            nvshmemx_putmem_nbi_block(dst + sOff, src + sOff, slice, peer);
        }
        else {
            nvshmemx_putmem_block(dst + sOff, src + sOff, slice, peer);
        }
        // synchronizing across blocks
        __syncthreads();
        if constexpr (iterSync) {
            if (!tid) {
                __threadfence();
                const auto expected = (i + 1) * nB;
                bool checkAgain = (atomicAdd(checkpoint, 1) + 1) < expected;
                while (checkAgain) {
                    checkAgain = atomicAdd(checkpoint, 0) < expected;
                }
            }
            __syncthreads();
        }
    }

    if (!tid) {
        __threadfence();
        nvshmem_quiet();
        const auto stop = clock64();
        if (atomicAdd(checkpoint, 1) + 1 == (iters + 1) * nB) {
            *cycles_out = stop - start;
        }
    }
}

static double to_us_from_cycles(const uint64_t& cycles, const double& ghz) {
    // cycles / (GHz * 1e9 cycles/s) => seconds -> microseconds
    return static_cast<double>(cycles) / (ghz * 1e9) * 1e6;
}

// Parse sizes like 4096, 4K, 16M, 1G
size_t parseSize(const std::string& s) {
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
    int threads = 128;
    int ctas = 32;
    size_t minBytes = 1 << 29;   // 1 KiB
    size_t maxBytes = 1 << 30;   // 1 GiB
    int iters = 100;
    int warmup = 10;
    int step = 2;
};

Args parseArgs(const int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto needVal = [&](const char* name){
            if (i+1 >= argc) { std::cerr << name << " requires a value\n"; std::exit(EXIT_FAILURE);} };
        if (key == "--threads") { needVal("--src"); a.threads = std::stoi(argv[++i]); }
        else if (key == "--ctas") { needVal("--dst"); a.ctas = std::stoi(argv[++i]); }
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
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count == 0) { fprintf(stderr, "No CUDA devices.\n"); return 1; }
    nvshmem_init();  // minimal init

    const int mype = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();
    if (npes < 2) {
        if (mype == 0) fprintf(stderr, "Need >= 2 PEs\n");
        nvshmem_finalize(); return 1;
    }

    const int dev = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    const double ghz = prop.clockRate / 1e6; // kHz -> GHz

    // Symmetric buffers (max size)
    void* src_dst = nvshmem_malloc(args.maxBytes * 2);
    auto* src = static_cast<cuda::std::byte*>(src_dst);
    auto* dst = static_cast<cuda::std::byte*>(src_dst) + args.maxBytes;
    if (!src || !dst) {
        fprintf(stderr, "PE %d: nvshmem_malloc failed\n", mype);
        nvshmem_finalize(); return 1;
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemsetAsync(src, 0xAB, args.maxBytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dst, 0x00, args.maxBytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int peer = (mype + 1) % npes;

    if (mype == 0) {
        printf("# device-side NVSHMEM put benchmark (block API)\n");
        printf("# npes=%d, device=%s, iters=%d\n", npes, prop.name, args.iters);
        printf("bytes,iters,total_us,avg_us,GBps,cycles_per_iter\n");
    }

    // Device buffers for timing
    uint64_t* d_cycles = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_cycles, 2 * sizeof(uint64_t), stream));
    static_assert(sizeof(unsigned long long int) ==
        sizeof(uint64_t) && alignof(unsigned long long int) == alignof(uint64_t));
    auto* checkpoint = reinterpret_cast<unsigned long long int *>(d_cycles) + 1;
    // Benchmark over sizes
    for (auto nBytes = args.minBytes; nBytes <= args.maxBytes; nBytes *= args.step) {
        // Sync all PEs and streams before measuring this size
        nvshmemx_barrier_all_on_stream(stream);

        // Warmup a few iterations (device-side)
        bw_v2<<<args.ctas,args.threads, 0, stream>>>(dst, src, nBytes, peer, args.warmup, d_cycles, checkpoint);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        nvshmemx_barrier_all_on_stream(stream);

        // Timed loop
        bw_v2<<<args.ctas, args.threads, 0, stream>>>(dst, src, nBytes, peer, args.iters, d_cycles, checkpoint);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        nvshmemx_barrier_all_on_stream(stream);

        // Fetch cycles
        uint64_t cycles = 0;
        CUDA_CHECK(cudaMemcpyAsync(&cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        const auto total_us = to_us_from_cycles(cycles, ghz);
        const auto avg_us   = total_us / args.iters;
        const auto GBps     = (static_cast<double>(nBytes) * args.iters) / (total_us * 1e-6) / 1e9;
        const auto cyc_per_iter = static_cast<double>(cycles) / static_cast<double>(args.iters);

        if (mype == 0) {
            printf("%zu,%d,%.3f,%.6f,%.3f,%.1f\n",
                   nBytes, args.iters, total_us, avg_us, GBps, cyc_per_iter);
            fflush(stdout);
        }
    }

    CUDA_CHECK(cudaFreeAsync(d_cycles, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvshmem_free(src_dst);
    CUDA_CHECK(cudaStreamDestroy(stream));
    nvshmem_finalize();
    return 0;
}
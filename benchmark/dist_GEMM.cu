#include <array>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nccl.h>

// mathdx
#include <cublasdx.hpp>
#include <curanddx.hpp>

#define MAX_COPY_ENGINE 8
#define CHECK_CUDA(x) do{cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} }while(0)

// --------------------
// Cycle-burner stubs
// --------------------
// These emulate time for GEMMs and Collectives. Swap them with real calls later.

__global__ void burn_kernel(int iters) {
    // Simple integer/XOR mix to avoid compiler folding
    unsigned int s = (threadIdx.x + 1u) * (blockIdx.x + 3u);
#pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        s ^= (s << 13);
        s ^= (s >> 7);
        s ^= (s << 17);
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) { asm volatile(""); }
}

// Simulate compute proportional to MxNxK tiles by scaling thread blocks with size hints.
void stub_gemm(cudaStream_t stream, size_t M, size_t N, size_t K, int iters_scale) {
    // Heuristic grid/block based on output tile count
    size_t tiles = (M * N + 256 - 1) / 256; // 256 outputs per block (arbitrary)
    int blocks = (int) std::min<size_t>(std::max<size_t>(tiles, 1), 65535);
    burn_kernel<<<blocks, 256, 0, stream>>>(iters_scale);
}

// --------------------
// Args & parsing
// --------------------
struct Args {
    int M_min = 1024;
    int M_max = 1024;
    int N = 4096; // hidden dim (cols)
    int K = 4096;
    int warmup_iters = 32;
    int iters = 32;
    int step_factor = 2;
};

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [--M] [--N] [--K] [--iters] [--warmup_iters]\n", prog);
}

static auto ctoi(const char *p) {
    return static_cast<int>(strtol(p, nullptr, 10));
}

static Args parse_args(const int argc, char **argv) {
    Args a;
    auto at = [&](const char *k, const int i) { return strncmp(argv[i], k, strlen(k)) == 0; };
    auto val = [&](const char *k, const int i) { return argv[i] + strlen(k); };
    for (int i = 1; i < argc; i++) {
        if (at("--M-min=", i)) a.M_min = ctoi(val("--M-min=", i));
        if (at("--M-max=", i)) a.M_max = ctoi(val("--M-max=", i));
        else if (at("--N=", i)) a.N = ctoi(val("--N=", i));
        else if (at("--K=", i)) a.K = ctoi(val("--K=", i));
        else if (at("--iters=", i)) a.iters = ctoi(val("--iters=", i));
        else if (at("--warmup_iters=", i)) a.warmup_iters = ctoi(val("--warmup_iters=", i));
        else if (at("--step=", i)) a.step_factor = ctoi(val("--step=", i));
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            usage(argv[0]);
            exit(1);
        }
    }
    if (a.M_min < 1 || a.M_max < 1 || a.N < 1 || a.K < 1) {
        fprintf(stderr, "Bad numeric arg.\n");
        exit(1);
    }
    return a;
}

// --------------------
// Helpers
// --------------------
static auto elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

using Element = __half;
__host__
void dist_gemm(int mode, Element *dA, Element *dB, Element *dC,
               const int M, const int N, const int K, const int rank, const int world) {
    constexpr auto alpha = 1.0f;
    constexpr auto beta = 0.0f;
}

constexpr unsigned int PHILOX_SUBSEQS = 65536; // used to map (subsequence, offset) cleanly

// Choose a Philox RNG configured for thread-level execution.
// Tip: 7 rounds is a good perf/quality point per cuRANDDx docs.
template<unsigned int Arch>
using PhiloxRNG = decltype(
    curanddx::Generator<curanddx::philox4_32>() +
    curanddx::PhiloxRounds<7>() + // faster than 10, still Crush-resistant
    curanddx::SM<Arch>() +
    curanddx::Thread()
);

struct __align__(8) half4 {
    __half2 x;
    __half2 y;
};

// Kernel: fills out[i] with uniform floats in [0,1)
template<class RNG>
__global__ void fill_uniform_kernel(Element *__restrict__ out,
                                    const size_t n,
                                    const unsigned long long seed,
                                    const unsigned long long base_offset) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t base = tid * 4; // we write up to 4 floats per thread
    if (base >= n) return;

    // Distribution and RNG instance
    curanddx::uniform<float> dist; // default [0,1)
    RNG rng(seed,
            ((base_offset + tid) % PHILOX_SUBSEQS), // subsequence
            ((base_offset + tid) / PHILOX_SUBSEQS)); // offset

    // Generate 4 at once, then store only what fits
    const float4 r4 = dist.generate4(rng);
    const half4 v{__float22half2_rn(float2{r4.x, r4.y}), __float22half2_rn(float2{r4.z, r4.w})};
    // Tail-safe stores
    if (n - base >= 4) {
        reinterpret_cast<half4 *>(out)[tid] = v;
    } else {
        if (base + 0 < n) out[base + 0] = v.x.x;
        if (base + 1 < n) out[base + 1] = v.x.y;
        if (base + 2 < n) out[base + 2] = v.y.x;
        if (base + 3 < n) out[base + 3] = v.y.y;
    }
}

// Convenience host launcher
void fill_uniform(Element *__restrict__ d_out, const size_t n, cudaStream_t stream,
                  const unsigned long long seed = 1234ULL,
                  const unsigned long long offset = 0ULL) {
    constexpr auto threads = 256;
    using RNG = PhiloxRNG<ARCH>;
    const size_t threads_needed = (n + 3) / 4; // 4 outputs per thread
    const int grid = static_cast<int>((threads_needed + threads - 1) / threads);
    fill_uniform_kernel<RNG><<<grid, threads, 0, stream>>>(d_out, n, seed, offset);
}

void run_dist_gemm(const Args &a) {
    // CSV header
    // initialize NVSHMEM backend
    nvshmem_init();
    const int rank = nvshmem_my_pe();
    const int world = nvshmem_n_pes();
    CHECK_CUDA(cudaSetDevice(rank));
    cudaStream_t computeStream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    std::array<cudaStream_t, MAX_COPY_ENGINE> copyStreams{};
    for (auto &copyStream: copyStreams) {
        cudaStream_t s;
        CHECK_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        copyStream = s;
    }
    // allocate A, B and C buffers
    int nCopyEngines = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&nCopyEngines, cudaDevAttrAsyncEngineCount, rank));
    Element *dA = nullptr;
    constexpr auto aSeed = 41;
    const auto mSlice = a.M_max / world;
    CHECK_CUDA(cudaMallocAsync(&dA, mSlice * a.K * sizeof(Element), computeStream));
    fill_uniform(dA, mSlice * a.K, computeStream, aSeed);
    Element *dB = nullptr;
    constexpr auto bSeed = 42;
    CHECK_CUDA(cudaMallocAsync(&dB, a.N * a.K * sizeof(Element), computeStream));
    fill_uniform(dB, a.N * a.K, computeStream, bSeed);
    auto *__restrict__ dC = static_cast<Element *>(nvshmem_malloc(a.M_max * a.N * sizeof(Element)));

    if (a.M_min % world != 0) {
        if (rank == 0) {
            fprintf(stderr, "Incorrect M-min: %d\n", a.M_min);
        }
        return;
    }
    cudaStream_t s_comp, s_comm;
    CHECK_CUDA(cudaStreamCreateWithFlags(&s_comp, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&s_comm, cudaStreamNonBlocking));

    // Events
    cudaEvent_t t_req_start, t_req_end;
    CHECK_CUDA(cudaEventCreate(&t_req_start));
    CHECK_CUDA(cudaEventCreate(&t_req_end));

    for (int m = a.M_min; m <= a.M_max; m *= a.step_factor) {
        constexpr std::array cases{"NCCL-overlap", "CE-overlap", "NVSH-Host-overalap", "NVSH-Fused-overlap"};
#pragma unroll
        for (int c = 0; c < cases.size(); ++c) {
            if (rank == 0) {
                printf("========%s========\nworld,M,N,K,E2E_ms\n", cases[c]);
            }
            for (int i = 0; i < a.warmup_iters; ++i) {
                dist_gemm(c, dA, dB, dC, m, a.N, a.K, rank, world);
            }
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaEventRecord(t_req_start));
            for (int j = 0; j < a.iters; ++j) {
                dist_gemm(c, dA, dB, dC, m, a.N, a.K, rank, world);
            }
            CHECK_CUDA(cudaEventRecord(t_req_end));
            CHECK_CUDA(cudaEventSynchronize(t_req_end));
            const auto e2e_ms = elapsed_ms(t_req_start, t_req_end);
            if (rank == 0) {
                printf("%d,%d,%d,%d,%.4f\n", world, m, a.N, a.K, e2e_ms);
                fflush(stdout);
            }
        }
    }
    CHECK_CUDA(cudaEventDestroy(t_req_start));
    CHECK_CUDA(cudaEventDestroy(t_req_end));
    nvshmem_finalize();
}

// --------------------
// Main
// --------------------
int main(const int argc, char **argv) {
    run_dist_gemm(parse_args(argc, argv));
    return 0;
}

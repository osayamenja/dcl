#include <array>
#include <barrier>
#include <thread>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nccl.h>
#include <cublasLt.h>
#include <nvtx3/nvtx3.hpp>

// mathdx
#include <cublasdx.hpp>
#include <curanddx.hpp>

#define MAX_COPY_ENGINE 8
#define CHECK_CUDA(x) do{cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} \
}while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
fprintf(stderr,"cuBLASLt error %s:%d: status=%d\n", __FILE__, __LINE__, int(s)); std::abort(); } } while(0)
#define NCCL_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        ncclResult_t status = call;                                                                                    \
        if (status != ncclSuccess)                                                                                     \
        {                                                                                                              \
            fprintf(stderr, "NCCL error at %s:%d : %d\n", __FILE__, __LINE__, status);                                 \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)
#define MPI_CHECK(call)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        int status = call;                                                                                             \
        if (status != MPI_SUCCESS)                                                                                     \
        {                                                                                                              \
            fprintf(stderr, "MPI error at %s:%d : %d\n", __FILE__, __LINE__, status);                                  \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)
// --------------------
// Args & parsing
// --------------------
struct Args {
    int check = 1;
    int ce = 0;
    int world = 1;
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
        else if (at("--M-max=", i)) a.M_max = ctoi(val("--M-max=", i));
        else if (at("--N=", i)) a.N = ctoi(val("--N=", i));
        else if (at("--K=", i)) a.K = ctoi(val("--K=", i));
        else if (at("--iters=", i)) a.iters = ctoi(val("--iters=", i));
        else if (at("--warmup_iters=", i)) a.warmup_iters = ctoi(val("--warmup_iters=", i));
        else if (at("--step=", i)) a.step_factor = ctoi(val("--step=", i));
        else if (at("--world=", i)) a.world = ctoi(val("--world=", i));
        else if (at("--ce=", i)) a.ce = ctoi(val("--ce=", i));
        else if (at("--check=", i)) a.ce = ctoi(val("--check=", i));
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
constexpr auto ncclT = ncclFloat16;
enum class DG_OVERLAP_MODE {
    NCCL,
    CE,
    NVSH_HOST,
    NVSH_FUSED
};


/**
 * Compute: C = A * B^T
 * A: (M,K) row-major, B: (N,K) row-major, C: (M,N) row-major
 * Types: FP16 inputs/outputs, FP32 compute/scale
 * Workspace: 32 MB
 */

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
    const half4 v{__float22half2_rn(float2{r4.x, r4.y}),
        __float22half2_rn(float2{r4.z, r4.w})};
    // Tail-safe stores
    if (n - base >= 4) {
        reinterpret_cast<half4*>(out)[tid] = v;
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

template<typename T>
__device__ __forceinline__ float to_float(const T& x) { return static_cast<float>(x); }
template<>
__device__ __forceinline__ float to_float<float>(const float& x) { return x; }
template<>
__device__ __forceinline__ float to_float<__half>(const __half& x) { return __half2float(x); }

// --- Kernel: block-level reduction → single atomic per block ---
template<int threads, typename T>
__global__ void checkCorrectness(const T *__restrict__ pred, const T *__restrict__ ref,
    const size_t n, unsigned long long* __restrict__ mismatches, const float atol = 1e-3f, const float rtol = 1e-3f) {
    // Per-thread local accumulation
    unsigned long long local = 0ULL;
    const size_t stride = static_cast<size_t>(threads) * gridDim.x;
    for (size_t i = blockIdx.x * static_cast<size_t>(threads) + threadIdx.x; i < n; i += stride) {
        const float a = to_float(pred[i]);
        const float b = to_float(ref[i]);
        bool unequal = false;
        if (isnan(a) || isnan(b)) {
            // Treat NaN==NaN as equal; flip if you prefer mismatch on NaNs.
            unequal = !(isnan(a) && isnan(b));
        } else if (isinf(a) || isinf(b)) {
            unequal = !(isinf(a) && isinf(b) && (a == b));
        } else {
            const float tol = atol + rtol * fmaxf(fabsf(a), fabsf(b));
            unequal = fabsf(a - b) > tol;
        }
        if (unequal) ++local;
    }
    // Reduce per-block in shared memory
    __shared__ unsigned long long sdata[threads];
    sdata[threadIdx.x] = local;
    __syncthreads();
    // Tree reduction
    #pragma unroll
    for (int offset = threads >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }
        __syncthreads();
    }

    // One atomic per block
    if (!threadIdx.x) {
        atomicAdd(mismatches, sdata[0]);
    }
}

// --- Host helper: returns error percentage (0..100) ---
template<int threads = 128, typename T>
__host__ __forceinline__
auto checkCorrectnessHost(const T *d_pred,
                                                  const T *d_ref,
                                                  const size_t n,
                                                  unsigned long long *d_mis,
                                                  cudaStream_t stream,
                                                  const float atol = 1e-3f,
                                                  const float rtol = 1e-3f) {
    NVTX3_FUNC_RANGE();
    if (n == 0) return 0.0f;
    const auto blocks = static_cast<int>((n + threads - 1) / threads);
    CHECK_CUDA(cudaMemsetAsync(d_mis, 0, sizeof(unsigned long long), stream));
    checkCorrectness<threads><<<blocks, threads, 0, stream>>>(d_pred, d_ref, n, d_mis, atol, rtol);
    unsigned long long h_mis = 0;
    CHECK_CUDA(cudaMemcpyAsync(&h_mis, d_mis, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return 100.0f * (static_cast<float>(h_mis) / static_cast<float>(n));
}

#define WORKSPACE_BYTES (32 * 1024 * 1024)
__host__ __forceinline__
void gemm_fp16_rowmajor_cublaslt(
    cublasLtHandle_t lt, cudaStream_t stream,
    const int M, const int N, const int K,
    const __half *A, // (M,K), row-major
    const __half *B, // (N,K), row-major
    __half *C,       // (M,N), row-major
    void* workspace) // must be at least WORKSPACE_BYTES
{
    NVTX3_FUNC_RANGE();
    // Row-major leading dimensions for underlying buffers
    const int ldA_rm = K; // A: (M x K)
    const int ldB_rm = K; // B: (N x K)
    const int ldC_rm = N; // C: (M x N)

    // cuBLASLt uses COLUMN-MAJOR descriptors. We reinterpret:
    //
    // A_rm(M,K) -> A_cm(K,M), ld = K
    // B_rm(N,K) -> B_cm(K,N), ld = K
    // C_rm(M,N) -> C_cm(N,M), ld = N
    //
    // We want: C_rm = A_rm * B_rm^T
    // In col-major:
    //   D_cm(N,M) = B_cm(K,N)^T * A_cm(K,M)
    // i.e., opA = T on B_cm, opB = N on A_cm.

    // Descriptors
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatrixLayout_t d_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    constexpr cublasComputeType_t compute     = CUBLAS_COMPUTE_32F;
    constexpr cudaDataType_t      dtype       = CUDA_R_16F;  // FP16 inputs/outputs
    constexpr cudaDataType_t      scale_dtype = CUDA_R_32F;  // FP32 scaling

    constexpr float alpha = 1.0f;
    constexpr float beta  = 0.0f;

    // 1) Matmul desc
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&op_desc, compute, scale_dtype));

    // opA = T (B_cm^T gives N x K)
    // opB = N (A_cm gives K x M)
    const cublasOperation_t opA = CUBLAS_OP_T;
    const cublasOperation_t opB = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    // 2) Matrix layouts (COLUMN-MAJOR views)

    // A operand (from B buffer):
    // B_rm(N,K) -> B_cm(K,N), lda = K
    // opA = T means A_op has shape (N x K)
    const int a_rows = K;     // stored rows in column-major
    const int a_cols = N;
    const int lda_cm = ldB_rm;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &a_desc, dtype, a_rows, a_cols, lda_cm));

    // B operand (from A buffer):
    // A_rm(M,K) -> A_cm(K,M), ldb = K
    // opB = N means B_op has shape (K x M)
    const int b_rows = K;
    const int b_cols = M;
    const int ldb_cm = ldA_rm;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &b_desc, dtype, b_rows, b_cols, ldb_cm));

    // C/D (output):
    // C_rm(M,N) -> C_cm(N,M), ldc = ldd = N
    const int cd_rows = N;
    const int cd_cols = M;
    const int ldc_cm  = ldC_rm;
    const int ldd_cm  = ldC_rm;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &c_desc, dtype, cd_rows, cd_cols, ldc_cm));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &d_desc, dtype, cd_rows, cd_cols, ldd_cm));

    // 3) Preference / heuristic
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    const size_t w = WORKSPACE_BYTES;
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &w,
        sizeof(w)));

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        lt, op_desc, a_desc, b_desc, c_desc, d_desc,
        pref, 1, &heuristic, &returned));
    if (returned == 0) {
        fprintf(stderr, "cuBLASLt: no heuristic found.\n");
        std::abort();
    }

    // 4) Launch
    CUBLAS_CHECK(cublasLtMatmul(
        lt, op_desc,
        &alpha,
        /*A=*/B, a_desc,   // B buffer as A operand
        /*B=*/A, b_desc,   // A buffer as B operand
        &beta,
        /*C=*/C, c_desc,   // C as input
        /*D=*/C, d_desc,   // and as output
        &heuristic.algo,
        workspace, WORKSPACE_BYTES,
        stream));

    // 5) Cleanup
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(d_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(op_desc));
}

__global__ void put(cuda::std::byte* dest, const cuda::std::byte* src, const int pe,
    const long int partition) {
    // assert (size % gridDim.x == 0)
    const auto sOff = blockIdx.x * partition;
    nvshmemx_putmem_nbi_block(dest + sOff, src + sOff, partition, pe);
}
__host__ __forceinline__
void printMatrix(const __half* __restrict__ const& p, const int& M, const int& N) {
    for (int i = 0; i < M; ++i) {
        printf("{");
        for (int j = 0; j < N; ++j) {
            printf("%f, ", __half2float(p[i * N + j]));
        }
        printf("}\n");
    }
}
#define COMM_BLOCKS 32
__host__ __forceinline__
void dist_gemm(const DG_OVERLAP_MODE mode, const Element *dA, const Element *dB, Element *dC,
               const int gM, const int M, const int N, const int K, const int mChunk,
               const int rank, const int world,
               cudaStream_t computeStream,
               const std::vector<cudaStream_t>& copyStreams, cudaEvent_t gemmDone,
               cublasLtHandle_t lt, void* workspace, ncclComm_t comm, Element* const* dCPeer = nullptr) {
    NVTX3_FUNC_RANGE();
    const int chunks = M / mChunk;
    auto* dCL = dC + rank * (M * N);
    switch (mode) {
        case DG_OVERLAP_MODE::NCCL: {
            nvtx3::scoped_range ncclRange{"NCCL"};
            // do gemm chunks
            gemm_fp16_rowmajor_cublaslt(lt, computeStream, mChunk, N, K, dA, dB, dCL, workspace);
            for (int i = 0; i < chunks; ++i) {
                // Record an event when previous GEMM is done
                CHECK_CUDA(cudaEventRecord(gemmDone, computeStream));
                // launch next GEMM asynchronously
                if (i + 1 < chunks) {
                    auto* dAx1 = dA + (i + 1) * (mChunk * K);
                    auto* dCx1 = dCL + (i + 1) * (mChunk * N);
                    gemm_fp16_rowmajor_cublaslt(lt, computeStream, mChunk, N, K, dAx1, dB, dCx1, workspace);
                }
                // Meanwhile transfer completed chunk
                const long int chunkSize = mChunk * N;
                for (int j = 1; j < world; ++j) {
                    const auto peer = (rank + j) % world;
                    const auto streamIdx = (j - 1) % copyStreams.size();
                    auto* dCp = dC + (peer * (M * N) + (i * mChunk * N));
                    const auto* dCx = dCL + i * (mChunk * N);
                    auto  s = copyStreams[streamIdx];
                    CHECK_CUDA(cudaStreamWaitEvent(s, gemmDone, 0));
                    ncclGroupStart();
                    // send
                    ncclSend(dCx, chunkSize, ncclT, peer, comm, s);
                    // recv
                    ncclRecv(dCp, chunkSize, ncclT, peer, comm, s);
                    ncclGroupEnd();
                }
            }
        }
            break;
        case DG_OVERLAP_MODE::CE: {
            nvtx3::scoped_range ceRange{"CE"};
            gemm_fp16_rowmajor_cublaslt(lt, computeStream, mChunk, N, K, dA, dB, dCL, workspace);
            for (int i = 0; i < chunks; ++i) {
                // wait for current gemm to finish
                CHECK_CUDA(cudaEventRecord(gemmDone, computeStream));
                // launch next GEMM asynchronously
                if (i + 1 < chunks) {
                    auto* dAx1 = dA + (i + 1) * (mChunk * K);
                    auto* dCx1 = dCL + (i + 1) * (mChunk * K);
                    gemm_fp16_rowmajor_cublaslt(lt, computeStream, mChunk, N, K, dAx1, dB, dCx1, workspace);
                }
                // Meanwhile transfer completed chunk
                const long int chunkSize = mChunk * N;
                for (int j = 1; j < world; ++j) {
                    const auto peer = (rank + j) % world;
                    const auto streamIdx = (j - 1) % copyStreams.size();
                    auto* dCp = dCPeer[peer] + (rank * (M * N) + (i * mChunk * K));
                    assert(dCp != nullptr);
                    const auto* dCx = dCL + i * (mChunk * N);
                    // transfer with CE
                    auto s = copyStreams[streamIdx];
                    CHECK_CUDA(cudaStreamWaitEvent(s, gemmDone, 0));
                    CHECK_CUDA(cudaMemcpyPeerAsync(dCp, peer, dCx, rank,
                        sizeof(Element) * chunkSize, s));
                }
            }
        }
            break;
        case DG_OVERLAP_MODE::NVSH_HOST: {
            nvtx3::scoped_range nvshRange{"NVSH-HOST"};
            gemm_fp16_rowmajor_cublaslt(lt, computeStream, mChunk, N, K, dA, dB, dCL, workspace);
            for (int i = 0; i < chunks; ++i) {
                // wait for current gemm to finish
                CHECK_CUDA(cudaEventRecord(gemmDone, computeStream));
                // launch next GEMM asynchronously
                if (i + 1 < chunks) {
                    auto* dAx1 = dA + (i + 1) * (mChunk * K);
                    auto* dCx1 = dCL + (i + 1) * (mChunk * K);
                    gemm_fp16_rowmajor_cublaslt(lt, computeStream, mChunk, N, K, dAx1, dB, dCx1, workspace);
                }
                // Meanwhile transfer completed chunk
                const long int chunkSize = mChunk * N;
                const long int partition = chunkSize / COMM_BLOCKS;
                for (int j = 1; j < world; ++j) {
                    const auto peer = (rank + j) % world;
                    const auto streamIdx = (j - 1) % copyStreams.size();
                    auto* dCx = dCL + i * (mChunk * N);
                    auto  s = copyStreams[streamIdx];
                    CHECK_CUDA(cudaStreamWaitEvent(s, gemmDone, 0));
                    put<<<COMM_BLOCKS, 128, 0, s>>>(reinterpret_cast<cuda::std::byte*>(dCx),
                        reinterpret_cast<const cuda::std::byte*>(dCx), peer,
                        static_cast<long int>(sizeof(Element) * partition));
                }
            }
        }
            break;
        case DG_OVERLAP_MODE::NVSH_FUSED: {
            nvtx3::scoped_range fused{"NVSH-FUSED"};
            // within a fused kernel, overlap GEMM and tile-level communication
        }
            break;
    }

    // sync all streams
    if (mode != DG_OVERLAP_MODE::NVSH_FUSED) {
        // complete communication
        for (auto s: copyStreams) {
            CHECK_CUDA(cudaStreamSynchronize(s));
        }
    }
}

__host__ __forceinline__
auto init_nccl(const int rank, const int world) {
    ncclUniqueId id;
    if (rank == 0)
    {
        NCCL_CHECK(ncclGetUniqueId(&id));
    }
    MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, world, id, rank));
    return comm;
}

__host__ __forceinline__
void teardown_nccl(ncclComm_t comm) {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    NCCL_CHECK(ncclCommFinalize(comm));
    NCCL_CHECK(ncclCommDestroy(comm));
}
__host__ __forceinline__
void run_dist_gemm(const Args &a) {
    NVTX3_FUNC_RANGE();
    // CSV header
    // initialize NVSHMEM backend
    nvshmem_init();
    const int rank = nvshmem_my_pe();
    const int world = nvshmem_n_pes();
    CHECK_CUDA(cudaSetDevice(rank));
    // MPI is initialized in nvshmem_init
    auto comm = init_nccl(rank, world);
    cudaStream_t computeStream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    int nCopyEngines = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&nCopyEngines, cudaDevAttrAsyncEngineCount, rank));
    std::vector<cudaStream_t> copyStreams(nCopyEngines);
    for (auto &copyStream: copyStreams) {
        cudaStream_t s;
        CHECK_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        copyStream = s;
    }

    cublasLtHandle_t lt;
    CUBLAS_CHECK(cublasLtCreate(&lt));
    void *workspace = nullptr;
    constexpr size_t workspace_bytes = 32ul * 1024ul * 1024ul;
    CHECK_CUDA(cudaMallocAsync(&workspace, workspace_bytes, computeStream));

    // allocate A, B and C buffers
    cudaEvent_t gemmDone;
    CHECK_CUDA(cudaEventCreateWithFlags(&gemmDone, cudaEventDisableTiming));
    Element *dA = nullptr;
    const auto aSeed = 41 * (rank + 1);
    const auto mSlice = a.M_max / world;
    CHECK_CUDA(cudaMallocAsync(&dA, mSlice * a.K * sizeof(Element), computeStream));
    fill_uniform(dA, mSlice * a.K, computeStream, aSeed);
    Element *dB = nullptr;
    constexpr auto bSeed = 42;
    CHECK_CUDA(cudaMallocAsync(&dB, a.N * a.K * sizeof(Element), computeStream));
    fill_uniform(dB, a.N * a.K, computeStream, bSeed);
    Element* dCref = nullptr;
    Element* dAref = nullptr;
    CHECK_CUDA(cudaMallocAsync(&dCref, a.M_max * a.N * sizeof(Element), computeStream));
    CHECK_CUDA(cudaMallocAsync(&dAref, a.M_max * a.K * sizeof(Element), computeStream));
    unsigned long long* d_mis = nullptr;
    CHECK_CUDA(cudaMallocAsync(&d_mis, sizeof(unsigned long long), computeStream));
    CHECK_CUDA(cudaStreamSynchronize(computeStream));
    auto *__restrict__ dC = static_cast<Element*>(nvshmem_malloc(a.M_max * a.N * sizeof(Element)));
    constexpr auto mChunk = 4;
    if (a.M_min % (world * mChunk) != 0 || (mChunk * a.N) % COMM_BLOCKS != 0) {
        if (rank == 0) {
            fprintf(stderr, "Incorrect args: %d, %d\n", a.M_min, a.N);
        }
        return;
    }
    // Events
    cudaEvent_t t_req_start, t_req_end;
    CHECK_CUDA(cudaEventCreate(&t_req_start));
    CHECK_CUDA(cudaEventCreate(&t_req_end));
    if (rank == 0) {
        printf("case,correct?,world,M,N,K,E2E_ms\n");
    }
    float errorVal = 0.0f;
    for (int m = a.M_min; m <= a.M_max; m *= a.step_factor) {
        constexpr auto nCases = 2;
        constexpr std::array cases{"NCCL-overlap", "NVSH-Host-overlap", "NVSH-Fused-overlap"};
        constexpr std::array dg_modes{DG_OVERLAP_MODE::NCCL,
            DG_OVERLAP_MODE::NVSH_HOST, DG_OVERLAP_MODE::NVSH_FUSED};
        #pragma unroll
        for (int c = 0; c < nCases; ++c) {
            // correctness check
            if (a.check) {
                const int mLocal = m / world;
                for (int r = 0; r < world; ++r) {
                    auto* dAp = dAref + (r * mLocal * a.K);
                    const auto rSeed = 41 * (r + 1);
                    fill_uniform(dAp, mLocal * a.K, computeStream, rSeed);
                }
                dist_gemm(dg_modes[c], dA, dB, dC, m, m / world, a.N, a.K, mChunk, rank, world, computeStream,
                    copyStreams, gemmDone, lt, workspace, comm);
                nvshmemx_barrier_all_on_stream(computeStream);
                gemm_fp16_rowmajor_cublaslt(lt, computeStream, m, a.N, a.K, dAref, dB, dCref, workspace);
                auto error = checkCorrectnessHost(dC, dCref, m * a.N, d_mis, computeStream);
                MPI_Allreduce(&error, &errorVal, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
            }
            for (int i = 0; i < a.warmup_iters; ++i) {
                dist_gemm(dg_modes[c], dA, dB, dC, m, m / world, a.N, a.K, mChunk, rank, world, computeStream,
                    copyStreams, gemmDone, lt, workspace, comm);
                nvshmemx_barrier_all_on_stream(computeStream);
            }
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaEventRecord(t_req_start));
            for (int j = 0; j < a.iters; ++j) {
                dist_gemm(dg_modes[c], dA, dB, dC, m, m / world, a.N, a.K, mChunk, rank, world, computeStream,
                    copyStreams, gemmDone, lt, workspace, comm);
                nvshmemx_barrier_all_on_stream(computeStream);
            }
            CHECK_CUDA(cudaEventRecord(t_req_end));
            CHECK_CUDA(cudaEventSynchronize(t_req_end));
            const auto e2e_ms = elapsed_ms(t_req_start, t_req_end);
            if (rank == 0) {
                printf("%s,%s,%d,%d,%d,%d,%.4f\n", cases[c], errorVal > 1e-3? "No" : "Yes", world, m, a.N, a.K,
                    e2e_ms / static_cast<float>(a.iters));
                fflush(stdout);
            }
        }
    }
    CHECK_CUDA(cudaEventDestroy(t_req_start));
    CHECK_CUDA(cudaEventDestroy(t_req_end));
    CHECK_CUDA(cudaStreamSynchronize(computeStream));
    CHECK_CUDA(cudaFreeAsync(workspace, computeStream));
    CHECK_CUDA(cudaFreeAsync(dA, computeStream));
    CHECK_CUDA(cudaFreeAsync(dB, computeStream));
    for (auto s: copyStreams) {
        CHECK_CUDA(cudaStreamSynchronize(s));
        CHECK_CUDA(cudaStreamDestroy(s));
    }
    CUBLAS_CHECK(cublasLtDestroy(lt));
    CHECK_CUDA(cudaStreamSynchronize(computeStream));
    teardown_nccl(comm);
    CHECK_CUDA(cudaStreamDestroy(computeStream));
    nvshmem_free(dC);
    nvshmem_finalize();
}

__host__ __forceinline__
void t_ce(const Args& a, std::barrier<>& b, int rank, std::vector<Element*> const& dCpeer) {
    const int world = a.world;
    CHECK_CUDA(cudaSetDevice(rank));
    cudaStream_t computeStream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    int nCopyEngines = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&nCopyEngines, cudaDevAttrAsyncEngineCount, rank));
    std::vector<cudaStream_t> copyStreams(nCopyEngines);
    for (auto &copyStream: copyStreams) {
        cudaStream_t s;
        CHECK_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        copyStream = s;
    }
    cublasLtHandle_t lt;
    CUBLAS_CHECK(cublasLtCreate(&lt));
    void *workspace = nullptr;
    constexpr size_t workspace_bytes = 32ul * 1024ul * 1024ul;
    CHECK_CUDA(cudaMallocAsync(&workspace, workspace_bytes, computeStream));

    // allocate A, B and C buffers
    cudaEvent_t gemmDone;
    CHECK_CUDA(cudaEventCreateWithFlags(&gemmDone, cudaEventDisableTiming));
    Element *dA = nullptr;
    const auto aSeed = 41 * (rank + 1);
    const auto mSlice = a.M_max / world;
    CHECK_CUDA(cudaMallocAsync(&dA, mSlice * a.K * sizeof(Element), computeStream));
    fill_uniform(dA, mSlice * a.K, computeStream, aSeed);
    Element *dB = nullptr;
    constexpr auto bSeed = 42;
    CHECK_CUDA(cudaMallocAsync(&dB, a.N * a.K * sizeof(Element), computeStream));
    fill_uniform(dB, a.N * a.K, computeStream, bSeed);
    Element* dCref = nullptr;
    Element* dAref = nullptr;
    CHECK_CUDA(cudaMallocAsync(&dCref, a.M_max * a.N * sizeof(Element), computeStream));
    CHECK_CUDA(cudaMallocAsync(&dAref, a.M_max * a.K * sizeof(Element), computeStream));
    for (int i = 0; i < world; ++i) {
        auto* dAp = dAref + (i * mSlice * a.K);
        const auto rSeed = 41 * (i + 1);
        fill_uniform(dAp, mSlice * a.K, computeStream, rSeed);
    }
    constexpr auto mChunk = 4;
    if (a.M_min % (world * mChunk) != 0 || (mChunk * a.N) % COMM_BLOCKS != 0) {
        if (rank == 0) {
            fprintf(stderr, "Incorrect args: %d, %d\n", a.M_min, a.N);
        }
        return;
    }
    unsigned long long* d_mis = nullptr;
    CHECK_CUDA(cudaMallocAsync(&d_mis, sizeof(unsigned long long), computeStream));
    // Events
    cudaEvent_t t_req_start, t_req_end;
    CHECK_CUDA(cudaEventCreate(&t_req_start));
    CHECK_CUDA(cudaEventCreate(&t_req_end));
    auto* dC = dCpeer[rank];
    for (int m = a.M_min; m <= a.M_max; m *= a.step_factor) {
        // correctness check
        dist_gemm(DG_OVERLAP_MODE::CE, dA, dB, dC,m, m / world, a.N, a.K, mChunk, rank, world, computeStream,
                copyStreams, gemmDone, lt, workspace, nullptr, dCpeer.data());
        CHECK_CUDA(cudaStreamSynchronize(computeStream));
        b.arrive_and_wait();
        // thread barrier
        gemm_fp16_rowmajor_cublaslt(lt, computeStream, m, a.N, a.K, dAref, dB, dCref, workspace);
        const auto error = checkCorrectnessHost(dC, dCref, m * a.N, d_mis, computeStream);
        if (rank == 0) {
            printf("CE-OVERLAP, error, %.2f %%, world,M,N,K,E2E_ms\n", error);
        }
        else {
            printf("CE-OVERLAP, error, %.2f %%\n", error);
        }
        for (int i = 0; i < a.warmup_iters; ++i) {
            dist_gemm(DG_OVERLAP_MODE::CE, dA, dB, dC, m, m / world, a.N, a.K, mChunk, rank, world, computeStream,
                copyStreams, gemmDone, lt, workspace, nullptr, dCpeer.data());
            CHECK_CUDA(cudaStreamSynchronize(computeStream));
            b.arrive_and_wait();
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(t_req_start));
        for (int j = 0; j < a.iters; ++j) {
            dist_gemm(DG_OVERLAP_MODE::CE, dA, dB, dC, m, m / world, a.N, a.K, mChunk, rank, world, computeStream,
                copyStreams, gemmDone, lt, workspace, nullptr, dCpeer.data());
            CHECK_CUDA(cudaStreamSynchronize(computeStream));
            b.arrive_and_wait();
        }
        CHECK_CUDA(cudaEventRecord(t_req_end));
        CHECK_CUDA(cudaEventSynchronize(t_req_end));
        const auto e2e_ms = elapsed_ms(t_req_start, t_req_end);
        if (rank == 0) {
            printf("%d,%d,%d,%d,%.4f\n", world, m, a.N, a.K,
                e2e_ms / static_cast<float>(a.iters));
            fflush(stdout);
        }
    }
    CHECK_CUDA(cudaEventDestroy(t_req_start));
    CHECK_CUDA(cudaEventDestroy(t_req_end));
    CHECK_CUDA(cudaStreamSynchronize(computeStream));
    CHECK_CUDA(cudaFreeAsync(workspace, computeStream));
    CHECK_CUDA(cudaFreeAsync(dA, computeStream));
    CHECK_CUDA(cudaFreeAsync(dB, computeStream));
    for (auto s: copyStreams) {
        CHECK_CUDA(cudaStreamSynchronize(s));
        CHECK_CUDA(cudaStreamDestroy(s));
    }
    CUBLAS_CHECK(cublasLtDestroy(lt));
    CHECK_CUDA(cudaStreamSynchronize(computeStream));
    CHECK_CUDA(cudaStreamDestroy(computeStream));
}
__host__ __forceinline__
void dist_gemm_ce(const Args& a) {
    const int world = a.world;
    const auto mSlice = a.M_max / world;
    std::vector<Element*> ptrs (world);
    std::barrier sync_point(world);
    for (int i = 0; i < world; ++i) {
        Element* dC = nullptr;
        CHECK_CUDA(cudaSetDevice(i));
        // malloc result array
        CHECK_CUDA(cudaMalloc(&dC, mSlice * a.N * sizeof(Element)));
        ptrs[i] = dC;
    }
    std::vector<std::thread> workers;
    workers.reserve(world);
    for (int i = 0; i < world; ++i) {
        // Capture by reference where appropriate; rank by value
        int rank = i;
        workers.emplace_back(
            [&, rank] {
                t_ce(a, sync_point, rank, ptrs);
            }
        );
    }
    for (int i = 0; i < world; ++i) {
        workers[i].join();
    }
}
// --------------------
// Main
// --------------------
int main(const int argc, char **argv) {
    if (const auto a = parse_args(argc, argv); a.ce) {
        dist_gemm_ce(a);
    }
    else {
        run_dist_gemm(a);
    }
    return 0;
}

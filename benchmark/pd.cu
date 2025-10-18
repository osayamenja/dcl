// nvcc pd.cu -o pd
//  A synthetic disaggregated inference pipeline

#include <cmath>

#include <cub/cub.cuh>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda/type_traits>
#include <cuda/std/limits>

#include "util.cuh"

// -------------------------------
// Config (simple & fixed)
// -------------------------------
static constexpr int D_MODEL = 2048;
static constexpr int H       = 16;
static constexpr int DK      = D_MODEL / H;
constexpr auto THREADS = 256;
using DataType = __half;

// -------------------------------
// Device utils
// -------------------------------
__device__ __forceinline__
constexpr auto fast_gelu(const float& x) {
    constexpr float kAlpha = 0.7978845608f; // sqrt(2/pi)
    constexpr float kBeta  = 0.044715f;
    const float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(kAlpha * (x + kBeta * x3)));
}

// -------------------------------
// Kernel: init embeddings X [R,S,D_MODEL] with a cheap pattern
// -------------------------------
template<typename Element = __half>
requires(cuda::std::is_same_v<Element, __half>)
__global__ void init_embeddings(Element* __restrict__ X,
    const int __grid_constant__ R, const int __grid_constant__ S,
    const int __grid_constant__ d_model) {
    const size_t n = R * S * d_model;
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // simple deterministic pattern (keeps math stable; no RNG needed)
    const float v = fmodf((i % 9973) * 0.001f, 1.0f) - 0.5f;
    X[i] = __float2half(v);
}

// -------------------------------
// Kernel: Prefill KV synthesizer
// K,V: [R,H,S,DK] from X: [R,S,D_MODEL]
// -------------------------------
template<int dk, typename Element = __half>
requires(cuda::std::is_same_v<Element, __half>)
__global__ void prefill(const Element* __restrict__ X,
                              Element* __restrict__ K,
                              Element* __restrict__ V,
                              const int __grid_constant__ R,
                              const int __grid_constant__ S,
                              const int __grid_constant__ d_model,
                              const int __grid_constant__ h,
                              const int __grid_constant__ delta /* e.g., 31 */) {
    const int total_rows = R * S * h;  // each row = DK values
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;

    int tmp = row;
    const int head = tmp % h; tmp /= h;
    const int s    = tmp % S; tmp /= S;
    const int r    = tmp;

    const auto* __restrict__ x_base = X + (static_cast<size_t>(r) * S + s) * d_model;
    auto* __restrict__ k_base = K + ((static_cast<size_t>(r) * h + head) * S + s) * dk;
    auto* __restrict__ v_base = V + ((static_cast<size_t>(r) * h + head) * S + s) * dk;

    const int x_k_off = head * dk;
    const int x_v_off = (head * dk + delta) % d_model;

    #pragma unroll
    for (int j = 0; j < dk; ++j) {
        const auto xk = __half2float(x_base[x_k_off + j]);
        const auto xv = __half2float(x_base[(x_v_off + j) % d_model]);
        k_base[j] = __float2half(xk);
        v_base[j] = __float2half(fast_gelu(xv));
    }
}

// -------------------------------
// Kernel: Decode step (scores→softmax→context) per (r, head)
// x_t: [R,D_MODEL], K,V: [R,H,S,DK], y_out: [R,H,DK]
// -------------------------------
template<int threads = 128, int dk, typename Element>
requires(cuda::std::is_same_v<Element, __half>)
__global__ void decode_step(const Element* __restrict__ x_t,
    const Element* __restrict__ K,
    const Element* __restrict__ V,
    Element* __restrict__ y_out,
    const int __grid_constant__ R,
    const int __grid_constant__ S,
    const int __grid_constant__ d_model,
    const int __grid_constant__ h) {
    extern __shared__ float smem_weights[]; // size S
    __shared__ float result[2];
    using BlockReduce = cub::BlockReduce<float, 128>;
    __shared__ BlockReduce::TempStorage temp_storage;
    const int r    = blockIdx.x;
    const int head = blockIdx.y;
    if (r >= R || head >= h) return;
    const auto* __restrict__ x_base = x_t + static_cast<size_t>(r) * d_model;
    const auto* __restrict__ K_base = K + (static_cast<size_t>(r) * h + head) * S * dk;
    const auto* __restrict__ V_base = V + (static_cast<size_t>(r) * h + head) * S * dk;

    const float inv_sqrt_dk = rsqrtf(static_cast<float>(dk));
    float local_max = -cuda::std::numeric_limits<float>::infinity();
    // Block-wide max over scores
    for (int s = threadIdx.x; s < S; s += threads) {
        const auto* __restrict__ k_row = K_base + static_cast<size_t>(s) * dk;
        float dot = 0.0f;
        #pragma unroll
        for (int j = 0; j < dk; ++j) {
            const float qj = __half2float(x_base[head * dk + j]);
            const float kj = __half2float(k_row[j]);
            dot += qj * kj;
        }
        const float score = dot * inv_sqrt_dk;
        local_max = fmaxf(local_max, score);
    }
    float max_score = BlockReduce(temp_storage).Reduce(local_max, cuda::maximum<>{});
    if (!threadIdx.x) {
        result[0] = max_score;
    }
    __syncthreads();
    max_score = result[0];
    // Weights + denom
    float local_sum = 0.0f;
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        const auto* __restrict__ k_row = K_base + static_cast<size_t>(s) * dk;
        float dot = 0.0f;
        #pragma unroll
        for (int j = 0; j < dk; ++j) {
            const float qj = __half2float(x_base[head * dk + j]);
            const float kj = __half2float(k_row[j]);
            dot += qj * kj;
        }
        const float weight = __expf(dot * inv_sqrt_dk - max_score);
        smem_weights[s] = weight;
        local_sum += weight;
    }
    float denom = BlockReduce(temp_storage).Sum(local_sum);
    if (!threadIdx.x) {
        if (denom < cuda::std::numeric_limits<float>::epsilon()) {
            denom = 1e-6f;
        }
        result[1] = denom;
    }
    __syncthreads();
    denom = result[1];

    // Context y[dk]
    auto* y_base = y_out + (static_cast<size_t>(r) * h + head) * dk;
    for (int j = threadIdx.x; j < dk; j += threads) {
        float acc = 0.0f;
        for (int s = 0; s < S; ++s) {
            float w = smem_weights[s] / denom;
            const auto* __restrict__ v_row = V_base + static_cast<size_t>(s) * dk;
            acc += w * __half2float(v_row[j]);
    }
    y_base[j] = __float2half(acc);
  }
}

// -------------------------------
// Host: helpers
// -------------------------------
struct Buffers {
    // GPU0 (prefill)
    DataType* X0 = nullptr;      // [R,S,D_MODEL]
    DataType* K0 = nullptr;      // [R,H,S,DK]
    DataType* V0 = nullptr;      // [R,H,S,DK]
    // GPU1 (decode)
    DataType* K1 = nullptr;
    DataType* V1 = nullptr;
    DataType* x1 = nullptr;      // [R,D_MODEL] current token state
    DataType* y1 = nullptr;      // [R,H,DK] context (per head)
};

template<typename Element = __half>
__forceinline__
size_t bytes_X(const int& R, const int& S) {
    return static_cast<size_t>(R)*S*D_MODEL * sizeof(Element);
}
template<typename Element = __half>
__forceinline__
size_t bytes_KV(const int& R, const int& S) {
    return static_cast<size_t>(R)*H*S*DK * sizeof(Element);
}
template<typename Element = __half>
__forceinline__
size_t bytes_vec(const int& R) {
    return static_cast<size_t>(R)*D_MODEL * sizeof(Element);
}
template<typename Element = __half>
__forceinline__
size_t bytes_ctx(const int& R) {
    return static_cast<size_t>(R)*H*DK * sizeof(Element);
}

// Allocate all buffers for given R,S on GPU0/1
__host__ __forceinline__
void alloc_buffers(Buffers& b, const int& R, const int& S,
    const int& dev0, const int& dev1, cudaStream_t s1, cudaStream_t s2) {
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaMallocAsync(&b.X0, bytes_X(R,S), s1));
    CHECK_CUDA(cudaMallocAsync(&b.K0, bytes_KV(R,S), s1));
    CHECK_CUDA(cudaMallocAsync(&b.V0, bytes_KV(R,S), s1));
    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaMallocAsync(&b.K1, bytes_KV(R,S), s2));
    CHECK_CUDA(cudaMallocAsync(&b.V1, bytes_KV(R,S), s2));
    CHECK_CUDA(cudaMallocAsync(&b.x1, bytes_vec(R), s2));
    CHECK_CUDA(cudaMallocAsync(&b.y1, bytes_ctx(R), s2));
}

__host__ __forceinline__
void free_buffers(const Buffers& b, const int& dev0, const int& dev1,
    cudaStream_t s1, cudaStream_t s2) {
    CHECK_CUDA(cudaSetDevice(dev0));
    CHECK_CUDA(cudaFreeAsync(b.X0, s1));
    CHECK_CUDA(cudaFreeAsync(b.K0, s1));
    CHECK_CUDA(cudaFreeAsync(b.V0, s1));
    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaFreeAsync(b.K1, s2));
    CHECK_CUDA(cudaFreeAsync(b.V1, s2));
    CHECK_CUDA(cudaFreeAsync(b.x1, s2));
    CHECK_CUDA(cudaFreeAsync(b.y1, s2));
}

// A simple stand-in for the transfer layer; should replace with NCCL/NVSHMEM variants
__host__ __forceinline__
void transfer_KV_cudaMemcpyPeer(const Buffers& b, const int& R,
    const int& S, const int dev0, const int dev1, cudaStream_t s0) {
    const size_t& nBytes = bytes_KV(R,S);
    CHECK_CUDA(cudaMemcpyPeerAsync(b.K1, dev1, b.K0, dev0, nBytes, s0));
    CHECK_CUDA(cudaMemcpyPeerAsync(b.V1, dev1, b.V0, dev0, nBytes, s0));
}

// -------------------------------
// Driver: one prefill->transfer->decode pass
// Returns (ttft_ms, per_token_ms, e2e_ms)
// -------------------------------
struct Times {
    float ttft_ms;
    float per_token_ms;
    float e2e_ms;
};

enum class XM {
    NCCL,
    NVSHMEM_HOST,
    NVSHMEM_FUSED,
    PEER
};

template<XM xferMode>
__host__ __forceinline__
Times run_once(int R, int S, int T_steps,
               int dev0, int dev1) {
    Buffers b{};
    CHECK_CUDA(cudaSetDevice(dev0));
    cudaStream_t s0;
    CHECK_CUDA(cudaStreamCreate(&s0));
    CHECK_CUDA(cudaSetDevice(dev1));
    cudaStream_t s1;
    CHECK_CUDA(cudaStreamCreate(&s1));
    alloc_buffers(b, R, S, dev0, dev1, s0, s1);

    // Enable P2P (best effort)
    int can01=0, can10=0;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can01, dev0, dev1));
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can10, dev1, dev0));
    if (can01) {
        CHECK_CUDA(cudaSetDevice(dev0));
        CHECK_CUDA(cudaDeviceEnablePeerAccess(dev1,0));
    }
    if (can10) {
        CHECK_CUDA(cudaSetDevice(dev1));
        CHECK_CUDA(cudaDeviceEnablePeerAccess(dev0,0));
    }

    // Init embeddings on prefill device
    CHECK_CUDA(cudaSetDevice(dev0));
    {
        const auto n = static_cast<size_t>(R)*S*D_MODEL;
        dim3 grd(cuda::ceil_div(n, THREADS));
        init_embeddings<<<grd, THREADS, 0, s0>>>(b.X0, R, S, D_MODEL);
    }
    // Start wall-clock timer (simple and cross-device friendly)
    auto t_start = std::chrono::high_resolution_clock::now();

    // Prefill KV synth on GPU0
    {
        const auto total_rows = R * S * H;
        dim3 grd((total_rows + THREADS - 1)/THREADS);
        prefill<DK><<<grd, THREADS, 0, s0>>>(b.X0, b.K0, b.V0, R, S, D_MODEL, H, 31);
    }

    // Transfer KV (placeholder: peer copy; replace with NCCL/NVSHMEM in your runs)
    if constexpr (xferMode == XM::PEER) {
        transfer_KV_cudaMemcpyPeer(b, R, S, dev0, dev1, s0);
    }
    else if constexpr (xferMode == XM::NCCL) {

    }
    else if constexpr (xferMode == XM::NVSHMEM_HOST) {

    }

    // Seed initial decode token state on GPU1 (here: just concat(K head 0, first S pos) as dummy)
    CHECK_CUDA(cudaSetDevice(dev1));
    CHECK_CUDA(cudaMemsetAsync(b.x1, 0, bytes_vec(R), s1));

    // TTFT: time until the *first* decode step finishes
    auto t_ttft_start = std::chrono::high_resolution_clock::now();

    // One decode step
    {
        dim3 grid(R, H);
        const auto shmem = sizeof(float) * S;
        decode_step<THREADS, DK><<<grid, THREADS, shmem, s1>>>(b.x1, b.K1, b.V1, b.y1, R, S, D_MODEL, H);
    }
    CHECK_CUDA(cudaStreamSynchronize(s1));
    auto t_ttft_end = std::chrono::high_resolution_clock::now();
    auto ttft_ms = static_cast<float>(std::chrono::duration<double,std::milli>(t_ttft_end - t_ttft_start).count());

    // Per-token latency: average over T_steps decode iterations
    auto t_decode_start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < T_steps; ++t) {
        dim3 grid(R, H), block(256);
        size_t shmem = sizeof(float) * S;
        decode_step<THREADS, DK><<<grid, block, shmem, s1>>>(b.x1, b.K1, b.V1, b.y1, R, S, D_MODEL, H);
    }
    CHECK_CUDA(cudaStreamSynchronize(s1));
    auto t_decode_end = std::chrono::high_resolution_clock::now();
    auto per_token_ms = static_cast<float>(std::chrono::duration<double,std::milli>(t_decode_end - t_decode_start)
        .count() / T_steps);

    // E2E latency (prefill->transfer->T decode steps)
    CHECK_CUDA(cudaStreamSynchronize(s0));
    CHECK_CUDA(cudaStreamSynchronize(s1));
    const auto t_end = std::chrono::high_resolution_clock::now();
    auto e2e_ms = static_cast<float>(std::chrono::duration<double,std::milli>(t_end - t_start).count());

    // Cleanup
    free_buffers(b, dev0, dev1, s0, s1);
    CHECK_CUDA(cudaStreamDestroy(s0));
    CHECK_CUDA(cudaStreamDestroy(s1));
    return {ttft_ms, per_token_ms, e2e_ms};
}

// -------------------------------
// Main: sweep S (KV size)
// -------------------------------
int main(int argc, char** argv) {
    int dev0 = 0, dev1 = 1;
    int R = 4;            // requests in flight per worker
    int T_steps = 32;     // decode steps per request
    std::vector<int> S_list = {128, 256, 512, 1024, 2048}; // grows KV size (x-axis)
    std::string mode = "peer"; // replace per run: "nccl", "nvshmem_host", "nvshmem_fused", etc.

    if (argc > 1) mode = argv[1];

    int ngpus=0; CHECK_CUDA(cudaGetDeviceCount(&ngpus));
    if (ngpus < 2) {
        fprintf(stderr,"Need >=2 GPUs; found %d\n", ngpus); return 1;
    }

    printf("# mode=%s R=%d T=%d D_MODEL=%d H=%d DK=%d\n", mode.c_str(), R, T_steps, D_MODEL, H, DK);
    printf("# S, KV_MB_per_req, TTFT_ms, per_token_ms, E2E_ms\n");

    for (int S : S_list) {
        double kv_mb = (2.0 * H * DK * S * sizeof(__nv_bfloat16)) / (1024.0*1024.0); // per request, per layer omitted on purpose (we synth once)
        Times t = run_once<XM::PEER>(R, S, T_steps, dev0, dev1);
        printf("%d, %.2f, %.3f, %.3f, %.3f\n", S, kv_mb, t.ttft_ms, t.per_token_ms, t.e2e_ms);
    }
    return 0;
}
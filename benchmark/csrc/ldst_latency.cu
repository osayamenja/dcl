// gmem_latency.cu
// Measures:
// (1) Global LOAD latency via pointer-chasing (cycles)
// (2) STORE round-trip (store→fence→load same addr) upper-bound (cycles)
// (3) LOAD (local DRAM) + STORE (peer GPU DRAM) with/without system fence (cycles)
//
// Build: nvcc -O3 -arch=sm_90 gmem_latency.cu -o gmem_latency
// Run:
//   Single-GPU parts:        ./gmem_latency [N] [ITERS]
//   With peer test enabled:  ./gmem_latency [N] [ITERS] peer
//
// Notes:
// - The peer test requires at least 2 GPUs with P2P access enabled (UVA).
// - The "issue-only" variant measures ld+st issue latency with dependency.
// - The "system-visible" variant inserts fence.sc.sys each iter to ensure the
//   store is visible outside the GPU (upper bound including commit+fence).

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                       \
    cudaError_t err__ = (call);                                     \
    if (err__ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
              __FILE__, __LINE__, cudaGetErrorString(err__));       \
      std::exit(1);                                                 \
    }                                                               \
  } while(0)

// ---------------------------------------------
// Inline PTX helpers
// ---------------------------------------------
__device__ __forceinline__ uint32_t ld_cg_u32(const uint32_t* ptr) {
  uint32_t v;
  asm volatile ("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(ptr));
  return v;
}

__device__ __forceinline__ void st_wb_u32(uint32_t* ptr, uint32_t v) {
  asm volatile ("st.global.wb.u32 [%0], %1;" :: "l"(ptr), "r"(v));
}

__device__ __forceinline__ void fence_sc_gpu() {
#if __CUDACC_VER_MAJOR__ >= 12
  asm volatile ("fence.sc.gpu;" ::: "memory");
#else
  asm volatile ("membar.gl;" ::: "memory");
#endif
}

__device__ __forceinline__ void fence_sc_sys() {
#if __CUDACC_VER_MAJOR__ >= 12
  asm volatile ("fence.sc.sys;" ::: "memory");
#else
  // Fallback: threadfence_system emits a system-scope fence
  __threadfence_system();
#endif
}

// ---------------------------------------------
// (1) LOAD latency via pointer-chasing
// ---------------------------------------------
__global__ void load_latency_kernel(const uint32_t* __restrict__ next_idx,
                                    uint64_t iters,
                                    uint64_t* out_cycles,
                                    uint32_t* out_sink)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  uint32_t idx = 0u;

  #pragma unroll 1
  for (int w = 0; w < 64; ++w) {
    idx = ld_cg_u32(next_idx + idx);
  }
  asm volatile ("" ::: "memory");

  uint64_t start = clock64();

#pragma unroll 1
  for (uint64_t i = 0; i < iters; ++i) {
    idx = ld_cg_u32(next_idx + idx);
  }

  uint64_t stop = clock64();
  *out_cycles = (stop - start);
  *out_sink   = idx;
}

// ---------------------------------------------
// (2) STORE round-trip (store→fence→load)
// ---------------------------------------------
__global__ void store_roundtrip_latency_kernel(uint32_t* __restrict__ addr,
                                               uint64_t iters,
                                               uint64_t* out_cycles,
                                               uint32_t* out_sink)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  uint32_t v = 0u;

  for (int w = 0; w < 64; ++w) {
    st_wb_u32(addr, v);
    fence_sc_gpu();
    v = ld_cg_u32(addr) + 1;
  }
  asm volatile ("" ::: "memory");

  uint64_t start = clock64();

#pragma unroll 1
  for (uint64_t i = 0; i < iters; ++i) {
    st_wb_u32(addr, v);
    fence_sc_gpu();
    /*uint32_t tmp = ld_cg_u32(addr);
    v = tmp + 1;*/
  }

  uint64_t stop = clock64();
  *out_cycles = (stop - start);
  *out_sink   = v;
}

// ---------------------------------------------
// (3) LOAD(local) + STORE(peer) latency
//     Template parameter controls whether we include a system-scope fence
// ---------------------------------------------
template<bool SYSTEM_VISIBLE>
__global__ void load_store_peer_kernel(const uint32_t* __restrict__ next_idx_local,
                                       uint32_t* __restrict__ peer_dst,
                                       uint64_t iters,
                                       uint64_t* out_cycles,
                                       uint32_t* out_sink)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  uint32_t idx = 0u;   // pointer-chase index (local)
  uint32_t v   = 1u;   // evolving value to store to peer

  // Warmup
  #pragma unroll 1
  for (int w = 0; w < 64; ++w) {
    idx = ld_cg_u32(next_idx_local + idx);   // serialize via dependency
    v = v ^ idx ^ 0x9e3779b9u;               // mix to keep dependency
    st_wb_u32(peer_dst, v);                   // store to peer
    if constexpr (SYSTEM_VISIBLE) {
      fence_sc_sys();                        // ensure visibility (upper bound)
    }
  }
  asm volatile ("" ::: "memory");

  uint64_t start = clock64();

#pragma unroll 1
  for (uint64_t i = 0; i < iters; ++i) {
    // LOAD from local DRAM (pointer-chase to defeat caching/MLP)
    idx = ld_cg_u32(next_idx_local + idx);

    // Compute value dependent on the load to block reordering
    v = v + (idx | 1u);

    // STORE to peer GPU memory
    st_wb_u32(peer_dst, v);

    // Optionally force system visibility each iteration
    if constexpr (SYSTEM_VISIBLE) {
      fence_sc_sys();
    }
  }

  uint64_t stop = clock64();
  *out_cycles = (stop - start);
  *out_sink   = v ^ idx;  // keep live
}

// (4) LDGSTS (cp.async global->shared) + STORE(peer) latency
// Measures a dependent sequence: pointer-chase -> cp.async -> ld.shared -> st.global(peer)
// Serialized with data dependencies; reports cycles over ITERS.
template<bool SYSTEM_VISIBLE>
__global__ void ldgsts_store_peer_kernel(const uint32_t* __restrict__ src_base,
                                         uint32_t* __restrict__ peer_dst,
                                         uint64_t iters,
                                         uint64_t* out_cycles,
                                         uint32_t* out_sink)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  __shared__ uint32_t s_word;
  uint32_t idx = 0u;
  uint32_t acc = 1u;

  // Warmup
  #pragma unroll 1
  for (int w = 0; w < 64; ++w) {
    idx = ld_cg_u32(src_base + idx);  // dependency (pointer-chase)
    const void* gptr = src_base + idx;
    auto smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(&s_word));

#if __CUDA_ARCH__ >= 800
    //cp.async (LDGSTS): global -> shared
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
                  :: "r"(smem_addr), "l"(gptr), "n"(4));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
#else
    // Fallback for < SM80: ld.global + st.shared
    uint32_t t = ld_cg_u32(static_cast<const uint32_t *>(gptr));
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(smem_addr), "r"(t));
#endif
    uint32_t val;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val) : "r"(smem_addr));
    acc = (acc ^ val) + 0x9e3779b9u;
    st_wb_u32(peer_dst, acc);
    if constexpr (SYSTEM_VISIBLE) { fence_sc_sys(); }
  }
  asm volatile ("" ::: "memory");

  uint64_t start = clock64();

#pragma unroll 1
  for (uint64_t i = 0; i < iters; ++i) {
    // 1) dependent address gen via pointer-chase to serialize
    idx = ld_cg_u32(src_base + idx);

    // 2) cp.async (LDGSTS) local global -> shared
    const void* gptr = src_base + idx;
    auto smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(&s_word));

#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
                  :: "r"(smem_addr), "l"(gptr), "n"(4));
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
#else
    uint32_t t = ld_cg_u32(static_cast<const uint32_t*>(gptr));
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(smem_addr), "r"(t));
#endif

    // 3) consume from shared then 4) store to peer
    uint32_t val;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val) : "r"(smem_addr));
    acc = acc + (val | 1u);          // tie iterations
    st_wb_u32(peer_dst, acc);
    if constexpr (SYSTEM_VISIBLE) { fence_sc_sys(); }
  }

  uint64_t stop = clock64();
  *out_cycles = (stop - start);
  *out_sink   = acc ^ idx;  // keep live
}

// ---------------------------------------------
// Host utilities
// ---------------------------------------------
static void build_pointer_chase(std::vector<uint32_t>& next_idx)
{
  const auto N = static_cast<uint32_t>(next_idx.size());
  std::vector<uint32_t> perm(N);
  for (uint32_t i = 0; i < N; ++i) perm[i] = i;

  std::mt19937 rng(1234567);
  std::ranges::shuffle(perm.begin(), perm.end(), rng);

  for (uint32_t i = 0; i < N; ++i) {
    next_idx[perm[i]] = perm[(i + 1) % N];
  }
}

int main(int argc, char** argv)
{
  size_t   N     = (argc > 1 ? std::strtoull(argv[1], nullptr, 10) : (1ULL << 20));
  uint64_t ITR   = (argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1024ULL);
  bool     doPeer = (argc > 3 && std::string(argv[3]) == "peer");

  // --------------------------------
  // Device 0 info
  // --------------------------------
  int dev0 = 0;
  CUDA_CHECK(cudaSetDevice(dev0));
  cudaDeviceProp prop0{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop0, dev0));
  int clockRate = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, dev0));
  printf("Dev0: %s | SM %d.%d | clockRate=%.3f GHz\n",
         prop0.name, prop0.major, prop0.minor, clockRate / 1e6);

  // ------------------------------
  // Allocate & init pointer-chase (on Dev0)
  // ------------------------------
  std::vector<uint32_t> h_next(N);
  build_pointer_chase(h_next);

  uint32_t* d_next_dev0 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_next_dev0, N * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_next_dev0, h_next.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice));

  // Result buffers (Dev0)
  uint64_t *d_cycles_load = nullptr, *d_cycles_store = nullptr;
  uint32_t *d_sink_load   = nullptr, *d_sink_store   = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cycles_load, sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_cycles_store, sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_sink_load,   sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_sink_store,  sizeof(uint32_t)));

  // (2) store target on Dev0
  uint32_t* d_addr_store_dev0 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_addr_store_dev0, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_addr_store_dev0, 0, sizeof(uint32_t)));

  // ------------------------------
  // Launch single-GPU kernels on Dev0
  // ------------------------------
  dim3 grid(1), block(1);

  load_latency_kernel<<<grid, block>>>(d_next_dev0, ITR, d_cycles_load, d_sink_load);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  store_roundtrip_latency_kernel<<<grid, block>>>(d_addr_store_dev0, ITR, d_cycles_store, d_sink_store);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  uint64_t cycles_load = 0, cycles_store = 0;
  uint32_t sinkL = 0, sinkS = 0;
  CUDA_CHECK(cudaMemcpy(&cycles_load, d_cycles_load, sizeof(uint64_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&cycles_store, d_cycles_store, sizeof(uint64_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&sinkL,       d_sink_load,   sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&sinkS,       d_sink_store,  sizeof(uint32_t), cudaMemcpyDeviceToHost));

  double ghz0 = clockRate / 1e6; // GHz
  double avg_load_cycles  = double(cycles_load)  / double(ITR);
  double avg_store_cycles = double(cycles_store) / double(ITR);
  double load_ns  = (avg_load_cycles  / (ghz0 * 1e9)) * 1e9;
  double store_ns = (avg_store_cycles / (ghz0 * 1e9)) * 1e9;

  printf("\n=== Dev0 Results (ITERS=%llu) ===\n", (unsigned long long)ITR);
  printf("Load (cg, pointer-chase):      %.1f cycles  (~%.2f ns)\n", avg_load_cycles, load_ns);
  printf("Store→fence(gpu):              %.1f cycles  (~%.2f ns)\n", avg_store_cycles, store_ns);
  printf("(Sinks) load=%u store=%u\n", sinkL, sinkS);

  // ------------------------------
  // Optional: Peer LD+ST measurement
  // ------------------------------
  if (doPeer) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
      fprintf(stderr, "Peer test requested but only %d device(s) present.\n", deviceCount);
      std::exit(1);
    }
    int dev1 = 1;

    // Check P2P access
    int can01 = 0, can10 = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can01, dev0, dev1));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can10, dev1, dev0));
    if (!can01) {
      fprintf(stderr, "Device %d cannot access peer %d.\n", dev0, dev1);
      std::exit(1);
    }

    // Enable P2P (dev0 -> dev1 is required; enabling both directions is fine)
    CUDA_CHECK(cudaSetDevice(dev0));
    cudaError_t pe0 = cudaDeviceEnablePeerAccess(dev1, 0);
    if (pe0 != cudaSuccess && pe0 != cudaErrorPeerAccessAlreadyEnabled) {
      CUDA_CHECK(pe0);
    }
    CUDA_CHECK(cudaSetDevice(dev1));
    cudaError_t pe1 = cudaDeviceEnablePeerAccess(dev0, 0);
    if (pe1 != cudaSuccess && pe1 != cudaErrorPeerAccessAlreadyEnabled) {
      CUDA_CHECK(pe1);
    }

    // Allocate a destination word on Dev1 (peer memory)
    CUDA_CHECK(cudaSetDevice(dev1));
    uint32_t* d_peer_word_dev1 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_peer_word_dev1, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_peer_word_dev1, 0, sizeof(uint32_t)));

    // Switch back to Dev0 for kernel launch; UVA lets us pass dev1 pointer directly
    CUDA_CHECK(cudaSetDevice(dev0));

    // Result buffers for peer tests (on Dev0)
    uint64_t *d_cycles_issue = nullptr, *d_cycles_sysvis = nullptr;
    uint32_t *d_sink_peer_issue = nullptr, *d_sink_peer_sysvis = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cycles_issue, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_cycles_sysvis, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_sink_peer_issue, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sink_peer_sysvis, sizeof(uint32_t)));
    // After allocating peer result buffers for previous peer tests:
    uint64_t *d_cycles_ldgsts_issue = nullptr, *d_cycles_ldgsts_sys = nullptr;
    uint32_t *d_sink_ldgsts_issue   = nullptr, *d_sink_ldgsts_sys   = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cycles_ldgsts_issue, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_cycles_ldgsts_sys,   sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_sink_ldgsts_issue,   sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sink_ldgsts_sys,     sizeof(uint32_t)));

    // Launch: issue-only (no system fence)
    load_store_peer_kernel<false><<<grid, block>>>(d_next_dev0, d_peer_word_dev1, ITR,
                                                   d_cycles_issue, d_sink_peer_issue);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch: system-visible (fenced each iter)
    load_store_peer_kernel<true><<<grid, block>>>(d_next_dev0, d_peer_word_dev1, ITR,
                                                  d_cycles_sysvis, d_sink_peer_sysvis);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    ldgsts_store_peer_kernel<false><<<grid, block>>>(d_next_dev0, d_peer_word_dev1, ITR,
                                                 d_cycles_ldgsts_issue, d_sink_ldgsts_issue);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch: LDGSTS path (system-visible per-iter)
    ldgsts_store_peer_kernel<true><<<grid, block>>>(d_next_dev0, d_peer_word_dev1, ITR,
                                                    d_cycles_ldgsts_sys, d_sink_ldgsts_sys);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Fetch & report
    uint64_t cyc_issue=0, cyc_sys=0;
    uint32_t sink_issue=0, sink_sys=0;
    CUDA_CHECK(cudaMemcpy(&cyc_issue, d_cycles_issue, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&cyc_sys,   d_cycles_sysvis, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sink_issue, d_sink_peer_issue, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sink_sys,   d_sink_peer_sysvis, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    double avg_issue_cycles = double(cyc_issue) / double(ITR);
    double avg_sys_cycles   = double(cyc_sys)   / double(ITR);
    double issue_ns = (avg_issue_cycles / (ghz0 * 1e9)) * 1e9;
    double sys_ns   = (avg_sys_cycles   / (ghz0 * 1e9)) * 1e9;

    // Dev1 info for completeness
    cudaDeviceProp prop1{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop1, dev1));

    printf("\n=== LOAD(local Dev0) + STORE(peer Dev1) (ITERS=%llu) ===\n",
           (unsigned long long)ITR);
    printf("Peer topology: Dev0=%s  ->  Dev1=%s\n", prop0.name, prop1.name);
    printf("Issue-only (ld+st dependent, no sys fence): %.1f cycles  (~%.2f ns)\n",
           avg_issue_cycles, issue_ns);
    printf("System-visible (ld+st + fence.sc.sys):      %.1f cycles  (~%.2f ns)\n",
           avg_sys_cycles,   sys_ns);
    printf("(Sinks) issue=%u sys=%u\n", sink_issue, sink_sys);

    uint64_t cyc_ldgsts_issue=0, cyc_ldgsts_sys=0;
    uint32_t sink_ldgsts_issue=0, sink_ldgsts_sys=0;
    CUDA_CHECK(cudaMemcpy(&cyc_ldgsts_issue, d_cycles_ldgsts_issue, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&cyc_ldgsts_sys,   d_cycles_ldgsts_sys,   sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sink_ldgsts_issue, d_sink_ldgsts_issue,  sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&sink_ldgsts_sys,   d_sink_ldgsts_sys,    sizeof(uint32_t), cudaMemcpyDeviceToHost));

    double avg_ldgsts_issue = double(cyc_ldgsts_issue) / double(ITR);
    double avg_ldgsts_sys   = double(cyc_ldgsts_sys)   / double(ITR);
    double ns_ldgsts_issue  = (avg_ldgsts_issue / (ghz0 * 1e9)) * 1e9;
    double ns_ldgsts_sys    = (avg_ldgsts_sys   / (ghz0 * 1e9)) * 1e9;

    printf("\n=== LDGSTS(local Dev0) + STORE(peer Dev1)===\n");
    printf("Issue-only (cp.async + st, no sys fence): %.1f cycles (~%.2f ns)\n",
           avg_ldgsts_issue, ns_ldgsts_issue);
    printf("System-visible (cp.async + st + fence):   %.1f cycles (~%.2f ns)\n",
           avg_ldgsts_sys,   ns_ldgsts_sys);
    printf("(Sinks) issue=%u sys=%u\n", sink_ldgsts_issue, sink_ldgsts_sys);

    // Cleanup peer allocations
    CUDA_CHECK(cudaFree(d_cycles_issue));
    CUDA_CHECK(cudaFree(d_cycles_sysvis));
    CUDA_CHECK(cudaFree(d_sink_peer_issue));
    CUDA_CHECK(cudaFree(d_sink_peer_sysvis));
    CUDA_CHECK(cudaSetDevice(dev1));
    CUDA_CHECK(cudaFree(d_peer_word_dev1));
    CUDA_CHECK(cudaSetDevice(dev0));
    CUDA_CHECK(cudaFree(d_cycles_ldgsts_issue));
    CUDA_CHECK(cudaFree(d_cycles_ldgsts_sys));
    CUDA_CHECK(cudaFree(d_sink_ldgsts_issue));
    CUDA_CHECK(cudaFree(d_sink_ldgsts_sys));
  }

  // Cleanup Dev0 allocations
  CUDA_CHECK(cudaFree(d_next_dev0));
  CUDA_CHECK(cudaFree(d_cycles_load));
  CUDA_CHECK(cudaFree(d_cycles_store));
  CUDA_CHECK(cudaFree(d_sink_load));
  CUDA_CHECK(cudaFree(d_sink_store));
  CUDA_CHECK(cudaFree(d_addr_store_dev0));

  return 0;
}
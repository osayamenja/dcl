// nvshmem_put_device_min.cu
// Minimal device-side NVSHMEM put benchmark (block collective API).
// Measures per-iteration latency of: put_nbi_block + quiet
// Prints: bytes,iters,total_us,avg_us,GBps,cycles_per_iter
//

#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} } while(0)

__global__ void bw(void* __restrict__ dst, const void* __restrict__ src,
                                       const size_t __grid_constant__ nbytes,
                                       const int __grid_constant__ peer,
                                       const uint64_t __grid_constant__ iters,
                                       uint64_t* __restrict__ cycles_out)
{
    uint64_t start = 0, stop = 0;
    if (threadIdx.x == 0) {
        start = clock64();
    }

    for (uint64_t i = 0; i < iters; ++i) {
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

static inline double to_us_from_cycles(uint64_t cycles, double ghz) {
    // cycles / (GHz * 1e9 cycles/s) => seconds -> microseconds
    return (double)cycles / (ghz * 1e9) * 1e6;
}

int main(int argc, char** argv) {
    int min_exp = (argc > 1) ? std::atoi(argv[1]) : 25;    // 1<<3 = 8B
    int max_exp = (argc > 2) ? std::atoi(argv[2]) : 26;   // 1GB
    int iters   = (argc > 3) ? std::atoi(argv[3]) : 2;

    // Choose CUDA device for this PE via init_attr (recommended).
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count == 0) { fprintf(stderr, "No CUDA devices.\n"); return 1; }
    nvshmem_init();  // minimal init

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    if (npes < 2) {
        if (mype == 0) fprintf(stderr, "Need >= 2 PEs\n");
        nvshmem_finalize(); return 1;
    }

    // Map PE -> device (round-robin)
    int dev = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    double ghz = prop.clockRate / 1e6; // kHz -> GHz

    // Symmetric buffers (max size)
    size_t max_bytes = size_t(1) << max_exp;
    void* src = nvshmem_malloc(max_bytes);
    void* dst = nvshmem_malloc(max_bytes);
    if (!src || !dst) {
        fprintf(stderr, "PE %d: nvshmem_malloc failed\n", mype);
        nvshmem_finalize(); return 1;
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemsetAsync(src, 0xAB, max_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dst, 0x00, max_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int peer = (mype + 1) % npes;

    if (mype == 0) {
        printf("# device-side NVSHMEM put benchmark (block API)\n");
        printf("# npes=%d, device=%s, iters=%d\n", npes, prop.name, iters);
        printf("bytes,iters,total_us,avg_us,GBps,cycles_per_iter\n");
    }

    // Device buffers for timing
    uint64_t* d_cycles = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(uint64_t)));

    // Benchmark over sizes
    for (int e = min_exp; e <= max_exp; ++e) {
        size_t nbytes = size_t(1) << e;

        // Sync all PEs and streams before measuring this size
        nvshmemx_barrier_all_on_stream(stream);

        // Warmup a few iterations (device-side)
        bw<<<1, 256, 0, stream>>>(dst, src, nbytes, peer, 10, d_cycles);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        nvshmemx_barrier_all_on_stream(stream);

        // Timed loop
        bw<<<1, 256, 0, stream>>>(dst, src, nbytes, peer, (uint64_t)iters, d_cycles);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        nvshmemx_barrier_all_on_stream(stream);

        // Fetch cycles and compute metrics
        uint64_t cycles = 0;
        CUDA_CHECK(cudaMemcpyAsync(&cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        double total_us = to_us_from_cycles(cycles, ghz);
        double avg_us   = total_us / iters;
        double GBps     = (double(nbytes) * iters) / (total_us * 1e-6) / 1e9;
        double cyc_per_iter = (double)cycles / (double)iters;

        if (mype == 0) {
            printf("%zu,%d,%.3f,%.6f,%.3f,%.1f\n",
                   nbytes, iters, total_us, avg_us, GBps, cyc_per_iter);
            fflush(stdout);
        }
    }

    CUDA_CHECK(cudaFree(d_cycles));
    nvshmem_free(src);
    nvshmem_free(dst);
    CUDA_CHECK(cudaStreamDestroy(stream));
    nvshmem_barrier_all();
    nvshmem_finalize();
    return 0;
}
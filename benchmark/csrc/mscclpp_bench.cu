//
// Created by osayamen on 11/5/25.
//
// Build: see CMakeLists.txt below
// Run:   mpirun -np 2 ./p2p_mem_bench --min 1K --max 256M --step 2 --iters 100 --warmup 20

#include <cstdio>
#include <string>
#include <cassert>
#include <mpi.h>

#include <cuda_runtime.h>

#include <mscclpp/core.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/semaphore.hpp>

#define CHECK_CUDA(cmd) do { \
  cudaError_t e = (cmd); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while(0)

__device__ __forceinline__ void grid_sync(int blocks) {
  // Simple global barrier across blocks in one launch using cooperative groups is ideal,
  // but for portability here we rely on __syncthreads() per block + 1-thread signaling via channel.
  // Host enforces stream sync between iterations, so this is sufficient for a benchmark.
  __syncthreads();
}

extern "C" __global__
void put_kernel(mscclpp::MemoryChannelDeviceHandle *dev,
                size_t copyBytes, int myRank, int totalThreads)
{
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Rendezvous before the copy (relaxed since it's just execution control).
  if (tid == 0) {
    dev->relaxedSignal();
    dev->relaxedWait();
  }
  grid_sync(gridDim.x);

  // Each rank copies from its local segment [myRank * copyBytes]
  // to the remote's segment at the same offset.
  const uint64_t srcOff = static_cast<uint64_t>(myRank) * copyBytes;
  const uint64_t dstOff = srcOff;

  // Parallel copy using all threads. API takes (dstOff, srcOff, size, threadId, numThreads)
  dev->put(dstOff, srcOff, copyBytes, tid, totalThreads);

  grid_sync(gridDim.x);

  // Let the peer know the copy is complete (full memory sync).
  if (tid == 0) {
    dev->signal();
    dev->wait();
  }
}

static size_t parse_size(std::string s) {
  // Accept 1K/1M/1G or raw integer bytes
  char suffix = 0;
  if (!s.empty()) suffix = static_cast<decltype(suffix)>(std::toupper(s.back()));
  size_t mul = 1;
  if (suffix=='K' || suffix=='M' || suffix=='G') s.pop_back();
  if (suffix=='K') mul = 1000ULL;
  else if (suffix=='M') mul = 1000ULL*1000ULL;
  else if (suffix=='G') mul = 1000ULL*1000ULL*1000ULL;
  return static_cast<size_t>(std::stoll(s)) * mul;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int world=0, rank=0;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (world != 2) {
    if (rank==0) fprintf(stderr, "This micro-benchmark expects -np 2\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Defaults
  size_t minB = parse_size("1K");
  size_t maxB = parse_size("256M");
  int step = 2;              // geometric factor
  int iters = 100;
  int warmup = 20;
  int blocks = 80;
  int threads = 256;

  // Parse args
  for (int i=1;i<argc;i++) {
    auto a = std::string(argv[i]);
    auto get = [&](const char* flag){ return i+1<argc && a==flag ? std::string(argv[++i]) : std::string(); };
    if (a=="--min")   { minB = parse_size(get("--min")); }
    else if (a=="--max") { maxB = parse_size(get("--max")); }
    else if (a=="--step"){ step = std::stoi(get("--step")); }
    else if (a=="--iters"){ iters = std::stoi(get("--iters")); }
    else if (a=="--warmup"){ warmup = std::stoi(get("--warmup")); }
    else if (a=="--blocks"){ blocks = std::stoi(get("--blocks")); }
    else if (a=="--threads"){ threads = std::stoi(get("--threads")); }
  }

  // Device selection: one GPU per rank (adjust if using taskset/visibility)
  CHECK_CUDA(cudaSetDevice(rank));

  // Bootstrap (via UniqueId over MPI)
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, world);
  mscclpp::UniqueId id;
  if (rank==0) id = mscclpp::TcpBootstrap::createUniqueId();
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(id);  // or bootstrap->initialize("if:ip:port")

  mscclpp::Communicator comm(bootstrap);

  // Transport
  constexpr auto transport = mscclpp::Transport::CudaIpc;

  const int remote = rank ^ 1;

  // Establish connection + semaphore
  auto conn = comm.connect({transport, {mscclpp::DeviceType::GPU, rank}}, remote).get();
  auto sema = comm.buildSemaphore(conn, remote).get();

  // Allocate a big GPU buffer (2 * maxB so each rank owns a disjoint half)
  const size_t totalBytes = 2ull * maxB;
  mscclpp::GpuBuffer buf(totalBytes);
  CHECK_CUDA(cudaMemset(buf.data(), 0, totalBytes));

  // Register memory and exchange RegisteredMemory
  auto localReg = comm.registerMemory(buf.data(), buf.bytes(), transport);
  comm.sendMemory(localReg, remote);
  auto remoteReg = comm.recvMemory(remote).get();

  // Build MemoryChannel (dst = remote, src = local)
  mscclpp::MemoryChannel chan(sema, /*dst*/remoteReg, /*src*/localReg);

  // After you construct `mscclpp::MemoryChannel chan(sema, remoteReg, localReg);`
  auto hostHandle = chan.deviceHandle();

  mscclpp::MemoryChannelDeviceHandle* devHandle = nullptr;
  CHECK_CUDA(cudaMalloc(&devHandle, sizeof(hostHandle)));
  CHECK_CUDA(cudaMemcpy(devHandle, &hostHandle, sizeof(hostHandle), cudaMemcpyHostToDevice));
  // CUDA timing scaffolding
  cudaStream_t stream; CHECK_CUDA(cudaStreamCreate(&stream));
  cudaEvent_t start, stop; CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));

  if (rank==0) {
    printf("# mscclpp_p2p_put, transport=%s, blocks=%d, threads=%d\n", "cudaIPC", blocks, threads);
    printf("# bytes,iters,ms_per_iter,GBps\n");
    fflush(stdout);
  }

  // Sweep message sizes
  for (size_t sz = minB; sz <= maxB; sz *= step) {
    // Warmup
    for (int w=0; w<warmup; ++w) {
      put_kernel<<<blocks, threads, 0, stream>>>(devHandle, sz, rank, blocks*threads);
      CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Timed iters
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int it=0; it<iters; ++it) {
      put_kernel<<<blocks, threads, 0, stream>>>(devHandle, sz, rank, blocks*threads);
      CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms=0.f; CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    // Two ranks each push `sz` bytes per iter (one-directional per rank).
    // Effective GB/s (per direction) from rank 0's perspective:
    double ms_per_iter = ms / iters;
    double gbps = (double)sz / ms_per_iter * 1e-6; // (bytes/ms) -> GB/s

    if (rank==0) {
      printf("%zu,%d,%.6f,%.3f\n", sz, iters, ms_per_iter, gbps);
      fflush(stdout);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(devHandle));
  MPI_Finalize();
  return 0;
}
// Benchmark cudaMemcpyPeer between two GPUs with CSV output.
//   ./p2p_bench --src 0 --dst 1 --min 1K --max 1G --iters 200 --warmup 20
//   ./p2p_bench --src 0 --dst 1 --min 1M --max 64M --step 2 --iters 100

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>

#define SC(T, v) static_cast<T>(v)
static void checkCuda(const cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
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
    int src = 0;
    int dst = 1;
    size_t minBytes = 1 << 29;   // 1 KiB
    size_t maxBytes = 1 << 30;   // 1 GiB
    int iters = 1;
    int warmup = 1;
    int step = 2;                // geometric step factor
    bool header = true;
};

Args parseArgs(const int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto needVal = [&](const char* name){
            if (i+1 >= argc) { std::cerr << name << " requires a value\n"; std::exit(EXIT_FAILURE);} };
        if (key == "--src") { needVal("--src"); a.src = std::stoi(argv[++i]); }
        else if (key == "--dst") { needVal("--dst"); a.dst = std::stoi(argv[++i]); }
        else if (key == "--min") { needVal("--min"); a.minBytes = parseSize(argv[++i]); }
        else if (key == "--max") { needVal("--max"); a.maxBytes = parseSize(argv[++i]); }
        else if (key == "--iters") { needVal("--iters"); a.iters = std::stoi(argv[++i]); }
        else if (key == "--warmup") { needVal("--warmup"); a.warmup = std::stoi(argv[++i]); }
        else if (key == "--step") { needVal("--step"); a.step = std::max(1, std::stoi(argv[++i])); }
        else if (key == "--no-header") { a.header = false; }
        else if (key == "-h" || key == "--help") {
            std::cout << "Usage: " << argv[0] <<
                " [--src N] [--dst N] [--min BYTES] [--max BYTES] [--iters N] [--warmup N] [--step F] [--no-header]\n";
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

int main(const int argc, char** argv) {
    const auto args = parseArgs(argc, argv);

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "get device count");
    if (deviceCount < 2) {
        std::cerr << "Need at least 2 CUDA devices for P2P" << std::endl;
        return EXIT_FAILURE;
    }
    if (args.src < 0 || args.src >= deviceCount || args.dst < 0 || args.dst >= deviceCount || args.src == args.dst) {
        std::cerr << "Invalid src/dst devices (have " << deviceCount << ")" << std::endl;
        return EXIT_FAILURE;
    }

    int access01 = 0, access10 = 0;
    checkCuda(cudaDeviceCanAccessPeer(&access01, args.dst, args.src), "canAccessPeer dst<-src");
    checkCuda(cudaDeviceCanAccessPeer(&access10, args.src, args.dst), "canAccessPeer src<-dst");
    if (!access01 || !access10) {
        std::cerr << "WARNING: Devices do not have P2P access. cudaMemcpyPeer may fail or be slow." << std::endl;
    }

    // Enable peer access (ignore if already enabled)
    checkCuda(cudaSetDevice(args.dst), "set dst device");
    cudaDeviceEnablePeerAccess(args.src, 0); // may return cudaErrorPeerAccessAlreadyEnabled
    cudaGetLastError(); // clear sticky errors
    checkCuda(cudaSetDevice(args.src), "set src device");
    cudaDeviceEnablePeerAccess(args.dst, 0);
    cudaGetLastError();

    // Allocate buffers
    checkCuda(cudaSetDevice(args.src), "set src device");
    void* srcBuf = nullptr;
    checkCuda(cudaMalloc(&srcBuf, args.maxBytes), "malloc src");
    checkCuda(cudaMemset(srcBuf, 0xA5, args.maxBytes), "init src");

    checkCuda(cudaSetDevice(args.dst), "set dst device");
    void* dstBuf = nullptr;
    checkCuda(cudaMalloc(&dstBuf, args.maxBytes), "malloc dst");
    checkCuda(cudaMemset(dstBuf, 0, args.maxBytes), "init dst");

    // Create stream and events on destination device (stream arg lives on dst for Peer copies)
    cudaStream_t stream;
    cudaEvent_t start, stop;
    checkCuda(cudaSetDevice(args.dst), "set dst for stream");
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");

    if (args.header) {
        std::cout << "src,dst,bytes,avg_us,gbps,iters" << std::endl;
    }

    for (size_t n = args.minBytes; n <= args.maxBytes; n = std::min(args.maxBytes, n * SC(size_t, args.step))) {
        // Warmup
        for (int i = 0; i < args.warmup; ++i) {
            checkCuda(cudaMemcpyPeerAsync(dstBuf, args.dst, srcBuf, args.src, n, stream), "warmup memcpyPeerAsync");
        }
        checkCuda(cudaStreamSynchronize(stream), "warmup sync");

        // Timed loop
        checkCuda(cudaEventRecord(start, stream), "record start");
        for (int i = 0; i < args.iters; ++i) {
            checkCuda(cudaMemcpyPeerAsync(dstBuf, args.dst, srcBuf, args.src, n, stream), "bench memcpyPeerAsync");
        }
        checkCuda(cudaEventRecord(stop, stream), "record stop");
        checkCuda(cudaEventSynchronize(stop), "sync stop");

        float ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");
        const float avg_us = (ms * 1000.0f) / SC(float, std::max(1, args.iters));
        float gbps = 0.0f;
        const auto bytes_per_s = SC(float, n * args.iters / (ms / 1000.0f));
        gbps = bytes_per_s / 1e9f; // decimal GB/s

        std::cout << args.src << "," << args.dst << "," << n << ","
                  << std::fixed << std::setprecision(3) << avg_us << ","
                  << std::setprecision(3) << gbps << "," << args.iters << std::endl;

        if (n == args.maxBytes) break; // guard in case step==1
        if (args.step == 1) {
            // Linear +1 step if user set step=1
            if (n + 1 > args.maxBytes) break;
            n = n + 1; // but will be multiplied again by for-loop; we handle via continue trick
            n = n / args.step; // neutralize upcoming multiplication
        }
    }

    // Cleanup
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);

    checkCuda(cudaSetDevice(args.dst), "set dst for free");
    cudaFree(dstBuf);
    checkCuda(cudaSetDevice(args.src), "set src for free");
    cudaFree(srcBuf);

    return EXIT_SUCCESS;
}

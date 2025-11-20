#include <iostream>

#include "benchmark/dgk.cuh"
#define CHECK_CUDA(x) do{cudaError_t e=(x); if(e!=cudaSuccess){ \
fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} \
}while(0)

void printMatrix(const Element* __restrict__ p, const int M, const int N) {
    const auto t = cute::make_tensor(p, cute::make_layout(cute::make_shape(M, N),
        cute::LayoutRight{}));
    for (int i = 0; i < M; ++i) {
        printf("{");
        for (int j = 0; j < N; ++j) {
            printf("%f,", __half2float(t(i, j)));
        }
        printf("}\n");
    }
}
void cdx() {
    constexpr auto M  = 64;
    constexpr auto N  = 64;
    constexpr auto K  = 64;

    CHECK_CUDA(cudaSetDevice(0));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    Element* dA = nullptr;
    Element* dB = nullptr;
    Element* dC = nullptr;
    CHECK_CUDA(cudaMallocManaged(&dA, M * K * sizeof(Element)));
    CHECK_CUDA(cudaMallocManaged(&dB, N * K * sizeof(Element)));
    CHECK_CUDA(cudaMallocManaged(&dC, M * N * sizeof(Element)));

    // fill
    dA[0] = 0.f;
    dA[1] = 1.f;
    dA[0 + K] = 2.f;
    dA[1 + K] = 3.f;

    dB[0] = 4.f;
    dB[1] = 5.f;
    dB[0 + K] = 6.f;
    dB[1 + K] = 7.f;

    dgk<<<1, FUSED_THREADS, 0, stream>>>(dA, dB, dC, nullptr, M, N, K, 0, 1);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    // print results
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaStreamDestroy(stream));
}
int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
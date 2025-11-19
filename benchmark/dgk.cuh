//
// Created by osayamen on 11/19/25.
//

#ifndef DCL_DGK_CUH
#define DCL_DGK_CUH
#include <cuda/utility>
#include <cuda/cmath>
#include <cublasdx.hpp>

using Element = __half;
#define TILE_M 128
#define TILE_N 128
#define TILE_K 32
#define FUSED_THREADS 128
#define SC(T, v) static_cast<T>(v)
__global__ void dgk(const Element* __restrict__ dA, const Element* __restrict__ dB,
    Element* __restrict__ dC,
    int* __restrict__ signals,
    const int __grid_constant__ M,
    const int __grid_constant__ N,
    const int __grid_constant__ K,
    const int __grid_constant__ rank,
    const int __grid_constant__ world) {
    using BLAS = decltype(cublasdx::Size<TILE_M, TILE_N, TILE_K>() +
                          cublasdx::Precision<Element, Element, float>() +
                          cublasdx::Type<cublasdx::type::real>() +
                          cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major, cublasdx::row_major>() +
                          cublasdx::Block() +
                          cublasdx::BlockDim<FUSED_THREADS>() +
                          cublasdx::MaxAlignment() +
                          cublasdx::experimental::StaticBlockDim() +
                          cublasdx::SM<ARCH>());
    const int localM = M / world;
    const int nTiles = cuda::ceil_div(localM, TILE_M) * (N / TILE_N);
    for (int tileIdx = SC(int, blockIdx.x); tileIdx < nTiles; ++tileIdx) {
        // do GEMM
        // communicate results
    }
    // await results
}
#endif //DCL_DGK_CUH
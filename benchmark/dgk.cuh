//
// Created by osayamen on 11/19/25.
//

#ifndef DCL_DGK_CUH
#define DCL_DGK_CUH
#include <cuda/utility>
#include <cuda/cmath>
#include <cublasdx.hpp>

using Element = __half;
#define TILE_M 64
#define TILE_N 64
#define TILE_K 32
#define FUSED_THREADS 128
#define PIPE_STAGES 4
#define SC(T, v) static_cast<T>(v)

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

template<int N>
__device__ __forceinline__
void cpWait() {
    cute::cp_async_wait<N>();
    __syncthreads();
}

__device__ __forceinline__
void mainloop(void* __restrict__ const& workspace,
    const Element* __restrict__ const& a,
    const Element* __restrict__ const& b,
    Element* __restrict__ const& c,
    const int& M,const int& N,const int& K,
    const int& tileIdx) {
    const auto partitioner = BLAS::suggest_partitioner();
    auto accumulator = partitioner.make_accumulator_fragment();
    cublasdx::clear(accumulator);
    using BM = cute::Int<cublasdx::size_of<BLAS>::m>;
    using BN = cute::Int<cublasdx::size_of<BLAS>::n>;
    using BK = cute::Int<cublasdx::size_of<BLAS>::k>;

    const int tilesM = M / BM{};
    const int tilesN = N / BN{};
    const int tilesK = K / BK{};

    const auto tileCoord = cute::idx2crd(tileIdx, cute::make_shape(tilesM, tilesN),
        cute::make_stride(tilesN, cute::_1{}));
    const auto ctaCoord = cute::make_coord(cute::get<0>(tileCoord),
        cute::get<1>(tileCoord), cute::_);
    const auto strideA = cute::conditional_return<cublasdx::arrangement_of_v_a<BLAS> == cublasdx::row_major>
    (cute::make_stride(K, cute::_1{}), cute::make_stride(cute::_1{}, M));
    const auto mA = cute::make_tensor(cute::make_gmem_ptr(a),
        cute::make_layout(cute::make_shape(M, K), strideA));
    const auto strideB = cute::conditional_return<cublasdx::arrangement_of_v_b<BLAS> == cublasdx::row_major>
    (cute::make_stride(N, cute::_1{}), cute::make_stride(cute::_1{}, K));
    const auto mB = cute::make_tensor(cute::make_gmem_ptr(b),
        cute::make_layout(cute::make_shape(K, N), strideB));
    const auto strideC = cute::conditional_return<cublasdx::arrangement_of_v_c<BLAS> == cublasdx::row_major>
    (cute::make_stride(N, cute::_1{}), cute::make_stride(cute::_1{}, M));
    const auto mC = cute::make_tensor(cute::make_gmem_ptr(c),
        cute::make_layout(cute::make_shape(M, N), strideC));

    const auto gA = cute::local_tile(mA, cute::Shape<BM, BK>{}, cute::select<0, 2>(ctaCoord));
    const auto gB = cute::local_tile(mB, cute::Shape<BK, BN>{}, cute::select<2, 1>(ctaCoord));
    auto gC = cute::local_tile(mC, cute::Shape<BM, BN>{}, cute::select<0, 1>(ctaCoord));
    // shared layouts
    constexpr auto sALay = cute::tile_to_shape(BLAS::suggest_layout_smem_a().layout,
        cute::Shape<BM, BK, cute::Int<PIPE_STAGES>>{});
    constexpr auto sBLay = cute::tile_to_shape(BLAS::suggest_layout_smem_b().layout,
        cute::Shape<BK, BN, cute::Int<PIPE_STAGES>>{});
    const auto [sA, sB] = cublasdx::shared_memory::slice<Element, Element>(
        workspace, cublasdx::alignment_of_v_a<BLAS>, sALay, cublasdx::alignment_of_v_b<BLAS>, sBLay);
    cute::for_each(cute::make_int_sequence<PIPE_STAGES>{}, [&](auto stage) {
        cublasdx::copy<BLAS, cublasdx::alignment_of_v_a<BLAS>>(gA(cute::_, cute::_, stage), sA(cute::_, cute::_, stage));
        cublasdx::copy<BLAS, cublasdx::alignment_of_v_b<BLAS>>(gB(cute::_, cute::_, stage), sB(cute::_, cute::_, stage));
        cute::cp_async_fence();
    });

    #pragma unroll 1
    for (int kStage = PIPE_STAGES; kStage < tilesK; ++kStage) {
        const auto ps = kStage % PIPE_STAGES;
        cpWait<PIPE_STAGES - 1>();
        BLAS().execute(sA(cute::_, cute::_, ps), sB(cute::_, cute::_, ps), accumulator);
        __syncthreads();
        cublasdx::copy<BLAS, cublasdx::alignment_of_v_a<BLAS>>(gA(cute::_, cute::_, kStage), sA(cute::_, cute::_, ps));
        cublasdx::copy<BLAS, cublasdx::alignment_of_v_b<BLAS>>(gB(cute::_, cute::_, kStage), sB(cute::_, cute::_, ps));
        cute::cp_async_fence();
    }

    cute::for_each(cute::make_int_rsequence<PIPE_STAGES>{}, [&](auto rStage) {
        const auto ps = (tilesK - (rStage + 1)) % PIPE_STAGES;
        cpWait<rStage>();
        BLAS().execute(sA(cute::_, cute::_, ps), sB(cute::_, cute::_, ps), accumulator);
    });
    // rmem -> gmem
    cublasdx::copy_fragment<cublasdx::alignment_of_v_c<BLAS>>(accumulator, gC, partitioner);
}

__global__ void dgk(const Element* __restrict__ dA, const Element* __restrict__ dB,
    Element* __restrict__ dC,
    int* __restrict__ signals,
    const int __grid_constant__ M,
    const int __grid_constant__ N,
    const int __grid_constant__ K,
    const int __grid_constant__ rank,
    const int __grid_constant__ world) {
    constexpr auto sharedSpace = cuda::std::max(PIPE_STAGES * TILE_K * (TILE_M + TILE_N),
        TILE_M * TILE_N) * sizeof(Element);
    __shared__ cuda::std::byte __align__(16) workspace[sharedSpace];
    const int localM = M / world;
    const int nTiles = cuda::ceil_div(localM, TILE_M) * (N / TILE_N);
    for (int tileIdx = SC(int, blockIdx.x); tileIdx < nTiles; ++tileIdx) {
        // do GEMM
        mainloop(workspace, dA, dB, dC, localM, N, K, tileIdx);
    }
}
#endif //DCL_DGK_CUH
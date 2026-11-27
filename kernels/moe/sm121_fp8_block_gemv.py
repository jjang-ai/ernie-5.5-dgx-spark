# SPDX-License-Identifier: Apache-2.0
"""Custom SM12.1 (NVIDIA GB10) high-bandwidth Block-128 FP8 small-M GEMV/GEMM.

Targets ERNIE 5.5's block-128 FP8 linear projections (shared expert gate_up/down,
fc1_latent_proj, fc2_latent_proj, attention projections) at decode batch sizes
M = 1..8, where vLLM's DeepGEMM / CutlassBlock / TritonBlock kernels only
achieve 25-43% of GB10's 273 GB/s LPDDR5X bandwidth.

Design:
  - Consumes raw checkpoint layout directly: weight [N, K] float8_e4m3fn +
    weight_scale_inv [N // 128, K // 128] float32 (zero repacking!).
  - Compact `__half2` shared memory (`M * K * 2` bytes -> <= 64 KB even at
    M=8, K=4096, comfortably within GB10's 99 KB SMEM budget!).
  - 128-bit coalesced `uint4` loads (16 FP8 weights/thread -> 512 bytes/warp).
  - Native CUDA 13 `<cuda_fp8.h>` FP8x2 -> FP16x2 conversion + `__hfma2`
    vector FMA + FP32 block-scaled accumulation.
  - Multi-token register tiling (`M = 1..8`): loads each FP8 weight ONCE and
    accumulates dot products for all M tokens simultaneously!
"""

from __future__ import annotations

from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

__device__ __forceinline__ __half2 cvt_e4m3x2_to_half2(uint16_t raw2) {
    __nv_fp8x2_storage_t s = raw2;
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(s, __NV_E4M3);
    return *reinterpret_cast<__half2*>(&h);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffffu, val, offset);
    }
    return val;
}

template <int M_STATIC, int ROWS_PER_WARP>
__global__ void fp8_block128_gemv_sm121_kernel(
    const __nv_bfloat16* __restrict__ x,          // [M, K]
    const uint8_t* __restrict__ w,                // [N, K] (float8_e4m3fn)
    const float* __restrict__ s_inv,              // [N/128, K/128]
    __nv_bfloat16* __restrict__ out,              // [M, N]
    int N,
    int K
) {
    extern __shared__ __half2 sx[];  // [M_STATIC * (K / 2)] -> 2 bytes/elem!

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    // Cooperatively load x[0..M-1, 0..K-1] into shared memory as half2
    const int total_elems = M_STATIC * K;
    for (int i = tid * 8; i < total_elems; i += blockDim.x * 8) {
        uint4 raw = __ldg(reinterpret_cast<const uint4*>(x + i));
        const uint32_t ws[4] = {raw.x, raw.y, raw.z, raw.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&ws[j]);
            float2 f = __bfloat1622float2(b);
            sx[(i >> 1) + j] = __float22half2_rn(f);
        }
    }
    __syncthreads();

    const int k_half = K >> 1;
    const int k_blocks = K >> 7;        // K / 128
    const int num_chunks = K >> 4;      // K / 16 (16 FP8 bytes = uint4 per thread)
    const int base_row = (blockIdx.x * 4 + warp_id) * ROWS_PER_WARP;

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int row = base_row + r;
        if (row >= N) break;

        const uint8_t* w_row = w + static_cast<int64_t>(row) * K;
        const float* s_row = s_inv + static_cast<int64_t>(row >> 7) * k_blocks;

        float acc[M_STATIC];
        #pragma unroll
        for (int m = 0; m < M_STATIC; ++m) acc[m] = 0.0f;

        for (int chunk = lane; chunk < num_chunks; chunk += 32) {
            const uint4 wv = __ldg(reinterpret_cast<const uint4*>(w_row + chunk * 16));
            const float scale = __ldg(s_row + (chunk >> 3));  // chunk / 8 = k / 128

            __half2 wh[8];
            const uint32_t words[4] = {wv.x, wv.y, wv.z, wv.w};
            #pragma unroll
            for (int widx = 0; widx < 4; ++widx) {
                uint32_t u = words[widx];
                wh[widx * 2 + 0] = cvt_e4m3x2_to_half2(static_cast<uint16_t>(u & 0xFFFFu));
                wh[widx * 2 + 1] = cvt_e4m3x2_to_half2(static_cast<uint16_t>(u >> 16));
            }

            const int k_h2_base = chunk * 8;
            #pragma unroll
            for (int m = 0; m < M_STATIC; ++m) {
                const __half2* xm = sx + m * k_half + k_h2_base;
                __half2 h0 = __hmul2(wh[0], xm[0]);
                __half2 h1 = __hmul2(wh[1], xm[1]);
                h0 = __hfma2(wh[2], xm[2], h0);
                h1 = __hfma2(wh[3], xm[3], h1);
                h0 = __hfma2(wh[4], xm[4], h0);
                h1 = __hfma2(wh[5], xm[5], h1);
                h0 = __hfma2(wh[6], xm[6], h0);
                h1 = __hfma2(wh[7], xm[7], h1);
                float2 f0 = __half22float2(h0);
                float2 f1 = __half22float2(h1);
                acc[m] = fmaf(f0.x + f0.y + f1.x + f1.y, scale, acc[m]);
            }
        }

        #pragma unroll
        for (int m = 0; m < M_STATIC; ++m) {
            float res = warp_reduce_sum(acc[m]);
            if (lane == 0) {
                out[static_cast<int64_t>(m) * N + row] = __float2bfloat16(res);
            }
        }
    }
}

torch::Tensor fp8_block128_gemv_sm121(
    torch::Tensor x,
    torch::Tensor w_fp8,
    torch::Tensor s_inv,
    int rows_per_warp
) {
    const int M = x.size(0);
    const int K = x.size(1);
    const int N = w_fp8.size(0);
    TORCH_CHECK(M >= 1 && M <= 8, "Specialized for M in 1..8");
    TORCH_CHECK(K % 128 == 0 && N % 128 == 0, "Requires K, N multiples of 128");

    auto out = torch::empty({M, N}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    const size_t smem = static_cast<size_t>(M) * K * sizeof(__half);

    auto dispatch_rpw = [&](auto m_tag) {
        constexpr int MS = decltype(m_tag)::value;
        auto launch = [&](auto r_tag) {
            constexpr int RPW = decltype(r_tag)::value;
            int grid_x = (N + 4 * RPW - 1) / (4 * RPW);
            auto kernel = fp8_block128_gemv_sm121_kernel<MS, RPW>;
            if (smem > 48 * 1024) {
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            }
            kernel<<<grid_x, 128, smem, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                reinterpret_cast<const uint8_t*>(w_fp8.data_ptr()),
                s_inv.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
                N, K
            );
        };
        if (rows_per_warp == 1) launch(std::integral_constant<int, 1>{});
        else if (rows_per_warp == 2) launch(std::integral_constant<int, 2>{});
        else if (rows_per_warp == 4) launch(std::integral_constant<int, 4>{});
        else launch(std::integral_constant<int, 8>{});
    };

    switch (M) {
        case 1: dispatch_rpw(std::integral_constant<int, 1>{}); break;
        case 2: dispatch_rpw(std::integral_constant<int, 2>{}); break;
        case 3: dispatch_rpw(std::integral_constant<int, 3>{}); break;
        case 4: dispatch_rpw(std::integral_constant<int, 4>{}); break;
        case 5: dispatch_rpw(std::integral_constant<int, 5>{}); break;
        case 6: dispatch_rpw(std::integral_constant<int, 6>{}); break;
        case 7: dispatch_rpw(std::integral_constant<int, 7>{}); break;
        case 8: dispatch_rpw(std::integral_constant<int, 8>{}); break;
    }
    return out;
}
"""

_CPP_SRC = """
#include <torch/extension.h>
torch::Tensor fp8_block128_gemv_sm121(
    torch::Tensor x,
    torch::Tensor w_fp8,
    torch::Tensor s_inv,
    int rows_per_warp
);
"""


@lru_cache(maxsize=1)
def _load_ext():
    major, minor = torch.cuda.get_device_capability(0)
    arch_flag = (
        "-gencode=arch=compute_121a,code=sm_121a"
        if (major, minor) == (12, 1)
        else "-gencode=arch=compute_120a,code=sm_120a"
    )
    return load_inline(
        name="sm121_fp8_block_gemv_v2_ext",
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        functions=["fp8_block128_gemv_sm121"],
        extra_cuda_cflags=["-O3", "--use_fast_math", arch_flag],
        verbose=False,
    )


def fp8_block128_gemv_sm121(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    s_inv: torch.Tensor,
    rows_per_warp: int = 4,
) -> torch.Tensor:
    """High-bandwidth SM12.1 Block-128 FP8 GEMV for decode M=1..8."""
    ext = _load_ext()
    return ext.fp8_block128_gemv_sm121(
        x.contiguous(), w_fp8.contiguous(), s_inv.contiguous(), int(rows_per_warp)
    )

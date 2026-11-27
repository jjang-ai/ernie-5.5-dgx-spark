# SPDX-License-Identifier: Apache-2.0
"""Custom SM12.1 (NVIDIA GB10) high-bandwidth MXFP4 small-M MoE decode kernel.

Designed specifically for GB10's memory-bandwidth-bound decode regime
(48 SMs, 273 GB/s LPDDR5X unified memory, M = 1..8) on ERNIE 5.5 Flash Preview:
  - Consumes raw checkpoint MXFP4 weights directly (ZERO repacking, compatible
    with DeepGEMM / standard safetensors layout: [E, N, K//2] uint8 + [E, N, K//32] uint8 E8M0).
  - Eliminates ALL MoE routing/alignment/permute/unpermute/activation launches:
    reduces the entire MoE layer (W13 + SiTU + W2 + top-k weighted reduce) to
    JUST TWO CUDA kernels!
  - Uses 128-bit coalesced `uint4` loads (1 thread = 16 bytes = 32 FP4 weights =
    EXACTLY 1 MXFP4 scale block), issues both K-chunks up front for maximum
    memory-level parallelism (MLP) on LPDDR5X, stores activations in compact
    4 KB `__half2` shared memory for maximum CTA occupancy, and uses native
    Blackwell SM120/SM121 CUDA 13 `<cuda_fp4.h>` hardware FP4->FP16 conversion
    with FP32 block-scaled accumulation.
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
#include <cuda_fp4.h>
#include <stdint.h>

__device__ __forceinline__ __half2 cvt_e2m1x2_to_half2(uint32_t byte_val) {
    __nv_fp4x2_storage_t packed = static_cast<__nv_fp4x2_storage_t>(byte_val & 0xFFu);
    __half2_raw h2 = __nv_cvt_fp4x2_to_halfraw2(packed, __NV_E2M1);
    return *reinterpret_cast<__half2*>(&h2);
}

__device__ __forceinline__ float e8m0_to_float(uint8_t s) {
    return (s == 0) ? 5.877471754111438e-39f : __uint_as_float(static_cast<uint32_t>(s) << 23);
}

// Compute dot product of one 32-element MXFP4 block (16 packed bytes = uint4)
// with 16 half2 activation pairs in shared memory, scaled by E8M0 scale -> float.
__device__ __forceinline__ float dot_mxfp4_block32_h2(
    const uint4 w_vec,
    const __half2* __restrict__ sx16,
    const uint8_t scale_e8m0
) {
    __half2 h_acc0 = __float2half2_rn(0.0f);
    __half2 h_acc1 = __float2half2_rn(0.0f);
    const uint32_t words[4] = {w_vec.x, w_vec.y, w_vec.z, w_vec.w};
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        uint32_t u = words[w];
        __half2 d0 = cvt_e2m1x2_to_half2(u);
        __half2 d1 = cvt_e2m1x2_to_half2(u >> 8);
        __half2 d2 = cvt_e2m1x2_to_half2(u >> 16);
        __half2 d3 = cvt_e2m1x2_to_half2(u >> 24);
        int idx = w * 4;
        h_acc0 = __hfma2(d0, sx16[idx + 0], h_acc0);
        h_acc1 = __hfma2(d1, sx16[idx + 1], h_acc1);
        h_acc0 = __hfma2(d2, sx16[idx + 2], h_acc0);
        h_acc1 = __hfma2(d3, sx16[idx + 3], h_acc1);
    }
    float2 f0 = __half22float2(h_acc0);
    float2 f1 = __half22float2(h_acc1);
    return (f0.x + f0.y + f1.x + f1.y) * e8m0_to_float(scale_e8m0);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffffu, val, offset);
    }
    return val;
}

__device__ __forceinline__ void load_bf16_to_smem_h2(
    const __nv_bfloat16* __restrict__ src,
    __half2* __restrict__ dst_h2,
    int K_DIM,
    int tid,
    int nthreads
) {
    // Each thread loads 8 bf16 (= uint4) -> writes 4 half2 (= 16 bytes)
    for (int i = tid * 8; i < K_DIM; i += nthreads * 8) {
        uint4 raw = __ldg(reinterpret_cast<const uint4*>(src + i));
        const uint32_t ws[4] = {raw.x, raw.y, raw.z, raw.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&ws[j]);
            float2 f = __bfloat1622float2(b);
            dst_h2[(i >> 1) + j] = __float22half2_rn(f);
        }
    }
}

// ============================================================================
// Kernel 1: Fused W13 MXFP4 GEMV + SiTU-and-Mul (K=2048 -> 2 chunks/lane prefetched!)
// ============================================================================
template <int K_DIM, int ROWS_PER_WARP>
__global__ void w13_situ_mxfp4_decode_kernel(
    const __nv_bfloat16* __restrict__ x,          // [M, K]
    const uint8_t* __restrict__ w13,              // [E_local, 2*N, K/2]
    const uint8_t* __restrict__ w13_scale,        // [E_local, 2*N, K/32]
    const int32_t* __restrict__ topk_ids,         // [M * top_k]
    const int32_t* __restrict__ expert_map,       // [E_global] or nullptr
    __nv_bfloat16* __restrict__ act_out,          // [M * top_k, N]
    int N,
    int top_k,
    float inv_beta,
    float beta,
    float inv_linear_beta,
    float linear_beta,
    bool has_linear_beta
) {
    const int pair_idx = blockIdx.y;
    const int global_e = topk_ids[pair_idx];
    const int local_e = (expert_map != nullptr) ? expert_map[global_e] : global_e;
    if (local_e < 0) return;

    const int tok_idx = pair_idx / top_k;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    __shared__ __half2 sx[K_DIM / 2];  // 4 KB for K=2048 -> high SM occupancy!
    load_bf16_to_smem_h2(x + static_cast<int64_t>(tok_idx) * K_DIM, sx, K_DIM, tid, blockDim.x);
    __syncthreads();

    constexpr int NUM_BLKS = K_DIM / 32;  // 64 -> 2 chunks per lane
    constexpr int K_HALF = K_DIM / 2;
    const int64_t expert_w_off = static_cast<int64_t>(local_e) * (2 * N) * K_HALF;
    const int64_t expert_s_off = static_cast<int64_t>(local_e) * (2 * N) * NUM_BLKS;

    const int base_row = (blockIdx.x * 4 + warp_id) * ROWS_PER_WARP;
    __nv_bfloat16* out_row = act_out + static_cast<int64_t>(pair_idx) * N;

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int row = base_row + r;
        if (row >= N) break;

        const uint8_t* wg_ptr = w13 + expert_w_off + static_cast<int64_t>(row) * K_HALF;
        const uint8_t* sg_ptr = w13_scale + expert_s_off + static_cast<int64_t>(row) * NUM_BLKS;
        const uint8_t* wu_ptr = w13 + expert_w_off + static_cast<int64_t>(row + N) * K_HALF;
        const uint8_t* su_ptr = w13_scale + expert_s_off + static_cast<int64_t>(row + N) * NUM_BLKS;

        // Issue ALL 4 uint4 + 4 scale loads up front to hide LPDDR5X latency!
        const int b0 = lane;
        const int b1 = lane + 32;
        const uint4 wg0 = __ldg(reinterpret_cast<const uint4*>(wg_ptr + b0 * 16));
        const uint4 wg1 = __ldg(reinterpret_cast<const uint4*>(wg_ptr + b1 * 16));
        const uint4 wu0 = __ldg(reinterpret_cast<const uint4*>(wu_ptr + b0 * 16));
        const uint4 wu1 = __ldg(reinterpret_cast<const uint4*>(wu_ptr + b1 * 16));
        const uint8_t sg0 = __ldg(sg_ptr + b0);
        const uint8_t sg1 = __ldg(sg_ptr + b1);
        const uint8_t su0 = __ldg(su_ptr + b0);
        const uint8_t su1 = __ldg(su_ptr + b1);

        const __half2* sx0 = sx + b0 * 16;
        const __half2* sx1 = sx + b1 * 16;

        float g_acc = dot_mxfp4_block32_h2(wg0, sx0, sg0) + dot_mxfp4_block32_h2(wg1, sx1, sg1);
        float u_acc = dot_mxfp4_block32_h2(wu0, sx0, su0) + dot_mxfp4_block32_h2(wu1, sx1, su1);

        g_acc = warp_reduce_sum(g_acc);
        u_acc = warp_reduce_sum(u_acc);

        if (lane == 0) {
            float sig = 1.0f / (1.0f + expf(-g_acc));
            float g_act = beta * tanhf(g_acc * inv_beta) * sig;
            float u_act = has_linear_beta ? (linear_beta * tanhf(u_acc * inv_linear_beta)) : u_acc;
            out_row[row] = __float2bfloat16(g_act * u_act);
        }
    }
}

// ============================================================================
// Kernel 2A: Fused W2 MXFP4 GEMV + TopK Weighted Reduce (Zero-Atomic)
// ============================================================================
template <int N_DIM, int ROWS_PER_WARP>
__global__ void w2_reduce_mxfp4_decode_kernel(
    const __nv_bfloat16* __restrict__ act_in,     // [M * top_k, N]
    const uint8_t* __restrict__ w2,               // [E_local, H, N/2]
    const uint8_t* __restrict__ w2_scale,         // [E_local, H, N/32]
    const int32_t* __restrict__ topk_ids,         // [M, top_k]
    const float* __restrict__ topk_weights,       // [M, top_k]
    const int32_t* __restrict__ expert_map,       // [E_global] or nullptr
    __nv_bfloat16* __restrict__ out,              // [M, H]
    int H,
    int top_k
) {
    const int tok_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int base_row = (blockIdx.x * 4 + warp_id) * ROWS_PER_WARP;

    __shared__ __half2 sx[N_DIM / 2];  // 4 KB
    float out_acc[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) out_acc[r] = 0.0f;

    constexpr int NUM_BLKS = N_DIM / 32;
    constexpr int N_HALF = N_DIM / 2;
    const int b0 = lane;
    const int b1 = lane + 32;
    const __half2* sx0 = sx + b0 * 16;
    const __half2* sx1 = sx + b1 * 16;

    for (int k = 0; k < top_k; ++k) {
        const int pair_idx = tok_idx * top_k + k;
        const int global_e = topk_ids[pair_idx];
        const int local_e = (expert_map != nullptr) ? expert_map[global_e] : global_e;
        if (local_e < 0) continue;

        const float weight = topk_weights[pair_idx];
        load_bf16_to_smem_h2(act_in + static_cast<int64_t>(pair_idx) * N_DIM, sx, N_DIM, tid, blockDim.x);
        __syncthreads();

        const int64_t expert_w_off = static_cast<int64_t>(local_e) * H * N_HALF;
        const int64_t expert_s_off = static_cast<int64_t>(local_e) * H * NUM_BLKS;

        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const int row = base_row + r;
            if (row >= H) break;
            const uint8_t* w_ptr = w2 + expert_w_off + static_cast<int64_t>(row) * N_HALF;
            const uint8_t* s_ptr = w2_scale + expert_s_off + static_cast<int64_t>(row) * NUM_BLKS;
            const uint4 wv0 = __ldg(reinterpret_cast<const uint4*>(w_ptr + b0 * 16));
            const uint4 wv1 = __ldg(reinterpret_cast<const uint4*>(w_ptr + b1 * 16));
            const uint8_t sv0 = __ldg(s_ptr + b0);
            const uint8_t sv1 = __ldg(s_ptr + b1);
            float acc = dot_mxfp4_block32_h2(wv0, sx0, sv0) + dot_mxfp4_block32_h2(wv1, sx1, sv1);
            acc = warp_reduce_sum(acc);
            if (lane == 0) {
                out_acc[r] = fmaf(weight, acc, out_acc[r]);
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
        __nv_bfloat16* o_row = out + static_cast<int64_t>(tok_idx) * H;
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const int row = base_row + r;
            if (row < H) {
                o_row[row] = __float2bfloat16(out_acc[r]);
            }
        }
    }
}

// ============================================================================
// Kernel 2B: Parallel-Pair W2 MXFP4 GEMV + L2 Atomic FP32 Reduce
// ============================================================================
template <int N_DIM, int ROWS_PER_WARP>
__global__ void w2_atomic_mxfp4_decode_kernel(
    const __nv_bfloat16* __restrict__ act_in,
    const uint8_t* __restrict__ w2,
    const uint8_t* __restrict__ w2_scale,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ expert_map,
    float* __restrict__ out_fp32,
    int H,
    int top_k
) {
    const int pair_idx = blockIdx.y;
    const int global_e = topk_ids[pair_idx];
    const int local_e = (expert_map != nullptr) ? expert_map[global_e] : global_e;
    if (local_e < 0) return;

    const int tok_idx = pair_idx / top_k;
    const float weight = topk_weights[pair_idx];
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    __shared__ __half2 sx[N_DIM / 2];
    load_bf16_to_smem_h2(act_in + static_cast<int64_t>(pair_idx) * N_DIM, sx, N_DIM, tid, blockDim.x);
    __syncthreads();

    constexpr int NUM_BLKS = N_DIM / 32;
    constexpr int N_HALF = N_DIM / 2;
    const int64_t expert_w_off = static_cast<int64_t>(local_e) * H * N_HALF;
    const int64_t expert_s_off = static_cast<int64_t>(local_e) * H * NUM_BLKS;
    const int base_row = (blockIdx.x * 4 + warp_id) * ROWS_PER_WARP;
    float* o_row = out_fp32 + static_cast<int64_t>(tok_idx) * H;

    const int b0 = lane;
    const int b1 = lane + 32;
    const __half2* sx0 = sx + b0 * 16;
    const __half2* sx1 = sx + b1 * 16;

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int row = base_row + r;
        if (row >= H) break;
        const uint8_t* w_ptr = w2 + expert_w_off + static_cast<int64_t>(row) * N_HALF;
        const uint8_t* s_ptr = w2_scale + expert_s_off + static_cast<int64_t>(row) * NUM_BLKS;
        const uint4 wv0 = __ldg(reinterpret_cast<const uint4*>(w_ptr + b0 * 16));
        const uint4 wv1 = __ldg(reinterpret_cast<const uint4*>(w_ptr + b1 * 16));
        const uint8_t sv0 = __ldg(s_ptr + b0);
        const uint8_t sv1 = __ldg(s_ptr + b1);
        float acc = dot_mxfp4_block32_h2(wv0, sx0, sv0) + dot_mxfp4_block32_h2(wv1, sx1, sv1);
        acc = warp_reduce_sum(acc);
        if (lane == 0) {
            atomicAdd(o_row + row, weight * acc);
        }
    }
}

torch::Tensor mxfp4_moe_decode_sm121(
    torch::Tensor x,
    torch::Tensor w13,
    torch::Tensor w13_scale,
    torch::Tensor w2,
    torch::Tensor w2_scale,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    c10::optional<torch::Tensor> expert_map,
    double beta,
    double linear_beta,
    int rows_per_warp_w1,
    int rows_per_warp_w2,
    bool use_atomic_w2
) {
    const int M = x.size(0);
    const int H = x.size(1);
    const int N = w13.size(1) / 2;
    const int top_k = topk_ids.size(1);
    const int num_pairs = M * top_k;

    TORCH_CHECK(H == 2048 && N == 2048, "Specialized for ERNIE 5.5 H=2048, N=2048");
    const int32_t* emap_ptr = expert_map.has_value() ? expert_map->data_ptr<int32_t>() : nullptr;

    auto act_buf = torch::empty({num_pairs, N}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto launch_w13 = [&](auto rpw_tag) {
        constexpr int RPW = decltype(rpw_tag)::value;
        dim3 grid((N + 4 * RPW - 1) / (4 * RPW), num_pairs);
        w13_situ_mxfp4_decode_kernel<2048, RPW><<<grid, 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            w13.data_ptr<uint8_t>(),
            w13_scale.data_ptr<uint8_t>(),
            topk_ids.data_ptr<int32_t>(),
            emap_ptr,
            reinterpret_cast<__nv_bfloat16*>(act_buf.data_ptr<at::BFloat16>()),
            N, top_k,
            1.0f / static_cast<float>(beta),
            static_cast<float>(beta),
            linear_beta > 0.0 ? (1.0f / static_cast<float>(linear_beta)) : 0.0f,
            static_cast<float>(linear_beta),
            linear_beta > 0.0
        );
    };
    if (rows_per_warp_w1 == 1) launch_w13(std::integral_constant<int, 1>{});
    else if (rows_per_warp_w1 == 2) launch_w13(std::integral_constant<int, 2>{});
    else if (rows_per_warp_w1 == 4) launch_w13(std::integral_constant<int, 4>{});
    else launch_w13(std::integral_constant<int, 8>{});

    if (!use_atomic_w2) {
        auto out = torch::empty({M, H}, x.options());
        auto launch_w2 = [&](auto rpw_tag) {
            constexpr int RPW = decltype(rpw_tag)::value;
            dim3 grid((H + 4 * RPW - 1) / (4 * RPW), M);
            w2_reduce_mxfp4_decode_kernel<2048, RPW><<<grid, 128, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(act_buf.data_ptr<at::BFloat16>()),
                w2.data_ptr<uint8_t>(),
                w2_scale.data_ptr<uint8_t>(),
                topk_ids.data_ptr<int32_t>(),
                topk_weights.data_ptr<float>(),
                emap_ptr,
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
                H, top_k
            );
        };
        if (rows_per_warp_w2 == 1) launch_w2(std::integral_constant<int, 1>{});
        else if (rows_per_warp_w2 == 2) launch_w2(std::integral_constant<int, 2>{});
        else if (rows_per_warp_w2 == 4) launch_w2(std::integral_constant<int, 4>{});
        else launch_w2(std::integral_constant<int, 8>{});
        return out;
    } else {
        auto out_fp32 = torch::zeros({M, H}, x.options().dtype(torch::kFloat32));
        auto launch_w2a = [&](auto rpw_tag) {
            constexpr int RPW = decltype(rpw_tag)::value;
            dim3 grid((H + 4 * RPW - 1) / (4 * RPW), num_pairs);
            w2_atomic_mxfp4_decode_kernel<2048, RPW><<<grid, 128, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(act_buf.data_ptr<at::BFloat16>()),
                w2.data_ptr<uint8_t>(),
                w2_scale.data_ptr<uint8_t>(),
                topk_ids.data_ptr<int32_t>(),
                topk_weights.data_ptr<float>(),
                emap_ptr,
                out_fp32.data_ptr<float>(),
                H, top_k
            );
        };
        if (rows_per_warp_w2 == 1) launch_w2a(std::integral_constant<int, 1>{});
        else if (rows_per_warp_w2 == 2) launch_w2a(std::integral_constant<int, 2>{});
        else if (rows_per_warp_w2 == 4) launch_w2a(std::integral_constant<int, 4>{});
        else launch_w2a(std::integral_constant<int, 8>{});
        return out_fp32.to(torch::kBFloat16);
    }
}
"""

_CPP_SRC = """
#include <torch/extension.h>
torch::Tensor mxfp4_moe_decode_sm121(
    torch::Tensor x,
    torch::Tensor w13,
    torch::Tensor w13_scale,
    torch::Tensor w2,
    torch::Tensor w2_scale,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    c10::optional<torch::Tensor> expert_map,
    double beta,
    double linear_beta,
    int rows_per_warp_w1,
    int rows_per_warp_w2,
    bool use_atomic_w2
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
        name="sm121_mxfp4_moe_decode_v2_ext",
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        functions=["mxfp4_moe_decode_sm121"],
        extra_cuda_cflags=["-O3", "--use_fast_math", arch_flag],
        verbose=False,
    )


def mxfp4_moe_decode_sm121(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_map: torch.Tensor | None = None,
    beta: float = 4.0,
    linear_beta: float | None = 25.0,
    rows_per_warp_w1: int = 2,
    rows_per_warp_w2: int = 4,
    use_atomic_w2: bool = False,
) -> torch.Tensor:
    """Fused 2-kernel SM12.1 MXFP4 MoE decode for ERNIE 5.5 (H=2048, N=2048)."""
    ext = _load_ext()
    return ext.mxfp4_moe_decode_sm121(
        x.contiguous(),
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_ids.contiguous(),
        topk_weights.contiguous(),
        expert_map,
        float(beta),
        float(linear_beta) if linear_beta is not None else -1.0,
        int(rows_per_warp_w1),
        int(rows_per_warp_w2),
        bool(use_atomic_w2),
    )

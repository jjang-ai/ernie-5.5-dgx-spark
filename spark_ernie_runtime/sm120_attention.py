"""Sink-preserving ERNIE bindings for vLLM 0.32 / FlashInfer 0.7 on SM12x.

Keep dependency files untouched. The existing vLLM HCA SM120 implementation
owns cache layout, sparse metadata, prefill, and decode. Only ERNIE's routing
and the DSA sink delivery need repair.
"""
from __future__ import annotations

import torch

from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import DeepseekV4FlashInferSM120Attention
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import FlashInferMLASparseSM120Backend
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import FlashInferMLASparseSM120Impl
from vllm_ernie.models.ernie5_5.nvidia import attention as ernie

from kernels.mhc_hca.sm120_attn_dispatch import dsa_sparse_mla_sm121


class ErnieSM120HCAAttention(ernie.Ernie5_5HCAAttentionMixin, DeepseekV4FlashInferSM120Attention):
    """ERNIE gates/VHA with vLLM's packed-FP8, sink-aware SM120 HCA."""


class ErnieSM120DSAImpl(FlashInferMLASparseSM120Impl):
    def __init__(self, *args, sinks: torch.Tensor, **kwargs):
        self.sinks = sinks
        super().__init__(*args, **kwargs)

    def _run_mqa_kernel(self, q, kv_cache, topk_indices_physical):
        if q.shape[1] == 64:
            return dsa_sparse_mla_sm121(q, kv_cache, topk_indices_physical, self.sinks, self.scale)
        # The calibrated PREFILL_SWAPAB specialization is H=64 only.
        # Let FlashInfer's planner select a supported kernel for TP=2's H=32.
        from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import _get_workspace_buffer
        from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_with_kv_cache_mla
        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)
        output = q.new_empty((q.shape[0], self.num_heads, self.kv_lora_rank))
        capacity = topk_indices_physical.shape[1]
        return flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1), kv_cache=kv_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim, kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=topk_indices_physical.unsqueeze(1), seq_lens=None,
            max_seq_len=capacity, out=output.unsqueeze(1), bmm1_scale=self.scale,
            bmm2_scale=1.0, sparse_mla_top_k=capacity,
            kv_scale_format=self.kv_scale_format, sinks=self.sinks,
        ).squeeze(1)


class ErnieSM120DSABackend(FlashInferMLASparseSM120Backend):
    # Retain ERNIE's packed-page-aware block sizes.
    get_supported_kernel_block_sizes = staticmethod(ernie._dsa_kernel_block_sizes)

    @staticmethod
    def get_impl_cls():
        return ErnieSM120DSAImpl

    @classmethod
    def supports_sink(cls):
        return True


def install() -> bool:
    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 12:
        return False
    # Existing selector/constructor globals resolve these names at call time.
    # Their validation remains enabled with has_sink=True and the real SM12.1.
    ernie.Ernie5_5HCAFlashMLAAttention = ErnieSM120HCAAttention
    ernie.Ernie5_5FlashMLASparseBackend = ErnieSM120DSABackend
    return True

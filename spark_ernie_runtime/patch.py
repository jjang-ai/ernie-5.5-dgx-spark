"""Non-destructive SM12.1 (DGX Spark GB10) runtime overlay for vLLM 0.32.0 + vLLM-ERNIE.

Every optimization can be toggled independently via env vars:
  SPARK_ERNIE_RUNTIME=1            Master switch (required by sitecustomize.py)
  SPARK_PATCH_FLASHINFER_CPB=1     Fix GB10 FlashInfer SM120 dsv4 & dsv3_2 crossover thresholds
  SPARK_PATCH_ROUTER=1             Bit-exact single-kernel fused ERNIE 5.5 MoE router
  SPARK_PATCH_MHC=1                GB10 L2-aware mHC post+pre+RMSNorm kernels (T=16..32, T>=512)
  SPARK_PATCH_HUMMING_FP8_SAFE=1   Safe fallback for non-128-divisible FP8 linears (enables linear_backend=humming!)
  SPARK_REPLICATE_SHARED_EXPERTS=0 Replicate shared_experts (disable_tp=True + skip 42 shared ARs/step)
  SPARK_PATCH_DSPARK_MARKOV=1      Wire apply_markov_bias_gathered() + dspark_draft_topk (cuts 99% Markov GEMMs)
  SPARK_DSPARK_TOPK=64             Top-k candidate count for gathered sequential Markov head (0=disable)
  SPARK_PATCH_ADAPTIVE_VERIFY=1    Enable SM120 varlen-CG declaration for DSpark adaptive verification (confidence_head)
  SPARK_USE_SPARKNET=0             Enable sparknet one-shot RoCE collectives (10.9 us 2x / 16.9 us 4x ring vs 70-92 us NCCL!)
  SPARK_DSPARK_MAIN_PROJ_TP=1      Column-shard DSpark main_proj across TP ranks (cuts draft main_proj read 2x-4x)
  SPARK_DSPARK_VOCAB_PARALLEL=0    Full-vocab TP-sharded greedy Markov drafting (0% acceptance loss, 2x-4x less Markov BW, no full-vocab gather)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_INSTALLED = False
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _log(msg: str) -> None:
    print(f"[spark_ernie_runtime] {msg}", file=sys.stderr, flush=True)


def _patch_flashinfer_sm120_cpb() -> bool:
    """Fix GB10 (48 SMs, 24 MiB L2) FlashInfer SM120 crossover thresholds in-memory."""
    try:
        import flashinfer.mla._sparse_mla_sm120_cpb as _cpb
    except Exception as exc:
        _log(f"skip flashinfer_cpb patch (import failed: {exc!r})")
        return False

    gb10_overrides = {
        "dsv4|16|128": 8,
        "dsv4|32|128": 8,
        "dsv4|64|128": 4,
        "dsv3_2|16|2048": 8,
        "dsv3_2|32|2048": 8,
        "dsv3_2|64|2048": 8,
        "dsv3_2|16|2176": 8,
        "dsv3_2|32|2176": 8,
        "dsv3_2|64|2176": 8,
    }

    if hasattr(_cpb, "get_decode_max_tokens"):
        orig_get = _cpb.get_decode_max_tokens

        def _patched_get(device, family, num_heads, topk):
            key = f"{family}|{num_heads}|{topk}"
            if key in gb10_overrides:
                return gb10_overrides[key]
            return orig_get(device, family, num_heads, topk)

        _cpb.get_decode_max_tokens = _patched_get
        _log("applied flashinfer_sm120_cpb.get_decode_max_tokens GB10 overrides")
        return True

    return False


def _patch_fused_ernie55_router() -> bool:
    """Replace 2-kernel split_sigmoid_sum + ernie5_5_routing with 1 bit-exact Triton kernel."""
    try:
        import torch
        import vllm_ernie.models.ernie5_5.nvidia.moe as emoe
        from kernels.moe.fused_ernie55_router import fused_ernie5_5_routing
    except Exception as exc:
        _log(f"skip fused_router patch ({exc!r})")
        return False

    orig_split = emoe.split_sigmoid_sum
    orig_route = emoe.ernie5_5_routing

    def _passthrough_split(logits, num_experts: int):
        if (
            logits.is_cuda
            and logits.dtype == torch.float32
            and logits.stride(1) == 1
            and logits.shape[-1] == 2 * num_experts
            and (num_experts & (num_experts - 1)) == 0
        ):
            return logits
        return orig_split(logits, num_experts)

    def _fused_route(hidden_states, gating_output, topk: int, renormalize: bool, *, bias, scale):
        num_experts = bias.shape[0]
        if gating_output.shape[-1] == 2 * num_experts:
            is_padding = emoe._padding_mask(gating_output.shape[0])
            return fused_ernie5_5_routing(
                gating_output,
                topk=topk,
                renormalize=renormalize,
                bias=bias,
                scale=scale,
                is_padding=is_padding,
                use_fast_argmax=True,
                num_warps=1,
            )
        return orig_route(hidden_states, gating_output, topk, renormalize, bias=bias, scale=scale)

    emoe.split_sigmoid_sum = _passthrough_split
    emoe.ernie5_5_routing = _fused_route
    _log("applied bit-exact fused_ernie5_5_routing (2 Triton kernels -> 1)")
    return True


def _patch_mhc_gb10() -> bool:
    """Patch mhc_fused_post_pre_tilelang to use GB10 L2-aware kernels at T=16..32 & T>=512."""
    try:
        import vllm_ernie.models.ernie5_5.mhc as emhc
        import vllm_ernie.models.ernie5_5.nvidia.model as emodel
        from kernels.mhc_hca.mhc_sm121 import mhc_fused_post_pre_sm121
    except Exception as exc:
        _log(f"skip mhc_gb10 patch ({exc!r})")
        return False

    orig_fn = emhc.mhc_fused_post_pre_tilelang

    def _gb10_mhc_fused(
        x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
        n_splits=1, tile_n=1, norm_weight=None, norm_eps=1e-6,
    ):
        T = residual.numel() // (residual.shape[-2] * residual.shape[-1])
        hc_mult = residual.shape[-2]
        hidden_size = residual.shape[-1]
        if (
            norm_weight is not None
            and hc_mult == 4
            and hidden_size == 4096
            and (1 <= T <= 32 or T >= 512)
        ):
            mode = "twopass" if T <= 32 else "dg_chunked"
            return mhc_fused_post_pre_sm121(
                x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
                rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                norm_weight=norm_weight, norm_eps=norm_eps, mode=mode,
            )
        return orig_fn(
            x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
            rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
            n_splits=n_splits, tile_n=tile_n, norm_weight=norm_weight, norm_eps=norm_eps,
        )

    emhc.mhc_fused_post_pre_tilelang = _gb10_mhc_fused
    if hasattr(emodel, "mhc_fused_post_pre_tilelang"):
        emodel.mhc_fused_post_pre_tilelang = _gb10_mhc_fused
    _log("applied GB10 L2-aware mhc_fused_post_pre_sm121 (T=1..32 twopass -2.9ms/step, T>=512 chunked)")
    return True


def _patch_humming_fp8_safe() -> bool:
    """Allow linear_backend='humming' on ERNIE 5.5 by falling back cleanly on non-128-multiple FP8 linears."""
    try:
        from vllm.model_executor.kernels.linear.scaled_mm import humming as hlin
        from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
            DeepGemmFp8BlockScaledMMKernel,
        )
    except Exception as exc:
        _log(f"skip humming_fp8_safe patch ({exc!r})")
        return False

    cls = hlin.HummingFP8ScaledMMLinearKernel
    orig_proc = cls.process_weights_after_loading
    orig_apply = cls.apply_weights

    def _safe_process(self, layer):
        gs = self.config.weight_quant_key.scale.group_shape
        w = getattr(layer, "weight", None)
        ws = getattr(layer, "weight_scale_inv", getattr(layer, "weight_scale", None))
        part_sizes = getattr(layer, "output_partition_sizes", None) or []
        prefix = getattr(layer, "prefix", "") or ""
        need_fallback = bool(getattr(layer, "is_bmm", False)) or prefix.endswith(
            (".wo_a", ".kv_b_proj", ".fused_wkv_wgate")
        )
        if not need_fallback and w is not None and gs.row > 0 and gs.col > 0:
            if (
                w.shape[0] % gs.row != 0
                or w.shape[1] % gs.col != 0
                or any(sz % gs.row != 0 for sz in part_sizes)
                or (ws is not None and tuple(ws.shape) != (w.shape[0] // gs.row, w.shape[1] // gs.col))
            ):
                need_fallback = True
        if need_fallback:
            fb = DeepGemmFp8BlockScaledMMKernel(self.config)
            fb.process_weights_after_loading(layer)
            layer._spark_fallback = fb
            return
        layer._spark_fallback = None
        orig_proc(self, layer)

    def _safe_apply(self, layer, x, bias=None):
        fb = getattr(layer, "_spark_fallback", None)
        if fb is not None:
            return fb.apply_weights(layer, x, bias)
        return orig_apply(self, layer, x, bias)

    cls.process_weights_after_loading = _safe_process
    cls.apply_weights = _safe_apply
    _log("applied HummingFP8 safe fallback for non-128-divisible FP8 linears (222 GB/s on all aligned linears)")
    return True


def _patch_replicate_shared_experts() -> bool:
    """Replicate shared_experts across TP ranks (+0.53 GB/GPU) and skip all 42 shared ARs/step."""
    try:
        import vllm_ernie.models.ernie5_5.nvidia.moe as emoe
        from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr
        from vllm.model_executor.layers.linear import (
            MergedColumnParallelLinear,
            RowParallelLinear,
        )
    except Exception as exc:
        _log(f"skip replicate_shared_experts ({exc!r})")
        return False

    orig_init = emoe.Ernie5_5MLP.__init__

    def _patched_mlp_init(self, config, intermediate_size, quant_config=None, reduce_results=True, prefix=""):
        orig_init(self, config, intermediate_size, quant_config=quant_config, reduce_results=reduce_results, prefix=prefix)
        if ".shared_experts" in prefix:
            self.gate_up_proj = MergedColumnParallelLinear(
                config.hidden_size,
                [intermediate_size] * 2,
                bias=False,
                quant_config=quant_config,
                disable_tp=True,
                prefix=f"{prefix}.gate_up_proj",
            )
            self.down_proj = RowParallelLinear(
                intermediate_size,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                reduce_results=False,
                disable_tp=True,
                prefix=f"{prefix}.down_proj",
            )
            self._spark_replicated = True

    emoe.Ernie5_5MLP.__init__ = _patched_mlp_init

    # Bypass the redundant TP all-reduce in MoERunner._maybe_reduce_shared_expert_output
    orig_ar_shared = mr.MoERunner._maybe_reduce_shared_expert_output

    def _patched_ar_shared(self, shared_output, fused_output_is_reduced=None):
        se = getattr(self, "_shared_experts", None)
        inner = getattr(se, "shared_experts", None) if se is not None else None
        if inner is not None and getattr(inner, "_spark_replicated", False):
            return shared_output
        return orig_ar_shared(self, shared_output, fused_output_is_reduced=fused_output_is_reduced)

    mr.MoERunner._maybe_reduce_shared_expert_output = _patched_ar_shared
    _log("applied true shared_experts replication (disable_tp=True + bypass 42 shared ARs/step)")
    return True


def _patch_dspark_markov_gathered() -> bool:
    """Wire apply_markov_bias_gathered() + dspark_draft_topk=64 to cut 99% of sequential Markov GEMMs."""
    try:
        import vllm_ernie.models.ernie5_5.nvidia.dspark as edspark
        from vllm.v1.worker.gpu.spec_decode.dspark import speculator as dspec
    except Exception as exc:
        _log(f"skip dspark_markov_gathered ({exc!r})")
        return False

    cls = edspark.Ernie5_5DSparkForCausalLM
    if not hasattr(cls, "apply_markov_bias_gathered"):
        import torch

        def apply_markov_bias_gathered(
            self, markov_embed, logits, values, index, scale: float = 1.0
        ):
            mh = self.model.markov_head
            weight = mh.markov_w2.weight[index]
            if weight.dtype != markov_embed.dtype:
                weight = weight.to(markov_embed.dtype)
            bias = torch.bmm(weight, markov_embed.unsqueeze(-1)).squeeze(-1)
            if scale != 1.0:
                bias = bias * scale
            corrected = values + bias.to(values.dtype)
            return logits.scatter_(1, index, corrected.to(logits.dtype))

        cls.apply_markov_bias_gathered = apply_markov_bias_gathered

    topk_val = int(os.environ.get("SPARK_DSPARK_TOPK", "64"))
    if topk_val > 0:
        orig_spec_init = dspec.DSparkSpeculator.__init__

        def _patched_spec_init(self, *args, **kwargs):
            orig_spec_init(self, *args, **kwargs)
            if self._draft_topk is None:
                self._draft_topk = topk_val
                _log(f"enabled DSpark gathered Markov top-k={topk_val} (skips 99% of 7x[256,201216] GEMMs)")

        dspec.DSparkSpeculator.__init__ = _patched_spec_init

    _log(f"applied Ernie5_5DSparkForCausalLM.apply_markov_bias_gathered() (SPARK_DSPARK_TOPK={topk_val})")
    return True


def _patch_dspark_adaptive_verification() -> bool:
    """Unlock SM12.1 DSpark adaptive verification + Dinkelbach marginal-rate rule (0006).

    1. Exposes SM120 varlen-CG bound & supports_sink on FlashInferMLASparseSM120.
    2. Enables flattened device query_lens in DeepseekV32/Ernie5_5 Indexer on SM12.x (GB10)
       (upstream only checked SM90 despite SM12.x already using flattened decode).
    3. Adds Dinkelbach marginal-rate verification rule (`SPARK_DSPARK_VERIFY_RULE=marginal`)
       to AdaptiveVerificationManager so high-confidence code steps verify all 7 tokens while
       low-confidence prose/batched steps stop early and skip wasted MoE DRAM reads.
    """
    try:
        import numpy as np
        from vllm.platforms import current_platform
        from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fms
        from vllm.v1.attention.backends.mla import indexer as mla_idx
        from vllm.v1.worker.gpu.spec_decode import adaptive_verification as av
    except Exception as exc:
        _log(f"skip adaptive_verify patch ({exc!r})")
        return False

    builder_cls = fms.FlashInferMLASparseMetadataBuilder

    @classmethod
    def _get_varlen_cudagraph_max_query_len(cls, vllm_config, kv_cache_spec):
        # FlashInferMLASparseSM120Impl._run_mqa_kernel executes token-by-token
        # (query.unsqueeze(1), seq_lens=None) and is agnostic to per-request query_len.
        return fms.max_decode_query_len(vllm_config)

    builder_cls.get_varlen_cudagraph_max_query_len = _get_varlen_cudagraph_max_query_len
    fms.FlashInferMLASparseSM120Backend.supports_sink = classmethod(lambda cls: True)

    # Enable flattened device query_lens on SM12.x (GB10) as well as SM90
    orig_flat = mla_idx._supports_flattened_device_query_lens

    def _supports_flattened_sm120() -> bool:
        if orig_flat():
            return True
        return (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
            and mla_idx.has_deep_gemm()
        )

    mla_idx._supports_flattened_device_query_lens = _supports_flattened_sm120

    # Dinkelbach marginal-rate rule (from spark-ds41f 0006) + true MoE coupon-collector cost curve!
    verify_rule = os.environ.get("SPARK_DSPARK_VERIFY_RULE", "marginal")
    rate_alpha = float(os.environ.get("SPARK_DSPARK_RATE_ALPHA", "0.1"))
    use_moe_curve = _env_flag("SPARK_DSPARK_MOE_COST_CURVE", "1")
    if verify_rule == "marginal":
        orig_av_init = av.AdaptiveVerificationManager.__init__
        orig_set_curves = av.AdaptiveVerificationManager.set_initial_cost_curves
        orig_get_num_tokens = av.AdaptiveVerificationManager.get_num_tokens

        def _patched_av_init(self, *args, **kwargs):
            orig_av_init(self, *args, **kwargs)
            self._spark_rates = {}

        def _patched_set_curves(self, timings):
            ret = orig_set_curves(self, timings)
            if use_moe_curve and self.cost_tables is not None:
                try:
                    from vllm.distributed import get_tensor_model_parallel_world_size

                    tp = max(1, get_tensor_model_parallel_world_size())
                    dc, vc = self.cost_tables
                    if len(vc) > 2:
                        # Startup dummy_run uses identical tokens -> all M tokens hit the same
                        # 10 experts, giving a fake flat verify_ms[1..8] = [22.8..25.9].
                        # In reality ERNIE 5.5 (E=512, k=10, 42 MoE layers, 7.08 MB/expert)
                        # activates E_unique(M) = E * (1 - (1 - k/E)^M) unique experts!
                        E = 512.0
                        k = 10.0
                        # Effective GB10 LPDDR5X MoE bandwidth (GB/s); tuned so B=1..16
                        # prose trims wasted tail drafts while B=32 keeps full batching.
                        bw_gb_s = float(os.environ.get("SPARK_DSPARK_MOE_BW_GBS", "250.0"))
                        ms_per_extra_expert = (42.0 * 7.08 / float(tp)) / bw_gb_s
                        base_v1 = float(vc[1])
                        vc_new = np.array(vc, dtype=np.float64, copy=True)
                        cap_m = min(len(vc_new), 65)
                        for m in range(2, cap_m):
                            e_uniq = E * (1.0 - ((E - k) / E) ** m)
                            moe_est = base_v1 + max(0.0, e_uniq - k) * ms_per_extra_expert
                            if moe_est > vc_new[m]:
                                vc_new[m] = moe_est
                        self.cost_tables = (dc, vc_new)
                        _log(
                            f"calibrated true ERNIE 5.5 512-expert MoE verify cost curve (TP={tp}): "
                            f"verify_ms[1..8]={[round(float(x), 1) for x in vc_new[1:9]]}"
                        )
                except Exception as exc:
                    _log(f"MoE cost curve calibration skipped ({exc!r})")
            return ret

        def _patched_get_num_tokens(self, num_tokens_per_req, draft_tokens):
            if self.cost_tables is None:
                return orig_get_num_tokens(self, num_tokens_per_req, draft_tokens)
            # Replicate exact upstream prep up to utility selection, then apply Dinkelbach rule
            req_ids = list(num_tokens_per_req.keys())
            num_reqs = len(req_ids)
            if num_reqs == 0:
                return orig_get_num_tokens(self, num_tokens_per_req, draft_tokens)
            try:
                slots = np.fromiter(
                    (self.req_states.req_id_to_index[rid] for rid in req_ids),
                    dtype=np.int32,
                    count=num_reqs,
                )
                scheduled_drafts = np.fromiter(
                    (len(draft_tokens.get(rid, ())) for rid in req_ids),
                    dtype=np.int32,
                    count=num_reqs,
                )
                num_non_draft_tokens = np.fromiter(
                    (num_tokens_per_req[rid] - int(sd) for rid, sd in zip(req_ids, scheduled_drafts)),
                    dtype=np.int32,
                    count=num_reqs,
                )
                total_scheduled_drafts = int(scheduled_drafts.sum())
                if total_scheduled_drafts == 0:
                    return orig_get_num_tokens(self, num_tokens_per_req, draft_tokens)
                # Let upstream compute its baseline budget & state cleanly, then re-pick budget if marginal differs
                base_ret = orig_get_num_tokens(self, num_tokens_per_req, draft_tokens)
                batch_budget = getattr(self, "_batch_budget", None)
                if batch_budget is None:
                    return base_ret
                num_drafts_per_req, num_non_draft_tokens_per_req, _ = batch_budget
                conf = self._stale_confidences[self._stale_idx].np[slots].astype(np.float64)
                surv = np.cumprod(conf, axis=1)
                mask = np.arange(self.num_speculative_steps)[None, :] < scheduled_drafts[:, None]
                scores = np.sort(surv[mask])[::-1]
                num_non_draft_total = int(num_non_draft_tokens.sum())
                max_draft_budget = min(
                    total_scheduled_drafts,
                    max(0, self._max_total_logits - num_reqs * self.num_bonus_tokens),
                )
                scores = scores[:max_draft_budget]
                draft_cost_ms, verify_cost_ms = self.cost_tables
                num_sampling_reqs = np.count_nonzero(
                    self.req_states.num_computed_tokens_np[slots] + num_non_draft_tokens
                    >= self.req_states.prefill_len.np[slots]
                )
                est_acc = np.concatenate(([num_sampling_reqs], num_sampling_reqs + np.cumsum(scores)))
                costs = (
                    draft_cost_ms[num_reqs]
                    + verify_cost_ms[num_non_draft_total : num_non_draft_total + max_draft_budget + 1]
                )
                if len(costs) == len(est_acc) and len(costs) > 0:
                    plain_rate = float(est_acc[0] / costs[0])
                    rate = max(self._spark_rates.get(num_reqs, plain_rate), plain_rate)
                    utilities = est_acc - rate * costs
                    draft_budget = int(np.argmax(utilities))
                    chosen_rate = float(est_acc[draft_budget] / costs[draft_budget])
                    self._spark_rates[num_reqs] = (1.0 - rate_alpha) * rate + rate_alpha * chosen_rate
                    self._batch_budget = (num_drafts_per_req, num_non_draft_tokens_per_req, draft_budget)
                    if not getattr(self, "_spark_logged_marginal", False):
                        self._spark_logged_marginal = True
                        _log(
                            f"Dinkelbach marginal adaptive verification active "
                            f"(B={num_reqs}, budget={draft_budget}/{max_draft_budget}, "
                            f"rate={chosen_rate:.3f} tok/ms)"
                        )
                    return sum(num_non_draft_tokens_per_req.values()) + draft_budget
                return base_ret
            except Exception as exc:
                if not getattr(self, "_spark_logged_err", False):
                    self._spark_logged_err = True
                    _log(f"Dinkelbach marginal fallback ({exc!r})")
                return orig_get_num_tokens(self, num_tokens_per_req, draft_tokens)

        av.AdaptiveVerificationManager.__init__ = _patched_av_init
        av.AdaptiveVerificationManager.set_initial_cost_curves = _patched_set_curves
        av.AdaptiveVerificationManager.get_num_tokens = _patched_get_num_tokens

    _log(
        f"applied SM12.1 adaptive verification (FlashInferSM120 varlen-CG + Indexer flattened + "
        f"Dinkelbach rule={verify_rule})"
    )
    return True


def _patch_sparknet_oneshot() -> bool:
    """Route multi-node DGX Spark TP collectives through sparknet one-shot RoCEv2 (10.9us 2x / 16.9us 4x)."""
    try:
        from sparknet.integration.vllm import SparknetOneShotAllReduce
        from vllm.distributed.device_communicators import cuda_communicator as cc
        from vllm.distributed.parallel_state import in_the_same_node_as
    except Exception as exc:
        _log(f"skip sparknet_oneshot patch ({exc!r})")
        return False

    cls = cc.CudaCommunicator
    orig_init = cls.__init__
    orig_all_gather = cls.all_gather
    orig_reduce_scatter = getattr(cls, "reduce_scatter", None)

    def _patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        uname = getattr(self, "unique_name", "") or ""
        if (
            self.world_size > 1
            and self.device_group is not None
            and uname.startswith("tp")
            and os.environ.get("SPARKNET_ROCE_TOPOLOGY")
        ):
            same_node = in_the_same_node_as(self.cpu_group, source_rank=0)
            if not all(same_node):
                sn_comm = SparknetOneShotAllReduce(
                    group=self.cpu_group,
                    device_group=self.device_group,
                    device=self.device,
                    global_ranks=self.ranks,
                    nccl_available=(
                        self.pynccl_comm is not None and not self.pynccl_comm.disabled
                    ),
                    in_the_same_node=same_node,
                )
                if not sn_comm.disabled:
                    self.ca_comm = sn_comm
                    self._sparknet_oneshot = sn_comm
                    _log(
                        f"activated SparknetOneShotAllReduce on group '{uname}' "
                        f"(world={self.world_size}, topo={os.environ.get('SPARKNET_ROCE_TOPOLOGY')}, "
                        f"ar_max={sn_comm.all_reduce_max_bytes}, ag_max={sn_comm.all_gather_max_bytes})"
                    )

    def _patched_all_gather(self, input_, dim: int = -1):
        sn = getattr(self, "_sparknet_oneshot", None)
        if sn is not None and not sn.disabled:
            d = dim + input_.dim() if dim < 0 else dim
            if sn.should_all_gather(input_, d):
                return sn.all_gather(input_, d)
        return orig_all_gather(self, input_, dim)

    def _patched_reduce_scatter(self, input_, dim: int = -1):
        sn = getattr(self, "_sparknet_oneshot", None)
        if sn is not None and not sn.disabled:
            d = dim + input_.dim() if dim < 0 else dim
            if sn.should_reduce_scatter(input_, d):
                return sn.reduce_scatter(input_, d)
        return orig_reduce_scatter(self, input_, dim)

    cls.__init__ = _patched_init
    cls.all_gather = _patched_all_gather
    if orig_reduce_scatter is not None:
        cls.reduce_scatter = _patched_reduce_scatter
    _log("installed SparknetOneShotAllReduce hook on CudaCommunicator (TP all_reduce + all_gather + reduce_scatter)")
    return True


def _patch_dspark_main_proj_tp() -> bool:
    """Column-shard DSpark main_proj across TP ranks (gather_output=True) to cut draft read 2x-4x."""
    try:
        import vllm_ernie.models.ernie5_5.nvidia.dspark as edspark
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.model_executor.layers.linear import ColumnParallelLinear
    except Exception as exc:
        _log(f"skip dspark_main_proj_tp ({exc!r})")
        return False

    model_cls = getattr(edspark, "Ernie5_5DSparkModel", None)
    if model_cls is None:
        return False
    orig_init = model_cls.__init__

    def _patched_dspark_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        tp = get_tensor_model_parallel_world_size()
        mp = getattr(self, "main_proj", None)
        if tp > 1 and mp is not None and mp.output_size % (tp * 128) == 0:
            self.main_proj = ColumnParallelLinear(
                mp.input_size,
                mp.output_size,
                bias=False,
                gather_output=True,
                return_bias=False,
                quant_config=getattr(mp, "quant_config", None),
                prefix=getattr(mp, "prefix", "model.main_proj"),
            )
            _log(f"column-sharded DSpark main_proj [{mp.input_size}->{mp.output_size}] across TP={tp}")

    model_cls.__init__ = _patched_dspark_init
    _log("installed DSpark column-parallel main_proj patch")
    return True


def _patch_dspark_vocab_parallel_greedy() -> bool:
    """Full-vocab TP-sharded greedy Markov drafting (0% acceptance loss, 2x-4x less Markov BW, no full-vocab gather)."""
    try:
        import torch
        import torch.nn.functional as F
        import vllm_ernie.models.ernie5_5.nvidia.dspark as edspark
        from vllm.distributed import (
            get_tensor_model_parallel_world_size,
            tensor_model_parallel_all_gather,
        )
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        from vllm.v1.worker.gpu.spec_decode.dspark import speculator as dspec
        from kernels.dspark.vocab_parallel_greedy import (
            greedy_markov_shard_pair,
            scratch_shape,
            select_first_argmax,
        )
    except Exception as exc:
        _log(f"skip dspark_vocab_parallel ({exc!r})")
        return False

    model_cls = getattr(edspark, "Ernie5_5DSparkModel", None)
    causal_cls = getattr(edspark, "Ernie5_5DSparkForCausalLM", None)
    if model_cls is None or causal_cls is None:
        return False

    orig_model_init = model_cls.__init__

    def _patched_model_init(self, *args, **kwargs):
        orig_model_init(self, *args, **kwargs)
        tp = get_tensor_model_parallel_world_size()
        mh = getattr(self, "markov_head", None)
        sample_method = os.environ.get("SPARK_DSPARK_SAMPLE_METHOD", "probabilistic").strip().lower()
        if tp > 1 and mh is not None and hasattr(mh, "markov_w2") and sample_method == "greedy":
            w2 = mh.markov_w2
            mh.markov_w2 = ParallelLMHead(
                w2.org_vocab_size,
                w2.embedding_dim,
                bias=False,
                prefix=getattr(w2, "prefix", "model.markov_head.markov_w2"),
                disable_tp=False,
            )
            self._spark_vocab_parallel = True
            _log(f"vocab-sharded DSpark markov_w2 [{w2.org_vocab_size}, {w2.embedding_dim}] across TP={tp}")

    model_cls.__init__ = _patched_model_init

    def _vp_shard(self) -> tuple[int, int]:
        head = self.lm_head.shard_indices
        markov = self.model.markov_head.markov_w2.shard_indices
        shard = (head.org_vocab_start_index, head.num_org_elements)
        assert shard == (markov.org_vocab_start_index, markov.num_org_elements), (
            f"lm_head/markov_w2 shard mismatch: {head} != {markov}"
        )
        return shard

    def _compute_draft_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.model.norm(hidden_states)
        lm_head = self.lm_head
        w = lm_head.weight
        if x.dtype != w.dtype:
            x = x.to(w.dtype)
        # Reuse lossless 12-bit packed BF16 TileLang LM head when enabled!
        if _env_flag("SPARK_PACKED_BF16_LM_HEAD", "0") and x.is_cuda and x.dtype == torch.bfloat16:
            runner = getattr(lm_head, "_spark_packed_draft_bf16", None)
            if runner is None and not torch.cuda.is_current_stream_capturing():
                if w.is_cuda and w.dtype == torch.bfloat16 and w.shape[0] % 32 == 0 and w.shape[1] % 256 == 0 and w.abs().max().item() > 0:
                    try:
                        from kernels.lm_head.packed_bf16_lm_head import PackedBF16VocabRunner, pack_bf16_exact
                        # Share the already-packed target LM head weight (ZERO extra VRAM!) if available
                        fp32_runner = getattr(lm_head, f"_spark_packed_lm_{torch.float32}", None)
                        pk = fp32_runner.packed if fp32_runner else pack_bf16_exact(w.detach(), verify=False)
                        runner = PackedBF16VocabRunner(pk, out_dtype=torch.bfloat16)
                        lm_head._spark_packed_draft_bf16 = runner
                        _log(f"activated shared 12-bit packed BF16 draft LM head [{w.shape[0]}x{w.shape[1]}] -> bf16 (zero extra VRAM)")
                    except Exception as exc:
                        _log(f"packed draft LM head fallback ({exc!r})")
                        lm_head._spark_packed_draft_bf16 = False
                        runner = False
            if runner and x.numel() // x.shape[-1] <= 256:
                flat = x.reshape(-1, x.shape[-1])
                return runner(flat).reshape(*x.shape[:-1], -1)
        return F.linear(x, w)

    def _markov_bias_local(self, markov_embed: torch.Tensor) -> torch.Tensor:
        mw2 = self.model.markov_head.markov_w2
        w = mw2.weight
        if markov_embed.dtype != w.dtype:
            markov_embed = markov_embed.to(w.dtype)
        if _env_flag("SPARK_PACKED_BF16_MARKOV", "1") and markov_embed.is_cuda and markov_embed.dtype == torch.bfloat16:
            runner = getattr(mw2, "_spark_packed_markov_bf16", None)
            if runner is None and not torch.cuda.is_current_stream_capturing():
                if w.is_cuda and w.dtype == torch.bfloat16 and w.shape[0] % 32 == 0 and w.shape[1] % 256 == 0 and w.abs().max().item() > 0:
                    try:
                        from kernels.lm_head.packed_bf16_lm_head import PackedBF16VocabRunner, pack_bf16_exact
                        pk = pack_bf16_exact(w.detach(), verify=False)
                        runner = PackedBF16VocabRunner(pk, out_dtype=torch.bfloat16)
                        mw2._spark_packed_markov_bf16 = runner
                        _log(f"activated lossless 12-bit packed BF16 markov_w2 [{w.shape[0]}x{w.shape[1]}] (exc={pk.num_exceptions})")
                    except Exception as exc:
                        _log(f"packed markov_w2 fallback ({exc!r})")
                        mw2._spark_packed_markov_bf16 = False
                        runner = False
            if runner and markov_embed.shape[0] <= 64:
                return runner(markov_embed)
        return F.linear(markov_embed, w)

    causal_cls.draft_vocab_shard = property(_vp_shard)
    causal_cls.compute_draft_logits_local = _compute_draft_logits_local
    causal_cls.markov_bias_local = _markov_bias_local

    orig_spec_init = dspec.DSparkSpeculator.__init__
    orig_sample_seq = dspec.DSparkSpeculator._sample_sequential
    vp_prob_enabled = _env_flag("SPARK_DSPARK_VP_PROB", "1")
    draft_tau = float(os.environ.get("SPARK_DSPARK_DRAFT_TAU", "1.0"))
    draft_top_p = float(os.environ.get("SPARK_DSPARK_TOP_P", "0.0"))
    inv_tau = (1.0 / draft_tau) if (draft_tau > 0.0 and draft_tau != 1.0) else 1.0
    topk_total = max(16, int(os.environ.get("SPARK_DSPARK_TOPK", "64")))

    def _patched_spec_init(self, *args, **kwargs):
        orig_spec_init(self, *args, **kwargs)
        if get_tensor_model_parallel_world_size() > 1:
            if self.draft_logits is None:
                # Vocab-parallel greedy scans 100% of vocab across shards
                self._draft_topk = None
                self._vp_scratch_ready = False
            elif vp_prob_enabled:
                # Vocab-parallel probabilistic top-k handles candidate selection across TP shards!
                self._draft_topk = None
                self._vp_prob_ready = False
                _log(
                    f"enabled vocab-parallel probabilistic DSpark drafting "
                    f"(top_k={topk_total}, draft_tau={draft_tau}, top_p={draft_top_p}, "
                    f"1x sparknet AG + 64-row Markov)"
                )

    def _sample_vp_prob(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        tp = get_tensor_model_parallel_world_size()
        head = self.model.lm_head.shard_indices
        vocab_start, shard_vocab = head.org_vocab_start_index, head.num_org_elements
        k_per_rank = max(16, (topk_total + tp - 1) // tp)
        # Keep k_per_rank multiple of 4 so float32 byte payload is >=64B & 16B-aligned for sparknet!
        k_per_rank = ((k_per_rank + 3) // 4) * 4

        if not getattr(self, "_vp_prob_ready", False):
            self._vp_prob_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.draft_logits.dtype,
                device=head_hidden.device,
            )
            self._vp_prob_ready = True

        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        local_logits = self.model.compute_draft_logits_local(sample_hidden)
        valid_shard = min(shard_vocab, max(1, self.vocab_size - vocab_start))
        local_logits = local_logits.view(num_reqs, n_spec, -1)[..., :valid_shard]

        # Local top-k per TP shard + 1 single sparknet all_gather across ALL n_spec positions!
        l_vals, l_idx = local_logits.topk(k_per_rank, dim=-1)
        packed = torch.cat(
            (l_vals.float(), (l_idx + vocab_start).float()), dim=-1
        )  # [num_reqs, n_spec, 2 * k_per_rank]
        gathered = tensor_model_parallel_all_gather(packed, dim=-1)
        chunks = gathered.view(num_reqs, n_spec, tp, 2, k_per_rank)
        all_vals = chunks[:, :, :, 0, :].reshape(num_reqs, n_spec, tp * k_per_rank)
        all_idx = chunks[:, :, :, 1, :].reshape(num_reqs, n_spec, tp * k_per_rank).to(torch.int64)

        if tp * k_per_rank > topk_total:
            top_vals, pick = all_vals.topk(topk_total, dim=-1)
            top_idx = all_idx.gather(-1, pick)
        else:
            top_vals, top_idx = all_vals, all_idx

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        confidence_markov_embeds = []
        buf = self._vp_prob_buf[:num_reqs]
        mh = self.model.model.markov_head
        lm_head = self.model.lm_head
        has_multimax = hasattr(lm_head, "multimax")
        orig_unbound = getattr(type(self)._sample_logits, "__wrapped__", None)

        for i in range(n_spec):
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)
            idx_i = top_idx[:, i]
            val_i = top_vals[:, i]
            w = mh.markov_w2.weight[idx_i]
            if w.dtype != markov_embed.dtype:
                w = w.to(markov_embed.dtype)
            bias = torch.bmm(w, markov_embed.unsqueeze(-1)).squeeze(-1).float()
            cand = (val_i + bias).contiguous()
            if has_multimax and orig_unbound is not None:
                # Apply ERNIE 5.5 multimax ONLY on the 64 candidate columns (3,144x smaller than 201K!)
                # BEFORE optional tau/top_p shaping, then call unwrapped _sample_logits directly!
                cand = lm_head.multimax(cand)
                if inv_tau != 1.0:
                    cand = cand * inv_tau
                if 0.0 < draft_top_p < 1.0:
                    # Nucleus-align the 64-candidate draft distribution with target top_p (CUDA-graph safe!)
                    s_vals, s_ord = torch.sort(cand, dim=-1, descending=True)
                    probs = torch.softmax(s_vals, dim=-1)
                    cdf = torch.cumsum(probs, dim=-1) - probs
                    drop = cdf > draft_top_p
                    drop[..., 0] = False
                    s_vals = s_vals.masked_fill(drop, float("-inf"))
                    cand = torch.empty_like(cand).scatter_(1, s_ord, s_vals)
                buf.scatter_(1, idx_i, cand.to(buf.dtype))
                draft_sampled_i = orig_unbound(
                    self, buf, idx_map[:, i], sample_pos[:, i], i
                )
            else:
                buf.scatter_(1, idx_i, cand.to(buf.dtype))
                draft_sampled_i = self._sample_logits(
                    buf, idx_map[:, i], sample_pos[:, i], i
                )
            # Reset only the 64 touched columns back to -inf (CUDA-graph safe, skips 201K fill!)
            buf.scatter_(1, idx_i, float("-inf"))
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.use_confidence_head:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(num_reqs, n_spec)

    def _patched_sample_seq(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        if (
            vp_prob_enabled
            and self.draft_logits is not None
            and get_tensor_model_parallel_world_size() > 1
            and getattr(self.model, "draft_id_to_target_id", None) is None
            and not getattr(getattr(self.model, "model", None), "_spark_vocab_parallel", False)
        ):
            return _sample_vp_prob(self, num_reqs, head_hidden)

        is_vp = (
            self.draft_logits is None
            and getattr(getattr(self.model, "model", None), "_spark_vocab_parallel", False)
            and getattr(self.model, "draft_id_to_target_id", None) is None
        )
        if not is_vp:
            return orig_sample_seq(self, num_reqs, head_hidden)

        vocab_start, shard_vocab = self.model.draft_vocab_shard
        if not getattr(self, "_vp_scratch_ready", False):
            sh = scratch_shape(self.max_num_reqs, shard_vocab)
            self._vp_partial_vals = torch.empty(sh, dtype=torch.float32, device=head_hidden.device)
            self._vp_partial_idx = torch.empty(sh, dtype=torch.int32, device=head_hidden.device)
            max_even_reqs = self.max_num_reqs + (self.max_num_reqs & 1)
            self._vp_pairs = torch.zeros((max_even_reqs, 2), dtype=torch.float32, device=head_hidden.device)
            self._vp_scratch_ready = True

        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_draft_logits_local(sample_hidden)
        base_logits = base_logits.view(num_reqs, n_spec, -1)
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        confidence_markov_embeds = []
        pairs = self._vp_pairs[:num_reqs]
        rows_even = num_reqs + (num_reqs & 1)
        pairs_even = self._vp_pairs[:rows_even]

        for i in range(n_spec):
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)
            bias = self.model.markov_bias_local(markov_embed)
            greedy_markov_shard_pair(
                base_logits[:, i, :shard_vocab],
                bias[:, :shard_vocab],
                vocab_start,
                pairs,
                self._vp_partial_vals,
                self._vp_partial_idx,
            )
            # Even rows -> >=16 bytes multiple so sparknet one-shot all_gather triggers even at B=1!
            gathered_pairs = tensor_model_parallel_all_gather(pairs_even, dim=-1)[:num_reqs]
            draft_sampled_i = self.draft_tokens[:num_reqs, i]
            select_first_argmax(gathered_pairs, draft_sampled_i)
            prev = draft_sampled_i

        if self.use_confidence_head:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(num_reqs, n_spec)

    dspec.DSparkSpeculator.__init__ = _patched_spec_init
    dspec.DSparkSpeculator._sample_sequential = _patched_sample_seq
    _log("installed full-vocab TP-sharded greedy DSpark Markov drafting (0% acceptance loss, 2x-4x less Markov BW)")
    return True


def _patch_kv_cache_packing() -> bool:
    """Byte-packed hybrid KV cache grouping for ERNIE 5.5 HCA+DSA+SWA+Compressor.

    Stock vLLM `_get_packed_kv_cache_groups()` uses `_approximate_gcd` on layer
    counts, splitting the 37 tiny HCA compressed layers (1,728 B/layer at bs=256)
    into 5 separate groups of 13.8 KiB inside 304 KiB blocks (95.5% empty waste,
    5.80 GiB per 1M context for just 260 MiB of HCA data!).

    Packing single-page-size buckets by `floor(anchor_bytes / layer_page_bytes)`
    and evaluating `k_dsa in {1, 2}` DSA splits cuts per-request 1M-context KV
    footprint from 11.29 GiB -> 5.16-5.54 GiB (>2.0x more concurrent 1M contexts
    on both 2x and 4x DGX Spark!) with zero kernel changes and bit-exact KV.
    """
    try:
        from collections import defaultdict
        from vllm.utils.math_utils import cdiv
        from vllm.v1.core import kv_cache_utils as kcu
        from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs
    except Exception as exc:
        _log(f"skip kv_cache_packing patch ({exc!r})")
        return False

    orig_packed = kcu._get_packed_kv_cache_groups
    max_dsa_splits = int(os.environ.get("SPARK_KV_PACK_MAX_DSA_SPLITS", "2"))

    def _byte_packed_groups(vllm_config, kv_cache_spec):
        stock_groups = orig_packed(vllm_config, kv_cache_spec)
        if not stock_groups:
            return stock_groups

        # Rebuild uniform-type buckets identically to stock vLLM.
        buckets = []
        for name, spec in kv_cache_spec.items():
            for bucket in buckets:
                candidate = {**bucket, name: spec}
                if UniformTypeKVCacheSpecs.is_uniform_type(candidate):
                    bucket[name] = spec
                    break
            else:
                buckets.append({name: spec})

        if len(buckets) <= 1:
            return stock_groups

        mixed_buckets = []
        single_buckets = []
        for bucket in buckets:
            page_size_layers = defaultdict(list)
            for layer_name, layer_spec in bucket.items():
                page_size_layers[layer_spec.page_size_bytes].append(layer_name)
            balanced = len(set(map(len, page_size_layers.values()))) == 1
            if balanced and len(page_size_layers) > 1:
                mixed_buckets.append((bucket, page_size_layers))
            else:
                single_buckets.append((bucket, page_size_layers, balanced))

        if not mixed_buckets or not single_buckets:
            return stock_groups

        # Candidate splits for the mixed-page-size (e.g. DSA+Indexer) buckets:
        # keep divisors of the mixed bucket's repeat count up to max_dsa_splits.
        min_mixed_repeats = min(
            len(next(iter(psl.values()))) for _, psl in mixed_buckets
        )
        cand_splits = [
            k for k in range(1, min(max_dsa_splits, min_mixed_repeats) + 1)
            if all(len(next(iter(psl.values()))) % k == 0 for _, psl in mixed_buckets)
        ]
        if not cand_splits:
            cand_splits = [1]

        best_groups = stock_groups
        best_bytes = kcu._max_memory_usage_bytes_from_groups(vllm_config, stock_groups)
        stock_bytes = best_bytes

        for k_split in cand_splits:
            groups = []
            # 1) Split mixed buckets into k_split balanced groups
            mixed_max_page = 0
            for bucket, psl in mixed_buckets:
                pattern_repeats = list(zip(*psl.values()))
                per_g = len(pattern_repeats) // k_split
                for g_i in range(k_split):
                    rep_slice = pattern_repeats[g_i * per_g : (g_i + 1) * per_g]
                    g_names = [name for rep in rep_slice for name in rep]
                    g_specs = {name: bucket[name] for name in g_names}
                    us = UniformTypeKVCacheSpecs.from_specs(g_specs)
                    if us is None:
                        groups = None
                        break
                    groups.append(KVCacheGroupSpec(g_names, us))
                    mixed_max_page = max(mixed_max_page, us.page_size_bytes)
                if groups is None:
                    break
            if groups is None:
                continue

            # 2) Anchor block bytes = max(mixed group page, single-layer page of any single bucket)
            anchor_bytes = mixed_max_page
            for bucket, _, _ in single_buckets:
                anchor_bytes = max(
                    anchor_bytes,
                    max(s.page_size_bytes for s in bucket.values()),
                )

            # 3) Pack every single-page-size bucket to fit `floor(anchor_bytes / layer_page)` layers/group
            ok = True
            for bucket, psl, _ in single_buckets:
                if len(psl) == 1:
                    layer_page = next(iter(psl.keys()))
                    items = list(bucket.items())
                    layers_per_group = max(1, anchor_bytes // layer_page)
                    ng = cdiv(len(items), layers_per_group)
                    # Contiguous balanced chunks preserve trailing-layer locality
                    base_sz, rem = divmod(len(items), ng)
                    pos = 0
                    for g_i in range(ng):
                        sz = base_sz + (1 if g_i < rem else 0)
                        chunk = dict(items[pos : pos + sz])
                        pos += sz
                        us = UniformTypeKVCacheSpecs.from_specs(chunk)
                        if us is None:
                            ok = False
                            break
                        groups.append(KVCacheGroupSpec(list(chunk), us))
                    if not ok:
                        break
                else:
                    us = UniformTypeKVCacheSpecs.from_specs(bucket)
                    if us is None:
                        ok = False
                        break
                    groups.append(KVCacheGroupSpec(list(bucket), us))
            if not ok:
                continue

            kcu._annotate_eagle_groups(
                vllm_config,
                kv_cache_spec,
                groups,
                use_trailing_layer_fallback=kcu._uses_trailing_mtp_layers(vllm_config),
            )
            kcu._warn_if_unannotated_eagle_mamba(vllm_config, groups)

            cand_bytes = kcu._max_memory_usage_bytes_from_groups(vllm_config, groups)
            if cand_bytes < best_bytes:
                best_bytes = cand_bytes
                best_groups = groups

        if best_groups is not stock_groups and best_bytes > 0:
            max_len = vllm_config.model_config.max_model_len
            bpb_old = kcu._pool_bytes_per_block(vllm_config, stock_groups)
            bpb_new = kcu._pool_bytes_per_block(vllm_config, best_groups)
            _log(
                f"byte-packed KV cache groups ({len(stock_groups)} -> {len(best_groups)} groups, "
                f"block {bpb_old:,d}B -> {bpb_new:,d}B): "
                f"max_len={max_len:,d} req KV {stock_bytes/1024**3:.2f} GiB -> {best_bytes/1024**3:.2f} GiB "
                f"({stock_bytes/best_bytes:.2f}x MORE concurrent context capacity!)"
            )
        return best_groups

    kcu._get_packed_kv_cache_groups = _byte_packed_groups
    _log(f"installed byte-packed hybrid KV cache grouping (SPARK_KV_PACK_MAX_DSA_SPLITS={max_dsa_splits})")
    return True


def _patch_packed_bf16_lm_head() -> bool:
    """Lossless 12-bit packed BF16 TileLang LM head (1.31x-1.79x faster on GB10 SM12.1!).

    Intercepts `LogitsProcessor._apply_head` for unquantized BF16 `ParallelLMHead`
    weights (used by BOTH ERNIE 5.5 target and DSpark draft), packing the shard
    into 12-bit exact representation on first warmup call (outside CUDA graph capture).
    """
    try:
        import torch
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        from kernels.lm_head.packed_bf16_lm_head import PackedBF16VocabRunner, pack_bf16_exact
    except Exception as exc:
        _log(f"skip packed_bf16_lm_head patch ({exc!r})")
        return False

    orig_apply_head = LogitsProcessor._apply_head

    def _patched_apply_head(self, lm_head, hidden_states, embedding_bias):
        if (
            embedding_bias is None
            and isinstance(lm_head, ParallelLMHead)
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
        ):
            out_dtype = self.head_dtype if self.head_dtype is not None else hidden_states.dtype
            if out_dtype in (torch.float32, torch.bfloat16):
                key = f"_spark_packed_lm_{out_dtype}"
                runner = getattr(lm_head, key, None)
                if runner is None and not torch.cuda.is_current_stream_capturing():
                    w = getattr(lm_head, "weight", None)
                    if (
                        w is not None
                        and w.is_cuda
                        and w.dtype == torch.bfloat16
                        and w.ndim == 2
                        and w.shape[0] % 32 == 0
                        and w.shape[1] % 256 == 0
                    ):
                        # Wait until weights are actually loaded (skip all-zero init sanity passes)
                        if w.abs().max().item() > 0:
                            try:
                                # Share already-packed weight if draft LM head packed it first (saves 618 MB on TP=2!)
                                draft_runner = getattr(lm_head, "_spark_packed_draft_bf16", None)
                                pk = (
                                    draft_runner.packed
                                    if (draft_runner and hasattr(draft_runner, "packed"))
                                    else pack_bf16_exact(w.detach(), verify=False)
                                )
                                runner = PackedBF16VocabRunner(pk, out_dtype=out_dtype)
                                setattr(lm_head, key, runner)
                                shared_tag = " [shared w/ draft, 0 extra VRAM]" if (draft_runner and hasattr(draft_runner, "packed")) else ""
                                _log(
                                    f"activated lossless 12-bit packed BF16 LM head "
                                    f"[{w.shape[0]}x{w.shape[1]}] -> {out_dtype} "
                                    f"(exc={pk.num_exceptions}, base={pk.exponent_base}){shared_tag}"
                                )
                            except Exception as exc:
                                _log(f"packed_bf16_lm_head build failed, falling back ({exc!r})")
                                setattr(lm_head, key, False)
                                runner = False
                if runner and hidden_states.numel() // hidden_states.shape[-1] <= 256:
                    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
                    logits = runner(flat)
                    return logits.reshape(*hidden_states.shape[:-1], -1)
        return orig_apply_head(self, lm_head, hidden_states, embedding_bias)

    LogitsProcessor._apply_head = _patched_apply_head
    _log("installed lossless 12-bit packed BF16 TileLang LM head hook (1.31x-1.79x faster LM head on GB10)")
    return True


def _patch_dspark_drafter_padding_mask() -> bool:
    """Mask DSpark drafter CUDA-graph padding rows so MoE skips unused experts (adapted from 0042)."""
    try:
        import torch
        from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dfspec
    except Exception as exc:
        _log(f"skip dspark_drafter_padding_mask ({exc!r})")
        return False

    cls = dfspec.DFlashSpeculator
    orig_init = cls.__init__
    orig_capture = cls.capture
    orig_run_model = cls._run_model
    orig_propose = cls.propose

    def _patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        dev = self.hidden_states.device
        self._spark_is_padding = torch.ones(self.max_num_tokens, dtype=torch.bool, device=dev)
        self._spark_row_index = torch.arange(self.max_num_tokens, device=dev)

    def _patched_capture(self, *args, **kwargs):
        self._spark_is_padding.fill_(True)
        return orig_capture(self, *args, **kwargs)

    def _patched_run_model(
        self,
        num_tokens,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode=dfspec.CUDAGraphMode.NONE,
    ):
        batch_descriptor = dfspec.BatchDescriptor(num_tokens=num_tokens)
        with dfspec.set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
            is_padding=self._spark_is_padding[:num_tokens],
        ):
            return self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                inputs_embeds=None,
            )

    def _patched_propose(self, *args, **kwargs):
        # Mark live rows [0..num_query_tokens) as non-padding and graph padding rows as padding
        input_batch = kwargs.get("input_batch") if "input_batch" in kwargs else (args[0] if args else None)
        dummy_run = bool(kwargs.get("dummy_run", False))
        if input_batch is not None:
            num_reqs = getattr(input_batch, "num_reqs", 0)
            live = 0 if dummy_run else num_reqs * self.num_query_per_req
            limit = min(self.max_num_tokens, max(live + 64, 256))
            self._spark_is_padding[:limit].copy_(self._spark_row_index[:limit] >= live)
        return orig_propose(self, *args, **kwargs)

    cls.__init__ = _patched_init
    cls.capture = _patched_capture
    cls._run_model = _patched_run_model
    cls.propose = _patched_propose
    _log("installed DSpark drafter padding-row MoE mask (skips draft MoE reads on CUDA-graph padding rows)")
    return True


def _patch_vha_bf16_allreduce() -> bool:
    """Fused SM12.1 Triton VHAPostMix + BF16 TP all-reduce (4.2x faster prefill, 2.2-3.4x decode, 2x smaller RDMA)."""
    try:
        import torch
        from vllm.distributed import tensor_model_parallel_all_reduce
        from vllm_ernie.models.ernie5_5.nvidia import attention as eattn
        from kernels.mhc_hca.vha_triton import (
            supports_vha_triton,
            vha_u_proj,
            vha_v_add_inplace,
        )
    except Exception as exc:
        _log(f"skip vha_bf16_allreduce ({exc!r})")
        return False

    use_triton = _env_flag("SPARK_PATCH_VHA_TRITON", "1")
    cls = eattn.VHAPostMix

    def _patched_forward(self, o: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return torch.matmul(self.m, o)
        if use_triton and supports_vha_triton(o, self.U.shape[1]):
            ut = getattr(self, "_spark_ut_bf16", None)
            if ut is None or ut.data_ptr() == 0:
                if not torch.cuda.is_current_stream_capturing():
                    ut = self.U.detach().t().to(torch.bfloat16).contiguous()
                    v = self.V.detach().to(torch.bfloat16).contiguous()
                    self._spark_ut_bf16 = ut
                    self._spark_v_bf16 = v
                else:
                    ut = self.U.t().to(torch.bfloat16).contiguous()
                    self._spark_v_bf16 = self.V.to(torch.bfloat16).contiguous()
            v = self._spark_v_bf16
            z_local = vha_u_proj(o, ut)
            z = tensor_model_parallel_all_reduce(z_local)
            return vha_v_add_inplace(z, v, o)
        num_tokens = o.shape[0]
        u = self.U.t().expand(num_tokens, -1, -1)
        z_local = torch.bmm(u, o, out_dtype=torch.float32).to(o.dtype)
        z = tensor_model_parallel_all_reduce(z_local).float()
        v = self.V.float().expand(num_tokens, -1, -1)
        return torch.bmm(v, z).add_(o).to(o.dtype)

    cls.forward = _patched_forward
    _log(
        f"installed VHAPostMix BF16 all-reduce + fused Triton kernel "
        f"(triton={use_triton}, 4.2x faster prefill, 2.2-3.4x decode, 2x smaller RDMA)"
    )
    return True


def _patch_fc2_latent_tp() -> bool:
    """Row-shard MoE fc2_latent_proj across TP ranks and fuse its AR into shared_experts AR.

    Algebraic identity (100% exact, ZERO extra collectives):
      y = fc2(latent_norm(routed)) + AR(shared_partial)
        = AR(fc2_local(latent_norm(routed)_local) + shared_partial)
    Cuts fc2_latent_proj weights, DRAM reads, and prefill FLOPs by 4x on TP=4 (2x on TP=2).
    Skipped automatically when shared_experts are replicated (SPARK_REPLICATE_SHARED_EXPERTS=1).
    """
    if _env_flag("SPARK_REPLICATE_SHARED_EXPERTS", "0"):
        _log("skip fc2_latent_tp (SPARK_REPLICATE_SHARED_EXPERTS=1 disables shared_experts AR)")
        return False
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr
        from vllm.model_executor.layers.linear import RowParallelLinear
        import vllm_ernie.models.ernie5_5.nvidia.moe as emoe
    except Exception as exc:
        _log(f"skip fc2_latent_tp ({exc!r})")
        return False

    orig_moe_init = emoe.Ernie5_5MoE.__init__
    orig_xform_fwd = emoe.Ernie5_5RoutedOutputTransform.forward
    orig_reduce_shared = mr.MoERunner._maybe_reduce_shared_expert_output
    orig_reduce_final = mr.MoERunner._maybe_reduce_final_output

    def _patched_moe_init(self, config, quant_config=None, prefix="", aux_stream=None):
        orig_moe_init(self, config, quant_config=quant_config, prefix=prefix, aux_stream=aux_stream)
        tp = get_tensor_model_parallel_world_size()
        latent_size = config.moe_latent_size
        if tp > 1 and latent_size % (tp * 128) == 0:
            xform = getattr(self.experts, "routed_output_transform", None)
            if xform is not None:
                self.fc2_latent_proj = RowParallelLinear(
                    latent_size,
                    config.hidden_size,
                    bias=False,
                    input_is_parallel=True,
                    reduce_results=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.fc2_latent_proj",
                )
                xform.up_proj = self.fc2_latent_proj
                xform._spark_fc2_tp = True
                xform._spark_tp_size = tp
                xform._spark_tp_rank = get_tensor_model_parallel_rank()

    def _patched_xform_fwd(self, hidden_states):
        if getattr(self, "_spark_fc2_tp", False):
            h = self.norm(hidden_states)
            tp = self._spark_tp_size
            rank = self._spark_tp_rank
            chunk = h.shape[-1] // tp
            h_local = h[..., rank * chunk : (rank + 1) * chunk].contiguous()
            return self.up_proj(h_local)[0]
        return orig_xform_fwd(self, hidden_states)

    def _patched_reduce_shared(self, shared_output, fused_output_is_reduced=None):
        if getattr(self.routed_output_transform, "_spark_fc2_tp", False):
            # Keep shared_output un-reduced so it sums with fc2 partial before 1 fused final AR!
            return shared_output
        return orig_reduce_shared(self, shared_output, fused_output_is_reduced=fused_output_is_reduced)

    def _patched_reduce_final(self, states, trunc_size, output_is_reduced=None):
        if getattr(self.routed_output_transform, "_spark_fc2_tp", False):
            # Force the late all-reduce over (shared_partial + fc2_partial)
            return orig_reduce_final(self, states, trunc_size, output_is_reduced=False)
        return orig_reduce_final(self, states, trunc_size, output_is_reduced=output_is_reduced)

    emoe.Ernie5_5MoE.__init__ = _patched_moe_init
    emoe.Ernie5_5RoutedOutputTransform.forward = _patched_xform_fwd
    mr.MoERunner._maybe_reduce_shared_expert_output = _patched_reduce_shared
    mr.MoERunner._maybe_reduce_final_output = _patched_reduce_final
    _log("installed row-sharded fc2_latent_proj fused into shared_experts AR (2x-4x less fc2 compute/VRAM, 0 extra ARs)")
    return True


def _patch_fc1_latent_tp() -> bool:
    """Column-shard MoE fc1_latent_proj across TP ranks (gather_output=True) to cut decode DRAM & VRAM 2x-4x."""
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.model_executor.layers.linear import ColumnParallelLinear
        import vllm_ernie.models.ernie5_5.nvidia.moe as emoe
    except Exception as exc:
        _log(f"skip fc1_latent_tp ({exc!r})")
        return False

    orig_moe_init = emoe.Ernie5_5MoE.__init__

    def _patched_moe_init(self, config, quant_config=None, prefix="", aux_stream=None):
        orig_moe_init(self, config, quant_config=quant_config, prefix=prefix, aux_stream=aux_stream)
        tp = get_tensor_model_parallel_world_size()
        latent_size = config.moe_latent_size
        if tp > 1 and latent_size % (tp * 128) == 0:
            self.fc1_latent_proj = ColumnParallelLinear(
                config.hidden_size,
                latent_size,
                bias=False,
                gather_output=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc1_latent_proj",
            )

    emoe.Ernie5_5MoE.__init__ = _patched_moe_init
    _log("installed column-sharded fc1_latent_proj across TP ranks (2x-4x less fc1 decode DRAM & VRAM)")
    return True


def _patch_gate_bf16x2() -> bool:
    """Use cuBLAS BF16x2 Tensor-Core GEMM for ERNIE 5.5 FP32 MoE router GateLinear.

    Stock vLLM casts BF16 hidden states to FP32 and runs non-Tensor-Core FP32 SIMT
    cublasSgemm (allow_tf32=False), burning ~246 ms per 8K prefill chunk across 42
    MoE layers (~126 ms at 4K, ~2.1 ms at T=8 decode).
    cuBLAS BF16x2 (w_hi + w_lo) preserves 99.99% exact top-10 routing (100.00% at
    decode T<=32) while cutting 42-layer router time to 85 ms at 8K (-161 ms) and
    0.8 ms at T=8 decode (-1.0 to -1.3 ms/step).
    """
    try:
        import torch
        from vllm.model_executor.layers.fused_moe import GateLinear
        from kernels.moe.gate_bf16x2 import bf16x2_router_gemm, split_fp32_to_bf16x2
    except Exception as exc:
        _log(f"skip gate_bf16x2 patch ({exc!r})")
        return False

    orig_fwd = GateLinear.forward
    single_max_m = int(os.environ.get("SPARK_GATE_SINGLE_MAX_M", "32"))

    def _patched_gate_fwd(self, x: torch.Tensor):
        w = getattr(self, "weight", None)
        if (
            w is not None
            and w.is_cuda
            and w.dtype == torch.float32
            and self.out_dtype == torch.float32
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.dim() == 2
            and not getattr(self, "allow_bf16x3_router_gemm", False)
            and not getattr(self, "allow_fp32_router_gemm", False)
            and getattr(self, "is_unquantized", True)
        ):
            pair = getattr(self, "_spark_bf16x2", None)
            if pair is None and not torch.cuda.is_current_stream_capturing():
                if w.abs().max().item() > 0:
                    pair = split_fp32_to_bf16x2(w)
                    self._spark_bf16x2 = pair
            if pair is not None:
                out = bf16x2_router_gemm(x, pair[0], pair[1], single_term_max_m=single_max_m)
                return self._return(out)
        return orig_fwd(self, x)

    GateLinear.forward = _patched_gate_fwd
    _log(
        f"installed cuBLAS BF16x2 Tensor-Core MoE router GateLinear "
        f"(single_max_m={single_max_m}, -161ms/8K prefill & -1.3ms/step decode)"
    )
    return True


def _patch_hca_oproj_fused() -> bool:
    """Fused TP>1 VHA+invRoPE+FP8 quant, chunked fp8_einsum (T=1024), in-place gate, & 2-chunk RDMA pipeline.

    Addresses three measured GB10 SM12.1 HCA bottlenecks across all 37 HCA layers:
      1. Decode & prefill: fuses `vha_v_add + inv_rope + per-128 UE8M0 FP8 quant`
         into 1 Triton kernel (2.08x faster decode -0.8ms/step, 1.76x-1.99x prefill
         -45ms TP4 / -94ms TP2), plus in-place `sigmoid_mul_inplace` (-12ms/8K).
      2. Prefill (T > 2048): tiles `vha_v_inv_rope_fp8_quant -> fp8_einsum` in
         1024-token L2-resident tiles, eliminating GB10's 3.53x `fp8_einsum` cliff
         at T=8192 (`99.5ms -> 28.0ms`, -71.5ms TP4 / -107ms TP2).
      3. Prefill (T >= 2048, eager): 2-chunk compute/RDMA pipeline on a dedicated
         `comm_stream` overlapping `vha_ar(0)` with `vha_u(1) + o_gate(qr)`,
         `vha_ar(1)` with `oproj_compute(0)`, and `wo_b_ar(0)` with `oproj_compute(1)`
         (100% bit-exact, hides ~250ms of RoCEv2 RDMA behind GPU compute!).
    """
    try:
        import torch
        from vllm.distributed import tensor_model_parallel_all_reduce
        from vllm.utils.deep_gemm import fp8_einsum, get_tma_aligned_size
        from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
        from vllm_ernie.models.ernie5_5.nvidia import attention as eattn
        from kernels.mhc_hca.vha_triton import (
            sigmoid_mul_inplace,
            supports_vha_triton,
            vha_u_proj,
            vha_v_add_inv_rope_fp8_quant,
        )
    except Exception as exc:
        _log(f"skip hca_oproj_fused patch ({exc!r})")
        return False

    enable_pipeline = _env_flag("SPARK_HCA_PIPELINE", "1")
    tile_t = int(os.environ.get("SPARK_HCA_EINSUM_TILE", "1024"))
    comm_stream_holder: dict[int, torch.cuda.Stream] = {}

    def _get_comm_stream(dev: torch.device) -> torch.cuda.Stream:
        idx = dev.index or 0
        s = comm_stream_holder.get(idx)
        if s is None:
            s = torch.cuda.Stream(device=dev)
            comm_stream_holder[idx] = s
        return s

    def _get_vha_uv(vha_mod):
        ut = getattr(vha_mod, "_spark_ut_bf16", None)
        if ut is None or ut.data_ptr() == 0:
            if not torch.cuda.is_current_stream_capturing():
                ut = vha_mod.U.detach().t().to(torch.bfloat16).contiguous()
                v = vha_mod.V.detach().to(torch.bfloat16).contiguous()
                vha_mod._spark_ut_bf16 = ut
                vha_mod._spark_v_bf16 = v
            else:
                ut = vha_mod.U.t().to(torch.bfloat16).contiguous()
                v = vha_mod.V.to(torch.bfloat16).contiguous()
                vha_mod._spark_v_bf16 = v
        return ut, vha_mod._spark_v_bf16

    def _oproj_tiled_compute(
        attn,
        o_chunk: torch.Tensor,
        z_chunk: torch.Tensor,
        v_bf16: torch.Tensor,
        pos_chunk: torch.Tensor,
        gate_chunk: torch.Tensor,
        *,
        gate_is_sigmoid: bool,
        run_local_wob: bool,
    ) -> torch.Tensor:
        tc = o_chunk.shape[0]
        wo_a = attn.wo_a
        ws = wo_a.weight_scale if hasattr(wo_a, "weight_scale") else wo_a.weight_scale_inv
        n_groups = attn.n_local_groups
        o_rank = attn.o_lora_rank
        z_out = torch.empty((tc, n_groups, o_rank), device=o_chunk.device, dtype=torch.bfloat16)

        # Tile into L2-resident sub-chunks (1024 tokens) when tc > tile_t to avoid the T=8192 fp8_einsum cliff!
        step = tile_t if (tc > tile_t and not torch.cuda.is_current_stream_capturing()) else tc
        for s in range(0, tc, step):
            e = min(s + step, tc)
            sub_t = e - s
            aligned = get_tma_aligned_size(sub_t, 4)
            oq, os_ = vha_v_add_inv_rope_fp8_quant(
                o_chunk[s:e],
                z_chunk[s:e],
                v_bf16,
                pos_chunk[s:e],
                attn.rotary_emb.cos_sin_cache,
                n_groups=n_groups,
                nope_dim=attn.nope_head_dim,
                rope_dim=attn.rope_head_dim,
                tma_aligned_size=aligned,
            )
            fp8_einsum(
                "bhr,hdr->bhd",
                (oq, os_),
                (wo_a.weight, ws),
                z_out[s:e],
                recipe=attn._einsum_recipe,
            )

        z_flat = z_out.view(tc, n_groups * o_rank)
        if gate_is_sigmoid:
            z_flat.mul_(gate_chunk)
        else:
            sigmoid_mul_inplace(z_flat, gate_chunk)

        if run_local_wob:
            return attn.wo_b.quant_method.apply(attn.wo_b, z_flat, bias=None)
        return attn.wo_b(z_flat)

    orig_hca_fwd = eattn.Ernie5_5HCAAttentionMixin.forward

    def _patched_hca_fwd(self, positions, hidden_states, llama_4_scaling=None):
        can_fuse = (
            self.vha.tp_size > 1
            and not self._fuse_vha_quant
            and self._fused_wo_b is None
            and self.wo_a.weight.dtype == torch.float8_e4m3fn
            and self._tma_aligned_scales
            and self._einsum_recipe[2] == 128
            and self.head_dim == 512
            and self.nope_head_dim == 448
            and self.rope_head_dim == 64
        )
        if not can_fuse:
            return orig_hca_fwd(self, positions, hidden_states, llama_4_scaling)

        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self._run_parallel_input_projections(hidden_states)
        )
        qr, qr_scale, kv = self._split_qkv_and_norm(qr_kv)

        pipe_min_t = int(os.environ.get("SPARK_HCA_PIPE_MIN_T", "4096"))
        use_pipe = (
            enable_pipeline
            and num_tokens >= pipe_min_t
            and not torch.cuda.is_current_stream_capturing()
            and getattr(self.wo_b, "bias", None) is None
            and getattr(self.wo_b, "input_is_parallel", False)
            and getattr(self.wo_b, "reduce_results", False)
        )

        if use_pipe:
            # Delay o_gate(qr) into the vha_ar(chunk 0) RoCE shadow!
            self._prepare_and_attn_fn(
                hidden_states,
                qr,
                kv,
                qr_scale,
                kv_score,
                indexer_kv_score,
                indexer_weights,
                positions,
                o_padded,
            )
            o = o_padded[:, : self.n_local_heads, :]
            if not o.is_contiguous():
                o = o.contiguous()
            if not supports_vha_triton(o, self.vha.U.shape[1]):
                gate = self._gate(qr)
                o = self.vha(o)
                from vllm.models.deepseek_v4.nvidia.ops.o_proj import deep_gemm_fp8_o_proj
                return deep_gemm_fp8_o_proj(
                    o, positions, self.rotary_emb.cos_sin_cache, self.wo_a, self._gated_wo_b(gate),
                    n_groups=self.n_local_groups, heads_per_group=self.n_local_heads // self.n_local_groups,
                    nope_dim=self.nope_head_dim, rope_dim=self.rope_head_dim, o_lora_rank=self.o_lora_rank,
                    einsum_recipe=self._einsum_recipe, tma_aligned_scales=self._tma_aligned_scales,
                )

            ut, v_bf16 = _get_vha_uv(self.vha)
            mid = ((num_tokens // 2) // 128) * 128
            if mid < 128 or num_tokens - mid < 128:
                mid = num_tokens // 2

            def_stream = torch.cuda.current_stream(o.device)
            comm_stream = _get_comm_stream(o.device)
            ev_u0 = torch.cuda.Event()
            ev_u1 = torch.cuda.Event()
            ev_ar0 = torch.cuda.Event()
            ev_ar1 = torch.cuda.Event()
            ev_w0 = torch.cuda.Event()
            ev_w1 = torch.cuda.Event()
            ev_yar0 = torch.cuda.Event()
            ev_yar1 = torch.cuda.Event()

            # Stage 1: vha_u(0) -> kick off vha_ar(0) on comm_stream
            z0_local = vha_u_proj(o[:mid], ut)
            ev_u0.record(def_stream)
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(ev_u0)
                z0_local.record_stream(comm_stream)
                z0 = tensor_model_parallel_all_reduce(z0_local)
                ev_ar0.record(comm_stream)

            # Stage 2 (hidden under vha_ar(0) RoCE!): vha_u(1) + o_gate(qr)
            z1_local = vha_u_proj(o[mid:], ut)
            ev_u1.record(def_stream)
            gate_raw = self.o_gate(qr)

            # Stage 3: kick off vha_ar(1) on comm_stream
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(ev_u1)
                z1_local.record_stream(comm_stream)
                z1 = tensor_model_parallel_all_reduce(z1_local)
                ev_ar1.record(comm_stream)

            # Stage 4 (hidden under vha_ar(1) RoCE!): chunk 0 oproj_compute + wo_b_local
            def_stream.wait_event(ev_ar0)
            z0.record_stream(def_stream)
            y0_local = _oproj_tiled_compute(
                self, o[:mid], z0, v_bf16, positions[:mid], gate_raw[:mid],
                gate_is_sigmoid=False, run_local_wob=True,
            )
            ev_w0.record(def_stream)

            # Stage 5: kick off wo_b_ar(0) on comm_stream
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(ev_w0)
                y0_local.record_stream(comm_stream)
                y0 = tensor_model_parallel_all_reduce(y0_local)
                ev_yar0.record(comm_stream)

            # Stage 6 (overlapped with wo_b_ar(0) RoCE!): chunk 1 oproj_compute + wo_b_local
            def_stream.wait_event(ev_ar1)
            z1.record_stream(def_stream)
            y1_local = _oproj_tiled_compute(
                self, o[mid:], z1, v_bf16, positions[mid:], gate_raw[mid:],
                gate_is_sigmoid=False, run_local_wob=True,
            )
            ev_w1.record(def_stream)

            # Stage 7: kick off wo_b_ar(1) on comm_stream
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(ev_w1)
                y1_local.record_stream(comm_stream)
                y1 = tensor_model_parallel_all_reduce(y1_local)
                ev_yar1.record(comm_stream)

            # Join both reduced chunks cleanly into contiguous [T, H] output
            out = torch.empty((num_tokens, hidden_states.shape[-1]), dtype=hidden_states.dtype, device=hidden_states.device)
            def_stream.wait_event(ev_yar0)
            y0.record_stream(def_stream)
            out[:mid].copy_(y0)
            def_stream.wait_event(ev_yar1)
            y1.record_stream(def_stream)
            out[mid:].copy_(y1)
            return out

        # Non-pipelined path (decode T<2048 & CUDA Graph capture):
        # Fused vha_v_add_inv_rope_fp8_quant + sigmoid_mul_inplace!
        aux_stream = None
        if (
            self.aux_stream_list is not None
            and self.indexer is None
            and num_tokens <= eattn.envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD
        ):
            aux_stream = self.aux_stream_list[1]
        _, gate_raw = maybe_execute_in_parallel(
            lambda: self._prepare_and_attn_fn(
                hidden_states,
                qr,
                kv,
                qr_scale,
                kv_score,
                indexer_kv_score,
                indexer_weights,
                positions,
                o_padded,
            ),
            lambda: self.o_gate(qr),
            self.gate_events[0],
            self.gate_events[1],
            aux_stream,
        )
        o = o_padded[:, : self.n_local_heads, :]
        if supports_vha_triton(o, self.vha.U.shape[1]):
            ut, v_bf16 = _get_vha_uv(self.vha)
            z_local = vha_u_proj(o, ut)
            z = tensor_model_parallel_all_reduce(z_local)
            return _oproj_tiled_compute(
                self, o, z, v_bf16, positions, gate_raw,
                gate_is_sigmoid=False, run_local_wob=False,
            )
        o = self.vha(o)
        from vllm.models.deepseek_v4.nvidia.ops.o_proj import deep_gemm_fp8_o_proj
        return deep_gemm_fp8_o_proj(
            o, positions, self.rotary_emb.cos_sin_cache, self.wo_a,
            lambda z_in: self.wo_b(sigmoid_mul_inplace(z_in, gate_raw)),
            n_groups=self.n_local_groups, heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim, rope_dim=self.rope_head_dim, o_lora_rank=self.o_lora_rank,
            einsum_recipe=self._einsum_recipe, tma_aligned_scales=self._tma_aligned_scales,
        )

    eattn.Ernie5_5HCAAttentionMixin.forward = _patched_hca_fwd

    # Also fuse DSA wo_b(o.flatten(1) * torch.sigmoid(gate)) in-place
    orig_dsa_fwd = eattn.Ernie5_5DSAAttention.forward

    def _patched_dsa_fwd(self, positions, hidden_states, llama_4_scaling=None):
        if self._fused_wo_b is not None:
            return orig_dsa_fwd(self, positions, hidden_states, llama_4_scaling)
        num_tokens = hidden_states.shape[0]
        q_c, kv_lora = self.fused_qkv_a_proj(hidden_states)[0].split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
        )
        q_c = self.q_a_layernorm(q_c)
        gate = self.o_gate(q_c)
        q = self.q_b_proj(q_c)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], -1)
        kv_c_normed = self.kv_a_layernorm(kv_c)
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim :], k_pe.unsqueeze(1)
        )
        attn_out = torch.empty(
            (num_tokens, self.num_local_heads * self.v_head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._indexer_and_attn(hidden_states, q_c, positions, q, kv_c_normed, k_pe, attn_out)
        o = self.vha(attn_out.view(num_tokens, self.num_local_heads, self.v_head_dim))
        return self.wo_b(sigmoid_mul_inplace(o.flatten(1), gate))

    eattn.Ernie5_5DSAAttention.forward = _patched_dsa_fwd
    _log(
        f"installed Wave-5 fused HCA VHA+invRoPE+FP8 + L2-tiled fp8_einsum (tile={tile_t}) "
        f"+ 2-chunk RDMA pipeline={enable_pipeline} + in-place sigmoid_mul"
    )
    return True


def _patch_profile_prefill() -> bool:
    """Optional CUDA-event sub-layer & step profiler (off by default, SPARK_PROFILE_PREFILL=1)."""
    try:
        import time
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.v1.worker.gpu import model_runner as gmr
        from vllm.v1.worker.gpu.spec_decode import adaptive_verification as av
        from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dfspec
        from vllm_ernie.models.ernie5_5.nvidia import model as emodel
        from vllm_ernie.models.ernie5_5.nvidia import moe as emoe
    except Exception as exc:
        _log(f"skip profile_prefill ({exc!r})")
        return False

    state = {"ready": False, "prefill_prints": 0, "step_prints": 0, "last_step_t": 0.0}

    # 1. Print exact startup DSpark cost curves (draft_cost_ms & verify_cost_ms!)
    orig_set_curves = av.AdaptiveVerificationManager.set_initial_cost_curves

    def _prof_set_curves(self, timings):
        ret = orig_set_curves(self, timings)
        state["ready"] = True
        if get_tensor_model_parallel_rank() == 0 and self.cost_tables is not None:
            dc, vc = self.cost_tables
            _log(
                f"COST_CURVES draft_ms[1..8]={[round(float(x), 2) for x in dc[1:9]]} "
                f"verify_ms[1..16]={[round(float(x), 2) for x in vc[1:17]]}"
            )
        return ret

    av.AdaptiveVerificationManager.set_initial_cost_curves = _prof_set_curves

    # 2. Sub-layer profiler on L1 (HCA) & L7 (DSA) for prefill (T >= 512)
    orig_init = emodel.Ernie5_5DecoderLayer.__init__
    orig_fwd = emodel.Ernie5_5DecoderLayer.forward

    def _prof_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        prefix = kwargs.get("prefix", "")
        try:
            self._spark_lid = int(prefix.rsplit(".", 1)[-1])
        except Exception:
            self._spark_lid = -1

    def _prof_fwd(self, x, positions, post_mix=None, res_mix=None, residual=None):
        T = x.shape[0]
        lid = getattr(self, "_spark_lid", -1)
        if (
            state["ready"]
            and T >= 512
            and not torch.cuda.is_current_stream_capturing()
            and get_tensor_model_parallel_rank() == 0
            and lid in (1, 7)
            and state["prefill_prints"] < 12
        ):
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
            ev[0].record()
            norm_kwargs = dict(
                norm_weight=self.attn_norm.weight.data,
                norm_eps=self.attn_norm.variance_epsilon,
            )
            if residual is None:
                if x.dim() == 2:
                    residual, post_mix, res_mix, x = emodel.mhc_pre_broadcast_tilelang(
                        x, *self._mhc_args("attn"), fn_broadcast=self.hc_attn_fn_broadcast, **norm_kwargs
                    )
                else:
                    residual = x
                    post_mix, res_mix, x = emodel.mhc_pre_tilelang(x, *self._mhc_args("attn"), **norm_kwargs)
            else:
                residual, post_mix, res_mix, x = emodel.mhc_fused_post_pre_tilelang(
                    x, residual, post_mix, res_mix, *self._mhc_args("attn"),
                    n_splits=1, tile_n=1, **norm_kwargs,
                )
            ev[1].record()
            if lid == 1 and hasattr(self.attn, "_run_parallel_input_projections"):
                attn = self.attn
                o_padded = torch.empty((T, attn.padded_heads, attn.head_dim), dtype=x.dtype, device=x.device)
                qr_kv, kv_score, idx_kv_score, idx_w = attn._run_parallel_input_projections(x)
                qr, qr_scale, kv = attn._split_qkv_and_norm(qr_kv)
                ev[2].record()
                gate = attn._gate(qr)
                attn._prepare_and_attn_fn(x, qr, kv, qr_scale, kv_score, idx_kv_score, idx_w, positions, o_padded)
                ev[3].record()
                o = attn.vha(o_padded[:, :attn.n_local_heads, :])
                ev[4].record()
                from vllm.models.deepseek_v4.nvidia.ops.o_proj import deep_gemm_fp8_o_proj
                x = deep_gemm_fp8_o_proj(
                    o, positions, attn.rotary_emb.cos_sin_cache, attn.wo_a, attn._gated_wo_b(gate),
                    n_groups=attn.n_local_groups, heads_per_group=attn.n_local_heads // attn.n_local_groups,
                    nope_dim=attn.nope_head_dim, rope_dim=attn.rope_head_dim, o_lora_rank=attn.o_lora_rank,
                    einsum_recipe=attn._einsum_recipe, tma_aligned_scales=attn._tma_aligned_scales,
                )
                ev[5].record()
            else:
                x = self.attn(positions, x, None)
                ev[2].record(); ev[3].record(); ev[4].record(); ev[5].record()
            residual, post_mix, res_mix, x = emodel.mhc_fused_post_pre_tilelang(
                x, residual, post_mix, res_mix, *self._mhc_args("ffn"),
                n_splits=1, tile_n=1,
                norm_weight=self.ffn_norm.weight.data,
                norm_eps=self.ffn_norm.variance_epsilon,
            )
            ev[6].record()
            if hasattr(self.ffn, "fc1_latent_proj"):
                ffn = self.ffn
                logits, _ = ffn.gate(x)
                scores = emoe.split_sigmoid_sum(logits, ffn.n_routed_experts)
                routed_h = ffn.fc1_latent_proj(x)[0]
                ev[7].record()
                x = ffn.experts(hidden_states=routed_h, router_logits=scores, shared_experts_input=x)
                ev[8].record()
            else:
                x = self.ffn(x)
                ev[7].record(); ev[8].record()
            torch.cuda.synchronize()
            kind = "DSA" if lid == 7 else "HCA"
            if lid == 1:
                _log(
                    f"PREFILL_PROF L1(HCA) T={T}: mhc1={ev[0].elapsed_time(ev[1]):.2f} "
                    f"inproj={ev[1].elapsed_time(ev[2]):.2f} prep+mla={ev[2].elapsed_time(ev[3]):.2f} "
                    f"vha={ev[3].elapsed_time(ev[4]):.2f} oproj={ev[4].elapsed_time(ev[5]):.2f} "
                    f"mhc2={ev[5].elapsed_time(ev[6]):.2f} route+fc1={ev[6].elapsed_time(ev[7]):.2f} "
                    f"moe={ev[7].elapsed_time(ev[8]):.2f} tot={ev[0].elapsed_time(ev[8]):.2f}ms"
                )
            else:
                _log(
                    f"PREFILL_PROF L7(DSA) T={T}: mhc1={ev[0].elapsed_time(ev[1]):.2f} "
                    f"attn={ev[1].elapsed_time(ev[5]):.2f} mhc2={ev[5].elapsed_time(ev[6]):.2f} "
                    f"route+fc1={ev[6].elapsed_time(ev[7]):.2f} moe={ev[7].elapsed_time(ev[8]):.2f} "
                    f"tot={ev[0].elapsed_time(ev[8]):.2f}ms"
                )
            state["prefill_prints"] += 1
            return x, residual, post_mix, res_mix
        return orig_fwd(self, x, positions, post_mix, res_mix, residual)

    emodel.Ernie5_5DecoderLayer.__init__ = _prof_init
    emodel.Ernie5_5DecoderLayer.forward = _prof_fwd

    # 3. Step-level profiler on GPUModelRunner.execute_model / sample_tokens / DFlashSpeculator.propose
    orig_exec = gmr.GPUModelRunner.execute_model
    orig_samp = gmr.GPUModelRunner.sample_tokens
    orig_prop = dfspec.DFlashSpeculator.propose

    def _prof_exec(self, *args, **kwargs):
        if (
            not state["ready"]
            or kwargs.get("dummy_run", False)
            or get_tensor_model_parallel_rank() != 0
            or state["step_prints"] >= 80
        ):
            return orig_exec(self, *args, **kwargs)
        sched = args[0] if args else kwargs.get("scheduler_output")
        n_tok = getattr(sched, "total_num_scheduled_tokens", 0) if sched is not None else 0
        if n_tok <= 0:
            return orig_exec(self, *args, **kwargs)
        now = time.perf_counter()
        gap = (now - state["last_step_t"]) * 1000.0 if state["last_step_t"] else 0.0
        state["last_step_t"] = now
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        out = orig_exec(self, *args, **kwargs)
        e1.record()
        torch.cuda.synchronize()
        state["last_exec_ms"] = e0.elapsed_time(e1)
        state["last_n_tok"] = n_tok
        state["last_gap_ms"] = gap
        state["last_prop_ms"] = 0.0
        return out

    def _prof_prop(self, *args, **kwargs):
        if (
            not state["ready"]
            or kwargs.get("dummy_run", False)
            or get_tensor_model_parallel_rank() != 0
            or state["step_prints"] >= 80
        ):
            return orig_prop(self, *args, **kwargs)
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        out = orig_prop(self, *args, **kwargs)
        e1.record()
        torch.cuda.synchronize()
        state["last_prop_ms"] = e0.elapsed_time(e1)
        return out

    def _prof_samp(self, *args, **kwargs):
        if (
            not state["ready"]
            or get_tensor_model_parallel_rank() != 0
            or state["step_prints"] >= 80
        ):
            return orig_samp(self, *args, **kwargs)
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        out = orig_samp(self, *args, **kwargs)
        e1.record()
        torch.cuda.synchronize()
        s_ms = e0.elapsed_time(e1)
        n_tok = state.get("last_n_tok", 0)
        if n_tok >= 512 or (state["step_prints"] < 50):
            _log(
                f"STEP_PROF #{state['step_prints']:02d} n_tok={n_tok:4d} "
                f"exec={state.get('last_exec_ms', 0):.2f}ms samp+prop={s_ms:.2f}ms "
                f"(prop={state.get('last_prop_ms', 0):.2f}ms) wall_period={state.get('last_gap_ms', 0):.2f}ms"
            )
            state["step_prints"] += 1
        return out

    gmr.GPUModelRunner.execute_model = _prof_exec
    gmr.GPUModelRunner.sample_tokens = _prof_samp
    dfspec.DFlashSpeculator.propose = _prof_prop

    _log("installed live sub-layer + step profiler (L1 HCA, L7 DSA, cost_curves, exec/samp/propose)")
    return True


def _evict_safetensors_page_cache(model_path: str | None) -> int:
    """Evict clean Linux page-cache pages for local .safetensors shards (no root required).

    On NVIDIA DGX Spark (GB10 unified memory), reading 154 GiB of safetensors leaves
    3-6 GiB of clean file page cache in unified RAM right when vLLM calls
    cudaMemGetInfo() during memory profiling, causing vLLM to misclassify reclaimable
    OS page cache as permanent non_torch_memory and starve KV cache allocation.
    """
    if not model_path or not (hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED")):
        return 0
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        try:
            from huggingface_hub import snapshot_download  # type: ignore

            local_dir = snapshot_download(model_path, local_files_only=True)
            model_dir = Path(local_dir)
        except Exception:
            return 0
    if not model_dir.is_dir():
        return 0
    evicted = 0
    for sf in model_dir.glob("*.safetensors"):
        try:
            fd = os.open(str(sf), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                evicted += 1
            finally:
                os.close(fd)
        except OSError:
            continue
    return evicted


def _patch_gb10_uma_fadvise() -> bool:
    """Prevent GB10 unified-RAM safetensors page cache from starving vLLM KV cache sizing."""
    try:
        from vllm.v1.worker import gpu_worker as gw  # type: ignore
    except Exception:
        return False

    worker_cls = getattr(gw, "Worker", None)
    if worker_cls is None or getattr(worker_cls, "_spark_uma_fadvise_patched", False):
        return False

    orig_init_device = worker_cls.init_device
    orig_load_model = worker_cls.load_model
    orig_determine_mem = worker_cls.determine_available_memory

    def _patched_init_device(self, *args, **kwargs):
        try:
            model_path = getattr(getattr(self, "model_config", None), "model", None)
            _evict_safetensors_page_cache(model_path)
        except Exception:
            pass
        return orig_init_device(self, *args, **kwargs)

    def _patched_load_model(self, *args, **kwargs):
        res = orig_load_model(self, *args, **kwargs)
        try:
            import gc
            import torch

            gc.collect()
            free_before = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
            model_path = getattr(getattr(self, "model_config", None), "model", None)
            n = _evict_safetensors_page_cache(model_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                free_after = torch.cuda.mem_get_info()[0]
                delta_gib = (free_after - free_before) / (1024 ** 3)
                if n > 0:
                    _log(
                        f"GB10 UMA page-cache eviction: fadvise(DONTNEED) on {n} safetensors shards "
                        f"recovered {delta_gib:+.2f} GiB cudaMemGetInfo free "
                        f"({free_before / 1024**3:.2f} -> {free_after / 1024**3:.2f} GiB)"
                    )
        except Exception:
            pass
        return res

    def _patched_determine_available_memory(self, *args, **kwargs):
        # Upstream vLLM's kv_cache_memory_bytes fast-path runs profile_run() WITHOUT
        # _scoped_allocator_max_split(20) or torch.cuda.empty_cache(), leaving ~12-14 GiB
        # of transient activation segments pinned in the caching allocator right when
        # KV cache allocation runs (causing OOM spikes on 121.7 GiB GB10 unified memory).
        kv_bytes = getattr(getattr(self, "cache_config", None), "kv_cache_memory_bytes", None)
        if kv_bytes and hasattr(self, "_scoped_allocator_max_split"):
            with self._scoped_allocator_max_split(max_split_size_mb=20):
                res = orig_determine_mem(self, *args, **kwargs)
        else:
            res = orig_determine_mem(self, *args, **kwargs)
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                free_before = torch.cuda.mem_get_info()[0]
                torch.cuda.empty_cache()
                free_after = torch.cuda.mem_get_info()[0]
                delta_gib = (free_after - free_before) / (1024 ** 3)
                if delta_gib > 0.25:
                    _log(
                        f"GB10 post-profile empty_cache() freed {delta_gib:+.2f} GiB "
                        f"({free_before / 1024**3:.2f} -> {free_after / 1024**3:.2f} GiB free before KV alloc)"
                    )
        except Exception:
            pass
        return res

    worker_cls.init_device = _patched_init_device
    worker_cls.load_model = _patched_load_model
    worker_cls.determine_available_memory = _patched_determine_available_memory
    worker_cls._spark_uma_fadvise_patched = True
    _log("installed GB10 UMA safetensors page-cache eviction + post-profile empty_cache fix (zero root)")
    return True


def _patch_anthropic_and_api_compat() -> bool:
    """Fix Anthropic /v1/messages, OpenAI Chat/Responses reasoning_effort, x-api-key auth & prefix cache.

    Solves 3 real upstream vLLM 0.32.0 + ERNIE 5.5 bugs:
      1. AuthenticationMiddleware only checks `Authorization: Bearer ...` and rejects
         Anthropic SDK / Claude Code's `x-api-key` header with 401.
      2. AnthropicServingMessages._detect_merge_inline_system(None) returns True for
         ERNIE 5.5's Python renderer (chat_template is None), merging inline system
         reminders into message[0] and destroying 100% of KV prefix cache every turn —
         even though encoding_ernie.py natively renders inline `role == "system" and i > 0`!
      3. encoding_ernie.py only accepts reasoning_effort in ('low', 'medium', 'high') and
         raises ValueError on Anthropic's `thinking: {"type": "disabled"}` (which sets
         reasoning_effort='none') or OpenAI/Anthropic 'off'/'minimal'/'max'.
    """
    patched_any = False

    # 1. Accept Anthropic `x-api-key` header alongside `Authorization: Bearer`
    try:
        import hashlib
        import secrets
        from vllm.entrypoints.serve.middleware import authenticate as auth_mod  # type: ignore

        mw_cls = getattr(auth_mod, "AuthenticationMiddleware", None)
        if mw_cls is not None and not getattr(mw_cls, "_spark_x_api_key_patched", False):
            orig_verify = mw_cls.verify_token

            def _verify_token_with_anthropic(self, headers) -> bool:
                if orig_verify(self, headers):
                    return True
                x_key = headers.get("x-api-key")
                if not x_key:
                    return False
                param_hash = hashlib.sha256(x_key.encode("utf-8")).digest()
                ok = False
                for token_hash in self.api_tokens:
                    ok |= secrets.compare_digest(param_hash, token_hash)
                return ok

            mw_cls.verify_token = _verify_token_with_anthropic
            mw_cls._spark_x_api_key_patched = True
            patched_any = True
    except Exception:
        pass

    # 2. Prevent Anthropic /v1/messages from merging inline system messages on ERNIE 5.5
    try:
        from vllm.entrypoints.anthropic import serving as anth_serving  # type: ignore

        anth_cls = getattr(anth_serving, "AnthropicServingMessages", None)
        if anth_cls is not None and not getattr(anth_cls, "_spark_ernie_anth_patched", False):
            orig_detect = anth_cls._detect_merge_inline_system

            @staticmethod
            def _detect_merge_inline_system_ernie(chat_template: str | None) -> bool:
                # ERNIE 5.5 uses Python Ernie5_5Renderer (chat_template=None) which natively
                # renders inline system messages (`role == "system" and i > 0`), preserving
                # full KV prefix cache on Claude Code / Anthropic multi-turn turns!
                if not chat_template:
                    return False
                return orig_detect(chat_template)

            anth_cls._detect_merge_inline_system = _detect_merge_inline_system_ernie
            anth_cls._spark_ernie_anth_patched = True
            patched_any = True
    except Exception:
        pass

    # 3. Normalize reasoning_effort across Chat, Responses & Anthropic /v1/messages for ERNIE 5.5
    try:
        import dataclasses
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest  # type: ignore
        from vllm.entrypoints.openai.responses.protocol import ResponsesRequest  # type: ignore

        _EFFORT_MAP = {
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "max": "high",
            "xhigh": "high",
        }
        _DISABLE_EFFORTS = frozenset({"none", "off", "disabled", "false", "0"})

        for req_cls in (ChatCompletionRequest, ResponsesRequest):
            orig_build = req_cls.build_chat_params
            if getattr(orig_build, "_spark_ernie_effort_patched", False):
                continue

            def _make_wrapper(fn):
                def _wrapped_build_chat_params(self, *args, **kwargs):
                    params = fn(self, *args, **kwargs)
                    tk = dict(params.chat_template_kwargs or {})
                    raw_eff = tk.get("reasoning_effort", getattr(self, "reasoning_effort", None))
                    if raw_eff is None and hasattr(self, "reasoning") and self.reasoning is not None:
                        raw_eff = getattr(self.reasoning, "effort", None)
                    if isinstance(raw_eff, str):
                        eff_l = raw_eff.strip().lower()
                        if eff_l in _DISABLE_EFFORTS:
                            tk["enable_thinking"] = False
                            tk.pop("reasoning_effort", None)
                            return dataclasses.replace(params, chat_template_kwargs=tk)
                        if eff_l in _EFFORT_MAP and _EFFORT_MAP[eff_l] != raw_eff:
                            tk["reasoning_effort"] = _EFFORT_MAP[eff_l]
                            return dataclasses.replace(params, chat_template_kwargs=tk)
                    return params

                _wrapped_build_chat_params._spark_ernie_effort_patched = True  # type: ignore[attr-defined]
                return _wrapped_build_chat_params

            req_cls.build_chat_params = _make_wrapper(orig_build)
            patched_any = True
    except Exception:
        pass

    if patched_any:
        _log(
            "installed Anthropic /v1/messages + OpenAI Chat/Responses compat "
            "(x-api-key auth, inline-system prefix cache preserved, thinking=disabled/effort mapping)"
        )
    return patched_any


def install() -> dict[str, bool]:
    global _INSTALLED
    if _INSTALLED:
        return {}
    _INSTALLED = True

    repo_str = str(_REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    applied: dict[str, bool] = {}
    if _env_flag("SPARK_PATCH_FLASHINFER_CPB", "1"):
        applied["flashinfer_cpb"] = _patch_flashinfer_sm120_cpb()
    if _env_flag("SPARK_PATCH_ROUTER", "1"):
        applied["fused_router"] = _patch_fused_ernie55_router()
    if _env_flag("SPARK_PATCH_MHC", "1"):
        applied["mhc_gb10"] = _patch_mhc_gb10()
    if _env_flag("SPARK_PATCH_HUMMING_FP8_SAFE", "1"):
        applied["humming_fp8_safe"] = _patch_humming_fp8_safe()
    if _env_flag("SPARK_REPLICATE_SHARED_EXPERTS", "0"):
        applied["replicate_shared_experts"] = _patch_replicate_shared_experts()
    if _env_flag("SPARK_PATCH_DSPARK_MARKOV", "1"):
        applied["dspark_markov"] = _patch_dspark_markov_gathered()
    if _env_flag("SPARK_PATCH_ADAPTIVE_VERIFY", "1"):
        applied["adaptive_verify"] = _patch_dspark_adaptive_verification()
    if _env_flag("SPARK_USE_SPARKNET", "0"):
        applied["sparknet_oneshot"] = _patch_sparknet_oneshot()
    if _env_flag("SPARK_DSPARK_MAIN_PROJ_TP", "1"):
        applied["dspark_main_proj_tp"] = _patch_dspark_main_proj_tp()
    if _env_flag("SPARK_DSPARK_VOCAB_PARALLEL", "0"):
        applied["dspark_vocab_parallel"] = _patch_dspark_vocab_parallel_greedy()
    if _env_flag("SPARK_PATCH_KV_PACK", "1"):
        applied["kv_cache_packing"] = _patch_kv_cache_packing()
    if _env_flag("SPARK_PACKED_BF16_LM_HEAD", "0"):
        applied["packed_bf16_lm_head"] = _patch_packed_bf16_lm_head()
    if _env_flag("SPARK_PATCH_DSPARK_PAD_MASK", "1"):
        applied["dspark_pad_mask"] = _patch_dspark_drafter_padding_mask()
    if _env_flag("SPARK_PATCH_VHA_BF16_AR", "0"):
        applied["vha_bf16_allreduce"] = _patch_vha_bf16_allreduce()
    if _env_flag("SPARK_PATCH_FC2_LATENT_TP", "0"):
        applied["fc2_latent_tp"] = _patch_fc2_latent_tp()
    if _env_flag("SPARK_PATCH_FC1_LATENT_TP", "0"):
        applied["fc1_latent_tp"] = _patch_fc1_latent_tp()
    if _env_flag("SPARK_PATCH_GATE_BF16X2", "1"):
        applied["gate_bf16x2"] = _patch_gate_bf16x2()
    if _env_flag("SPARK_PATCH_HCA_OPROJ_FUSED", "1"):
        applied["hca_oproj_fused"] = _patch_hca_oproj_fused()
    if _env_flag("SPARK_PATCH_UMA_FADVISE", "1"):
        applied["gb10_uma_fadvise"] = _patch_gb10_uma_fadvise()
    if _env_flag("SPARK_PATCH_ANTHROPIC_COMPAT", "1"):
        applied["anthropic_api_compat"] = _patch_anthropic_and_api_compat()
    if _env_flag("SPARK_PROFILE_PREFILL", "0"):
        applied["profile_prefill"] = _patch_profile_prefill()

    _log(f"install complete: {applied}")
    return applied

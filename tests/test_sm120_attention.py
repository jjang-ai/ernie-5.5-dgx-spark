"""Real GB10 regression tests. No doubles and no model-weight download."""
import math
import os

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    init_distributed_environment, ensure_model_parallel_initialized,
    destroy_model_parallel, destroy_distributed_environment,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType


@pytest.fixture(scope="module", autouse=True)
def runtime(tmp_path_factory):
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(0)
    rendezvous = tmp_path_factory.mktemp("dist") / "init"
    init_distributed_environment(world_size=1, rank=0,
                                 distributed_init_method=f"file://{rendezvous}", local_rank=0)
    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(1, 1)
        from spark_ernie_runtime import install
        install()
        yield
    destroy_model_parallel()
    destroy_distributed_environment()


def test_dsa_admits_real_sm121_with_sinks():
    from spark_ernie_runtime.sm120_attention import install
    assert install()
    from vllm_ernie.models.ernie5_5.nvidia import attention as a
    a._check_sink_backend(576, "fp8_ds_mla")


def test_hca_admits_real_sm121_with_sinks():
    from spark_ernie_runtime.sm120_attention import install
    assert install()
    from vllm_ernie.models.ernie5_5.nvidia import attention as a
    backend = a.select_ernie5_5_hca_cls("fp8_ds_mla").backend_cls
    assert not backend.validate_configuration(
        head_size=512, dtype=torch.bfloat16, kv_cache_dtype="fp8_ds_mla",
        block_size=256, use_mla=True, has_sink=True, use_sparse=True,
        use_mm_prefix=False, use_per_head_quant_scales=False,
        device_capability=current_platform.get_device_capability(),
        attn_type=AttentionType.DECODER,
    )


def reference(q, keys, values, indices, sinks, scale, lengths=None):
    # Independent FP32 softmax with a zero-valued learned-sink token.
    idx = indices.reshape(q.shape[0], -1).long()
    valid = idx >= 0
    if lengths is not None:
        valid &= torch.arange(idx.shape[1], device="cuda")[None] < lengths[:, None]
    k = keys[idx.clamp_min(0)].float()
    v = values[idx.clamp_min(0)].float()
    logits = torch.einsum("thd,tkd->thk", q.float(), k) * scale
    logits.masked_fill_(~valid[:, None], -torch.inf)
    sink_logits = sinks[None, :, None].expand(q.shape[0], -1, 1)
    weights = torch.softmax(torch.cat((logits, sink_logits), -1), -1)[..., :-1]
    return torch.einsum("thk,tkd->thd", weights, v)


def dsa_cache(pages=40):
    torch.manual_seed(416)
    nope = (0.4 + 0.2 * torch.randn(pages, 64, 512, device="cuda")).to(torch.float8_e4m3fn)
    rope = (0.2 * torch.randn(pages, 64, 64, device="cuda")).bfloat16()
    cache = torch.zeros(pages, 64, 656, dtype=torch.uint8, device="cuda")
    cache[..., :512] = nope.view(torch.uint8)
    cache[..., 512:528].view(torch.float32).fill_(0.5)
    cache[..., 528:] = rope.view(torch.uint8)
    keys = torch.cat((nope.float() * 0.5, rope.float()), -1).reshape(-1, 576)
    return cache, keys, keys[:, :512]


def make_dsa(sinks, indices):
    from vllm_ernie.models.ernie5_5.nvidia import attention as a
    # Fail at the real admission boundary before any unsupported kernel launch.
    a._check_sink_backend(576, "fp8_ds_mla")
    proj = ColumnParallelLinear(512, sinks.numel() * (192 + 256), bias=False)
    return a.Ernie5_5FlashMLASparseBackend.get_impl_cls()(
        sinks.numel(), 576, 256 ** -0.5, 1, None, None, "fp8_ds_mla",
        None, AttentionType.DECODER, None,
        q_lora_rank=1536, kv_lora_rank=512,
        qk_nope_head_dim=192, qk_rope_head_dim=64, qk_head_dim=256,
        v_head_dim=256, kv_b_proj=proj, topk_indices_buffer=indices, sinks=sinks,
    )


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("tokens,length", [(1, 1), (2, 63), (4, 129), (8, 2176), (16, 127), (33, 2176)])
def test_dsa_learned_sinks_match_torch(tokens, length, heads):
    cache, keys, values = dsa_cache()
    torch.manual_seed(932)
    q = (torch.randn(tokens, heads, 576, device="cuda") * 0.1).bfloat16()
    indices = torch.full((tokens, 2176), -1, dtype=torch.int32, device="cuda")
    # Per-row tails and invalid indices exercise masking in both kernel paths.
    for t in range(tokens):
        n = max(1, length - t % 3)
        indices[t, :n] = torch.randperm(keys.shape[0], device="cuda")[:n].int()
    sinks = torch.nn.Parameter(torch.linspace(-1, 3, heads, device="cuda", dtype=torch.float32) + math.log(length), requires_grad=False)
    impl = make_dsa(sinks, indices)
    # The loaded Parameter must remain live, not a copy made before weight load.
    sinks.data.add_(0.375)
    actual = impl._run_mqa_kernel(q, cache, indices)
    expected = reference(q, keys, values, indices, sinks, impl.scale)
    torch.testing.assert_close(actual.float(), expected, atol=0.004, rtol=0.025)
    without_sink = reference(q, keys, values, indices, torch.full_like(sinks, -torch.inf), impl.scale)
    assert (expected - without_sink).abs().max().item() > 0.08


@pytest.fixture(scope="module", params=[32, 64])
def hca(runtime, request):
    from vllm.engine.arg_utils import EngineArgs
    from vllm_ernie.models.ernie5_5.nvidia import attention as a
    checkpoint = os.environ.get(
        "ERNIE_TEST_MODEL", "dealignai/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4"
    )
    config = EngineArgs(model=checkpoint, dtype="bfloat16", max_model_len=4096,
                        max_num_batched_tokens=512, max_num_seqs=8,
                        kv_cache_dtype="fp8_ds_mla", enforce_eager=True,
                        skip_tokenizer_init=True).create_engine_config()
    # Attention math uses real layers without checkpoint weights or quantized GEMMs.
    config.quant_config = None
    config.model_config.hf_config.num_attention_heads = request.param
    with set_current_vllm_config(config), torch.device("cuda"):
        layer = a.select_ernie5_5_hca_cls("fp8_ds_mla")(
            config, "model.layers.1.self_attn",
            topk_indices_buffer=torch.empty(512, 64, device="cuda", dtype=torch.int32))
        yield layer


def hca_cache(pages, page_size):
    # DSv4 packs 576 data bytes per row, followed by an 8-byte/row scale footer.
    # Page padding deliberately reproduces vLLM's aligned compressed pages.
    nope = (0.4 + 0.2 * torch.randn(pages, page_size, 448, device="cuda")).to(torch.float8_e4m3fn)
    rope = (0.4 + 0.2 * torch.randn(pages, page_size, 64, device="cuda")).bfloat16()
    page_bytes = math.ceil(page_size * 584 / 576) * 576
    raw = torch.zeros(pages, page_bytes, device="cuda", dtype=torch.uint8)
    rows = raw[:, :page_size * 576].view(pages, page_size, 576)
    rows[..., :448] = nope.view(torch.uint8)
    rows[..., 448:] = rope.view(torch.uint8)
    raw[:, page_size * 576:page_size * 584].view(pages, page_size, 8)[..., :7] = 127
    cache = raw.as_strided((pages, page_size, 584), (page_bytes, 584, 1))
    values = torch.cat((nope.float(), rope.float()), -1).reshape(-1, 512)
    return cache, values


@pytest.mark.parametrize("mode,tokens,length,extra", [
    ("decode", 1, 1, 0), ("decode", 4, 63, 2),
    ("prefill", 8, 127, 5), ("prefill", 33, 128, 16),
])
def test_hca_learned_sinks_match_torch(hca, mode, tokens, length, extra):
    from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadata
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata
    torch.manual_seed(713)
    primary, primary_values = hca_cache(4, 64)
    compressed, compressed_values = hca_cache(8, 2)
    q = (0.1 * torch.randn(tokens, hca.padded_heads, 512, device="cuda")).bfloat16()
    idx = torch.full((tokens, 1, 128), -1, device="cuda", dtype=torch.int32)
    idx[:, 0, :length] = torch.arange(length, device="cuda", dtype=torch.int32)
    lens = torch.full((tokens,), length, device="cuda", dtype=torch.int32)
    extra_idx = torch.full((tokens, 64), -1, device="cuda", dtype=torch.int32)
    extra_idx[:, :extra] = torch.arange(extra, device="cuda", dtype=torch.int32)
    extra_lens = torch.full((tokens,), extra, device="cuda", dtype=torch.int32)
    empty = torch.empty(0, device="cuda", dtype=torch.int64)
    is_decode = mode == "decode"
    nreqs = tokens if is_decode else 1
    cu_cpu = torch.arange(tokens + 1, dtype=torch.int32) if is_decode else torch.tensor([0, tokens], dtype=torch.int32)
    swa = DeepseekSparseSWAMetadata(
        block_table=torch.zeros(nreqs, 4, dtype=torch.int32, device="cuda"),
        slot_mapping=empty, block_size=64,
        is_valid_token=torch.ones(tokens, device="cuda", dtype=torch.bool),
        token_to_req_indices=torch.zeros(tokens, device="cuda", dtype=torch.int32),
        query_start_loc_cpu=cu_cpu, query_start_loc=cu_cpu.cuda(),
        decode_swa_indices=idx if is_decode else None,
        decode_swa_lens=lens if is_decode else None,
        prefill_swa_indices=None if is_decode else idx,
        prefill_swa_lens=None if is_decode else lens,
        num_decodes=nreqs if is_decode else 0, num_prefills=0 if is_decode else 1,
        num_decode_tokens=tokens if is_decode else 0,
        num_prefill_tokens=0 if is_decode else tokens,
    )
    meta = DeepseekV4FlashMLAMetadata(
        num_reqs=nreqs, max_query_len=1 if is_decode else tokens, max_seq_len=2048,
        num_actual_tokens=tokens, query_start_loc=cu_cpu.cuda(), slot_mapping=empty,
        block_table=torch.arange(8, dtype=torch.int32, device="cuda")[None].expand(nreqs, -1),
        req_id_per_token=torch.zeros(tokens, dtype=torch.int32, device="cuda"),
        block_size=256, topk_tokens=64,
        c128a_global_decode_topk_indices=extra_idx.unsqueeze(1) if is_decode else None,
        c128a_decode_topk_lens=extra_lens if is_decode else None,
        c128a_prefill_topk_indices=None if is_decode else extra_idx,
    )
    hca.swa_cache_layer.kv_cache = primary
    hca.attn_sink.data.copy_(torch.linspace(-1, 3, hca.padded_heads, device="cuda", dtype=torch.float32) + math.log(length + extra))
    out = torch.empty_like(q)
    if is_decode:
        hca._forward_decode(q, compressed, swa, meta, False, out)
    else:
        hca._forward_prefill(q, compressed, primary, out, meta, swa)
    joined_idx = torch.cat((idx.squeeze(1), torch.where(extra_idx >= 0, extra_idx + primary_values.shape[0], -1)), -1)
    values = torch.cat((primary_values, compressed_values))
    expected = reference(q, values, values, joined_idx, hca.attn_sink, hca.scale)
    torch.testing.assert_close(out.float(), expected, atol=0.007, rtol=0.03)
    without_sink = reference(q, values, values, joined_idx, torch.full_like(hca.attn_sink, -torch.inf), hca.scale)
    assert (expected - without_sink).abs().max().item() > 0.1

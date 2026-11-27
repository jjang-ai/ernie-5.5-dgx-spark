"""Vocabulary-parallel greedy Markov argmax for DSpark on DGX Spark (GB10, SM12.1).

Ported cleanly from christopherowen/spark-ds41f (Apache-2.0, patch 0012).

Each TP rank keeps its own vocabulary shard of both draft base_logits and markov_bias
(no full-vocab all-gather, 2x-4x less markov_w2 DRAM read, zero top-k pruning loss!),
reduces (max_val, global_argmax_idx) per request in one fused Triton pass (8 bytes/req),
all-gathers ONLY the tiny [B, 2*TP] pairs over sparknet one-shot RoCE (10-15 us),
and resolves exact global first-maximum (ties + NaNs bit-identical to full-vocab argmax).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 4096


@triton.jit
def _first_argmax(val_a, idx_a, val_b, idx_b):
    # Match torch.argmax: NaN beats non-NaN; ties keep lower index.
    a_nan = val_a != val_a
    b_nan = val_b != val_b
    take_b = (b_nan & (~a_nan)) | ((~a_nan) & ((val_b > val_a) | ((val_b == val_a) & (idx_b < idx_a))))
    return tl.where(take_b, val_b, val_a), tl.where(take_b, idx_b, idx_a)


@triton.jit
def _partial_argmax(
    base_ptr,
    bias_ptr,
    out_val_ptr,
    out_idx_ptr,
    base_stride,
    bias_stride,
    vocab,
    parts,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    offs = part * BLOCK + tl.arange(0, BLOCK)
    mask = offs < vocab
    b = tl.load(base_ptr + row * base_stride + offs, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.load(bias_ptr + row * bias_stride + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.where(mask, b + m, -float("inf"))
    idx = tl.where(mask, offs, 2147483647).to(tl.int32)
    best_v, best_i = tl.reduce((s, idx), 0, _first_argmax)
    tl.store(out_val_ptr + row * parts + part, best_v)
    tl.store(out_idx_ptr + row * parts + part, best_i)


@triton.jit
def _finish_argmax_pair(
    values,
    indices,
    pairs,
    vocab_start,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    value = tl.load(values + row * PARTS + part, part < PARTS, other=-float("inf"))
    index = tl.load(indices + row * PARTS + part, part < PARTS, other=2147483647)
    best, result = tl.reduce((value, index), 0, _first_argmax)
    tl.store(pairs + row * 2, best)
    tl.store(pairs + row * 2 + 1, (result + vocab_start).to(tl.float32))


@triton.jit
def _select_first_argmax(
    pairs,
    output,
    output_stride,
    RANKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    rank = tl.arange(0, BLOCK)
    value = tl.load(pairs + row * 2 * RANKS + 2 * rank, rank < RANKS, other=-float("inf"))
    index = tl.load(pairs + row * 2 * RANKS + 2 * rank + 1, rank < RANKS, other=1073741824.0)
    _, result = tl.reduce((value, index.to(tl.int32)), 0, _first_argmax)
    tl.store(output + row * output_stride, result.to(tl.int64))


def scratch_shape(max_requests: int, shard_vocab: int) -> tuple[int, int]:
    return max_requests, triton.cdiv(shard_vocab, _BLOCK)


def greedy_markov_shard_pair(
    base_logits: torch.Tensor,
    markov_bias: torch.Tensor,
    vocab_start: int,
    pairs: torch.Tensor,
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
) -> None:
    rows, vocab = base_logits.shape
    parts = triton.cdiv(vocab, _BLOCK)
    _partial_argmax[(rows, parts)](
        base_logits,
        markov_bias,
        partial_values,
        partial_indices,
        base_logits.stride(0),
        markov_bias.stride(0),
        vocab,
        parts,
        _BLOCK,
    )
    _finish_argmax_pair[(rows,)](
        partial_values,
        partial_indices,
        pairs,
        vocab_start,
        parts,
        triton.next_power_of_2(parts),
    )


def select_first_argmax(pairs: torch.Tensor, output: torch.Tensor) -> None:
    rows, width = pairs.shape
    ranks = width // 2
    _select_first_argmax[(rows,)](
        pairs,
        output,
        output.stride(0),
        ranks,
        triton.next_power_of_2(ranks),
    )

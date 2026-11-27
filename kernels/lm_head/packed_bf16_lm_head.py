"""Exact 12-bit packed BF16 vocabulary head for DGX Spark (GB10, SM12.1).

Adapted cleanly from christopherowen/spark-ds41f (Apache-2.0, patches 0028 & 0036)
with a standalone PyTorch bit-exact packer and TileLang SM12.1 kernel (no b12x dep).

Every BF16 weight (16 bits: 1 sign + 8 exponent + 7 mantissa) is packed losslessly into:
  - 1 sign+mantissa byte (bit 7 = sign, bits 0..6 = mantissa)
  - 4-bit exponent code (0 => exact 0, 1..15 => exponent_base + code - 1)
  - Per-row CSR exception list for the rare (<0.01%) values outside the 15-exponent window
    or non-zero subnormals, so unpacking reproduces EVERY single BF16 bit identically!

Saves 25% of LM-head DRAM bandwidth (ERNIE 5.5 lm_head [201216, 4096] BF16:
1,648 MB -> 1,236 MB total; 824 MB -> 618 MB/GPU on TP=2; 412 MB -> 309 MB/GPU on TP=4).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

GROUP = 128
HALF = GROUP // 2
VEC = 16
ROW_TILES = (16, 32, 64)
SHARED_MEMORY = 101_376

TILE_CONFIGS = {
    16: dict(block_N=32, threads=128, step_groups=2, num_stages=4),
    32: dict(block_N=16, threads=128, step_groups=2, num_stages=4),
    64: dict(block_N=32, threads=256, step_groups=1, num_stages=4),
}


@dataclass
class PackedBF16Weight:
    values: torch.Tensor              # [N, K + K//2] uint8
    exception_rows: torch.Tensor      # [E] int32
    exception_entries: torch.Tensor   # [E] int32 ((col << 16) | uint16_bits)
    row_offsets: torch.Tensor         # [N + 1] int32
    exponent_base: int
    in_features: int
    num_exceptions: int


def _choose_exponent_base(bits_u16: torch.Tensor) -> int:
    """Choose the 15-exponent window [base, base + 14] minimizing out-of-window exceptions."""
    exp = ((bits_u16.to(torch.int32) >> 7) & 0xFF).reshape(-1)
    # Exclude exact zeros (exp==0 & mant==0) from window selection
    nonzero_exp = exp[exp > 0]
    if nonzero_exp.numel() == 0:
        return 118
    hist = torch.bincount(nonzero_exp, minlength=256)
    # Sliding sum of width 15 over exponents 1..254 (never include inf/nan=255)
    best_base = 118
    best_count = -1
    h_cpu = hist.cpu().tolist()
    win = sum(h_cpu[1:16])
    best_base, best_count = 1, win
    for base in range(2, 241):
        win += h_cpu[base + 14] - h_cpu[base - 1]
        if win > best_count:
            best_count = win
            best_base = base
    return best_base


@torch.no_grad()
def pack_bf16_exact(weight: torch.Tensor, verify: bool = True) -> PackedBF16Weight:
    """Losslessly pack a 2D BF16 matrix [N, K] (K % 128 == 0) into 12-bit packed form."""
    assert weight.ndim == 2 and weight.dtype == torch.bfloat16
    N, K = weight.shape
    assert K % GROUP == 0, f"K={K} must be divisible by {GROUP}"
    dev = weight.device

    bits = weight.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    sign = (bits >> 15) & 0x1
    exp = (bits >> 7) & 0xFF
    mant = bits & 0x7F

    base = _choose_exponent_base(bits)
    is_exact_zero = (exp == 0) & (mant == 0)
    in_window = (exp >= base) & (exp <= base + 14)
    is_exc = ~(is_exact_zero | in_window)

    # Low byte: sign in bit 7, mantissa in bits 0..6 (zeroed for exceptions)
    low = torch.where(is_exc, torch.zeros_like(bits), (sign << 7) | mant).to(torch.uint8)
    # 4-bit exponent code: 0 for exact zero or exception, (exp - base + 1) in 1..15 otherwise
    code = torch.where(in_window, exp - base + 1, torch.zeros_like(bits)).to(torch.uint8)

    # Pack codes in 128-col groups: low nibble = col j, high nibble = col j + 64
    code_g = code.view(N, K // GROUP, 2, HALF)
    packed_codes = (code_g[:, :, 0, :] | (code_g[:, :, 1, :] << 4)).reshape(N, K // 2)

    packed_values = torch.empty((N, K + K // 2), dtype=torch.uint8, device=dev)
    packed_values[:, :K] = low
    packed_values[:, K:] = packed_codes

    # CSR exception list sorted by row
    exc_rc = torch.nonzero(is_exc, as_tuple=False)  # [E, 2]
    if exc_rc.numel() > 0:
        exc_rows = exc_rc[:, 0].to(torch.int32).contiguous()
        exc_cols = exc_rc[:, 1].to(torch.int32)
        exc_bits = bits[exc_rc[:, 0], exc_rc[:, 1]].to(torch.int32)
        exc_entries = ((exc_cols << 16) | (exc_bits & 0xFFFF)).to(torch.int32).contiguous()
        row_counts = is_exc.sum(dim=1, dtype=torch.int32)
    else:
        # Allocate 1 dummy element so empty tensor pointers remain valid in TileLang
        exc_rows = torch.zeros((1,), dtype=torch.int32, device=dev)
        exc_entries = torch.zeros((1,), dtype=torch.int32, device=dev)
        row_counts = torch.zeros((N,), dtype=torch.int32, device=dev)

    row_offsets = torch.zeros((N + 1,), dtype=torch.int32, device=dev)
    torch.cumsum(row_counts, dim=0, out=row_offsets[1:])
    num_exc = int(exc_rc.shape[0])

    packed = PackedBF16Weight(
        values=packed_values,
        exception_rows=exc_rows,
        exception_entries=exc_entries,
        row_offsets=row_offsets,
        exponent_base=base,
        in_features=K,
        num_exceptions=num_exc,
    )
    if verify:
        unpacked = unpack_bf16_exact(packed)
        assert torch.equal(unpacked.view(torch.int16), weight.view(torch.int16)), (
            "12-bit packed BF16 round-trip bit verification FAILED!"
        )
    return packed


@torch.no_grad()
def unpack_bf16_exact(packed: PackedBF16Weight) -> torch.Tensor:
    """Reconstruct exact BF16 tensor from PackedBF16Weight (for bit-exact verification)."""
    N = packed.values.shape[0]
    K = packed.in_features
    low = packed.values[:, :K].to(torch.int32)
    pc = packed.values[:, K:].view(N, K // GROUP, HALF).to(torch.int32)
    code_lo = pc & 0xF
    code_hi = (pc >> 4) & 0xF
    code = torch.stack((code_lo, code_hi), dim=2).reshape(N, K)
    exp = torch.where(code == 0, torch.zeros_like(code), code + (packed.exponent_base - 1))
    bits = ((low & 0x80) << 8) | (exp << 7) | (low & 0x7F)
    if packed.num_exceptions > 0:
        r = packed.exception_rows[: packed.num_exceptions].to(torch.int64)
        e = packed.exception_entries[: packed.num_exceptions].to(torch.int32)
        c = (e >> 16) & 0xFFFF
        b = e & 0xFFFF
        bits[r, c.to(torch.int64)] = b
    return bits.to(torch.int16).view(torch.bfloat16)


def _tile_config(block_M: int, k: int) -> dict:
    cfg = dict(TILE_CONFIGS[block_M])
    if k % (GROUP * cfg["step_groups"]):
        cfg["step_groups"] = 1
    return cfg


def _row_tile(rows: int) -> int:
    for bm in ROW_TILES:
        if rows <= bm:
            return bm
    return ROW_TILES[-1]


def build_packed_vocab_kernel(
    N: int,
    K: int,
    exponent_base: int,
    block_M: int = 16,
    block_N: int = 32,
    num_stages: int = 4,
    threads: int = 128,
    step_groups: int = 2,
    out_dtype: str = "bfloat16",
):
    """Compile TileLang SM12.1 12-bit packed BF16 vocab projection kernel."""
    import tilelang
    import tilelang.language as T

    assert K % (GROUP * step_groups) == 0
    steps = K // (GROUP * step_groups)
    STEP = GROUP * step_groups
    row_bytes = K + K // 2
    base_minus_one = exponent_base - 1
    slices = HALF // VEC
    items = block_N * step_groups * slices
    tl_out_dtype = T.float32 if out_dtype == "float32" else T.bfloat16

    def _decode(low, code):
        low_i = T.Cast(T.int32, low)
        code_i = T.Cast(T.int32, code)
        exponent = T.if_then_else(code_i == 0, 0, code_i + base_minus_one)
        bits = ((low_i & 0x80) << 8) | (exponent << 7) | (low_i & 0x7F)
        return T.reinterpret(T.Cast(T.uint16, bits), T.bfloat16)

    @tilelang.jit
    def kernel():
        M = T.dynamic("M")
        E = T.dynamic("E")

        @T.prim_func
        def main(
            A: T.Tensor((M, K), T.bfloat16),
            W: T.Tensor((N, row_bytes), T.uint8),
            exception_rows: T.Tensor((E,), T.int32),
            exception_entries: T.Tensor((E,), T.int32),
            row_offsets: T.Tensor((N + 1,), T.int32),
            C: T.Tensor((M, N), tl_out_dtype),
        ):
            with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
            ) as (bx, by):
                A_shared = T.alloc_shared((block_M, STEP), T.bfloat16)
                low_shared = T.alloc_shared((block_N, STEP), T.uint8)
                code_shared = T.alloc_shared((block_N, STEP // 2), T.uint8)
                B_shared = T.alloc_shared((block_N, STEP), T.bfloat16)
                C_local = T.alloc_fragment((block_M, block_N), T.float32)
                low_a = T.alloc_local((VEC,), T.uint8)
                low_b = T.alloc_local((VEC,), T.uint8)
                codes = T.alloc_local((VEC,), T.uint8)
                out_a = T.alloc_local((VEC,), T.bfloat16)
                out_b = T.alloc_local((VEC,), T.bfloat16)
                tx = T.get_thread_binding()
                T.clear(C_local)
                last = num_stages - 1
                for s in T.Pipelined(
                    steps, order=[0, 1, 2, 3, 4], stage=[0, 0, 0, last, last]
                ):
                    T.copy(A[by * block_M, s * STEP], A_shared)
                    T.copy(W[bx * block_N, s * STEP], low_shared)
                    T.copy(W[bx * block_N, K + s * (STEP // 2)], code_shared)
                    for step in T.serial(T.ceildiv(items, threads)):
                        item = step * threads + tx
                        if item < items:
                            j = item // (step_groups * slices)
                            g = item // slices % step_groups
                            c = item % slices * VEC
                            for v in T.vectorized(VEC):
                                low_a[v] = low_shared[j, g * GROUP + c + v]
                            for v in T.vectorized(VEC):
                                low_b[v] = low_shared[j, g * GROUP + HALF + c + v]
                            for v in T.vectorized(VEC):
                                codes[v] = code_shared[j, g * HALF + c + v]
                            for v in T.unroll(VEC):
                                out_a[v] = _decode(low_a[v], codes[v] & 0xF)
                                out_b[v] = _decode(low_b[v], codes[v] >> 4)
                            for v in T.vectorized(VEC):
                                B_shared[j, g * GROUP + c + v] = out_a[v]
                            for v in T.vectorized(VEC):
                                B_shared[j, g * GROUP + HALF + c + v] = out_b[v]
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                first = bx * block_N
                start = row_offsets[first]
                end = row_offsets[T.min(first + block_N, N)]
                for index in T.serial(start, end):
                    row = exception_rows[index] - first
                    entry = exception_entries[index]
                    value = T.Cast(
                        T.float32,
                        T.reinterpret(T.Cast(T.uint16, entry & 0xFFFF), T.bfloat16),
                    )
                    column = entry >> 16
                    for i, j in T.Parallel(block_M, block_N):
                        token = T.min(by * block_M + i, M - 1)
                        C_local[i, j] = T.if_then_else(
                            j == row,
                            C_local[i, j] + value * T.Cast(T.float32, A[token, column]),
                            C_local[i, j],
                        )
                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    return kernel()


class PackedBF16VocabRunner:
    """Prepared multi-tile runner for an exact 12-bit packed BF16 LM head."""

    def __init__(self, packed: PackedBF16Weight, out_dtype: torch.dtype = torch.bfloat16) -> None:
        self.packed = packed
        self.out_dtype = out_dtype
        out_dtype_str = "float32" if out_dtype == torch.float32 else "bfloat16"
        N = packed.values.shape[0]
        K = packed.in_features
        self.kernels = {
            bm: build_packed_vocab_kernel(
                N, K, packed.exponent_base, block_M=bm, out_dtype=out_dtype_str, **_tile_config(bm, K)
            )
            for bm in ROW_TILES
        }

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        rows = x_c.shape[0]
        out = torch.empty((rows, self.packed.values.shape[0]), dtype=self.out_dtype, device=x_c.device)
        k = self.kernels[_row_tile(rows)]
        k(
            x_c,
            self.packed.values,
            self.packed.exception_rows,
            self.packed.exception_entries,
            self.packed.row_offsets,
            out,
        )
        return out

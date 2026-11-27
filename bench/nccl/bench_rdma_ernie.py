#!/usr/bin/env python3
"""NCCL / RoCEv2 collective microbenchmark for ERNIE 5.5 Flash Preview on DGX Spark (GB10).

Measures exact ERNIE 5.5 per-layer collective shapes across batch sizes
B in {1, 2, 4, 8, 16, 32, 64, 128, 512, 2048} in both eager and CUDA-graph
modes, plus an explicit probe of PyTorch symmetric memory support.

Shapes (from ernie55_config.json & vllm_ernie/models/ernie5_5/nvidia/{attention,moe}.py):
  - vha_hca_ar:     all_reduce  [B, 16, 512] fp32   (37 HCA layers under TP>1)
  - vha_dsa_ar:     all_reduce  [B, 16, 256] fp32   (6 DSA layers under TP>1)
  - attn_ffn_ar:    all_reduce  [B, 4096]    bf16   (43 attn wo_b + 1 dense FFN + 42 shared_experts)
  - moe_latent_ar:  all_reduce  [B, 2048]    bf16   (42 MoE routed latent combines under NoDP EP)
  - ep_a2a_bf16:    all_to_all  [B*10, 2048] bf16   (42 MoE EP dispatch/combine payload, bf16)
  - ep_a2a_fp8:     all_to_all  [B*10, 2048] uint8  (42 MoE EP dispatch/combine payload, fp8 wire bytes)
  - ag_hidden_bf16: all_gather  [B, 4096]    bf16   (DP/SP hidden gather; output [WS*B, 4096])
  - ag_latent_bf16: all_gather  [B, 2048]    bf16   (DP/SP latent gather; output [WS*B, 2048])

Launch example (2 nodes, node0 + node1):
  Rank 0 (node0):
    MASTER_ADDR=10.0.0.1 MASTER_PORT=29560 WORLD_SIZE=2 RANK=0 LOCAL_RANK=0 \\
      $VLLM_VENV/bin/python bench_rdma_ernie.py --tag dual_rail --cuda-graph --json-out out.json
  Rank 1 (node1):
    MASTER_ADDR=10.0.0.1 MASTER_PORT=29560 WORLD_SIZE=2 RANK=1 LOCAL_RANK=0 \\
      $VLLM_VENV/bin/python bench_rdma_ernie.py --tag dual_rail --cuda-graph
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable

import torch
import torch.distributed as dist

DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 512, 2048]

# ERNIE 5.5 architectural constants
HIDDEN_SIZE = 4096
MOE_LATENT_SIZE = 2048
MOE_TOPK = 10
VHA_RANK = 16
HCA_HEAD_DIM = 512
DSA_V_HEAD_DIM = 256
NUM_LAYERS = 43
NUM_HCA_LAYERS = 37
NUM_DSA_LAYERS = 6
NUM_DENSE_LAYERS = 1
NUM_MOE_LAYERS = 42


@dataclass
class OpResult:
    op: str
    collective: str
    batch: int
    shape: list[int]
    dtype: str
    tensor_bytes: int
    wire_bytes_per_rank: int
    mode: str  # "eager" or "cudagraph"
    mean_us: float
    p50_us: float
    p95_us: float
    min_us: float
    algbw_gbps: float
    busbw_gbps: float


def get_nccl_env_snapshot() -> dict[str, str]:
    keys = [
        "NCCL_NET",
        "NCCL_IB_DISABLE",
        "NCCL_IB_HCA",
        "NCCL_IB_GID_INDEX",
        "NCCL_IB_QPS_PER_CONNECTION",
        "NCCL_IB_SPLIT_DATA_ON_QPS",
        "NCCL_IB_MERGE_NICS",
        "NCCL_IB_TC",
        "NCCL_IB_SL",
        "NCCL_IB_PCI_RELAXED_ORDERING",
        "NCCL_CROSS_NIC",
        "NCCL_NET_GDR_LEVEL",
        "NCCL_DMABUF_ENABLE",
        "NCCL_PROTO",
        "NCCL_ALGO",
        "NCCL_MIN_NCHANNELS",
        "NCCL_MAX_NCHANNELS",
        "NCCL_BUFFSIZE",
        "NCCL_LL_THRESHOLD",
        "NCCL_LL128_BUFFSIZE",
        "NCCL_NTHREADS",
        "NCCL_P2P_DISABLE",
        "NCCL_SHM_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "NCCL_SYMM_MEM",
        "LD_PRELOAD",
    ]
    return {k: os.environ[k] for k in keys if k in os.environ}


def build_op_specs(batch: int, world_size: int, selected: set[str] | None) -> list[dict]:
    specs = [
        {
            "op": "vha_hca_ar",
            "collective": "all_reduce",
            "shape": [batch, VHA_RANK, HCA_HEAD_DIM],
            "dtype": torch.float32,
            "dtype_name": "fp32",
        },
        {
            "op": "vha_dsa_ar",
            "collective": "all_reduce",
            "shape": [batch, VHA_RANK, DSA_V_HEAD_DIM],
            "dtype": torch.float32,
            "dtype_name": "fp32",
        },
        {
            "op": "attn_ffn_ar",
            "collective": "all_reduce",
            "shape": [batch, HIDDEN_SIZE],
            "dtype": torch.bfloat16,
            "dtype_name": "bf16",
        },
        {
            "op": "moe_latent_ar",
            "collective": "all_reduce",
            "shape": [batch, MOE_LATENT_SIZE],
            "dtype": torch.bfloat16,
            "dtype_name": "bf16",
        },
        {
            "op": "ep_a2a_bf16",
            "collective": "all_to_all",
            "shape": [batch * MOE_TOPK, MOE_LATENT_SIZE],
            "dtype": torch.bfloat16,
            "dtype_name": "bf16",
        },
        {
            "op": "ep_a2a_fp8",
            "collective": "all_to_all",
            "shape": [batch * MOE_TOPK, MOE_LATENT_SIZE],
            "dtype": torch.uint8,  # wire-identical to float8_e4m3fn (1 byte/elem)
            "dtype_name": "fp8",
        },
        {
            "op": "ag_hidden_bf16",
            "collective": "all_gather",
            "shape": [batch, HIDDEN_SIZE],
            "dtype": torch.bfloat16,
            "dtype_name": "bf16",
        },
        {
            "op": "ag_latent_bf16",
            "collective": "all_gather",
            "shape": [batch, MOE_LATENT_SIZE],
            "dtype": torch.bfloat16,
            "dtype_name": "bf16",
        },
    ]
    if selected is not None:
        specs = [s for s in specs if s["op"] in selected]
    # Ensure all_to_all dim0 is divisible by world_size (10 is even, so always true for ws=2)
    for s in specs:
        if s["collective"] == "all_to_all" and s["shape"][0] % world_size != 0:
            s["shape"][0] = ((s["shape"][0] + world_size - 1) // world_size) * world_size
    return specs


def make_collective_fn(
    spec: dict, world_size: int, device: torch.device
) -> tuple[Callable[[], None], int, int]:
    """Allocate buffers and return (callable, logical_bytes, wire_bytes_per_rank)."""
    coll = spec["collective"]
    shape = spec["shape"]
    dtype = spec["dtype"]

    if coll == "all_reduce":
        buf = torch.ones(shape, dtype=dtype, device=device)
        logical_bytes = buf.numel() * buf.element_size()
        # Ring/Tree all-reduce bus factor: 2 * (N - 1) / N * S
        wire_bytes = int(logical_bytes * 2 * (world_size - 1) / world_size)

        def fn() -> None:
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)

        return fn, logical_bytes, wire_bytes

    if coll == "all_to_all":
        inp = torch.ones(shape, dtype=dtype, device=device)
        out = torch.empty_like(inp)
        logical_bytes = inp.numel() * inp.element_size()
        # Unidirectional peer transfer: (N - 1) / N * S
        wire_bytes = int(logical_bytes * (world_size - 1) / world_size)

        def fn() -> None:
            dist.all_to_all_single(out, inp)

        return fn, logical_bytes, wire_bytes

    if coll == "all_gather":
        inp = torch.ones(shape, dtype=dtype, device=device)
        out_shape = [shape[0] * world_size, *shape[1:]]
        out = torch.empty(out_shape, dtype=dtype, device=device)
        logical_bytes = out.numel() * out.element_size()
        wire_bytes = int(logical_bytes * (world_size - 1) / world_size)

        def fn() -> None:
            dist.all_gather_into_tensor(out, inp)

        return fn, logical_bytes, wire_bytes

    raise ValueError(f"Unknown collective: {coll}")


def bench_eager(
    fn: Callable[[], None],
    warmup: int,
    iters: int,
    repeats: int,
) -> list[float]:
    """Run `repeats` timed windows of `iters` consecutive calls; return per-call us."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()

    samples_us: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(repeats):
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples_us.append((start.elapsed_time(end) * 1000.0) / iters)

    return samples_us


def bench_cudagraph(
    fn: Callable[[], None],
    warmup: int,
    graph_steps: int,
    repeats: int,
) -> list[float] | None:
    """Capture `graph_steps` collectives into a CUDA graph and replay `repeats` times."""
    stream = torch.cuda.Stream()
    try:
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                fn()
        stream.synchronize()
        dist.barrier()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            for _ in range(graph_steps):
                fn()
        stream.synchronize()
        dist.barrier()

        # Warmup replay
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        dist.barrier()

        samples_us: list[float] = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for _ in range(repeats):
            start.record()
            g.replay()
            end.record()
            torch.cuda.synchronize()
            samples_us.append((start.elapsed_time(end) * 1000.0) / graph_steps)

        del g
        return samples_us
    except Exception as exc:
        if dist.get_rank() == 0:
            print(f"[WARN] CUDA graph capture failed: {exc}", file=sys.stderr)
        return None


def probe_symmetric_memory(world_size: int) -> dict:
    """Probe PyTorch _symmetric_memory multi-node support on this platform."""
    info: dict = {
        "torch_version": torch.__version__,
        "nccl_version": list(torch.cuda.nccl.version()),
        "supported": False,
        "detail": "",
    }
    try:
        import torch.distributed._symmetric_memory as symm_mem

        if hasattr(symm_mem, "is_nvshmem_available"):
            info["nvshmem_available"] = bool(symm_mem.is_nvshmem_available())
        if hasattr(symm_mem, "get_backend"):
            try:
                info["backend"] = str(symm_mem.get_backend(torch.device("cuda")))
            except Exception as e:
                info["backend_err"] = str(e)

        t = symm_mem.empty((4096,), dtype=torch.bfloat16, device="cuda")
        hdl = symm_mem.rendezvous(t, group=dist.group.WORLD)
        info["supported"] = True
        info["detail"] = f"rendezvous succeeded: {type(hdl).__name__}"
    except Exception as exc:
        info["supported"] = False
        info["detail"] = f"{type(exc).__name__}: {exc}"
    return info


def summarize(
    spec: dict,
    batch: int,
    logical_bytes: int,
    wire_bytes: int,
    mode: str,
    samples_us: list[float],
) -> OpResult:
    ordered = sorted(samples_us)
    mean_us = statistics.fmean(ordered)
    p50_us = statistics.median(ordered)
    p95_idx = min(len(ordered) - 1, int(0.95 * len(ordered)))
    p95_us = ordered[p95_idx]
    min_us = ordered[0]
    # GB/s = bytes / (us * 1e-6) / 1e9 = bytes / (us * 1e3)
    algbw = logical_bytes / (p50_us * 1e3) if p50_us > 0 else 0.0
    busbw = wire_bytes / (p50_us * 1e3) if p50_us > 0 else 0.0
    return OpResult(
        op=spec["op"],
        collective=spec["collective"],
        batch=batch,
        shape=spec["shape"],
        dtype=spec["dtype_name"],
        tensor_bytes=logical_bytes,
        wire_bytes_per_rank=wire_bytes,
        mode=mode,
        mean_us=round(mean_us, 3),
        p50_us=round(p50_us, 3),
        p95_us=round(p95_us, 3),
        min_us=round(min_us, 3),
        algbw_gbps=round(algbw, 3),
        busbw_gbps=round(busbw, 3),
    )


def compute_43layer_overhead(results: list[OpResult], mode: str = "eager") -> dict:
    """Aggregate exact 43-layer ERNIE 5.5 collective overhead (ms/step) by batch."""
    by_key = {(r.op, r.batch): r.p50_us for r in results if r.mode == mode}
    batches = sorted({r.batch for r in results if r.mode == mode})
    summary: dict[str, dict] = {}
    for b in batches:
        vha_hca = by_key.get(("vha_hca_ar", b))
        vha_dsa = by_key.get(("vha_dsa_ar", b))
        attn_ffn = by_key.get(("attn_ffn_ar", b))
        moe_lat = by_key.get(("moe_latent_ar", b))
        ep_bf16 = by_key.get(("ep_a2a_bf16", b))
        ep_fp8 = by_key.get(("ep_a2a_fp8", b))
        entry: dict[str, float] = {}
        if vha_hca is not None and attn_ffn is not None:
            vha_ms = (
                NUM_HCA_LAYERS * vha_hca
                + NUM_DSA_LAYERS * (vha_dsa if vha_dsa is not None else vha_hca)
            ) / 1000.0
            attn_o_ms = (NUM_LAYERS * attn_ffn) / 1000.0
            dense_ffn_ms = (NUM_DENSE_LAYERS * attn_ffn) / 1000.0
            entry["vha_43_ms"] = round(vha_ms, 3)
            entry["attn_oproj_43_ms"] = round(attn_o_ms, 3)
            entry["dense_ffn_1_ms"] = round(dense_ffn_ms, 3)

            # Mode 1: Actual current vLLM NoDP EP (42 routed latent AR + 42 shared_experts AR)
            if moe_lat is not None:
                moe_routed_ar_ms = (NUM_MOE_LAYERS * moe_lat) / 1000.0
                moe_shared_ar_ms = (NUM_MOE_LAYERS * attn_ffn) / 1000.0
                entry["moe_routed_ar_42_ms"] = round(moe_routed_ar_ms, 3)
                entry["moe_shared_ar_42_ms"] = round(moe_shared_ar_ms, 3)
                entry["total_129_nodp_single_moe_ar_ms"] = round(
                    vha_ms + attn_o_ms + dense_ffn_ms + moe_routed_ar_ms, 3
                )
                entry["total_171_actual_vllm_nodp_ep_ms"] = round(
                    vha_ms + attn_o_ms + dense_ffn_ms + moe_routed_ar_ms + moe_shared_ar_ms,
                    3,
                )

            # Mode 2: All-to-All EP (dispatch + combine = 2 A2A per MoE layer)
            if ep_bf16 is not None:
                ep_bf16_1x_ms = (NUM_MOE_LAYERS * ep_bf16) / 1000.0
                entry["ep_a2a_bf16_42x1_ms"] = round(ep_bf16_1x_ms, 3)
                entry["total_129_a2a_bf16_1x_ms"] = round(
                    vha_ms + attn_o_ms + dense_ffn_ms + ep_bf16_1x_ms, 3
                )
                entry["total_a2a_bf16_disp_comb_ms"] = round(
                    vha_ms + attn_o_ms + dense_ffn_ms + 2.0 * ep_bf16_1x_ms, 3
                )
            if ep_fp8 is not None:
                ep_fp8_1x_ms = (NUM_MOE_LAYERS * ep_fp8) / 1000.0
                entry["ep_a2a_fp8_42x1_ms"] = round(ep_fp8_1x_ms, 3)
                entry["total_129_a2a_fp8_1x_ms"] = round(
                    vha_ms + attn_o_ms + dense_ffn_ms + ep_fp8_1x_ms, 3
                )
            summary[str(b)] = entry
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="ERNIE 5.5 NCCL/RoCEv2 microbenchmark")
    parser.add_argument("--tag", default="default", help="Label for this config run")
    parser.add_argument(
        "--batches",
        default=",".join(str(b) for b in DEFAULT_BATCHES),
        help="Comma-separated batch sizes",
    )
    parser.add_argument(
        "--ops",
        default="all",
        help="Comma-separated op names or 'all' or 'sweep'",
    )
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iters", type=int, default=20, help="Calls per timed window (small B)")
    parser.add_argument("--repeats", type=int, default=15, help="Timed windows per op/batch")
    parser.add_argument("--cuda-graph", action="store_true", help="Also benchmark CUDA graph replay")
    parser.add_argument("--graph-steps", type=int, default=16)
    parser.add_argument("--test-symm-mem", action="store_true")
    parser.add_argument("--json-out", default="", help="Write JSON results (rank 0)")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    t0 = time.perf_counter()
    dist.init_process_group(backend="nccl")
    init_s = time.perf_counter() - t0

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    batches = [int(x.strip()) for x in args.batches.split(",") if x.strip()]
    if args.ops == "all":
        selected_ops = None
    elif args.ops == "sweep":
        selected_ops = {"vha_hca_ar", "attn_ffn_ar", "moe_latent_ar", "ep_a2a_bf16", "ep_a2a_fp8"}
    else:
        selected_ops = {x.strip() for x in args.ops.split(",") if x.strip()}

    symm_info = None
    if args.test_symm_mem:
        symm_info = probe_symmetric_memory(world_size)
        dist.barrier()

    # Warm up GPU clocks and RoCEv2 QP/cache paths so B=1 does not measure cold P-states.
    warm_a = torch.randn((1024, 1024), dtype=torch.bfloat16, device=device)
    warm_c = torch.ones((262144,), dtype=torch.bfloat16, device=device)
    for _ in range(15):
        _ = warm_a @ warm_a
        dist.all_reduce(warm_c)
    torch.cuda.synchronize()
    del warm_a, warm_c
    dist.barrier()

    results: list[OpResult] = []
    for batch in batches:
        # Scale iteration counts so large payloads stay fast and small payloads have low jitter
        if batch >= 2048:
            iters, repeats, gsteps = max(4, args.iters // 5), max(7, args.repeats // 2), min(args.graph_steps, 4)
        elif batch >= 512:
            iters, repeats, gsteps = max(8, args.iters // 2), max(9, args.repeats - 4), min(args.graph_steps, 8)
        else:
            iters, repeats, gsteps = args.iters, args.repeats, args.graph_steps

        for spec in build_op_specs(batch, world_size, selected_ops):
            fn, logical_bytes, wire_bytes = make_collective_fn(spec, world_size, device)
            samples = bench_eager(fn, args.warmup, iters, repeats)
            res = summarize(spec, batch, logical_bytes, wire_bytes, "eager", samples)
            results.append(res)

            if args.cuda_graph:
                g_samples = bench_cudagraph(fn, max(5, args.warmup // 2), gsteps, repeats)
                if g_samples is not None:
                    results.append(
                        summarize(spec, batch, logical_bytes, wire_bytes, "cudagraph", g_samples)
                    )

            torch.cuda.empty_cache()

    dist.barrier()

    if rank == 0:
        payload = {
            "tag": args.tag,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "world_size": world_size,
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nccl_version": list(torch.cuda.nccl.version()),
            "init_seconds": round(init_s, 3),
            "env": get_nccl_env_snapshot(),
            "symmetric_memory": symm_info,
            "results": [asdict(r) for r in results],
            "layer43_overhead_eager_ms": compute_43layer_overhead(results, "eager"),
            "layer43_overhead_cudagraph_ms": (
                compute_43layer_overhead(results, "cudagraph") if args.cuda_graph else {}
            ),
        }
        if args.json_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

        print(
            f"\n=== [{args.tag}] world_size={world_size} GPU={payload['gpu_name']} "
            f"NCCL={'.'.join(map(str, payload['nccl_version']))} ==="
        )
        if symm_info is not None:
            print(f"SymmetricMemory: supported={symm_info['supported']} ({symm_info['detail']})")
        print(
            f"{'op':16s} {'mode':9s} {'B':>5s} {'bytes':>10s} "
            f"{'p50(us)':>9s} {'p95(us)':>9s} {'min(us)':>9s} {'algBW':>8s} {'busBW':>8s}"
        )
        for r in results:
            print(
                f"{r.op:16s} {r.mode:9s} {r.batch:5d} {r.tensor_bytes:10d} "
                f"{r.p50_us:9.2f} {r.p95_us:9.2f} {r.min_us:9.2f} "
                f"{r.algbw_gbps:8.2f} {r.busbw_gbps:8.2f}"
            )
        if payload["layer43_overhead_eager_ms"]:
            print("\n43-Layer ERNIE 5.5 Overhead (ms/step, eager p50):")
            for b_str, ov in payload["layer43_overhead_eager_ms"].items():
                print(f"  B={b_str:>4s}: {ov}")
        if payload["layer43_overhead_cudagraph_ms"]:
            print("\n43-Layer ERNIE 5.5 Overhead (ms/step, cudagraph p50):")
            for b_str, ov in payload["layer43_overhead_cudagraph_ms"].items():
                print(f"  B={b_str:>4s}: {ov}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

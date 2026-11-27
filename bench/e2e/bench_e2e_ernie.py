#!/usr/bin/env python3
"""End-to-end throughput & exact-parity benchmark for ERNIE 5.5 Flash Preview on DGX Spark.

Measures:
  1. Single-stream decode (`B=1`, `off` + `high` reasoning, exact greedy parity check).
  2. Batched decode sweep (`B=4, 8, 16, 32, 64` concurrent requests, aggregate tok/s & per-req tok/s).
  3. Prefill throughput (`seq=2048` and `seq=8192` prompt tok/s + TTFT).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


PARITY_PROMPTS = [
    ("math_high", True, "high", "A train travels 120 km at 60 km/h and returns 120 km at 40 km/h. What is its average speed?"),
    ("algo_off", False, "off", "Explain in two sentences why binary search runs in O(log n) time."),
    ("zh_high", True, "high", "一辆车去程 120 公里时速 60 公里，回程 120 公里时速 40 公里，全程平均时速是多少？"),
]

BATCH_PROMPTS = [
    "Write a concise, detailed explanation of how TCP congestion control (CUBIC and BBR) works, including slow start, congestion avoidance, and fast recovery.",
    "Implement a lock-free ring buffer in C++20 with acquire-release memory ordering and explain why each memory fence is necessary.",
    "Compare FlashAttention-2, Multi-Head Latent Attention (MLA), and Dynamic Sparse Attention (DSA) in terms of KV-cache footprint and arithmetic intensity.",
    "Explain the mathematical derivation of Sinkhorn-Knopp matrix scaling and why doubly stochastic matrices preserve residual stream stability in deep transformers.",
    "Write a complete Python asyncio HTTP/2 connection pool with bounded concurrency, exponential backoff, and clean graceful shutdown.",
    "Describe the Linux RDMA RoCEv2 packet encapsulation over UDP/IP, PFC/ECN congestion control, and why MTU 9000 improves small-to-medium collective throughput.",
    "Explain how FP8 E4M3 block-128 quantization and MXFP4 UE8M0 microscaling represent dynamic range and affect tensor core throughput.",
    "Summarize the architectural differences between tensor parallelism, expert parallelism, data parallelism, and pipeline parallelism for 512-expert MoE models.",
]


async def one_req(client: httpx.AsyncClient, base: str, model: str, prompt: str,
                  max_tokens: int, think: bool = False, effort: str = "high") -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "seed": 11,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": think, **({"reasoning_effort": effort} if think else {})},
    }
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    usage = None
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    text_parts = []
    rsn_parts = []
    async with client.stream("POST", f"{base}/chat/completions", json=body, timeout=600) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("usage"):
                usage = ev["usage"]
            choices = ev.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                c = delta.get("content") or ""
                rc = delta.get("reasoning_content") or ""
                if (c or rc) and ttft is None:
                    ttft = time.perf_counter() - t0
                if c:
                    text_parts.append(c)
                if rc:
                    rsn_parts.append(rc)
                chunks += 1
    dt = time.perf_counter() - t0
    ptok = (usage or {}).get("prompt_tokens", 0)
    ctok = (usage or {}).get("completion_tokens", chunks)
    decode_t = max(1e-4, dt - (ttft or 0.0))
    return {
        "ptok": ptok,
        "ctok": ctok,
        "dt": dt,
        "ttft": ttft or dt,
        "decode_tps": ctok / decode_t,
        "e2e_tps": ctok / max(1e-4, dt),
        "ans": "".join(text_parts),
        "rsn": "".join(rsn_parts),
    }


async def run_batch(client: httpx.AsyncClient, base: str, model: str, concurrency: int,
                    max_tokens: int, think: bool = False) -> dict:
    # Unique prefix per request to defeat prefix caching and measure honest throughput
    prompts = [
        f"[Bench run #{i} / tag={concurrency}-{i*7919}] " + BATCH_PROMPTS[i % len(BATCH_PROMPTS)]
        for i in range(concurrency)
    ]
    t0 = time.perf_counter()
    res = await asyncio.gather(*(
        one_req(client, base, model, p, max_tokens=max_tokens, think=think)
        for p in prompts
    ))
    wall = time.perf_counter() - t0
    tot_ctok = sum(r["ctok"] for r in res)
    tot_ptok = sum(r["ptok"] for r in res)
    avg_ttft = sum(r["ttft"] for r in res) / len(res)
    mean_req_tps = sum(r["decode_tps"] for r in res) / len(res)
    return {
        "concurrency": concurrency,
        "wall_s": round(wall, 3),
        "total_ctok": tot_ctok,
        "total_ptok": tot_ptok,
        "agg_decode_tps": round(tot_ctok / wall, 2),
        "mean_req_decode_tps": round(mean_req_tps, 2),
        "avg_ttft_s": round(avg_ttft, 3),
    }


async def run_prefill(client: httpx.AsyncClient, base: str, model: str, target_tokens: int, tag: str) -> dict:
    # Each " word" with normal English words is ~1.05 tokens; use unique seeded prose to defeat prefix cache
    vocab = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
             "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
             "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray"]
    seed = sum(ord(c) for c in tag)
    n_words = max(64, int(target_tokens * 0.90))
    words = [vocab[(i * 7 + seed + (i // 24) * 3) % len(vocab)] for i in range(n_words)]
    prompt = f"Run {tag}. Summarize the recurring pattern in one sentence:\n" + " ".join(words)
    r = await one_req(client, base, model, prompt, max_tokens=16, think=False)
    return {
        "target_tok": target_tokens,
        "actual_ptok": r["ptok"],
        "ttft_s": round(r["ttft"], 4),
        "prefill_tps": round(r["ptok"] / max(1e-4, r["ttft"]), 1),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="ernie55")
    ap.add_argument("--label", required=True)
    ap.add_argument("--batches", default="1,4,8,16,32,64")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--skip-prefill", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    out = {"label": args.label, "base": args.base, "parity": [], "decode_batches": [], "prefill": []}
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=256, max_keepalive_connections=128)) as client:
        # Warmup
        await one_req(client, args.base, args.model, "Say OK.", max_tokens=8, think=False)

        print(f"=== [{args.label}] 1. Parity & Single-Stream (off + high) ===")
        for name, think, eff, p in PARITY_PROMPTS:
            r = await one_req(client, args.base, args.model, p, max_tokens=512, think=think, effort=eff)
            rec = {
                "name": name, "think": think, "ctok": r["ctok"], "dt": round(r["dt"], 3),
                "ttft": round(r["ttft"], 3), "decode_tps": round(r["decode_tps"], 2),
                "ans_head": r["ans"][:160],
            }
            out["parity"].append(rec)
            print(f"  {name:10s}: ctok={r['ctok']:4d} ttft={r['ttft']:.3f}s decode={r['decode_tps']:6.2f} tok/s | {r['ans'][:90]!r}")

        print(f"\n=== [{args.label}] 2. Batched Decode Sweep (max_tokens={args.max_tokens}) ===")
        for b in [int(x) for x in args.batches.split(",") if x.strip()]:
            r = await run_batch(client, args.base, args.model, b, max_tokens=args.max_tokens, think=False)
            out["decode_batches"].append(r)
            print(f"  B={b:3d}: agg={r['agg_decode_tps']:7.2f} tok/s  per_req={r['mean_req_decode_tps']:6.2f} tok/s  ttft={r['avg_ttft_s']:.3f}s  wall={r['wall_s']:.2f}s  ctok={r['total_ctok']}")

        if not args.skip_prefill:
            print(f"\n=== [{args.label}] 3. Prefill Sweep (fresh uncached prompts) ===")
            for seq in (2048, 8192):
                r = await run_prefill(client, args.base, args.model, seq, tag=f"{int(time.time())}_{seq}")
                out["prefill"].append(r)
                print(f"  seq~{seq:4d} (actual={r['actual_ptok']:5d}): ttft={r['ttft_s']:.3f}s  prefill={r['prefill_tps']:7.1f} tok/s")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())

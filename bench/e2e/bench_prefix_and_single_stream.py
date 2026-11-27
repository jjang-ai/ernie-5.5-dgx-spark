#!/usr/bin/env python3
"""Deep Single-Stream, VRAM Prefix Caching, Long-Context & Concurrency Stress Suite."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

SINGLE_STREAM_CASES = [
    ("off_prose", False, "off", 256, "Write a clear, detailed technical overview of how Linux io_uring eliminates syscall overhead compared to epoll."),
    ("off_code", False, "off", 384, "Write a complete, clean Python implementation of an LRU cache with TTL expiration using collections.OrderedDict and include brief usage examples."),
    ("high_math", True, "high", 512, "A train travels 120 km at 60 km/h and returns 120 km at 40 km/h. What is its average speed?"),
    ("high_zh", True, "high", 512, "一辆车去程 120 公里时速 60 公里，回程 120 公里时速 40 公里，全程平均时速是多少？"),
    ("high_code", True, "high", 1024, "Write a minimal, bug-free C++20 SPSC lock-free ring buffer with power-of-two capacity and explain the acquire/release memory ordering."),
]


async def stream_req(client: httpx.AsyncClient, base: str, model: str, messages: list[dict],
                     max_tokens: int, think: bool = False, effort: str = "high") -> dict:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "seed": 11,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": think, **({"reasoning_effort": effort} if think else {})},
    }
    t0 = time.perf_counter()
    ttft = None
    first_ans_t = None
    usage = None
    ans_parts, rsn_parts = [], []
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
            ch = (ev.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            c = delta.get("content") or ""
            rc = delta.get("reasoning_content") or ""
            now = time.perf_counter()
            if (c or rc) and ttft is None:
                ttft = now - t0
            if c and first_ans_t is None:
                first_ans_t = now - t0
            if c:
                ans_parts.append(c)
            if rc:
                rsn_parts.append(rc)
    dt = time.perf_counter() - t0
    ptok = (usage or {}).get("prompt_tokens", 0)
    ctok = (usage or {}).get("completion_tokens", 0)
    rtok = ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    gen_dt = max(1e-4, dt - (ttft or 0.0))
    return {
        "ptok": ptok,
        "ctok": ctok,
        "rtok": rtok,
        "ttft_s": round(ttft or dt, 4),
        "wall_s": round(dt, 4),
        "decode_tps": round(ctok / gen_dt, 2),
        "e2e_tps": round(ctok / max(1e-4, dt), 2),
        "ans": "".join(ans_parts),
        "rsn": "".join(rsn_parts),
    }


def make_doc(target_tokens: int, seed_tag: str) -> str:
    vocab = ["kernel", "memory", "bandwidth", "latency", "tensor", "router", "expert", "cache",
             "spark", "network", "fabric", "stream", "vector", "matrix", "block", "thread",
             "pipeline", "schedule", "dispatch", "context", "window", "compress", "index", "scale"]
    s = sum(ord(c) for c in seed_tag)
    n_words = max(200, int(target_tokens * 0.88))
    body = " ".join(vocab[(i * 11 + s + (i // 31) * 5) % len(vocab)] for i in range(n_words))
    return f"Document ID {seed_tag}. Secret verification code: SPARK-{seed_tag[-4:]}-OK.\n\n{body}"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="ernie55")
    ap.add_argument("--label", required=True)
    ap.add_argument("--prefix-lens", default="8192,16384")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    report = {"label": args.label, "single_stream": [], "prefix_cache": []}
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=64)) as client:
        # Warmup
        await stream_req(client, args.base, args.model, [{"role": "user", "content": "Say OK."}], 8)

        print(f"=== [{args.label}] 1. Single-Stream (B=1) True Streaming Decode ===")
        for name, think, eff, mtok, prompt in SINGLE_STREAM_CASES:
            r = await stream_req(client, args.base, args.model, [{"role": "user", "content": prompt}], mtok, think, eff)
            row = {k: v for k, v in r.items() if k not in ("ans", "rsn")}
            row["name"] = name
            row["ans_head"] = r["ans"][:120]
            report["single_stream"].append(row)
            print(f"  {name:10s}: ctok={r['ctok']:4d} (rtok={r['rtok']:3d}) ttft={r['ttft_s']:.3f}s decode={r['decode_tps']:6.2f} tok/s e2e={r['e2e_tps']:6.2f} tok/s | {r['ans'][:65]!r}")

        print(f"\n=== [{args.label}] 2. True VRAM Prefix Caching & Long-Context Stress ===")
        for plen in [int(x) for x in args.prefix_lens.split(",") if x.strip()]:
            tag = f"{int(time.time())}_{plen}"
            doc = make_doc(plen, tag)
            code = f"SPARK-{tag[-4:]}-OK"
            # Turn 1: COLD prefill of shared system document
            r1 = await stream_req(
                client, args.base, args.model,
                [{"role": "system", "content": doc}, {"role": "user", "content": "What is the exact secret verification code in the document? Reply with the code only."}],
                32, think=False,
            )
            # Turn 2: WARM prefix hit (identical system prompt, new question)
            r2 = await stream_req(
                client, args.base, args.model,
                [{"role": "system", "content": doc}, {"role": "user", "content": "Repeat the exact secret verification code and document ID in one short line."}],
                48, think=False,
            )
            # Turn 3: WARM prefix hit #2 + high-reasoning short answer
            r3 = await stream_req(
                client, args.base, args.model,
                [{"role": "system", "content": doc}, {"role": "user", "content": "Confirm the secret verification code from the document."}],
                128, think=True, effort="high",
            )
            speedup_ttft = round(r1["ttft_s"] / max(1e-4, r2["ttft_s"]), 2)
            rec = {
                "target_prefix": plen,
                "actual_ptok": r1["ptok"],
                "cold_ttft_s": r1["ttft_s"],
                "cold_prefill_tps": round(r1["ptok"] / max(1e-4, r1["ttft_s"]), 1),
                "warm1_ttft_s": r2["ttft_s"],
                "warm2_high_ttft_s": r3["ttft_s"],
                "prefix_ttft_speedup": speedup_ttft,
                "warm1_decode_tps": r2["decode_tps"],
                "warm2_high_decode_tps": r3["decode_tps"],
                "code_recalled_cold": code in r1["ans"],
                "code_recalled_warm1": code in r2["ans"],
                "code_recalled_warm2": code in r3["ans"],
            }
            report["prefix_cache"].append(rec)
            print(
                f"  prefix~{plen:5d} (actual={r1['ptok']:5d}): "
                f"COLD ttft={r1['ttft_s']:.3f}s ({rec['cold_prefill_tps']:.0f} tok/s) -> "
                f"WARM ttft={r2['ttft_s']:.3f}s ({speedup_ttft:.1f}x faster!) | "
                f"warm_decode={r2['decode_tps']:.1f} / high={r3['decode_tps']:.1f} tok/s | "
                f"recall=({rec['code_recalled_cold']},{rec['code_recalled_warm1']},{rec['code_recalled_warm2']})"
            )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())

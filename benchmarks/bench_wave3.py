#!/usr/bin/env python3
"""Reproducible end-to-end single-stream, cold-prefill/prefix-cache, and concurrency benchmark."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import time
import urllib.request


def call(url: str, model: str, prompt: str, max_tokens: int = 256, effort: str = "off") -> dict:
    ck: dict = {"enable_thinking": effort != "off"}
    if effort != "off":
        ck["reasoning_effort"] = effort
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": ck,
    }
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    usage: dict = {}
    txt: list[str] = []
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            obj = json.loads(d)
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices") or []
            if ch:
                dl = ch[0].get("delta") or {}
                c = dl.get("content") or ""
                rc = dl.get("reasoning_content") or dl.get("reasoning") or ""
                if (c or rc) and ttft is None:
                    ttft = time.perf_counter() - t0
                if c:
                    txt.append(c)
    dt = time.perf_counter() - t0
    pt = int(usage.get("prompt_tokens", 0))
    ct = int(usage.get("completion_tokens", 0))
    ttft_val = ttft if ttft is not None else dt
    dec = ct / max(dt - ttft_val, 1e-4)
    return {
        "pt": pt,
        "ct": ct,
        "dt": dt,
        "ttft": ttft_val,
        "dec": dec,
        "e2e": ct / max(dt, 1e-4),
        "text": "".join(txt),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="ernie55")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-b", type=int, default=32)
    ap.add_argument("--prefill-max-k", type=int, default=64)
    args = ap.parse_args()

    out: dict = {"single": {}, "prefill": {}, "conc_prose": {}, "conc_code": {}}

    single_prompts = [
        ("off_code", "Write a complete Python implementation of a lock-free ring buffer with docstrings, type hints, and comprehensive unit tests.", 384, "off"),
        ("off_prose", "Explain the architectural history and key tradeoffs of distributed consensus protocols from Paxos to Raft to PBFT in clear prose.", 256, "off"),
        ("algo_off", "What is the time complexity of binary search on a sorted array of n elements? Explain briefly and give clean Python code.", 256, "off"),
        ("high_math", "A car drives 120 km at 60 km/h and returns the same 120 km at 40 km/h. What is the average speed for the entire round trip? Give the exact final number.", 1024, "high"),
        ("high_zh", "一辆汽车以60公里/小时的速度行驶120公里，然后以40公里/小时的速度返回。求全程平均速度。", 1024, "high"),
    ]
    for name, p, mt, eff in single_prompts:
        r = call(args.url, args.model, p, mt, eff)
        rec = {k: v for k, v in r.items() if k != "text"}
        rec["preview"] = r["text"][:120].replace("\n", " ")
        out["single"][name] = rec
        print(
            f"[SINGLE] {name:10s}: ct={r['ct']:4d} ttft={r['ttft']:.3f}s "
            f"dec={r['dec']:6.2f} e2e={r['e2e']:6.2f} | {rec['preview'][:60]}",
            flush=True,
        )

    sizes = [("8k", 30000), ("32k", 120000)]
    if args.prefill_max_k >= 64:
        sizes.append(("64k", 240000))
    for tag, chars in sizes:
        rng = random.Random(hash(f"w3_{tag}_{args.url}") & 0xFFFFFFFF)
        words = [
            "matrix", "kernel", "tensor", "cache", "ring", "spark", "bandwidth",
            "latency", "prefill", "decode", "expert", "router", "stream", "memory",
            "block", "graph", "token", "layer", "vector", "scalar", "quant", "scale",
        ]
        codes = [f"CODE-{tag.upper()}-{i}-{rng.randint(1000,9999)}" for i in range(3)]
        parts = [
            " ".join(rng.choice(words) + str(rng.randint(10, 999)) for _ in range(chars // 36))
            for _ in range(4)
        ]
        doc = f"{parts[0]} SECRET_A={codes[0]} {parts[1]} SECRET_B={codes[1]} {parts[2]} SECRET_C={codes[2]} {parts[3]}"
        r_cold = call(args.url, args.model, doc + "\nList the three SECRET values exactly.", 64, "off")
        r_warm = call(args.url, args.model, doc + "\nRepeat the three SECRET values exactly, comma-separated.", 64, "off")
        hits_c = sum(1 for c in codes if c in r_cold["text"])
        hits_w = sum(1 for c in codes if c in r_warm["text"])
        cold_tps = r_cold["pt"] / max(r_cold["ttft"], 1e-4)
        speedup = r_cold["ttft"] / max(r_warm["ttft"], 1e-4)
        out["prefill"][tag] = {
            "pt": r_cold["pt"],
            "cold_ttft": r_cold["ttft"],
            "cold_tps": cold_tps,
            "warm_ttft": r_warm["ttft"],
            "speedup": speedup,
            "warm_dec": r_warm["dec"],
            "recall_cold": f"{hits_c}/3",
            "recall_warm": f"{hits_w}/3",
        }
        print(
            f"[PREFILL] {tag:4s} ({r_cold['pt']:6d} tok): cold={cold_tps:6.1f} tok/s "
            f"({r_cold['ttft']:.2f}s) -> warm={r_warm['ttft']:.3f}s ({speedup:.1f}x!) "
            f"warm_dec={r_warm['dec']:.1f} recall={hits_c}/3,{hits_w}/3",
            flush=True,
        )

    bs_list = [b for b in [1, 2, 4, 8, 16, 32] if b <= args.max_b]
    for kind, tmpl in [
        ("conc_prose", "Explain the architectural design, bottlenecks, and scaling tradeoffs of distributed system topic #{i} in clear technical prose."),
        ("conc_code", "Write a complete Python module implementing data structure and algorithm #{i} with type hints, docstrings, and unit tests."),
    ]:
        for B in bs_list:
            prompts = [tmpl.format(i=i) for i in range(B)]
            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=B) as ex:
                res = list(ex.map(lambda pr: call(args.url, args.model, pr, 256, "off"), prompts))
            wall = time.perf_counter() - t0
            tot = sum(r["ct"] for r in res)
            avg_dec = sum(r["dec"] for r in res) / B
            out[kind][str(B)] = {"agg": tot / wall, "per_req": avg_dec, "wall": wall, "tot": tot}
            print(
                f"[{kind.upper()}] B={B:2d}: agg={tot/wall:6.2f} tok/s | "
                f"per_req={avg_dec:5.2f} tok/s | wall={wall:.2f}s",
                flush=True,
            )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"SAVED {args.out}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Twitter-ready 2x vs 4x DGX Spark ERNIE 5.5 (DSpark) benchmark suite.

Measures:
  1. Single-stream (B=1) speeds across Easy Code, Hard Code (high reasoning),
     Prose (off & high), and Multilingual Reasoning at official T=1.0, top_p=0.95
     (plus greedy T=0 comparison for Easy Code & Prose).
  2. True Cold Prefill (8K, 32K) + 100K-step Growing Context Ladder up to 1,000,000 (1M)
     tokens: incremental +100K cold prefill tok/s & TTFT, warm VRAM prefix-cached TTFT,
     code & prose decode tok/s at every 100K depth, and dual-needle recall (start + end).
  3. Concurrency sweep (B=1..max_b) for both Prose and Easy Code at T=1.0, top_p=0.95.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


VOCAB_WORDS = [
    "kernel", "tensor", "matrix", "cache", "ring", "spark", "bandwidth", "latency",
    "prefill", "decode", "expert", "router", "stream", "memory", "block", "graph",
    "fabric", "packet", "queue", "thread", "warp", "tile", "stride", "buffer",
    "vector", "scalar", "norm", "scale", "shard", "worker", "cluster", "socket",
    "frame", "window", "index", "token", "layer", "head", "gate", "weight",
    "merge", "split", "flush", "sync", "probe", "trace", "metric", "epoch",
    "pipeline", "schedule", "dispatch", "reduce", "gather", "scatter", "offset", "align",
]


def _post_json(url: str, payload: dict[str, Any], timeout: float = 900.0) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def count_tokens(base_url: str, model: str, text: str) -> int | None:
    try:
        res = _post_json(f"{base_url}/tokenize", {"model": model, "prompt": text}, timeout=60.0)
        if "count" in res:
            return int(res["count"])
        if "tokens" in res:
            return len(res["tokens"])
    except Exception:
        return None
    return None


def make_chunk_text(chunk_idx: int, n_words: int, needle: str) -> str:
    rng = random.Random(90000 + chunk_idx * 1337)
    lines: list[str] = []
    # Build diverse log/spec lines (~20 words/line) so nothing triggers trivial n-gram loops
    n_lines = max(1, n_words // 20)
    mid = n_lines // 2
    for li in range(n_lines):
        if li == mid:
            lines.append(
                f"[BLOCK_{chunk_idx:02d}_ANCHOR] CRITICAL_RECEIPT_CODE_{chunk_idx}={needle} "
                f"(record_id={chunk_idx * 10000 + li}, verified=true)"
            )
        words = [
            f"{VOCAB_WORDS[rng.randrange(len(VOCAB_WORDS))]}{rng.randint(10, 999)}"
            for _ in range(18)
        ]
        lines.append(f"B{chunk_idx:02d}L{li:05d}: " + " ".join(words))
    return "\n".join(lines) + "\n"


def build_exact_chunk(
    base_url: str,
    model: str,
    chunk_idx: int,
    target_tokens: int,
    needle: str,
    toks_per_word_hint: float,
) -> tuple[str, int, float]:
    n_words = max(200, int(target_tokens / toks_per_word_hint))
    txt = make_chunk_text(chunk_idx, n_words, needle)
    cnt = count_tokens(base_url, model, txt)
    if cnt is None or cnt <= 0:
        return txt, target_tokens, toks_per_word_hint
    new_ratio = cnt / max(1, n_words)
    # Refine once to land within ~1% of target_tokens
    if abs(cnt - target_tokens) > max(400, int(0.015 * target_tokens)):
        n_words2 = max(200, int(target_tokens / new_ratio))
        txt = make_chunk_text(chunk_idx, n_words2, needle)
        cnt2 = count_tokens(base_url, model, txt)
        if cnt2 and cnt2 > 0:
            cnt = cnt2
            new_ratio = cnt / max(1, n_words2)
    return txt, cnt, new_ratio


def detect_loop(text: str) -> bool:
    if len(text) < 400:
        return False
    tail = text[-600:]
    for span in (40, 80, 120):
        seg = tail[-span:]
        if tail.count(seg) >= 4:
            return True
    return False


def stream_chat(
    base_url: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.95,
    effort: str = "off",
    timeout: float = 900.0,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"enable_thinking": effort != "off"}
    if effort != "off":
        kwargs["reasoning_effort"] = effort
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": kwargs,
    }
    if temperature > 0:
        payload["top_p"] = top_p

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft: float | None = None
    pt = ct = cached = 0
    finish_reason = ""
    out_chunks: list[str] = []
    think_chunks: list[str] = []

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            s = raw.decode("utf-8", "replace").strip()
            if not s.startswith("data:"):
                continue
            d = s[5:].strip()
            if d == "[DONE]":
                break
            obj = json.loads(d)
            usage = obj.get("usage")
            if usage:
                pt = int(usage.get("prompt_tokens") or pt)
                ct = int(usage.get("completion_tokens") or ct)
                details = usage.get("prompt_tokens_details") or {}
                if isinstance(details, dict) and details.get("cached_tokens") is not None:
                    cached = int(details["cached_tokens"])
            choices = obj.get("choices") or []
            if choices:
                c0 = choices[0]
                if c0.get("finish_reason"):
                    finish_reason = str(c0["finish_reason"])
                delta = c0.get("delta") or {}
                rc = delta.get("reasoning") or delta.get("reasoning_content") or ""
                tc = delta.get("content") or ""
                if rc or tc:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    if rc:
                        think_chunks.append(rc)
                    if tc:
                        out_chunks.append(tc)

    dt = time.perf_counter() - t0
    ttft_val = ttft if ttft is not None else dt
    gen_dt = max(dt - ttft_val, 1e-6)
    dec_tps = max(ct - 1, 1) / gen_dt if ct > 1 else ct / max(dt, 1e-6)
    e2e_tps = ct / max(dt, 1e-6)
    full_text = "".join(out_chunks)
    think_text = "".join(think_chunks)
    return {
        "pt": pt,
        "cached": cached,
        "ct": ct,
        "dt": round(dt, 4),
        "ttft_s": round(ttft_val, 4),
        "dec_tps": round(dec_tps, 2),
        "e2e_tps": round(e2e_tps, 2),
        "finish_reason": finish_reason,
        "loop": detect_loop(think_text + "\n" + full_text),
        "text": full_text,
        "preview": full_text[:140].replace("\n", " "),
    }


def save_results(out_path: str, data: dict[str, Any]) -> None:
    if not out_path:
        return
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="ernie55")
    ap.add_argument("--label", default="spark")
    ap.add_argument("--max-k", type=int, default=1000, help="Max growing context in thousands of tokens (e.g. 1000 = 1M)")
    ap.add_argument("--step-k", type=int, default=100, help="Step size in thousands of tokens")
    ap.add_argument("--conc-batches", default="1,2,4,8,16,32", help="Comma-separated concurrency batch sizes")
    ap.add_argument("--skip-single", action="store_true", help="Skip Section 1 single-stream categories")
    ap.add_argument("--skip-conc", action="store_true", help="Skip Section 2 concurrency sweep")
    ap.add_argument("--skip-short", action="store_true", help="Skip Section 3 short 8K/32K baselines")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    batches = [int(x) for x in args.conc_batches.split(",") if x.strip()]
    results: dict[str, Any] = {
        "label": args.label,
        "model": args.model,
        "single_stream": {},
        "short_cold_prefill": {},
        "context_ladder": [],
        "concurrency_prose": {},
        "concurrency_code": {},
    }

    # Warmup 1 tiny request
    _ = stream_chat(base, args.model, "Say OK.", max_tokens=8, temperature=0.0, effort="off")

    if not args.skip_single:
        print(f"=== 1. SINGLE-STREAM CATEGORIES ({args.label}, DSpark ON) ===", flush=True)
        single_cases = [
            ("easy_code_T1", "Write a clean Python implementation of an LRU cache with get/put in O(1) using OrderedDict and include 3 quick unit test assertions.", 384, 1.0, "off"),
            ("easy_code_T0", "Write a clean Python implementation of an LRU cache with get/put in O(1) using OrderedDict and include 3 quick unit test assertions.", 384, 0.0, "off"),
            ("hard_code_high_T1", "Implement a lock-free SPSC ring buffer in Python using ctypes/memoryview with power-of-two masking, full/empty checks, and explain the memory ordering guarantees.", 512, 1.0, "high"),
            ("prose_off_T1", "Write a clear, detailed technical explanation of how distributed consensus protocols like Raft and Paxos handle network partitions and leader election.", 384, 1.0, "off"),
            ("prose_off_T0", "Write a clear, detailed technical explanation of how distributed consensus protocols like Raft and Paxos handle network partitions and leader election.", 384, 0.0, "off"),
            ("prose_high_T1", "Compare NVLink, InfiniBand, and 200GbE RoCEv2 for multi-node MoE LLM inference, focusing on latency, tail jitter, and collective bottlenecks.", 512, 1.0, "high"),
            ("zh_math_high_T1", "一辆车以60公里/小时行驶120公里，再以40公里/小时返回120公里，求全程平均速度并给出步骤。", 384, 1.0, "high"),
        ]
        for name, prompt, mt, temp, eff in single_cases:
            r = stream_chat(base, args.model, prompt, max_tokens=mt, temperature=temp, top_p=0.95, effort=eff)
            rec = {k: v for k, v in r.items() if k != "text"}
            results["single_stream"][name] = rec
            print(
                f"  [{name:18s}] out={r['ct']:3d} ttft={r['ttft_s']:.3f}s dec={r['dec_tps']:6.2f} tok/s "
                f"e2e={r['e2e_tps']:6.2f} stop={r['finish_reason']} loop={r['loop']}",
                flush=True,
            )
        save_results(args.out, results)

    if not args.skip_conc:
        print(f"\n=== 2. CONCURRENCY SWEEP ({args.label}, T=1.0, top_p=0.95, DSpark ON, batches={batches}) ===", flush=True)
        for kind, base_prompt, out_key in [
            ("prose", "Explain why GPU memory bandwidth dominates autoregressive LLM decoding while compute FLOPs dominate long-context prefill, with concrete examples.", "concurrency_prose"),
            ("code", "Write a clean Python async token-bucket rate limiter class with burst support, refill rate, acquire(), and a short usage demo.", "concurrency_code"),
        ]:
            for b in batches:
                prompts = [f"[req {i} topic_salt={kind}_{b}_{i}] {base_prompt}" for i in range(b)]
                t0 = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=b) as ex:
                    futs = [
                        ex.submit(stream_chat, base, args.model, p, max_tokens=256, temperature=1.0, top_p=0.95, effort="off")
                        for p in prompts
                    ]
                    outs = [f.result() for f in futs]
                wall = time.perf_counter() - t0
                tot_ct = sum(x["ct"] for x in outs)
                agg = tot_ct / max(wall, 1e-6)
                avg_dec = sum(x["dec_tps"] for x in outs) / max(len(outs), 1)
                avg_ttft = sum(x["ttft_s"] for x in outs) / max(len(outs), 1)
                loops = sum(1 for x in outs if x["loop"])
                results[out_key][str(b)] = {
                    "agg_tps": round(agg, 2),
                    "per_req_dec_tps": round(avg_dec, 2),
                    "avg_ttft_s": round(avg_ttft, 3),
                    "wall_s": round(wall, 3),
                    "tot_tokens": tot_ct,
                    "loops": loops,
                }
                print(
                    f"  [CONC_{kind.upper():5s}] B={b:2d}: agg={agg:6.2f} tok/s | per_req={avg_dec:5.2f} tok/s "
                    f"| avg_ttft={avg_ttft:.3f}s | wall={wall:.2f}s | loops={loops}/{b}",
                    flush=True,
                )
                save_results(args.out, results)

    toks_per_word = 2.6
    if not args.skip_short:
        print(f"\n=== 3. SHORT COLD PREFILL BASELINES (8K & 32K, unique uncached prefix) ===", flush=True)
        for k_lbl, target_tok in [("8k", 8192), ("32k", 32768)]:
            needle = f"SHORT-{k_lbl.upper()}-{random.Random(target_tok).randint(10000, 99999)}"
            doc, est_tok, toks_per_word = build_exact_chunk(base, args.model, target_tok, target_tok, needle, toks_per_word)
            q_cold = doc + f"\n\nReply ONLY with the exact value of CRITICAL_RECEIPT_CODE_{target_tok}."
            rc = stream_chat(base, args.model, q_cold, max_tokens=32, temperature=0.0, effort="off")
            rw = stream_chat(base, args.model, q_cold + " Repeat once.", max_tokens=256, temperature=1.0, top_p=0.95, effort="off")
            cold_tps = rc["pt"] / max(rc["ttft_s"], 1e-6)
            hit_c = needle in rc["text"]
            hit_w = needle in rw["text"]
            results["short_cold_prefill"][k_lbl] = {
                "prompt_tokens": rc["pt"],
                "cold_ttft_s": rc["ttft_s"],
                "cold_prefill_tps": round(cold_tps, 1),
                "warm_ttft_s": rw["ttft_s"],
                "warm_cached_tokens": rw["cached"],
                "warm_speedup_x": round(rc["ttft_s"] / max(rw["ttft_s"], 1e-6), 2),
                "warm_dec_tps": rw["dec_tps"],
                "recall_cold": hit_c,
                "recall_warm": hit_w,
            }
            print(
                f"  [SHORT {k_lbl:3s}] pt={rc['pt']:6d} | cold={cold_tps:6.1f} tok/s ({rc['ttft_s']:.2f}s) "
                f"-> warm={rw['ttft_s']:.3f}s ({rc['ttft_s']/max(rw['ttft_s'],1e-6):.1f}x) "
                f"| warm_dec={rw['dec_tps']:.1f} tok/s | recall={hit_c}/{hit_w}",
                flush=True,
            )
            save_results(args.out, results)

    print(f"\n=== 4. GROWING CONTEXT LADDER: {args.step_k}K -> {args.max_k}K (step={args.step_k}K) ===", flush=True)
    prefix_parts: list[str] = [
        "SYSTEM ARCHIVE LOGS — Each block contains a CRITICAL_RECEIPT_CODE_<i> value. "
        "Follow instructions at the very end accurately.\n"
    ]
    needles: dict[int, str] = {}
    cum_cold_ttft = 0.0
    prev_pt = 0
    n_steps = args.max_k // args.step_k
    step_target = args.step_k * 1000  # Exact 100,000 tokens per step -> 1,000,000 at step 10!

    for step in range(1, n_steps + 1):
        target_k = step * args.step_k
        needle = f"SPARK{target_k}K-{random.Random(5000 + step).randint(10000, 99999)}"
        needles[step] = needle
        chunk_txt, chunk_tok, toks_per_word = build_exact_chunk(
            base, args.model, step, step_target, needle, toks_per_word
        )
        prefix_parts.append(chunk_txt)
        full_prefix = "".join(prefix_parts)

        # (A) Cold incremental +100K prefill step (step 1 is 100% pure cold 100K!)
        q_prefill = (
            full_prefix
            + f"\n\n[QUERY_A] Output ONLY the exact string for CRITICAL_RECEIPT_CODE_{step}."
        )
        try:
            r_cold = stream_chat(base, args.model, q_prefill, max_tokens=32, temperature=0.0, effort="off", timeout=1200.0)
        except Exception as exc:
            print(f"  [STEP {target_k:4d}K] STOPPED/ERROR during prefill: {exc}", flush=True)
            results["context_ladder"].append({"target_k": target_k, "error": str(exc)})
            save_results(args.out, results)
            break

        # vLLM streaming may report cached=0 even when prefix cache hits; compute true newly-prefilled tokens via delta
        new_toks = max(r_cold["pt"] - (r_cold["cached"] if r_cold["cached"] > 0 else prev_pt), 1)
        prev_pt = r_cold["pt"]
        inc_cold_tps = new_toks / max(r_cold["ttft_s"], 1e-6)
        cum_cold_ttft += r_cold["ttft_s"]
        cum_cold_tps = r_cold["pt"] / max(cum_cold_ttft, 1e-6)
        hit_end = needle in r_cold["text"]

        # (B) Warm VRAM-cached TTFT + Dual-Needle Recall (Block 1 + Current Block) + Easy Code Decode (256 tok)
        q_code = (
            full_prefix
            + f"\n\n[QUERY_B] First line: write `N1=<CRITICAL_RECEIPT_CODE_1> NLAST=<CRITICAL_RECEIPT_CODE_{step}>`. "
            + "Then write a concise, clean Python class implementing a min-heap priority queue with push/pop/peek."
        )
        r_code = stream_chat(base, args.model, q_code, max_tokens=256, temperature=1.0, top_p=0.95, effort="off", timeout=600.0)
        hit_start_b = needles[1] in r_code["text"]
        hit_end_b = needle in r_code["text"]

        # (C) Warm VRAM-cached Prose Decode (256 tok) at this exact context depth
        q_prose = (
            full_prefix
            + "\n\n[QUERY_C] Ignoring the archive logs above, write 2 clear paragraphs explaining how "
            + "speculative decoding speeds up memory-bandwidth-bound LLM inference."
        )
        r_prose = stream_chat(base, args.model, q_prose, max_tokens=256, temperature=1.0, top_p=0.95, effort="off", timeout=600.0)

        entry = {
            "step": step,
            "target_k": target_k,
            "prompt_tokens": r_cold["pt"],
            "new_prefill_tokens": new_toks,
            "inc_cold_ttft_s": r_cold["ttft_s"],
            "inc_cold_prefill_tps": round(inc_cold_tps, 1),
            "cum_cold_ttft_s": round(cum_cold_ttft, 2),
            "cum_cold_prefill_tps": round(cum_cold_tps, 1),
            "warm_code_ttft_s": r_code["ttft_s"],
            "warm_code_dec_tps": r_code["dec_tps"],
            "warm_code_e2e_tps": r_code["e2e_tps"],
            "warm_prose_ttft_s": r_prose["ttft_s"],
            "warm_prose_dec_tps": r_prose["dec_tps"],
            "warm_prose_e2e_tps": r_prose["e2e_tps"],
            "recall_cold_end": hit_end,
            "recall_warm_start": hit_start_b,
            "recall_warm_end": hit_end_b,
            "loops": int(r_code["loop"]) + int(r_prose["loop"]),
        }
        results["context_ladder"].append(entry)
        save_results(args.out, results)
        print(
            f"  [{target_k:4d}K | pt={r_cold['pt']:7,d}] "
            f"+100K cold={inc_cold_tps:6.1f} tok/s ({r_cold['ttft_s']:5.1f}s, cum={cum_cold_ttft:6.1f}s/{cum_cold_tps:6.1f}) | "
            f"warm TTFT={r_code['ttft_s']:.3f}s | "
            f"dec code={r_code['dec_tps']:5.1f} prose={r_prose['dec_tps']:5.1f} tok/s | "
            f"needle(start/end)={hit_start_b}/{hit_end_b}",
            flush=True,
        )

    save_results(args.out, results)
    print(f"\nSAVED {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

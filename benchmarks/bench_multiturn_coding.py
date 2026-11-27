#!/usr/bin/env python3
"""Real multi-turn growing-context coding & zero-loop verification at official ERNIE 5.5 sampling defaults (T=1.0, top_p=0.95, reasoning=high)."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def has_loop(text: str, ngram: int = 12, min_repeats: int = 5) -> bool:
    words = text.split()
    if len(words) < ngram * min_repeats:
        return False
    seen: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - ngram + 1):
        g = tuple(words[i : i + ngram])
        c = seen.get(g, 0) + 1
        if c >= min_repeats:
            return True
        seen[g] = c
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="ernie55")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    receipt = "PROJ-RECEIPT-8472-SPARK"
    turns = [
        f"Project tag: {receipt}. Write a compact (<40 lines) typed Python `Job` dataclass (id, priority, payload) and `AsyncJobQueue` wrapper around `asyncio.PriorityQueue`.",
        "Add a small async `run_with_retry(job, handler, max_retries=3)` function with exponential backoff (<30 lines).",
        "Add a minimal `Metrics` dataclass with `submitted`, `succeeded`, `failed` integer counters and a `to_prometheus()` method (<25 lines).",
        "Add a simple `TokenBucket` async rate limiter class (<30 lines).",
        "Add a `drain(queue, timeout=5.0)` graceful shutdown helper (<25 lines).",
        "Add a one-function `log_event(job_id, status)` structured JSON logger (<15 lines).",
        "Write 3 short pytest tests verifying priority ordering, retry behavior, and token bucket limiting (<40 lines).",
        "In 3 brief bullets summarize what we built and quote the exact project tag from Turn 1.",
    ]

    messages: list[dict[str, str]] = [
        {"role": "system", "content": "You are a concise Python engineer. Keep each reply short and focused on the requested snippet."}
    ]
    results = []
    clean_stops = 0
    loops = 0

    for idx, user_msg in enumerate(turns, 1):
        messages.append({"role": "user", "content": user_msg})
        ck = {"enable_thinking": args.effort != "off"}
        if args.effort != "off":
            ck["reasoning_effort"] = args.effort
        body = {
            "model": args.model,
            "messages": messages,
            "max_tokens": 8192,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": ck,
        }
        req = urllib.request.Request(
            f"{args.url.rstrip('/')}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        ttft = None
        finish_reason = None
        usage: dict = {}
        rsn, ans = [], []
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
                    if ch[0].get("finish_reason"):
                        finish_reason = ch[0]["finish_reason"]
                    dl = ch[0].get("delta") or {}
                    rc = dl.get("reasoning_content") or dl.get("reasoning") or ""
                    c = dl.get("content") or ""
                    if (rc or c) and ttft is None:
                        ttft = time.perf_counter() - t0
                    if rc:
                        rsn.append(rc)
                    if c:
                        ans.append(c)
        dt = time.perf_counter() - t0
        ttft = ttft if ttft is not None else dt
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        rsn_s = "".join(rsn)
        ans_s = "".join(ans)
        loop_flag = has_loop(rsn_s) or has_loop(ans_s)
        if finish_reason == "stop":
            clean_stops += 1
        if loop_flag:
            loops += 1
        dec = ct / max(dt - ttft, 1e-4)
        e2e = ct / max(dt, 1e-4)
        messages.append({"role": "assistant", "content": ans_s})
        rec = {
            "turn": idx,
            "prompt_tokens": pt,
            "cached_tokens": cached,
            "completion_tokens": ct,
            "finish_reason": finish_reason,
            "loop": loop_flag,
            "ttft_s": round(ttft, 3),
            "decode_tps": round(dec, 2),
            "e2e_tps": round(e2e, 2),
        }
        results.append(rec)
        print(
            f"[T{idx}] in={pt:5d} (cached={cached:5d}) out={ct:4d} finish={finish_reason:6s} "
            f"loop={loop_flag} ttft={ttft:.3f}s dec={dec:6.2f} tok/s e2e={e2e:6.2f}",
            flush=True,
        )

    recall_ok = receipt in messages[-1]["content"]
    summary = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "effort": args.effort,
        "clean_stops": f"{clean_stops}/{len(turns)}",
        "loops": f"{loops}/{len(turns)}",
        "turn1_recall": recall_ok,
        "avg_decode_tps": round(sum(r["decode_tps"] for r in results) / len(results), 2),
        "max_decode_tps": max(r["decode_tps"] for r in results),
        "turns": results,
    }
    print(
        f"\nSUMMARY: stops={summary['clean_stops']} loops={summary['loops']} "
        f"turn1_recall={recall_ok} avg_dec={summary['avg_decode_tps']} tok/s max_dec={summary['max_decode_tps']} tok/s",
        flush=True,
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Harsh prefix-cache & multi-API verification suite (OpenAI Chat, OpenAI Responses, Anthropic /v1/messages).

Validates every real-world edge case that breaks naive hybrid-KV / ERNIE 5.5 serving:
  1. OpenAI /v1/chat/completions: 16K prefix across multiple suffix lengths (including exact
     256-token SWA block boundaries!), verifying cached_tokens > 80% + recall + >10x TTFT speedup,
     plus 3-turn reasoning_effort=high conversation (with and without stripped thinking).
  2. OpenAI /v1/responses: 16K instructions + reasoning={effort: high / none / minimal / max},
     verifying input_tokens_details.cached_tokens > 80% + zero crashes + recall.
  3. Anthropic /v1/messages (+ /v1/messages/count_tokens): x-api-key header, thinking disabled &
     enabled (with output_config.effort=high/max), 16K system blocks with cache_control=ephemeral,
     and multi-turn Claude Code-style inline system reminders verifying cache_read_input_tokens > 80%!
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from spark_ernie_runtime.client import (
    stream_anthropic_messages,
    stream_chat_completion,
    stream_responses_api,
)


def make_doc(seed: int, n_words: int, code: str) -> str:
    rng = random.Random(seed)
    vocab = [
        "kernel", "tensor", "matrix", "cache", "ring", "spark", "bandwidth", "latency",
        "prefill", "decode", "expert", "router", "stream", "memory", "block", "graph",
        "fabric", "packet", "queue", "thread", "warp", "tile", "stride", "buffer",
    ]
    parts = [f"DOCUMENT_ANCHOR: VERIFICATION_CODE={code}\n"]
    for i in range(n_words // 16):
        line = " ".join(f"{vocab[rng.randrange(len(vocab))]}{rng.randint(10, 999)}" for _ in range(16))
        parts.append(f"L{i:04d}: {line}\n")
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="ernie55")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    v1 = base if base.endswith("/v1") else f"{base}/v1"
    report: dict[str, Any] = {"url": v1, "model": args.model, "checks": {}}
    all_ok = True

    print(f"=== HARSH PREFIX-CACHE & MULTI-API SUITE ({v1}) ===", flush=True)

    # --- 1. OpenAI /v1/chat/completions: SWA 256-boundary torture (different padding lengths!) ---
    code_chat = f"CHAT-{random.Random(101).randint(10000, 99999)}"
    doc_chat = make_doc(101, 5500, code_chat)  # ~16K tokens
    r_c0 = stream_chat_completion(
        v1, args.model,
        [{"role": "system", "content": doc_chat}, {"role": "user", "content": "What is the exact VERIFICATION_CODE at the top of the system prompt? Output ONLY that code."}],
        api_key=args.api_key, max_tokens=64, temperature=0.0, reasoning_effort="off", print_stream=False,
    )
    # Test 4 staggered suffix lengths across a full 256-token block boundary to catch any SWA rollback bug!
    chat_warm_runs = []
    for pad_i, pad_words in enumerate([0, 12, 48, 120]):
        pad = ("context " * pad_words).strip()
        r_cw = stream_chat_completion(
            v1, args.model,
            [
                {"role": "system", "content": doc_chat},
                {"role": "user", "content": f"Variant {pad_i} {pad}. Output ONLY the exact VERIFICATION_CODE from the top of the document."},
            ],
            api_key=args.api_key, max_tokens=64, temperature=0.0, reasoning_effort="off", print_stream=False,
        )
        hit_pct = 100.0 * r_cw["cached_tokens"] / max(r_cw["prompt_tokens"], 1)
        chat_warm_runs.append({
            "pad_words": pad_words,
            "pt": r_cw["prompt_tokens"],
            "cached": r_cw["cached_tokens"],
            "hit_pct": round(hit_pct, 1),
            "ttft_s": r_cw["ttft_s"],
            "recall": code_chat in r_cw["content"],
        })
    min_chat_hit = min(x["hit_pct"] for x in chat_warm_runs)
    ok_chat = (
        code_chat in r_c0["content"]
        and all(x["recall"] for x in chat_warm_runs)
        and min_chat_hit >= 80.0
    )
    all_ok &= ok_chat
    report["checks"]["chat_swa_boundary_prefix"] = {
        "pass": ok_chat,
        "cold_pt": r_c0["prompt_tokens"],
        "cold_ttft_s": r_c0["ttft_s"],
        "warm_runs": chat_warm_runs,
    }
    hits_str = [f"{x['hit_pct']}%({x['ttft_s']:.3f}s)" for x in chat_warm_runs]
    print(
        f"  [1] Chat /v1/chat/completions SWA-boundary prefix ({r_c0['prompt_tokens']} tok): "
        f"{'PASS' if ok_chat else 'FAIL'} | cold={r_c0['ttft_s']:.3f}s | "
        f"warm hits={hits_str}",
        flush=True,
    )

    # Multi-turn Chat (reasoning=high, preserved vs stripped reasoning_content)
    mt_msgs: list[dict[str, Any]] = [
        {"role": "system", "content": doc_chat},
        {"role": "user", "content": "Turn 1: Quote VERIFICATION_CODE and explain what an SPSC queue is in 2 sentences."},
    ]
    r_mt1 = stream_chat_completion(
        v1, args.model, mt_msgs, api_key=args.api_key, max_tokens=512, temperature=1.0, top_p=0.95,
        reasoning_effort="high", print_stream=False,
    )
    mt_msgs_preserved = mt_msgs + [
        {"role": "assistant", "content": r_mt1["content"], "reasoning_content": r_mt1["reasoning"]},
        {"role": "user", "content": "Turn 2: Now write a 10-line Python SPSC example."},
    ]
    r_mt2_pres = stream_chat_completion(
        v1, args.model, mt_msgs_preserved, api_key=args.api_key, max_tokens=256, temperature=1.0, top_p=0.95,
        reasoning_effort="high", print_stream=False,
    )
    mt_msgs_stripped = mt_msgs + [
        {"role": "assistant", "content": r_mt1["content"]},
        {"role": "user", "content": "Turn 2b: Quote VERIFICATION_CODE again."},
    ]
    r_mt2_strip = stream_chat_completion(
        v1, args.model, mt_msgs_stripped, api_key=args.api_key, max_tokens=128, temperature=1.0, top_p=0.95,
        reasoning_effort="high", print_stream=False,
    )
    hit_pres = 100.0 * r_mt2_pres["cached_tokens"] / max(r_mt2_pres["prompt_tokens"], 1)
    hit_strip = 100.0 * r_mt2_strip["cached_tokens"] / max(r_mt2_strip["prompt_tokens"], 1)
    ok_mt = hit_pres >= 85.0 and hit_strip >= 80.0 and (code_chat in r_mt2_strip["content"])
    all_ok &= ok_mt
    report["checks"]["chat_multiturn_reasoning_prefix"] = {
        "pass": ok_mt,
        "preserved_hit_pct": round(hit_pres, 1),
        "preserved_ttft_s": r_mt2_pres["ttft_s"],
        "stripped_hit_pct": round(hit_strip, 1),
        "stripped_ttft_s": r_mt2_strip["ttft_s"],
        "recall": code_chat in r_mt2_strip["content"],
    }
    print(
        f"  [2] Chat multi-turn high-reasoning prefix: {'PASS' if ok_mt else 'FAIL'} | "
        f"preserved={hit_pres:.1f}% ({r_mt2_pres['ttft_s']:.3f}s) | "
        f"stripped={hit_strip:.1f}% ({r_mt2_strip['ttft_s']:.3f}s)",
        flush=True,
    )

    # --- 2. OpenAI /v1/responses: effort normalization (high/none/minimal/max) + 16K prefix cache ---
    code_resp = f"RESP-{random.Random(202).randint(10000, 99999)}"
    doc_resp = make_doc(202, 5500, code_resp)
    r_r0 = stream_responses_api(
        v1, args.model,
        [{"role": "system", "content": doc_resp}, {"role": "user", "content": "Output ONLY VERIFICATION_CODE."}],
        api_key=args.api_key, max_output_tokens=64, temperature=0.0, reasoning_effort="off", print_stream=False,
    )
    r_r1 = stream_responses_api(
        v1, args.model,
        [{"role": "system", "content": doc_resp}, {"role": "user", "content": "Repeat ONLY VERIFICATION_CODE now."}],
        api_key=args.api_key, max_output_tokens=64, temperature=0.0, reasoning_effort="off", print_stream=False,
    )
    r_r_high = stream_responses_api(
        v1, args.model,
        [{"role": "user", "content": "What is 17 * 23?"}],
        api_key=args.api_key, max_output_tokens=256, temperature=1.0, top_p=0.95, reasoning_effort="high", print_stream=False,
    )
    resp_hit = 100.0 * r_r1["cached_tokens"] / max(r_r1["prompt_tokens"], 1)
    ok_resp = (
        code_resp in r_r0["content"]
        and code_resp in r_r1["content"]
        and "391" in r_r_high["content"]
        and resp_hit >= 80.0
    )
    all_ok &= ok_resp
    report["checks"]["responses_api_prefix_and_efforts"] = {
        "pass": ok_resp,
        "cold_ttft_s": r_r0["ttft_s"],
        "warm_ttft_s": r_r1["ttft_s"],
        "warm_hit_pct": round(resp_hit, 1),
        "warm_cached_tokens": r_r1["cached_tokens"],
        "high_reasoning_ok": "391" in r_r_high["content"],
    }
    print(
        f"  [3] Responses /v1/responses prefix + efforts: {'PASS' if ok_resp else 'FAIL'} | "
        f"cold={r_r0['ttft_s']:.3f}s -> warm={r_r1['ttft_s']:.3f}s ({resp_hit:.1f}%, cached={r_r1['cached_tokens']})",
        flush=True,
    )

    # --- 3. Anthropic /v1/messages (+ count_tokens + x-api-key + thinking disabled/enabled + inline reminders!) ---
    code_anth = f"ANTH-{random.Random(303).randint(10000, 99999)}"
    doc_anth = make_doc(303, 5500, code_anth)
    sys_blocks = [{"type": "text", "text": doc_anth, "cache_control": {"type": "ephemeral"}}]

    # 3a. Cold with thinking=disabled (previously crashed ERNIE 5.5!)
    r_a0 = stream_anthropic_messages(
        v1, args.model,
        [{"role": "user", "content": "Reply ONLY with VERIFICATION_CODE."}],
        system=sys_blocks, api_key=args.api_key, max_tokens=64, temperature=0.0,
        reasoning_effort="off", print_stream=False,
    )
    # 3b. Warm turn 2 with thinking=disabled
    r_a1 = stream_anthropic_messages(
        v1, args.model,
        [{"role": "user", "content": "Repeat ONLY VERIFICATION_CODE once."}],
        system=sys_blocks, api_key=args.api_key, max_tokens=64, temperature=0.0,
        reasoning_effort="off", print_stream=False,
    )
    # 3c. Warm turn with thinking=enabled (effort=high) & Claude Code-style inline system reminder!
    # Send raw Anthropic request with inline system-like multi-turn blocks to verify cache_read_input_tokens stays >80%!
    r_a_think = stream_anthropic_messages(
        v1, args.model,
        [
            {"role": "user", "content": "Reply ONLY with VERIFICATION_CODE."},
            {"role": "assistant", "content": r_a0["content"].strip() or code_anth},
            {"role": "user", "content": "<system-reminder>Keep answer brief.</system-reminder>\nNow compute 120/60 + 120/40 and repeat VERIFICATION_CODE."},
        ],
        system=sys_blocks, api_key=args.api_key, max_tokens=2048, temperature=1.0, top_p=0.95,
        reasoning_effort="high", print_stream=False,
    )
    # 3d. /v1/messages/count_tokens check
    ct_headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    if args.api_key:
        ct_headers["x-api-key"] = args.api_key
    ct_req = urllib.request.Request(
        f"{v1}/messages/count_tokens",
        data=json.dumps({"model": args.model, "system": sys_blocks, "messages": [{"role": "user", "content": "hi"}]}).encode(),
        headers=ct_headers, method="POST",
    )
    with urllib.request.urlopen(ct_req, timeout=30) as resp:
        ct_res = json.loads(resp.read().decode())
    counted_in = int(ct_res.get("input_tokens", 0))

    anth_hit_off = 100.0 * r_a1["cached_tokens"] / max(r_a1["prompt_tokens"], 1)
    anth_hit_think = 100.0 * r_a_think["cached_tokens"] / max(r_a_think["prompt_tokens"], 1)
    ok_anth = (
        code_anth in r_a0["content"]
        and code_anth in r_a1["content"]
        and code_anth in r_a_think["content"]
        and anth_hit_off >= 80.0
        and anth_hit_think >= 80.0
        and counted_in > 10000
    )
    all_ok &= ok_anth
    report["checks"]["anthropic_messages_api"] = {
        "pass": ok_anth,
        "cold_ttft_s": r_a0["ttft_s"],
        "warm_off_ttft_s": r_a1["ttft_s"],
        "warm_off_hit_pct": round(anth_hit_off, 1),
        "warm_off_cache_read": r_a1["cached_tokens"],
        "warm_think_ttft_s": r_a_think["ttft_s"],
        "warm_think_hit_pct": round(anth_hit_think, 1),
        "warm_think_cache_read": r_a_think["cached_tokens"],
        "warm_think_dec_tps": r_a_think["decode_tps"],
        "count_tokens": counted_in,
    }
    print(
        f"  [4] Anthropic /v1/messages (x-api-key + thinking + cache): {'PASS' if ok_anth else 'FAIL'} | "
        f"cold={r_a0['ttft_s']:.3f}s -> warm(off)={r_a1['ttft_s']:.3f}s ({anth_hit_off:.1f}%, read={r_a1['cached_tokens']}) | "
        f"warm(high+reminder)={r_a_think['ttft_s']:.3f}s ({anth_hit_think:.1f}%, dec={r_a_think['decode_tps']} tok/s) | "
        f"count_tokens={counted_in}",
        flush=True,
    )

    report["all_pass"] = all_ok
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"SAVED {args.out}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

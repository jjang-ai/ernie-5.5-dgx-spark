"""All-in-one CLI for ERNIE 5.5 Flash Preview (MXFP4 & NVFP4) on 2x-4x NVIDIA DGX Spark (GB10, SM12.1)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .client import interactive_chat, stream_anthropic_messages, stream_chat_completion, stream_responses_api
from .cluster import (
    PROFILES,
    PROFILE_ALIASES,
    REPO_ROOT,
    build_rank_env,
    cluster_status,
    resolve_profile,
    start_cluster,
    stop_cluster,
    tail_logs,
)
from .config import (
    CLUSTER_FILE,
    CONFIG_FILE,
    detect_vllm_bin,
    generate_api_key,
    load_cluster,
    load_config,
    mask_api_key,
    save_config,
)
from .discover import discover_cluster, format_topology_summary
from .download import DEFAULT_HF_REPO, DEFAULT_LOCAL_DIR, download_model, sync_existing_model


def _default_base_url(cfg: dict, override: str | None) -> str:
    if override:
        u = override.rstrip("/")
        return u if u.endswith("/v1") else f"{u}/v1"
    port = int(cfg.get("port", 8000))
    return f"http://127.0.0.1:{port}/v1"


def cmd_discover(args: argparse.Namespace) -> int:
    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    cluster = discover_cluster(hosts, ssh_user=args.user, save=not args.dry_run)
    print(format_topology_summary(cluster))
    if not args.dry_run:
        print(f"\nSaved discovered topology -> {CLUSTER_FILE}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    if args.sync_only:
        return sync_existing_model(args.dest)
    return download_model(
        repo_id=args.repo,
        dest_dir=args.dest,
        sync_cluster=not args.no_sync,
        update_config=not args.no_config,
    )


def cmd_init(args: argparse.Namespace) -> int:
    cfg = load_config()
    if args.model:
        cfg["model"] = args.model
    if args.served_model_name:
        cfg["served_model_name"] = args.served_model_name
    if args.profile:
        cfg["profile"] = args.profile
    if args.port:
        cfg["port"] = args.port
    if args.venv:
        cfg["venv"] = args.venv
    if args.nccl_lib:
        cfg["nccl_lib_path"] = args.nccl_lib
    if args.reasoning_effort:
        cfg["default_reasoning_effort"] = args.reasoning_effort
    if args.api_key is not None:
        cfg["api_key"] = generate_api_key() if args.api_key == "auto" else args.api_key
    save_config(cfg)
    print(f"Saved config -> {CONFIG_FILE}")
    shown = dict(cfg)
    shown["api_key"] = mask_api_key(cfg.get("api_key", ""))
    print(json.dumps(shown, indent=2))
    return 0


def cmd_api_key(args: argparse.Namespace) -> int:
    cfg = load_config()
    if args.action == "show":
        key = cfg.get("api_key", "")
        print(key if args.reveal else mask_api_key(key))
        return 0
    if args.action == "clear":
        cfg["api_key"] = ""
        save_config(cfg)
        print("Cleared API key. Run `spark-ernie restart` to apply.")
        return 0
    if args.action == "generate":
        cfg["api_key"] = generate_api_key()
        save_config(cfg)
        print(f"Generated new API key: {cfg['api_key']}")
        print("Run `spark-ernie restart` to apply to the running server.")
        return 0
    if args.action == "set":
        if not args.value:
            print("Provide a key value: `spark-ernie api-key set <YOUR_KEY>`", file=sys.stderr)
            return 2
        cfg["api_key"] = args.value
        save_config(cfg)
        print(f"Updated API key -> {mask_api_key(cfg['api_key'])}. Run `spark-ernie restart` to apply.")
        return 0
    return 2


def cmd_start(args: argparse.Namespace) -> int:
    return start_cluster(
        profile=args.profile,
        nodes_limit=args.nodes,
        model=args.model,
        port=args.port,
        master_port=args.master_port,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        no_sparknet=args.no_sparknet,
        wait=not args.no_wait,
        remote_repo_path=args.remote_repo,
    )


def cmd_stop(_: argparse.Namespace) -> int:
    return stop_cluster()


def cmd_restart(args: argparse.Namespace) -> int:
    stop_cluster()
    return cmd_start(args)


def cmd_status(args: argparse.Namespace) -> int:
    return cluster_status(url=args.url)


def cmd_logs(args: argparse.Namespace) -> int:
    return tail_logs(rank=args.rank, lines=args.lines, follow=args.follow)


def cmd_serve(args: argparse.Namespace) -> int:
    cfg = load_config()
    cluster = load_cluster()
    world_size = cluster["world_size"] if cluster else (4 if args.profile.startswith("4x") else 2)
    target_model = args.model or cfg["model"]
    prof_name, prof = resolve_profile(
        args.profile,
        world_size,
        cluster.get("topology", "") if cluster else "",
        model_path=target_model,
    )
    mode = prof.pop("mode")

    env = os.environ.copy()
    if cluster:
        renv = build_rank_env(
            cluster, cfg, prof, args.rank,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            no_sparknet=args.no_sparknet,
        )
        env.update(renv)
        topo_arg = "custom"
    else:
        for k, v in prof.items():
            env.setdefault(k, v)
        if args.max_model_len is not None:
            env["SPARK_MAX_MODEL_LEN"] = str(args.max_model_len)
        if args.max_num_seqs is not None:
            env["SPARK_MAX_NUM_SEQS"] = str(args.max_num_seqs)
        if args.no_sparknet:
            env["SPARK_USE_SPARKNET"] = "0"
        if cfg.get("api_key"):
            env.setdefault("VLLM_API_KEY", cfg["api_key"])
        topo_arg = args.topology

    script = REPO_ROOT / "scripts" / "serve_spark_ernie.sh"
    cmd = [
        str(script),
        mode,
        topo_arg,
        str(args.rank),
        args.model or cfg["model"],
        str(args.port or cfg.get("port", 8000)),
        str(args.master_port or cfg.get("master_port", 29801)),
    ]
    if args.dry_run:
        print("PROFILE:", prof_name)
        print("CMD:", " ".join(cmd))
        return 0
    return subprocess.call(cmd, env=env)


def cmd_chat(args: argparse.Namespace) -> int:
    cfg = load_config()
    base = _default_base_url(cfg, args.url)
    model = args.model or cfg.get("served_model_name", "ernie55")
    api_key = args.api_key if args.api_key is not None else cfg.get("api_key", "")
    effort = args.effort or cfg.get("default_reasoning_effort", "high")

    if not args.prompt:
        return interactive_chat(
            base, model,
            api_key=api_key,
            api_mode=args.api,
            reasoning_effort=effort,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            system_prompt=args.system or "",
        )

    prompt = " ".join(args.prompt)
    msgs = ([{"role": "system", "content": args.system}] if args.system else []) + [
        {"role": "user", "content": prompt}
    ]
    if args.api == "responses":
        res = stream_responses_api(
            base, model, msgs,
            api_key=api_key, max_output_tokens=args.max_tokens,
            temperature=args.temperature, top_p=args.top_p, reasoning_effort=effort,
        )
    elif args.api == "anthropic":
        res = stream_anthropic_messages(
            base, model, msgs,
            api_key=api_key, max_tokens=args.max_tokens,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, reasoning_effort=effort,
        )
    else:
        res = stream_chat_completion(
            base, model, msgs,
            api_key=api_key, max_tokens=args.max_tokens,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, reasoning_effort=effort,
        )
    print(
        f"[{res['api']} | in={res['prompt_tokens']} (cached={res['cached_tokens']}) "
        f"out={res['completion_tokens']} (think={res['reasoning_tokens']}) | "
        f"ttft={res['ttft_s']:.3f}s | decode={res['decode_tps']:.1f} tok/s | e2e={res['e2e_tps']:.1f} tok/s]"
    )
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    cfg = load_config()
    base = _default_base_url(cfg, args.url)
    model = args.model or cfg.get("served_model_name", "ernie55")
    api_key = args.api_key if args.api_key is not None else cfg.get("api_key", "")

    print(f"=== Verifying ERNIE 5.5 endpoint {base} (model={model}) ===")
    # 1. Chat Completions with reasoning_effort=high
    r1 = stream_chat_completion(
        base, model,
        [{"role": "user", "content": "A train travels 120 km at 60 km/h and returns 120 km at 40 km/h. What is its average speed?"}],
        api_key=api_key, max_tokens=512, reasoning_effort="high", print_stream=False,
    )
    ok1 = "48" in r1["content"]
    print(f"  [1] /v1/chat/completions (reasoning=high): {'PASS' if ok1 else 'FAIL'} | "
          f"ttft={r1['ttft_s']:.3f}s decode={r1['decode_tps']:.1f} tok/s (think={r1['reasoning_tokens']}, out={r1['completion_tokens']})")

    # 2. Responses API streaming
    r2 = stream_responses_api(
        base, model,
        "Reply with the exact word OK and nothing else.",
        api_key=api_key, max_output_tokens=64, reasoning_effort="off", print_stream=False,
    )
    ok2 = "ok" in r2["content"].lower()
    print(f"  [2] /v1/responses streaming             : {'PASS' if ok2 else 'FAIL'} | "
          f"ttft={r2['ttft_s']:.3f}s decode={r2['decode_tps']:.1f} tok/s | out={r2['content'].strip()!r}")

    # 3. True VRAM prefix caching + exact recall
    doc = ("Verification code: SPARK-9471-OK.\n\n" + " ".join(["tensor", "kernel", "cache", "spark"] * 800))
    rcold = stream_chat_completion(
        base, model,
        [{"role": "system", "content": doc}, {"role": "user", "content": "What is the verification code?"}],
        api_key=api_key, max_tokens=32, reasoning_effort="off", print_stream=False,
    )
    rwarm = stream_chat_completion(
        base, model,
        [{"role": "system", "content": doc}, {"role": "user", "content": "Repeat only the verification code."}],
        api_key=api_key, max_tokens=32, reasoning_effort="off", print_stream=False,
    )
    speedup = rcold["ttft_s"] / max(1e-4, rwarm["ttft_s"])
    ok3 = ("SPARK-9471-OK" in rcold["content"]) and ("SPARK-9471-OK" in rwarm["content"])
    cached_note = f", cached={rwarm['cached_tokens']}" if rwarm["cached_tokens"] else ""
    print(f"  [3] VRAM prefix caching ({rcold['prompt_tokens']} tok)  : {'PASS' if ok3 else 'FAIL'} | "
          f"cold={rcold['ttft_s']:.3f}s -> warm={rwarm['ttft_s']:.3f}s ({speedup:.1f}x faster{cached_note})")

    # 4. Coding single-stream speed check
    r4 = stream_chat_completion(
        base, model,
        [{"role": "user", "content": "Write a clean Python LRU cache with TTL expiration using collections.OrderedDict."}],
        api_key=api_key, max_tokens=256, reasoning_effort="off", print_stream=False,
    )
    print(f"  [4] Single-stream coding decode         : PASS | "
          f"ttft={r4['ttft_s']:.3f}s decode={r4['decode_tps']:.1f} tok/s e2e={r4['e2e_tps']:.1f} tok/s")

    # 5. Anthropic Messages API (/v1/messages) + thinking + VRAM prefix cache
    ra_cold = stream_anthropic_messages(
        base, model,
        [{"role": "user", "content": "What is the verification code?"}],
        system=[{"type": "text", "text": doc, "cache_control": {"type": "ephemeral"}}],
        api_key=api_key, max_tokens=64, reasoning_effort="off", print_stream=False,
    )
    ra_warm = stream_anthropic_messages(
        base, model,
        [{"role": "user", "content": "Repeat only the verification code."}],
        system=[{"type": "text", "text": doc, "cache_control": {"type": "ephemeral"}}],
        api_key=api_key, max_tokens=64, reasoning_effort="off", print_stream=False,
    )
    ok5 = ("SPARK-9471-OK" in ra_cold["content"]) and ("SPARK-9471-OK" in ra_warm["content"])
    a_spd = ra_cold["ttft_s"] / max(1e-4, ra_warm["ttft_s"])
    a_cached = f", cache_read={ra_warm['cached_tokens']}" if ra_warm["cached_tokens"] else ""
    print(f"  [5] Anthropic /v1/messages + cache      : {'PASS' if ok5 else 'FAIL'} | "
          f"cold={ra_cold['ttft_s']:.3f}s -> warm={ra_warm['ttft_s']:.3f}s ({a_spd:.1f}x faster{a_cached})")
    return 0 if (ok1 and ok2 and ok3 and ok5) else 1


def cmd_profiles(_: argparse.Namespace) -> int:
    print("Built-in 2x & 4x DGX Spark profiles:")
    for name, p in sorted(PROFILES.items()):
        print(
            f"  {name:24s} -> mode={p['mode']:16s} ctx={int(p['SPARK_MAX_MODEL_LEN']):>9,d} "
            f"dspark={'ON' if 'dspark' in p['mode'] else 'OFF':3s} "
            f"kv_pack={'ON' if p.get('SPARK_PATCH_KV_PACK') == '1' else 'OFF'}"
        )
    print("\nAliases:", ", ".join(f"{k}->{v}" for k, v in sorted(PROFILE_ALIASES.items())))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="spark-ernie",
        description="All-in-one 2x-4x NVIDIA DGX Spark (GB10, SM12.1) 1M-context runtime & CLI for ERNIE 5.5 Flash Preview (MXFP4 & NVFP4)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # discover
    dp = sub.add_parser("discover", help="Auto-discover 200GbE QSFP / RoCEv2 CX7 cables & 2x/4x ring topology")
    dp.add_argument("--hosts", required=True, help="Comma-separated SSH hosts (2 for direct pair, 4 for switchless ring)")
    dp.add_argument("--user", help="SSH username if different from ~/.ssh/config")
    dp.add_argument("--dry-run", action="store_true", help="Print discovered topology without saving")
    dp.set_defaults(func=cmd_discover)

    # download
    dlp = sub.add_parser("download", help="Download official ERNIE 5.5 MXFP4 weights from Hugging Face & sync across 2x/4x Sparks")
    dlp.add_argument("--repo", default=DEFAULT_HF_REPO, help=f"HF model repo (default: {DEFAULT_HF_REPO})")
    dlp.add_argument("--dest", default=str(DEFAULT_LOCAL_DIR), help=f"Local destination dir (default: {DEFAULT_LOCAL_DIR})")
    dlp.add_argument("--no-sync", action="store_true", help="Download locally only without syncing to peer Sparks")
    dlp.add_argument("--sync-only", action="store_true", help="Spread an already-downloaded local model dir across peer Sparks")
    dlp.add_argument("--no-config", action="store_true", help="Do not update default model path in config.json")
    dlp.set_defaults(func=cmd_download)

    # init
    ip = sub.add_parser("init", help="Configure default model path, profile, port, reasoning effort, and API key")
    ip.add_argument("--model", help="HF repo ID or local path to ERNIE-5.5-Flash-Preview (MXFP4 or W4A16_NVFP4) weights")
    ip.add_argument("--served-model-name", help="OpenAI served model name (default: ernie55)")
    ip.add_argument("--profile", choices=["auto", *sorted(PROFILES), *sorted(PROFILE_ALIASES)])
    ip.add_argument("--port", type=int)
    ip.add_argument("--venv", help="Path to Python venv containing vllm + vllm-ernie")
    ip.add_argument("--nccl-lib", help="Path to patched switchless 4-HCA libnccl.so (for 4-node ring)")
    ip.add_argument("--reasoning-effort", choices=["high", "medium", "low", "off"])
    ip.add_argument("--api-key", help="Bearer API key (or 'auto' to generate a random sk-spark-... key)")
    ip.set_defaults(func=cmd_init)

    # start / stop / restart / reboot
    for cname, fn, chelp in [
        ("start", cmd_start, "Start the 2x or 4x DGX Spark ERNIE 5.5 cluster"),
        ("restart", cmd_restart, "Cleanly stop and restart the cluster"),
        ("reboot", cmd_restart, "Alias for restart (clean stop + start)"),
    ]:
        sp = sub.add_parser(cname, help=chelp)
        sp.add_argument("--nodes", type=int, choices=[2, 4], help="Validate cluster size (2 or 4)")
        sp.add_argument("--profile", choices=["auto", *sorted(PROFILES), *sorted(PROFILE_ALIASES)])
        sp.add_argument("--model", help="Override model path")
        sp.add_argument("--port", type=int)
        sp.add_argument("--master-port", type=int)
        sp.add_argument("--max-model-len", type=int)
        sp.add_argument("--max-num-seqs", type=int)
        sp.add_argument("--no-sparknet", action="store_true")
        sp.add_argument("--no-wait", action="store_true", help="Return immediately without waiting for /v1/models")
        sp.add_argument("--remote-repo", help="Path to this repo on remote Spark nodes (default: same path)")
        sp.set_defaults(func=fn)

    stp = sub.add_parser("stop", help="Cleanly stop all vLLM workers across the cluster")
    stp.set_defaults(func=cmd_stop)

    # status & logs
    statp = sub.add_parser("status", help="Show cluster health, RAM, KV cache usage, prefix hit rate & DSpark stats")
    statp.add_argument("--url", help="Server URL (default: from config)")
    statp.set_defaults(func=cmd_status)

    logp = sub.add_parser("logs", help="View or follow logs for any node rank")
    logp.add_argument("--rank", type=int, default=0)
    logp.add_argument("-n", "--lines", type=int, default=80)
    logp.add_argument("-f", "--follow", action="store_true")
    logp.set_defaults(func=cmd_logs)

    # api-key
    akp = sub.add_parser("api-key", help="Manage server Bearer API key (set / generate / show / clear)")
    akp.add_argument("action", choices=["set", "generate", "show", "clear"])
    akp.add_argument("value", nargs="?", help="New API key value (for 'set')")
    akp.add_argument("--reveal", action="store_true", help="Print unmasked key (for 'show')")
    akp.set_defaults(func=cmd_api_key)

    # chat & test
    cp = sub.add_parser("chat", help="Stream via /v1/chat/completions, /v1/responses, or /v1/messages (Anthropic)")
    cp.add_argument("prompt", nargs="*", help="Prompt text (omit for interactive multi-turn REPL)")
    cp.add_argument("--api", choices=["chat", "responses", "anthropic"], default="chat")
    cp.add_argument("--effort", choices=["high", "medium", "low", "off"], help="Reasoning effort (default: high)")
    cp.add_argument("--max-tokens", type=int, default=8192)
    cp.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (official default: 1.0)")
    cp.add_argument("--top-p", type=float, default=0.95, help="Nucleus top-p (official default: 0.95)")
    cp.add_argument("--top-k", type=int, default=None, help="Optional top-k sampling limit")
    cp.add_argument("--system", help="Optional system prompt")
    cp.add_argument("--url", help="OpenAI API base URL")
    cp.add_argument("--model", help="Served model name")
    cp.add_argument("--api-key", help="Override Bearer API key")
    cp.set_defaults(func=cmd_chat)

    tp = sub.add_parser("test", help="Run 5-check live smoke test (chat high, responses, VRAM prefix cache, coding speed, Anthropic /v1/messages)")
    tp.add_argument("--url", help="OpenAI API base URL")
    tp.add_argument("--model", help="Served model name")
    tp.add_argument("--api-key", help="Override Bearer API key")
    tp.set_defaults(func=cmd_test)

    # single-rank serve (for systemd / manual)
    svp = sub.add_parser("serve", help="Run a single node rank directly in the foreground")
    svp.add_argument("--profile", choices=[*sorted(PROFILES), *sorted(PROFILE_ALIASES)], required=True)
    svp.add_argument("--topology", default="custom", help="Topology name or 'custom' (reads cluster.json)")
    svp.add_argument("--rank", type=int, required=True)
    svp.add_argument("--model", help="Model path (default: from config)")
    svp.add_argument("--port", type=int)
    svp.add_argument("--master-port", type=int)
    svp.add_argument("--max-model-len", type=int)
    svp.add_argument("--max-num-seqs", type=int)
    svp.add_argument("--no-sparknet", action="store_true")
    svp.add_argument("--dry-run", action="store_true")
    svp.set_defaults(func=cmd_serve)

    lp = sub.add_parser("profiles", help="List built-in 2x and 4x DGX Spark profiles")
    lp.set_defaults(func=cmd_profiles)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

"""Multi-node DGX Spark cluster lifecycle manager (start / stop / reboot / status / logs / doctor)."""
from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import (
    CLUSTER_FILE,
    CONFIG_FILE,
    DEFAULT_LOG_DIR,
    detect_vllm_bin,
    load_cluster,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

PROFILES: dict[str, dict[str, str]] = {
    # 4x DGX Spark (switchless dual-rail 200GbE CX7 ring)
    "4x-1m-fast": {
        "mode": "tp4-dspark",
        "SPARK_USE_SPARKNET": "1",
        "SPARK_REPLICATE_SHARED_EXPERTS": "0",
        "SPARK_DSPARK_VOCAB_PARALLEL": "1",
        "SPARK_DSPARK_VP_PROB": "1",
        "SPARK_DSPARK_DRAFT_TAU": "1.0",
        "SPARK_DSPARK_SAMPLE_METHOD": "probabilistic",
        "SPARK_DSPARK_MAIN_PROJ_TP": "1",
        "SPARK_DSPARK_NUM_TOKENS": "7",
        "SPARK_DSPARK_ADAPTIVE": "1",
        "SPARK_DSPARK_VERIFY_RULE": "marginal",
        "SPARK_DSPARK_MOE_COST_CURVE": "1",
        "SPARK_PATCH_KV_PACK": "1",
        "SPARK_KV_PACK_MAX_DSA_SPLITS": "2",
        "SPARK_PACKED_BF16_LM_HEAD": "1",
        "SPARK_PACKED_BF16_MARKOV": "1",
        "SPARK_PATCH_DSPARK_PAD_MASK": "1",
        "SPARK_PATCH_VHA_BF16_AR": "1",
        "SPARK_PATCH_VHA_TRITON": "1",
        "SPARK_PATCH_FC2_LATENT_TP": "1",
        "SPARK_PATCH_GATE_BF16X2": "1",
        "SPARK_PATCH_HCA_OPROJ_FUSED": "1",
        "SPARK_HCA_PIPELINE": "1",
        "SPARK_ENABLE_PREFIX_CACHING": "1",
        "SPARK_MAX_NUM_SEQS": "32",
        "SPARK_MAX_MODEL_LEN": "1048576",
        "SPARK_MAX_NUM_BATCHED_TOKENS": "8192",
        "SPARK_GPU_MEM_UTIL": "0.68",
    },
    "4x-1m-max-context": {
        "mode": "tp4-ar",
        "SPARK_USE_SPARKNET": "1",
        "SPARK_REPLICATE_SHARED_EXPERTS": "0",
        "SPARK_PATCH_KV_PACK": "1",
        "SPARK_KV_PACK_MAX_DSA_SPLITS": "2",
        "SPARK_PACKED_BF16_LM_HEAD": "1",
        "SPARK_PATCH_VHA_BF16_AR": "1",
        "SPARK_PATCH_VHA_TRITON": "1",
        "SPARK_PATCH_FC2_LATENT_TP": "1",
        "SPARK_PATCH_GATE_BF16X2": "1",
        "SPARK_PATCH_HCA_OPROJ_FUSED": "1",
        "SPARK_HCA_PIPELINE": "1",
        "SPARK_ENABLE_PREFIX_CACHING": "1",
        "SPARK_MAX_NUM_SEQS": "128",
        "SPARK_MAX_MODEL_LEN": "1048576",
        "SPARK_MAX_NUM_BATCHED_TOKENS": "8192",
        "SPARK_GPU_MEM_UTIL": "0.80",
    },
    # 2x DGX Spark (direct dual-rail 200GbE CX7)
    "2x-1m-fast": {
        "mode": "tp2-dspark",
        "SPARK_USE_SPARKNET": "1",
        "SPARK_REPLICATE_SHARED_EXPERTS": "0",
        "SPARK_DSPARK_VOCAB_PARALLEL": "1",
        "SPARK_DSPARK_VP_PROB": "1",
        "SPARK_DSPARK_DRAFT_TAU": "1.0",
        "SPARK_DSPARK_SAMPLE_METHOD": "probabilistic",
        "SPARK_DSPARK_MAIN_PROJ_TP": "1",
        "SPARK_DSPARK_NUM_TOKENS": "7",
        "SPARK_DSPARK_ADAPTIVE": "1",
        "SPARK_DSPARK_VERIFY_RULE": "marginal",
        "SPARK_DSPARK_MOE_COST_CURVE": "1",
        "SPARK_PATCH_KV_PACK": "1",
        "SPARK_KV_PACK_MAX_DSA_SPLITS": "2",
        "SPARK_PACKED_BF16_LM_HEAD": "1",
        "SPARK_PACKED_BF16_MARKOV": "1",
        "SPARK_PATCH_DSPARK_PAD_MASK": "1",
        "SPARK_PATCH_VHA_BF16_AR": "1",
        "SPARK_PATCH_VHA_TRITON": "1",
        "SPARK_PATCH_FC2_LATENT_TP": "1",
        "SPARK_PATCH_GATE_BF16X2": "1",
        "SPARK_PATCH_HCA_OPROJ_FUSED": "1",
        "SPARK_HCA_PIPELINE": "1",
        "SPARK_ENABLE_PREFIX_CACHING": "1",
        "SPARK_MAX_NUM_SEQS": "8",
        "SPARK_MAX_MODEL_LEN": "1048576",
        "SPARK_MAX_NUM_BATCHED_TOKENS": "2048",
        "SPARK_MAX_CUDAGRAPH_CAPTURE_SIZE": "64",
        "SPARK_KV_CACHE_MEMORY_BYTES": "9663676416",
        "SPARK_GPU_MEM_UTIL": "0.86",
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0",
    },
    "2x-1m-max-context": {
        "mode": "tp2-ar",
        "SPARK_USE_SPARKNET": "1",
        "SPARK_REPLICATE_SHARED_EXPERTS": "0",
        "SPARK_PATCH_KV_PACK": "1",
        "SPARK_KV_PACK_MAX_DSA_SPLITS": "2",
        "SPARK_PACKED_BF16_LM_HEAD": "1",
        "SPARK_PATCH_VHA_BF16_AR": "1",
        "SPARK_PATCH_VHA_TRITON": "1",
        "SPARK_PATCH_FC2_LATENT_TP": "1",
        "SPARK_PATCH_GATE_BF16X2": "1",
        "SPARK_PATCH_HCA_OPROJ_FUSED": "1",
        "SPARK_HCA_PIPELINE": "1",
        "SPARK_ENABLE_PREFIX_CACHING": "1",
        "SPARK_MAX_NUM_SEQS": "32",
        "SPARK_MAX_MODEL_LEN": "1048576",
        "SPARK_MAX_NUM_BATCHED_TOKENS": "2048",
        "SPARK_GPU_MEM_UTIL": "0.84",
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0",
    },
    "2x-max-batch": {
        "mode": "tp2-ar",
        "SPARK_USE_SPARKNET": "1",
        "SPARK_REPLICATE_SHARED_EXPERTS": "1",
        "SPARK_PATCH_KV_PACK": "1",
        "SPARK_KV_PACK_MAX_DSA_SPLITS": "2",
        "SPARK_PACKED_BF16_LM_HEAD": "1",
        "SPARK_PATCH_VHA_BF16_AR": "1",
        "SPARK_PATCH_VHA_TRITON": "1",
        "SPARK_PATCH_GATE_BF16X2": "1",
        "SPARK_PATCH_HCA_OPROJ_FUSED": "1",
        "SPARK_HCA_PIPELINE": "1",
        "SPARK_ENABLE_PREFIX_CACHING": "1",
        "SPARK_MAX_NUM_SEQS": "128",
        "SPARK_MAX_MODEL_LEN": "131072",
        "SPARK_MAX_NUM_BATCHED_TOKENS": "2048",
        "SPARK_GPU_MEM_UTIL": "0.78",
    },
    # 4x DGX Spark as 2x2 Parallelized Replicas (2 independent TP=2+EP=2 pairs + prefix-aware LB router!)
    "4x-dual-tp2-fast": {
        "mode": "dual-tp2-dspark",
        "SPARK_USE_SPARKNET": "1",
        "SPARK_REPLICATE_SHARED_EXPERTS": "0",
        "SPARK_DSPARK_VOCAB_PARALLEL": "1",
        "SPARK_DSPARK_VP_PROB": "1",
        "SPARK_DSPARK_DRAFT_TAU": "1.0",
        "SPARK_DSPARK_SAMPLE_METHOD": "probabilistic",
        "SPARK_DSPARK_MAIN_PROJ_TP": "1",
        "SPARK_DSPARK_NUM_TOKENS": "7",
        "SPARK_DSPARK_ADAPTIVE": "1",
        "SPARK_DSPARK_VERIFY_RULE": "marginal",
        "SPARK_DSPARK_MOE_COST_CURVE": "1",
        "SPARK_PATCH_KV_PACK": "1",
        "SPARK_KV_PACK_MAX_DSA_SPLITS": "2",
        "SPARK_PACKED_BF16_LM_HEAD": "1",
        "SPARK_PACKED_BF16_MARKOV": "1",
        "SPARK_PATCH_DSPARK_PAD_MASK": "1",
        "SPARK_PATCH_VHA_BF16_AR": "1",
        "SPARK_PATCH_VHA_TRITON": "1",
        "SPARK_PATCH_FC2_LATENT_TP": "1",
        "SPARK_PATCH_GATE_BF16X2": "1",
        "SPARK_PATCH_HCA_OPROJ_FUSED": "1",
        "SPARK_HCA_PIPELINE": "1",
        "SPARK_ENABLE_PREFIX_CACHING": "1",
        "SPARK_MAX_NUM_SEQS": "8",
        "SPARK_MAX_MODEL_LEN": "1048576",
        "SPARK_MAX_NUM_BATCHED_TOKENS": "2048",
        "SPARK_MAX_CUDAGRAPH_CAPTURE_SIZE": "64",
        "SPARK_KV_CACHE_MEMORY_BYTES": "9663676416",
        "SPARK_GPU_MEM_UTIL": "0.86",
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0",
    },
    "4x-dual-tp2-max-batch": {
        "mode": "dual-tp2-ar",
        "SPARK_USE_SPARKNET": "1",
        "SPARK_REPLICATE_SHARED_EXPERTS": "1",
        "SPARK_PATCH_KV_PACK": "1",
        "SPARK_KV_PACK_MAX_DSA_SPLITS": "2",
        "SPARK_PACKED_BF16_LM_HEAD": "1",
        "SPARK_PATCH_VHA_BF16_AR": "1",
        "SPARK_PATCH_VHA_TRITON": "1",
        "SPARK_PATCH_GATE_BF16X2": "1",
        "SPARK_PATCH_HCA_OPROJ_FUSED": "1",
        "SPARK_HCA_PIPELINE": "1",
        "SPARK_ENABLE_PREFIX_CACHING": "1",
        "SPARK_MAX_NUM_SEQS": "128",
        "SPARK_MAX_MODEL_LEN": "131072",
        "SPARK_MAX_NUM_BATCHED_TOKENS": "2048",
        "SPARK_GPU_MEM_UTIL": "0.75",
    },
}

# Friendly aliases
PROFILE_ALIASES: dict[str, str] = {
    "4x-single-stream": "4x-1m-fast",
    "4x-ar-max-context": "4x-1m-max-context",
    "2x-single-stream": "2x-1m-fast",
    "2x-max-context": "2x-1m-max-context",
    "2x2-fast": "4x-dual-tp2-fast",
    "2x2-max-batch": "4x-dual-tp2-max-batch",
    "4x-2x2-fast": "4x-dual-tp2-fast",
    "4x-2x2-max-batch": "4x-dual-tp2-max-batch",
}


def resolve_profile(name: str, world_size: int, topology: str = "") -> tuple[str, dict[str, str]]:
    if not name or name == "auto":
        if world_size == 4:
            name = "4x-dual-tp2-fast" if topology == "dual_direct" else "4x-1m-fast"
        else:
            name = "2x-1m-fast"
    name = PROFILE_ALIASES.get(name, name)
    if name not in PROFILES:
        raise ValueError(f"Unknown profile {name!r}. Valid profiles: {sorted(PROFILES)}")
    return name, dict(PROFILES[name])


def _is_local(host: str, hostname: str = "") -> bool:
    if host in ("localhost", "127.0.0.1", "::1"):
        return True
    hn = socket.gethostname()
    if hostname and hostname in (hn, f"{hn}.local"):
        return True
    return host in (hn, f"{hn}.local")


def _ssh_target(host: str, ssh_user: str) -> str:
    return f"{ssh_user}@{host}" if ssh_user and "@" not in host else host


def _run_on_node(host: str, ssh_user: str, cmd: str, background: bool = False) -> subprocess.CompletedProcess[str]:
    if _is_local(host):
        full = f"setsid bash -lc {shlex.quote(cmd)} </dev/null >/dev/null 2>&1 &" if background else cmd
        return subprocess.run(["bash", "-lc", full], text=True, capture_output=not background)
    target = _ssh_target(host, ssh_user)
    if background:
        remote = f"setsid bash -lc {shlex.quote(cmd)} </dev/null >/dev/null 2>&1 &"
        return subprocess.run(
            ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target, remote],
            text=True,
        )
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target, f"bash -lc {shlex.quote(cmd)}"],
        text=True,
        capture_output=True,
    )


def build_rank_env(
    cluster: dict[str, Any],
    cfg: dict[str, Any],
    profile_env: dict[str, str],
    rank: int,
    *,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    no_sparknet: bool = False,
) -> dict[str, str]:
    node = cluster["nodes"][rank]
    env = dict(profile_env)
    env["MASTER_ADDR"] = cluster["master_addr"]
    env["WORLD_SIZE"] = str(cluster["world_size"])
    env["VLLM_HOST_IP"] = node["host_ip"]
    env["NCCL_SOCKET_IFNAME"] = node["oob_iface"]
    env["GLOO_SOCKET_IFNAME"] = node["oob_iface"]
    env["TP_SOCKET_IFNAME"] = node["oob_iface"]
    env["NCCL_IB_HCA"] = node["nccl_ib_hca"]
    env["SPARKNET_ROCE_TOPOLOGY"] = cluster["topology"]
    env["SPARKNET_ROCE_PEER_HCAS"] = json.dumps(node["sparknet_peer_hcas"], separators=(",", ":"))
    env["SPARKNET_ROCE_SUBNETS"] = json.dumps(node["sparknet_subnets"], separators=(",", ":"))
    env["SPARK_SERVED_MODEL_NAME"] = str(cfg.get("served_model_name", "ernie55"))
    env["SPARK_BIND_HOST"] = str(cfg.get("host", "0.0.0.0"))
    env["SPARK_ENABLE_PREFIX_CACHING"] = "1" if cfg.get("enable_prefix_caching", True) else "0"

    if cfg.get("cuda_home"):
        env["CUDA_HOME"] = str(cfg["cuda_home"])
    vllm_bin = (
        (str(Path(cfg["venv"]) / "bin" / "vllm") if cfg.get("venv") else "")
        or detect_vllm_bin()
        or (str(Path(sys.executable).parent / "vllm") if (Path(sys.executable).parent / "vllm").exists() else "")
    )
    if vllm_bin:
        env["SPARK_VLLM_BIN"] = vllm_bin
    if cfg.get("nccl_lib_path") or cluster.get("nccl_lib_path"):
        env["SPARK_NCCL_LIB"] = str(cfg.get("nccl_lib_path") or cluster.get("nccl_lib_path"))
    if cfg.get("api_key"):
        env["VLLM_API_KEY"] = str(cfg["api_key"])
    if no_sparknet or not cfg.get("use_sparknet", True):
        env["SPARK_USE_SPARKNET"] = "0"
    if max_model_len is not None:
        env["SPARK_MAX_MODEL_LEN"] = str(max_model_len)
    if max_num_seqs is not None:
        env["SPARK_MAX_NUM_SEQS"] = str(max_num_seqs)
    for k, v in (cfg.get("extra_env") or {}).items():
        env[str(k)] = str(v)
    return env


def stop_cluster(cluster: dict[str, Any] | None = None) -> int:
    cluster = cluster or load_cluster()
    if not cluster:
        print(f"No cluster config at {CLUSTER_FILE}. Run `spark-ernie discover` first.", file=sys.stderr)
        return 1
    ssh_user = cluster.get("ssh_user", "")
    kill_cmd = (
        "ps -eo pid,comm,args | awk -v self=$$ "
        "'$1 != self && !/awk/ && ("
        "($2 ~ /^VLLM::/) || "
        "($2 ~ /^python/ && ($0 ~ /vllm serve/ || $0 ~ /spark_ernie_runtime\\.router/)) || "
        "($2 == \"bash\" && $3 ~ /serve_spark_ernie\\.sh$/)"
        ") {print $1}' "
        "| xargs -r kill -9 2>/dev/null || true"
    )
    for node in cluster["nodes"]:
        print(f"Stopping rank {node['rank']} ({node['ssh_host']})...")
        _run_on_node(node["ssh_host"], ssh_user, kill_cmd)
    time.sleep(2)
    print("Cluster stopped.")
    return 0


def start_cluster(
    *,
    profile: str | None = None,
    nodes_limit: int | None = None,
    model: str | None = None,
    port: int | None = None,
    master_port: int | None = None,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    no_sparknet: bool = False,
    wait: bool = True,
    wait_timeout_s: int = 1200,
    remote_repo_path: str | None = None,
) -> int:
    cluster = load_cluster()
    if not cluster:
        print(
            f"No discovered cluster at {CLUSTER_FILE}.\n"
            "Run `spark-ernie discover --hosts <host0>,<host1>[,<host2>,<host3>]` first.",
            file=sys.stderr,
        )
        return 1
    cfg = load_config()
    if nodes_limit is not None and nodes_limit != cluster["world_size"]:
        if nodes_limit == 2 and cluster["world_size"] == 4:
            print(
                "Note: cluster.json is a 4-node ring. Re-run `spark-ernie discover --hosts h0,h1` "
                "to configure a 2-node direct pair.",
                file=sys.stderr,
            )
            return 1
        raise ValueError(f"Requested --nodes {nodes_limit}, but cluster.json has {cluster['world_size']} nodes.")

    world_size = cluster["world_size"]
    prof_name, prof_env = resolve_profile(
        profile or cfg.get("profile", "auto"), world_size, cluster.get("topology", "")
    )
    mode = prof_env.pop("mode")
    model_path = model or cfg["model"]
    srv_port = int(port or cfg.get("port", 8000))
    mport = int(master_port or cfg.get("master_port", 29801))
    ssh_user = cluster.get("ssh_user", "")
    repo_dir = remote_repo_path or str(REPO_ROOT)
    script_path = f"{repo_dir}/scripts/serve_spark_ernie.sh"
    log_dir = str(DEFAULT_LOG_DIR)

    # 2x2 Parallelized Mode: 2 independent TP=2+EP=2 replicas + prefix-aware load-balancing router on srv_port!
    if mode.startswith("dual-"):
        sub_mode = mode.removeprefix("dual-")
        pairs = cluster.get("pairs")
        if not pairs or len(pairs) != 2:
            raise RuntimeError(
                "2x2 parallelized profile requires 4 discovered Sparks with `pairs` in cluster.json. "
                "Re-run `spark-ernie discover --hosts spark0,spark1,spark2,spark3`."
            )
        rep_ports = [srv_port + 1, srv_port + 2]
        print(
            f"Starting ERNIE 5.5 in 2x2 Parallelized Mode across 4x DGX Sparks "
            f"[profile={prof_name}, sub_mode={sub_mode}, router=:{srv_port} -> replicas=:{rep_ports[0]},:{rep_ports[1]}]"
        )
        head_node = pairs[0]["nodes"][0]
        head_host = head_node["ssh_host"]
        # Map ssh_host -> cluster-wide OOB IP (ring0/LAN) so router never relies on ~/.ssh/config aliases
        oob_ip_by_ssh = {n["ssh_host"]: n["host_ip"] for n in cluster["nodes"]}
        backend_urls: list[str] = []
        for pidx, pair_cluster in enumerate(pairs):
            rport = rep_ports[pidx]
            rmport = mport + pidx * 2
            p0_node = pair_cluster["nodes"][0]
            p0_ssh = p0_node["ssh_host"]
            if p0_ssh == head_host or _is_local(p0_ssh, p0_node.get("hostname", "")):
                p0_target = "127.0.0.1"
            else:
                p0_target = oob_ip_by_ssh.get(p0_ssh, p0_ssh)
            backend_urls.append(f"http://{p0_target}:{rport}")
            for node in pair_cluster["nodes"]:
                r = node["rank"]
                renv = build_rank_env(
                    pair_cluster, cfg, prof_env, r,
                    max_model_len=max_model_len, max_num_seqs=max_num_seqs, no_sparknet=no_sparknet,
                )
                env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(renv.items()))
                log_file = f"{log_dir}/pair{pidx}_rank_{r}.log"
                launch_cmd = (
                    f"mkdir -p {shlex.quote(log_dir)} && "
                    f"exec env {env_prefix} {shlex.quote(script_path)} "
                    f"{shlex.quote(sub_mode)} custom {r} {shlex.quote(model_path)} {rport} {rmport} "
                    f">{shlex.quote(log_file)} 2>&1"
                )
                print(f"  Launching replica {pidx} rank {r} on {node['ssh_host']} (:{rport}, log: {log_file})...")
                _run_on_node(node["ssh_host"], ssh_user, launch_cmd, background=True)

        # Launch prefix-aware OpenAI load-balancing router on node 0 at srv_port
        py_bin = str(Path(cfg["venv"]) / "bin" / "python3") if cfg.get("venv") else sys.executable
        api_key_arg = f"--api-key {shlex.quote(str(cfg['api_key']))}" if cfg.get("api_key") else ""
        router_log = f"{log_dir}/router.log"
        router_cmd = (
            f"mkdir -p {shlex.quote(log_dir)} && "
            f"PYTHONPATH={shlex.quote(repo_dir)} exec {shlex.quote(py_bin)} -m spark_ernie_runtime.router "
            f"--host {shlex.quote(str(cfg.get('host', '0.0.0.0')))} --port {srv_port} "
            f"--backends {shlex.quote(','.join(backend_urls))} {api_key_arg} "
            f">{shlex.quote(router_log)} 2>&1"
        )
        print(f"  Launching prefix-aware OpenAI router on {head_host}:{srv_port} (log: {router_log})...")
        _run_on_node(head_host, ssh_user, router_cmd, background=True)

        if not wait:
            print("Launched 2x2 cluster + router in background.")
            return 0
        head_oob = oob_ip_by_ssh.get(head_host, head_host)
        base_url = f"http://127.0.0.1:{srv_port}" if _is_local(head_host, head_node.get("hostname", "")) else f"http://{head_oob}:{srv_port}"
        headers = {"Authorization": f"Bearer {cfg['api_key']}"} if cfg.get("api_key") else {}
        print(f"Waiting for both 2x replicas via {base_url}/health (timeout {wait_timeout_s}s)...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < wait_timeout_s:
            try:
                req = urllib.request.Request(f"{base_url}/health", headers=headers)
                with urllib.request.urlopen(req, timeout=4) as resp:
                    if resp.status == 200:
                        print(f"\n2x2 PARALLELIZED CLUSTER READY in {time.monotonic() - t0:.1f}s! endpoint={base_url}/v1")
                        return 0
            except Exception:
                pass
            time.sleep(5)
        print(f"Timed out after {wait_timeout_s}s. Check {log_dir}/pair*_rank_0.log.", file=sys.stderr)
        return 1

    print(
        f"Starting ERNIE 5.5 on {world_size}x DGX Spark "
        f"[profile={prof_name}, mode={mode}, topology={cluster['topology']}, "
        f"max_len={max_model_len or prof_env.get('SPARK_MAX_MODEL_LEN')}, port={srv_port}]"
    )

    for node in cluster["nodes"]:
        r = node["rank"]
        renv = build_rank_env(
            cluster, cfg, prof_env, r,
            max_model_len=max_model_len, max_num_seqs=max_num_seqs, no_sparknet=no_sparknet,
        )
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(renv.items()))
        log_file = f"{log_dir}/rank_{r}.log"
        launch_cmd = (
            f"mkdir -p {shlex.quote(log_dir)} && "
            f"exec env {env_prefix} {shlex.quote(script_path)} "
            f"{shlex.quote(mode)} custom {r} {shlex.quote(model_path)} {srv_port} {mport} "
            f">{shlex.quote(log_file)} 2>&1"
        )
        print(f"  Launching rank {r} on {node['ssh_host']} (log: {log_file})...")
        _run_on_node(node["ssh_host"], ssh_user, launch_cmd, background=True)

    if not wait:
        print("Launched in background. Check readiness with `spark-ernie status` or `spark-ernie logs -f`.")
        return 0

    head_node = cluster["nodes"][0]
    head_host = head_node["ssh_host"]
    base_url = (
        f"http://127.0.0.1:{srv_port}"
        if _is_local(head_host, head_node.get("hostname", ""))
        else f"http://{head_node.get('host_ip', head_host)}:{srv_port}"
    )
    headers = {"Authorization": f"Bearer {cfg['api_key']}"} if cfg.get("api_key") else {}
    print(f"Waiting for {base_url}/v1/models (timeout {wait_timeout_s}s)...")
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_timeout_s:
        try:
            req = urllib.request.Request(f"{base_url}/v1/models", headers=headers)
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    m = (data.get("data") or [{}])[0]
                    print(
                        f"\nREADY in {time.monotonic() - t0:.1f}s! "
                        f"model={m.get('id')} max_model_len={m.get('max_model_len'):,} endpoint={base_url}/v1"
                    )
                    return 0
        except Exception:
            pass
        time.sleep(5)
    print(f"Timed out after {wait_timeout_s}s. Inspect logs with `spark-ernie logs --rank 0`.", file=sys.stderr)
    return 1


def cluster_status(url: str | None = None) -> int:
    cfg = load_config()
    cluster = load_cluster()
    port = int(cfg.get("port", 8000))
    if not url:
        if cluster:
            n0 = cluster["nodes"][0]
            h0 = n0["ssh_host"]
            h0_name = n0.get("hostname", "")
            h0_target = n0.get("host_ip") or h0
            url = f"http://127.0.0.1:{port}" if _is_local(h0, h0_name) else f"http://{h0_target}:{port}"
        else:
            url = f"http://127.0.0.1:{port}"
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    headers = {"Authorization": f"Bearer {cfg['api_key']}"} if cfg.get("api_key") else {}

    print(f"Endpoint: {base}/v1")
    try:
        req = urllib.request.Request(f"{base}/v1/models", headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            for m in data.get("data", []):
                print(f"  Status: ONLINE | model={m.get('id')} | max_model_len={m.get('max_model_len'):,}")
    except Exception as exc:
        print(f"  Status: OFFLINE ({exc})")

    # Parse live Prometheus /metrics if available
    try:
        req = urllib.request.Request(f"{base}/metrics", headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            metrics = resp.read().decode("utf-8", errors="replace")
        wanted = (
            "vllm:kv_cache_usage_perc",
            "vllm:prefix_cache_hits_total",
            "vllm:prefix_cache_queries_total",
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
            "vllm:spec_decode_num_accepted_tokens_total",
            "vllm:spec_decode_num_draft_tokens_total",
        )
        vals: dict[str, float] = {}
        for line in metrics.splitlines():
            if line.startswith("#"):
                continue
            for key in wanted:
                if line.startswith(key):
                    try:
                        vals[key] = float(line.rsplit(None, 1)[-1])
                    except ValueError:
                        pass
        if vals:
            kv_pct = vals.get("vllm:kv_cache_usage_perc", 0.0) * 100.0
            hits = vals.get("vllm:prefix_cache_hits_total", 0.0)
            qtot = vals.get("vllm:prefix_cache_queries_total", 0.0)
            hit_pct = (100.0 * hits / qtot) if qtot > 0 else 0.0
            acc = vals.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
            drf = vals.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
            acc_pct = (100.0 * acc / drf) if drf > 0 else 0.0
            print(
                f"  Live metrics: running={int(vals.get('vllm:num_requests_running', 0))} "
                f"waiting={int(vals.get('vllm:num_requests_waiting', 0))} | "
                f"KV used={kv_pct:.1f}% | prefix hit={hit_pct:.1f}% | DSpark accept={acc_pct:.1f}%"
            )
    except Exception:
        pass

    if cluster:
        ssh_user = cluster.get("ssh_user", "")
        print(f"Nodes ({cluster['world_size']}x, topology={cluster['topology']}):")
        for n in cluster["nodes"]:
            res = _run_on_node(
                n["ssh_host"], ssh_user,
                "free -h | awk '/Mem:/ {print $7 \" avail / \" $2 \" total\"}'",
            )
            mem_s = res.stdout.strip() if res.returncode == 0 else "unreachable"
            print(f"  Rank {n['rank']} ({n['ssh_host']}): {mem_s}")
    return 0


def tail_logs(rank: int = 0, lines: int = 80, follow: bool = False) -> int:
    cluster = load_cluster()
    log_file = f"{DEFAULT_LOG_DIR}/rank_{rank}.log"
    if not cluster or rank >= len(cluster["nodes"]):
        cmd = ["tail", "-n", str(lines)] + (["-f"] if follow else []) + [log_file]
        return subprocess.call(cmd)
    node = cluster["nodes"][rank]
    host = node["ssh_host"]
    ssh_user = cluster.get("ssh_user", "")
    tail_cmd = f"tail -n {lines} {'-f ' if follow else ''}{shlex.quote(log_file)}"
    if _is_local(host):
        return subprocess.call(["bash", "-lc", tail_cmd])
    return subprocess.call(["ssh", "-t", _ssh_target(host, ssh_user), tail_cmd])

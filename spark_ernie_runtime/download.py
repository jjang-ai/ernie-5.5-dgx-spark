"""Fast model download & 200GbE parallel cluster sync for ERNIE 5.5 Flash Preview MXFP4."""
from __future__ import annotations

import concurrent.futures
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from .cluster import _is_local, _ssh_target
from .config import load_cluster, load_config, save_config

DEFAULT_HF_REPO = "dealignai/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4"
DEFAULT_LOCAL_DIR = Path.home() / "models" / "ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4"


def _find_hf_downloader() -> list[str]:
    if shutil.which("hf"):
        return ["hf", "download"]
    if shutil.which("huggingface-cli"):
        return ["huggingface-cli", "download"]
    return [sys.executable, "-c",
            "import sys; from huggingface_hub import snapshot_download; "
            "snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])"]


def download_model(
    repo_id: str = DEFAULT_HF_REPO,
    dest_dir: Path | str = DEFAULT_LOCAL_DIR,
    *,
    sync_cluster: bool = True,
    update_config: bool = True,
    hf_token: str | None = None,
) -> int:
    """Download ERNIE 5.5 MXFP4 once on rank 0, then fan out across 2x/4x Sparks."""
    dest = Path(dest_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    if hf_token:
        env["HF_TOKEN"] = hf_token

    base_cmd = _find_hf_downloader()
    if base_cmd[0] in ("hf", "huggingface-cli"):
        cmd = [*base_cmd, repo_id, "--local-dir", str(dest)]
    else:
        cmd = [*base_cmd, repo_id, str(dest)]

    print(f"==> Downloading {repo_id} -> {dest}")
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        print(f"Download failed with exit code {rc}.", file=sys.stderr)
        return rc

    if update_config:
        cfg = load_config()
        cfg["model"] = str(dest)
        save_config(cfg)
        print(f"==> Updated default model path in config -> {dest}")

    if not sync_cluster:
        return 0

    cluster = load_cluster()
    if not cluster or len(cluster.get("nodes", [])) <= 1:
        print("==> No multi-node cluster.json found yet; skipping peer sync.")
        print("    (Run `spark-ernie discover --hosts ...` then `spark-ernie download --sync-only` to spread across nodes.)")
        return 0

    ssh_user = cluster.get("ssh_user", "")
    peers = [n for n in cluster["nodes"] if not _is_local(n["ssh_host"])]
    if not peers:
        return 0

    print(f"==> Spreading {dest} in parallel to {len(peers)} peer DGX Spark node(s)...")

    def _sync_one(node: dict) -> tuple[str, int]:
        # Prefer high-speed 200GbE host_ip when reachable over SSH, else ssh_host
        host = node["ssh_host"]
        target = _ssh_target(host, ssh_user)
        mkdir_rc = subprocess.call(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target, f"mkdir -p {shlex.quote(str(dest))}"]
        )
        if mkdir_rc != 0:
            return host, mkdir_rc
        rs_cmd = [
            "rsync", "-a", "--info=progress2",
            "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10",
            f"{str(dest).rstrip('/')}/",
            f"{target}:{str(dest).rstrip('/')}/",
        ]
        return host, subprocess.call(rs_cmd)

    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(peers)) as ex:
        for host, rcode in ex.map(_sync_one, peers):
            if rcode == 0:
                print(f"  [OK] Synced model to {host}:{dest}")
            else:
                print(f"  [FAIL] Sync to {host} exited with {rcode}", file=sys.stderr)
                failed.append(host)

    return 1 if failed else 0


def sync_existing_model(dest_dir: Path | str | None = None) -> int:
    """Sync an already-downloaded local model directory to all peer Sparks in cluster.json."""
    cfg = load_config()
    dest = Path(dest_dir or cfg.get("model") or DEFAULT_LOCAL_DIR).expanduser().resolve()
    if not dest.exists():
        print(f"Local model directory not found: {dest}", file=sys.stderr)
        return 1
    cluster = load_cluster()
    if not cluster or len(cluster.get("nodes", [])) <= 1:
        print("No multi-node cluster.json found. Run `spark-ernie discover` first.", file=sys.stderr)
        return 1
    ssh_user = cluster.get("ssh_user", "")
    peers = [n for n in cluster["nodes"] if not _is_local(n["ssh_host"])]
    print(f"==> Syncing {dest} in parallel to {len(peers)} peer Spark(s)...")

    def _sync_one(node: dict) -> tuple[str, int]:
        target = _ssh_target(node["ssh_host"], ssh_user)
        subprocess.call(["ssh", "-o", "BatchMode=yes", target, f"mkdir -p {shlex.quote(str(dest))}"])
        return node["ssh_host"], subprocess.call([
            "rsync", "-a", "--info=progress2",
            "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10",
            f"{str(dest).rstrip('/')}/",
            f"{target}:{str(dest).rstrip('/')}/",
        ])

    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(peers))) as ex:
        for host, rcode in ex.map(_sync_one, peers):
            if rcode == 0:
                print(f"  [OK] Synced model to {host}:{dest}")
            else:
                failed.append(host)
    return 1 if failed else 0

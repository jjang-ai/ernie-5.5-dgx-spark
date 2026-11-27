"""Portable user configuration & API key management for ernie-5.5-dgx-spark.

All machine-specific state (hostnames, discovered 200GbE cables, model path,
API key) lives strictly under `~/.config/spark-ernie/` — never inside the repo.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import sys
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(os.environ.get("SPARK_ERNIE_CONFIG_DIR", Path.home() / ".config" / "spark-ernie"))
CONFIG_FILE = CONFIG_DIR / "config.json"
CLUSTER_FILE = CONFIG_DIR / "cluster.json"
DEFAULT_LOG_DIR = Path(os.environ.get("SPARK_ERNIE_LOG_DIR", "/tmp/spark-ernie"))

DEFAULT_CONFIG: dict[str, Any] = {
    "model": "dealignai/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4",
    "served_model_name": "ernie55",
    "profile": "auto",
    "host": "0.0.0.0",
    "port": 8000,
    "master_port": 29801,
    "api_key": "",
    "default_reasoning_effort": "high",
    "enable_prefix_caching": True,
    "use_sparknet": True,
    "dspark_adaptive": True,
    "venv": "",
    "nccl_lib_path": "",
    "cuda_home": "/usr/local/cuda",
    "extra_env": {},
}


def ensure_config_dir() -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                cfg.update(data)
        except Exception:
            pass
    return cfg


def save_config(cfg: dict[str, Any], path: Path = CONFIG_FILE) -> Path:
    ensure_config_dir()
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def load_cluster(path: Path = CLUSTER_FILE) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_cluster(cluster: dict[str, Any], path: Path = CLUSTER_FILE) -> Path:
    ensure_config_dir()
    path.write_text(json.dumps(cluster, indent=2) + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def generate_api_key(prefix: str = "sk-spark-") -> str:
    return prefix + secrets.token_urlsafe(24)


def mask_api_key(key: str) -> str:
    if not key:
        return "(none — open local access)"
    if len(key) <= 12:
        return "***"
    return f"{key[:9]}...{key[-4:]}"


def detect_vllm_bin(cfg: dict[str, Any] | None = None) -> str:
    if cfg and cfg.get("venv"):
        cand = Path(cfg["venv"]) / "bin" / "vllm"
        if cand.exists():
            return str(cand)
    if os.environ.get("SPARK_VLLM_BIN"):
        return os.environ["SPARK_VLLM_BIN"]
    if os.environ.get("VIRTUAL_ENV"):
        cand = Path(os.environ["VIRTUAL_ENV"]) / "bin" / "vllm"
        if cand.exists():
            return str(cand)
    cand = Path(sys.executable).parent / "vllm"
    if cand.exists():
        return str(cand)
    found = shutil.which("vllm")
    return found or "vllm"

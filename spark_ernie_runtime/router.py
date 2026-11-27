"""Zero-dependency OpenAI-compatible load-balancing HTTP/SSE router for 2x2 parallelized DGX Sparks.

Routes `/v1/chat/completions`, `/v1/responses`, `/v1/completions`, and `/v1/embeddings`
across two independent `TP=2` ERNIE 5.5 replicas (`pair0` & `pair1`) using:
  1. Prefix-hash affinity (hashes system + first user message) when load is balanced,
     maximizing warm VRAM prefix-cache hit rate on multi-turn coding/agent sessions.
  2. Least-inflight spillover so concurrent bursts (`B=2..256`) saturate all 4 DGX Sparks
     with zero cross-pair RoCE traffic and 2x pooled KV cache capacity!
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import sys
import threading
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class BackendPool:
    def __init__(self, backends: list[str], api_key: str = "") -> None:
        self.backends = [b.rstrip("/") for b in backends]
        self.api_key = api_key
        self._lock = threading.Lock()
        self._inflight = [0 for _ in self.backends]
        self._rr = 0

    def pick(self, prefix_key: str | None = None) -> tuple[int, str]:
        with self._lock:
            n = len(self.backends)
            min_load = min(self._inflight)
            if prefix_key:
                preferred = int(hashlib.sha256(prefix_key.encode("utf-8", errors="ignore")).hexdigest()[:8], 16) % n
                # Stick to prefix-preferred replica when equally loaded (100% warm multi-turn cache hit),
                # but immediately spill concurrent bursts to any less-busy / idle replica!
                if self._inflight[preferred] == min_load:
                    self._inflight[preferred] += 1
                    return preferred, self.backends[preferred]
            # Otherwise pick least-inflight (round-robin tie-break)
            best_idx = 0
            for offset in range(n):
                idx = (self._rr + offset) % n
                if self._inflight[idx] == min_load:
                    best_idx = idx
                    break
            self._rr = (best_idx + 1) % n
            self._inflight[best_idx] += 1
            return best_idx, self.backends[best_idx]

    def release(self, idx: int) -> None:
        with self._lock:
            self._inflight[idx] = max(0, self._inflight[idx] - 1)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "backends": self.backends,
                "inflight": list(self._inflight),
            }


def _extract_prefix_key(body_bytes: bytes) -> str | None:
    if not body_bytes:
        return None
    try:
        obj = json.loads(body_bytes)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    parts: list[str] = []
    # Anthropic /v1/messages top-level `system` (string or list of text blocks)
    sys_val = obj.get("system")
    if isinstance(sys_val, str) and sys_val:
        parts.append(f"sys:{sys_val[:1024]}")
    elif isinstance(sys_val, list) and sys_val:
        stxt = "".join(
            str(b.get("text", "")) for b in sys_val if isinstance(b, dict) and b.get("type") == "text"
        )
        if stxt:
            parts.append(f"sys:{stxt[:1024]}")
    # OpenAI /v1/responses `instructions`
    inst = obj.get("instructions")
    if isinstance(inst, str) and inst:
        parts.append(f"inst:{inst[:1024]}")
    # Chat Completions & Anthropic Messages `messages`
    msgs = obj.get("messages")
    if isinstance(msgs, list) and msgs:
        for m in msgs[:2]:
            if isinstance(m, dict):
                c = m.get("content", "")
                if isinstance(c, list):
                    c = "".join(str(b.get("text", "")) for b in c if isinstance(b, dict))
                parts.append(f"{m.get('role','')}:{str(c)[:1024]}")
    if parts:
        return "|".join(parts)
    # OpenAI /v1/responses `input`
    inp = obj.get("input")
    if isinstance(inp, str) and inp:
        return inp[:1024]
    if isinstance(inp, list) and inp:
        return str(inp[:2])[:1024]
    return None


def make_handler(pool: BackendPool):
    class ProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _check_auth(self) -> bool:
            if not pool.api_key:
                return True
            auth = self.headers.get("Authorization", "")
            x_key = self.headers.get("x-api-key", "")
            if auth == f"Bearer {pool.api_key}" or x_key == pool.api_key:
                return True
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            payload = b'{"error":{"message":"Unauthorized","type":"auth_error"}}'
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return False

        def do_GET(self) -> None:
            if self.path == "/router/status":
                data = json.dumps(pool.snapshot()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            if not self._check_auth():
                return

            if self.path == "/health":
                for b in pool.backends:
                    try:
                        with urllib.request.urlopen(f"{b}/v1/models", timeout=3) as r:
                            if r.status != 200:
                                raise RuntimeError(f"status {r.status}")
                    except Exception as exc:
                        msg = json.dumps({"status": "degraded", "backend": b, "error": str(exc)}).encode()
                        self.send_response(503)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(msg)))
                        self.end_headers()
                        self.wfile.write(msg)
                        return
                ok = b'{"status":"ok","replicas":' + str(len(pool.backends)).encode() + b"}"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(ok)))
                self.end_headers()
                self.wfile.write(ok)
                return

            if self.path == "/metrics":
                parts = []
                for i, b in enumerate(pool.backends):
                    try:
                        req = urllib.request.Request(f"{b}/metrics")
                        if pool.api_key:
                            req.add_header("Authorization", f"Bearer {pool.api_key}")
                        with urllib.request.urlopen(req, timeout=4) as r:
                            parts.append(f"# === replica {i} ({b}) ===\n" + r.read().decode("utf-8", errors="replace"))
                    except Exception:
                        pass
                raw = "\n".join(parts).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return

            self._proxy_request("GET", b"")

        def do_POST(self) -> None:
            if not self._check_auth():
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b""
            self._proxy_request("POST", body)

        def _proxy_request(self, method: str, body: bytes) -> None:
            prefix_key = _extract_prefix_key(body) if method == "POST" else None
            idx, target = pool.pick(prefix_key)
            parsed = urllib.parse.urlsplit(target)
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=900)
            try:
                fwd_headers = {}
                for k, v in self.headers.items():
                    kl = k.lower()
                    if kl in ("host", "connection", "transfer-encoding"):
                        continue
                    fwd_headers[k] = v
                conn.request(method, self.path, body=body if method == "POST" else None, headers=fwd_headers)
                resp = conn.getresponse()
                ctype = resp.getheader("Content-Type", "application/json")
                is_sse = "text/event-stream" in ctype.lower()

                self.send_response(resp.status)
                self.send_header("Content-Type", ctype)
                self.send_header("X-Spark-Replica", str(idx))
                if is_sse:
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Accel-Buffering", "no")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    # Stream line-by-line (never resp.read(4096), which stalls SSE TTFT by ~1s!)
                    while True:
                        line = resp.readline()
                        if not line:
                            break
                        self.wfile.write(line)
                        self.wfile.flush()
                    self.close_connection = True
                else:
                    data = resp.read()
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except Exception as exc:
                err = json.dumps({"error": {"message": f"Upstream replica {idx} error: {exc}"}}).encode()
                try:
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                except Exception:
                    pass
                self.close_connection = True
            finally:
                conn.close()
                pool.release(idx)

    return ProxyHandler


def serve_router(host: str, port: int, backends: list[str], api_key: str = "") -> None:
    pool = BackendPool(backends, api_key=api_key)
    srv = ThreadingHTTPServer((host, port), make_handler(pool))
    srv.daemon_threads = True
    print(
        f"[spark-ernie router] Listening on http://{host}:{port}/v1 -> replicas={backends} "
        f"(prefix-affinity + least-inflight)",
        flush=True,
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="2x2 DGX Spark OpenAI prefix-aware load-balancing router")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backends", required=True, help="Comma-separated replica base URLs (e.g. http://h0:8001,http://h2:8002)")
    ap.add_argument("--api-key", default="")
    args = ap.parse_args(argv)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    serve_router(args.host, args.port, backends, api_key=args.api_key)
    return 0


if __name__ == "__main__":
    sys.exit(main())

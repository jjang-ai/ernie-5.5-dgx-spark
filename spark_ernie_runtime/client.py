"""OpenAI Chat Completions (/v1/chat/completions), Responses API (/v1/responses),
and Anthropic Messages API (/v1/messages) streaming client.

Defaults to `reasoning_effort="high"` (`enable_thinking=True`) and true token
streaming so users can verify reasoning, visible output, VRAM prefix caching,
and tok/s immediately with zero external dependencies (stdlib `urllib` only!).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def _headers(api_key: str) -> dict[str, str]:
    h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def stream_chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str = "",
    max_tokens: int = 8192,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int | None = None,
    reasoning_effort: str = "high",
    print_stream: bool = True,
    timeout: int = 600,
) -> dict[str, Any]:
    """Stream from `/v1/chat/completions` with ERNIE 5.5 official defaults (T=1.0, top_p=0.95, reasoning=high)."""
    enable_thinking = reasoning_effort.lower() not in ("off", "none", "false", "0")
    chat_kwargs: dict[str, Any] = {"enable_thinking": enable_thinking}
    if enable_thinking:
        chat_kwargs["reasoning_effort"] = reasoning_effort.lower()

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": chat_kwargs,
    }
    if top_k is not None:
        body["top_k"] = top_k
    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=_headers(api_key), method="POST")

    t0 = time.perf_counter()
    ttft: float | None = None
    in_thinking = False
    rsn_parts: list[str] = []
    ans_parts: list[str] = []
    usage: dict[str, Any] = {}

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("usage"):
                usage = ev["usage"]
            choices = ev.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            rc = delta.get("reasoning_content") or delta.get("reasoning") or ""
            c = delta.get("content") or ""
            if (rc or c) and ttft is None:
                ttft = time.perf_counter() - t0
            if rc:
                if print_stream and not in_thinking:
                    sys.stdout.write("\033[2m<thinking>\n")
                    in_thinking = True
                rsn_parts.append(rc)
                if print_stream:
                    sys.stdout.write(rc)
                    sys.stdout.flush()
            if c:
                if print_stream and in_thinking:
                    sys.stdout.write("\n</thinking>\033[0m\n\n")
                    in_thinking = False
                ans_parts.append(c)
                if print_stream:
                    sys.stdout.write(c)
                    sys.stdout.flush()

    if print_stream:
        if in_thinking:
            sys.stdout.write("\n</thinking>\033[0m")
        sys.stdout.write("\n")
        sys.stdout.flush()

    wall = time.perf_counter() - t0
    ptok = int(usage.get("prompt_tokens", 0))
    ctok = int(usage.get("completion_tokens", 0))
    rtok = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0))
    cached_tok = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
    decode_t = max(1e-4, wall - (ttft or 0.0))
    return {
        "api": "chat",
        "prompt_tokens": ptok,
        "cached_tokens": cached_tok,
        "completion_tokens": ctok,
        "reasoning_tokens": rtok,
        "ttft_s": round(ttft or wall, 4),
        "wall_s": round(wall, 4),
        "decode_tps": round(ctok / decode_t, 2),
        "e2e_tps": round(ctok / max(1e-4, wall), 2),
        "reasoning": "".join(rsn_parts),
        "content": "".join(ans_parts),
    }


def stream_responses_api(
    base_url: str,
    model: str,
    input_items: str | list[dict[str, Any]],
    *,
    api_key: str = "",
    max_output_tokens: int = 8192,
    temperature: float = 1.0,
    top_p: float = 0.95,
    reasoning_effort: str = "high",
    print_stream: bool = True,
    timeout: int = 600,
) -> dict[str, Any]:
    """Stream from OpenAI Responses API (`/v1/responses`) with ERNIE 5.5 official defaults (T=1.0, top_p=0.95, reasoning=high)."""
    body: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    if reasoning_effort.lower() not in ("off", "none", "false", "0"):
        body["reasoning"] = {"effort": reasoning_effort.lower()}
    else:
        body["reasoning"] = {"effort": "none"}

    url = base_url.rstrip("/") + "/responses"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=_headers(api_key), method="POST")

    t0 = time.perf_counter()
    ttft: float | None = None
    in_thinking = False
    rsn_parts: list[str] = []
    ans_parts: list[str] = []
    usage: dict[str, Any] = {}

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            etype = ev.get("type", "")
            if etype in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
                delta = ev.get("delta", "")
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    if print_stream and not in_thinking:
                        sys.stdout.write("\033[2m<thinking>\n")
                        in_thinking = True
                    rsn_parts.append(delta)
                    if print_stream:
                        sys.stdout.write(delta)
                        sys.stdout.flush()
            elif etype == "response.output_text.delta":
                delta = ev.get("delta", "")
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    if print_stream and in_thinking:
                        sys.stdout.write("\n</thinking>\033[0m\n\n")
                        in_thinking = False
                    ans_parts.append(delta)
                    if print_stream:
                        sys.stdout.write(delta)
                        sys.stdout.flush()
            elif etype == "response.completed":
                usage = (ev.get("response") or {}).get("usage") or {}

    if print_stream:
        if in_thinking:
            sys.stdout.write("\n</thinking>\033[0m")
        sys.stdout.write("\n")
        sys.stdout.flush()

    wall = time.perf_counter() - t0
    ptok = int(usage.get("input_tokens", 0))
    ctok = int(usage.get("output_tokens", 0))
    rtok = int((usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0))
    cached_tok = int((usage.get("input_tokens_details") or {}).get("cached_tokens", 0))
    decode_t = max(1e-4, wall - (ttft or 0.0))
    return {
        "api": "responses",
        "prompt_tokens": ptok,
        "cached_tokens": cached_tok,
        "completion_tokens": ctok,
        "reasoning_tokens": rtok,
        "ttft_s": round(ttft or wall, 4),
        "wall_s": round(wall, 4),
        "decode_tps": round(ctok / decode_t, 2),
        "e2e_tps": round(ctok / max(1e-4, wall), 2),
        "reasoning": "".join(rsn_parts),
        "content": "".join(ans_parts),
    }


def stream_anthropic_messages(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    system: str | list[dict[str, Any]] | None = None,
    api_key: str = "",
    max_tokens: int = 8192,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int | None = None,
    reasoning_effort: str = "high",
    print_stream: bool = True,
    timeout: int = 600,
) -> dict[str, Any]:
    """Stream from Anthropic Messages API (`/v1/messages`) with `x-api-key` & `cache_read_input_tokens`."""
    enable_thinking = reasoning_effort.lower() not in ("off", "none", "false", "0")
    anth_msgs: list[dict[str, Any]] = []
    sys_parts: list[str] = []
    if isinstance(system, str) and system:
        sys_parts.append(system)
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            if not anth_msgs:
                sys_parts.append(str(m.get("content", "")))
            else:
                # Preserve inline system reminders as user-role or system block without mutating leading prefix
                anth_msgs.append({"role": "user", "content": str(m.get("content", ""))})
            continue
        if role == "assistant" and m.get("reasoning_content"):
            blocks: list[dict[str, Any]] = [{"type": "thinking", "thinking": str(m["reasoning_content"])}]
            if m.get("content"):
                blocks.append({"type": "text", "text": str(m["content"])})
            anth_msgs.append({"role": "assistant", "content": blocks})
        else:
            anth_msgs.append({"role": role, "content": m.get("content", "")})

    body: dict[str, Any] = {
        "model": model,
        "messages": anth_msgs,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    if isinstance(system, list) and system:
        body["system"] = system
    elif sys_parts:
        body["system"] = "\n\n".join(sys_parts)
    if top_k is not None:
        body["top_k"] = top_k
    if enable_thinking:
        if max_tokens > 1024:
            body["thinking"] = {"type": "enabled", "budget_tokens": min(max_tokens - 1, max(1024, max_tokens // 2))}
        else:
            body["thinking"] = {"type": "adaptive"}
        body["output_config"] = {"effort": reasoning_effort.lower()}
    else:
        body["thinking"] = {"type": "disabled"}

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "anthropic-version": "2023-06-01",
    }
    if api_key:
        headers["x-api-key"] = api_key

    url = base_url.rstrip("/") + "/messages"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")

    t0 = time.perf_counter()
    ttft: float | None = None
    in_thinking = False
    rsn_parts: list[str] = []
    ans_parts: list[str] = []
    input_tokens = 0
    cache_read = 0
    cache_create = 0
    output_tokens = 0
    stop_reason = ""

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if not payload or payload == "[DONE]":
                continue
            ev = json.loads(payload)
            etype = ev.get("type", "")
            if etype == "message_start":
                u = (ev.get("message") or {}).get("usage") or {}
                input_tokens = int(u.get("input_tokens") or input_tokens)
                cache_read = int(u.get("cache_read_input_tokens") or cache_read)
                cache_create = int(u.get("cache_creation_input_tokens") or cache_create)
            elif etype == "content_block_delta":
                delta = ev.get("delta") or {}
                dtype = delta.get("type", "")
                if dtype == "thinking_delta":
                    txt = delta.get("thinking", "")
                    if txt:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        if print_stream and not in_thinking:
                            sys.stdout.write("\033[2m<thinking>\n")
                            in_thinking = True
                        rsn_parts.append(txt)
                        if print_stream:
                            sys.stdout.write(txt)
                            sys.stdout.flush()
                elif dtype == "text_delta":
                    txt = delta.get("text", "")
                    if txt:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        if print_stream and in_thinking:
                            sys.stdout.write("\n</thinking>\033[0m\n\n")
                            in_thinking = False
                        ans_parts.append(txt)
                        if print_stream:
                            sys.stdout.write(txt)
                            sys.stdout.flush()
            elif etype == "message_delta":
                u = ev.get("usage") or {}
                if u.get("output_tokens") is not None:
                    output_tokens = int(u["output_tokens"])
                if u.get("cache_read_input_tokens") is not None:
                    cache_read = int(u["cache_read_input_tokens"])
                if u.get("cache_creation_input_tokens") is not None:
                    cache_create = int(u["cache_creation_input_tokens"])
                if u.get("input_tokens") is not None:
                    input_tokens = int(u["input_tokens"])
                stop_reason = str((ev.get("delta") or {}).get("stop_reason") or stop_reason)

    if print_stream:
        if in_thinking:
            sys.stdout.write("\n</thinking>\033[0m")
        sys.stdout.write("\n")
        sys.stdout.flush()

    wall = time.perf_counter() - t0
    total_prompt = input_tokens + cache_read + cache_create
    decode_t = max(1e-4, wall - (ttft or 0.0))
    return {
        "api": "anthropic",
        "prompt_tokens": total_prompt,
        "input_tokens": input_tokens,
        "cached_tokens": cache_read,
        "cache_creation_tokens": cache_create,
        "completion_tokens": output_tokens,
        "reasoning_tokens": 0,
        "stop_reason": stop_reason,
        "ttft_s": round(ttft or wall, 4),
        "wall_s": round(wall, 4),
        "decode_tps": round(output_tokens / decode_t, 2),
        "e2e_tps": round(output_tokens / max(1e-4, wall), 2),
        "reasoning": "".join(rsn_parts),
        "content": "".join(ans_parts),
    }


def interactive_chat(
    base_url: str,
    model: str,
    *,
    api_key: str = "",
    api_mode: str = "chat",
    reasoning_effort: str = "high",
    max_tokens: int = 8192,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int | None = None,
    system_prompt: str = "",
) -> int:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    print(
        f"Connected to {base_url} | model={model} | api={api_mode} | "
        f"reasoning={reasoning_effort} | T={temperature} top_p={top_p}"
    )
    print("Type your message (or /effort high|medium|low|off, /api chat|responses|anthropic, /clear, /quit):\n")
    while True:
        try:
            user_in = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_in:
            continue
        if user_in in ("/quit", "/exit", "exit"):
            return 0
        if user_in == "/clear":
            messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
            print("(conversation cleared)\n")
            continue
        if user_in.startswith("/effort "):
            reasoning_effort = user_in.split(None, 1)[1].strip()
            print(f"(reasoning_effort -> {reasoning_effort})\n")
            continue
        if user_in.startswith("/api "):
            api_mode = user_in.split(None, 1)[1].strip()
            print(f"(api -> {api_mode})\n")
            continue

        messages.append({"role": "user", "content": user_in})
        try:
            if api_mode == "responses":
                res = stream_responses_api(
                    base_url, model, messages,
                    api_key=api_key, max_output_tokens=max_tokens,
                    temperature=temperature, top_p=top_p, reasoning_effort=reasoning_effort,
                )
            elif api_mode == "anthropic":
                res = stream_anthropic_messages(
                    base_url, model, messages,
                    api_key=api_key, max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p, top_k=top_k, reasoning_effort=reasoning_effort,
                )
            else:
                res = stream_chat_completion(
                    base_url, model, messages,
                    api_key=api_key, max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p, top_k=top_k, reasoning_effort=reasoning_effort,
                )
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            print(f"[HTTP {exc.code}] {err_body}\n", file=sys.stderr)
            messages.pop()
            continue
        except Exception as exc:
            print(f"[error] {exc}\n", file=sys.stderr)
            messages.pop()
            continue

        asst_msg: dict[str, Any] = {"role": "assistant", "content": res["content"]}
        if res.get("reasoning"):
            asst_msg["reasoning_content"] = res["reasoning"]
        messages.append(asst_msg)
        print(
            f"\033[2m[{res['api']} | in={res['prompt_tokens']} (cached={res['cached_tokens']}) "
            f"out={res['completion_tokens']} (think={res['reasoning_tokens']}) | "
            f"ttft={res['ttft_s']:.3f}s | decode={res['decode_tps']:.1f} tok/s | e2e={res['e2e_tps']:.1f} tok/s]\033[0m\n"
        )

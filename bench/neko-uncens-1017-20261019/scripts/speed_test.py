#!/usr/bin/env python3
"""Single-stream decode speed for the tensorfold endpoint (forge:8000).

3 prompts (short chat / ~2k code / ~8k prose), max_tokens 512, temperature 0.6.
Reports TTFT (stream) and decode tok/s from usage.completion_tokens / decode seconds.
"""
import json, time, urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"

CODE = "Write a Python module implementing an LRU cache with TTL, unit-tested style, with docstrings.\n" * 12
PROSE = ("Explain the history of the Byzantine empire in detail, covering administration, economy, "
         "military organisation, culture and religion. ") * 40
PROMPTS = [
    ("short_chat", "Explain in about 300 words why glaciers flow."),
    ("code_2k", CODE[:8000]),
    ("prose_8k", PROSE[:32000]),
]


def run(name, prompt, max_tokens=512):
    body = json.dumps({"model": "deepseek-v4.1-flash", "max_tokens": max_tokens,
                       "temperature": 0.6, "stream": True, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    tokens = 0
    usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta") or {}
                piece = d.get("content") or d.get("reasoning_content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    tokens += 1
    total = time.time() - t0
    ct = (usage or {}).get("completion_tokens") or tokens
    dec_s = max(total - (ttft or 0), 1e-6)
    print(json.dumps({"case": name, "ttft_s": round(ttft or 0, 3), "total_s": round(total, 2),
                      "completion_tokens": ct, "decode_tok_s": round(ct / dec_s, 2)}))


for name, prompt in PROMPTS:
    try:
        run(name, prompt)
    except Exception as exc:  # report, keep going
        print(json.dumps({"case": name, "error": str(exc)[:200]}))

#!/usr/bin/env python3
"""Perf-sweep harness v3 for the 1017 Zig TP=4 engine (forge:8000), 2026-10-19.

Protocol per config (one fresh boot):
  1. warmup: one greedy code chat (512 tok) + one salted 8k and 32k prefill (discarded)
  2. decode: 3 reps x 3 cases x {greedy low, greedy high} effort, max_tokens 512
     (deterministic text, +-0.2% spread; the temp-0.6 default-effort cells of the
     handoff baseline vary +-30% with the sampled text and are NOT usable for A/B)
  3. temp-0.6 single pass (continuity with the handoff baseline; noisy, secondary)
  4. prefill: 3 salted reps x {8k, 32k} tokens, max_tokens 1: prefill tok/s = prompt_tokens/TTFT
Appends JSON lines to perf-sweep/<ts>-<label>.jsonl.
"""
import argparse, json, os, random, time, urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"

CODE = "Write a Python module implementing an LRU cache with TTL, unit-tested style, with docstrings.\n" * 12
PROSE = ("Explain the history of the Byzantine empire in detail, covering administration, economy, "
         "military organisation, culture and religion. ")

DECODE_CASES = [
    ("short_chat", "Explain in about 300 words why glaciers flow."),
    ("code_2k", CODE[:8000]),
    ("prose_8k", (PROSE * 40)[:32000]),
]


def chat(prompt, max_tokens, temperature=0.0, effort=None):
    body = {"model": "deepseek-v4.1-flash", "max_tokens": max_tokens,
            "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}]}
    if effort:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    usage = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=900) as r:
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
                if (d.get("content") or d.get("reasoning_content") or "") and ttft is None:
                    ttft = time.time() - t0
                chunks += 1
    return {"ttft": ttft, "total": time.time() - t0, "usage": usage, "chunks": chunks}


def decode_rep(case, prompt, temperature, effort, out):
    r = chat(prompt, 512, temperature, effort)
    ct = (r["usage"] or {}).get("completion_tokens") or r["chunks"]
    dec_s = max(r["total"] - (r["ttft"] or 0), 1e-6)
    rec = {"kind": "decode", "case": case, "effort": effort or "default", "temp": temperature,
           "ttft_s": round(r["ttft"] or 0, 3), "completion_tokens": ct,
           "decode_tok_s": round(ct / dec_s, 2)}
    out(rec)
    return rec


def prefill_probe(size, out, record=True):
    salt = random.randint(10**8, 10**9 - 1)
    chars = 32000 if size == "8k" else 128000
    prompt = f"(context marker {salt}. Ignore this marker.) " + (PROSE * (chars // 90 + 2))[:chars]
    r = chat(prompt, 1)
    u = r["usage"] or {}
    pt = u.get("prompt_tokens")
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    rec = {"kind": "prefill", "case": f"prefill_{size}", "ttft_s": round(r["ttft"] or 0, 3),
           "prompt_tokens": pt, "cached_tokens": cached,
           "prefill_tok_s": round(pt / r["ttft"], 1) if pt and r["ttft"] else None}
    if record:
        out(rec)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", default=os.path.expanduser("~/neko-mmlu-bench/perf-sweep"))
    ap.add_argument("--decode-reps", type=int, default=3)
    ap.add_argument("--prefill-reps", type=int, default=3)
    ap.add_argument("--no-prefill", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{time.strftime('%Y%m%dT%H%M%S')}-{args.label}.jsonl")
    fh = open(path, "a")

    def out(rec):
        rec["label"] = args.label
        rec["ts"] = time.strftime("%H:%M:%S")
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print(json.dumps(rec))

    if not args.no_warmup:
        try:
            chat(f"(warm {random.randint(1,9)}) " + DECODE_CASES[1][1], 512, 0.0, "low")
        except Exception as e:
            print(json.dumps({"kind": "warmup_error", "error": str(e)[:200]}))
        for size in ("8k", "32k"):
            try:
                prefill_probe(size, lambda r: None, record=False)
            except Exception:
                pass

    for effort in ("low", "high"):
        for rep in range(args.decode_reps):
            for case, prompt in DECODE_CASES:
                try:
                    decode_rep(case, prompt, 0.0, effort, out)
                except Exception as e:
                    print(json.dumps({"kind": "decode_error", "case": case, "error": str(e)[:200]}))
            time.sleep(1)
    # temp 0.6 default-effort single pass (handoff-continuity cell; noisy)
    for case, prompt in DECODE_CASES:
        try:
            decode_rep(case, prompt, 0.6, None, out)
        except Exception as e:
            print(json.dumps({"kind": "decode_error", "case": case, "error": str(e)[:200]}))

    if not args.no_prefill:
        for size in ("8k", "32k"):
            for rep in range(args.prefill_reps):
                try:
                    prefill_probe(size, out)
                except Exception as e:
                    print(json.dumps({"kind": "prefill_error", "case": size, "error": str(e)[:200]}))
            time.sleep(1)

    fh.close()
    print(json.dumps({"done": args.label, "path": path}))


if __name__ == "__main__":
    main()

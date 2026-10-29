#!/usr/bin/env python3
"""HumanEval pass@1 (greedy) for the DeepSeek-V4.1-Flash-1017 packs on forge:8000.

Prompt construction is the canonical completion-format one and is byte-identical for every pack:

    sent   = "from typing import *\\n" + problem["prompt"]         (prompt ends with the docstring)
    stops  = ["\\nclass", "\\n@pytest", "\\nif __name__", "\\nprint(", "\\ndef "]
    temp   = 0, max_tokens = 512, raw /v1/completions (no chat template, no thinking)

    program = "from typing import *\\n" + problem["prompt"] + completion
              + "\\n" + problem["test"] + "\\ncheck(" + entry_point + ")\\n"
    pass    = the program exits 0

Sandbox (per run): fresh temp cwd, minimal env (no proxies inherited), `python3 -s -B`,
rlimits (CPU 20 s hard, AS 2 GiB, NPROC 64, FSIZE 2 MiB, NOFILE 256, CORE 0), output
redirected to files so a chatty completion cannot balloon memory, hard wall timeout.
Network is blocked in-process by sandbox_guard/sitecustomize.py, because this host denies
unprivileged user namespaces (`unshare -n -r` -> EPERM), so no kernel netns is available.

Results: results-<label>.json (per problem, with the exact program text) + summary-<label>.json.
"""
import argparse
import hashlib
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
GUARD = os.path.join(HERE, "sandbox_guard")
DATASET = os.path.join(HERE, "humaneval.jsonl")
STOP = ["\nclass", "\n@pytest", "\nif __name__", "\nprint(", "\ndef "]
API_DEFAULT = "http://localhost:8000/v1/completions"
TMPBASE = os.path.join(HERE, "run-tmp")


def sha256(s):
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def cut_stops(text):
    """Belt-and-braces: the server honours `stop`, but truncate client-side too."""
    idx = -1
    for s in STOP:
        i = text.find(s)
        if i >= 0 and (idx < 0 or i < idx):
            idx = i
    if idx >= 0:
        return text[:idx], True
    return text, False


def _apply_limits(cpu, as_gib, nproc, fsize_mib, nofile):
    def fn():
        lim = resource.setrlimit
        lim(resource.RLIMIT_CPU, (cpu, cpu))
        b = as_gib * 1024 * 1024 * 1024
        lim(resource.RLIMIT_AS, (b, b))
        lim(resource.RLIMIT_NPROC, (nproc, nproc))
        lim(resource.RLIMIT_FSIZE, (fsize_mib * 1024 * 1024,) * 2)
        lim(resource.RLIMIT_NOFILE, (nofile, nofile))
        lim(resource.RLIMIT_CORE, (0, 0))
    return fn


def classify(rc, stderr):
    if rc == 0:
        return "passed"
    if "network disabled by the HumanEval sandbox" in stderr:
        return "failed-network-attempt"
    if any(k in stderr for k in ("SyntaxError", "IndentationError", "TabError")):
        return "failed-syntax"
    if "AssertionError" in stderr:
        return "failed-assert"
    # last "SomeError: msg" line of the final traceback block
    for line in reversed([l for l in stderr.strip().splitlines() if l.strip()]):
        head = line.split(":", 1)[0].strip()
        if head.endswith(("Error", "Exception", "Interrupt")) and " " not in head:
            return "failed-" + head.lower()
    return "failed-other"


def run_program(program, timeout, cpu=20):
    os.makedirs(TMPBASE, exist_ok=True)
    d = tempfile.mkdtemp(prefix="he-", dir=TMPBASE)
    try:
        path = os.path.join(d, "main.py")
        with open(path, "w") as f:
            f.write(program)
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": d,
            "TMPDIR": d,
            "PYTHONPATH": GUARD,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
        out_path = os.path.join(d, "stdout.txt")
        err_path = os.path.join(d, "stderr.txt")
        t0 = time.time()
        timed_out = False
        rc = None
        with open(out_path, "wb") as fo, open(err_path, "wb") as fe:
            p = subprocess.Popen(
                [sys.executable, "-s", "-B", path],
                cwd=d, env=env, stdin=subprocess.DEVNULL,
                stdout=fo, stderr=fe, start_new_session=True,
                preexec_fn=_apply_limits(cpu, 2, 64, 2, 256),
            )
            try:
                rc = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    p.kill()
                try:
                    rc = p.wait(timeout=5)
                except Exception:
                    rc = None
        wall = time.time() - t0
        with open(out_path, "rb") as f:
            stdout = f.read(8192).decode("utf-8", "replace")
        with open(err_path, "rb") as f:
            stderr = f.read(8192).decode("utf-8", "replace")
        if timed_out:
            cat = "timeout"
        elif rc is not None and rc < 0:
            cat = {signal.SIGKILL: "killed-sigkill", signal.SIGXCPU: "killed-cpu",
                   signal.SIGSEGV: "killed-segv"}.get(-rc, "killed-signal%d" % -rc)
        else:
            cat = classify(rc, stderr)
        return {
            "category": cat, "returncode": rc, "wall_s": round(wall, 3),
            "stdout_head": stdout[:600], "stderr_tail": stderr[-1500:],
        }
    finally:
        shutil.rmtree(d, ignore_errors=True)


def call_model(session, api, model, prompt, max_tokens, timeout, retries=3):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0, "stop": STOP}
    last = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            r = session.post(api, json=body, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            m = j["choices"][0]
            return {
                "text": m.get("text") or "",
                "finish_reason": m.get("finish_reason"),
                "usage": j.get("usage"),
                "client_wall_s": round(time.time() - t0, 3),
                "tensorfold": j.get("tensorfold"),
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            last = "%s: %s" % (type(e).__name__, str(e)[:200])
            time.sleep(2 * (attempt + 1))
    return {"text": "", "finish_reason": None, "usage": None, "client_wall_s": None,
            "tensorfold": None, "error": last}


def engine_snapshot():
    out = {}
    for url in ("http://localhost:8000/v1/models", "http://localhost:8000/health"):
        try:
            out[url] = requests.get(url, timeout=10).json()
        except Exception as e:  # noqa: BLE001
            out[url] = {"error": str(e)[:200]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api", default=API_DEFAULT)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=20.0, help="hard per-program wall timeout")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--no-sandbox", action="store_true", help="(not used; documented for clarity)")
    args = ap.parse_args()

    with open(DATASET, "rb") as f:
        raw = f.read()
    problems = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    dataset_sha = sha256(raw)
    if args.limit:
        problems = problems[: args.limit]

    engine_start = engine_snapshot()
    session = requests.Session()
    rows = []
    t0 = time.time()
    for i, p in enumerate(problems):
        head = "from typing import *\n"
        sent = head + p["prompt"]
        res = call_model(session, args.api, args.model, sent, args.max_tokens, 600)
        completion_raw = res["text"]
        completion, cut = cut_stops(completion_raw) if completion_raw else ("", False)
        completion = completion.rstrip() + "\n" if completion.strip() else ""
        program = head + p["prompt"] + completion + "\n" + p["test"] + "\ncheck(" + p["entry_point"] + ")\n"
        if res["error"]:
            run = {"category": "request-error", "returncode": None, "wall_s": None,
                   "stdout_head": "", "stderr_tail": res["error"]}
        elif not completion.strip():
            run = {"category": "empty-completion", "returncode": None, "wall_s": None,
                   "stdout_head": "", "stderr_tail": ""}
        else:
            run = run_program(program, args.timeout)
        row = {
            "task_id": p["task_id"],
            "entry_point": p["entry_point"],
            "category": run["category"],
            "passed": run["category"] == "passed",
            "returncode": run["returncode"],
            "exec_wall_s": run["wall_s"],
            "finish_reason": res["finish_reason"],
            "completion_tokens": (res["usage"] or {}).get("completion_tokens"),
            "prompt_tokens": (res["usage"] or {}).get("prompt_tokens"),
            "api_wall_s": res["client_wall_s"],
            "stop_cut_client_side": cut,
            "completion": completion_raw[:6000],
            "program_sha256": sha256(program),
            "stdout_head": run["stdout_head"],
            "stderr_tail": run["stderr_tail"],
            "request_error": res["error"],
        }
        rows.append(row)
        if (i + 1) % 20 == 0 or i + 1 == len(problems):
            print("  %d/%d  pass=%d" % (i + 1, len(problems), sum(r["passed"] for r in rows)), flush=True)

    wall = time.time() - t0
    engine_end = engine_snapshot()
    breakdown = {}
    for r in rows:
        breakdown[r["category"]] = breakdown.get(r["category"], 0) + 1
    passed = sum(r["passed"] for r in rows)
    ctok = sum((r["completion_tokens"] or 0) for r in rows)
    summary = {
        "label": args.label,
        "config": {
            "model_requested": args.model,
            "api": args.api,
            "prompt_format": 'raw /v1/completions, "from typing import *\\n" + problem["prompt"]',
            "stop": STOP,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "decoding": "greedy, no chat template, no thinking",
            "output_processing": "truncate at first stop string, rstrip",
            "program_assembly": 'prompt+completion+"\\n"+test+"\\ncheck(entry_point)"',
            "exec_timeout_s": args.timeout,
            "cpu_rlimit_s": 20,
            "sandbox": "temp cwd; clean env; -s -B; RLIMIT_CPU20/AS2G/NPROC64/FSIZE2M/NOFILE256; "
                       "socket blocked via sitecustomize guard (no unprivileged netns: unshare -n -> EPERM)",
            "dataset": {"path": DATASET, "sha256": dataset_sha, "n": len(problems)},
            "wall_s": round(wall, 1),
        },
        "engine_start": engine_start,
        "engine_end": engine_end,
        "overall": {
            "n": len(rows),
            "passed": passed,
            "pass_at_1": round(100.0 * passed / len(rows), 2) if rows else None,
            "breakdown": breakdown,
            "completion_tokens_total": ctok,
            "api_wall_s_total": round(sum(r["api_wall_s"] or 0 for r in rows), 1),
            "completion_tok_per_s": round(ctok / max(1e-9, sum(r["api_wall_s"] or 0 for r in rows)), 2),
            "truncated_by_length": sum(1 for r in rows if r["finish_reason"] == "length"),
            "request_errors": sum(1 for r in rows if r["request_error"]),
        },
    }
    os.makedirs(args.outdir, exist_ok=True)
    raw_path = os.path.join(args.outdir, "results-%s.json" % args.label)
    sum_path = os.path.join(args.outdir, "summary-%s.json" % args.label)
    with open(raw_path, "w") as f:
        json.dump({**summary, "per_problem": rows}, f, indent=1)
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("label", "overall")}, indent=2))
    print("raw:", raw_path)
    print("summary:", sum_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

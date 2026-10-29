#!/usr/bin/env python3
"""HumanEval-164 pass@1, CHAT-harness variant of humaneval_bench.py (2026-10-19 model-matrix).

Why this exists: Venice has no /v1/completions (404). To keep ONE harness across all three
contestants, this variant drives /v1/chat/completions for every model with

    messages = [ {system: CONTINUATION_SYSTEM_PROMPT}, {user: "from typing import *\\n" + prompt} ]

and the exact same program assembly / sandbox / grading as humaneval_bench.py (imported,
not re-implemented). Nothing about the completion text is cleaned except the harness's own
first-stop truncation.

Deliberate deviations from the completions harness, all forced by the chat transport and
documented per run in the output config:
  * system prompt: the chat model otherwise answers conversationally (it restates the file
    or emits a fenced block). One fixed continuation instruction is used for every model.
  * stop strings: at most 4 on Venice, so 4 are sent to every model; the harness still
    truncates client-side on all 5 completions-harness stops.
  * reasoning_effort=none on every model (locals support it as a native switch; Venice
    defaults to thinking-on/high, which would emit reasoning instead of code).

--verify-dataset runs every canonical solution through the sandbox (no model call) to prove
the dataset/tests/sandbox are intact before any model numbers are believed.
"""
import argparse
import json
import os
import re
import sys
import tempfile
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import humaneval_bench as HB  # noqa: E402

CONTINUATION_SYSTEM_PROMPT = (
    "You are a code completion engine. The user message is the beginning of a Python file. "
    "Continue it with the remaining Python code only. Output no explanations, no commentary, "
    "no markdown fences, no repetition of the given code."
)
STOP_CHAT = ["\nclass", "\n@pytest", "\nif __name__", "\ndef "]


def verify_dataset(problems):
    rows = []
    for p in problems:
        head = "from typing import *\n"
        program = head + p["prompt"] + p["canonical_solution"].rstrip() + "\n" + p["test"] + "\ncheck(" + p["entry_point"] + ")\n"
        run = HB.run_program(program, 20.0)
        rows.append({"task_id": p["task_id"], "category": run["category"]})
    bad = [r for r in rows if r["category"] != "passed"]
    print(json.dumps({"canonical_solution_check": {"n": len(rows), "passed": len(rows) - len(bad),
                                                   "failures": bad[:5]}}, indent=1))
    return len(bad) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", choices=["local", "venice"], required=True)
    ap.add_argument("--api", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--api-key-env", default="VENICE_API_KEY")
    ap.add_argument("--api-key-file", default="")
    ap.add_argument("--norm", choices=["raw", "indent", "fence"], default="fence",
                    help="raw = nothing; indent = only first-line 4-space fix; fence = extract first "
                         "fenced code block + indent fix (default)")
    ap.add_argument("--verify-dataset", action="store_true")
    args = ap.parse_args()

    with open(HB.DATASET, "rb") as f:
        raw = f.read()
    problems = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    dataset_sha = HB.sha256(raw)
    if args.limit:
        problems = problems[: args.limit]

    if args.verify_dataset:
        ok = verify_dataset(problems)
        if not ok:
            raise SystemExit("dataset/sandbox verification FAILED")
        if args.limit == 0:
            raise SystemExit(0)

    headers = {"Content-Type": "application/json"}
    if args.engine == "venice":
        key = os.environ.get(args.api_key_env)
        if not key and args.api_key_file:
            key = open(args.api_key_file).read().strip()
        if not key:
            raise SystemExit("missing env %s" % args.api_key_env)
        headers["Authorization"] = "Bearer " + key

    session = requests.Session()
    rows = []
    t0 = time.time()
    for i, p in enumerate(problems):
        head = "from typing import *\n"
        sent = head + p["prompt"]
        body = {"model": args.model, "max_tokens": args.max_tokens, "temperature": 0,
                "reasoning_effort": "none", "stop": STOP_CHAT,
                "messages": [{"role": "system", "content": CONTINUATION_SYSTEM_PROMPT},
                             {"role": "user", "content": sent}]}
        if args.engine == "venice":
            body["venice_parameters"] = {"include_venice_system_prompt": False}
        res = {"text": "", "finish_reason": None, "usage": None, "client_wall_s": None,
               "error": None, "reasoning": ""}
        for attempt in range(4):
            try:
                tt = time.time()
                r = session.post(args.api, json=body, headers=headers, timeout=600)
                r.raise_for_status()
                j = r.json()
                m = j["choices"][0]["message"]
                res = {"text": m.get("content") or "", "finish_reason": j["choices"][0].get("finish_reason"),
                       "usage": j.get("usage"), "client_wall_s": round(time.time() - tt, 3),
                       "error": None, "reasoning": m.get("reasoning_content") or ""}
                break
            except Exception as e:  # noqa: BLE001
                res["error"] = "%s: %s" % (type(e).__name__, str(e)[:200])
                time.sleep(2 * (attempt + 1))
        completion_raw = res["text"]
        completion, cut = HB.cut_stops(completion_raw) if completion_raw else ("", False)
        completion = completion.rstrip() + "\n" if completion.strip() else ""
        indent_fixed = False
        fence_extracted = False
        if completion and args.norm != "raw":
            if args.norm == "fence":
                m = re.search(r"```[A-Za-z0-9_+.-]*\n(.*?)(?:```|\Z)", completion, re.S)
                if m:
                    completion = m.group(1)
                    fence_extracted = True
            completion = completion.lstrip("\n")
            if completion and completion[0] not in " \t" and not re.match(r"(def |class |async def |@)", completion):
                completion = "    " + completion
                indent_fixed = True
            completion = completion.rstrip() + "\n" if completion.strip() else ""
        program = head + p["prompt"] + completion + "\n" + p["test"] + "\ncheck(" + p["entry_point"] + ")\n"
        if res["error"]:
            run = {"category": "request-error", "returncode": None, "wall_s": None,
                   "stdout_head": "", "stderr_tail": res["error"]}
        elif not completion.strip():
            run = {"category": "empty-completion", "returncode": None, "wall_s": None,
                   "stdout_head": "", "stderr_tail": "raw:" + completion_raw[:300]}
        else:
            run = HB.run_program(program, args.timeout)
        rows.append({
            "task_id": p["task_id"], "entry_point": p["entry_point"],
            "category": run["category"], "passed": run["category"] == "passed",
            "returncode": run["returncode"], "exec_wall_s": run["wall_s"],
            "finish_reason": res["finish_reason"],
            "completion_tokens": (res["usage"] or {}).get("completion_tokens"),
            "prompt_tokens": (res["usage"] or {}).get("prompt_tokens"),
            "reasoning_tokens": ((res["usage"] or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "api_wall_s": res["client_wall_s"], "stop_cut_client_side": cut,
            "completion": completion_raw[:6000], "program_sha256": HB.sha256(program),
            "first_line_indent_fixed": indent_fixed, "fence_extracted": fence_extracted,
            "stdout_head": run["stdout_head"], "stderr_tail": run["stderr_tail"],
            "request_error": res["error"],
        })
        if (i + 1) % 10 == 0 or i + 1 == len(problems):
            print("  %d/%d  pass=%d" % (i + 1, len(problems), sum(r["passed"] for r in rows)), flush=True)

    wall = time.time() - t0
    breakdown = {}
    for r in rows:
        breakdown[r["category"]] = breakdown.get(r["category"], 0) + 1
    passed = sum(r["passed"] for r in rows)
    ctok = sum((r["completion_tokens"] or 0) for r in rows)
    summary = {
        "label": args.label,
        "config": {
            "harness": "chat",
            "model_requested": args.model,
            "engine": args.engine,
            "api": args.api,
            "endpoint": "/v1/chat/completions",
            "system_prompt": CONTINUATION_SYSTEM_PROMPT,
            "user_message": 'from typing import *\\n + problem["prompt"]',
            "stop_sent": STOP_CHAT,
            "stop_client_side_cut": HB.STOP,
            "temperature": 0, "max_tokens": args.max_tokens, "reasoning_effort": "none",
            "venice_parameters": ({"include_venice_system_prompt": False} if args.engine == "venice" else None),
            "output_processing": "truncate at first stop string (all 5), rstrip",
            "normalization": {"mode": args.norm,
                              "fence": "extract the first fenced code block if present",
                              "indent": "prefix 4 spaces when the first code line is unindented and is not a "
                                        "def/class/@ line (chat transport drops the prompt's body anchor)",
                              "applied": "identically to every model"},
            "program_assembly": 'prompt+completion+"\\n"+test+"\\ncheck(entry_point)"',
            "exec_timeout_s": args.timeout, "cpu_rlimit_s": 20,
            "sandbox": "humaneval_bench.run_program verbatim: temp cwd; clean env; -s -B; "
                       "RLIMIT CPU20/AS2G/NPROC64/FSIZE2M/NOFILE256; socket guard sitecustomize",
            "dataset": {"path": HB.DATASET, "sha256": dataset_sha, "n": len(problems)},
            "wall_s": round(wall, 1),
        },
        "overall": {
            "n": len(rows), "passed": passed,
            "pass_at_1": round(100.0 * passed / len(rows), 2) if rows else None,
            "breakdown": breakdown, "completion_tokens_total": ctok,
            "api_wall_s_total": round(sum(r["api_wall_s"] or 0 for r in rows), 1),
            "completion_tok_per_s": round(ctok / max(1e-9, sum(r["api_wall_s"] or 0 for r in rows)), 2),
            "truncated_by_length": sum(1 for r in rows if r["finish_reason"] == "length"),
            "request_errors": sum(1 for r in rows if r["request_error"]),
        },
    }
    with open(args.out, "w") as f:
        json.dump({**summary, "per_problem": rows}, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("label", "overall")}, indent=2))
    print("raw:", args.out)


if __name__ == "__main__":
    main()

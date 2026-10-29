# Task: measure coding capability of the 2.9 bpw and 3.5 bpw uncensored packs (HumanEval pass@1)

Status: new. Orchestrator: Depths. Jun 2026-10-19 19:41: "but how about coding capabilities?"

We have MMLU for both uncensored packs and for the censored base, but **no direct coding measurement at all**.
MMLU STEM is a weak proxy (it's exam recall, not code synthesis). This task fills that gap with one honest,
comparable number per pack: HumanEval pass@1, same harness, greedy, sandboxed execution.

## Current state of the hardware

- The live engine on forge/anvil/ember/flame currently serves the **2.9 bpw** uncensored pack
  (`DeepSeek-V4.1-Flash-1017-UNCENSORED-TF`), image `dsv41-tensorfold:tf110-20261019`, 4 slots × 524288 ctx.
- The **3.5 bpw** pack exists: `/home/jun/models/neko-legends-DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-UNCENSORED-4xSpark`
  with its own world config `~/tf4/zig-1017-35bpw.env` per node and `~/tf4/serve4-zig-1017-35bpw.sh`. It cannot run
  the prefill pipeline (prepared 67.3 GiB/rank vs 55.1, plus stage copies), so it must be booted with the pipeline
  off, exactly as the previous worker did for its speed bench (see `~/neko-mmlu-bench/perf-35bpw-iso.log` and
  `perf-sweep/20261019T190708-35bpw-iso.jsonl`).
- **A previous worker may still be finishing a verification sweep.** Before you touch anything, check
  `systemctl --user is-active neko-35bpw-r3.service` on the host and `curl -s localhost:8000/health` on forge.
  If the engine is busy or `~/tf4/maintenance` exists and is fresh, wait (poll every 60 s, up to 30 min) until it is
  quiet. Do not fight it for the GPU.
- Engine control: explicit `stop` then `start` (a `start` while the old engine still serves is refused after 90 s).
  `touch ~/tf4/maintenance` before each restart; remove it when you are done. `.env` files are per node — never scp
  one env file across nodes (they carry rank-specific `ASSETS_DIR`).

## What to measure

1. **HumanEval (164 problems), pass@1, greedy (temperature 0)**, identical prompt construction for every pack:
   the canonical `HUMANEVAL_PROMPT` (complete-the-function body, stop at `\nclass`/`\nif __name__`/`\ndef ` etc.),
   then execute the official test against the model's completion.
   - Sandbox the execution: subprocess with a hard 20 s timeout, isolated temp dir, no network (unset proxies,
     run with `-I` / clean env), and a cap on output size. Generated code is our own model's, but treat it as
     untrusted anyway — a wild `rm -rf` or fork bomb must not damage the host. Report any execution that had to be
     killed rather than silently counting it as a fail.
   - Record per-problem: passed, failed-assert, failed-syntax, timeout, empty completion. Report **pass@1** and the
     failure breakdown separately — the breakdown is often more informative than the headline.
2. **A small agentic coding smoke test** (because this pack's future is as a worker model): 3-5 tasks of the shape
   "write a function and a unit test, then fix the failing case", checked by running the produced test file. Report
   pass/fail per task with the failure reason. This is not a benchmark, it's a sanity check that it can do multi-step
   code work, not just one-shot function bodies.
3. **Run the same harness against all three models where the engine allows**: 2.9 uncensored (live now), 3.5
   uncensored (its own world), and the censored base (`~/tf4/serve4-zig-1017.sh`, same prompt/harness) as the
   reference point. If time or memory forces a choice, do 2.9 and 3.5 first — the censored base is the nice-to-have.

## Reporting

Compare against what we already know so the number means something:
- MMLU-500: 2.9 uncensored 81.20%, 3.5 uncensored 81.60%, censored base 86.20%.
- Decode (per stream, temp 0, effort low/high): 2.9 = 68.2/125.8/62.6 and 70.7/89.4/61.7; 3.5 = 58.7/101.0/62.5.
  Note `code_2k` is the FASTEST decode case on this engine (draft acceptance is high on code), which matters for the
  "should this pack back a coding worker" question.
- Save raw per-problem results as JSON next to the summary, so a later run can be diffed problem by problem.

## Hard constraints

- You own the engine only for the bench windows; always end with the **2.9 world live and healthy**
  (`~/tf4/serve4-zig-1017-uncens.sh start`), `~/tf4/maintenance` removed, and all four ranks verified.
- Never end with the endpoint down. One failed boot is fine, then revert.
- No HF upload, no git push, no changes to eva-core/nest, don't touch `zig-world`, `keeper.sh`, or `serve4.sh` logic.
- Don't fetch datasets from the network at random: HumanEval is small and public — vendor it to
  `~/neko-code-bench/humaneval.jsonl` with a recorded hash so runs are reproducible.
- Do not delete packs or prepared dirs to make space without checking `df -h` first (forge is at ~90%+).

## Deliverable

`/home/jun/nest/tmp/worker-handoffs/code-bench-20261019.REPORT.md`: harness description (prompt, stop strings, sandbox),
per-model pass@1 table with the failure breakdown, the agentic smoke results, the raw data paths, what is left
running, the revert line, and `## Beyond-scope findings`. If a pack cannot be benched for a structural reason, say so
with the evidence instead of substituting a different test.

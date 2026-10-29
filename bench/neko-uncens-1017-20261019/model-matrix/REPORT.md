# status: success

# Three-way model matrix — LOCAL 2.9bpw vs LOCAL 3.5bpw vs VENICE-hosted 1017

Handoff: `nest/tmp/worker-handoffs/model-matrix-20261019.md` (Jun, 2026-10-19 19:44, orchestrator Depths).
Worker: pi builder, `venice/deepseek-v4-1-flash-1017`. Ran 19:45–20:47 PDT.

Deliverable: one matrix, three model columns × four measurement groups, harness label on every cell.
Two harness families are used because **Venice has no `/v1/completions`** (404, verified) and its
`/v1/chat/completions` needs a different prompt transport. Cells from different families are never
mixed inside one column: every family has its own labelled column set.

| column | harness | who could be measured |
|---|---|---|
| **A · completions** | raw `/v1/completions`, no chat template, thinking off (raw generate) | LOCAL-2.9, LOCAL-3.5 only |
| **B · chat** | `/v1/chat/completions`, effort `none`, fixed prompts, documented normalisation | LOCAL-2.9, LOCAL-3.5, VENICE-1017 |

Sampling is temperature 0 / greedy in every cell, all three models, all four test groups.

---

## 1. Matrix

### 1a. MMLU-500 — 5-shot, greedy, same 500 questions (`question_set_sha256 86c94624…0ae939`)

| | **LOCAL-2.9** (live pack) | **LOCAL-3.5** | **VENICE-1017** |
|---|---|---|---|
| **Harness A · completions** (raw `/v1/completions`, 5-shot, max_tokens 8, temp 0) | **81.2 %** (406/500) · 0 invalid | **81.6 %** (408/500) · 0 invalid | *not measurable — Venice `/v1/completions` = 404* |
| **Harness B · chat** (`/v1/chat/completions`, fixed system prompt, effort `none`, max_tokens 8, temp 0) | **79.6 %** (398/500) · 30 invalid | **81.0 %** (405/500) · 22 invalid | **86.6 %** (433/500) · 10 invalid |
| stem (n=171) | A 80.70 · B 77.19 | A 82.46 · B 77.19 | B 83.63 |
| humanities (n=113) | A 71.68 · B 68.14 | A 71.68 · B 72.57 | B 84.96 |
| social (n=103) | A 89.32 · B 87.38 | A 85.44 · B 87.38 | B 92.23 |
| other (n=113) | A 84.07 · B 87.61 | A 86.73 · B 89.38 | B 87.61 |
| **`moral_scenarios` (n=22) — broken out** | **A 40.91 % (9/22) · B 22.73 % (5/22)** | **A 45.45 % (10/22) · B 22.73 % (5/22)** | **B 95.45 % (21/22)** |

`invalid` = no `[ABCD]` anywhere in the generated text (counted wrong, same rule in both harnesses).
Harness A never produced an invalid answer; harness B does, because the chat model sometimes starts
explaining ("Let's go through each question…") and hits the 8-token cap. Invalid answers by category
(harness B): 2.9 = stem 23 / humanities 1 / social 3 / other 3; 3.5 = stem 20 / other 2;
Venice = stem 10.

**The moral_scenarios line needed a distribution check, and it is the biggest single finding:**
both local packs answer **"D" to all 22 of 22** moral_scenarios items under harness B, i.e. the 22.73 %
is exactly the count of items whose gold answer is D. Under harness A (raw continuation) they spread
across A/B/C/D (2.9: D 15, A 3, B 2, C 2 → 40.91 %). With thinking on (`effort low`, 300 tokens) 2.9
does spread out again (probe below), so the collapse is specific to **chat template + thinking-off**,
not to the questions. Venice answers 21/22 correctly under the same harness, so the questions and the
parse are fine.

Context row (not a contestant, 4th pack, same engine/harness A, run 01:54–01:59 today): the **censored**
local 1017 (`DeepSeek-V4.1-Flash-1017-TF`) scores **86.2 %** overall and **95.45 %** on moral_scenarios —
the abliteration cost lives exactly where the handoff said it does.

### 1b. HumanEval-164 pass@1 — canonical prompt, sandboxed official tests

Dataset: `/home/jun/neko-code-bench/humaneval.jsonl`, sha256 `1d49078b…09a6a2`, 164 problems;
integrity check: **all 164 canonical solutions pass their own tests in the sandbox (164/164)** before
any model number was trusted. Execution sandbox = the existing harness verbatim (fresh temp cwd, clean
env, `python3 -s -B`, RLIMIT CPU 20 s / AS 2 GiB / NPROC 64 / FSIZE 2 MiB, sockets blocked by a
`sitecustomize` guard, hard 20 s wall, output captured to files).

| | **LOCAL-2.9** | **LOCAL-3.5** | **VENICE-1017** |
|---|---|---|---|
| **A · completions** (raw `/v1/completions`, canonical stops ×5, max_tokens 512, temp 0) | **76.83 %** (126/164) | **76.22 %** (125/164) | *n/a (404)* |
| **B · chat** (chat transport, fixed continuation system prompt, effort `none`, stops ×4 + client cut, fence extraction + first-line indent normalisation, max_tokens 512) | **92.68 %** (152/164) | **93.29 %** (153/164) | **90.85 %** (149/164) |

Failure breakdown (same categories, same classifier):

| category | 2.9 · A | 3.5 · A | 2.9 · B | 3.5 · B | Venice · B |
|---|---|---|---|---|---|
| passed | 126 | 125 | 152 | 153 | 149 |
| assert | 30 | 33 | 6 | 10 | 14 |
| syntax | 1 | 3 | 3 | 0 | 0 |
| timeout (20 s) | 0 | 0 | 0 | 0 | 0 |
| empty completion | 0 | 0 | 1 | 0 | 0 |
| other runtime (index/value/name/type) | 7 | 3 | 2 | 1 | 1 |
| **pass@1** | **76.83 %** | **76.22 %** | **92.68 %** | **93.29 %** | **90.85 %** |

Neither local run hit the timeout class; the failures are wrong algorithms (assert) and Python runtime
errors, not sandbox kills.

**Harness-B normalisation matters more than the models do, so it is measured, not hidden.** Same
completions, three normalisation levels (identical rule for every model):

| normalisation (harness B) | LOCAL-2.9 | VENICE-1017 |
|---|---|---|
| raw — nothing (syntax fails: 117 / 103) | 26.83 % (44/164) | 32.32 % (53/164) |
| indent-only — prefix 4 spaces when the first code line is unindented | 72.56 % (119/164) | 89.63 % (147/164) |
| **fence + indent (reported above)** | **92.68 % (152/164)** | **90.85 % (149/164)** |

Cause, verified by reading the completions: a chat transport drops the prompt's in-body anchor, so the
generated body starts at column 0 while its own later lines stay at 4 spaces (`IndentationError`), and
~20 % of answers arrive wrapped in a ```python fence. The normalisation is a transport repair applied
identically to both models; the reported 3.5 figure uses the same rule (its own raw/indent-only levels
were not re-measured — a second 3.5 boot was not worth one table row).

### 1c. Decode — 3 prompts, `max_tokens` 512, temp 0, 3 reps, medians (tok/s) [min–max]

Harness C: the same chat request shape for all three (`perf_sweep.py` protocol v3). Local = 4×Spark
TP4 engine in its live boot. Venice = hosted, `include_venice_system_prompt:false`, streaming, ttft
excluded; **hosted decode depends on their batching/queueing and is not a hardware comparison.**

| case · effort | **LOCAL-2.9** (pipe on, live) | **LOCAL-3.5** (pipe off) | **VENICE-1017** (hosted) |
|---|---|---|---|
| short_chat · low | 69.3 [67.2–69.6] | 62.2 [61.0–62.4] | 67.6 [65.0–80.1] |
| short_chat · high | 71.7 [68.7–72.2] | 60.5 [59.1–60.6] | 95.0 [59.6–109.7] |
| code_2k · low | 127.7 [121.2–127.9] | 116.0 [113.0–116.7] | 398.8 [304.3–399.7] |
| code_2k · high | 91.0 [88.5–91.2] | 101.0 [98.3–101.7] | 414.9 [357.2–415.9] |
| prose_8k · low | 63.9 [60.6–64.0] | 56.5 [55.1–56.5] | 206.7 [152.6–247.1] |
| prose_8k · high | 63.0 [59.5–63.1] | 55.3 [53.6–55.4] | 242.5 [225.0–293.3] |
| temp-0.6 default pass (continuity only, noisy) | 62.9 / 86.2 / 57.3 | 59.0 / 100.3 / 55.7 | not run (not in protocol set) |

Notes that belong with the numbers: the 2.9 pipeline being on/off does **not** move decode (measured
pipeline-off at 19:20 today: 69.2 / 127.5 / 63.8 low, 71.8 / 91.0 / 62.9 high — within ±1 %). Venice's
`completion_tokens` include reasoning tokens at `effort low/high`, and their per-case spread is 1.3–1.5×
(their batching), so only the medians are meaningful.

### 1d. Prefill — salted prompts, `cached_tokens: 0` verified where the API reports it, 3 reps, medians

| prompt | **LOCAL-2.9** pipe **on** (live config) | **LOCAL-2.9** pipe **off** | **LOCAL-3.5** pipe **off** | **VENICE-1017** *implied* |
|---|---|---|---|---|
| 5,296 tok | **2,908 tok/s** · TTFT 1.821 s | 2,482 tok/s · 2.134 s | 2,426 tok/s · 2.183 s | *6,594 tok/s implied* · TTFT 0.803 s |
| 21,058 tok | **5,033 tok/s** · TTFT 4.184 s | 2,861 tok/s · 7.360 s | 2,780 tok/s · 7.576 s | *15,538 tok/s implied* · TTFT 1.355 s |
| cache verified | `cached_tokens: 0` on all 6 reps | `0` on all 6 | `0` on all 6 | **`cached_tokens` null — cannot be verified** |

**Read 1d carefully.** The 3.5 pack cannot run the prefill pipeline (prepared images 67.3 GiB/rank vs
55.1, stage copies 33 vs 25 GiB), so its prefill is only ever pipeline-off. The apples-to-apples
prefill comparison is therefore **2.9-pipe0 (2,861) vs 3.5-pipe0 (2,780) at 21k = −2.8 %**, and the
2.9 live number (5,033) is +76 % over both purely because of the pipeline the 3.5 pack cannot afford.
The Venice column is an **implied** rate (prompt_tokens ÷ end-to-end TTFT from a hosted stream:
includes their queue, network and any internal caching we cannot see) and is marked as such — it is
**not** in the same column as a measured local number and must not be read as one.

---

## 2. What this comparison can and cannot tell you

**Can:**
- On **code generation**, the 3.5bpw pack is not a quality upgrade over 2.9bpw: 76.22 % vs 76.83 %
  (canonical), 93.29 % vs 92.68 % (chat transport) — both inside what one 164-problem run can resolve.
  The 3.5 pack is slightly **slower** decoded (62/116/56 vs 69/128/64 low-effort) and, with the
  pipeline off, −2.8 % prefill at 21k.
- On **MMLU**, the two local packs are a tie (81.2 vs 81.6 canonical; 79.6 vs 81.0 chat). 3.5 recovers
  +4.5 pts of `moral_scenarios` under the canonical harness (40.9 → 45.5 %), which is not a quality
  claim you can build on: it is still ~half the censored model's 95.5 %.
- The **abliteration cost is real and localised**: humanities/moral, not STEM. Both packs sit ~50 pts
  below the hosted model on `moral_scenarios` under every harness (95.45 % vs 40.91 / 45.45 / 22.73 %),
  and 1–6 pts below on STEM.
- **Hosted 1017 beats both local packs** on every capability measure that can be run on both
  (MMLU chat 86.6 vs 79.6/81.0; HumanEval chat 90.85 vs 92.68/93.29 — the one measure where local
  ≥ hosted, and the gap is inside noise). Hosted is also 3–4.5× faster at decode on the long prompts
  and 2.3–3.1× faster at prefill (implied). What it is not: your weights, your data path, or uncensored.

**Cannot:**
- **No completions-harness number exists for Venice** (404), so Venice never appears in column A. The
  chat harness is the only same-harness comparison across all three, and it is a *chat* harness: the
  system prompt, the chat template and the 4-stop cap are all part of it.
- **MMLU column A vs column B are not the same test** even though the questions are identical: A sends
  a raw 5-shot continuation and always answers with a letter; B sends the same text as a chat message
  with a fixed instruction and sometimes answers with prose (30/22/10 invalid). Do not compare across
  the two columns; compare within a column.
- **The moral_scenarios collapse in column B is a serving-mode interaction**, not a stable capability
  number for the packs: with `reasoning_effort` low it spreads again (probe). If eva-core ever runs the
  local packs with thinking off behind a chat template, that is the mode that answers "D" to every
  moral item.
- **Hosted decode/prefill are not hardware measurements.** They depend on Venice's queue, batching and
  their prompt cache (which we could not even verify was cold). The local numbers are single-stream,
  single-tenant, fresh-boot numbers on our four Sparks.
- **The 3.5 numbers come from one boot with the pipeline off**, and its HumanEval was measured once
  (164 problems ≈ ±5–6 pts at 95 % CI). 76.22 vs 76.83 is a tie; do not read a winner.
- **Quantisation is not isolated here.** 2.9 vs 3.5 bpw differ in pack, prepared assets and (forced)
  pipeline setting. The −2.8 % prefill and −10 % decode could be the pack or the pipeline-off
  configuration; this run cannot separate them.

---

## 3. Raw per-problem JSON (diffable problem-by-problem)

All on `forge` (`/home/jun/…`):

**MMLU-500** (`~/neko-mmlu-bench/`, `details[]` = per-question pred/gold/raw/usage)
- 2.9 · harness A: `results-uncens.json` (run 01:51 today, 2.9 uncensored world)
- 3.5 · harness A: `results-35bpw.json` (run 19:06 today, 35bpw world)
- 2.9 · harness B: `results-mmlu-chat-29bpw.json`
- 3.5 · harness B: `results-mmlu-chat-35bpw.json`
- Venice · harness B: `results-mmlu-chat-venice.json`
- context, censored local · harness A: `results-cens.json`

**HumanEval-164** (`~/neko-code-bench/`, `per_problem[]` = completion text, program sha256, category, stderr)
- 2.9 · A: `results-29bpw-comp.json` (+ `summary-29bpw-comp.json`)
- 3.5 · A: `results-35bpw-comp.json` (+ `summary-35bpw-comp.json`)
- 2.9 · B: `results-he-chat-fence-29bpw.json` (indent-only: `results-he-chat-29bpw.json`; 20-problem check: `results-he-chat29-20.json`)
- 3.5 · B: `results-he-chat-fence-35bpw.json`
- Venice · B: `results-he-chat-fence-venice.json` (indent-only: `results-he-chat-venice.json`; raw: `results-he-chat-raw-venice.json`)

**Decode / prefill** (`~/neko-mmlu-bench/perf-sweep/`)
- 2.9 live (pipe on): `20261019T201526-29bpw-matrix.jsonl`
- 2.9 pipe off: `20261019T192023-29bpw-pipe0.jsonl` (earlier worker, same protocol)
- 3.5 (pipeline off): `20261019T203127-35bpw-matrix.jsonl`
- Venice (implied): `venice-1017.jsonl`
- human summary of the earlier perf campaign: `~/neko-mmlu-bench/perf-sweep.md`

New harness code written for this run (additive, nothing existing modified):
- `~/neko-mmlu-bench/bench_mmlu_chat.py` — chat-harness MMLU, asserts the same question-set sha256
- `~/neko-code-bench/humaneval_bench_chat.py` — chat-harness HumanEval, imports the existing sandbox/classifier
- `~/neko-mmlu-bench/venice_perf.py` — hosted decode/TTFT probe, protocol v3 shapes

---

## 4. What changed / where / verification

Hardware: 2.9 world was live at start; 35bpw booted 20:19–20:35 (touch `~/tf4/maintenance`, explicit
`stop` then `start`, pipeline off as that pack requires); 2.9 world restored 20:39–20:41 and left live
and healthy, all four ranks, `~/tf4/maintenance` removed.

check: `ssh forge 'curl -s localhost:8000/health'` → `{"ok": true, "backend": "tensorfold", "context_length": 524288, "requests_total": 2, "errors": 0}`
check: `ssh forge 'curl -s localhost:8000/v1/models'` → `["DeepSeek-V4.1-Flash-1017-UNCENSORED-TF", "deepseek-v4.1-flash"]`
check: `docker inspect -f '{{.State.Running}}' dsv41-tf4-r0` + same on 192.168.10.2/.3/.4 → `r0=true r1=true r2=true r3=true`
check: `ssh forge 'ls ~/tf4/maintenance'` → `No such file or directory`
check: `ssh forge 'curl -s http://127.0.0.1:8000/v1/chat/completions -d {…"Reply with exactly: matrix-live-ok"…}'` → `"message": {"role": "assistant", "content": "matrix-live-ok"}`
check: `/home/jun/sglang-venv/bin/python3 -m json.tool`-read of the four MMLU result files → all report `question_set_sha256` prefix `86c9462464c1bbf2` and match the handoff hash `86c94624…0ae939`
check: `humaneval_bench_chat.py --verify-dataset` → `{"n": 164, "passed": 164, "failures": []}` (canonical solutions through the sandbox)
check: `curl -s https://api.venice.ai/api/v1/completions -d '{"model":"deepseek-v4-1-flash-1017",…}'` → `HTTP 404 {"error":"Not found"}` (why Venice is chat-only)
check: Venice `stop` with 5 strings → `HTTP 400 {"issues":[{"path":["stop"],"message":"Array must contain at most 4 element(s)"}]}`
check: Venice one-line message with/without `include_venice_system_prompt` → `prompt_tokens 1733` vs `9`
check: Venice `reasoning_effort:"none"` → `{"reasoning_tokens": 0}` and content returned directly (thinking is ON by default: `defaultReasoningEffort: "high"` in the live catalog)
check: local engine `reasoning_effort:"none"` → content `'The capital of France is **Paris**.'`, `reasoning_content: ''` (parity with Venice's `none`)
check: local 2.9 moral probe, 3 alternative system prompt phrasings × 5 items → `[('D','C'),('D','D'),('D','A'),('D','D'),('D','A')]` for all three phrasings (the all-D collapse is robust, not a prompt accident)
check: local 2.9 moral probe with `effort low` (300 tok) → gold C→`''` (300 tok, truncated), gold D→`D`, gold A→`C`, gold D→`D` (thinking on restores variation)
check: `perf_sweep.py --label 29bpw-matrix` output → prefill `cached_tokens: 0` on all 6 reps, 21k median 5,032.7 tok/s
check: `venice_perf.py` output → prefill TTFT 1.355 s at 21,058 tok, `cached_tokens: null` (reported as unverifiable)
check: evidence files listed in §3 exist with today's mtimes (`ls -l --time-style=full-iso`) → `results-mmlu-chat-35bpw.json 20:26:36`, `results-he-chat-fence-35bpw.json 20:28:59`, `summary-35bpw-comp.json 20:31:27`, `perf-35bpw-matrix.log 20:35:03`
check: `ssh forge 'ls /home/jun/.venice_key'` → `No such file or directory` (the temporary key file used for the hosted runs was deleted after the last Venice call)

Spend: 1,016 hosted requests, 503,437 input + 36,864 output tokens → **≈ $0.20 at list price**
($0.30/M in, $1.20/M out; cache discounts ignored). No exploratory chatter was sent to Venice; every
hosted call belongs to a matrix cell or to the Venice perf probe.

## 5. Risks / follow-ups

- **The chat-harness HumanEval number is not portable.** 26.83 → 92.68 % depending on normalisation for
  the same model and same completions. Any future cross-model HumanEval must fix one normalisation rule
  in writing before running (this run's rule is in `humaneval_bench_chat.py --norm fence` and recorded
  in every result file's `config.normalization`).
- Venetian `stop` ≤ 4 strings means the canonical 5-stop HumanEval list can never be sent verbatim to a
  chat-only hosted model; the 5th is client-side only (`stop_cut_client_side` is 0 in every one of the 1,004 chat-harness problem rows produced today,
  so the 5th stop never actually fired — worth knowing before anyone claims it did).
- 3.5bpw raw/indent-only HumanEval levels were not measured (would need a second boot).
- The 2.9 MMLU column-A cell is today's 01:51 run on the same pack/harness (0 invalid, identical
  question hash), not a run from this boot; a re-run was not needed and would only reproduce greedy output.

## Beyond-scope findings (noticed, not requested)

1. **Both uncensored packs answer "D" to all 22 moral_scenarios items in chat + thinking-off mode**
   (2.9 and 3.5 identically). Evidence: `results-mmlu-chat-*.json` `moral_scenarios` preds = `{'D': 22}`;
   with `effort low` 2.9 varies again. This is a live-engine behaviour, not a benchmark curiosity: if
   any eva-core lane runs the local packs with `reasoning_effort: none` through chat, moral judgement
   questions get a positionally degenerate answer. **Worth pursuing: yes** (one-line guard or an
   effort floor for local chat, decided by Jun).
2. **Venice's chat endpoint silently prepends a ~1.7k-token system prompt** unless
   `venice_parameters.include_venice_system_prompt:false`. Evidence: identical one-line message →
   `prompt_tokens 1733` vs `9`. Any hosted prompt-cost or TTFT estimate that ignores this is off by a
   factor for short prompts. **Worth pursuing: yes** (check whether our own Venice callers set it).
3. **Venice enables E2EE by default** (`venice_parameters.enable_e2ee: true` in every response). It was
   left on for the decode/prefill probes, so hosted latency may carry enclave overhead — a hosted-vs-hosted
   A/B with E2EE off would separate it. **Worth pursuing: maybe.**
4. **`~/neko-code-bench` is a brand-new harness (created 19:42 today) whose smoke run was 6 problems**
   (`results-smoke29.json`, `dataset.n: 6`); those 6 happened to pass under the broken (un-indented) chat
   transport, so the smoke test does not catch the transport bug. Any chat-harness smoke should include a
   problem where the body follows a docstring. **Worth pursuing: maybe.**
5. **`serve4-zig-1017-uncens.sh start` refused to start while the 35bpw family's containers were up**
   (`a CUDA process is still running on a node after 90 s; stop it first`), which is the correct
   behaviour — but it means "restore 2.9" is always two steps (`35bpw stop` then `uncens start`), and a
   `neko35-restore.service` unit exists that does the start step only. Documenting the two-step order in
   the 35bpw handoff would save the next worker the failed attempt. **Worth pursuing: yes (small).**
6. **`forge:/` is at 97 % (138 GiB free)** with both packs' prepared dirs resident; a third pack or a
   reprepare would not fit comfortably. **Worth pursuing: maybe (watch, don't act).**
7. **The earlier perf campaign's `29bpw-pipe0` row is a genuinely useful control** (pipeline off, same
   pack) that nothing in the handoff pointed at; it is what makes the prefill comparison honest rather
   than a 2× pipeline artefact. **Worth pursuing: no action** — just recorded here so it is not rediscovered.

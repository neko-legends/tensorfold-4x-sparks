# REPORT — neko-1017-perf-sweep-20261019

status: success

## What changed

**Nothing in the live config — that is the result.** 15 knob variants were swept against the
stock profile over full engine reboots and none beat stock by ≥3% outside boot-to-boot spread
on single-stream decode or prefill. The engine is left running healthy on the **stock**
`~/tf4/zig-1017-uncens.env` (byte-identical on all 4 nodes, verified). Two new evidence
artifacts: `~/tf4/zig-1017-uncens.env.best-20261019` (stock + note) and
`~/neko-mmlu-bench/perf-sweep.md` (+ raw JSONL in `~/neko-mmlu-bench/perf-sweep/`, 20 files).
Reusable tooling left on forge: `~/neko-mmlu-bench/perf_sweep.py` (protocol v3 harness),
`perf_summary.py`, `sweep-knob.sh` (v2, all-4-node env override block),
`sweep-restart.sh` (maintenance-aware restart + canary). Stock env backup on every node:
`~/tf4/zig-1017-uncens.env.stock-20261019`.

## Method (measurement discipline)

One knob per full reboot; env override block appended to the env file on **all four nodes**
(bug in my first knob-setter used `ssh -n` and silently never reached ranks 1-3 — caught by the
TP settings mismatch check via `TF_TP_MAILBOX_MAX_KB`, fixed, and the two affected configs
re-measured with all-rank verification). Decode: 3 prompts (speed_test.py's short_chat /
code_2k / prose_8k) × 3 reps (final baseline ×5) at **temperature 0** (deterministic —
identical reply hashes run-to-run, ±0.5% warm spread) × {effort low, high}, max_tokens 512,
single stream, medians reported. The handoff's temp-0.6 default-effort numbers (39.8/37.2/41.3)
were shown to be unusable for A/B: thinking is on by default, so replies are mostly
reasoning_content, sampled seeds differ per run, and identical-prompt reruns swing 25→43 tok/s
with the sampled text (evidence: back-to-back runs 02:57-03:05 in raw logs). Rep0 TTFTs on
stock match the handoff exactly (0.181/0.349/0.540 vs 0.20/0.37/0.55), so the protocol is
anchored to the same endpoint behavior. Prefill: 3 salted reps per size (5,296 and 21,058
tokens), max_tokens 1, `cached_tokens: 0` verified in every probe.

## Results table (medians, deltas vs mean of two stock baselines)

Full table with spreads: `~/neko-mmlu-bench/perf-sweep.md`. Summary (decode tok/s
low-effort short/code/prose; prefill tok/s 5.3k/21k):

- stock baseline ×2 boots: 68.2 / 125.8 / 62.6 (low), 70.7 / 89.4 / 61.7 (high); 2,891 / 5,001
- TF_DSV41_ENGRAM_POLL=1 (2 boots): +0.5-2% — noise
- TF_DSV41_CODE_ACCEPT=1 (2 boots): +0.5-2.4% (prose low +2.3%) — best candidate, still <3%, not adopted
- TF_DSV41_ENGRAM_WARM=1 / COEF_LATE=1 / LIN2_CLUSTER=1 / ROCE_POLLERS=2 / DRAFT_DEPTH=8: +0-2% — noise
- TF_DSV41_X3LD_EPI=1: −1-2% — reject; SAMP_BATCH=1: −0.5-1.5% — reject
- TF_TP_ROCE_BLOCKS=8: −1-1.5% — reject (stock 16 better)
- TF_DSV41_SLOTS=2: −1-2% single-stream AND halves capacity — reject
- TF_DSV41_MHC_DEFER_AT_ROWS=8: 0 to −1.5% — noise
- GLM53_TF_ROCE_MAX_KB=4096: flat — no effect
- TF_DSV41_PF_TBO=1: prefill +0.7-0.9% for +3 GiB workspace — not kept
- TF_DSV41_EXPERT_TOPP=0.80 (data-only): code_2k high +38% (123.5 tok/s) but **lossy** and short_chat −4.4% — rejected on quality; interesting data point only
- Final stock re-baseline (5 reps): 68.1 / 125.5 / 62.5 (low), 70.5 / 89.1 / 61.4 (high);
  2,883 / 5,006 — matches first baseline within ~1% ⇒ no session drift.

## Rejected by boot failure / engine rules (recovered within one attempt each)

- `TF_DSV41_PIPE_ROWS=4096`: boots, dies at first 32k prefill with `MissingTritonVariant`
  (fail-fast; the 1017 AOT fill lacks 4K pipeline-stage programs — usable only after an asset
  refill, which is out of scope).
- `TF_DSV41_PROMPT_TAIL=prefill`: refused at boot (`PREFILL=replay` requires `verify`).
- Ruled out by code (not swept): PREFILL_CHUNK/ROWS≠4096 while PF_4K=1; L2PF_MB/PACE (fixed
  values only); DRAFT_HEAD (no such knob in the Zig engine — Python-era); SLOTS=8 (no
  single-stream signal at 2, AOT slot-shaped); MHC_PFDEC variants (≥17-row windows only);
  PF_OVERLAP (TP=2 only); ENGRAM_QD (thread-pool only, live uses native AIO); draft
  trees/more candidates (prior campaign: noise or loss).

## Config left running / revert

Stock env, stock image, 4 slots × 524,288 ctx, healthy (health ok, 4 ranks `true`, canary 391).
`~/tf4/maintenance` removed after final restore (keeper cron resumed; ~06:55 last check).
Revert/rollback: nothing to revert (no live change). Emergency rollback files:
`zig-1017-uncens.env.stock-20261019` and `zig-1017-uncens.env.best-20261019` on all nodes.

## Verification (check lines)

- check: `ssh forge 'bash ~/neko-mmlu-bench/sweep-knob.sh TF_TP_ROCE_BLOCKS=8'` → block tail
  printed on anvil/ember/flame/forge (v2 setter appends on all 4 nodes)
- check: first-setter bug evidence: boot log `warning: tp bootstrap: rank 1's settings
  (… TF_TP_MAILBOX_MAX_KB …) differs from rank 0's; both ranks refuse to start` → caught,
  fixed (v2), the two configs measured before the fix (engram_poll1, code_accept1) re-run as
  *b labels and the originals excluded from conclusions
- check: `grep "waits" …boot-r1.log` after ENGRAM_POLL boot → "rows read by native AIO … waits
  polled (TF_DSV41_ENGRAM_POLL)" on rank 1 (knob reached all ranks)
- check: `grep "pipeline: prompts" …boot-r0.log` after PIPE_ROWS=4096 → "(4096 rows a segment)",
  then `MissingTritonVariant` crash at 32k prefill; after revert → "(2048 rows a segment)"
- check: `python3 ~/neko-mmlu-bench/perf_summary.py` → all 20 configs aggregated, spreads
  0.1-8%, medians as tabled above
- check: greedy determinism probe (2 identical calls, effort low/high) → identical md5 reply
  hashes, tok/s 123.84 vs 123.86 (low) — protocol foundation
- check: prefill probes → `cached_tokens: 0` on every recorded rep (salts defeat the prefix
  cache; earlier unsalted attempts showed cache hits)
- check: `grep -E "tbo|workspace" …boot-r0.log` under PF_TBO → "prefill workspace 5.22 GiB …
  worst MemAvailable 15.48 GiB" (booted safe, measured, reverted)
- check: md5sum of `zig-1017-uncens.env` vs `.stock-20261019` on all 4 nodes → same=YES ×4
- check: `~/tf4/serve4-zig-1017-uncens.sh status` → rank 0/1/2/3: true; `curl /health` → ok,
  4 slots, ctx 524288; canary 17*23 → "391"
- check: `ls ~/tf4/maintenance` → No such file (removed after final restore)
- check: `wc -l ~/neko-mmlu-bench/perf-sweep.md` → 76 lines; 20 jsonl files in perf-sweep/

## Risks / follow-ups

- Engine was rebooted ~14 times this window; final boot is stock and healthy — no risk left.
- If a future window wants the 4K pipeline stages (the one blocked prefill lever): regenerate
  the 1017 AOT fill with `TF_DSV41_PIPE_ROWS=4096` exported, then re-sweep (assets-only change,
  still no model-pack change).
- CODE_ACCEPT=1 is the leading near-candidate (+2% prose/short-chat, twice reproduced); a
  5-boot confirmation would settle whether it's a real ~2% or boot luck.
- The topp=0.80 code-decode +38% shows decode is acceptance-bound on code-like text; a
  *quality-gated* pruning study (MMLU + tool-call canaries at 0.80/0.85) is the only lever
  class that moved decode double-digits.

## Beyond-scope findings (noticed, not requested)

- `ssh -n` + stdin pipe silently drops the payload — my v1 sweep-knob.sh never applied knobs on
  ranks 1-3 and the engine *booted fine* for knobs that don't change protocol settings
  (evidence: the TP mismatch only fired on GLM53_TF_ROCE_MAX_KB). Any future multi-node env
  override must verify remote file state, not assume. — worth-pursuing: no (fixed in the v2
  script left on forge; recorded here as the trap)
- The live engine answers "first request after idle" ~30-40% slower than warm (25-30 vs 43
  tok/s on identical prompts; warms within one pass). Evidence: 02:57-03:05 runs in
  `perf-sweep/` raw logs. If anyone benches this engine casually, warm it first or the number
  is garbage. — worth-pursuing: maybe (a one-line warmup in speed_test.py would protect the
  MMLU-bench workflow)
- The engine's `cached_tokens_total` health counter only counts some cache use: repeated
  identical prompts show `cached_tokens: 0` in per-request usage yet TTFT drops 0.55→0.19 s
  (evidence: decode rep0 vs rep1-2 TTFTs in every jsonl). The per-request `cached_tokens`
  field under-reports prefix reuse. — worth-pursuing: no (cosmetic, engine-side)
- Boot log on every rank prints `dsv41 kernel refused on this GPU (… shared > 101376)` for ~7
  x3gm variants — pre-existing shape-probe noise, same on stock boots, not from this sweep.
  — worth-pursuing: no
- `~/tf4/zig-1017-uncens.env` per-node files carry rank-specific ASSETS_DIR lines, so a naive
  single-file scp rollout across nodes would corrupt ranks 1-3 paths. Evidence: forge-vs-anvil
  diff = rank comment + ASSETS_DIR only. The stock backup + best copy are per-node; any future
  rollout must use the append-block approach. — worth-pursuing: maybe (fold into any future
  ops runbook)
- decode tok/s at temp 0.6 default-effort is dominated by *what* gets generated (reasoning
  vs content, repetition): identical prompts rerun minutes apart gave 25.5 and 43.1 tok/s.
  Evidence: speed_test.py runs 03:02/03:04 vs perf_sweep varcheck. Any future speed claims
  from speed_test.py should switch to temp 0 or state the seed/content dependence.
  — worth-pursuing: yes (cheap fix, protects every future bench of this engine)

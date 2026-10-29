# Perf sweep — DeepSeek-V4.1-Flash-1017-UNCENSORED-TF, Zig TP=4 over RoCE (2026-10-19)

Window: overnight 02:43-05:30, engine owned end-to-end (orchestrator Depths, worker per
`nest/tmp/worker-handoffs/neko-1017-perf-sweep-20261019.md`). Raw data: `perf-sweep/*.jsonl`
(one file per boot). Aggregator: `perf_summary.py`.

## Method

One knob change per full reboot (`~/tf4/serve4-zig-1017-uncens.sh`, ~2.5 min), env override
block appended to `~/tf4/zig-1017-uncens.env` on ALL FOUR nodes (verified by md5 / TP settings
check). Decode: 3 prompts (short_chat / code_2k / prose_8k = the speed_test.py prompts) x 3 reps
(final baseline 5 reps) at **temperature 0** (greedy, deterministic: identical reply hashes,
+-0.5% warm spread) x {reasoning_effort low, high}, max_tokens 512, single stream; medians
reported. The temp-0.6 default-effort cells of the 2026-10-19 handoff baseline vary +-30% with
the sampled text (thinking on => reply is mostly reasoning; sampled seeds differ per run) and
are NOT usable for A/B — single noisy pass per config kept for continuity only. Prefill: 3
salted (cache-busting, cached_tokens=0 verified) reps per size, max_tokens 1: tok/s =
prompt_tokens/TTFT. "8k" = 5,296 tokens, "32k" = 21,058 tokens.

**Baseline (stock, fresh boot 03:17):** decode low-effort 68.3 / 126.1 / 62.7 tok/s
(short/code/prose), high-effort 70.9 / 89.7 / 61.9; prefill 2,899 / 4,997 tok/s.
**Final stock re-baseline (05:14, 5 reps): 68.1 / 125.5 / 62.5 (low), 70.5 / 89.1 / 61.4 (high),
2,883 / 5,006 prefill** — matches the first baseline within ~1%: no session drift; the +-1-2%
wobble between mid-session configs is boot-to-boot variance. Acceptance bar: >=3% outside spread.

## Results (medians; delta vs mean of the two stock baselines)

| config (one knob vs stock) | dec low short/code/prose | dec high short/code/prose | prefill 5.3k/21k | verdict |
|---|---|---|---|---|
| **stock (baseline x2)** | **68.2 / 125.8 / 62.6** | **70.7 / 89.4 / 61.7** | **2,891 / 5,001** | reference |
| TF_DSV41_ENGRAM_POLL=1 (x2 boots) | 69.4 / 127.7 / 63.7 | 71.8 / 91.0 / 63.0 | 2,906 / 5,025 | +0.5-2%, noise |
| TF_DSV41_CODE_ACCEPT=1 (x2 boots) | 68.9 / 127.2 / 64.2 | 72.2 / 91.3 / 63.1 | 2,908 / 5,046 | +0.5-2.4%, best candidate, still <3% |
| TF_DSV41_ENGRAM_WARM=1 | 69.2 / 127.9 / 63.2 | 72.6 / 91.5 / 63.3 | 2,921 / 5,055 | +0.5-2%, noise |
| TF_DSV41_COEF_LATE=1 | 68.6 / 127.5 / 63.8 | 71.9 / 91.0 / 62.9 | 2,901 / 5,057 | +0-1.5%, noise |
| TF_DSV41_X3LD_EPI=1 | 67.7 / 124.8 / 62.3 | 70.2 / 88.7 / 61.5 | 2,902 / 5,050 | -1-2%, reject |
| TF_DSV41_SAMP_BATCH=1 | 67.6 / 125.3 / 62.4 | 70.3 / 89.1 / 61.4 | 2,890 / 5,041 | -0.5-1.5%, reject |
| TF_TP_ROCE_BLOCKS=8 | 67.3 / 124.8 / 62.0 | 69.6 / 88.8 / 61.1 | 2,864 / 4,970 | -1-1.5%, reject |
| TF_TP_ROCE_POLLERS=2 | 68.8 / 127.2 / 63.6 | 72.1 / 90.7 / 63.0 | 2,896 / 5,024 | +0-1.5%, noise |
| TF_DSV41_LIN2_CLUSTER=1 | 69.0 / 126.9 / 63.4 | 71.0 / 90.0 / 62.1 | 2,901 / 5,042 | 0-1%, noise |
| TF_DSV41_SLOTS=2 (data-only) | 67.0 / 124.7 / 61.5 | 70.3 / 89.3 / 61.6 | 2,891 / 5,002 | -1-2% single-stream AND halves capacity, reject |
| TF_DSV41_EXPERT_TOPP=0.80 (lossy, data-only) | 72.8 / 109.4 / 63.0 | 67.7 / **123.5** / 61.9 | 2,894 / 5,045 | code_2k high +38% but lossy + short_chat -4%: rejected on quality grounds |
| TF_DSV41_MHC_DEFER_AT_ROWS=8 | 68.1 / 126.3 / 62.1 | 69.4 / 88.7 / 61.8 | 2,912 / 5,045 | 0 to -1.5%, noise |
| TF_DSV41_DRAFT_DEPTH=8 | 69.5 / 127.4 / 63.7 | 71.9 / 91.1 / 62.9 | 2,898 / 5,036 | +0-1%, noise (cost-priced depth already optimal at cap 5) |
| GLM53_TF_ROCE_MAX_KB=4096 | 68.5 / 126.2 / 62.9 | 70.8 / 89.9 / 61.9 | 2,884 / 4,947 | flat, no effect |
| TF_DSV41_PF_TBO=1 | 68.3 / 126.1 / 62.8 | 70.8 / 89.5 / 61.8 | 2,921 / 5,054 | +0.7-0.9% prefill for +3 GiB workspace, not kept |

## Boot-refused (not measurable)

- `TF_DSV41_PIPE_ROWS=4096`: boots, then the pipeline line dies at a 32k prefill with
  `MissingTritonVariant` (fail-fast, clean) — the 1017 AOT fill lacks the 4K stage programs.
  Usable only after regenerating assets with the same env.
- `TF_DSV41_PROMPT_TAIL=prefill`: refused at boot — `PREFILL=replay` requires
  `PROMPT_TAIL=verify` (engine rule).

## Ruled out by code / prior reports (not swept)

- `TF_DSV41_PREFILL_CHUNK/ROWS` != 4096: forced to 4096 while PF_4K=1 (`prod_knobs.prefillRows`).
- `TF_DSV41_L2PF_MB/PACE_GBPS`: other values refused (fixed 12 MiB / 150 GB/s); prior campaign:
  within noise anyway.
- `TF_DSV41_DRAFT_HEAD`: does not exist in the Zig engine (was a Python-engine knob).
- Draft trees / more candidates / joint depth: prior 2-Spark campaign — noise or loss.
- `TF_DSV41_SLOTS=8`: SLOTS=2 showed no single-stream gain and the AOT fill is slot-shaped; not
  pursued (multi-slot throughput was out of protocol).
- `TF_DSV41_MHC_PFDEC` variants: only affect windows >=17 rows (multi-slot rounds), not
  single-stream.
- `TF_DSV41_PF_OVERLAP`: TP=2 only (refused at world 4).
- `TF_DSV41_ENGRAM_QD`: thread-pool knob, not applicable under native AIO.

## Conclusion

**No knob beats the stock profile by >=3% on single-stream decode or prefill at 5.3k/21k
tokens.** The stock `zig-1017-uncens.env` stays live (byte-identical to
`zig-1017-uncens.env.best-20261019`; stock backup `zig-1017-uncens.env.stock-20261019` on every
node). Decode on this engine is memory-roofline-bound per the fork's DECODE.md; the remaining
headroom levers are asset-level (4K pipeline stages need an AOT refill) or model-level (expert
pruning tradeoffs), not env-level.

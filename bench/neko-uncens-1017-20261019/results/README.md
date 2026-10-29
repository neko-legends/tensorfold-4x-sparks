---
language:
- en
- es
- fr
- ru
- it
- th
license: other
license_name: deepseek-v41-flash-1017
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/LICENSE
base_model: dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: text-generation
tags:
- exllamav3
- exl3
- deepseek
- deepseek-v41
- uncensored
- abliterated
- tensorfold
- dgx-spark
- gb10
---

# DeepSeek-V4.1-Flash-1017 — UNCENSORED (Neko Legends 4× DGX Spark edition)

The [dealignai](https://x.com/dealignai) uncensored 1017 bundle, made loadable on the **TensorFold Zig TP4 engine** running DeepSeek-V4.1-Flash across **four NVIDIA DGX Spark (GB10)** nodes — the exact stack Neko Legends serves.

## What this is

[`dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED`](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED) ships two kinds of tensors our TensorFold serving path cannot consume as-is:

- the Engram n-gram tables (`layers.{1,14}.engram.*`) as ~100 GB **F8_E4M3** tensors, and
- the 54 abliterated modules (`layers.{10..36}.attn.wo_b`, `layers.{10..36}.ffn.shared_experts.w2`) stored only as **dense BF16** — their EXL3 trellis codes were stripped.

This pack is a **repack, not a re-quantization of the whole model**:

- The 54 abliterated modules are re-encoded from the dealignai BF16 weights to **EXL3 trellis** at the base pack's own bitrate per module (K=5 `wo_b`, K=4 `shared_experts.w2`), with the `mul1` codebook, **reusing each module's searched `suh`/`svh` scales from the [LibertAI](https://huggingface.co/LibertAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) base pack** (same tensor geometry — the abliteration edit is a small perturbation of the base weights).
- The Engram F8 tables are **not included** (see Serving — TensorFold serves Engram from the deepseek-ai source tables, not from the EXL3 pack).
- **Everything else is byte-identical** to the dealignai bundle (and to the LibertAI base pack wherever dealignai did not edit): tensor name set, dtypes, shapes and bytes match the base 2.9 bpw pack exactly.

## How it was built

Per edited module, with exllamav3's own codec (1.6.2-era trellis quantizer, `mul1` codebook, uncalibrated per-tile search):

1. read the module's dense BF16 weight `W` from the dealignai bundle;
2. invert the EXL3 input transform using the **base pack's fixed** `suh`/`svh`: `W_r = H₁₂₈( H₁₂₈(W / svh) / suh )`;
3. trellis-quantize `W_r` at the module's base bitrate (K=5 / K=4), `mul1` codebook;
4. pack the trellis codes with the same layout as the base pack (`pack_trellis`).

Pilot module (layer 20), measured against the dealignai BF16 weights:

| module | K | rel. error ‖dequant(new)−W‖/‖W‖ | edit retention ⟨·⟩/‖Δ‖² | cosine |
|---|---:|---:|---:|---:|
| layers.20.attn.wo_b | 5 | 0.0335 | 1.000 | 0.955 |
| layers.20.ffn.shared_experts.w2 | 4 | 0.0647 | 0.998 | 0.893 |

Edit retention ≈ 1.0 means the abliteration direction `Δ = W_uncens − dequant(base)` survives the re-quantization; values for all 54 modules are in `NEKO_BUILD.json`.

Full per-module table: all 54 modules retention 0.964–1.021 (median 1.000, every module ≥ 0.96); rel. error vs the dealignai BF16 weights 0.0269–0.0647 (wo_b, K=5: 0.0282–0.0337; shared_experts.w2: K=4 modules 0.0647, K=5 modules ≤ 0.0336). The full per-module table is in `NEKO_BUILD.json` (`pilot_metrics`/`modules`).

## Results

Measured on **4x DGX Spark (GB10)**, TensorFold Zig TP4, 524288-token context (2026-10-19). Both models run on the *same* engine, same knobs, same day:

| test | **UNCENSORED (this pack)** | censored 1017 (reference) |
|---|---:|---:|
| MMLU-500 accuracy | **81.2%** (406/500) | 86.2% (431/500) |
| - STEM (n=171) | 80.7% | 83.6% |
| - Humanities (n=113) | 71.7% | 82.3% |
| - Social (n=103) | 89.3% | 94.2% |
| - Other (n=113) | 84.1% | 86.7% |
| Refusal probe (10 harmful-trigger prompts) | **0/10 refused** (10/10 answered) | 10/10 refused |
| Benign controls (2 prompts) | 2/2 answered | 2/2 answered |
| Decode tok/s - short chat / code 2k / prose 8k | 39.8 / 37.2 / 41.3 | 40.8 / 37.8 / 41.1 |
| TTFT s - short / code 2k / prose 8k | 0.20 / 0.37 / 0.55 | 0.23 / 0.36 / 0.55 |

Protocol notes:

- **MMLU-500**: identical seeded 500-question subset for both runs (`question_set_sha256 86c9462464c1bbf244b44b475555e43b02a1dc56396839c1d5b5f0499f0ae939`), category totals stem 171 / humanities 113 / social 103 / other 113, raw `/v1/completions` (no chat template), max_tokens 8, temperature 0, concurrency 4, seed 20261019. Invalid/errors: 0/0 (uncensored) and 0/0 (censored). Wall 282s uncensored.
- **Refusal probe**: local 10-prompt harmful-trigger set (lock picking, hot-wiring, phishing, paywall bypass, destructive script, fake ID, money laundering, IT-desk social engineering, bank-impersonation phishing, GPS disabling) plus 2 benign controls, thinking disabled, temperature 0. The censored count includes 3 *soft* refusals ("I understand..., but I can't provide..."); with a strict first-sentence rule it reads 7/10. **HarmBench-320 was not run** - this is not a HarmBench number.
- The -5.0pp MMLU delta is the abliteration cost on this 2.9 bpw quantization; dealignai reports about -4.2pp for the edit at full precision, so the pack preserves the expected shape of the trade-off.
- Speed: single-stream, 4 slots x 524288, drafts on; uncensored decode is within run-to-run noise of the censored pack.

## Serving

Serve with the Neko Legends TensorFold fork for 4× DGX Spark: [`neko-legends/tensorfold-4x-sparks`](https://github.com/neko-legends/tensorfold-4x-sparks) (the Zig CUDA engine, TP=4 over RoCE). Point `PACK_DIR` (the EXL3 weights) at this folder.

The **Engram tables are served separately**: TensorFold reads the n-gram tables from the deepseek-ai 1017 source checkpoint (`deepseek-ai/DeepSeek-V4.1-Flash-1017`, shards 47/48), repacked per-rank for TP=4 by the fork's `scripts/pack_engram.py` (see the repo's `dsv41-1017-engram-pack.sh`: per-rank `engram-*.bin` shards, sha256-verified against the source). No F8 tensors are read from this pack.

## License & credits

- Base model & license: [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), under the DeepSeek V4.1 Flash license.
- 2.9 bpw EXL3 base pack: [LibertAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw).
- Uncensoring: [dealignai](https://x.com/dealignai).

Based on [`dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED`](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED) — please see their card for the abliteration methodology and their benchmark numbers.

## Safety

This model has refusal behavior removed. It will provide information it would otherwise decline. You remain fully responsible for how you use it. Do not use it to harm real people.

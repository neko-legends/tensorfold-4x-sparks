#!/usr/bin/env python3
"""Write README.md + NEKO_BUILD.json into the assembled Neko Legends 3.5bpw pack on forge.
Run after assemble35 + verify35. Optional --results <json> embeds the bench results (run again post-bench)."""
import argparse, hashlib, json, time
from pathlib import Path

ROOT = Path("/home/jun/models")
NEW = ROOT / "neko-legends-DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-UNCENSORED-4xSpark"
STAGE = ROOT / "neko35-stage"

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=None, help="path to results-35bpw.json + perf summary to embed")
args = ap.parse_args()

metrics = json.load(open(STAGE / "encoded35" / "metrics.json"))
assert len(metrics) == 54
by = {m["module"]: m for m in metrics}
pilot = [by["layers.20.attn.wo_b"], by["layers.20.ffn.shared_experts.w2"]]
rets = [m["edit_retention"] for m in metrics]
rels = [m["rel_err_new"] for m in metrics]
wo = [m for m in metrics if m["module"].endswith("wo_b")]
w2 = [m for m in metrics if m["module"].endswith("w2")]

pilot_rows = "\n".join(
    f"| {m['module']} | 6 | {m['rel_err_new']:.4f} | {m['edit_retention']:.3f} | {m['edit_cosine']:.3f} |"
    for m in pilot)
summary = (
    f"all 54 modules retention {min(rets):.3f}–{max(rets):.3f} (median {sorted(rets)[27]:.3f}); rel. error vs the "
    f"dealignai 3.5bpw BF16 weights {min(rels):.4f}–{max(rels):.4f} (wo_b {min(m['rel_err_new'] for m in wo):.4f}–"
    f"{max(m['rel_err_new'] for m in wo):.4f}; shared_experts.w2 {min(m['rel_err_new'] for m in w2):.4f}–"
    f"{max(m['rel_err_new'] for m in w2):.4f}). K=6 (96 words) matches the pack's native wo_b/shared bitrate. "
    "Full per-module table in `NEKO_BUILD.json`."
)

results_md = ""
if args.results:
    r = json.load(open(args.results))
    res = r["mmlu500"]
    u = res["uncensored_35bpw"]
    v29 = res["uncensored_29bpw"]
    vc = res["censored_29bpw"]
    perf = r["perf"]
    c29 = perf["control_29bpw_pipe0"]
    pon = perf["published_29bpw_pipe_on"]
    cat = res["categories"]
    cat29 = res["categories_29bpw"]
    catc = res["categories_censored"]
    moral = res["per_subject"]["moral_scenarios"]
    moral29 = res["per_subject_29bpw"]["moral_scenarios"]
    moralc = res["per_subject_censored"]["moral_scenarios"]
    qh = r["harness"]["question_set_sha256"]

    def mrow(name, a, b, c, n=None):
        nn = f" (n={n})" if n else ""
        return f"| {name}{nn} | {a}% | {b}% | {c}% |"

    results_md = f"""
## Results

Measured on **4x DGX Spark (GB10)**, TensorFold Zig TP4, 524288-token context, 4 slots, same harness as the 2.9bpw
Neko pack (2026-10-19). MMLU-500: 5-shot, temperature 0, raw `/v1/completions`, identical question set
(`question_set_sha256 {qh[:16]}…`), run with port 8000 closed to non-loopback peers. Perf: `perf_sweep.py`
protocol v3 (temperature 0, 3 reps, medians; salted prefill with `cached_tokens: 0`).

| MMLU-500 | **3.5bpw UNCENSORED (this pack)** | 2.9bpw UNCENSORED (Neko) | censored 1017 (reference) |
|---|---:|---:|---:|
| accuracy | **{u['accuracy_pct']}%** ({u['correct']}/{u['n']}) | {v29['accuracy_pct']}% ({v29['correct']}/{v29['n']}) | {vc['accuracy_pct']}% ({vc['correct']}/{vc['n']}) |
{mrow("- STEM", cat['stem']['accuracy_pct'], cat29['stem']['accuracy_pct'], catc['stem']['accuracy_pct'], cat['stem']['n'])}
{mrow("- Humanities", cat['humanities']['accuracy_pct'], cat29['humanities']['accuracy_pct'], catc['humanities']['accuracy_pct'], cat['humanities']['n'])}
{mrow("- Social", cat['social']['accuracy_pct'], cat29['social']['accuracy_pct'], catc['social']['accuracy_pct'], cat['social']['n'])}
{mrow("- Other", cat['other']['accuracy_pct'], cat29['other']['accuracy_pct'], catc['other']['accuracy_pct'], cat['other']['n'])}
{mrow("- **moral_scenarios**", moral['accuracy_pct'], moral29['accuracy_pct'], moralc['accuracy_pct'], moral['n'])}

**The extra bits do not recover the abliteration gap.** 3.5bpw scores {u['accuracy_pct']}% against the 2.9 pack\'s
{v29['accuracy_pct']}% — two questions of 500 — and {round(vc['accuracy_pct'] - u['accuracy_pct'], 2)} pp under the
censored model, barely different from the 2.9 pack\'s {round(vc['accuracy_pct'] - v29['accuracy_pct'], 2)} pp gap.
The humanities gap is entirely `moral_scenarios`, and K=6 recovers exactly one of that subject\'s questions
({moral['correct']}/{moral['n']} vs {moral29['correct']}/{moral29['n']}; censored {moralc['correct']}/{moralc['n']}).
Bitrate is not what ablates the model here.

### Speed

The 3.5bpw pack cannot run the 2.9 world\'s prefill pipeline: its prepared images are **67.3 GiB a rank**
(2.9: 55.1) and the pipeline-stage copies another 33 GiB (2.9: 25), and with both resident the engine\'s measured
kv gate leaves under its 4 GiB hard floor on a 121 GiB node. This pack therefore runs with
`TF_DSV41_PREFILL_PIPE=0`; the 2.9 pack was re-measured at the same setting (control) so the numbers compare like
for like, next to the published pipe-on baseline.

| tok/s (median) | **3.5bpw, PIPE=0** | 2.9, PIPE=0 (control) | 2.9, PIPE=1024 (published) |
|---|---:|---:|---:|
| decode low — short chat / code 2k / prose 8k | **{perf['low']['short_chat']} / {perf['low']['code_2k']} / {perf['low']['prose_8k']}** | {c29['low']['short_chat']} / {c29['low']['code_2k']} / {c29['low']['prose_8k']} | {pon['low']['short_chat']} / {pon['low']['code_2k']} / {pon['low']['prose_8k']} |
| decode high — short chat / code 2k / prose 8k | **{perf['high']['short_chat']} / {perf['high']['code_2k']} / {perf['high']['prose_8k']}** | {c29['high']['short_chat']} / {c29['high']['code_2k']} / {c29['high']['prose_8k']} | {pon['high']['short_chat']} / {pon['high']['code_2k']} / {pon['high']['prose_8k']} |
| prefill ~5.3k tokens | **{perf['prefill']['p5k']:.0f}** | {c29['prefill']['p5k']:.0f} | {pon['prefill']['p5k']:.0f} |
| prefill ~21k tokens | **{perf['prefill']['p21k']:.0f}** | {c29['prefill']['p21k']:.0f} | {pon['prefill']['p21k']:.0f} |

Decode is 8–12% slower than the 2.9 pack at the same config: the 54 re-encoded modules now carry K=6 (96 words) in
every layer\'s dense path instead of K=5/K=4, and single-stream decode here is memory-roofline-bound. Prefill is
within 2–3% of the 2.9 pack at the same config; the 2.9 pipe-on column shows what the pack gives up by having to
run the line off.
"""

readme = f"""---
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
base_model: dealignai/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-UNCENSORED
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

# DeepSeek-V4.1-Flash-1017 — 3.5bpw UNCENSORED (Neko Legends 4× DGX Spark edition)

The [dealignai](https://x.com/dealignai) uncensored 3.5bpw bundle, made loadable on the **TensorFold Zig TP4 engine**
running DeepSeek-V4.1-Flash across **four NVIDIA DGX Spark (GB10)** nodes — the exact stack Neko Legends serves.

## What this is

[`dealignai/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-UNCENSORED`](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-UNCENSORED)
ships the same two structural catches as the 2.9bpw bundle, plus one difference:

- the Engram n-gram tables (`layers.{{1,14}}.engram.*`) as F8_E4M3 tensors in shards 47/48 — deliberately **not
  downloaded**; TensorFold serves Engram from the deepseek-ai source tables, not from the EXL3 pack;
- the 54 abliterated modules (`layers.10..36.attn.wo_b`, `layers.10..36.ffn.shared_experts.w2`) stored only as
  **dense BF16** — the engine requires EXL3 trellis codes.

This pack re-encodes exactly those 54 modules and leaves everything else byte-identical to the dealignai 3.5bpw
bundle:

- the 54 modules are encoded to **EXL3 trellis K=6 `mul1`** — the 3.5bpw pack's own native bitrate for `wo_b` and
  the shared experts (96 words/tile; the 2.9bpw Neko build used K=5/K=4 at its lower bitrate) — **reusing each
  module's searched `suh`/`svh` scales and `mul1` marker from the 2.9bpw [Neko/LibertAI]
  (https://huggingface.co/LibertAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) lineage** (same tensor geometry, same
  codebook family; the scales are copied through byte-identical);
- the quantization is **edit-aware LDLQ** around the 2.9bpw Neko pack's own trellis: the Hessian is built from the
  3.5-vs-2.9 edit direction so the quantization error lands orthogonal to the edit;
- everything else (tensor names, dtypes, shapes, bytes) is **byte-identical** to the dealignai 3.5bpw bundle.

## How it was built

Per edited module, with exllamav3's codec (`mul1` codebook, edit-aware LDLQ, λ=0.1):

1. read the module's dense BF16 weight `W` from the dealignai 3.5bpw bundle;
2. invert the EXL3 input transform using the **2.9bpw Neko pack's fixed** `suh`/`svh`:
   `W_r = H₁₂₈( H₁₂₈(W / svh) / suh )`;
3. build `D_reg = W_r − dequant(trellis_2.9, K₂.₉)` (the 3.5 edit in regularized space) and the Hessian
   `H = D_reg D_regᵀ/n + 0.1·(‖D_reg‖²/kn)·I`; `L = block_ldl(H, 16)`;
4. trellis-quantize `W_r` at **K=6** with `ldlq(W_r, L)`, pack with `pack_trellis`.

Pilot modules (layer 20), measured against the dealignai 3.5bpw BF16 weights:

| module | K | rel. error ‖dequant(new)−W‖/‖W‖ | edit retention ⟨·⟩/‖Δ‖² | cosine |
|---|---:|---:|---:|---:|
{pilot_rows}

Edit retention ≈ 1.0 means the direction `Δ = W_3.5 − dequant(2.9 pack)` survives the re-quantization.
{summary}
{results_md}
## Serving (TensorFold Zig TP4)

Same as the 2.9bpw Neko pack: `MODEL_DIR` → this dir, Engram from `/home/jun/dsv41-1017-engram-w4`, assets from
the 512k rope/aot set. First boot builds a fresh prepared dir.

## Credits

- **[dealignai](https://huggingface.co/dealignai)** — the uncensoring (abliteration) of DeepSeek-V4.1-Flash-1017 and
  the 3.5bpw UNCENSORED EXL3 bundle this pack repacks.
- **[LibertAI/dealignai](https://huggingface.co/LibertAI)** (LibertAIDAI) — the 3.5bpw EXL3 base quantization lineage
  and the searched `suh`/`svh` scale structure reused here.
- **[deepseek-ai](https://huggingface.co/deepseek-ai)** — DeepSeek-V4.1-Flash-1017, the model itself.

Upstream dealignai card preserved as `README.source-dealign.md`.
"""

(NEW / "README.md").write_text(readme)
print("README.md written")

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

t0 = time.time()
shas = {}
for p in sorted(NEW.rglob("*")):
    if p.is_file() and p.name != "NEKO_BUILD.json":
        shas[str(p.relative_to(NEW))] = sha256(p)
print(f"hashed {len(shas)} files in {time.time()-t0:.0f}s")

build = {
    "pack": "neko-legends/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-UNCENSORED-4xSpark",
    "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "source": {
        "repo": "dealignai/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-UNCENSORED",
        "revision": "7af7f28516f5",
        "local_dir": "/home/jun/models/dealign-3.5bpw-uncens",
    },
    "scale_donor": {
        "pack": "neko-legends/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED-4xSpark",
        "note": "suh/svh/mul1 reused byte-identical; edit base = that pack's trellis (K=5 wo_b, K=4/5 w2)",
    },
    "credits": {
        "uncensoring": "dealignai",
        "exl3_base_quantization": "LibertAIDAI (3.5bpw EXL3 lineage)",
        "model": "deepseek-ai",
    },
    "re_encoded_modules": [m["module"] for m in metrics],
    "method": (
        "Per module: invert the EXL3 input transform with the 2.9 Neko pack's fixed suh/svh "
        "(W_r = H128(H128(W/svh)/suh)), then edit-aware LDLQ trellis quantization at K=6 mul1 with Hessian "
        "H = D_reg D_reg^T/n + 0.1*(||D_reg||^2/(k*n))*I where D_reg = W_r - dequant_reg(2.9 Neko trellis, K2.9). "
        "Scales suh/svh and the mul1 marker are copied from the 2.9 Neko pack unchanged (byte-identical). "
        "Encoder: exllamav3 1.4.4 tree (/home/jun/git/exllamav3, torch 2.11.0+cu128, CUDA ext at "
        "~/.cache/torch_extensions/py312_cu128), block_ldl/ldlq/pack_trellis/reconstruct."
    ),
    "codec_params": {"codebook": "mul1", "mul1_multiplier": 2212286765, "K": 6,
                     "lambda_edit_hessian": 0.1, "out_scales": "2.9 Neko pack suh/svh reused unchanged"},
    "dropped_tensors": [
        "layers.{1,14}.engram.{embed,wkv} F8 shards (model-00047/48-of-00048, never downloaded; "
        "TensorFold serves Engram from the deepseek-ai 1017 source via /home/jun/dsv41-1017-engram-w4)",
    ],
    "pilot_metrics": pilot,
    "modules": metrics,
    "notes": [
        "54 modules = layers.10..36 attn.wo_b + ffn.shared_experts.w2 (abliterated by dealignai, shipped dense BF16)",
        "17 untouched shards hard-linked from the dealignai 3.5 dir; 27 shards rewritten (dense .weight dropped, "
        "all other tensors byte-identical, trellis/suh/svh/mul1 appended); index = source - 54 .weight + 216 new",
        "ONE index correction vs the source (fix_engram_index.py): layers.{1,14}.engram.{k,q}_weight were mapped "
        "to the never-downloaded FP8 shards 47/48 while the identical F32 [4,5120] tensors physically live in "
        "shards 2/15 (the source index is mis-mapped). Remapped to 2/15, exactly where the 2.9 Neko pack keeps "
        "them; the engine reads tensor ownership from the index and otherwise fails boot with MissingTensor.",
        "The two absent FP8 Engram shards (wrapper names layers.{1,14}.engram.{embed.*,wkv.weight,wkv.scale}) are "
        "still listed in the index, as in the source pack; the engine ignores names it does not plan to read "
        "(pack.zig) and Engram tables come from /home/jun/dsv41-1017-engram-w4.",
        "SHA256SUMS.source-dealign = the upstream hashes of the SOURCE pack; it does not describe these shards. "
        "This pack's own file hashes are files_sha256 below.",
        "K=6 (96 words) is the 3.5bpw pack's own native bitrate for wo_b and shared experts",
        "Edit retention = <dequant(new,K6) - dequant(2.9pack,K2.9), W_3.5 - dequant(2.9pack,K2.9)> / ||W_3.5 - dequant(2.9pack,K2.9)||^2",
    ],
    "files_sha256": shas,
}
if args.results:
    build["results"] = json.load(open(args.results))
(NEW / "NEKO_BUILD.json").write_text(json.dumps(build, indent=1))
print("NEKO_BUILD.json written")

# TensorFold on 4× DGX Spark: DeepSeek V4.1 Flash and Nemotron 3.5 Super VL

Two large models served across **four** NVIDIA DGX Sparks with [TensorFold](https://github.com/ashhart/TensorFold)'s
Zig engine: tensor-parallel (TP-4) over a switched CX7 RoCE fabric, one OpenAI-compatible endpoint, image input,
speculative decoding from each model's own draft head.

> **The engine is ours, not the upstream release.** The DeepSeek lane runs our rebase of TensorFold 1.1.2 plus this
> repo's TP-4 delta, on our four Sparks and our 2.9 bpw pack (see [BENCHMARKS §4](docs/BENCHMARKS.md)). Upstream 1.1.2
> alone does not run DeepSeek-V4.1 at four ranks. Other TensorFold versions and other packs are untested here, so the
> figures below belong to this engine, not to TensorFold in general.

| | **DeepSeek V4.1 Flash 1017** (live) | Nemotron 3.5 Super VL 120B-A12B | DeepSeek V4.1 Flash |
| --- | --- | --- | --- |
| weights | the official 1017 weights, third-party 2.9 bpw EXL3 quant | NVIDIA's NVFP4 checkpoint, 70 GB | 2.9 bpw EXL3 pack |
| engine | the DeepSeek engine, unchanged: [a config-only drop-in](#deepseek-v41-flash-1017-a-config-only-drop-in) | TensorFold 1.0.11 (Zig, CUDA) + this repo's TP-4 port, [`engine/zig-nemotron`](engine/zig-nemotron) | Jay Leaton's DeepSeek CUDA port of the Zig engine + this repo's TP-4 work, [`engine/zig`](engine/zig) |
| one stream, code / prose | 152 / 67 tok/s (DSpark drafting) | **185 / 95 tok/s** (MTP drafting) | 155 / 72 tok/s (DSpark drafting) |
| at once | up to **512K tokens** per request (524,288; passkey correct at 520K; the shared KV pool fits about 2.3 such contexts across the 4 slots) | **16 conversations**, 307,200-token window each | 4 request slots of up to 420K tokens |
| serving | since 2026-10-17 | 2026-10-15 → 2026-10-16 | 2026-10-11 → 2026-10-15, 2026-10-16 → 2026-10-17 |

DeepSeek V4.1 Flash 1017 is what these four Sparks serve today, on the same DeepSeek engine. Nemotron's and the earlier
DeepSeek pack's numbers come from their own runs on the same cluster, and their recipes below still work. Built on **Ash Hart's [TensorFold](https://github.com/ashhart/TensorFold)**
(the engine, the server, the kernels) and **Jay Leaton's [DeepSeek V4.1 Flash port](https://github.com/jayleaton/deepseek-v41-tensorfold-spark)**;
this repository adds the four-Spark parts and the measurements.

> Measured on one cluster (four GB10s, one switch). Every number links to its raw files. Read the known limits of each
> model ([Nemotron](#nemotron-known-limits), [DeepSeek](#what-is-not-solved)) before relying on it.

- [Head to head](#head-to-head)
- [Nemotron 3.5 Super VL](#nemotron-35-super-vl-120b-a12b-nvfp4): [serving capacity](#serving-capacity) ·
  [memory](#memory-per-spark) · [images under load](#images-under-load) · [how to run](#how-to-run-nemotron) ·
  [what we changed](#what-we-changed-for-nemotron)
- [DeepSeek V4.1 Flash](#deepseek-v41-flash): [the 1017 checkpoint as a drop-in](#deepseek-v41-flash-1017-a-config-only-drop-in)
  ([how to run it](#run-deepseek-v41-flash-1017-config-only-drop-in)),
  results, quick start, the engine and its tests
- [Shared engine notes](#shared-engine-notes): TP-4 over RoCE, the prompt cache under TP and its disk tier, staggered
  prompt segments
- [Documentation](#documentation) · [Licensing](#licensing) · [Credits](#credits)

## Head to head

Same four Sparks, same harnesses (spark-bench's frozen `qbench` suite and the same vision reads), thinking off,
greedy. Chart sources and the script that draws them: [`engine/zig-nemotron/results/charts-20261016/`](engine/zig-nemotron/results/charts-20261016/)
(`numbers.json` names every input file).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="engine/zig-nemotron/results/charts-20261016/nemotron-vs-deepseek-quality-dark.svg">
  <img alt="Quality: MMLU 85.0% Nemotron vs 88.5% DeepSeek, math 96.7% vs 100%, coding 80% both, instruction following 95% vs 90%, JSON, tools, long context and summaries 100% both" src="engine/zig-nemotron/results/charts-20261016/nemotron-vs-deepseek-quality-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="engine/zig-nemotron/results/charts-20261016/nemotron-vs-deepseek-speed-dark.svg">
  <img alt="Writing speed, one stream: code 185 tok/s Nemotron vs 155 DeepSeek, prose 95 vs 72; four streams 207 vs 172 tok/s aggregate; first token 43 ms vs 167 ms on the code prompt" src="engine/zig-nemotron/results/charts-20261016/nemotron-vs-deepseek-speed-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="engine/zig-nemotron/results/charts-20261016/nemotron-vs-deepseek-prefill-dark.svg">
  <img alt="Cold prompt reading: Nemotron 32K tokens 6.5 s, 131K 29.0 s, 200K 48.3 s (from 11.4, 49.6 and 79.9 s before staggered segments); DeepSeek 200K 36.8 s" src="engine/zig-nemotron/results/charts-20261016/nemotron-vs-deepseek-prefill-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="engine/zig-nemotron/results/charts-20261016/nemotron-vs-deepseek-vision-dark.svg">
  <img alt="Vision on code-rendered images: text read exactly 166/166 for both; shapes 26/30 both; objects counted 22/23 DeepSeek vs 1/23 Nemotron; text position error 9.9 px Nemotron vs 16.3 px DeepSeek" src="engine/zig-nemotron/results/charts-20261016/nemotron-vs-deepseek-vision-light.svg">
</picture>

| | Nemotron 3.5 Super VL | DeepSeek V4.1 Flash |
| --- | ---: | ---: |
| MMLU (200) | 85.0% | **88.5%** |
| math (30) | 96.7% | **100%** |
| coding (5) | 80% | 80% |
| instruction following (20) | **95%** | 90% |
| JSON schema / tool calls / long context (25K/100K/229K-token prompts) / summaries | 100% / 100% / 100% / 100% | 100% / 100% / 100% / 100% |

¹ Same harness, same engine; the earlier pack's Blender 3D score (79) came from a hosted endpoint in a race, so the 3D columns are not like-for-like. Long-context prompts measured 25K/100K/229K tokens.
| one stream, code / prose (tok/s) | **185 / 95** | 155 / 72 |
| four streams, aggregate (tok/s) | **207** | 172 |
| first token, short prompt | **43 ms** | 167 ms |
| cold 200K-token prompt | 48.3 s | **36.8 s** |
| vision: text read exactly | 100% (166/166) | 100% (166/166) |
| vision: shapes matched / objects counted | 87% / 4% | 87% / **96%** |

Where each wins: Nemotron writes faster (the MTP head drafts code well), answers sooner, holds 16 long conversations
at once and places text more precisely; DeepSeek is ahead on MMLU and math, reads a cold long prompt faster and
counts objects in an image far better (Nemotron 1/23 on the counting scene). The vision shapes figure is with the
live setting `NCCL_MAX_NCHANNELS=2`: Nemotron's shapes scene read 17/20 with that cap and 19/20 without it (the cap
makes long prompts read faster by not starving the staggered segments; it changes the arithmetic at rounding level,
which moved this one scene; today's reads at the same setting gave 17/20 with the zoomed re-read and 19/20 without).

## Nemotron 3.5 Super VL 120B-A12B (NVFP4)

[`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4):
88 blocks (40 Mamba-2, 40 latent mixture-of-experts with 512 experts top-22, 8 attention), 12B active parameters, an
MTP draft head and a C-RADIO vision encoder, served from NVIDIA's NVFP4 checkpoint as stored. The live build is
`c3ede919` on this repository's history.

### Serving capacity

On the live server, four Sparks, one endpoint:

- **16 conversations at once**, each with a **307,200-token** window.
- **~214-230 tok/s** total writing speed with all 16 streams busy (prose 217, code 230, mixed 214).
- First token with 16 streams arriving together: **0.8 s median, 1.9 s worst**.
- One stream: **185 tok/s** on code, **95** on prose; first token in 43 ms.
- A **40 GiB prompt cache in memory on every rank** plus a **disk tier**: 200 agents coming back to a 10K-token
  history wait 0.09 s median (200/200 hits); after a server restart, the disk tier brings 8 30K-token conversations
  back in 0.33 s instead of 6.0 s.
- Images: ~70 a minute for images the server has seen, ~23 a minute for new ones (below).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="engine/zig-nemotron/results/charts-20261016/nemotron-serving-dark.svg">
  <img alt="16 conversations at once: first-token wait 0.8-0.9 s median with 16 slots vs 5.0-9.4 s with the earlier 4 slots; coming back to a long conversation 0.09 s with the 40 GiB cache vs 2.1 s, and 0.33 s after a restart with the disk tier vs 6.0 s" src="engine/zig-nemotron/results/charts-20261016/nemotron-serving-light.svg">
</picture>

Raw files: [`spare-mem-20261015/`](engine/zig-nemotron/results/spare-mem-20261015/) (16-stream load and warm-cache
runs), [`live-speed-20261016/`](engine/zig-nemotron/results/live-speed-20261016/) (one-stream speed on the live build).

### Memory per Spark

Each GB10 has 121.7 GiB shared by CPU and GPU. Per rank:

| | GiB |
| --- | ---: |
| model weights (a quarter of the model, plus the replicated parts) | 25.7 |
| prompt cache (`--prompt-cache-gib 40`) | 39.8 |
| one sequence's state at the full 307,200-token window | 1.37 (16 slots: 21.9) |
| free, idle (all four Sparks) | 85.0-86.8 |
| free, prompt cache full and all 16 slots used | 46.7-48.6 |
| free if every slot also held a full-window conversation (computed) | ~24.8 |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="engine/zig-nemotron/results/charts-20261016/nemotron-memory-dark.svg">
  <img alt="Memory per Spark, 122 GiB: idle 25.7 GiB weights + 11.0 other, 85.0 GiB free; prompt cache full 25.7 + 39.8 + 9.4, 46.7 GiB free; ceiling with 16 full-window sequences (21.9 GiB) 24.8 GiB free" src="engine/zig-nemotron/results/charts-20261016/nemotron-memory-light.svg">
</picture>

Measured read-only on the live server (`free -b`, `nvidia-smi`) on 2026-10-16, and with the cache full on
2026-10-15: [`memory-20261016/memory.json`](engine/zig-nemotron/results/memory-20261016/memory.json).

### Images under load

1 to 16 image requests at once on the live server, 8 code-rendered bench images, one transcribe-and-describe prompt:

| at once | new images: per minute / first token (median, worst) | seen images: per minute / first token |
| ---: | ---: | ---: |
| 1 | 18.5 / 1.9 s, 2.7 s | 45 / 0.1 s, 0.3 s |
| 4 | 22.9 / 3.8 s, 7.8 s | 67 / 0.2 s, 0.5 s |
| 8 | 24.1 / 6.4 s, 13.7 s | 69 / 0.4 s, 1.0 s |
| 16 | 22.5 / 25.3 s, 31.1 s | 70 / 1.3 s, 2.1 s |

- **New images are taken one at a time**: ~2.6 s each (reading the picture in, encoding it and reading the prompt),
  so throughput stays at ~23 a minute and the wait grows with the queue. Images the server has already read (a
  follow-up question) are answered almost at once.
- **Accuracy does not change under load**: the transcriptions found 304 of 314 strings at every level; 16 structured
  tiled reads at once read 277/280 text strings (98.9%) against 69/70 (98.6%) on one stream.
- **Image replies decode faster than prose**: 140 tok/s on one stream (78% of MTP drafts accepted) vs 95 for prose
  (65%) and 185 for code (86%).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="engine/zig-nemotron/results/charts-20261016/nemotron-vision-serving-dark.svg">
  <img alt="Images under load: new images 18-24 a minute at 1-16 at once with first-token median rising from 1.9 to 25.3 s; images the server has seen 45-70 a minute with 0.1-1.3 s; one stream writes 140 tok/s on image prompts, 95 on prose, 185 on code" src="engine/zig-nemotron/results/charts-20261016/nemotron-vision-serving-light.svg">
</picture>

Method and every request: [`vision-serving-20261016/`](engine/zig-nemotron/results/vision-serving-20261016/README.md).

### Drafting

The MTP head drafts ahead and stops when its confidence drops, instead of drafting a fixed depth from the running
hit rate. Code went from 165 to 183 tok/s with token-for-token the same reply; prose is unchanged at ~95.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="engine/zig-nemotron/results/charts-20261016/nemotron-drafting-dark.svg">
  <img alt="Drafts sized by confidence vs by running hit rate: code 165 to 183 tok/s (same text), explain 117 to 127, prose 95 to 94, story 99 to 98, essay 112 to 111" src="engine/zig-nemotron/results/charts-20261016/nemotron-drafting-light.svg">
</picture>

### How to run Nemotron

Requirements: four DGX Sparks on one switch with a RoCE link (one CX7 port per Spark on one subnet), passwordless
ssh from rank 0 to the others, the checkpoint at the same path on all four (70 GB each), Zig 0.17.0 and CUDA on one
Spark to build.

```bash
git clone https://github.com/neko-legends/tensorfold-4x-sparks.git
cd tensorfold-4x-sparks/engine/zig-nemotron

# 1. build on a Spark (CUDA kernels in the binary), copy zig-out/native/bin/tensorfold-native to all four
zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dtarget=aarch64-linux-gnu

# 2. weights, same path on every Spark
hf download nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 --local-dir <MODEL>

# 3. one rank per Spark, r = 0..3 (rank 0 serves HTTP; the others follow it)
env TF_TP_WORLD=4 TF_TP_RANK=<r> TF_TP_MASTER=<rank 0 link address> TF_TP_PORT=29650 \
    TF_COMM_BACKEND=roce TF_TP_ROCE_HCA=<RoCE device> NCCL_IB_HCA=<RoCE device> GLM53_TF_ROCE_FAST=1 \
    TF_TP_ROCE_FALLBACK=nccl TF_NCCL_LIB=<libnccl.so.2> NCCL_SOCKET_IFNAME=<link interface> NCCL_MAX_NCHANNELS=2 \
    TF_NEMOTRON_DISK=<prompt-cache disk folder> \
  tensorfold-native serve <MODEL> --host 0.0.0.0 --port 8100 --no-update-check \
    --context 307200 --parallel 16 --segments 4 --prompt-cache-gib 40
```

Then `curl <rank 0>:8100/v1/chat/completions` with any OpenAI client (`chat_template_kwargs: {"enable_thinking":
false}` turns thinking off for a request). The operator scripts we use, including start, clean stop (rank 0 first, so
the disk tier is written) and status: [`tools/nemotron/swap/`](engine/zig-nemotron/tools/nemotron/swap/)
(`nemotron-window.sh`; the Sparks' names and link address come from `sparks.sh`). Serving notes (exactness and the
cache state, installing a new binary, the env switches, the disk tier's identity): [`WINDOW.md`](engine/zig-nemotron/tools/nemotron/swap/WINDOW.md).

### What we changed for Nemotron

Upstream TensorFold 1.0.11 serves Nemotron 3.5 Lightning (30B) on one GPU. To serve the 120B Super VL across four
Sparks we added
([port log](engine/zig-nemotron/NEMOTRON-PORT-LOG.md)):

- **Super VL support**: the checkpoint's config wrapper and names, the latent mixture-of-experts (router on the hidden
  width, NVFP4 experts at the 1,024-wide latent, shared expert), the 512-expert top-22 router, FP8 attention
  projections, the MTP head through the same path; checked block by block against the checkpoint's own Hugging Face
  code (cosine ≥ 0.9994 on every block type).
- **TP-4**: Mamba heads and groups, attention heads and the expert width (672, padded to 704) split four ways, with
  gathers and all-reduces between them; vocabulary-split head and draft head. Per-block agreement with one rank ≥ 0.9999.
- **Serving under TP**: rank 0 runs the server and the others replay its calls over the plan link, images included;
  16 slots sharing each round.
- **Prompt cache under TP, and a disk tier** (below), **staggered prompt segments** (below).
- **Drafting sized by confidence** for a lone stream (the chart above).
- **Images**: the image front end checked against the checkpoint's processor (same grids, same prompt length,
  pixels ≥ 99.96% bf16-exact).

### Nemotron: known limits

- New images are prepared one at a time (~23 a minute at any concurrency); a burst of new images queues.
- Object counting in images is weak (1/23 on the counting scene, DeepSeek 22/23).
- A cold 200K-token prompt takes 48 s (DeepSeek 37 s).
- With the prompt cache on, greedy text depends on the cache's state at rounding level (deterministic for the same
  request sequence); A/B tests pin the cache state.
- One cluster, measured over two days.

## DeepSeek V4.1 Flash

*DeepSeek served these four Sparks from 2026-10-04 to 2026-10-15 and again from 2026-10-16. Since 2026-10-17 the same
engine serves the official 1017 checkpoint ([below](#deepseek-v41-flash-1017-a-config-only-drop-in)). The rest of this
section is written as of the earlier pack: "live" below means live then. The recipe and the engine still run as described.*

Serve DeepSeek-V4.1-Flash (the 2.9 bpw EXL3 pack) across **four** NVIDIA DGX Sparks, tensor-parallel (TP=4) over a
switched CX7 RoCE fabric, behind an OpenAI-compatible API: exact DSpark speculative decoding, Engram rows from local
NVMe, 4 request slots of up to 420K tokens, image input, NVMe sessions, structured output and DSML tool calls.

This part of the repository began as the four-Spark fork of **[jayleaton/deepseek-v41-tensorfold-spark](https://github.com/jayleaton/deepseek-v41-tensorfold-spark)**.
The engine, the DeepSeek-V4.1-Flash family and nearly everything in this repository are Jay Leaton's work on
[TensorFold](https://github.com/ashhart/TensorFold); this fork adds what four Sparks need (`patches/0003`), a
four-node launcher, and the measurements. Its Python engine ran Jay's G19 build from 2026-10-07; since 2026-10-11
the serving engine is the CUDA Zig one (below). **Two Sparks? Use Jay's repository**: it is the maintained
two-Spark recipe.

> Measured on one cluster (four GB10s, one switch), one boot a configuration. Read [What is not solved](#what-is-not-solved)
> before relying on it.

Jay's [CUDA Zig serving path](docs/ZIG-SERVE.md) (TensorFold 1.0.2 plus the DeepSeek CUDA port, source in
[`engine/zig`](engine/zig)) is merged here as he ships it for **two Sparks (TP=2) and his q28-v2 2.8 bpw pack** —
and this fork extends it to four Sparks on the 2.9 bpw pack: **since 2026-10-11 the four-Spark service is the Zig
engine, TP=4 over RoCE, with pipelined prompt reading** ([ZIG-RESULTS](docs/ZIG-RESULTS.md); the port was planned
in [ZIG-TP4-PLAN](docs/ZIG-TP4-PLAN.md), the pipeline's design and proofs are in
[ZIG-PREFILL-PIPE](docs/ZIG-PREFILL-PIPE.md)). The Python engine it replaced stays on this page as the previous
engine, its numbers kept as history.

### DeepSeek V4.1 Flash 1017: a config-only drop-in

Live since 2026-10-17. DeepSeek's official `deepseek-ai/DeepSeek-V4.1-Flash-1017`, as the third-party 2.9 bpw EXL3 quant
[`LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw`](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)
(rev `afc7aa0a`), runs on **the same engine, image and serving settings** as the pack below, at TP=4 over RoCE. No code
change and no new image were needed. The config, tokenizer and every tensor name and shape match the earlier pack; the
1017 pack adds 43 `gate.bias_vl` tensors, and its drafter matrices are 6 bpw (the loader reads the width from the
weights). Three weight-derived inputs changed in 1017 and had to be rebuilt:

- **Engram tables:** repacked per rank from the 1017 base checkpoint's Engram shards (47 and 48), as in
  [docs/FOUR_SPARKS.md](docs/FOUR_SPARKS.md). The earlier tables do not match the 1017 weights.
- **Image-routing bias (`TF_DSV41_BIAS_VL`):** rebuilt from the 1017 pack's `gate.bias_vl` tensors; 34 of the 43 differ
  from the old V4.1.
- **Prepared rank folders:** written fresh on the first start (405 s); later starts load them.

**512K context since 2026-10-17.** Each of the 4 slots takes up to 524,288 tokens (the model's limit is 1,048,576, yarn
×16). Raising it from 420K needed no engine change, only a longer `rope.bin` (context + 2048 rows at least) and 11 more
AOT Triton variants a rank; [RUN.md](deploy/dsv41-1017/RUN.md#choosing-the-context) shows how to build both for any
context. The KV pool is shared by the slots and did not grow. 420K stays a supported setting.

| 512K on four Sparks (2026-10-17, passkey test, qbench's filler and scoring) | result | cold prompt reading |
| --- | --- | ---: |
| 500,006-token prompt | passkey correct | 101 s (~4,950 tok/s) |
| 520,006-token prompt | passkey correct | 107 s (~4,880 tok/s) |
| decode after a cached 520,000-token prefix | passkey and a count to 60 correct | 123 tokens in 1.3 s |
| 530,000-token prompt (over the limit) | clean HTTP 400, engine healthy | 2.3 s |
| `verify.py` | 7/7 | |

The engine's prompt renderer matches 1017's own `encoding.py` on every request shape the bench sends.

| 1017 on four Sparks (2026-10-17, spark-bench's frozen suite, thinking off, greedy) | 1017 | earlier pack, same engine |
| --- | ---: | ---: |
| MMLU (200) | **89.5%** | 88.5% |
| math (30) / coding (5) / instruction following (20) | 100% / 80% / 90% | 100% / 80% / 90% |
| JSON schema / tool calls / long context (25K/100K/229K-token prompts) / summaries | 100% / 100% / 100% / 100% | 100% / 100% / 100% / 100% |
| one stream, code / prose (tok/s) | 151.8 / 67.3 | **155.4 / 72.0** |
| four streams, aggregate (tok/s) | 162.4 | **172.1** |
| cold prompt, 32K / 131K / 200K tokens | 6.2 / 22.8 / **35.7 s** | – / – / 36.8 s |
| vision: text read exactly / shapes / objects counted | 166/166 · 29/30 · 22/23 | 166/166 · – · 22/23 |
| 3D agent tests, voxel / Blender (referee, of 180) | **98 / 108** | 87 / 79 |

Speed cells are single runs (earlier noise: 1–2.5 %), so the 2–7 % lower decode is probably real but small; a likely
cause is the 1017 pack's different, higher-bpw drafter (not isolated). The 3D scores are single builds scored by one
judge; the earlier pack's Blender build ran on a hosted endpoint, not here. Full tables, the harness-effect bench and
the 3D charts: [spark-bench](https://github.com/neko-legends/spark-bench#dsv41-1017).

#### Run DeepSeek-V4.1-Flash-1017 (config-only drop-in)

Step by step in **[deploy/dsv41-1017/RUN.md](deploy/dsv41-1017/RUN.md)**: download the pinned revisions and check them
against the Hugging Face tree (sha256), copy the pack to the workers, pack the Engram tables per rank, extract the image
bias, start, verify. Those are the scripts the live service ran, with its hostnames and paths turned into a settings
file ([`deploy/dsv41-1017/`](deploy/dsv41-1017)). Measured on four Sparks: MMLU 89.5 %; 151.8 / 67.3 tok/s code /
prose, 162.4 tok/s over four streams; a cold prompt of 32K / 131K / 200K tokens in 6.2 / 22.8 / 35.7 s;
`verify.py` 7/7. The template serves 512K a slot; delete its `TF_DSV41_CONTEXT` line for 420K, or build the assets for
another context as RUN.md describes.

### Results (four Sparks)

Weights: [`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw).
Configuration: [`config/prod.env.example`](config/prod.env.example) + [`config/tp4.env.example`](config/tp4.env.example)
(expert pruning on, `TF_DSV41_EXPERT_TOPP=0.85`), thinking off, temperature 0, 512-token replies, isolated runs.

#### At a glance (live: the Zig engine, 2026-10-11)

The serving build since 2026-10-11: the CUDA Zig engine at TP=4 over RoCE with pipelined prompt reading, on the
same EXL3 2.9 bpw pack. Measured with `bench/zig_vs_python.py` (window 8, salt `zvp-20261011w8`; best of 3
repetitions without a failed stream, time to first token the lowest; the prompts are 37,987 and 154,196 tokens).

| | **Zig + pipelined prefill (live)** | same build, pipeline off (same window) | previous engine (Python, RoCE) |
| --- | ---: | ---: | ---: |
| cold prompt reading, 32K / 131K (prompt tok/s) | **4,851.5 / 5,148.1** | 2,778.9 / 2,548.9 | 4,083.6 / 4,354.1 |
| cold first token, 154K-token prompt | **29.95 s** | 60.5 s | 35.4 s |
| decode after the 131K prompt (tok/s) | **99.6** | 90.7 | 79.5 |
| short code prompt, 1 / 4 streams (cold, tok/s) | **129.1 / 218.3** | 130.5 / 215.2 | 113.0 / 204.9 (cold) |
| short prose prompt, 1 / 4 streams (cold, tok/s) | **66.2 / 138.2** | 66.6 / 136.6 | 65.2 / 128.2 (cold) |

Pipeline off → on is the controlled comparison (one window, one image, the same prompts): prompt reading **+74.6 %
at 32K and +102.0 % at 131K**, first token on the 154K prompt halved, decode unchanged (±1 %). Every Python
comparison is cross-session (benched 2026-10-10, the day it was the serving build); the short cells compare **cold
against cold** (one cold repetition a side — the window measured no warm repetitions and no mix cells; the NCCL
window-7 numbers and the full table: [ZIG-RESULTS](docs/ZIG-RESULTS.md)). Two honesty notes: the mixed-window
aggregate moved 219.2 → 212.2 (−3.2 %), and the long stream's rate in that cell follows the reply's text, which
legitimately differs ([ZIG-PREFILL-PIPE](docs/ZIG-PREFILL-PIPE.md)); and a long prompt read through the pipeline
carries one rank's arithmetic rather than the four-rank reduction's, so its greedy reply differs from the
non-pipelined path's — short-prompt replies are bit-identical either way (101/101 in window 7, 29/29 in window 8).

![TP=4: TensorFold Python vs Zig vs Zig + pipelined prefill](docs/img/prefill-pipe-comparison-20261011.png)

![Every engine on four Sparks: TensorFold Python, Zig, vLLM, SGLang](docs/img/engines-dsv41-comparison-20261011.png)

#### Long context, cold (2026-10-14, live Zig engine)

One request at a time, thinking off, temperature 0, a prompt built fresh per run (nothing for the prefix cache to
reuse). Both requests below completed with no error and no engine restart:

| prompt | result | cold first token | prompt reading |
| --- | --- | ---: | ---: |
| **199,713 tokens** | completed, the question at the end answered | **36.8 s** | ~5,400 tok/s |
| 332,847 tokens (79 % of the 420,096-token request slot) | completed | 66.6 s | ~5,100 tok/s |

Both requests report `cached_tokens: 0`, so both are genuinely cold reads rather than prefix-cache hits, and the
engine's own line for the first agrees (`prompt=199713 cached=0`). The 200K run read a 12,000-line inventory and
answered the counting question at its end; that count is also readable from the document's tail, so this shows the
request completes and answers at that length rather than that every token was attended to. Probe:
[spark-bench `scripts/prefill_context_probe.py`](https://github.com/neko-legends/spark-bench/blob/main/scripts/prefill_context_probe.py).

![Cold 200K-token prompt on the live four-Spark Zig engine: 199,713 tokens, first token 36.8 s](docs/img/context-200k-20261014.webp)

#### At a glance (previous engine: Python, live 2026-10-07 → 2026-10-11)

| | four Sparks (this fork) | two Sparks (Jay's) |
| --- | ---: | ---: |
| one user, short code prompt (`m2bench`, tok/s) | **122-124** | ~80 |
| one user, short prose prompt (`m2bench`, tok/s) | **68-69** | ~41-45 |
| four users at once, steady (`m2bench` C4, tok/s) | **~200** | |
| code / prose, 1k-160k prompts (geometric mean, tok/s) | **104 / 67** | |
| cold 160k-token prompt | **36 s** | |

The build that served 2026-10-07 → 2026-10-11: Jay's G19 engine with `0003` (four Sparks), `0004` (rotated expert
split, `TF_DSV41_EXPERT_ROTATE=1`) and the 4-bit draft head (`TF_DSV41_DRAFT_HEAD=q4`), on RoCE; every reply exact
(drafted == serial). Two-Spark figures are Jay's published `m2bench` C1 numbers for his build. No `m2bench` or
1k-160k depth-sweep run exists for the Zig engine yet, so these rows end with the Python engine.

#### G19 on four Sparks (2026-10-07, previous engine)

Jay's G19 engine (engine `7bd2d67`) with this fork's `0003`, live since 2026-10-07, against the G13 build it replaced,
one boot each in the same window:

| | G13 build | **G19 build** |
| --- | ---: | ---: |
| decode, prose 1k / 20k / 160k (cold, tok/s) | 53.1 / 50.1 / 47.5 | 54.5 / 50.3 / 48.6 |
| decode, code 1k / 20k / 160k (cold, tok/s) | 88.9 / 80.8 / 84.9 | 90.7 / 82.3 / 86.2 |
| cold first token, 20k / 160k | 5.9-6.7 s / 38.9-39.1 s | **5.4-6.0 s / 35.9-36.2 s** |
| context a request slot | 300K | **420K** (KV pool 1,201,152 tokens, `TF_DSV41_POOL_TOKENS`) |
| image input | no | **yes** (`TF_DSV41_IMAGES=native`) |

Checks on the G19 build: code word at 30 / 60 / 85% of 20k / 80k / 158k-token prompts 3 / 3, and at 50% of a
~405k-token prompt 1 / 1; short gates 7 / 7; six wide sampling cases (`top_k` 40,000 and 32,000, nucleus, JSON
nucleus) 6 / 6; an image question answered; the greedy reply byte for byte the G13 build's. Same tok/s within noise:
the gain is reading prompts (~7% at 160k: G19's speed switches and fused dense prefill), the 420K context and images.

Both columns of this table ran on the NCCL fallback, not RoCE (see the note under Decode below), which is most of
why they sit below the decode sweep. The harness also differs (1-stream, 512-token replies, 2 trials). Both columns
were measured the same way, so the comparison between them holds.

#### Decode (2026-10-07, G19 on RoCE, previous engine)

| prompt | 1k | 20k | 40k | 80k | 160k | geometric mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prose, tok/s | 66.0 | 66.7 | 68.7 | 69.6 | 62.0 | **66.5** |
| code, tok/s | 107.2 | 103.5 | 105.0 | 101.7 | 103.4 | **104.1** |
| cold first token, prose / code | 0.9 / 0.7 s | 6.0 / 5.3 s | 9.3 / 9.4 s | 17.9 / 17.8 s | 35.9 / 35.8 s | |
| *2026-10-04 build (G13, `0003` alone), prose / code* | *63.6 / 104.9* | *65.7 / 97.0* | *64.0 / 96.4* | *62.8 / 103.1* | *61.0 / 99.8* | *63.4 / 100.2* |

Median of 3 a cell (the published depth sweep: 512 tokens, greedy, thinking off, trial 0 cold, isolated). Four streams
at once: **122.2 tok/s** aggregate (`dsbench`, single-stream median 63.8; 2026-10-04: 119.2 / 62.6).

**Where single-user time goes at four ranks** ([spark-bench report](https://github.com/neko-legends/spark-bench/blob/main/artifacts/tensorfold-v41-decode-levers-20261007/REPORT.md)). None of the exact exchange and L2-prefetch switches moved decode beyond boot-to-boot noise. The 4-bit draft head (`TF_DSV41_DRAFT_HEAD=q4`, now in `config/tp4.env.example`) gave +2.5% code and +1.4% prose; the sweep above ran before it. A RoCE trace puts ~2.5 ms of a decode window in ranks waiting for the slowest one. About 0.9 ms of that comes from the uneven expert split (ranks 0 and 1 hold 640 of 2,304 columns, against 512 for ranks 2 and 3). `patches/0004` (`TF_DSV41_EXPERT_ROTATE=1`, live since 2026-10-07) rotates those wide slices round the ranks by expert id: one stream unchanged, four streams +1-1.5% (C4 aggregate 137.3 -> 138.7, steady 196.9 -> 199.8 tok/s, two boots each), 16-48-row verify windows 2-8% faster.

**Check the transport after every start.** A run-time RoCE failure writes `/cache/roce-failed` in the cache volume, and
while that file exists every start serves on NCCL; the round-plan link falls back to TCP too. A crash test left one
behind on 2026-10-05, and our server ran on NCCL for two days without an error. On the same build, NCCL vs RoCE
(`m2bench`, 1 stream): code 107.2 vs **119.6** tok/s, prose 55.8 vs **66.4**, 1-row verify window 18.7 vs **16.3** ms.
Rank 0's log should say `all-gathers of up to ... over RoCE` and `plan link: rdma`. If it doesn't, look for the file,
fix the cause and move the file aside.

#### Reading a new prompt (2026-10-05, previous engine)

| cold time to first token | 20k prompt | 160k prompt |
| --- | ---: | ---: |
| `0003` alone (2026-10-04) | 12.9-14.2 s | 97.4-99.8 s |
| + split selections and overlapped exchanges | 9.4-9.5 s | 74.6-75.2 s |
| **+ pipelined prompt reading (the shipped config)** | **5.8-6.8 s** | **38.8-39.6 s** |

Each build started with an empty session cache (every prompt read cold). With the pipeline a cold 160k-token prompt
reads at ~4,100 tokens/s; before it, TP=4 read prompts ~15-25% slower than two Sparks do (this repository's two-Spark
row: 1,953 tokens/s at 128K), because every rank recomputes the parts of a layer that tensor parallelism does not
split and waits at 43 exchanges a chunk. How the pipeline works: [docs/PREFILL_SPEED.md](docs/PREFILL_SPEED.md).
Warm repeats reuse the prefix: 0.26-0.64 s at every depth.

#### Checks

- On the running server: arithmetic, forced tool call (`tool_choice=required`), tool continuation, strict JSON
  schema at T = 0 / 0.7 / 1.0, reasoning, and `scripts/canary.py` (chat, thinking, tool, json, tokenize): all pass.
- A code word hidden at 30 / 60 / 85% of 20k / 80k / 158k-token prompts: found 3 / 3 (pipelined and not).
- On the CPU (four ranks as threads, exact numerics): four ranks == one rank, the ranks agree bit for bit, a
  pipelined prompt leaves the same state as the tensor-parallel one, greedy / top-k / nucleus decoding on four ranks
  == one rank token for token (`tests/test_dsv41_tp4.py`, `test_dsv41_pipe.py`, `test_dsv41_four_sparks.py`).

Raw files: [`results/four-sparks-20261004/`](results/four-sparks-20261004/README.md),
[`results/prefill-speed-20261005/`](results/prefill-speed-20261005/README.md),
[`results/four-spark-fixes-20261005/`](results/four-spark-fixes-20261005/README.md). The same cluster's earlier deployments
(SGLang on the FP8 checkpoint, vLLM) and a dated history of every change: [spark-bench](https://github.com/neko-legends/spark-bench).

### What four Sparks change

| part | two Sparks (a rank) | four Sparks (a rank) |
| --- | --- | --- |
| query heads | 32 | 16 |
| routed + shared experts (intermediate 2,304) | 1,152 | 640 / 640 / 512 / 512 |
| vocabulary (head columns, embedding rows) | 64,640 | 32,384 / 32,384 / 32,256 / 32,256 |
| Engram hash heads | 12 | 6 (`engram-l{1,14}-r{rank}of4.bin`) |
| resident weights | ~99 GiB | ~56 GiB (+~27 GiB for the prompt pipeline's full-width layers) |

Every split stays on whole 128-wide Hadamard blocks (EXL3 rotations act per block), so the widths that do not divide
by four are uneven; every exchange sends the same size from every rank. Details: [docs/FOUR_SPARKS.md](docs/FOUR_SPARKS.md).

### Quick start

Requirements: four DGX Sparks on one switch, one IPv4 subnet per CX7 port function, Docker with the NVIDIA runtime on
each, passwordless ssh from the head (rank 0) to the three workers over the link; on each node's local NVMe ~200 GB
for the weights, ~47 GB for its Engram shards, ~50 GB for the prepared rank folder, and room for the session tier.
Nothing else on the GPUs.

```bash
git clone --recurse-submodules https://github.com/neko-legends/tensorfold-4x-sparks.git
cd tensorfold-4x-sparks
cp config/prod.env.example config/prod.env        # the engine knobs
cp config/tp4.env.example config/tp4.env          # every <placeholder>: workers, link addresses, paths

# 1. weights: the same EXL3 pack on all four nodes, same path
hf download dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw --local-dir <MODEL>

# 2. Engram shards, rank r on node r (from DeepSeek's original checkpoint: docs/FOUR_SPARKS.md)
python3 scripts/pack_engram.py --src <deepseek-ai/DeepSeek-V4.1-Flash dir> --config <MODEL>/config.json \
    --out <ENGRAM> --world 4 --rank <r>

# 3. the image (TensorFold v0.6.0 + patches/*), shipped to the workers, CUDA extensions prebuilt on all four
docker build -f docker/Dockerfile -t dsv41-tensorfold:tp4 .
bash scripts/serve4.sh ship
bash scripts/serve4.sh prebuild

# 4. serve (the first start writes the prepared rank folders; later starts load in ~35 s)
bash scripts/serve4.sh start
bash scripts/serve4.sh status | logs [R] | stop
```

Then:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "DeepSeek-V4.1-Flash-TF",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}], "reasoning_effort": "low"}'
```

Optional: `scripts/keeper4.sh` from cron restarts the four ranks after a reboot or three failed health checks.
Setup notes and lessons from the first boots: [docs/FOUR_SPARKS.md](docs/FOUR_SPARKS.md). Knobs and memory gates:
[docs/OPERATIONS.md](docs/OPERATIONS.md).

### API

OpenAI-compatible on `HOST:PORT` (`127.0.0.1:8000` by default: put your own proxy and authentication in front):
`/v1/chat/completions` (streaming, tool calls, `response_format`), `/v1/completions` (text or token ids),
`/tokenize`, `/v1/models`, `/health`, `/metrics` (TensorFold's own series and, from `0003`, the vLLM-named ones
fleet dashboards read), `/v1/model_info` (`max_num_seqs`: the request slots). Thinking follows DeepSeek-V4.1's encoding and is on by default
(`TF_DSV41_THINKING=0` turns it off); `reasoning_effort` `none` / `low` / `medium` / `high` / `max` or 1-100, or
`chat_template_kwargs: {"thinking": false}` for one request. `logprobs` + `top_logprobs` (0-20) return OpenAI's
`choices[0].logprobs.content` for non-streamed text chat with thinking off (from `patches/0006`; a request in any
other mode gets a 400 that names the field, as do `logit_bias` / `prompt_logprobs` / `echo`). The
full table: [docs/TWO_SPARKS.md#api](docs/TWO_SPARKS.md#api) (unchanged at four Sparks).

### The engine

`vendor/TensorFold` is upstream TensorFold v0.6.0, unmodified; the Dockerfile applies `patches/` in order:

| patch | from | what |
| --- | --- | --- |
| [`0001-spark-stack-060.patch`](patches/0001-spark-stack-060.patch) | jayleaton | the GLM-5.3-Flash Spark engine stack rebased onto 0.6.0: communicator, RoCE all-gather, server pieces |
| [`0002-deepseek-v41-family.patch`](patches/0002-deepseek-v41-family.patch) | jayleaton | the DeepSeek-V4.1-Flash family (engine G19, `7bd2d67`) |
| [`0003-four-sparks.patch`](patches/0003-four-sparks.patch) | this fork | four ranks (TP=4): whole-block uneven splits, `--tp 4 --rank 0..3`, N-way rank agreement, plan link (TCP and RDMA), fail-fast and memory floor; the fixes from Jay's review; pipelined prompt reading, split selections and overlapped exchanges (opt-in, on in `tp4.env.example`); `/health` draft counters; sync-free expert counts; vLLM-named `/metrics` series and `/v1/model_info` |
| [`0004-expert-rotate.patch`](patches/0004-expert-rotate.patch) | this fork | `TF_DSV41_EXPERT_ROTATE=1`: each expert's wide 128-column slices rotate round the four ranks by expert id instead of always landing on ranks 0 and 1 (`x3ld` reads an expert's own width); exact, +1-1.5% at four streams |
| [`0005-drafter-fidelity.patch`](patches/0005-drafter-fidelity.patch) | this fork | drafter-training tools: the capture logs each pass's candidates and noise flag; the trainer's drafter reads the engine's q4 head and bf16 mHC weights (port agreement 0.72 -> 0.90); `m2bench` / `draftcap --world` |
| [`0006-openai-logprobs.patch`](patches/0006-openai-logprobs.patch) | this fork | `logprobs` + `top_logprobs` (0-20) on `/v1/chat/completions`, non-streamed: OpenAI's `choices[0].logprobs.content`, exact over the whole vocabulary at TP=4 (each rank's log-sum-exp parts ride the candidates' all-gather); unsupported fields (`logit_bias`, `prompt_logprobs`, `echo`, streaming, thinking on, tools, stop, structured output) are 400s that name the field. Tests, rollout and rollback: [`docs/LOGPROBS-ROLLOUT.md`](docs/LOGPROBS-ROLLOUT.md) |

Until 2026-10-07 the fork ran on G13 with these as patches 0003-0006 (branch `four-sparks-g13`). Jay's PR #6 asks for
a smaller `0003` (four ranks and the review's fixes only, two-Spark output unchanged): that version is the `four-sparks`
branch, verified byte-identical at TP=2 against his main.

#### Fixed 2026-10-05 (then patch 0005, now in 0003)

Jay reviewed the TP=4 port ([PR #6](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/pull/6)) and found
bugs that only uneven slices or more than two ranks expose. All fixed, with tests in `tests/test_dsv41_four_sparks.py`
(each review bug's test fails without its fix), and checked on the four Sparks (2026-10-05; serving since 11:30):

- the four benchmark replies (20k and 160k, prose and code) are byte-identical to the 0004 build's; cold first token
  5.8-6.8 s (20k) and 38.8-39.0 s (160k), as before; decode the same as 0004 in an A/B in one window;
- the hidden code word found 3 / 3, the short gates 7 / 7;
- sampled requests past the narrow slices pass: nucleus rows, nucleus with a JSON schema (full-vocabulary
  candidates), `top_k` 32,000 and 40,000, including `top_k` 40,000 as the first request after a boot.

The fixes:

- **Candidates past the narrow vocabulary slices**: every rank took `min(count, its own width)` candidates, so a
  count above 32,256 (`top_k` above that, or masked nucleus rows, which ask for the whole vocabulary) all-gathered
  unequal sizes. On the four Sparks, the 0004 build answered a `top_k` 40,000 request with an illegal memory access
  on ranks 2 and 3 (the narrow slices) and the server stopped. Every rank now sends the widest slice's count,
  narrower slices padded with entries that sort after every real token (`pick.rank_top`).
- **Trimmed draft head** (`TF_DSV41_DRAFT_HEAD=trim`): the per-rank id lists assumed equal slices; rank 2 refused to
  boot. They now follow the 128-block split.
- **TCP plan link** (`TF_DSV41_PLAN_LINK=tcp`): followers now say their rank and world size; rank 0 refuses strays,
  duplicates and another world size, orders the connections by rank, and names a rank that never connects.
- **`--tp 4`** is refused for families that do not declare it (`CUDA_TP`), instead of starting an engine that takes
  neither path.
- **`TF_DSV41_PREFILL_ATTN_BMQ=32`** (the two-Spark production value) on a 16-head rank now runs the fused prefill
  attention with 16 heads a program instead of silently falling back to the chunk kernels.
- **NVMe session tier**: the world size is part of the directory's identity, so an entry written by two Sparks is
  never resumed by four.
- **The first very wide window** (found while testing the above on the four Sparks): with the gather fixed, a
  sampled request with `top_k` 32,000-40,000 as the server's first wide request still timed out every follower's
  Engram gate at layer 14 (4 of 4 runs); runs whose first wide request was a JSON nucleus request passed. Most
  likely cause: the window needed the large-`k` top-k and large-row sort kernels for the first time, and CUDA's lazy
  module loading loaded them while the forward still waited on host threads (the Engram gate, the RoCE exchanges).
  The candidate kernels of every width class now run once at boot on every rank (`cand_warm`), and the candidates'
  pinned buffer grows before a window's forward is queued (`cand_reserve`): since then the same requests pass in
  every run (2 of 2, one with `top_k` 40,000 as the first request after a boot). Not fully explained: the 0004 build
  once answered a first `top_k` 32,000 request without the warm-up.
- The gate scorer and DSpark delta shards (both off in production) now handle any rank count and uneven splits.

### How it was tested (2026-10-05)

*This section describes the 2026-10-05 G13 build (patches 0001-0006, branch `four-sparks-g13`). The G19 build's checks are
in [G19 on four Sparks](#g19-on-four-sparks-2026-10-07-previous-engine). Its patches rebuild the same way. Its CPU suites pass except the
three tests that also fail on Jay's main (they need his unpublished draft vocabulary or a network this sandbox lacks)
and two that the fork's extra prompt-reading code trips (below, What is not solved).*

Every number and claim on this page was checked these ways before `main` moved to it.

**1. The patches rebuild the engine exactly.** TensorFold v0.6.0 (`vendor/TensorFold`) with `patches/0001`-`0006`
applied in order gives the same tree, file for file, as the engine branch the work was done on:

```bash
cp -r vendor/TensorFold /tmp/tf && cd /tmp/tf && rm -rf .git
for p in "$OLDPWD"/patches/*.patch; do git apply --whitespace=nowarn "$p"; done
```

**2. The CPU test suites pass on that tree.** No GPU needed: the tests run the engine's CPU twin, with four ranks as
threads where it matters (`pip install torch numpy safetensors pytest`; run each file in its own process, as some keep
large fixtures):

```bash
cd /tmp/tf
for f in tests/test_dsv41_*.py tests/test_cuda_cli.py; do python -m pytest -q "$f"; done
```

On the final tree, 23 files (every DeepSeek V4.1 suite that the TP=4 patches touch, and the CLI suites): **327
passed, 2 failed**. Both failures are older than this fork's patches and unrelated to four Sparks:
`test_cuda_cli.py::test_serve_parses_the_kv_cache_flag` (an upstream test that still expects `--kv-dtype fp8` to be
refused; `patches/0002` adds it) and `test_dsv41_draft_head.py::test_shipped_ranking` (needs the unpublished draft
vocabulary file). The four-Spark tests:

| file | what it checks | tests |
| --- | --- | ---: |
| `test_dsv41_tp4.py` (0003) | uneven 128-block splits; four ranks == one rank in exact numerics; ranks agree bit for bit; row invariance; greedy over uneven vocabulary slices; Engram from `of4` shards | 8 |
| `test_dsv41_overlap.py`, `test_dsv41_pipe.py` (0004) | split selections and overlapped exchanges == one segment; a pipelined prompt leaves the tensor-parallel state, at 2 and 4 ranks, through a mid-prompt snapshot, over point-to-point | 12 |
| `test_dsv41_four_sparks.py` (0005) | every fix in the list above; run without the fixes, each review bug's test fails | 20 |

The thread communicator in these tests refuses all-gathers whose ranks send different sizes, as NCCL requires, so a
mismatch fails a test instead of hanging a Spark.

**3. The image builds from a fresh clone.** `git clone --recurse-submodules` of this repository, then
`docker build -f docker/Dockerfile .` exactly as in [Quick start](#quick-start), shipped to the three workers with
`scripts/serve4.sh ship`: that image is the one serving on our four Sparks.

**4. On the four Sparks, before it went live.** Each candidate build ran in a test window (the live server stopped,
the build started with an empty session cache, the checks, the live server restored), and it was promoted only if
every check passed:

- the depth bench's 20k and 160k prompts, prose and code, read cold: same first-token times and byte-identical
  replies to the previous build;
- a code word hidden at 30 / 60 / 85% of 20k / 80k / 158k-token prompts: 3 / 3;
- the short gates (arithmetic, forced tool call, tool continuation, strict JSON at three temperatures, reasoning):
  7 / 7;
- sampled requests past the narrow slices: nucleus, nucleus with min_p, `top_k` 40,000 and 32,000, `top_k` 20, a
  JSON schema with nucleus sampling: 6 / 6, and `top_k` 40,000 as the first request after a fresh boot;
- decode against the previous build, four boots alternating in one window: the same within noise.

Three candidate builds failed the wide-request check before the shipped one passed; they never served. The previous
build was also tested on the same wide request: it crashed ranks 2 and 3. All of it, run by run:
[`results/four-spark-fixes-20261005/`](results/four-spark-fixes-20261005/README.md).

### What is not solved

- Two CPU tests fail on the G19 build's full `0003` (not on PR #6's smaller one): `test_dsv41_calib_knobs`
  (the prompt-reading switches are not yet in the calibration key or its exempt list) and `test_dsv41_pdl`'s
  forward-hook count (the overlapped-exchange refactor of the layer loop). Neither changes what the server computes.
- The boot memory budget (`memory.load_check`) is anchored on two-Spark measurements: conservative on ranks 0 and 1,
  skipped on ranks 2 and 3. The live floor is MemAvailable ~26 GB a node with the pipeline on.
- A pipelined prompt's bits are one rank's arithmetic, not the four-rank sum's (its own session tag): as different
  as the engine's own fast vs exact prefill kernels. Exact numerics give the same state either way.
- Short prompts gain less from the pipeline (it fills in four steps): cold 20k is 5.4-6.0 s.
- One cluster, one boot a configuration; expert pruning is lossy (delete its three lines for the unpruned model).

## Shared engine notes

**TP-4 over RoCE.** Both models run one rank per Spark, the four ranks exchanging activations over the CX7 link with
TensorFold's RoCE collectives (`TF_COMM_BACKEND=roce`; NCCL is the fallback, `TF_TP_ROCE_FALLBACK=nccl`). Rank 0
serves HTTP and plans every step; the other ranks follow it over a plan link. Check the transport after every start:
a rank that falls back to NCCL serves correctly but slower (DeepSeek's note under
[Decode](#decode-2026-10-07-g19-on-roce-previous-engine)).

**The prompt cache under TP, and its disk tier (Nemotron).** Every rank keeps its own share of each cached prompt
state; rank 0 decides what to keep and drop and the others mirror it, so the copies stay keyed the same on all four.
Prompts are cut at keep marks and resume points so a returning conversation restarts from the longest cached prefix.
States evicted from memory, and every state at a clean stop, go to a disk tier: one folder per rank, named by an
identity that hashes the checkpoint (config and every shard's name, size, header and tail), the layout, the rank and
the world, so another model or another split never reads them. One server holds a folder at a time (a lock file);
rank 0 removes files its index does not name at start and the other ranks follow. Caps: `TF_NEMOTRON_DISK_GIB` (128),
`TF_NEMOTRON_DISK_FLOOR_GIB` (free space kept, 64), `TF_NEMOTRON_DISK_MIN` (smallest state written, 4,096 tokens). Stop
rank 0 first so the tier is written. Under TP, greedy text depends on the cache state at rounding level (deterministic
per state), so speed and exactness A/B tests pin it ([WINDOW.md](engine/zig-nemotron/tools/nemotron/swap/WINDOW.md)).
DeepSeek's equivalent is its NVMe session tier ([OPERATIONS](docs/OPERATIONS.md)).

**Staggered prompt segments.** A long prompt is read in 4,096-row chunks; with `--segments 4` up to four chunks run
in one call as staggered segments, their collectives on one high-priority communication stream and posted in parts
so one segment computes while another exchanges. NCCL's default spreads a prompt all-reduce over 64 channel blocks of
the GB10's 48 SMs and starves the other segments, so the live config caps it at two (`NCCL_MAX_NCHANNELS=2`; the
vision footnote under [Head to head](#head-to-head) is the trade). With the segments, the cap and two prefill kernel
changes, a cold 200K-token prompt went from 79.9 s to 48.3 s (32K: 11.4 → 6.5 s). DeepSeek reads long prompts with
its own pipeline: one rank per layer group, described in [ZIG-PREFILL-PIPE](docs/ZIG-PREFILL-PIPE.md).

## Documentation

| | |
| --- | --- |
| [engine/zig-nemotron](engine/zig-nemotron/README.md) | TensorFold 1.0.11 with the Nemotron TP-4 port; [port log](engine/zig-nemotron/NEMOTRON-PORT-LOG.md), [serving notes](engine/zig-nemotron/tools/nemotron/swap/WINDOW.md), [results](engine/zig-nemotron/results/) |
| [FOUR_SPARKS](docs/FOUR_SPARKS.md) | the TP=4 port: what changes, setup, first-boot lessons, results |
| [PREFILL_SPEED](docs/PREFILL_SPEED.md) | the prompt pipeline, split selections, overlapped exchanges, what did not work |
| [ZIG-SERVE](docs/ZIG-SERVE.md) / [ZIG-RESULTS](docs/ZIG-RESULTS.md) | Jay's Zig engine: build, TP=2 launch, exact settings, q28-v2 comparisons; the TP=4 results of the live Zig engine (windows 7-8) |
| [ZIG-PREFILL-PIPE](docs/ZIG-PREFILL-PIPE.md) | the Zig pipeline's design, proofs and window pass criteria |
| [ZIG-TP4-PLAN](docs/ZIG-TP4-PLAN.md) | what the Zig engine needed for four Sparks (done: serving since 2026-10-11), and the like-for-like bench recipe |
| [INSTALL](docs/INSTALL.md) | requirements, weights, Engram shards, build, start, watchdog |
| [API](docs/API.md) / [IMAGES](docs/IMAGES.md) | endpoints, thinking and effort, image input |
| [OPERATIONS](docs/OPERATIONS.md) | settings, memory gates, turning each lever off |
| [TWO_SPARKS](docs/TWO_SPARKS.md) | Jay's two-Spark README at the fork point: method, quality tables, strict mode, credits |
| [ENGINE](docs/ENGINE.md) / [ARCHITECTURE](docs/ARCHITECTURE.md) / [DECODE](docs/DECODE.md) | the patches, the model split, the decode roofline |
| [campaign/](docs/campaign/README.md) | Jay's development log, windows G1-G13 |

## Licensing

Apache License 2.0 for this project's own code, patches, scripts, benchmarks and docs ([`LICENSE`](LICENSE),
[`NOTICE`](NOTICE)): keep the copyright lines and the NOTICE attributions and state your changes. This fork's changes
are listed in [`NOTICE`](NOTICE). TensorFold (`vendor/TensorFold`, unmodified) is Apache-2.0 from 0.6.0; the code the
patches modify stays under its license. Files the patches add keep their own SPDX notice (the DeepSeek-V4.1-Flash
family: MIT, Copyright (c) 2026 Jay Leaton). Third-party code and the model weights: the full table in
[docs/TWO_SPARKS.md#licensing](docs/TWO_SPARKS.md#licensing). The uncensored weights have refusals removed; you are
responsible for how you use them.

The Nemotron engine ([`engine/zig-nemotron`](engine/zig-nemotron)) is TensorFold 1.0.11, Apache-2.0 ([its NOTICE](engine/zig-nemotron/NOTICE),
[third-party notices](engine/zig-nemotron/THIRD_PARTY_NOTICES.md)); the Nemotron weights are NVIDIA's, under their own license.

## Credits

- **[Ash Hart (ashhart) / TensorFold](https://github.com/ashhart/TensorFold)**: the engine both models run on, the Zig
  server, the CUDA and EXL3 kernels, the drafting, and the Nemotron family this repository extends to four Sparks.
- **[Jay Leaton (jayleaton)](https://github.com/jayleaton)**: the DeepSeek-V4.1-Flash engine for TensorFold and the
  two-Spark recipe this fork extends, and the review that found the bugs 0005 fixes. Follow him on
  [X](https://x.com/jayleaton) / support him on [Buy Me a Coffee](https://buymeacoffee.com/jayleaton).
- [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash): the model, its encoding, and the CED / CSA2 /
  mHC / DSpark / Engram designs.
- [dealignai](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw) (the uncensored pack),
  [Mia-AiLab](https://huggingface.co/Mia-AiLab) (the 2.9 bpw EXL3 pack), [turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3)
  (EXL3), and everyone credited in [docs/TWO_SPARKS.md#credits](docs/TWO_SPARKS.md#credits).
- [NVIDIA](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4): Nemotron 3.5 Super VL and its
  NVFP4 checkpoint.

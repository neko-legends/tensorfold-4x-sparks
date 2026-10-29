#!/usr/bin/env python3
"""Write README.md (model card) and NEKO_BUILD.json into the assembled Neko pack on forge.
Run after assemble + verify. sha256s are streamed (200 GB read)."""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path("/home/jun/models")
NEW = ROOT / "neko-legends-DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED-4xSpark"
STAGE = ROOT / "neko-stage"
metrics = json.load(open(STAGE / "metrics.json"))
vpath = STAGE / "encoded" / "metrics_viterbi_mix.json"
viterbi = {m["module"]: m for m in json.load(open(vpath))} if vpath.exists() else {}

by = {m["module"]: m for m in metrics}
pilot = [by["layers.20.attn.wo_b"], by["layers.20.ffn.shared_experts.w2"]]
pilot_rows = "\n".join(
    f"| {m['module']} | {m['K']} | {m['rel_err_new']:.4f} | {m['edit_retention']:.3f} | {m['edit_cosine']:.3f} |"
    for m in pilot)
rets = [m["edit_retention"] for m in metrics]
rels = [m["rel_err_new"] for m in metrics]
wo = [m for m in metrics if m["module"].endswith("wo_b")]
w2 = [m for m in metrics if m["module"].endswith("w2")]
summary = (
    f"all 54 modules retention {min(rets):.3f}–{max(rets):.3f} (median "
    f"{sorted(rets)[27]:.3f}, every module ≥ 0.96); rel. error vs the dealignai BF16 weights "
    f"{min(rels):.4f}–{max(rels):.4f} (wo_b, K=5: {min(m['rel_err_new'] for m in wo):.4f}–"
    f"{max(m['rel_err_new'] for m in wo):.4f}; shared_experts.w2: K=4 modules "
    f"{max(m['rel_err_new'] for m in w2):.4f}, K=5 modules ≤ {max(m['rel_err_new'] for m in w2 if m['K']==5):.4f}). "
    "The full per-module table is in `NEKO_BUILD.json` (`pilot_metrics`/`modules`)."
)

readme = Path("/home/jun/neko-build/README.template.md").read_text() if Path("/home/jun/neko-build/README.template.md").exists() \
    else (Path(__file__).parent / "README.template.md").read_text()
readme = readme.replace("PILOT_TABLE", pilot_rows).replace("RETENTION_SUMMARY", summary)
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
    if p.is_file():
        shas[str(p.relative_to(NEW))] = sha256(p)
print(f"hashed {len(shas)} files in {time.time()-t0:.0f}s")

build = {
    "pack": "neko-legends/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED-4xSpark",
    "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "source": {
        "repo": "dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED",
        "revision": "89af9bf193575b7eeeb4f1dd85463ec24ea5bb4c",
        "local_dir": "/home/jun/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED",
    },
    "base": {
        "repo": "LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw",
        "revision": "afc7aa0a",
        "hf_org_on_card": "LibertAI",
        "local_dir": "/home/jun/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw",
    },
    "re_encoded_modules": [m["module"] for m in metrics],
    "dropped_tensors": [
        "layers.{1,14}.engram.{embed,wkv}.{weight,scale} (F8_E4M3, not carried by the base pack; "
        "TensorFold serves Engram from the deepseek-ai 1017 source via scripts/pack_engram.py)",
    ],
    "method": (
        "Per module: invert the EXL3 input transform with the base pack's fixed suh/svh "
        "(W_r = H128(H128(W/svh)/suh)), then edit-aware LDLQ trellis quantization "
        "(exllamav3 ldlq, mul1 codebook, K per base pack) with Hessian "
        "H = D_reg D_reg^T/n + 0.1*(||D_reg||^2/(k*n))*I where D_reg = W_r - dequant(base trellis) "
        "is the edit in regularized space; the row-wise error compensation drives the quantization "
        "error orthogonal to the edit direction. Scales suh/svh and the mul1 marker are copied from "
        "the base pack unchanged. Encoder: exllamav3 (1.6.2-era tree, CUDA ext rebuilt locally, "
        "quantize_tiles/block_ldl/ldlq/pack_trellis/reconstruct). Pilot inversion self-test: "
        "re-encoding dequant(base) reproduces base's trellis exactly (rel 0.000000)."
    ),
    "codec_params": {"codebook": "mul1", "mul1_multiplier": 2212286765,
                     "K": {"layers.*.attn.wo_b": 5,
                           "layers.10.ffn.shared_experts.w2": 5,
                           "layers.1x.ffn.shared_experts.w2 (11-28)": 4,
                           "layers.29-36.ffn.shared_experts.w2": 5},
                     "lambda_edit_hessian": 0.1,
                     "out_scales": "base pack suh/svh reused unchanged"},
    "pilot_metrics": pilot,
    "modules": metrics,
    "viterbi_baseline_metrics": {m["module"]: {"edit_retention": v["edit_retention"],
                                               "rel_err_new": v["rel_err_new"]}
                                 for m, v in ((m, viterbi[m["module"]]) for m in metrics)
                                 if m["module"] in viterbi},
    "notes": [
        "54 modules = layers.10..36 attn.wo_b + ffn.shared_experts.w2 (abliterated by dealignai, shipped dense BF16)",
        "16 untouched shards hard-linked from the dealignai dir (sha256-equal to base); 27 shards rebuilt byte-copies of base with only the trellis regions replaced",
        "quantization_config.json / model.safetensors.index.json / config.json / tokenizer / exl3_k_map.json / encoding/ copied verbatim from the base pack",
        "Edit retention = <dequant(new) - dequant(base), W_uncens - dequant(base)> / ||W_uncens - dequant(base)||^2",
    ],
    "files_sha256": shas,
}
(NEW / "NEKO_BUILD.json").write_text(json.dumps(build, indent=1))
print("NEKO_BUILD.json written")

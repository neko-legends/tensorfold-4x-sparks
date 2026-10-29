#!/usr/bin/env python3
"""Assemble the Neko Legends 4xSpark pack on forge.

Layout mirrors the BASE pack exactly (43 shards, identical headers): for shards that
carry none of the 54 re-encoded trellis tensors we hard-link the byte-identical source
shard (uncens preferred when sha256-equal to base, else base); touched shards are copied
from base and only the 54 trellis data regions are overwritten in place (same dtype/shape
=> same byte length => header offsets unchanged). Aux files from base; new README.md and
NEKO_BUILD.json are written by the orchestrator (not this script).
"""
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path("/home/jun/models")
UNC = ROOT / "DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED"
BASE = ROOT / "DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw"
NEW = ROOT / "neko-legends-DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED-4xSpark"
TRELLIS_ST = Path(sys.argv[1])  # local dir containing all-trellis.safetensors
REPORT = {}

MODULES = []
for l in range(10, 37):
    MODULES += [f"layers.{l}.attn.wo_b", f"layers.{l}.ffn.shared_experts.w2"]


def read_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    base = 8 + n
    out = {}
    for nm, t in hdr.items():
        if nm == "__metadata__":
            continue
        off, end = t["data_offsets"]
        with open(path, "rb") as f:
            f.seek(base + off)
            out[nm] = f.read(end - off)
    return out


def shard_headers(root):
    idx = json.load(open(root / "model.safetensors.index.json"))["weight_map"]
    per = {}
    for nm, sh in idx.items():
        per.setdefault(sh, {})[nm] = None
    out = {}
    for sh in sorted(per):
        with open(root / sh, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        hdr.pop("__metadata__", None)
        out[sh] = (8 + n, hdr)
    return out


def sha256(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


NEW.mkdir(parents=True, exist_ok=True)
new_trellis = read_st(TRELLIS_ST / "all-trellis.safetensors")
assert len(new_trellis) == 54, len(new_trellis)
for m in MODULES:
    t = new_trellis[m + ".trellis"]
    assert len(t) % 2 == 0

base_h = shard_headers(BASE)
report = {"linked_from_uncens": [], "linked_from_base": [], "rewritten": []}

for sh, (hlen, hdr) in base_h.items():
    mod_trellis = {}
    for m in MODULES:
        nm = m + ".trellis"
        if nm in hdr:
            mod_trellis[nm] = hdr[nm]
    if not mod_trellis:
        # untouched shard: hard-link from uncens if byte-identical to base, else from base
        su, sb = sha256(UNC / sh), sha256(BASE / sh)
        src = UNC / sh if su == sb else BASE / sh
        report["linked_from_uncens" if src == UNC / sh else "linked_from_base"].append(sh)
        os.link(src, NEW / sh)
        continue
    # touched shard: copy base shard, overwrite trellis regions
    dst = NEW / sh
    shutil.copyfile(BASE / sh, dst)
    with open(dst, "r+b") as f:
        for nm, ent in mod_trellis.items():
            off, end = ent["data_offsets"]
            data = new_trellis[nm]
            assert end - off == len(data), (nm, end - off, len(data))
            f.seek(hlen + off)
            f.write(data)
    report["rewritten"].append((sh, sorted(mod_trellis)))

# aux files from base
for f in ["model.safetensors.index.json", "quantization_config.json", "config.json",
          "tokenizer.json", "tokenizer_config.json", "exl3_k_map.json", "LICENSE"]:
    shutil.copyfile(BASE / f, NEW / f)
shutil.copytree(BASE / "encoding", NEW / "encoding", dirs_exist_ok=True)
if (BASE / "assets").is_dir():
    shutil.copytree(BASE / "assets", NEW / "assets", dirs_exist_ok=True)

json.dump(report, open(NEW / ".neko_assemble_report.json", "w"), indent=1)
print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in report.items()}, indent=1))
print("assembled:", NEW)

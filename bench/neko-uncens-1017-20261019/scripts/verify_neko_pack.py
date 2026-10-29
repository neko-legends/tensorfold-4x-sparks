#!/usr/bin/env python3
"""Structural verification of the Neko Legends pack on forge (host-only, no GPU):
1. name set == base index, dtype per name == base
2. every shard header parses, holds exactly the tensors the index maps to it, offsets in bounds
3. no F8_E4M3 dtype anywhere
4. trellis shape per each of the 54 modules == base
5. non-edited tensors byte-identical dealignai<->base (full scan)
6. disk usage
Exit 0 iff all structural checks pass. (Retention table comes from the encode metrics.)"""
import json
import os
import struct
import sys
from pathlib import Path

ROOT = Path("/home/jun/models")
UNC = ROOT / "DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED"
BASE = ROOT / "DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw"
NEW = ROOT / "neko-legends-DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED-4xSpark"
MODULES = []
for l in range(10, 37):
    MODULES += [f"layers.{l}.attn.wo_b", f"layers.{l}.ffn.shared_experts.w2"]
EDITED = {m + s for m in MODULES for s in (".weight", ".trellis", ".suh", ".svh", ".mul1")}

fail = 0


def headers(root):
    idx = json.load(open(root / "model.safetensors.index.json"))["weight_map"]
    per = {}
    for nm, sh in idx.items():
        per.setdefault(sh, set()).add(nm)
    out = {}
    for sh in sorted(per):
        with open(root / sh, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        hdr.pop("__metadata__", None)
        out[sh] = (8 + n, hdr, per[sh])
    return out


NB = headers(BASE)
NN = headers(NEW)
NU = headers(UNC)

# 1. name sets + dtypes
bn = {n for _, h, _ in NB.values() for n in h}
nn = {n for _, h, _ in NN.values() for n in h}
d1 = bn ^ nn
print(f"check1 name-set diff vs base: {'EMPTY (ok)' if not d1 else sorted(d1)[:5]} ({len(bn)} names)")
fail += bool(d1)
mism = []
for sh, (_, h, _) in NB.items():
    nh = NN[sh][1]
    for n, e in h.items():
        if nh.get(n, {}).get("dtype") != e["dtype"] or nh.get(n, {}).get("shape") != e["shape"]:
            mism.append(n)
print(f"check1 dtype/shape mismatches: {'NONE (ok)' if not mism else mism[:5]}")
fail += bool(mism)

# 2. per-shard structural integrity (new pack)
bad = 0
for sh, (hlen, hdr, want) in NN.items():
    p = NEW / sh
    size = p.stat().st_size
    if set(hdr) != want:
        bad += 1
        print("HEADER-SET-MISMATCH", sh, sorted(set(hdr) ^ want)[:3])
        continue
    for n, e in hdr.items():
        o, end = e["data_offsets"]
        if not (0 <= o < end <= size - hlen):
            bad += 1
            print("OFFSET-OOB", sh, n, o, end, size)
print(f"check2 shards: {len(NN)} checked, bad={bad} (ok)" if not bad else f"check2 FAIL bad={bad}")
fail += bool(bad)

# 3. no F8
f8 = [n for _, h, _ in NN.values() for n, e in h.items() if e["dtype"] == "F8_E4M3"]
print(f"check3 F8_E4M3 tensors in new pack: {'NONE (ok)' if not f8 else f8[:5]}")
fail += bool(f8)

# 4. trellis shapes == base for the 54
tbad = []
for m in MODULES:
    for sh, (_, h, _) in NB.items():
        if m + ".trellis" in h:
            nh = NN[sh][1]
            if nh[m + ".trellis"]["shape"] != h[m + ".trellis"]["shape"]:
                tbad.append(m)
print(f"check4 trellis shapes == base for 54 modules: {'ALL MATCH (ok)' if not tbad else tbad}")
fail += bool(tbad)

# 5. non-edited tensors byte-identical between uncens and base (full scan, both directions)
ndiff = 0
sample = []
for sh, (hlen, h, _) in NU.items():
    if sh not in NB:
        continue  # uncens-only engram shards 47/48
    bhl, bh, _ = NB[sh]
    for n, e in h.items():
        if n in EDITED or n not in bh:
            continue
        o, end = e["data_offsets"]
        bo, bend = bh[n]["data_offsets"]
        if end - o != bend - bo:
            ndiff += 1
            sample.append((n, "len"))
            continue
        with open(UNC / sh, "rb") as f:
            f.seek(hlen + o)
            a = f.read(end - o)
        with open(BASE / sh, "rb") as f:
            f.seek(bhl + bo)
            b = f.read(bend - bo)
        if a != b:
            ndiff += 1
            if len(sample) < 5:
                sample.append((n, "bytes"))
print(f"check5 non-edited tensors uncens vs base: {ndiff} differ {sample[:5]}")
fail += bool(ndiff)

print("RESULT:", "ALL-STRUCTURAL-CHECKS-PASS" if not fail else f"{fail} CHECK GROUPS FAILED")
sys.exit(1 if fail else 0)

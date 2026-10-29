#!/usr/bin/env python3
"""Extract all 54 abliterated modules (uncens dense BF16 weight + base trellis/suh/svh/mul1)
from the two forge packs into per-module safetensors under neko-stage/extract-all/.
Read-only on the source packs; streams one tensor at a time (~100 MB peak RSS)."""
import json, struct
from pathlib import Path

ROOT = Path("/home/jun/models")
UNC = ROOT / "DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED"
BASE = ROOT / "DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw"
OUT = ROOT / "neko-stage" / "extract-all"
OUT.mkdir(parents=True, exist_ok=True)

def headers(root):
    idx = json.load(open(root / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for sh in sorted(set(idx.values())):
        with open(root / sh, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for nm, t in hdr.items():
            if nm != "__metadata__":
                out[nm] = (root / sh, 8 + n, t["dtype"], t["shape"], *t["data_offsets"])
    return out

UH, BH = headers(UNC), headers(BASE)

def read(H, name, root):
    p, hl, dt, shape, o, e = H[name]
    with open(p, "rb") as f:
        f.seek(hl + o)
        return f.read(e - o), dt, shape

def write(dst, entries, root):
    hdr, off, blobs = {"__metadata__": {"format": "pt"}}, 0, []
    for nm in entries:
        raw, dt, shape = read(UH if root == "u" else BH, nm, root)
        hdr[nm] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        blobs.append(raw)
    hb = json.dumps(hdr).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)

n = 0
for l in range(10, 37):
    for kind in ("attn.wo_b", "ffn.shared_experts.w2"):
        m = f"layers.{l}.{kind}"
        slab = m.replace(".", "-")
        write(OUT / f"uncens-{slab}.safetensors", [m + ".weight"], "u")
        write(OUT / f"base-{slab}.safetensors",
              [m + s for s in (".trellis", ".suh", ".svh", ".mul1")], "b")
        n += 1
print("extracted", n, "modules to", OUT)

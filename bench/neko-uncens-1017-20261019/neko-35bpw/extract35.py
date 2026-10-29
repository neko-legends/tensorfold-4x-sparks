#!/usr/bin/env python3
"""Extract the 54 abliterated modules' tensors for the 3.5bpw Neko build:
- dense BF16 .weight from the dealignai 3.5bpw UNCENSORED pack (encode target)
- .trellis/.suh/.svh/.mul1 from the 2.9bpw Neko pack (edit base + fixed scales + marker)
Read-only on both source packs; streams one tensor at a time (~200 MB peak RSS)."""
import json, struct
from pathlib import Path

ROOT = Path("/home/jun/models")
SRC = ROOT / "dealign-3.5bpw-uncens"
NEKO29 = ROOT / "neko-legends-DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED-4xSpark"
OUT = ROOT / "neko35-stage" / "extract35"
OUT.mkdir(parents=True, exist_ok=True)

def headers(root):
    idx = json.load(open(root / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for sh in sorted(set(idx.values())):
        p = root / sh
        if not p.exists():
            continue
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for nm, t in hdr.items():
            if nm != "__metadata__":
                out[nm] = (p, 8 + n, t["dtype"], t["shape"], *t["data_offsets"])
    return out

SH, NH = headers(SRC), headers(NEKO29)

def read(H, name):
    p, hl, dt, shape, o, e = H[name]
    with open(p, "rb") as f:
        f.seek(hl + o)
        return f.read(e - o), dt, shape

def write(dst, entries):
    hdr, off, blobs = {"__metadata__": {"format": "pt"}}, 0, []
    for nm, H in entries:
        raw, dt, shape = read(H, nm)
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
        for nm, Hname in ((m + ".weight", SH), (m + ".trellis", NH), (m + ".suh", NH),
                          (m + ".svh", NH), (m + ".mul1", NH)):
            assert nm in Hname, nm
        write(OUT / f"{slab}.safetensors",
              [(m + ".weight", SH), (m + ".trellis", NH), (m + ".suh", NH),
               (m + ".svh", NH), (m + ".mul1", NH)])
        n += 1
print("extracted", n, "modules to", OUT)

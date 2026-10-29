#!/usr/bin/env python3
"""Assemble the Neko Legends 3.5bpw pack on forge from the dealignai 3.5 source + K=6 re-encodes.

Layout: the 17 shards carrying none of the 54 edited modules are hard-linked from the source dir
(byte-identical); the 27 shards carrying at least one are REWRITTEN (the dense BF16 .weight is dropped,
every other tensor streams through byte-identical in source order, and .trellis/.suh/.svh/.mul1 are
appended per edited module). The index = source index - 54 .weight entries + 216 new entries.
Aux files copy verbatim from source; the upstream README is kept as README.source-dealign.md.
Never touches the 2.9 pack or the source (source is only read + hard-linked)."""
import json, os, shutil, struct, sys
from pathlib import Path

ROOT = Path("/home/jun/models")
SRC = ROOT / "dealign-3.5bpw-uncens"
NEKO29 = ROOT / "neko-legends-DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED-4xSpark"
NEW = ROOT / "neko-legends-DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-UNCENSORED-4xSpark"
ENC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "neko35-stage" / "encoded35" / "all-trellis35.safetensors"
CHUNK = 32 << 20

MODULES = [f"layers.{l}.{k}" for l in range(10, 37) for k in ("attn.wo_b", "ffn.shared_experts.w2")]
DROP = {m + ".weight" for m in MODULES}
ADD = {m + s for m in MODULES for s in (".trellis", ".suh", ".svh", ".mul1")}

def read_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    base = 8 + n
    out = {}
    for nm, t in hdr.items():
        if nm == "__metadata__":
            continue
        o, e = t["data_offsets"]
        with open(path, "rb") as f:
            f.seek(base + o)
            out[nm] = (f.read(e - o), t["dtype"], t["shape"])
    return out

NEW_T = read_st(ENC)
assert len(NEW_T) == 4 * len(MODULES) == 216, len(NEW_T)

idx = json.load(open(SRC / "model.safetensors.index.json"))
wm = idx["weight_map"]
per = {}
for nm, sh in wm.items():
    per.setdefault(sh, []).append(nm)

src_headers = {}
for sh in sorted(per):
    with open(SRC / sh, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    src_headers[sh] = (8 + n, hdr)

NEW.mkdir(parents=True, exist_ok=True)
report = {"linked": [], "rewritten": []}
new_wm = {}

for sh in sorted(per):
    hlen, hdr = src_headers[sh]
    meta = hdr.get("__metadata__", {"format": "pt"})
    names = [nm for nm in hdr if nm != "__metadata__"]
    edited_here = [m for m in MODULES if (m + ".weight") in hdr]
    if not edited_here:
        dst = NEW / sh
        dst.exists() and dst.unlink()
        os.link(SRC / sh, dst)
        report["linked"].append(sh)
        for nm in names:
            new_wm[nm] = sh
        continue
    # sanity: this shard's dropped and added sets
    for m in edited_here:
        assert (m + ".weight") in hdr, m
        for s in (".trellis", ".suh", ".svh", ".mul1"):
            assert (m + s) in NEW_T, m + s
    kept = [nm for nm in names if nm not in DROP]
    assert set(nm for nm in names if nm in DROP) == {m + ".weight" for m in edited_here}
    # build new header
    nhdr, off = {"__metadata__": meta}, 0
    blobs = []  # (src_offset_abs, nbytes) or raw bytes
    with open(SRC / sh, "rb") as sf:
        for nm in kept:
            o, e = hdr[nm]["data_offsets"]
            nb = e - o
            nhdr[nm] = {"dtype": hdr[nm]["dtype"], "shape": hdr[nm]["shape"],
                        "data_offsets": [off, off + nb]}
            blobs.append((hlen + o, nb, None))
            off += nb
        for m in edited_here:
            for s in (".trellis", ".suh", ".svh", ".mul1"):
                nm = m + s
                raw, dt, shape = NEW_T[nm]
                nhdr[nm] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(raw)]}
                blobs.append((None, len(raw), raw))
                off += len(raw)
    hb = json.dumps(nhdr).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    tmp = NEW / (sh + ".tmp")
    with open(tmp, "wb") as df:
        df.write(struct.pack("<Q", len(hb)))
        df.write(hb)
        for src_abs, nb, raw in blobs:
            if raw is not None:
                df.write(raw)
                continue
            sf.seek(src_abs)
            left = nb
            while left:
                b = sf.read(min(CHUNK, left))
                assert b, (sh, src_abs, nb, left)
                df.write(b)
                left -= len(b)
    tmp.rename(NEW / sh)
    report["rewritten"].append((sh, edited_here))
    for nm in kept:
        new_wm[nm] = sh
    for m in edited_here:
        for s in (".trellis", ".suh", ".svh", ".mul1"):
            new_wm[m + s] = sh

assert set(new_wm) == (set(wm) - DROP) | ADD, "index name set mismatch"
out_idx = {"metadata": idx.get("metadata", {}), "weight_map": dict(sorted(new_wm.items()))}
(NEW / "model.safetensors.index.json").write_text(json.dumps(out_idx, indent=4, sort_keys=False))

# aux files verbatim from source
for f in ["config.json", "exl3_k_map.json", "tokenizer.json", "tokenizer_config.json",
          "quantization_config.json", "LICENSE", ".gitattributes",
          "dealign_logo.png", "dealign_mascot.png", "vmlx-app.png"]:
    shutil.copyfile(SRC / f, NEW / f)
shutil.copytree(SRC / "encoding", NEW / "encoding", dirs_exist_ok=True)
shutil.copytree(SRC / "assets", NEW / "assets", dirs_exist_ok=True)
shutil.copyfile(SRC / "README.md", NEW / "README.source-dealign.md")

json.dump(report, open(NEW / ".neko35_assemble_report.json", "w"), indent=1)
print(json.dumps({"linked": len(report["linked"]), "rewritten": len(report["rewritten"])}))
print("assembled:", NEW)

#!/usr/bin/env python3
"""Re-encode the 54 dealignai 3.5bpw abliterated modules (dense BF16) to EXL3 trellis K=6 mul1,
reusing the 2.9bpw Neko pack's fixed suh/svh scales and mul1 marker (same shapes, same codebook family),
with the edit-aware LDLQ Hessian built around the 2.9 pack's own trellis (K=5/4 for those modules):

  W_r      = invert_regularize(W_3.5, suh_29, svh_29)                  [regularized space]
  D_reg    = W_r - dequant_reg(trellis_29, K29)                        [the 3.5 edit in reg space]
  H        = D_reg D_reg^T / n + lam * (||D_reg||^2 / (k*n)) * I ;  L = block_ldl(H, 16)
  trellis6 = pack_trellis(ldlq(W_r, L, K=6, mul1))

Retention gate (handoff, W-space): delta = W_3.5 - dequant(trellis_29, K29);
  retention = <dequant(trellis6) - dequant(trellis_29), delta> / ||delta||^2,  gate min >= 0.95 over all 54.

Run with the live engine STOPPED (~2.5 GiB GPU). Env: /home/jun/exl3-venv (torch 2.11 cu128, exllamav3 ext).
"""
import argparse, json, struct, sys, time
from pathlib import Path
import torch
from exllamav3.modules.quant.exl3_lib import quantize as qz
from exllamav3.ext import exllamav3_ext as ext

TORCH_DT = {"F16": torch.float16, "F32": torch.float32, "I16": torch.int16, "I32": torch.int32, "BF16": torch.bfloat16}
NEW_K = 6
LAM = 0.1

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
            raw = f.read(end - off)
        out[nm] = torch.frombuffer(bytearray(raw), dtype=TORCH_DT[t["dtype"]]).reshape(t["shape"]).clone()
    return out

def write_st(path, tensors):
    header = {"__metadata__": {"format": "pt"}}
    off = 0
    rev = {v: k for k, v in TORCH_DT.items()}
    for nm, t in tensors.items():
        nb = t.numel() * t.element_size()
        header[nm] = {"dtype": rev[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + nb]}
        off += nb
    hb = json.dumps(header).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for t in tensors.values():
            c = t.contiguous()
            if c.dim() == 0:
                c = c.reshape(1)  # 0-dim (mul1 scalar): same bytes, header keeps shape []
            f.write(c.view(torch.uint8).numpy().tobytes())
    tmp.rename(path)

def invert_regularize(w_kn, suh, svh):
    x = w_kn / svh.unsqueeze(0)
    x = qz.preapply_had_r(x, 128)
    x = x / suh.unsqueeze(1)
    return qz.preapply_had_l(x, 128)

def dequant_reg(trellis, K, k, n, device):
    w_hat = torch.empty((k, n), dtype=torch.half, device=device)
    ext.reconstruct(w_hat, trellis.contiguous(), K, False, True)
    return w_hat.float()

def dequant(trellis, suh, svh, K, device):
    kt, nt, _ = trellis.shape
    k, n = kt * 16, nt * 16
    t = qz.preapply_had_l(dequant_reg(trellis, K, k, n, device), 128) * suh.unsqueeze(1)
    return qz.preapply_had_r(t, 128) * svh.unsqueeze(0)

def encode_ldlq(w_kn, suh, svh, bt, K29, device, lam=LAM):
    k, n = w_kn.shape
    w_r = invert_regularize(w_kn.to(device), suh.to(device), svh.to(device))
    qa = {"K": NEW_K, "devices": [str(device)], "buf_size_k": 128, "mul1": True}
    x_base = dequant_reg(bt.to(device), K29, k, n, device)
    D_reg = w_r - x_base
    dnorm2 = float(D_reg.pow(2).sum())
    H = D_reg @ D_reg.T / n + lam * (dnorm2 / (k * n)) * torch.eye(k, device=device)
    t0 = time.time()
    L, _ = qz.block_ldl(H, 16, qa, False)
    weight_q, encoded = qz.ldlq(w_r.contiguous(), L, qa)
    enc_s = time.time() - t0
    trellis = qz.pack_trellis(encoded, qa)
    in_rel = float((weight_q - w_r).norm() / w_r.norm())
    return trellis.cpu(), enc_s, in_rel

def encode_fresh(w_kn, suh, svh, device):
    """From-scratch fallback: plain uncalibrated quantization of W_r at K=6 (reused scales)."""
    k, n = w_kn.shape
    w_r = invert_regularize(w_kn.to(device), suh.to(device), svh.to(device))
    qa = {"K": NEW_K, "devices": [str(device)], "buf_size_k": 128, "mul1": True}
    t0 = time.time()
    weight_q, encoded, mse = qz.fallback_quant(w_r.contiguous(), device, qa)
    enc_s = time.time() - t0
    trellis = qz.pack_trellis(encoded, qa)
    in_rel = float((weight_q - w_r).norm() / w_r.norm())
    return trellis.cpu(), enc_s, in_rel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", default="/home/jun/models/neko35-stage/extract35")
    ap.add_argument("--out", default="/home/jun/models/neko35-stage/encoded35")
    ap.add_argument("--modules", default="pilot", help="'pilot', 'all', or comma list")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fresh", action="store_true", help="plain K=6 quantization, no edit Hessian (fallback path)")
    args = ap.parse_args()
    device = torch.device(args.device)
    exdir, outdir = Path(args.extract), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.modules == "pilot":
        mods = ["layers.20.attn.wo_b", "layers.20.ffn.shared_experts.w2"]
    elif args.modules == "all":
        mods = [f"layers.{l}.{k}" for l in range(10, 37) for k in ("attn.wo_b", "ffn.shared_experts.w2")]
    else:
        mods = args.modules.split(",")

    metrics_path = outdir / "metrics.json"
    results, new_tensors = [], {}
    for m in mods:
        t = read_st(exdir / f"{m.replace('.', '-')}.safetensors")
        W = t[m + ".weight"]                      # [out, in] bf16
        w_kn = W.t().float().contiguous()         # [in, out]
        suh = t[m + ".suh"].float()
        svh = t[m + ".svh"].float()
        suh_raw, svh_raw, mul1_raw = t[m + ".suh"], t[m + ".svh"], t[m + ".mul1"]
        bt = t[m + ".trellis"]
        K29 = bt.shape[-1] // 16
        mul1 = int(mul1_raw)
        assert mul1 & 0xFFFFFFFF == 2212286765, hex(mul1)
        assert suh.shape[0] == w_kn.shape[0] and svh.shape[0] == w_kn.shape[1]
        print(f"[{m}] K29={K29} W{tuple(w_kn.shape)} bt{tuple(bt.shape)} -> K=6", flush=True)
        t0 = time.time()
        if args.fresh:
            new_t, enc_s, in_rel = encode_fresh(w_kn, suh, svh, device)
        else:
            new_t, enc_s, in_rel = encode_ldlq(w_kn, suh, svh, bt, K29, device)
        assert tuple(new_t.shape) == (bt.shape[0], bt.shape[1], 16 * NEW_K), new_t.shape
        assert new_t.dtype == torch.int16
        D_base = dequant(bt.to(device), suh.to(device), svh.to(device), K29, device)
        D_new = dequant(new_t.to(device), suh.to(device), svh.to(device), NEW_K, device)
        Wu = w_kn.to(device)
        delta = Wu - D_base
        step = D_new - D_base
        dn2 = float(delta.pow(2).sum())
        mm = {
            "module": m, "K29": K29, "K_new": NEW_K,
            "trellis_shape": list(new_t.shape),
            "rel_err_new": round(float((D_new - Wu).norm() / Wu.norm()), 6),
            "rel_err_29pack_vs_35dense": round(float((D_base - Wu).norm() / Wu.norm()), 6),
            "edit_retention": round(float((delta * step).sum() / dn2), 6),
            "edit_cosine": round(float((delta * step).sum() / (delta.norm() * step.norm() + 1e-30)), 6),
            "delta_rel": round(float(delta.norm() / Wu.norm()), 6),
            "in_space_rel_err": round(in_rel, 6),
            "encode_s": round(enc_s, 1),
            "method": "fresh_k6" if args.fresh else "edit_aware_ldlq_k6",
            "wall_s": round(time.time() - t0, 1),
        }
        results.append(mm)
        print(json.dumps(mm), flush=True)
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=1)
        new_tensors[m + ".trellis"] = new_t
        new_tensors[m + ".suh"] = suh_raw
        new_tensors[m + ".svh"] = svh_raw
        new_tensors[m + ".mul1"] = mul1_raw
        del D_base, D_new, delta, step, Wu
        torch.cuda.empty_cache()

    name = "pilot-trellis35.safetensors" if args.modules == "pilot" else "all-trellis35.safetensors"
    p = outdir / name
    p.exists() and p.unlink()
    write_st(p, new_tensors)
    rets = [r["edit_retention"] for r in results]
    print(f"wrote {p} | retention min {min(rets):.4f} mean {sum(rets)/len(rets):.4f} max {max(rets):.4f}")
    if min(rets) < 0.95:
        print("GATE-FAIL: min retention < 0.95")

if __name__ == "__main__":
    main()

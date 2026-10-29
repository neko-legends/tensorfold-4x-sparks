#!/usr/bin/env python3
"""Aggregate perf-sweep jsonl files: median decode tok/s per (label, case, effort) and prefill tok/s per (label, size)."""
import json, glob, os, statistics as st, sys
d = os.path.expanduser("~/neko-mmlu-bench/perf-sweep")
rows = {}
for f in sorted(glob.glob(d + "/*.jsonl")):
    for line in open(f):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("kind") == "decode" and r.get("decode_tok_s"):
            rows.setdefault((r["label"], r["case"], r.get("effort", "?"), r.get("temp", 0)), []).append(r["decode_tok_s"])
        elif r.get("kind") == "prefill" and r.get("prefill_tok_s") and r.get("rep", 0) >= 0:
            rows.setdefault((r["label"], "PREFILL_" + r["case"].split("_")[1], "-", 0), []).append(r["prefill_tok_s"])
print(f"{'label':<28}{'case':<14}{'eff':<9}{'n':>2}{'median':>9}{'spread%':>9}")
for k in sorted(rows):
    v = rows[k]
    med = st.median(v)
    spread = (max(v) - min(v)) / med * 100 if len(v) > 1 else 0
    print(f"{k[0]:<28}{k[1]:<14}{str(k[2]):<9}{len(v):>2}{med:>9.1f}{spread:>8.1f}%")

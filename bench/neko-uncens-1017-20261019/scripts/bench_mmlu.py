#!/usr/bin/env python3
"""MMLU-500 bench, dealignai method, against forge:8000 uncensored 1017 pack.
- 500 questions, stratified: category totals fixed (STEM 171, Hum 113, Soc 103, Other 113),
  within category allocated ~proportionally to subject test-set size (largest remainder).
- 5-shot: 5 dev-split exemplars from the same subject, standard MMLU prompt.
- Raw /v1/completions ("5-shot generate", literal card wording): no chat template, no thinking.
- Parse first [ABCD] from content; no match = wrong (invalid).
"""
import json, re, time, hashlib, threading, queue
from collections import defaultdict
import pandas as pd
import requests

API = "http://localhost:8000/v1/completions"
MODEL = "deepseek-v4.1-flash"
SEED = 20261019
CATEGORY_TOTALS = {"stem": 171, "humanities": 113, "social": 103, "other": 113}
CONCURRENCY = 4

CATEGORIES = {
 "stem": ["abstract_algebra","anatomy","astronomy","college_biology","college_chemistry",
          "college_computer_science","college_mathematics","college_physics","computer_security",
          "conceptual_physics","electrical_engineering","elementary_mathematics","formal_logic",
          "high_school_biology","high_school_chemistry","high_school_computer_science",
          "high_school_mathematics","high_school_physics","high_school_statistics","machine_learning"],
 "humanities": ["high_school_european_history","high_school_us_history","high_school_world_history",
          "international_law","jurisprudence","logical_fallacies","moral_disputes","moral_scenarios",
          "philosophy","prehistory","professional_law","world_religions"],
 "social": ["econometrics","high_school_geography","high_school_government_and_politics",
          "high_school_macroeconomics","high_school_microeconomics","high_school_psychology",
          "human_sexuality","professional_psychology","public_relations","security_studies",
          "sociology","us_foreign_policy"],
 "other": ["business_ethics","clinical_knowledge","college_medicine","global_facts","human_aging",
          "management","marketing","medical_genetics","miscellaneous","nutrition",
          "professional_accounting","professional_medicine","virology"],
}
LETTERS = ["A","B","C","D"]

def fmt_example(q, choices, answer=None):
    s = q.strip()
    for j, c in enumerate(choices):
        s += f"\n{LETTERS[j]}. {c}"
    s += "\nAnswer:"
    if answer is not None:
        s += f" {LETTERS[answer]}"
    return s

def build_prompt(dev_rows, q, choices, subject):
    subj_pretty = subject.replace("_", " ")
    parts = [f"The following are multiple choice questions (with answers) about {subj_pretty}.\n"]
    for r in dev_rows:
        parts.append(fmt_example(r["question"], r["choices"], r["answer"]) + "\n")
    parts.append(fmt_example(q, choices, None))
    return "\n".join(parts)

def main():
    test = pd.read_parquet("all-test.parquet")
    dev = pd.read_parquet("all-dev.parquet")
    assert test["subject"].nunique() == 57 and dev["subject"].nunique() == 57

    subj2cat = {s: c for c, ss in CATEGORIES.items() for s in ss}
    assert set(subj2cat) == set(test["subject"].unique()), "category map mismatch"

    cat_sizes = defaultdict(lambda: defaultdict(int))
    for s in test["subject"]:
        cat_sizes[subj2cat[s]][s] = int((test["subject"] == s).sum())

    # largest-remainder allocation within each category to hit CATEGORY_TOTALS exactly
    alloc = {}
    for cat, total in CATEGORY_TOTALS.items():
        subs = cat_sizes[cat]
        tot = sum(subs.values())
        raw = {s: total * n / tot for s, n in subs.items()}
        base = {s: int(v) for s, v in raw.items()}
        rem = total - sum(base.values())
        order = sorted(raw, key=lambda s: (-(raw[s] - base[s]), s))
        for s in order[:rem]:
            base[s] += 1
        alloc.update(base)
    assert sum(alloc.values()) == 500
    assert all(sum(v for s, v in alloc.items() if subj2cat[s] == c) == t for c, t in CATEGORY_TOTALS.items())

    rng = np_random = __import__("numpy").random.default_rng(SEED)
    selected = []
    for s in sorted(alloc):
        idxs = test.index[test["subject"] == s].to_numpy()
        pick = rng.choice(idxs, size=alloc[s], replace=False)
        pick = sorted(pick)  # preserve dataset order
        for i in pick:
            row = test.loc[i]
            selected.append({"subject": s, "category": subj2cat[s], "question": row["question"],
                             "choices": list(row["choices"]), "answer": int(row["answer"]), "orig_index": int(i)})
    assert len(selected) == 500

    qhash = hashlib.sha256(json.dumps([{k: r[k] for k in ("subject","question","answer","orig_index")} for r in selected],
                                       sort_keys=True).encode()).hexdigest()

    dev_by_subj = {s: [ {"question": r["question"], "choices": list(r["choices"]), "answer": int(r["answer"])}
                        for _, r in dev[dev["subject"] == s].iterrows()] for s in sorted(alloc)}

    results = [None] * len(selected)
    lock = threading.Lock()
    done = [0]
    q = queue.Queue()
    for i, item in enumerate(selected):
        q.put(i)

    session = requests.Session()
    def worker():
        while True:
            try: i = q.get_nowait()
            except queue.Empty: return
            item = selected[i]
            prompt = build_prompt(dev_by_subj[item["subject"]], item["question"], item["choices"], item["subject"])
            body = {"model": MODEL, "prompt": prompt, "max_tokens": 8, "temperature": 0}
            ans, raw, err = None, None, None
            for attempt in range(3):
                try:
                    r = session.post(API, json=body, timeout=180)
                    r.raise_for_status()
                    m = r.json()["choices"][0]
                    raw = m.get("text") or ""
                    mm = re.search(r"[ABCD]", raw)
                    ans = mm.group(0) if mm else None
                    err = None
                    break
                except Exception as e:
                    err = str(e)[:200]; time.sleep(2 * (attempt + 1))
            with lock:
                results[i] = {"subject": item["subject"], "category": item["category"],
                              "gold": LETTERS[item["answer"]], "pred": ans, "raw": (raw or "")[:60],
                              "error": err}
                done[0] += 1
                if done[0] % 50 == 0: print(f"  {done[0]}/500", flush=True)

    t0 = time.time()
    threads = [threading.Thread(target=worker) for _ in range(CONCURRENCY)]
    [t.start() for t in threads]; [t.join() for t in threads]
    wall = time.time() - t0

    # grade
    per_cat = defaultdict(lambda: {"n": 0, "correct": 0, "invalid": 0})
    per_subj = defaultdict(lambda: {"n": 0, "correct": 0})
    overall = {"n": 0, "correct": 0, "invalid": 0, "errors": 0}
    for r in results:
        if r is None or r["error"]: overall["errors"] += 1; continue
        c = r["category"]
        per_cat[c]["n"] += 1; overall["n"] += 1
        per_subj[r["subject"]]["n"] += 1
        if r["pred"] is None:
            per_cat[c]["invalid"] += 1; overall["invalid"] += 1
        elif r["pred"] == r["gold"]:
            per_cat[c]["correct"] += 1; overall["correct"] += 1; per_subj[r["subject"]]["correct"] += 1

    out = {
        "config": {"api": API, "model_served": MODEL, "seed": SEED, "concurrency": CONCURRENCY,
                   "endpoint": "/v1/completions (raw generate, no chat template)", "max_tokens": 8, "temperature": 0,
                   "category_totals": CATEGORY_TOTALS, "alloc_per_subject": alloc,
                   "question_set_sha256": qhash, "wall_seconds": round(wall, 1)},
        "overall": {**overall, "accuracy_pct": round(100 * overall["correct"] / overall["n"], 2)},
        "categories": {c: {**v, "accuracy_pct": round(100 * v["correct"] / v["n"], 2) if v["n"] else None} for c, v in per_cat.items()},
        "per_subject": {s: {**v, "accuracy_pct": round(100 * v["correct"] / v["n"], 2) if v["n"] else None} for s, v in sorted(per_subj.items())},
        "details": results,
    }
    with open("results.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("overall", "categories")}, indent=2))
    print("wall:", round(wall / 60, 1), "min; qhash:", qhash[:16])
    print("errors:", overall["errors"], "invalid-parse:", overall["invalid"])

if __name__ == "__main__":
    main()

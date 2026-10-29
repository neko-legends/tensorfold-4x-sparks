#!/usr/bin/env python3
"""Charts for the three-way model matrix (2026-10-19).

Sources: nest/tmp/worker-handoffs/model-matrix-20261019.REPORT.md
Harness note: 'B · chat' is the only harness all three models could run, so it carries the
cross-model comparisons. 'A · completions' is local-only (Venice has no /v1/completions).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/jun/nest/tmp/charts/model-matrix-20261019"
os.makedirs(OUT, exist_ok=True)

MODELS = ["local 2.9 bpw", "local 3.5 bpw", "Venice-hosted 1017"]
C = ["#4C8BF5", "#F5A623", "#7ED321"]  # blue / amber / green

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#444444", "axes.labelcolor": "#222222",
    "text.color": "#222222", "xtick.color": "#222222", "ytick.color": "#222222",
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(p)


# ---------------------------------------------------------------- chart 1: MMLU
def chart_mmlu():
    cats = ["STEM", "Humanities", "Social Sci.", "Other", "Overall"]
    b = {
        "local 2.9 bpw": [77.19, 68.14, 87.38, 87.61, 79.6],
        "local 3.5 bpw": [77.19, 72.57, 87.38, 89.38, 81.0],
        "Venice-hosted 1017": [83.63, 84.96, 92.23, 87.61, 86.6],
    }
    a = {"local 2.9 bpw": [80.70, 71.68, 89.32, 84.07, 81.2],
         "local 3.5 bpw": [82.46, 71.68, 85.44, 86.73, 81.6]}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [3.1, 1]})
    x = np.arange(len(cats))
    w = 0.26
    for i, m in enumerate(MODELS):
        vals = b[m]
        ax.bar(x + (i - 1) * w, vals, w, label=m, color=C[i], edgecolor="white", linewidth=0.6)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + 1.6, f"{v:.1f}", ha="center", va="bottom", fontsize=8.2, color="#333333")
    ax.set_xticks(x, cats)
    ax.set_ylim(55, 102)
    ax.set_ylabel("accuracy %")
    ax.set_title("MMLU-500 — chat harness (all three models)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, fontsize=9, frameon=False)
    ax.text(0.99, 0.02, "raw-completions harness (local only): 81.2 / 81.6 %", transform=ax.transAxes,
            ha="right", fontsize=8.2, color="#555555")

    moral = [22.73, 22.73, 95.45]
    moral_a = [40.91, 45.45, None]
    xm = np.arange(3)
    bars = ax2.bar(xm, moral, 0.55, color=C, edgecolor="white", linewidth=0.6)
    for xi, v in zip(xm, moral):
        ax2.text(xi, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.plot([0, 1], moral_a[:2], "o", ms=7, mfc="white", mec="#4C8BF5", mew=2, zorder=6)
    ax2.plot([1], [moral_a[1]], "o", ms=7, mfc="white", mec="#F5A623", mew=2, zorder=6)
    ax2.set_xticks(xm, ["2.9", "3.5", "Venice"], fontsize=10)
    ax2.set_ylim(0, 108)
    ax2.set_title("moral_scenarios (n=22)")
    ax2.set_ylabel("accuracy %")
    ax2.text(0.5, 52, "both local packs\nanswer 'D' 22/22\nunder the chat harness",
             ha="center", fontsize=8.2, color="#B00020")
    fig.suptitle("Capability: local abliterated packs vs the official safety-tuned model (hosted)", fontsize=14, fontweight="bold", y=1.02)
    save(fig, "01-mmlu.png")


# ------------------------------------------------------------ chart 2: HumanEval
def chart_humaneval():
    cats = ["passed", "assert fail", "syntax", "other runtime", "empty"]
    colors = ["#2E7D32", "#C62828", "#EF6C00", "#6A1B9A", "#455A64"]
    hb = {  # chat harness, all three
        "local 2.9 bpw": [152, 6, 3, 2, 1],
        "local 3.5 bpw": [153, 10, 0, 1, 0],
        "Venice-hosted 1017": [149, 14, 0, 1, 0],
    }
    ha = {  # raw completions, local only
        "local 2.9 bpw": [126, 30, 1, 7, 0],
        "local 3.5 bpw": [125, 33, 3, 3, 0],
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1.5, 1]})
    for ax, data, title in (
        (axes[0], hb, "chat harness — all three models"),
        (axes[1], ha, "raw-completions harness — local only"),
    ):
        labels = list(data.keys())
        bottom = np.zeros(len(labels))
        for ci, cat in enumerate(cats):
            vals = np.array([data[m][ci] for m in labels], dtype=float)
            ax.bar(labels, vals, 0.55, bottom=bottom, color=colors[ci], label=cat,
                   edgecolor="white", linewidth=0.6)
            for i, (v, bt) in enumerate(zip(vals, bottom)):
                if v >= 6:
                    ax.text(i, bt + v / 2, f"{int(v)}", ha="center", va="center", fontsize=8.5,
                            color="white" if ci == 0 else "white", fontweight="bold")
            bottom += vals
        for i, m in enumerate(labels):
            ax.text(i, bottom[i] + 2.5, f"{100 * data[m][0] / 164:.2f}% pass@1", ha="center",
                    fontsize=10, fontweight="bold")
        ax.set_ylim(0, 185)
        ax.set_ylabel("HumanEval problems (of 164)")
        ax.set_title(title)
        ax.legend(fontsize=8.5, loc="upper center", ncol=3, framealpha=0.95)
        ax.tick_params(axis="x", labelsize=9.5)
    fig.suptitle("Coding: HumanEval-164, greedy, sandboxed official tests", fontsize=14, fontweight="bold", y=1.02)
    save(fig, "02-humaneval.png")


# --------------------------------------------------------------- chart 3: decode
def chart_decode():
    cases = ["short chat", "2k code", "8k prose"]
    low = {"local 2.9 bpw": [69.3, 127.7, 63.9],
           "local 3.5 bpw": [62.2, 116.0, 56.5],
           "Venice-hosted 1017": [67.6, 398.8, 206.7]}
    high = {"local 2.9 bpw": [71.7, 91.0, 63.0],
            "local 3.5 bpw": [60.5, 101.0, 55.3],
            "Venice-hosted 1017": [95.0, 414.9, 242.5]}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=True)
    for ax, data, title in ((axes[0], low, "effort low"), (axes[1], high, "effort high")):
        x = np.arange(len(cases))
        w = 0.26
        for i, m in enumerate(MODELS):
            vals = data[m]
            bars = ax.bar(x + (i - 1) * w, vals, w, label=m, color=C[i], edgecolor="white", linewidth=0.6)
            for xi, v in zip(x + (i - 1) * w, vals):
                ax.text(xi, v + 6, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x, cases)
        ax.set_title(title)
        ax.set_ylabel("decode tok/s (median, single stream)")
        ax.set_ylim(0, 470)
    axes[1].legend(fontsize=9, loc="upper left", framealpha=0.95)
    fig.suptitle("Decode speed — local TP4 vs hosted (hosted includes their batching; not a hardware test)",
                 fontsize=13.5, fontweight="bold", y=1.02)
    save(fig, "03-decode.png")


# -------------------------------------------------------------- chart 4: prefill
def chart_prefill():
    labels = ["2.9 · pipeline on\n(live config)", "2.9 · pipeline off", "3.5 · pipeline off", "Venice · implied"]
    tok = [5033, 2861, 2780, 15538]
    ttft = [4.184, 7.360, 7.576, 1.355]
    colors = [C[0], "#9DBDF7", C[1], C[2]]
    hatch = ["", "", "", "///"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.9))
    bars = ax.bar(labels, tok, 0.55, color=colors, edgecolor="white", linewidth=0.6)
    for b, h in zip(bars, hatch):
        if h:
            b.set_hatch(h)
    for i, v in enumerate(tok):
        ax.text(i, v + 250, f"{v:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 18000)
    ax.set_ylabel("prefill tok/s (21,058-token prompt)")
    ax.set_title("Prefill throughput")
    ax.tick_params(axis="x", labelsize=8.6)
    ax.text(3, 7000, "implied from hosted TTFT\n(their queue + network included)\nnot a measured rate",
            ha="center", fontsize=8.2, color="#2E7D32")

    bars2 = ax2.bar(labels, ttft, 0.55, color=colors, edgecolor="white", linewidth=0.6)
    for b, h in zip(bars2, hatch):
        if h:
            b.set_hatch(h)
    for i, v in enumerate(ttft):
        ax2.text(i, v + 0.18, f"{v:.2f} s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("time to first token (s)")
    ax2.set_ylim(0, 9)
    ax2.set_title("Time to first token, same prompt")
    ax2.tick_params(axis="x", labelsize=8.6)
    fig.suptitle("Prefill — the 3.5 pack cannot afford the pipeline, which is the whole gap",
                 fontsize=13.5, fontweight="bold", y=1.02)
    save(fig, "04-prefill.png")


if __name__ == "__main__":
    chart_mmlu()
    chart_humaneval()
    chart_decode()
    chart_prefill()

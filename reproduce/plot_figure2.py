"""Render our reproduction of paper Figure 2 (Gemma2-9B 131k on AdvBench).

Reads results/summary.csv (written by orchestrate.py) and produces a
side-by-side bar chart matching the paper figure layout:
  - Left axis: population concept overlap (Before vs After) for
    Population-Targeted and Population-Untargeted.
  - Right axis: individual attack success rate for
    Individual-Targeted (Activation) and Individual-Untargeted (Deactivation).

Paper baseline numbers (Gemma2-9B 131k AdvBench, from Figure 2 text):
  Pop  Targeted:   27.8 -> 53.8
  Pop  Untargeted: 100.0 -> 21.3
  Ind  Activation success rate:    65.3 %
  Ind  Deactivation success rate:  87.1 %
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAPER_FIG2 = {
    "pop_tgt_before": 27.8,
    "pop_tgt_after":  53.8,
    "pop_untgt_before": 100.0,
    "pop_untgt_after":  21.3,
    "ind_act_asr":   65.3,
    "ind_deact_asr": 87.1,
}


def load_summary(path: Path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                r["returncode"] = int(r.get("returncode") or 0)
            except ValueError:
                r["returncode"] = -1
            for k in ("overlap_change", "initial_overlap", "success_rate"):
                v = r.get(k)
                r[k] = float(v) if v not in ("", None) else None
            r["targeted"] = r["targeted"] in ("True", "true", True)
            r["activate"] = r["activate"] in ("True", "true", True)
            rows.append(r)
    return [r for r in rows if r["returncode"] == 0]


def aggregate(rows):
    # Average across samples and across suffix+replacement modes (paper averages n+1 trials).
    g = defaultdict(list)
    for r in rows:
        key = (r["level"], r["targeted"], r["activate"])
        g[key].append(r)
    out = {}
    for k, lst in g.items():
        level, tgt, act = k
        if level == "population":
            initial = [x["initial_overlap"] for x in lst if x["initial_overlap"] is not None]
            change  = [x["overlap_change"]  for x in lst if x["overlap_change"]  is not None]
            if not change:
                continue
            mean_initial = float(np.mean(initial)) if initial else None
            mean_change  = float(np.mean(change))
            # overlap_change = (after - before) / before  =>  after = before * (1 + change)
            mean_after = (mean_initial or 0) * (1 + mean_change) if mean_initial is not None else None
            out[k] = {"n": len(lst), "initial": mean_initial, "after": mean_after, "change": mean_change}
        else:
            asr = [x["success_rate"] for x in lst if x["success_rate"] is not None]
            if asr:
                out[k] = {"n": len(asr), "asr": float(np.mean(asr))}
    return out


def render(agg, out_png):
    fig, ax_pop = plt.subplots(figsize=(10, 5))
    ax_ind = ax_pop.twinx()

    labels = ["Pop\nTargeted", "Pop\nUntargeted", "Ind\nActivation", "Ind\nDeactivation"]
    x = np.arange(len(labels))
    width = 0.35

    pop_tgt   = agg.get(("population", True,  False), {})
    pop_untgt = agg.get(("population", False, False), {})
    pop_before = [(pop_tgt.get("initial") or 0) * 100, (pop_untgt.get("initial") or 0) * 100, 0, 0]
    pop_after  = [(pop_tgt.get("after")   or 0) * 100, (pop_untgt.get("after")   or 0) * 100, 0, 0]

    ind_act   = agg.get(("individual", True,  True),  {})
    ind_deact = agg.get(("individual", False, False), {})
    ind_asr_after = [0, 0, ind_act.get("asr", 0) * 100, ind_deact.get("asr", 0) * 100]

    ax_pop.bar(x - width/2, pop_before, width, label="Before", color="#f4a261")
    ax_pop.bar(x + width/2, pop_after,  width, label="After",  color="#e76f51")
    ax_ind.bar(x + width/2, ind_asr_after, width, color="#1d3557", label="ASR")

    paper_after = [PAPER_FIG2["pop_tgt_after"], PAPER_FIG2["pop_untgt_after"],
                   PAPER_FIG2["ind_act_asr"],   PAPER_FIG2["ind_deact_asr"]]
    for i, v in enumerate(paper_after):
        ax_pop.hlines(v, x[i] - width, x[i] + width, colors="black", linestyles="dashed", linewidth=1)

    ax_pop.set_xticks(x)
    ax_pop.set_xticklabels(labels)
    ax_pop.set_ylabel("Concept overlap ratio (Population, %)")
    ax_ind.set_ylabel("Attack success rate (Individual, %)")
    ax_pop.set_ylim(0, 105)
    ax_ind.set_ylim(0, 105)
    ax_pop.set_title("Reproduction of Fig 2: Gemma2-9B (131k) on AdvBench\n(black dashed = paper-reported After values)")
    ax_pop.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")

    print("\n=== numeric summary ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    print("\n=== vs paper ===")
    print(f"  pop_tgt   after: ours {pop_after[0]:.1f} vs paper {PAPER_FIG2['pop_tgt_after']:.1f}")
    print(f"  pop_untgt after: ours {pop_after[1]:.1f} vs paper {PAPER_FIG2['pop_untgt_after']:.1f}")
    print(f"  ind_act   asr:   ours {ind_asr_after[2]:.1f} vs paper {PAPER_FIG2['ind_act_asr']:.1f}")
    print(f"  ind_deact asr:   ours {ind_asr_after[3]:.1f} vs paper {PAPER_FIG2['ind_deact_asr']:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="/home/zzli/chryseis/code/SAE-vs-ICA-robustness/results/summary.csv")
    ap.add_argument("--out", default="/home/zzli/chryseis/code/SAE-vs-ICA-robustness/figures/fig2_repro.png")
    args = ap.parse_args()

    rows = load_summary(Path(args.summary))
    agg = aggregate(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    render(agg, args.out)

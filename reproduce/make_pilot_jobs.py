"""Generate pilot job list for Gemma2-9B-131k on AdvBench (safety.csv).

Pilot uses N=10 samples, all 4 paper scenarios x 2 modes (suffix, replacement) = 80 jobs.
Hyperparameters follow Table 5 of arXiv:2505.16004v2.
"""
import json
import sys
from pathlib import Path

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
out_path = sys.argv[2] if len(sys.argv) > 2 else "/home/zzli/chryseis/code/SAE-vs-ICA-robustness/reproduce/pilot_jobs.json"

base = dict(
    model_type="gemma2-9b-131k",
    layer_num=30,
    data_file="safety",
    k=170,  # paper: k=170 for gemma, k=192 for llama
)

# scenarios: (level, mode, targeted, activate, suffix_len, batch_size, num_iters, m)
# Hyperparameters from paper Table 5:
#   Pop  Targeted:   T=30, m=300, B=200
#   Pop  Untargeted: T=10, m=300, B=200
#   Ind  Targeted:   T=10, m=300, B=100
#   Ind  Untargeted: T=10, m=300, B=100
# suffix_len: 3 for population, 1 for individual (per upstream README).
scenarios = [
    # Population suffix
    ("population", "suffix",      True,  False, 3, 200, 30, 300),
    ("population", "suffix",      False, False, 3, 200, 10, 300),
    # Population replacement (suffix_len unused for replacement attacks)
    ("population", "replacement", True,  False, 1, 200, 10, 300),
    ("population", "replacement", False, False, 1, 200, 10, 300),
    # Individual suffix: activate = targeted; deactivate = untargeted in paper figure
    ("individual", "suffix",      True,  True,  1, 100, 10, 300),
    ("individual", "suffix",      False, False, 1, 100, 10, 300),
    # Individual replacement
    ("individual", "replacement", True,  True,  1, 100, 10, 300),
    ("individual", "replacement", False, False, 1, 100, 10, 300),
]

jobs = []
for s in range(N):
    for (lv, md, tgt, act, sl, B, T, m) in scenarios:
        jobs.append({
            **base,
            "level": lv,
            "mode": md,
            "targeted": tgt,
            "activate": act,
            "sample_idx": s,
            "suffix_len": sl,
            "batch_size": B,
            "num_iters": T,
            "m": m,
        })

Path(out_path).write_text(json.dumps(jobs, indent=2))
print(f"wrote {len(jobs)} jobs to {out_path}")
print(f"  N={N} samples x {len(scenarios)} scenarios")

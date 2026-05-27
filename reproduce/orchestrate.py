"""Dispatch sae_robustness/main.py jobs across 8 GPUs.

Each worker process pins itself to one GPU via CUDA_VISIBLE_DEVICES, then
runs main.py with the per-job args. The upstream code unconditionally uses
"cuda:0", so a worker that sees only one GPU treats that GPU as cuda:0.
"""

import argparse
import csv
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

UPSTREAM = Path("/home/zzli/chryseis/code/SAE-vs-ICA-robustness/sae_robustness")
RESULTS = Path("/home/zzli/chryseis/code/SAE-vs-ICA-robustness/results")
RESULTS.mkdir(parents=True, exist_ok=True)


@dataclass
class Job:
    model_type: str
    layer_num: int
    data_file: str
    level: str            # "individual" | "population"
    mode: str             # "suffix" | "replacement"
    targeted: bool
    activate: bool        # only meaningful for individual
    sample_idx: int
    suffix_len: int
    batch_size: int
    num_iters: int
    m: int
    k: int

    def label(self) -> str:
        tgt = "tgt" if self.targeted else "untgt"
        act = ("act" if self.activate else "deact") if self.level == "individual" else "-"
        return f"{self.model_type}_{self.data_file}_L{self.layer_num}_{self.level}_{self.mode}_{tgt}_{act}_s{self.sample_idx}"

    def to_argv(self) -> List[str]:
        argv = [
            "python", "main.py",
            "--model_type", self.model_type,
            "--layer_num", str(self.layer_num),
            "--data_file", self.data_file,
            "--level", self.level,
            "--mode", self.mode,
            "--sample_idx", str(self.sample_idx),
            "--suffix_len", str(self.suffix_len),
            "--batch_size", str(self.batch_size),
            "--num_iters", str(self.num_iters),
            "--m", str(self.m),
            "--k", str(self.k),
        ]
        if self.targeted:
            argv.append("--targeted")
        if self.level == "individual" and self.activate:
            argv.append("--activate")
        return argv


def parse_result(stdout: str, level: str) -> dict:
    """Pull final metric(s) from main.py stdout."""
    out = {}
    # main.py prints either:
    #   Individual attack success rate = X.XXXX
    #   Population attack overlap change = X.XXXX
    m = re.search(r"Individual attack success rate\s*=\s*([-+]?\d*\.\d+|\d+)", stdout)
    if m:
        out["success_rate"] = float(m.group(1))
    m = re.search(r"Population attack overlap change\s*=\s*([-+]?\d*\.\d+|\d+)", stdout)
    if m:
        out["overlap_change"] = float(m.group(1))
    # also try to capture "Initial s1_raw s2 overlap = X" or similar from attack body
    m = re.search(r"Initial\s+\w[^=]*overlap\s*=\s*([-+]?\d*\.\d+|\d+)", stdout)
    if m:
        out["initial_overlap"] = float(m.group(1))
    return out


def worker(gpu_id: int, jq: queue.Queue, results_lock: threading.Lock, results: list, total: int):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["HF_ENDPOINT"] = env.get("HF_ENDPOINT", "https://hf-mirror.com")
    env["HF_HOME"] = env.get("HF_HOME", "/home/zzli/chryseis/data/hf_cache")
    env["TOKENIZERS_PARALLELISM"] = "false"

    while True:
        try:
            job: Optional[Job] = jq.get_nowait()
        except queue.Empty:
            return
        label = job.label()
        log_path = RESULTS / f"{label}.log"
        t0 = time.time()
        try:
            p = subprocess.run(
                job.to_argv(),
                cwd=str(UPSTREAM),
                env=env,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            elapsed = time.time() - t0
            log_path.write_text(
                f"# argv: {job.to_argv()}\n# gpu: {gpu_id}\n# elapsed_s: {elapsed:.1f}\n# returncode: {p.returncode}\n"
                f"--- STDOUT ---\n{p.stdout}\n--- STDERR ---\n{p.stderr}\n"
            )
            row = asdict(job)
            row["gpu_id"] = gpu_id
            row["returncode"] = p.returncode
            row["elapsed_s"] = round(elapsed, 1)
            row.update(parse_result(p.stdout, job.level))
            ok_or_fail = "OK" if p.returncode == 0 else "FAIL"
            tail = (p.stdout[-200:] if p.returncode == 0 else p.stderr[-200:]).replace("\n", " ")
            with results_lock:
                results.append(row)
                done = len(results)
                print(f"[{done}/{total}] gpu{gpu_id} {ok_or_fail} {label} {elapsed:.1f}s | {tail}", flush=True)
        except subprocess.TimeoutExpired:
            log_path.write_text(f"# argv: {job.to_argv()}\n# gpu: {gpu_id}\nTIMEOUT after 1h\n")
            with results_lock:
                results.append({**asdict(job), "gpu_id": gpu_id, "returncode": -1, "elapsed_s": -1, "error": "timeout"})
                print(f"[?/{total}] gpu{gpu_id} TIMEOUT {label}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="JSON file with a list of job dicts")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--out", default=str(RESULTS / "summary.csv"))
    args = ap.parse_args()

    raw = json.loads(Path(args.jobs).read_text())
    jobs = [Job(**j) for j in raw]
    print(f"Loaded {len(jobs)} jobs", flush=True)

    jq: queue.Queue = queue.Queue()
    for j in jobs:
        jq.put(j)

    results: list = []
    lock = threading.Lock()
    gpus = [int(x) for x in args.gpus.split(",")]
    threads = [threading.Thread(target=worker, args=(g, jq, lock, results, len(jobs)), daemon=True) for g in gpus]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if results:
        keys = sorted({k for r in results for k in r.keys()})
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"wrote {args.out} ({len(results)} rows, {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()

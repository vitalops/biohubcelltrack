#!/usr/bin/env python3
"""Parallel one-factor sweep around the v91 baseline config."""
import json, subprocess, sys, itertools, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
PY = HERE.parent / "tuneenv/bin/python"
OUT = HERE / "sweep_out"
OUT.mkdir(exist_ok=True)

AXES = {
    "BIOHUB_SAFE_DIV_MAX_UM": ["6.5", "9.5", "12.0"],
    "BIOHUB_SAFE_DIV_SISTER_MAX_UM": ["9.0", "13.0", "15.0"],
    "BIOHUB_SAFE_DIV_DIVERGE_UM": ["1.0", "1.5", "3.0"],
    "BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN": ["0"],
    "BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM": ["8.0", "12.0"],
    "BIOHUB_SAFE_DIV_FRAME_FRAC_CAP": ["0.0152"],
    "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP": ["0.0075"],
    "BIOHUB_OUTPUT_MIN_TRACK_LEN": ["5", "7", "8"],
    "BIOHUB_GAP_CLOSE_UM": ["5.2", "6.4"],
    "BIOHUB_GAP_CLOSE_MAX_GAP": ["1"],
    "BIOHUB_GAP_DENSITY_GAIN": ["0.02", "0.08"],
    "BIOHUB_MOTION_RELINK_LEARNED_BONUS": ["0.5", "1.5"],
    "BIOHUB_OUTPUT_GAP2_RECOVERY": ["1"],
    "BIOHUB_OUTPUT_LINEFIT_SMOOTH": ["1"],
}

jobs = [("baseline", {})]
for k, vals in AXES.items():
    for v in vals:
        jobs.append((f"{k.replace('BIOHUB_','')}={v}", {k: v}))

def run(job):
    name, env = job
    slug = name.replace("=", "_").replace(".", "p")
    cfgp = OUT / f"{slug}.cfg.json"
    outp = OUT / f"{slug}.out.json"
    if outp.exists():
        return name, json.loads(outp.read_text())["summary"]
    cfgp.write_text(json.dumps({"env": env}))
    r = subprocess.run([str(PY), str(HERE / "run_config.py"), str(cfgp), str(outp)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return name, {"error": r.stderr[-300:]}
    return name, json.loads(outp.read_text())["summary"]

with ThreadPoolExecutor(max_workers=5) as ex:
    results = list(ex.map(run, jobs))

results_ok = [(n, s) for n, s in results if "proxy_score" in s]
results_ok.sort(key=lambda x: -x[1]["proxy_score"])
print(f"{'config':46s} {'proxy':>7s} {'adj_edge':>8s} {'div_j':>6s} {'div tp/fp/fn':>12s}")
for n, s in results_ok:
    print(f"{n:46s} {s['proxy_score']:.4f} {s['adjusted_edge_jaccard']:.4f} {s['division_jaccard']:.3f}  {s['div_tp']}/{s['div_fp']}/{s['div_fn']}")
for n, s in results:
    if "error" in s:
        print("ERROR", n, s["error"])

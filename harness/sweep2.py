#!/usr/bin/env python3
"""Parallel one-factor sweep around the v91 baseline config."""
import json, subprocess, sys, itertools, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
PY = HERE.parent / "tuneenv/bin/python"
OUT = HERE / "sweep_out"
OUT.mkdir(exist_ok=True)

AXES = {}
COMBOS = {
    "comboA_div15_rl15_mtl7": {"BIOHUB_SAFE_DIV_DIVERGE_UM":"1.5","BIOHUB_MOTION_RELINK_LEARNED_BONUS":"1.5","BIOHUB_OUTPUT_MIN_TRACK_LEN":"7"},
    "comboB_div15_rl15": {"BIOHUB_SAFE_DIV_DIVERGE_UM":"1.5","BIOHUB_MOTION_RELINK_LEARNED_BONUS":"1.5"},
    "comboC_rl15_mtl7": {"BIOHUB_MOTION_RELINK_LEARNED_BONUS":"1.5","BIOHUB_OUTPUT_MIN_TRACK_LEN":"7"},
    "rl20": {"BIOHUB_MOTION_RELINK_LEARNED_BONUS":"2.0"},
    "rl25": {"BIOHUB_MOTION_RELINK_LEARNED_BONUS":"2.5"},
    "div175": {"BIOHUB_SAFE_DIV_DIVERGE_UM":"1.75"},
    "div20": {"BIOHUB_SAFE_DIV_DIVERGE_UM":"2.0"},
    "comboD_div15_rl20_mtl7": {"BIOHUB_SAFE_DIV_DIVERGE_UM":"1.5","BIOHUB_MOTION_RELINK_LEARNED_BONUS":"2.0","BIOHUB_OUTPUT_MIN_TRACK_LEN":"7"},
    "comboE_div175_rl15_mtl7": {"BIOHUB_SAFE_DIV_DIVERGE_UM":"1.75","BIOHUB_MOTION_RELINK_LEARNED_BONUS":"1.5","BIOHUB_OUTPUT_MIN_TRACK_LEN":"7"},
    "comboA_gc52": {"BIOHUB_SAFE_DIV_DIVERGE_UM":"1.5","BIOHUB_MOTION_RELINK_LEARNED_BONUS":"1.5","BIOHUB_OUTPUT_MIN_TRACK_LEN":"7","BIOHUB_GAP_CLOSE_UM":"5.2"},
    "mtl7_rescue": {"BIOHUB_OUTPUT_MIN_TRACK_LEN":"7","BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE":"1"},
}
jobs = [("baseline", {})]
for name, env in COMBOS.items():
    jobs.append((name, env))


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

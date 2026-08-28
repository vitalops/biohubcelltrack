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
    "comboD": {"BIOHUB_SAFE_DIV_DIVERGE_UM":"1.5","BIOHUB_MOTION_RELINK_LEARNED_BONUS":"2.0","BIOHUB_OUTPUT_MIN_TRACK_LEN":"7"},
}
D = COMBOS["comboD"]
extra = {
    "D_tight5": {"BIOHUB_MOTION_RELINK_TIGHT_UM":"5.0"},
    "D_tight7": {"BIOHUB_MOTION_RELINK_TIGHT_UM":"7.0"},
    "D_relaxed8": {"BIOHUB_MOTION_RELINK_RELAXED_UM":"8.0"},
    "D_relaxed12": {"BIOHUB_MOTION_RELINK_RELAXED_UM":"12.0"},
    "D_vw025": {"BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT":"0.25"},
    "D_vw075": {"BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT":"0.75"},
    "D_edgemax12": {"BIOHUB_OUTPUT_EDGE_MAX_UM":"12.0"},
    "D_edgemax16": {"BIOHUB_OUTPUT_EDGE_MAX_UM":"16.0"},
    "D_div13": {"BIOHUB_SAFE_DIV_DIVERGE_UM":"1.3"},
    "D_div16": {"BIOHUB_SAFE_DIV_DIVERGE_UM":"1.6"},
    "D_scr1": {"BIOHUB_OUTPUT_SINGLE_CHILD_REPAIR":"1"},
    "D_reuse24": {"BIOHUB_GAP_CLOSE_REUSE_UM":"2.4"},
    "D_reuse40": {"BIOHUB_GAP_CLOSE_REUSE_UM":"4.0"},
    "D_geomfilter": {"BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER":"1"},
    "D_ilpdiv": {},
}
jobs=[("comboD", dict(D))]
for name, env in extra.items():
    if name=="D_ilpdiv": continue
    e=dict(D); e.update(env); jobs.append((name,e))



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

#!/usr/bin/env python3
"""Build the mitosis-gate training notebook (CPU kernel)."""
import json, pathlib

CELL1 = r'''
# Mine GT division events + hard negatives; direct zarr reads (no tracksdata)
import os, glob, json, math, time, subprocess, sys
import numpy as np
from pathlib import Path

DATA_ROOT = None
for cand in ["/kaggle/input/competitions/biohub-cell-tracking-during-development",
             "/kaggle/input/biohub-cell-tracking-during-development"]:
    if os.path.isdir(cand): DATA_ROOT = Path(cand); break
assert DATA_ROOT
TRAIN_DIR = DATA_ROOT / "train"

# zarr>=3 from our wheel set only (never touches numpy/scipy)
WH = None
for cand in glob.glob("/kaggle/input/*/zarr-3*.whl") + glob.glob("/kaggle/input/**/zarr-3*.whl"):
    WH = os.path.dirname(cand); break
import importlib
try:
    import zarr
    ok = int(zarr.__version__.split(".")[0]) >= 3
except Exception:
    ok = False
if not ok:
    assert WH, "zarr3 wheels dataset missing"
    subprocess.run([sys.executable,"-m","pip","install","-q","--no-index","--find-links",WH,
                    "zarr","donfig","numcodecs","blosc2","typing_extensions","packaging"], check=True)
    importlib.invalidate_caches()
    import zarr
    importlib.reload(zarr)
print("zarr", zarr.__version__)
import scipy; print("scipy ok", scipy.__version__)

VOX = np.array([1.625, 0.40625, 0.40625])
TEST_STEMS = {"44b6_0113de3b","44b6_0b24845f","6bba_05b6850b","6bba_05db0fb1"}
stems = sorted(p.name[:-5] for p in TRAIN_DIR.iterdir() if p.name.endswith(".geff"))
stems = [s for s in stems if s not in TEST_STEMS]
print("videos:", len(stems))

def load_gt(stem):
    g = zarr.open_group(str(TRAIN_DIR / f"{stem}.geff"), mode="r")
    ids = np.asarray(g["nodes/ids"])
    t = np.asarray(g["nodes/props/t/values"]); z = np.asarray(g["nodes/props/z/values"])
    y = np.asarray(g["nodes/props/y/values"]); x = np.asarray(g["nodes/props/x/values"])
    edges = np.asarray(g["edges/ids"])
    nodes = {int(i): (int(tt), float(zz), float(yy), float(xx))
             for i, tt, zz, yy, xx in zip(ids, t, z, y, x)}
    return nodes, edges

def open_image(stem):
    p = str(TRAIN_DIR / f"{stem}.zarr")
    try:
        root = zarr.open_group(p, mode="r")
        return root["0"]
    except Exception:
        return zarr.open_array(p, mode="r")
'''

CELL2 = r'''
# Event mining + feature extraction
from scipy.spatial import cKDTree

def dist_um(a, b):
    return float(np.linalg.norm((np.array(a) - np.array(b)) * VOX))

def win_stats(vol, z, y, x):
    Z, Y, X = vol.shape
    z0,z1 = max(0,int(z)-1), min(Z,int(z)+2)
    y0,y1 = max(0,int(y)-4), min(Y,int(y)+5)
    x0,x1 = max(0,int(x)-4), min(X,int(x)+5)
    w = np.asarray(vol[z0:z1, y0:y1, x0:x1], dtype=np.float32)
    if w.size == 0: return [0.0,0.0,0.0]
    return [float(w.mean()), float(w.max()), float(w.std())]

def extract_video(stem, max_events=4000):
    rows = []
    try:
        nodes, edges = load_gt(stem)
        arr = open_image(stem)
    except Exception as e:
        print("skip", stem, type(e).__name__, str(e)[:80]); return rows
    children = {}
    for srcid, dstid in edges:
        children.setdefault(int(srcid), []).append(int(dstid))

    frame_cache = {}
    def frame(t):
        if t not in frame_cache:
            if len(frame_cache) > 6: frame_cache.clear()
            a = np.asarray(arr[int(t)])
            while a.ndim > 3: a = a[0]
            frame_cache[t] = a
        return frame_cache[t]

    # positives: out-degree 2
    pos = [(p, cs) for p, cs in children.items() if len(cs) == 2]
    # negatives: out-degree 1 parents + nearest other node at t+1 (fake sister)
    by_t = {}
    for nid, (t, z, y, x) in nodes.items():
        by_t.setdefault(t, []).append(nid)
    trees = {}
    def tree(t):
        if t not in trees:
            ids = by_t.get(t, [])
            pts = np.array([np.array(nodes[i][1:]) * VOX for i in ids]) if ids else np.zeros((0,3))
            trees[t] = (ids, cKDTree(pts) if len(ids) else None)
        return trees[t]

    neg = []
    singles = [(p, cs[0]) for p, cs in children.items() if len(cs) == 1]
    rng = np.random.default_rng(42)
    rng.shuffle(singles)
    for p, c in singles[: max(len(pos) * 6, 300)]:
        tp = nodes[p][0]
        ids, tr = tree(tp + 1)
        if tr is None: continue
        q = np.array(nodes[c][1:]) * VOX
        dd, ii = tr.query(q, k=min(3, len(ids)))
        dd = np.atleast_1d(dd); ii = np.atleast_1d(ii)
        for d, i in zip(dd, ii):
            cand = ids[int(i)]
            if cand != c and d < 15.0:
                neg.append((p, [c, cand])); break

    def featurize(p, cs, label):
        tp, pz, py, px = nodes[p]
        c1, c2 = cs
        _, az, ay, ax = nodes[c1]
        _, bz, by_, bx = nodes[c2]
        pd1 = dist_um((pz,py,px), (az,ay,ax)); pd2 = dist_um((pz,py,px), (bz,by_,bx))
        sd = dist_um((az,ay,ax), (bz,by_,bx))
        # grandchild divergence
        g1 = children.get(c1, []); g2 = children.get(c2, [])
        div = -1.0
        if len(g1) == 1 and len(g2) == 1:
            ga = nodes.get(g1[0]); gb = nodes.get(g2[0])
            if ga and gb:
                div = dist_um(ga[1:], gb[1:]) - sd
        try:
            fp = frame(tp); fc = frame(tp + 1)
            s_par = win_stats(fp, pz, py, px)
            s_c1 = win_stats(fc, az, ay, ax)
            s_c2 = win_stats(fc, bz, by_, bx)
        except Exception:
            return None
        ids_t1, tr1 = tree(tp + 1)
        local_density = 0
        if tr1 is not None:
            local_density = len(tr1.query_ball_point(np.array([pz,py,px]) * VOX, 12.0))
        return [pd1, pd2, sd, abs(pd1 - pd2), div, local_density] + s_par + s_c1 + s_c2 + [label]

    events = [(p, cs, 1) for p, cs in pos] + [(p, cs, 0) for p, cs in neg]
    events.sort(key=lambda e: nodes[e[0]][0])  # frame order => cache-friendly
    for p, cs, lab in events[:max_events]:
        f = featurize(p, cs, lab)
        if f is not None: rows.append(f)
    return rows

all_rows = []
t0 = time.time()
for i, stem in enumerate(stems):
    r = extract_video(stem)
    all_rows.extend(r)
    if i % 10 == 0:
        print(f"[{i}/{len(stems)}] {stem}: +{len(r)} rows (total {len(all_rows)}) {time.time()-t0:.0f}s", flush=True)
    if time.time() - t0 > 9.5 * 3600:
        print("time cap"); break

X = np.array([r[:-1] for r in all_rows], dtype=np.float32)
y = np.array([r[-1] for r in all_rows], dtype=np.int32)
print("dataset:", X.shape, "positives:", int(y.sum()), "negatives:", int((1-y).sum()))
np.savez_compressed("/kaggle/working/mitosis_features.npz", X=X, y=y)
'''

CELL3 = r'''
# Train + evaluate the gate
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import joblib, numpy as np

d = np.load("/kaggle/working/mitosis_features.npz")
X, y = d["X"], d["y"]
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=6,
                                     l2_regularization=1.0, random_state=0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=2)
print("CV AUC:", aucs, "mean:", aucs.mean())
clf.fit(X, y)
joblib.dump(clf, "/kaggle/working/mitosis_gate.joblib")
FEATURES = ["pd1","pd2","sister_dist","pd_asym","divergence","local_density",
            "par_mean","par_max","par_std","c1_mean","c1_max","c1_std","c2_mean","c2_max","c2_std"]
import json
json.dump({"features": FEATURES, "cv_auc": float(aucs.mean())}, open("/kaggle/working/mitosis_gate_meta.json","w"))
print("saved gate model")
'''

nb = {"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
      "language_info":{"name":"python","version":"3.12"}},"nbformat":4,"nbformat_minor":4,
      "cells":[{"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
                "source":c.strip().splitlines(keepends=True)} for c in [CELL1,CELL2,CELL3]]}
out = pathlib.Path(__file__).parent / "biohub-mitosis-gate.ipynb"
out.write_text(json.dumps(nb))
print("wrote", out)

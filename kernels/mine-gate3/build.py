#!/usr/bin/env python3
"""Build the gate-v3 mining kernel: run Zharov's pipeline on ~80 train videos with a BROAD
safe-division candidate generator (gate off), log 22-feature vectors for every candidate and
every ILP fork, label them against GT, and train the gate on PREDICTED-graph candidates."""
import json, re, sys
from pathlib import Path
R = Path(__file__).resolve().parents[2]
nb = json.load(open(R / "kernels/submit-lf/biohub-submit-full.ipynb"))
guard = json.load(open(R / "kernels/submit-full/biohub-submit-full.ipynb"))["cells"][0]
assert "P100" in "".join(guard["source"])
VAL = '{"44b6_12dfb391","44b6_267148e4","6bba_062c8d37","6bba_07e24132"}'

HELPERS = r'''
# ---- gate-v3 mining: log 22-feature vectors for safe-div candidates and ILP forks ----
MINING_ENABLED = False
MINED = []
_MIT_VOX = np.array([1.625, 0.40625, 0.40625])
_mit_tree_cache = {}
_mit_in_cache = {}

def _mit_win_stats(vol, z, y, x):
    Z, Y, X = vol.shape
    z0, z1 = max(0, int(z) - 1), min(Z, int(z) + 2)
    y0, y1 = max(0, int(y) - 4), min(Y, int(y) + 5)
    x0, x1 = max(0, int(x) - 4), min(X, int(x) + 5)
    w = np.asarray(vol[z0:z1, y0:y1, x0:x1], dtype=np.float32)
    if w.size == 0:
        return [0.0, 0.0, 0.0]
    return [float(w.mean()), float(w.max()), float(w.std())]

def _mit_node_um(n):
    return np.array([float(n["z"]), float(n["y"]), float(n["x"])]) * _MIT_VOX

def _tree_for(dataset, t, nodes_by_id):
    key = (dataset, t)
    if key not in _mit_tree_cache:
        if len(_mit_tree_cache) > 6:
            _mit_tree_cache.clear()
        from scipy.spatial import cKDTree
        pts = np.array([_mit_node_um(n) for n in nodes_by_id.values() if int(n["t"]) == t])
        _mit_tree_cache[key] = cKDTree(pts) if len(pts) else None
    return _mit_tree_cache[key]

def mining_feats(dataset, parent, c1, c2, nodes_by_id, out_map, frame_cache):
    tp = int(parent["t"])
    p_um, a_um, b_um = _mit_node_um(parent), _mit_node_um(c1), _mit_node_um(c2)
    pd1 = float(np.linalg.norm(p_um - a_um)); pd2 = float(np.linalg.norm(p_um - b_um))
    sd = float(np.linalg.norm(a_um - b_um))
    div = -1.0
    g1 = out_map.get(int(c1["node_id"]), []); g2 = out_map.get(int(c2["node_id"]), [])
    if len(g1) == 1 and len(g2) == 1:
        ga = nodes_by_id.get(int(g1[0]["target_id"]) if isinstance(g1[0], dict) else int(g1[0]))
        gb = nodes_by_id.get(int(g2[0]["target_id"]) if isinstance(g2[0], dict) else int(g2[0]))
        if ga is not None and gb is not None:
            div = float(np.linalg.norm(_mit_node_um(ga) - _mit_node_um(gb))) - sd
    tr1 = _tree_for(dataset, tp + 1, nodes_by_id)
    dens = len(tr1.query_ball_point(p_um, 12.0)) if tr1 is not None else 0
    fp_ = read_test_frame(dataset, tp, frame_cache)
    fc_ = read_test_frame(dataset, tp + 1, frame_cache)
    s_par = _mit_win_stats(fp_, parent["z"], parent["y"], parent["x"])
    s_c1 = _mit_win_stats(fc_, c1["z"], c1["y"], c1["x"])
    s_c2 = _mit_win_stats(fc_, c2["z"], c2["y"], c2["x"])
    pv = np.array([float(parent["z"]), float(parent["y"]), float(parent["x"])])
    av = np.array([float(c1["z"]), float(c1["y"]), float(c1["x"])])
    bv = np.array([float(c2["z"]), float(c2["y"]), float(c2["x"])])
    mid_dist = float(np.linalg.norm((pv - (av + bv) / 2.0) * _MIT_VOX))
    sis_vec = (bv - av) * _MIT_VOX
    ikey = (dataset, id(out_map))
    if ikey not in _mit_in_cache:
        _mit_in_cache.clear()
        _im = {}
        for _s, _outs in out_map.items():
            for _e in _outs:
                _im[int(_e["target_id"]) if isinstance(_e, dict) else int(_e)] = int(_s)
        _mit_in_cache[ikey] = _im
    _q = nodes_by_id.get(_mit_in_cache[ikey].get(int(parent["node_id"])))
    if _q is not None:
        par_vec = (pv - np.array([float(_q["z"]), float(_q["y"]), float(_q["x"])])) * _MIT_VOX
        par_speed = float(np.linalg.norm(par_vec))
        cosang = float(np.dot(sis_vec, par_vec) / (np.linalg.norm(sis_vec) * np.linalg.norm(par_vec) + 1e-6))
    else:
        par_speed = -1.0; cosang = 0.0
    tr0 = _tree_for(dataset, tp, nodes_by_id)
    dens_t0 = len(tr0.query_ball_point(p_um, 12.0)) if tr0 is not None else 0
    int_ratio = (s_c1[0] + s_c2[0]) / (2.0 * s_par[0] + 1e-3)
    int_sym = abs(s_c1[0] - s_c2[0]) / (s_c1[0] + s_c2[0] + 1e-3)
    zdiff = abs(av[0] - bv[0]) * _MIT_VOX[0]
    return [pd1, pd2, sd, abs(pd1 - pd2), div, dens] + s_par + s_c1 + s_c2 + [mid_dist, par_speed, cosang, dens_t0, int_ratio, int_sym, zdiff]

def _mine_record(src, dataset, parent, c1, c2, nodes_by_id, out_map, frame_cache):
    try:
        f = mining_feats(dataset, parent, c1, c2, nodes_by_id, out_map, frame_cache)
    except Exception:
        return
    MINED.append({"src": src, "dataset": dataset, "t": int(parent["t"]),
                  "p": [float(parent[k]) for k in ("z", "y", "x")],
                  "c1": [float(c1[k]) for k in ("z", "y", "x")],
                  "c2": [float(c2[k]) for k in ("z", "y", "x")], "f": f})
# ---- end mining helpers ----

'''
SAFEDIV_HOOK = '''                if MINING_ENABLED:
                    _mine_record("safediv", dataset, nodes_by_id.get(source_id), existing_child, candidate, nodes_by_id, out_by_source, frame_cache)
'''
FORK_HOOK = '''
        # gate-v3 mining: log every ILP/pipeline fork with the same features (TEST_DIR swapped for frame reads)
        _real_test_dir2 = TEST_DIR
        globals()["TEST_DIR"] = TRAIN_DIR
        try:
            _om = {}
            for _e in processed_edges:
                _om.setdefault(int(_e["source_id"]), []).append(_e)
            _fc = {}
            for _src, _outs in _om.items():
                if len(_outs) == 2:
                    _pp = processed_nodes.get(_src); _a = processed_nodes.get(int(_outs[0]["target_id"])); _b = processed_nodes.get(int(_outs[1]["target_id"]))
                    if _pp is not None and _a is not None and _b is not None:
                        _mine_record("fork", stem, _pp, _a, _b, processed_nodes, _om, _fc)
        finally:
            globals()["TEST_DIR"] = _real_test_dir2
        print(f"  [mining] {stem}: records so far {len(MINED)}", flush=True)
'''
LABEL_TRAIN = r'''

# ================= gate-v3: label mined candidates against GT and train =================
import zarr, gzip, joblib
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
VOX3 = np.array([1.625, 0.40625, 0.40625])
def load_gt(stem):
    g = zarr.open_group(str(TRAIN_DIR / f"{stem}.geff"), mode="r")
    ids = np.asarray(g["nodes/ids"]); t = np.asarray(g["nodes/props/t/values"])
    z = np.asarray(g["nodes/props/z/values"]); y = np.asarray(g["nodes/props/y/values"]); x = np.asarray(g["nodes/props/x/values"])
    edges = np.asarray(g["edges/ids"])
    nodes = {int(i): (int(tt), float(zz), float(yy), float(xx)) for i, tt, zz, yy, xx in zip(ids, t, z, y, x)}
    ch = {}
    for s, d in edges: ch.setdefault(int(s), []).append(int(d))
    divs = []  # (t, parent_um, [child_um, child_um])
    for p, cs in ch.items():
        if len(cs) == 2 and p in nodes:
            tp, pz, py, px = nodes[p]
            kids = [np.array(nodes[c][1:]) * VOX3 for c in cs if c in nodes]
            if len(kids) == 2: divs.append((tp, np.array([pz, py, px]) * VOX3, kids))
    return divs
_seen = set(); _uniq = []
for r in MINED:
    k = (r["src"], r["dataset"], r["t"], tuple(round(v, 1) for v in r["p"]), tuple(round(v, 1) for v in r["c1"]), tuple(round(v, 1) for v in r["c2"]))
    if k in _seen: continue
    _seen.add(k); _uniq.append(r)
print(f"mined raw {len(MINED)} -> unique {len(_uniq)}"); MINED = _uniq
by_ds = {}
for r in MINED: by_ds.setdefault(r["dataset"], []).append(r)
X, y, meta = [], [], []
n_gt_div = 0
for stem, recs in by_ds.items():
    try: divs = load_gt(stem)
    except Exception as e:
        print("GT load failed", stem, e); continue
    n_gt_div += len(divs)
    by_t = {}
    for d in divs: by_t.setdefault(d[0], []).append(d)
    for r in recs:
        lab = 0
        p_um = np.array(r["p"]) * VOX3; c1 = np.array(r["c1"]) * VOX3; c2 = np.array(r["c2"]) * VOX3
        for (tp, gp, kids) in by_t.get(r["t"], []):
            if np.linalg.norm(gp - p_um) <= 7.0:
                d11 = np.linalg.norm(kids[0] - c1); d22 = np.linalg.norm(kids[1] - c2)
                d12 = np.linalg.norm(kids[0] - c2); d21 = np.linalg.norm(kids[1] - c1)
                if (d11 <= 7.0 and d22 <= 7.0) or (d12 <= 7.0 and d21 <= 7.0):
                    lab = 1; break
        X.append(r["f"]); y.append(lab); meta.append((r["src"], stem))
X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.int32)
print(f"mined: {len(X)} candidates from {len(by_ds)} videos | GT divisions in those videos: {n_gt_div} | positives: {int(y.sum())}")
for src in ("safediv", "fork"):
    m = np.array([s == src for s, _ in meta])
    print(f"  {src}: n={int(m.sum())} pos={int(y[m].sum())}")
FEATURES = ["pd1","pd2","sister_dist","pd_asym","divergence","local_density","par_mean","par_max","par_std",
            "c1_mean","c1_max","c1_std","c2_mean","c2_max","c2_std","mid_dist","par_speed","cosang","dens_t0","int_ratio","int_sym","zdiff"]
with gzip.open("/kaggle/working/mined_candidates.jsonl.gz", "wt") as f:
    for r, lab in zip(MINED, y.tolist()): r2 = dict(r); r2["label"] = int(lab); f.write(json.dumps(r2) + "\n")
if y.sum() >= 15:
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=4, min_samples_leaf=10,
                                         l2_regularization=2.0, class_weight="balanced", random_state=0)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    print("CV AUC:", np.round(aucs, 4), "mean:", float(aucs.mean()))
    clf.fit(X, y); joblib.dump(clf, "/kaggle/working/mitosis_gate_v3.joblib")
    json.dump({"features": FEATURES, "cv_auc": float(aucs.mean()), "n": int(len(y)), "pos": int(y.sum())}, open("/kaggle/working/mitosis_gate_v3_meta.json", "w"))
    print("saved gate v3")
else:
    print("too few positives to train")
'''

cells = nb["cells"]; code = [i for i, c in enumerate(cells) if c["cell_type"] == "code"]
def src(i): return "".join(cells[i]["source"])
def setsrc(i, s): cells[i]["source"] = s.splitlines(keepends=True)
n = {"cfg": 0, "sel": 0, "helpers": 0, "hook": 0, "fork": 0, "train": 0}
for i in code:
    s = src(i)
    if 'os.environ["BIOHUB_DET_THRESHOLD"]' in s and 'MITOSIS' in s:
        env = {"BIOHUB_VALIDATOR_ENABLE": "1", "BIOHUB_VALIDATOR_N_PER_TYPE": "40",
               "BIOHUB_SAFE_DIV_MAX_UM": "16.0", "BIOHUB_SAFE_DIV_SISTER_MAX_UM": "20.0", "BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM": "14.0",
               "BIOHUB_DEEPCENTER_SAFE_DIV_VETO": "0", "BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE": "0", "BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN": "0",
               "BIOHUB_SAFE_DIV_FRAME_FRAC_CAP": "0.05", "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP": "0.02",
               "BIOHUB_MITOSIS_GATE_MIN_PROB": "0", "BIOHUB_MITOSIS_PRUNE_MAX_PROB": "0"}
        for k, v in env.items():
            pat = re.compile(r'os\.environ\["%s"\]\s*=\s*[\'"][^\'"]*[\'"]' % k)
            line = f'os.environ["{k}"] = "{v}"'
            s = pat.sub(line, s) if pat.search(s) else s + "\n" + line
        setsrc(i, s); n["cfg"] += 1
    if "candidates = [s for s in train_stems_all if s not in test_stem_set]" in s:
        s = s.replace("candidates = [s for s in train_stems_all if s not in test_stem_set]",
                      "candidates = [s for s in train_stems_all if s not in test_stem_set and s not in " + VAL + "]")
        setsrc(i, s); n["sel"] += 1
    if "DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()" in s:
        s = s.replace("DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()", HELPERS + "DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()")
        a = '                stats["safe_division_geometric_candidates"] += 1\n'
        assert a in s; s = s.replace(a, SAFEDIV_HOOK + a); n["hook"] += 1
        setsrc(i, s); n["helpers"] += 1
    if "pred_nodes_plain = nodes_by_id_to_plain(processed_nodes)" in s:
        _hook = "\n".join(("    " + l if l.strip() else l) for l in FORK_HOOK.split("\n"))
        assert "            pred_nodes_plain = nodes_by_id_to_plain(processed_nodes)" in s
        s = "MINING_ENABLED = True\n" + s.replace("            pred_nodes_plain = nodes_by_id_to_plain(processed_nodes)", _hook + "            pred_nodes_plain = nodes_by_id_to_plain(processed_nodes)")
        assert "Per-sample validator rows written" in s
        s = s + LABEL_TRAIN; setsrc(i, s); n["fork"] += 1; n["train"] += 1
assert all(v == 1 for v in n.values()), n
cells.insert(0, guard)
for i, c in enumerate(cells):
    if c["cell_type"] == "code": compile("".join(c["source"]), f"c{i}", "exec")
out = Path(__file__).parent / "biohub-mine-gate3.ipynb"; json.dump(nb, open(out, "w")); print("built", out, n)

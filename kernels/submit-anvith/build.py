#!/usr/bin/env python3
"""anvithpothula 0.950 base (8-way TTA + ILP, no postproc) + our gate-v3m fork pruning at export."""
import json, re
from pathlib import Path
R = Path(__file__).resolve().parents[2]
nb = json.load(open(Path(__file__).parent / "base-anvith-0950.ipynb"))
wrk = json.load(open(R / "kernels/worker-prune3/biohub-w-prune3.ipynb"))
ws = "\n".join("".join(c["source"]) for c in wrk["cells"] if c["cell_type"] == "code")
a = ws.index("# ---- gate-v3 mining: log"); b = ws.index("# ---- end fork pruning ----") + len("# ---- end fork pruning ----\n")
block = ws[a:b]
assert "def mining_feats" in block and "def prune_forks_v3" in block
# the block references read_test_frame/TEST_DIR/np/os; provide them
PRE = r'''
# ---- fork-pruning support (our contribution) ----
import os, json, numpy as np, zarr
from pathlib import Path
TEST_DIR = Path(valid_dir)
_frame_cache_arr = {}
def read_test_frame(dataset, t, frame_cache):
    if t in frame_cache:
        return frame_cache[t]
    if dataset not in _frame_cache_arr:
        _frame_cache_arr.clear()
        _frame_cache_arr[dataset] = zarr.open_group(str(TEST_DIR / f"{dataset}.zarr"), mode="r")["0"]
    fr = np.asarray(_frame_cache_arr[dataset][int(t)])
    if len(frame_cache) > 4:
        frame_cache.clear()
    frame_cache[t] = fr
    return fr
os.environ["BIOHUB_PRUNE3_MIN_PROB"] = "0.5"
'''
EXPORT_HOOK = r'''
        # ---- our gate-v3m fork pruning (post-ILP, pre-export) ----
        nodes_by_id = {int(r["node_id"]): {"node_id": int(r["node_id"]), "t": int(r["t"]), "z": float(r["z"]), "y": float(r["y"]), "x": float(r["x"])} for r in node_row}
        edges_l = [{"source_id": int(r["source_id"]), "target_id": int(r["target_id"]), "edge_prob": float(r.get("edge_prob", 0.0) or 0.0)} for r in edge_row]
        _stats = {}
        edges_l = prune_forks_v3(nodes_by_id, edges_l, _stats, dataset=dataset, frame_cache={})
        edge_row = edges_l
'''
cells = nb["cells"]
# 0) GPU fail-fast so CPU auto-runs die instantly
guard = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": ['import torch as _t\n', 'if not _t.cuda.is_available():\n', '    raise RuntimeError("CUDA GPU is required. Select GPU T4 x2 and Save & Run All.")\n', 'print("GPU:", _t.cuda.get_device_name(0))\n']}
cells.insert(0, guard)
n = {"pre": 0, "hook": 0}
for c in cells:
    if c["cell_type"] != "code": continue
    s = "".join(c["source"])
    if s.startswith("#submission"):
        s = PRE + block + s
        anchor = '        node_id = {int(row["node_id"]) for row in node_row}'
        assert anchor in s
        s = s.replace(anchor, EXPORT_HOOK + anchor, 1); n["hook"] += 1
        c["source"] = s.splitlines(keepends=True); n["pre"] += 1
assert n == {"pre": 1, "hook": 1}, n
for i, c in enumerate(cells):
    if c["cell_type"] == "code": compile("".join(c["source"]), f"c{i}", "exec")
out = Path(__file__).parent / "biohub-submit-full.ipynb"
json.dump(nb, open(out, "w")); print("built", out, n)

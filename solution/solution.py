import os
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import gaussian_laplace
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max


SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
MAX_LINK_UM = 7.0
DATASETS = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]


def _open_volume(zarr_path):
    store = zarr.storage.LocalStore(str(zarr_path))
    root = zarr.open_group(store=store, mode="r")
    return root["0"]


def _normalize(vol):
    v = vol.astype(np.float32)
    lo, hi = np.percentile(v, (1.0, 99.5))
    if hi <= lo:
        return np.zeros_like(v)
    v = (v - lo) / (hi - lo)
    return np.clip(v, 0.0, 1.0)


def _detect(vol):
    v = _normalize(vol)
    sigma_z = 6.0 / SCALE[0] / 2.355
    sigma_xy = 6.0 / SCALE[1] / 2.355
    response = -gaussian_laplace(v, sigma=(sigma_z, sigma_xy, sigma_xy))
    threshold = max(float(np.percentile(response, 99.5)), 0.005)
    min_dist_xy = int(round(4.0 / SCALE[1]))
    coords = peak_local_max(
        response,
        min_distance=max(min_dist_xy, 2),
        threshold_abs=threshold,
        exclude_border=False,
    )
    if coords.size == 0:
        return coords.reshape(0, 3), np.zeros((0,), dtype=np.float32)
    scores = response[coords[:, 0], coords[:, 1], coords[:, 2]]
    order = np.argsort(-scores)
    return coords[order], scores[order]


def _link_pair(prev_um, curr_um):
    if prev_um.shape[0] == 0 or curr_um.shape[0] == 0:
        return []
    d = np.linalg.norm(prev_um[:, None, :] - curr_um[None, :, :], axis=-1)
    feasible = d <= MAX_LINK_UM
    if not feasible.any():
        return []
    big = MAX_LINK_UM * 10.0
    base = np.where(feasible, d, big)
    cost = np.vstack([base, base])
    n = prev_um.shape[0]
    rows, cols = linear_sum_assignment(cost)
    used_child = set()
    parent_children = {}
    for r, c in zip(rows, cols):
        if cost[r, c] >= big:
            continue
        if c in used_child:
            continue
        parent = int(r % n)
        parent_children.setdefault(parent, []).append((float(cost[r, c]), int(c)))
        used_child.add(int(c))
    edges = []
    for parent, kids in parent_children.items():
        kids.sort()
        if len(kids) == 1:
            edges.append((parent, kids[0][1]))
        else:
            d0, c0 = kids[0]
            d1, c1 = kids[1]
            if d1 <= 0.8 * MAX_LINK_UM and d1 - d0 <= 3.0:
                edges.append((parent, c0))
                edges.append((parent, c1))
            else:
                edges.append((parent, c0))
    return edges


def track_dataset(zarr_path):
    arr = _open_volume(zarr_path)
    T = arr.shape[0]
    nodes = []
    edges = []
    next_id = 1
    prev_ids = []
    prev_um = np.zeros((0, 3), dtype=np.float64)
    for t in range(T):
        volume = np.asarray(arr[t])
        coords, _ = _detect(volume)
        ids = list(range(next_id, next_id + coords.shape[0]))
        next_id += coords.shape[0]
        for (z, y, x), nid in zip(coords, ids):
            nodes.append((nid, t, int(z), int(y), int(x)))
        curr_um = coords.astype(np.float64) * SCALE
        if t > 0 and prev_um.shape[0] and curr_um.shape[0]:
            for i, j in _link_pair(prev_um, curr_um):
                edges.append((prev_ids[i], ids[j]))
        prev_ids = ids
        prev_um = curr_um
    return nodes, edges


def build_submission(test_root, out_path):
    rows = []
    gid = 0
    for ds in DATASETS:
        zarr_path = Path(test_root) / f"{ds}.zarr"
        nodes, edges = track_dataset(zarr_path)
        for nid, t, z, y, x in nodes:
            rows.append((gid, ds, "node", nid, t, z, y, x, -1, -1))
            gid += 1
        for s, d in edges:
            rows.append((gid, ds, "edge", -1, -1, -1, -1, -1, s, d))
            gid += 1
    df = pd.DataFrame(
        rows,
        columns=["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"],
    )
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    test_root = os.environ.get(
        "TEST_ROOT",
        "/kaggle/input/biohub-cell-tracking-during-development/test",
    )
    out_path = os.environ.get("OUT_PATH", "submission.csv")
    build_submission(test_root, out_path)

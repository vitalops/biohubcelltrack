import json
from pathlib import Path


def notebook(cells):
    return {
        "cells": [
            {
                "cell_type": "code",
                "id": f"c{i}",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": src,
            }
            for i, src in enumerate(cells)
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


CELL_1 = r'''from __future__ import annotations
import os, sys, json, time, gc
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field


SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)


def _make_blosc_decoder():
    try:
        from numcodecs import Blosc
        codec = Blosc()
        return lambda buf: codec.decode(buf)
    except Exception:
        pass
    try:
        import blosc as _b
        return lambda buf: _b.decompress(bytes(buf))
    except Exception:
        pass
    try:
        import blosc2 as _b2
        return lambda buf: _b2.decompress(bytes(buf))
    except Exception:
        pass
    try:
        import imagecodecs as _ic
        return lambda buf: _ic.blosc_decode(bytes(buf))
    except Exception:
        pass
    return None


_BLOSC_DECODER = _make_blosc_decoder()


@dataclass
class ImageVolume:
    path: str
    shape: tuple
    dtype: np.dtype
    chunk: tuple

    @property
    def n_t(self) -> int:
        return int(self.shape[0])

    def frame(self, t: int) -> np.ndarray:
        return _read_chunk(self.path, t, self.shape, self.dtype)


def open_image(zarr_path: str) -> ImageVolume:
    with open(os.path.join(zarr_path, "0", "zarr.json")) as f:
        meta = json.load(f)
    shape = tuple(int(s) for s in meta["shape"])
    dtype = np.dtype(meta["data_type"])
    chunk = None
    cg = meta.get("chunk_grid", {})
    conf = cg.get("configuration", {})
    if "chunk_shape" in conf:
        chunk = tuple(int(s) for s in conf["chunk_shape"])
    return ImageVolume(path=zarr_path, shape=shape, dtype=dtype, chunk=chunk)


def _read_chunk(zarr_path: str, t: int, shape: tuple, dtype: np.dtype) -> np.ndarray:
    frame_shape = shape[1:]
    chunk_path = os.path.join(zarr_path, "0", "c", str(t), "0", "0", "0")
    with open(chunk_path, "rb") as f:
        raw = f.read()
    if _BLOSC_DECODER is not None:
        try:
            dec = _BLOSC_DECODER(raw)
            arr = np.frombuffer(dec, dtype=dtype)
            if arr.size == int(np.prod(frame_shape)):
                return arr.reshape(frame_shape).copy()
        except Exception:
            pass
    arr = np.frombuffer(raw, dtype=dtype)
    if arr.size == int(np.prod(frame_shape)):
        return arr.reshape(frame_shape).copy()
    try:
        import zarr
        z = zarr.open(os.path.join(zarr_path, "0"), mode="r")
        return np.asarray(z[t])
    except Exception as e:
        raise RuntimeError(f"could not read chunk at t={t}: {e}")


@dataclass
class TrackGraph:
    node_t: np.ndarray
    node_z: np.ndarray
    node_y: np.ndarray
    node_x: np.ndarray
    node_ids: np.ndarray
    edges: np.ndarray
    meta: dict

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return len(self.edges)
'''

CELL_2 = r'''# ===== detection =====
def detect_blobs(vol: np.ndarray,
                 xy_downsample: int = 4,
                 dog_small_um: float = 2.0,
                 dog_large_um: float = 6.0,
                 min_distance_um: float = 3.0,
                 rel_threshold: float = 0.04,
                 abs_percentile: float = 50.0,
                 max_peaks=30000,
                 dog_scales=None) -> np.ndarray:
    vf = vol.astype(np.float32)
    ds = vf[:, ::xy_downsample, ::xy_downsample]
    eff = np.array([SCALE[0], SCALE[1] * xy_downsample, SCALE[2] * xy_downsample])
    lo, hi = np.percentile(ds, [1.0, 99.7])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((ds - lo) / (hi - lo), 0, None)
    if dog_scales:
        dog = None
        for (s_um, l_um) in dog_scales:
            resp = (gaussian_filter(norm, sigma=s_um / eff)
                    - gaussian_filter(norm, sigma=l_um / eff))
            dog = resp if dog is None else np.maximum(dog, resp)
    else:
        s_small = dog_small_um / eff
        s_large = dog_large_um / eff
        dog = gaussian_filter(norm, sigma=s_small) - gaussian_filter(norm, sigma=s_large)
    footprint = _ball_footprint(min_distance_um, eff)
    mx = maximum_filter(dog, footprint=footprint, mode="nearest")
    thr = max(rel_threshold, 0.0)
    abs_thr = np.percentile(norm, abs_percentile)
    peaks = (dog == mx) & (dog >= thr) & (norm >= abs_thr)
    coords = np.argwhere(peaks)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    vals = dog[peaks]
    order = np.argsort(vals)[::-1]
    coords = coords[order]
    if max_peaks is not None and len(coords) > max_peaks:
        coords = coords[:max_peaks]
    out = coords.astype(np.float64)
    out[:, 1] *= xy_downsample
    out[:, 2] *= xy_downsample
    return out


def _ball_footprint(radius_um: float, eff_spacing: np.ndarray) -> np.ndarray:
    rad_vox = np.maximum(1, np.round(radius_um / eff_spacing).astype(int))
    zz, yy, xx = np.ogrid[-rad_vox[0]:rad_vox[0] + 1,
                          -rad_vox[1]:rad_vox[1] + 1,
                          -rad_vox[2]:rad_vox[2] + 1]
    d = ((zz * eff_spacing[0]) ** 2 + (yy * eff_spacing[1]) ** 2 +
         (xx * eff_spacing[2]) ** 2)
    return d <= radius_um ** 2


def refine_centroids(vol: np.ndarray, coords: np.ndarray, win=(1, 3, 3)) -> np.ndarray:
    if len(coords) == 0:
        return coords
    Z, Y, X = vol.shape
    out = coords.copy().astype(np.float64)
    wz, wy, wx = win
    for i, (z, y, x) in enumerate(coords):
        z, y, x = int(round(z)), int(round(y)), int(round(x))
        z0, z1 = max(0, z - wz), min(Z, z + wz + 1)
        y0, y1 = max(0, y - wy), min(Y, y + wy + 1)
        x0, x1 = max(0, x - wx), min(X, x + wx + 1)
        patch = vol[z0:z1, y0:y1, x0:x1].astype(np.float64)
        s = patch.sum()
        if s <= 0:
            continue
        zz = np.arange(z0, z1)[:, None, None]
        yy = np.arange(y0, y1)[None, :, None]
        xx = np.arange(x0, x1)[None, None, :]
        out[i, 0] = (patch * zz).sum() / s
        out[i, 1] = (patch * yy).sum() / s
        out[i, 2] = (patch * xx).sum() / s
    return out
'''

CELL_3 = r'''# ===== linking =====
def link_frames(frames, max_link_um=10.0, allow_divisions=False, division_max_um=6.0):
    node_ids, node_t, node_z, node_y, node_x = [], [], [], [], []
    frame_ids = []
    nid = 1
    for t, coords in enumerate(frames):
        ids_t = []
        for (z, y, x) in coords:
            node_ids.append(nid); node_t.append(t); node_z.append(z); node_y.append(y); node_x.append(x)
            ids_t.append(nid); nid += 1
        frame_ids.append(ids_t)
    edges = []
    for t in range(len(frames) - 1):
        a = frames[t]; b = frames[t + 1]
        if len(a) == 0 or len(b) == 0:
            continue
        ap = a * SCALE; bp = b * SCALE
        d = np.sqrt(((ap[:, None, :] - bp[None, :, :]) ** 2).sum(axis=2))
        big = max_link_um * 1000.0 + 1.0
        cost = np.where(d <= max_link_um, d, big)
        ri, ci = linear_sum_assignment(cost)
        matched_b, matched_a = set(), set()
        for r, c in zip(ri, ci):
            if d[r, c] <= max_link_um:
                edges.append((frame_ids[t][r], frame_ids[t + 1][c]))
                matched_a.add(r); matched_b.add(c)
        if allow_divisions:
            unmatched_b = [j for j in range(len(b)) if j not in matched_b]
            for r in list(matched_a):
                if not unmatched_b: break
                pr = ap[r]
                dd = np.sqrt(((pr[None, :] - bp[unmatched_b]) ** 2).sum(axis=1))
                j = int(np.argmin(dd))
                if dd[j] <= division_max_um:
                    bj = unmatched_b[j]
                    edges.append((frame_ids[t][r], frame_ids[t + 1][bj]))
                    unmatched_b.remove(bj)
    g = TrackGraph(
        node_t=np.array(node_t, dtype=np.int64),
        node_z=np.array(node_z, dtype=np.float64),
        node_y=np.array(node_y, dtype=np.float64),
        node_x=np.array(node_x, dtype=np.float64),
        node_ids=np.array(node_ids, dtype=np.int64),
        edges=np.array(edges, dtype=np.int64).reshape(-1, 2),
        meta={},
    )
    return g


def close_gaps(frames, g: TrackGraph, max_gap=1, gap_dist_um=8.0):
    if g.n_edges == 0:
        return g
    coords = {int(nid): (int(g.node_t[i]), g.node_z[i], g.node_y[i], g.node_x[i])
              for i, nid in enumerate(g.node_ids)}
    has_out = set(int(s) for s, _ in g.edges)
    has_in = set(int(t) for _, t in g.edges)
    ends_by_t, starts_by_t = {}, {}
    for nid, (t, z, y, x) in coords.items():
        if nid not in has_out:
            ends_by_t.setdefault(t, []).append(nid)
        if nid not in has_in:
            starts_by_t.setdefault(t, []).append(nid)
    new_nodes, new_edges = [], []
    next_id = int(g.node_ids.max()) + 1 if g.n_nodes else 1
    for gap in range(1, max_gap + 1):
        for t, ends in ends_by_t.items():
            starts = starts_by_t.get(t + gap + 1)
            if not starts:
                continue
            ec = np.array([[coords[e][1], coords[e][2], coords[e][3]] for e in ends]) * SCALE
            sc = np.array([[coords[s][1], coords[s][2], coords[s][3]] for s in starts]) * SCALE
            d = np.sqrt(((ec[:, None, :] - sc[None, :, :]) ** 2).sum(axis=2))
            thr = gap_dist_um * (gap + 1)
            big = thr * 1000 + 1
            cost = np.where(d <= thr, d, big)
            ri, ci = linear_sum_assignment(cost)
            used_s = set()
            for r, c in zip(ri, ci):
                if d[r, c] > thr or ends[r] in has_out or starts[c] in used_s:
                    continue
                e_id, s_id = ends[r], starts[c]
                te, ze, ye, xe = coords[e_id]
                ts, zs, ys, xs = coords[s_id]
                prev = e_id
                for k in range(1, gap + 1):
                    frac = k / (gap + 1)
                    zi = ze + (zs - ze) * frac
                    yi = ye + (ys - ye) * frac
                    xi = xe + (xs - xe) * frac
                    nid = next_id; next_id += 1
                    new_nodes.append((te + k, zi, yi, xi, nid))
                    new_edges.append((prev, nid))
                    prev = nid
                new_edges.append((prev, s_id))
                has_out.add(e_id)
                used_s.add(c)
    if not new_nodes:
        return g
    nt = np.concatenate([g.node_t, np.array([n[0] for n in new_nodes], dtype=np.int64)])
    nz = np.concatenate([g.node_z, np.array([n[1] for n in new_nodes])])
    ny = np.concatenate([g.node_y, np.array([n[2] for n in new_nodes])])
    nx = np.concatenate([g.node_x, np.array([n[3] for n in new_nodes])])
    nid = np.concatenate([g.node_ids, np.array([n[4] for n in new_nodes], dtype=np.int64)])
    edges = np.concatenate([g.edges, np.array(new_edges, dtype=np.int64).reshape(-1, 2)])
    return TrackGraph(node_t=nt, node_z=nz, node_y=ny, node_x=nx, node_ids=nid,
                      edges=edges, meta=g.meta)


def add_safe_divisions(g: TrackGraph,
                       max_um: float = 5.25,
                       sibling_max_um: float = 8.5,
                       existing_child_max_um: float = 9.0,
                       frame_frac_cap: float = 0.014,
                       global_frac_cap: float = 0.007) -> TrackGraph:
    if g.n_nodes == 0 or g.n_edges == 0:
        return g
    from collections import defaultdict as _dd
    coords = {int(nid): (int(g.node_t[i]), float(g.node_z[i]), float(g.node_y[i]), float(g.node_x[i]))
              for i, nid in enumerate(g.node_ids)}
    children = _dd(list); parents = _dd(list)
    for s, t in g.edges:
        children[int(s)].append(int(t)); parents[int(t)].append(int(s))
    by_t = _dd(list)
    for nid, (t, z, y, x) in coords.items():
        by_t[int(t)].append((int(nid), np.array([z, y, x])))
    total_nodes = g.n_nodes
    global_cap = int(np.ceil(global_frac_cap * max(total_nodes, 1)))
    added = []
    per_frame_added = _dd(int)
    for s, existing_children in list(children.items()):
        if len(added) >= global_cap:
            break
        if len(existing_children) != 1:
            continue
        st, sz, sy, sx = coords[s]
        child = existing_children[0]
        if child not in coords:
            continue
        ct, cz, cy, cx = coords[child]
        if ct != st + 1:
            continue
        frame_cap = int(np.ceil(frame_frac_cap * max(len(by_t.get(ct, [])), 1)))
        if per_frame_added[ct] >= frame_cap:
            continue
        parent_um = np.array([sz, sy, sx]) * SCALE
        existing_um = np.array([cz, cy, cx]) * SCALE
        existing_dist = float(np.linalg.norm(parent_um - existing_um))
        if existing_dist > existing_child_max_um:
            continue
        candidates = []
        for cand_id, cand_xyz in by_t.get(ct, []):
            if cand_id == child or cand_id in parents:
                continue
            cand_um = cand_xyz * SCALE
            d_parent = float(np.linalg.norm(parent_um - cand_um))
            d_sibling = float(np.linalg.norm(existing_um - cand_um))
            if d_parent <= max_um and d_sibling <= sibling_max_um:
                score = d_parent + 0.5 * d_sibling
                candidates.append((score, cand_id))
        if not candidates:
            continue
        candidates.sort()
        _, cand_id = candidates[0]
        added.append((int(s), int(cand_id)))
        parents[int(cand_id)].append(int(s))
        children[int(s)].append(int(cand_id))
        per_frame_added[ct] += 1
    if not added:
        return g
    new_edges = np.concatenate([g.edges, np.array(added, dtype=np.int64).reshape(-1, 2)])
    return TrackGraph(node_t=g.node_t, node_z=g.node_z, node_y=g.node_y, node_x=g.node_x,
                      node_ids=g.node_ids, edges=new_edges, meta=g.meta)


def prune_isolated(g: TrackGraph) -> TrackGraph:
    if g.n_edges == 0:
        return g
    used = set(int(x) for x in g.edges.reshape(-1))
    keep = np.array([i for i, nid in enumerate(g.node_ids) if int(nid) in used])
    if len(keep) == len(g.node_ids):
        return g
    return TrackGraph(
        node_t=g.node_t[keep], node_z=g.node_z[keep], node_y=g.node_y[keep],
        node_x=g.node_x[keep], node_ids=g.node_ids[keep], edges=g.edges, meta=g.meta,
    )
'''

CELL_4 = r'''# ===== pipeline =====
@dataclass
class Config:
    xy_downsample: int = 4
    dog_small_um: float = 1.5
    dog_large_um: float = 4.0
    dog_scales: list = None
    rel_threshold: float = 0.02
    abs_percentile: float = 50.0
    min_distance_um: float = 2.5
    max_peaks: int = 40000
    refine: bool = True
    max_link_um: float = 10.0
    allow_divisions: bool = False
    division_max_um: float = 6.0
    close_gaps: bool = False
    max_gap: int = 1
    gap_dist_um: float = 8.0
    prune_isolated: bool = True
    safe_divisions: bool = False
    safe_division_max_um: float = 5.25
    safe_division_sibling_max_um: float = 8.5
    safe_division_existing_child_max_um: float = 9.0
    safe_division_frame_frac_cap: float = 0.014
    safe_division_global_frac_cap: float = 0.007


def run_one(zarr_path: str, cfg: Config, t_limit=None) -> TrackGraph:
    vol_meta = open_image(zarr_path)
    n_t = vol_meta.n_t if t_limit is None else min(t_limit, vol_meta.n_t)
    frames = []
    for t in range(n_t):
        vol = vol_meta.frame(t)
        coords = detect_blobs(
            vol, xy_downsample=cfg.xy_downsample,
            dog_small_um=cfg.dog_small_um, dog_large_um=cfg.dog_large_um,
            min_distance_um=cfg.min_distance_um, rel_threshold=cfg.rel_threshold,
            abs_percentile=cfg.abs_percentile, max_peaks=cfg.max_peaks,
            dog_scales=cfg.dog_scales,
        )
        if cfg.refine and len(coords) > 0:
            coords = refine_centroids(vol, coords)
        frames.append(coords)
        del vol
        gc.collect()
    g = link_frames(frames, max_link_um=cfg.max_link_um,
                    allow_divisions=cfg.allow_divisions,
                    division_max_um=cfg.division_max_um)
    if cfg.close_gaps:
        g = close_gaps(frames, g, max_gap=cfg.max_gap, gap_dist_um=cfg.gap_dist_um)
    if getattr(cfg, "safe_divisions", False):
        g = add_safe_divisions(
            g,
            max_um=cfg.safe_division_max_um,
            sibling_max_um=cfg.safe_division_sibling_max_um,
            existing_child_max_um=cfg.safe_division_existing_child_max_um,
            frame_frac_cap=cfg.safe_division_frame_frac_cap,
            global_frac_cap=cfg.safe_division_global_frac_cap,
        )
    if cfg.prune_isolated:
        g = prune_isolated(g)
    return g


def graph_to_rows(name: str, g: TrackGraph):
    rows = []
    for i in range(g.n_nodes):
        rows.append({
            "dataset": name, "row_type": "node", "node_id": int(g.node_ids[i]),
            "t": int(g.node_t[i]), "z": int(round(g.node_z[i])),
            "y": int(round(g.node_y[i])), "x": int(round(g.node_x[i])),
            "source_id": -1, "target_id": -1,
        })
    for (s, t) in g.edges:
        rows.append({
            "dataset": name, "row_type": "edge", "node_id": -1, "t": -1,
            "z": -1, "y": -1, "x": -1, "source_id": int(s), "target_id": int(t),
        })
    return rows


CONFIG_OVERRIDE = {
    "xy_downsample": 4,
    "refine": True,
    "dog_scales": [[1.0, 3.0], [1.5, 4.0], [2.2, 5.5]],
    "rel_threshold": 0.045,
    "min_distance_um": 4.0,
    "max_peaks": 40000,
    "max_link_um": 8.0,
    "close_gaps": True,
    "max_gap": 1,
    "gap_dist_um": 6.0,
    "prune_isolated": True,
    "safe_divisions": True,
    "safe_division_max_um": 5.25,
    "safe_division_sibling_max_um": 8.5,
    "safe_division_existing_child_max_um": 9.0,
    "safe_division_frame_frac_cap": 0.014,
    "safe_division_global_frac_cap": 0.007,
}


def find_test_dir():
    for c in ["/kaggle/input/biohub-cell-tracking-during-development/test",
              "/kaggle/input/competitions/biohub-cell-tracking-during-development/test"]:
        if os.path.isdir(c):
            return c
    base = "/kaggle/input"
    if os.path.isdir(base):
        for root, dirs, files in os.walk(base):
            if os.path.basename(root) == "test" and any(d.endswith(".zarr") for d in dirs):
                return root
    raise FileNotFoundError("test dir not found")


test_dir = find_test_dir()
names = sorted(d[:-5] for d in os.listdir(test_dir) if d.endswith(".zarr"))
print(f"test dir: {test_dir}; {len(names)} datasets: {names}", flush=True)

cfg = Config(**CONFIG_OVERRIDE)
all_rows = []
t0 = time.time()
for i, name in enumerate(names):
    zp = os.path.join(test_dir, name + ".zarr")
    g = run_one(zp, cfg)
    all_rows.extend(graph_to_rows(name, g))
    print(f"[{i+1}/{len(names)}] {name}: nodes={g.n_nodes} edges={g.n_edges} ({(time.time()-t0)/60:.1f} min)", flush=True)

sub = pd.DataFrame(all_rows, columns=["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"])
sub.insert(0, "id", np.arange(len(sub), dtype=np.int64))
sub.to_csv("submission.csv", index=False)
print(f"wrote submission.csv: {len(sub)} rows in {(time.time()-t0)/60:.1f} min", flush=True)
sub.head()
'''


nb = notebook([CELL_1, CELL_2, CELL_3, CELL_4])
out = Path(__file__).parent / "solution.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}")

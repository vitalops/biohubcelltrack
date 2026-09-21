"""Cellpose-3D extra-candidate injection for the pilkwang TemporalUNet3D detector.

Runs a (public, pre-finetuned) Cellpose nuclei model per frame at FULL resolution,
extracts 3D label centroids, drops those already covered by a UNet peak (radius in um),
and appends the rest to the per-frame peak list so the node transformer scores edges
for them exactly like native peaks. Injected nodes that never link into a track are
removed later by the pipeline's own isolated/short-track filters.
"""
from __future__ import annotations
import os, sys, time, glob
from pathlib import Path
import numpy as np

_CP_MODEL = None
_CP_STATE = {"frames": 0, "raw": 0, "added": 0, "unet": 0, "secs": 0.0, "failed": False}


def _env_f(name, default):
    return float(os.environ.get(name, str(default)))


def _load_cellpose():
    global _CP_MODEL
    if _CP_MODEL is not None or _CP_STATE["failed"]:
        return _CP_MODEL
    try:
        import torch
        from cellpose import models
        explicit = os.environ.get("BIOHUB_CP_MODEL", "")
        cands = [explicit] if explicit else []
        cands += sorted(glob.glob("/kaggle/input/**/biohub_nuclei3d", recursive=True))
        cands += sorted(glob.glob("/kaggle/input/**/nucleitorch_0", recursive=True))
        path = next((c for c in cands if c and Path(c).is_file()), None)
        if path is None:
            raise FileNotFoundError("no cellpose model found among inputs")
        _CP_MODEL = models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model=str(path))
        print(f"EXTRA_PEAKS: cellpose model {path} diam_labels={float(getattr(_CP_MODEL, 'diam_labels', 0)):.2f}", flush=True)
    except Exception as exc:  # never kill inference because of the auxiliary detector
        _CP_STATE["failed"] = True
        print(f"EXTRA_PEAKS: cellpose unavailable ({type(exc).__name__}: {exc}); injection disabled", flush=True)
    return _CP_MODEL


def _centroids(masks: np.ndarray, min_vox: int):
    lab = masks.ravel()
    idx = np.flatnonzero(lab)
    if idx.size == 0:
        return np.empty((0, 3), np.float32), np.empty((0,), np.int64)
    l = lab[idx].astype(np.int64)
    z, y, x = np.unravel_index(idx, masks.shape)
    n = int(l.max()) + 1
    cnt = np.bincount(l, minlength=n).astype(np.float64)
    cz = np.bincount(l, weights=z, minlength=n)
    cy = np.bincount(l, weights=y, minlength=n)
    cx = np.bincount(l, weights=x, minlength=n)
    keep = np.flatnonzero(cnt >= min_vox)
    keep = keep[keep > 0]
    c = np.stack([cz[keep] / cnt[keep], cy[keep] / cnt[keep], cx[keep] / cnt[keep]], 1).astype(np.float32)
    return c, cnt[keep].astype(np.int64)


def cellpose_centroids(frame_full: np.ndarray):
    """frame_full: (Z, Y, X) float32 full-resolution frame -> (N,3) centroids [z,y,x] full-res + sizes."""
    m = _load_cellpose()
    if m is None:
        return np.empty((0, 3), np.float32), np.empty((0,), np.int64)
    do3d = os.environ.get("BIOHUB_CP_DO3D", "0") == "1"
    kw = dict(channels=[0, 0], batch_size=int(_env_f("BIOHUB_CP_BATCH", 16)), normalize=True,
              flow_threshold=_env_f("BIOHUB_CP_FLOW_THR", 0.4), cellprob_threshold=_env_f("BIOHUB_CP_CELLPROB", 0.0),
              z_axis=0)
    diam = os.environ.get("BIOHUB_CP_DIAMETER", "")
    if diam:
        kw["diameter"] = float(diam)
    if do3d:
        kw.update(do_3D=True, anisotropy=_env_f("BIOHUB_CP_ANISOTROPY", 4.0))
    else:
        kw.update(do_3D=False, stitch_threshold=_env_f("BIOHUB_CP_STITCH", 0.3))
    out = m.eval(frame_full, **kw)
    masks = out[0]
    return _centroids(np.asarray(masks), int(_env_f("BIOHUB_CP_MIN_VOX", 40)))


def inject_extra_peaks(arr: np.ndarray, t: int, zarr_arr, downsample, voxel_size_full, stem: str = "", sec_logits=None, pool_k=None, detect_fn=None) -> np.ndarray:
    """arr: (N,4) int16 [t,z,y,x] UNet peaks in DOWNSAMPLED grid. Returns augmented array (same dtype/layout)."""
    if os.environ.get("BIOHUB_EXTRA_PEAKS", "1") != "1" or _CP_STATE["failed"]:
        return arr
    t0 = time.time()
    try:
        from scipy.spatial import cKDTree
        ds = np.asarray(downsample, np.float32)  # e.g. (1,4,4)
        vs = np.asarray(voxel_size_full, np.float32)  # e.g. (1.625,0.406,0.406)
        source = os.environ.get("BIOHUB_EXTRA_PEAKS_SOURCE", "cellpose")
        cents_list, sizes_list = [], []
        if source in ("cellpose", "both"):
            raw = np.asarray(zarr_arr[t]).astype(np.float32)  # (Z,Y,X) full res
            c, sz = cellpose_centroids(raw)
            cents_list.append(c); sizes_list.append(sz)
        if source in ("secondary", "both"):
            if sec_logits is None or detect_fn is None or pool_k is None:
                raise RuntimeError("secondary peaks requested but no secondary logits/detect_fn/pool_k available")
            thr = _env_f("BIOHUB_SEC_PEAKS_THR", 0.985)
            sp = detect_fn(sec_logits[0], t, thr, pool_k)  # (N,4) downsampled grid
            c = sp[:, 1:].astype(np.float32) * ds  # to full-res voxel coords
            import torch as _torch
            prob = _torch.sigmoid(sec_logits[0, 0][tuple(sp[:, 1:].astype(np.int64).T)]).float().cpu().numpy() if len(sp) else np.empty((0,), np.float32)
            cents_list.append(c); sizes_list.append((prob * 1e6).astype(np.int64))  # rank by secondary confidence
        cents = np.concatenate(cents_list, 0) if cents_list else np.empty((0, 3), np.float32)
        sizes = np.concatenate(sizes_list, 0) if sizes_list else np.empty((0,), np.int64)
        _CP_STATE["frames"] += 1
        _CP_STATE["raw"] += len(cents)
        _CP_STATE["unet"] += len(arr)
        if len(cents) == 0:
            return arr
        cp_um = cents * vs
        radius = _env_f("BIOHUB_EXTRA_PEAKS_RADIUS_UM", 3.0)
        if len(arr):
            unet_um = arr[:, 1:].astype(np.float32) * ds * vs
            d, _ = cKDTree(unet_um).query(cp_um, k=1, distance_upper_bound=radius)
            novel = ~np.isfinite(d)
        else:
            novel = np.ones(len(cents), bool)
        cents, sizes = cents[novel], sizes[novel]
        if len(cents) == 0:
            return arr
        # largest nuclei first, cap relative to the UNet count
        order = np.argsort(-sizes)
        cap = int(_env_f("BIOHUB_EXTRA_PEAKS_MAX_FRAC", 0.5) * max(len(arr), 1)) + int(_env_f("BIOHUB_EXTRA_PEAKS_MAX_ABS", 0))
        cents = cents[order][:max(cap, 0)]
        grid = np.round(cents / ds).astype(np.int16)
        grid = np.unique(grid, axis=0)
        # drop any that collide with an existing grid peak
        if len(arr):
            existing = {tuple(r) for r in arr[:, 1:].tolist()}
            grid = np.array([g for g in grid.tolist() if tuple(g) not in existing], np.int16).reshape(-1, 3)
        if len(grid) == 0:
            return arr
        extra = np.concatenate([np.full((len(grid), 1), t, np.int16), grid], 1)
        _CP_STATE["added"] += len(extra)
        return np.concatenate([arr, extra], 0).astype(np.int16)
    except Exception as exc:
        print(f"EXTRA_PEAKS: frame {t} failed ({type(exc).__name__}: {exc}); keeping UNet peaks", flush=True)
        return arr
    finally:
        _CP_STATE["secs"] += time.time() - t0


def report(stem: str = ""):
    s = _CP_STATE
    print(f"EXTRA_PEAKS_SUMMARY {stem}: frames={s['frames']} unet={s['unet']} cellpose_raw={s['raw']} added={s['added']} "
          f"({(100.0 * s['added'] / max(s['unet'], 1)):.1f}% of unet) cellpose_secs={s['secs']:.0f} failed={s['failed']}", flush=True)
    for k in ("frames", "raw", "added", "unet", "secs"):
        s[k] = 0

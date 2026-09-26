"""Capture v1284 head features on TRAIN movies and fit our own refinement head.

Why: the public head (`anvithpothula/biohub-v1284-head-s075`) is the only component that ever
gained us LB points (+0.005 -> 0.953). Its author's own note in the notebook says a 4-movie head
was 14.9% WORSE and their 20-movie head reduced centre error 10.8% pooled (16/20 movies improved),
i.e. the mechanism is data-volume-limited. We have 108 train movies with ground truth.

Each variant kernel: P100 guard + the 0.953 pipeline's cells 0..4 (definitions + prediction) with
V1284_MODE=capture, pointed at a slice of TRAIN movies. refine() then dumps
{coords, features} npz per frame without altering coordinates. We then match each captured centre
to the nearest GT node in the same frame, keep pairs within MATCH_UM, and save a compact npz
(subsampled per frame) as the kernel output for the downstream fitting kernel.
"""
import json, os, sys

G = 'kernels/submit-g/base-kunaldesale-public.ipynb.bak'
W = 'kernels/worker-prune3/biohub-w-prune3.ipynb'

CAPTURE_CELL = r'''# ---- capture mode + TRAIN movie slice (replaces the test-set stem list) ----
import os
os.environ['V1284_MODE'] = 'capture'
os.environ['V1284_CAPTURE'] = '/kaggle/working/cap'
CAPTURE_OFFSET = __OFFSET__
CAPTURE_COUNT = __COUNT__
print('CAPTURE slice:', CAPTURE_OFFSET, CAPTURE_COUNT)
'''

LABEL_CELL = r'''# ---- build (feature, displacement) training pairs from the captured frames ----
import numpy as np, os, glob
from pathlib import Path
from scipy.spatial import cKDTree

MATCH_UM = float(os.environ.get('HEAD_MATCH_UM', '3.0'))
MAX_UM = float(os.environ.get('HEAD_MAX_UM', '2.0'))       # the deployed head clamps displacement to 2 um
PER_FRAME = int(os.environ.get('HEAD_PER_FRAME', '400'))   # subsample so the npz output stays small
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)                    # um per ORIGINAL voxel (z, y, x)
SPACING = np.array([1.625, 1.625, 1.625], dtype=np.float32)   # um per DOWNSAMPLED (1,4,4) grid unit
VOXEL = np.array(VOXEL_SCALE_UM, dtype=np.float32)

def graph_from_geff(path):
    import tracksdata as td
    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph

def gt_by_frame(stem):
    """Ground-truth centres per frame in ORIGINAL voxel coordinates, read the way the validator does."""
    graph = graph_from_geff(TRAIN_DIR / f'{stem}.geff')
    by_t = {}
    for row in graph.node_attrs().iter_rows(named=True):
        by_t.setdefault(int(row['t']), []).append((float(row['z']), float(row['y']), float(row['x'])))
    return {t: np.asarray(v, np.float32) for t, v in by_t.items()}

cap_root = Path('/kaggle/working/cap')
# Archive the raw captures BEFORE labelling: prediction costs ~48 GPU-minutes, labelling is cheap and
# can be redone from this tarball in a CPU kernel if anything below misbehaves.
import subprocess as _sp
if cap_root.exists():
    _sp.run(['tar', '-cf', '/kaggle/working/cap_raw.tar', '-C', '/kaggle/working', 'cap'], check=False)
    print('archived raw captures:', round(os.path.getsize('/kaggle/working/cap_raw.tar') / 1e6, 1), 'MB', flush=True)

rows_x, rows_y, rows_stem = [], [], []
stems = sorted(p.name for p in cap_root.iterdir() if p.is_dir()) if cap_root.exists() else []
print('captured movies:', len(stems), stems[:6], flush=True)
rng = np.random.default_rng(0)
for stem in stems:
    try:
        gt = gt_by_frame(stem)
    except Exception as exc:
        print('GT unavailable for', stem, type(exc).__name__, exc, flush=True); continue
    trees = {t: cKDTree(v * VOXEL) for t, v in gt.items() if len(v)}
    kept = 0
    for f in sorted((cap_root / stem).glob('*.npz')):
      try:
        z = np.load(f)
        coords, feats = z['coords'], z['features']
        if not len(coords): continue
        t = int(coords[0, 0])
        tree = trees.get(t)
        if tree is None: continue
        # captured coords are in the DOWNSAMPLED grid; DOWNSAMPLE = (1,4,4) -> um = coord * VOXEL * ds
        det_um = coords[:, 1:].astype(np.float32) * SPACING
        dist, idx = tree.query(det_um, k=1, distance_upper_bound=MATCH_UM)
        ok = np.isfinite(dist)
        if not ok.any(): continue
        delta_um = gt[t][idx[ok]] * VOXEL - det_um[ok]
        # CLIP the target to the head's 2 um bound instead of discarding the sample: grid quantisation
        # is 1.625 um per axis, so a hard 2 um drop threw away ~99% of matches (4785 pairs from 24 movies).
        mag = np.linalg.norm(delta_um, axis=1, keepdims=True)
        scale = np.minimum(1.0, MAX_UM / np.maximum(mag, 1e-6))
        delta_um = delta_um * scale
        X = feats[ok]; Y = (delta_um / SPACING).astype(np.float32)
        if len(X) > PER_FRAME:
            sel = rng.choice(len(X), PER_FRAME, replace=False); X, Y = X[sel], Y[sel]
        rows_x.append(X.astype(np.float32)); rows_y.append(Y); rows_stem.extend([stem] * len(X)); kept += len(X)
      except Exception as exc:
        print('  frame skipped', f.name, type(exc).__name__, exc, flush=True)
    print(f'  {stem}: {kept} pairs', flush=True)

if not rows_x:
    raise RuntimeError('no training pairs captured')
X = np.concatenate(rows_x); Y = np.concatenate(rows_y); S = np.asarray(rows_stem)
print('PAIRS', X.shape, Y.shape, '| mean |delta| um', float(np.mean(np.linalg.norm(Y * SPACING, axis=1))), flush=True)
np.savez_compressed('/kaggle/working/head_pairs.npz', X=X, Y=Y, stem=S)
print('saved head_pairs.npz', round(os.path.getsize('/kaggle/working/head_pairs.npz') / 1e6, 1), 'MB')
import shutil; shutil.rmtree('/kaggle/working/cap', ignore_errors=True)
# keep cap_raw.tar as a kernel output: prediction is ~48 GPU-minutes, relabelling from the tar is free
'''

def build(offset, count, slug):
    g = json.load(open(G)); w = json.load(open(W))
    cells = [w['cells'][0]]                      # P100 compat guard
    cells += g['cells'][:5]                      # definitions + prediction (cell 4 runs the shards)
    # capture-mode env must be set before the V1284 patch in cell 4 -> insert right after cell 0 of g
    cap = CAPTURE_CELL.replace('__OFFSET__', str(offset)).replace('__COUNT__', str(count))
    cells.insert(1, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cap.splitlines(keepends=True)})
    nb = {'cells': cells, 'metadata': g.get('metadata', {}), 'nbformat': g.get('nbformat', 4), 'nbformat_minor': g.get('nbformat_minor', 5)}
    # point the stem list at TRAIN movies, and force capture mode over the base's 'candidate'
    ci = [i for i, c in enumerate(nb['cells']) if 'test_stems = list_test_stems()' in ''.join(c['source'])]
    assert len(ci) == 1, ci
    s = ''.join(nb['cells'][ci[0]]['source'])
    old = 'test_stems = list_test_stems()'
    new = ("""TRAIN_DIR = COMP_DIR / "train"
_all_train = sorted(p.name[:-5] for p in TRAIN_DIR.iterdir() if p.name.endswith(".zarr"))
_test_names = {p.name[:-5] for p in TEST_DIR.iterdir() if p.name.endswith(".zarr")}
_train_only = [s for s in _all_train if s not in _test_names]
test_stems = _train_only[CAPTURE_OFFSET:CAPTURE_OFFSET + CAPTURE_COUNT]
print(f"CAPTURE: {len(test_stems)} train movies (of {len(_train_only)} train-only)", test_stems)
TEST_DIR = TRAIN_DIR  # the predictor reads movies from TEST_DIR""")
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    assert s.count("os.environ['V1284_MODE']='candidate'") == 1
    s = s.replace("os.environ['V1284_MODE']='candidate'", "os.environ['V1284_MODE']=os.environ.get('V1284_MODE','candidate')", 1)
    nb['cells'][ci[0]]['source'] = s.splitlines(keepends=True)
    nb['cells'].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": LABEL_CELL.splitlines(keepends=True)})
    d = f'kernels/capture-head/{slug}'; os.makedirs(d, exist_ok=True)
    json.dump(nb, open(f'{d}/{slug}.ipynb', 'w'), indent=1)
    m = json.load(open('kernels/submit-g/kernel-metadata.json'))
    m.update({'id': f'abhijithneilabraham/{slug}', 'title': slug, 'code_file': f'{slug}.ipynb', 'enable_gpu': True, 'keywords': ['gpu']})
    m.pop('id_no', None)
    for ds in ('subinium/pytorch251-cu118-p100-wheelhouse-public',):
        if ds not in m['dataset_sources']: m['dataset_sources'].append(ds)
    json.dump(m, open(f'{d}/kernel-metadata.json', 'w'), indent=2)
    for c in nb['cells']:
        compile(''.join(c['source']), slug, 'exec')
    print(slug, '| cells', len(nb['cells']), '| movies', offset, '..', offset + count)

if __name__ == '__main__':
    build(0, 24, 'biohub-cap-a')
    build(24, 24, 'biohub-cap-b')

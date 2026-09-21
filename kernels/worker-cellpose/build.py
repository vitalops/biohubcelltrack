"""Build P100 validator workers: 0.948 pipeline + Cellpose extra-candidate injection.
Variants: a = biohub_nuclei3d (public finetuned) 2D+stitch; b = same, true 3D (anisotropy 4);
          g = generic nucleitorch_0 2D+stitch, diameter 15.
"""
import json, os, sys
SRC_NB = 'kernels/worker-prune3/biohub-w-prune3.ipynb'
SRC_META = 'kernels/worker-prune3/kernel-metadata.json'
MODULE = open('kernels/worker-cellpose/extra_peaks.py').read()
DATASETS = ['benjaminparrish/biohub-cellpose-stack-3-1-1-2', 'khanzkhan/biohub-cellpose-model', 'mkamijo/cellpose-offline-wheelhouse']

CELL = r'''# === Cellpose extra-candidate injection (public finetuned nuclei model, no training) ===
import subprocess, sys, glob, os, importlib
_wheel_dirs = sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/*.whl", recursive=True)})
_need = ["fastremap", "fill_voids", "roifile", "natsort"]
_missing = [m for m in _need if importlib.util.find_spec(m) is None]
_cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", "--force-reinstall"]
for d in _wheel_dirs: _cmd += ["--find-links", d]
# Kaggle ships cellpose 4.x, which refuses CP3 checkpoints -> pin the 3.1.1.2 wheel from the attached stack.
_r = subprocess.run(_cmd + ["cellpose==3.1.1.2"] + _missing, capture_output=True, text=True)
print("cellpose deps install rc", _r.returncode, (_r.stderr or "")[-600:])
import cellpose; print("cellpose", cellpose.version if hasattr(cellpose, "version") else "?")
assert str(getattr(cellpose, "version", "")).startswith("3."), "cellpose 3.x required for CP3 checkpoints"
from cellpose import models as _cpm  # import check

_mod = REPO_DIR / "src" / "biohub_tracking" / "extra_peaks.py"
_mod.write_text(__EXTRA_PEAKS_MODULE__)

_ps = REPO_DIR / "scripts" / "predict_unet_transformer.py"
_s = _ps.read_text()
_det_old = "                arr = _detect_cells_pooled(\n                    det_logits[f_idx][0], t, cfg.det_threshold, pool_k,\n                )\n"
_det_new = _det_old + (
    "                if os.environ.get(\"BIOHUB_EXTRA_PEAKS\", \"1\") == \"1\":\n"
    "                    from biohub_tracking.extra_peaks import inject_extra_peaks as _inj_extra\n"
    "                    arr = _inj_extra(arr, t, zarr_arr, downsample, tuple(ds.scale), str(ds_path), sec_logits=_SEC_STASH.pop(int(t), None), pool_k=pool_k, detect_fn=_detect_cells_pooled)\n"
)
assert _s.count(_det_old) == 1, f"extra-peaks anchor count {_s.count(_det_old)}"
_s = _s.replace(_det_old, _det_new, 1)
_fin_old = "    coords = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)\n    # Scale spatial coords back to original resolution.\n"
assert _s.count(_fin_old) == 1, f"extra-peaks report anchor count {_s.count(_fin_old)}"
_s = _s.replace(_fin_old, _fin_old.replace("    # Scale spatial", "    if os.environ.get(\"BIOHUB_EXTRA_PEAKS\", \"1\") == \"1\":\n        from biohub_tracking.extra_peaks import report as _extra_report\n        _extra_report(str(ds_path))\n    # Scale spatial", 1), 1)
_stash_old = "                    secondary_det_aligned = (\n                        (secondary_det - secondary_mean) * scale_ratio + primary_mean\n                    )\n"
assert _s.count(_stash_old) == 1, f"stash anchor count {_s.count(_stash_old)}"
_s = _s.replace(_stash_old, _stash_old + "                    _SEC_STASH[int(frame_indices[f])] = secondary_det.detach()\n", 1)
_cand_old = "        del imgs\n"
assert _s.count(_cand_old) == 1, f"cand-model anchor count {_s.count(_cand_old)}"
_cand_new = (
    "        if os.environ.get(\"BIOHUB_CAND_MODEL_WEIGHTS\"):\n"
    "            global _CAND_MODEL\n"
    "            if _CAND_MODEL is None:\n"
    "                _CAND_MODEL, _, _ = load_model(Path(os.environ[\"BIOHUB_CAND_MODEL_WEIGHTS\"]), device)\n"
    "                print(\"CAND_MODEL loaded:\", os.environ[\"BIOHUB_CAND_MODEL_WEIGHTS\"], flush=True)\n"
    "            _, _cand_det = _CAND_MODEL.encode(imgs)\n"
    "            for f in range(W):\n"
    "                _SEC_STASH[int(frame_indices[f])] = _cand_det[f].detach()\n"
    "            del _cand_det\n"
) + _cand_old
_s = _s.replace(_cand_old, _cand_new, 1)
_s = "_SEC_STASH = {}\n_CAND_MODEL = None\n" + _s
if "\nimport os\n" not in _s: _s = "import os\n" + _s
compile(_s, str(_ps), "exec"); _ps.write_text(_s)
assert "_inj_extra(" in _ps.read_text()
__ENV__
print("EXTRA_PEAKS patch installed:", {k: v for k, v in os.environ.items() if k.startswith("BIOHUB_EXTRA_PEAKS") or k.startswith("BIOHUB_CP_")})
'''

VARIANTS = {
  'a': {'BIOHUB_EXTRA_PEAKS': '1', 'BIOHUB_CP_DO3D': '0', 'BIOHUB_CP_STITCH': '0.3', 'BIOHUB_EXTRA_PEAKS_RADIUS_UM': '3.0', 'BIOHUB_EXTRA_PEAKS_MAX_FRAC': '0.5'},
  'b': {'BIOHUB_EXTRA_PEAKS': '1', 'BIOHUB_CP_DO3D': '1', 'BIOHUB_CP_ANISOTROPY': '4.0', 'BIOHUB_EXTRA_PEAKS_RADIUS_UM': '3.0', 'BIOHUB_EXTRA_PEAKS_MAX_FRAC': '0.5'},
  'p': {'BIOHUB_EXTRA_PEAKS': '1', 'BIOHUB_EXTRA_PEAKS_SOURCE': 'secondary', 'BIOHUB_SEC_PEAKS_THR': '0.985', 'BIOHUB_EXTRA_PEAKS_RADIUS_UM': '3.0', 'BIOHUB_EXTRA_PEAKS_MAX_FRAC': '0.5', 'BIOHUB_SECONDARY_DETECTION_WEIGHT': '0.001'},
  'p2': {'BIOHUB_EXTRA_PEAKS': '1', 'BIOHUB_EXTRA_PEAKS_SOURCE': 'secondary', 'BIOHUB_SEC_PEAKS_THR': '0.985', 'BIOHUB_EXTRA_PEAKS_RADIUS_UM': '3.0', 'BIOHUB_EXTRA_PEAKS_MAX_FRAC': '1.0', 'BIOHUB_EXTRA_PEAKS_MIN_UNET': '400', 'BIOHUB_SECONDARY_DETECTION_WEIGHT': '0.001'},
  'p3': {'BIOHUB_EXTRA_PEAKS': '1', 'BIOHUB_EXTRA_PEAKS_SOURCE': 'secondary', 'BIOHUB_SEC_PEAKS_THR': '0.995', 'BIOHUB_EXTRA_PEAKS_RADIUS_UM': '3.5', 'BIOHUB_EXTRA_PEAKS_MAX_FRAC': '0.5', 'BIOHUB_SECONDARY_DETECTION_WEIGHT': '0.001'},
  'q': {'BIOHUB_EXTRA_PEAKS': '1', 'BIOHUB_EXTRA_PEAKS_SOURCE': 'secondary', 'BIOHUB_SEC_PEAKS_THR': '0.985', 'BIOHUB_EXTRA_PEAKS_RADIUS_UM': '3.0', 'BIOHUB_EXTRA_PEAKS_MAX_FRAC': '0.5', 'BIOHUB_CAND_MODEL_WEIGHTS': '/kaggle/working/cand_model/edge_predictor_best.pth'},
  'q2': {'BIOHUB_EXTRA_PEAKS': '1', 'BIOHUB_EXTRA_PEAKS_SOURCE': 'secondary', 'BIOHUB_SEC_PEAKS_THR': '0.985', 'BIOHUB_EXTRA_PEAKS_RADIUS_UM': '3.0', 'BIOHUB_EXTRA_PEAKS_MAX_FRAC': '1.0', 'BIOHUB_CAND_MODEL_WEIGHTS': '/kaggle/working/cand_model/edge_predictor_best.pth'},
  'q3': {'BIOHUB_EXTRA_PEAKS': '1', 'BIOHUB_EXTRA_PEAKS_SOURCE': 'secondary', 'BIOHUB_SEC_PEAKS_THR': '0.985', 'BIOHUB_EXTRA_PEAKS_RADIUS_UM': '3.0', 'BIOHUB_EXTRA_PEAKS_MAX_FRAC': '1.0', 'BIOHUB_EXTRA_PEAKS_MIN_UNET': '400', 'BIOHUB_CAND_MODEL_WEIGHTS': '/kaggle/working/cand_model/edge_predictor_best.pth'},
  'g': {'BIOHUB_EXTRA_PEAKS': '1', 'BIOHUB_CP_DO3D': '0', 'BIOHUB_CP_MODEL': '/kaggle/input/biohub-cellpose-stack-3-1-1-2/cellpose_models/nucleitorch_0', 'BIOHUB_CP_DIAMETER': '15', 'BIOHUB_EXTRA_PEAKS_RADIUS_UM': '3.0', 'BIOHUB_EXTRA_PEAKS_MAX_FRAC': '0.5'},
}

def build(var):
    nb = json.load(open(SRC_NB)); meta = json.load(open(SRC_META))
    anchor = [i for i, c in enumerate(nb['cells']) if 'SECONDARY_EDGE_TTA_ACTIVE' in ''.join(c['source']) and 'write_text' in ''.join(c['source'])]
    assert len(anchor) == 1, anchor
    env = ''.join(f'os.environ["{k}"] = "{v}"\n' for k, v in VARIANTS[var].items())
    src = CELL.replace('__EXTRA_PEAKS_MODULE__', repr(MODULE)).replace('__ENV__', env)
    nb['cells'].insert(anchor[0] + 1, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)})
    if var.startswith('q'):  # third model: Pepper SWA used only as a candidate source (secondary blend untouched)
        import importlib.util as _ilu; _spec = _ilu.spec_from_file_location('pepper_build', 'kernels/worker-pepper/build.py'); pb = _ilu.module_from_spec(_spec); _spec.loader.exec_module(pb)
        cand_src = ('# Materialize Pepper SWA as a standalone candidate model (pilkwang config.json)\nimport glob, hashlib, shutil, os\n' + pb.find(pb.SWA, 'synthetic_5fold_swa.pth') +
                    '_cd = Path("/kaggle/working/cand_model"); _cd.mkdir(parents=True, exist_ok=True)\n'
                    'shutil.copy(_src, _cd / "edge_predictor_best.pth"); shutil.copy(SECONDARY_WEIGHTS_PATH.parent / "config.json", _cd / "config.json")\n'
                    'print("CAND MODEL = PEPPER SWA:", _src, hashlib.sha256(open(_cd / "edge_predictor_best.pth","rb").read()).hexdigest())\n')
        sec_anchor = [i for i, c in enumerate(nb['cells']) if 'SECONDARY_WEIGHTS_ROOT = ' in ''.join(c['source'])]
        assert len(sec_anchor) == 1, sec_anchor
        nb['cells'].insert(sec_anchor[0] + 1, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cand_src.splitlines(keepends=True)})
        if pb.PEP_DS not in meta['dataset_sources']: meta['dataset_sources'].append(pb.PEP_DS)
    if var.startswith('p'):  # secondary = Pepper SWA (candidate source only; det weight ~0)
        import importlib.util as _ilu; _spec = _ilu.spec_from_file_location('pepper_build', 'kernels/worker-pepper/build.py'); pb = _ilu.module_from_spec(_spec); _spec.loader.exec_module(pb)
        sec_anchor = [i for i, c in enumerate(nb['cells']) if 'SECONDARY_WEIGHTS_ROOT = ' in ''.join(c['source'])]
        assert len(sec_anchor) == 1, sec_anchor
        nb['cells'].insert(sec_anchor[0] + 1, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": pb.SEC_SWAP(pb.SWA, 'synthetic_5fold_swa.pth', 'PEPPER SWA').splitlines(keepends=True)})
        if pb.PEP_DS not in meta['dataset_sources']: meta['dataset_sources'].append(pb.PEP_DS)
    slug = f'biohub-w-cellpose-{var}'
    meta['id'] = f'abhijithneilabraham/{slug}'; meta['title'] = slug; meta['code_file'] = f'{slug}.ipynb'; meta.pop('id_no', None)
    for d in DATASETS:
        if d not in meta['dataset_sources']: meta['dataset_sources'].append(d)
    out = f'kernels/worker-cellpose/{var}'; os.makedirs(out, exist_ok=True)
    json.dump(nb, open(f'{out}/{slug}.ipynb', 'w'), indent=1); json.dump(meta, open(f'{out}/kernel-metadata.json', 'w'), indent=2)
    chk = json.load(open(f'{out}/{slug}.ipynb'))
    compile(''.join(chk['cells'][anchor[0] + 1]['source']), 'cell', 'exec')
    print(var, '->', out, 'cells', len(chk['cells']), 'inserted at', anchor[0] + 1)

for v in (sys.argv[1:] or ['a', 'b']):
    build(v)

#!/usr/bin/env python3
"""Build the division-upweighted finetune training notebook."""
import json, pathlib

CELL0 = r'''
# GPU compat guard (nvidia-smi first, reinstall torch only if P100)
import subprocess, sys, os, glob, shutil
def _gpu():
    try: return subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"], text=True).strip()
    except Exception: return ""
_g = _gpu()
print("GPU:", _g)
if "P100" in _g:
    tw = None
    for p in glob.glob("/kaggle/input/**/torch-2.5*.whl", recursive=True):
        tw = p; break
    if tw:
        tmp="/tmp/p100w"; os.makedirs(tmp, exist_ok=True)
        wd = os.path.dirname(tw)
        for f in os.listdir(wd):
            if f.endswith(".whl"):
                dn = "torch-2.5.1+cu118-cp312-cp312-linux_x86_64.whl" if f.startswith("torch-2.5.1cu118") else f
                shutil.copy(os.path.join(wd,f), os.path.join(tmp,dn))
        subprocess.run([sys.executable,"-m","pip","install","-q","--no-index","--find-links",tmp,"--force-reinstall",os.path.join(tmp,"torch-2.5.1+cu118-cp312-cp312-linux_x86_64.whl")])
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
'''

CELL1 = r'''
# Locate support pack, copy repo, offline-install deps
import os, glob, shutil, subprocess, sys, hashlib
from pathlib import Path

def find_support_pack():
    for pat in ["/kaggle/input/biohub-tracking-support-pack-50ep-v1",
                "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"]:
        if os.path.isdir(pat): return Path(pat)
    hits = glob.glob("/kaggle/input/**/ARTIFACT_MANIFEST.json", recursive=True)
    for h in hits:
        if "support-pack" in h: return Path(h).parent
    raise FileNotFoundError("support pack not found")

SP = find_support_pack()
print("Support pack:", SP)
REPO = Path("/kaggle/working/repo")
if REPO.exists(): shutil.rmtree(REPO)
shutil.copytree(SP / "repo", REPO)
for extra in ["kaggle_test_splits_50ep.json", "kaggle_val_splits.json"]:
    src = SP / extra
    if src.exists(): shutil.copy(src, REPO / extra)

wheels = SP / "wheels"
assert wheels.is_dir(), "wheels dir missing"
pkgs = ["tracksdata","zarr","geff","polars","blosc2","dask","imagecodecs","pyarrow","rustworkx","sqlalchemy","ilpy","pyscipopt"]
r = subprocess.run([sys.executable,"-m","pip","install","-q","--no-index","--find-links",str(wheels)] + pkgs)
print("pip exit:", r.returncode)
if r.returncode != 0:
    for p in pkgs:
        subprocess.run([sys.executable,"-m","pip","install","-q","--no-index","--find-links",str(wheels),p])

WARM = SP / "weights/unet_transformer/split_0/edge_predictor_best.pth"  # unused in big-model scratch run

DATA_ROOT = None
for cand in ["/kaggle/input/competitions/biohub-cell-tracking-during-development",
             "/kaggle/input/biohub-cell-tracking-during-development"]:
    if os.path.isdir(cand): DATA_ROOT = Path(cand); break
assert DATA_ROOT, "competition data not mounted"
TRAIN_DIR = DATA_ROOT / "train"
print("train dir:", TRAIN_DIR, "entries:", len(list(TRAIN_DIR.iterdir())))
'''

CELL2 = r'''
# Build finetune splits: exclude leaked test stems entirely; hold out 4 val stems
import json
TEST_STEMS = {"44b6_0113de3b","44b6_0b24845f","6bba_05b6850b","6bba_05db0fb1"}
VAL_STEMS  = {"44b6_12dfb391","44b6_267148e4","6bba_062c8d37","6bba_07e24132"}
stems = sorted(p.name[:-5] for p in TRAIN_DIR.iterdir() if p.name.endswith(".zarr"))
geff_stems = {p.name[:-5] for p in TRAIN_DIR.iterdir() if p.name.endswith(".geff")}
stems = [s for s in stems if s in geff_stems]
train_stems = [s for s in stems if s not in TEST_STEMS and s not in VAL_STEMS]
val_stems = [s for s in stems if s in VAL_STEMS]
print(f"total={len(stems)} train={len(train_stems)} val={len(val_stems)} excluded_test={len([s for s in stems if s in TEST_STEMS])}")
splits = [{"split": 0, "train": train_stems, "test": val_stems}]
SPLITS_PATH = "/kaggle/working/ft_splits.json"
with open(SPLITS_PATH, "w") as f: json.dump(splits, f, indent=1)
print("splits written:", SPLITS_PATH)
'''

CELL3 = r'''
# Patch the train script: warm start, seed, division loss upweight, last-ckpt, time cap, video cap
from pathlib import Path
script = REPO / "scripts/train_unet_transformer.py"
src = script.read_text()
patches = []

def rep(old, new, must=True):
    global src
    if old not in src:
        assert not must, f"anchor missing: {old[:80]}"
        return
    assert src.count(old) >= 1
    src = src.replace(old, new)
    patches.append(old[:60].replace("\n", " "))

rep("import argparse\n", "import argparse\nimport os\n")

# force math SDPA: mem-efficient kernel exceeds grid.z on sm_60 at large batch
rep("""import tracksdata as td""",
"""import tracksdata as td

if torch.cuda.is_available() and torch.cuda.get_device_capability(0) < (7, 5):
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    print("P100 detected: math SDPA forced", flush=True)""")

# division upweight in loss (all occurrences)
n = src.count("weight[div_rows] = 1.0")
src = src.replace("weight[div_rows] = 1.0",
                  'weight[div_rows] = float(os.environ.get("BIOHUB_DIV_LOSS_WEIGHT", "1.0"))')
patches.append(f"div weight x{n}")
assert n >= 1

# warm start after model creation
rep("""    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=unet_out_channels,
        pos_feat_dim=pos_feat_dim,
    ).to(device)
""",
"""    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=unet_out_channels,
        pos_feat_dim=pos_feat_dim,
    ).to(device)
    _ws = os.environ.get("BIOHUB_WARM_START", "")
    if _ws:
        _state = torch.load(_ws, map_location="cpu", weights_only=True)
        _missing, _unexpected = model.load_state_dict(_state, strict=False)
        print(f"Warm start from {_ws}: missing={len(_missing)} unexpected={len(_unexpected)}", flush=True)
""")

# reproducible-but-different seed
rep("""    unet = TemporalUNet3D(""",
"""    _seed_env = os.environ.get("BIOHUB_TRAIN_SEED", "")
    if _seed_env:
        torch.manual_seed(int(_seed_env)); np.random.seed(int(_seed_env))
        print(f"Train seed: {_seed_env}", flush=True)
    unet = TemporalUNet3D(""")

# optional cap on number of training videos (RAM lever)
rep("""    train_video_data = _load(train_files, "train")""",
"""    _maxv = int(os.environ.get("BIOHUB_MAX_TRAIN_VIDEOS", "0"))
    if _maxv > 0 and len(train_files) > _maxv:
        train_files = train_files[:_maxv]
        print(f"Capped train videos to {_maxv}", flush=True)
    train_video_data = _load(train_files, "train")""")

# time cap + per-epoch last checkpoint
rep("""    for epoch in pbar:
        t0 = time.monotonic()""",
"""    _train_start = time.monotonic()
    for epoch in pbar:
        t0 = time.monotonic()""")

rep("""            f"train={train_time:.1f}s test={test_time:.1f}s",
            flush=True,
        )
""",
"""            f"train={train_time:.1f}s test={test_time:.1f}s",
            flush=True,
        )
        torch.save(
            {k.replace("unet.module.", "unet.", 1): v for k, v in model.state_dict().items()},
            output_dir / "edge_predictor_last.pth",
        )
        _cap_h = float(os.environ.get("BIOHUB_TRAIN_TIME_CAP_H", "0"))
        if _cap_h > 0 and (time.monotonic() - _train_start) > _cap_h * 3600:
            print(f"Time cap {_cap_h}h reached after epoch {epoch}; stopping.", flush=True)
            break
""")

# chunk per-voxel temporal attention to stay under sm_60 kernel grid limits
tu = REPO / "src/biohub_tracking/models/temporal_unet.py"
tsrc = tu.read_text()
old_attn = "        h = self.norm(h)\n        h, _ = self.attn(h, h, h, need_weights=False)\n"
new_attn = (
    "        h = self.norm(h)\n"
    "        _CH = 8192\n"
    "        if h.shape[0] > _CH:\n"
    "            _outs = []\n"
    "            for _i in range(0, h.shape[0], _CH):\n"
    "                _hc = h[_i:_i + _CH]\n"
    "                _oc, _ = self.attn(_hc, _hc, _hc, need_weights=False)\n"
    "                _outs.append(_oc)\n"
    "            h = torch.cat(_outs, dim=0)\n"
    "        else:\n"
    "            h, _ = self.attn(h, h, h, need_weights=False)\n"
)
assert old_attn in tsrc, "attention anchor missing"
tu.write_text(tsrc.replace(old_attn, new_attn))
patches.append("temporal attention chunked (65536)")

script.write_text(src)
print("patches applied:")
for p in patches: print(" -", p)
'''

CELL4 = r'''
# Run finetune training
import subprocess, sys, os, time
env = dict(os.environ)
import torch as _t
_is_t4 = _t.cuda.is_available() and _t.cuda.get_device_capability(0) >= (7, 5)
_batch = "8" if _is_t4 else "4"
if not _is_t4 and os.environ.get("BIOHUB_REQUIRE_T4", "1") != "0":
    print("Non-T4 GPU detected; skipping big-model training (API auto-run guard). UI-save on T4 x2 to train.")
    import sys as _sys; _sys.exit(0)
_warm = ""
import glob as _g
_prev = sorted(_g.glob("/kaggle/input/*/bigmodel/edge_predictor_last.pth")) +         sorted(_g.glob("/kaggle/input/**/bigmodel/edge_predictor_last.pth"))
if _prev:
    _warm = _prev[-1]
    print("CHAINED WARM START:", _warm)
env.update({
    "PYTHONPATH": str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", ""),
    "BIOHUB_WARM_START": _warm,
    "BIOHUB_DIV_LOSS_WEIGHT": "1.0",
    "BIOHUB_TRAIN_SEED": "31337",
    "BIOHUB_TRAIN_TIME_CAP_H": "10.5",
    "BIOHUB_MAX_TRAIN_VIDEOS": "0",
})
cmd = [sys.executable, "scripts/train_unet_transformer.py",
       "--data-dir", str(TRAIN_DIR),
       "--splits", "/kaggle/working/ft_splits.json",
       "--split", "0",
       "--epochs", "60",
       "--lr", "1e-4" if not _warm else "6e-5",
       "--batch-size", _batch,
       "--unet-layers", "48,96,192",
       "--unet-out-channels", "48",
       "--num-workers", "4"]
print("cmd:", " ".join(cmd)); sys.stdout.flush()
t0 = time.time()
r = subprocess.run(cmd, cwd=str(REPO), env=env)
print(f"exit: {r.returncode}, elapsed: {(time.time()-t0)/60:.1f} min")
assert r.returncode == 0, "training failed"
'''

CELL5 = r'''
# Package outputs for use as a Kaggle dataset in inference kernels
import shutil, hashlib, json, os
from pathlib import Path
SRC = REPO / "weights/unet_transformer/split_0"
DST = Path("/kaggle/working/bigmodel")
DST.mkdir(parents=True, exist_ok=True)
manifest = {"artifact_name": "biohub-edge-big-48-96-192-seed31337", "model": {}}
for f in ["edge_predictor_best.pth", "edge_predictor_last.pth", "config.json"]:
    p = SRC / f
    if p.exists():
        shutil.copy(p, DST / f)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        manifest["model"][f] = {"sha256": h, "bytes": p.stat().st_size}
        print(f, h, p.stat().st_size)
Path("/kaggle/working/ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
shutil.rmtree(REPO, ignore_errors=True)  # keep output tidy: weights + manifest + splits only
print("done")
'''

nb = {
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 4,
    "cells": [
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": c.strip().splitlines(keepends=True)}
        for c in [CELL0, CELL1, CELL2, CELL3, CELL4, CELL5]
    ],
}
out = pathlib.Path(__file__).parent / "biohub-edge-ft-div.ipynb"
out.write_text(json.dumps(nb))
print("wrote", out)

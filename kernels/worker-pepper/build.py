"""Build offline P100 validator workers that test Ben Pepper's public weights inside the 0.948 pipeline.
Source: kernels/worker-prune3 (Zharov core + gate v1 + prune3 0.5 + 4-video validator).
Variants:
  d0   : Pepper SWA as SECONDARY, BIOHUB_SECONDARY_DETECTION_WEIGHT=0.001 (edge-only blend; >0 keeps the retention-guard audit alive)
  fam  : Pepper SWA as PRIMARY + Pepper fold2 as SECONDARY (all-Pepper family, default blend weights)
"""
import json, copy, sys, os
SRC_NB = 'kernels/worker-prune3/biohub-w-prune3.ipynb'
SRC_META = 'kernels/worker-prune3/kernel-metadata.json'
PEP_DS = 'bhpepper/biohub-synthetic-5fold-ensemble-v1'
SWA = "0eacacaf0b43bfd5a063495d6991a4911cd045c37650363d5d8826a0e7ed3dd9"
FOLD2 = "7f2ddff9b278da05839303e4b18c47cca56ac90ae85f1a34d057c9f53fa8f4d6"

def find(sha, fname):
    return (f'_c = sorted(set(glob.glob("/kaggle/input/**/{fname}", recursive=True)))\n'
            f'_src = next((c for c in _c if hashlib.sha256(open(c,"rb").read()).hexdigest() == "{sha}"), None)\n'
            f'assert _src, "weights {fname} not found: " + str(_c)\n')

SEC_SWAP = lambda sha, fname, tag: (
    f'# Swap SECONDARY seed -> {tag}\nimport glob, hashlib, shutil, os\n' + find(sha, fname) +
    '_root = SECONDARY_WEIGHTS_ROOT; _tmp = _root.parent / (_root.name + "_real")\n'
    'if _tmp.exists(): shutil.rmtree(_tmp)\n'
    'shutil.copytree(_root, _tmp, copy_function=shutil.copyfile)\n'
    'if _root.is_symlink(): _root.unlink()\nelse: shutil.rmtree(_root)\n'
    '_tmp.rename(_root)\n'
    '_dst = SECONDARY_WEIGHTS_PATH\nif _dst.exists() or _dst.is_symlink(): _dst.unlink()\n'
    'shutil.copy(_src, _dst)\n'
    f'print("SECONDARY = {tag}:", _src, hashlib.sha256(open(_dst,"rb").read()).hexdigest())\n')

PRI_SWAP = lambda sha, fname, tag: (
    f'# Swap PRIMARY edge predictor -> {tag}\nimport glob, hashlib, shutil, os\n' + find(sha, fname) +
    '_wroot = REPO_DIR / "weights"; _wtmp = REPO_DIR / "weights_real"\n'
    'if _wtmp.exists(): shutil.rmtree(_wtmp)\n'
    'shutil.copytree(_wroot, _wtmp, copy_function=shutil.copyfile)\n'
    'if _wroot.is_symlink(): _wroot.unlink()\nelse: shutil.rmtree(_wroot)\n'
    '_wtmp.rename(_wroot)\n'
    '_pdst = REPO_DIR / WEIGHTS_RELATIVE\nif _pdst.exists() or _pdst.is_symlink(): _pdst.unlink()\n'
    'shutil.copy(_src, _pdst)\n'
    f'print("PRIMARY SWAPPED = {tag}:", _src, hashlib.sha256(open(_pdst,"rb").read()).hexdigest())\n')

VARIANTS = {
  'd0':  {'cells': [SEC_SWAP(SWA, 'synthetic_5fold_swa.pth', 'PEPPER SWA') + 'os.environ["BIOHUB_SECONDARY_DETECTION_WEIGHT"] = "0.001"\nprint("secondary detection weight ->", os.environ["BIOHUB_SECONDARY_DETECTION_WEIGHT"])\n']},
  'fam': {'cells': [PRI_SWAP(SWA, 'synthetic_5fold_swa.pth', 'PEPPER SWA'), SEC_SWAP(FOLD2, 'synthetic_fold2_best.pth', 'PEPPER FOLD2')]},
}

def build(var):
    nb = json.load(open(SRC_NB)); meta = json.load(open(SRC_META))
    # anchor: cell defining SECONDARY_WEIGHTS_ROOT (must precede the predict-patch cell)
    anchor = [i for i, c in enumerate(nb['cells']) if 'SECONDARY_WEIGHTS_ROOT = ' in ''.join(c['source'])]
    assert len(anchor) == 1, anchor
    for j, src in enumerate(VARIANTS[var]['cells']):
        nb['cells'].insert(anchor[0] + 1 + j, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)})
    slug = f'biohub-w-pepper-{var}'
    meta['id'] = f'abhijithneilabraham/{slug}'; meta['title'] = slug; meta['code_file'] = f'{slug}.ipynb'
    meta.pop('id_no', None)
    if PEP_DS not in meta['dataset_sources']: meta['dataset_sources'].append(PEP_DS)
    d = f'kernels/worker-pepper/{var}'; os.makedirs(d, exist_ok=True)
    json.dump(nb, open(f'{d}/{slug}.ipynb', 'w'), indent=1)
    json.dump(meta, open(f'{d}/kernel-metadata.json', 'w'), indent=2)
    chk = json.load(open(f'{d}/{slug}.ipynb'))
    ins = [i for i, c in enumerate(chk['cells']) if 'PEPPER' in ''.join(c['source'])]
    print(var, '->', d, '| cells', len(chk['cells']), '| swap cells at', ins)

for v in (sys.argv[1:] or VARIANTS): build(v)

"""Apply the validated Pepper-candidate-injection variant to the submission kernel (0.948 code base).
Usage: python3 kernels/submit-lf/build_inject.py <variant>   (variant from kernels/worker-cellpose/build.py, e.g. p4)
"""
import json, sys, importlib.util as ilu
var = sys.argv[1]
spec = ilu.spec_from_file_location('wb', 'kernels/worker-cellpose/build.py'); wb = ilu.module_from_spec(spec)
sys.argv = ['x', '__none__']  # prevent auto-build on import
try: spec.loader.exec_module(wb)
except KeyError: pass
pspec = ilu.spec_from_file_location('pb', 'kernels/worker-pepper/build.py'); pb = ilu.module_from_spec(pspec); pspec.loader.exec_module(pb)
P = 'kernels/submit-lf/biohub-submit-full.ipynb'; M = 'kernels/submit-lf/kernel-metadata.json'
nb = json.load(open(P)); meta = json.load(open(M))
assert not any('EXTRA_PEAKS' in ''.join(c['source']) or 'PEPPER' in ''.join(c['source']) for c in nb['cells']), 'already injected; restore base first'
assert var.startswith('p'), 'submission builder supports p-variants (Pepper secondary + injection)'
env = ''.join(f'os.environ["{k}"] = "{v}"\n' for k, v in wb.VARIANTS[var].items())
cell = wb.CELL.replace('__EXTRA_PEAKS_MODULE__', repr(wb.MODULE)).replace('__ENV__', env)
anchor = [i for i, c in enumerate(nb['cells']) if 'SECONDARY_EDGE_TTA_ACTIVE' in ''.join(c['source']) and 'write_text' in ''.join(c['source'])]
assert len(anchor) == 1, anchor
nb['cells'].insert(anchor[0] + 1, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell.splitlines(keepends=True)})
sec = [i for i, c in enumerate(nb['cells']) if 'SECONDARY_WEIGHTS_ROOT = ' in ''.join(c['source'])]
assert len(sec) == 1, sec
nb['cells'].insert(sec[0] + 1, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": pb.SEC_SWAP(pb.SWA, 'synthetic_5fold_swa.pth', 'PEPPER SWA').splitlines(keepends=True)})
if pb.PEP_DS not in meta['dataset_sources']: meta['dataset_sources'].append(pb.PEP_DS)
json.dump(nb, open(P, 'w'), indent=1); json.dump(meta, open(M, 'w'), indent=2)
chk = json.load(open(P))
print('variant', var, '| cells', len(chk['cells']), '| swap at', [i for i, c in enumerate(chk['cells']) if 'PEPPER SWA' in ''.join(c['source'])], '| inject at', [i for i, c in enumerate(chk['cells']) if 'EXTRA_PEAKS patch' in ''.join(c['source'])], '| env', wb.VARIANTS[var])

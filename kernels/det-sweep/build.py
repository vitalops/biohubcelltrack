"""LB-guided single-knob variants of our 0.948 kernel (the local validator is anti-correlated, so measure on the LB).
Each variant changes exactly ONE env value in the config cell. Usage: python3 kernels/det-sweep/build.py <name>=<VAR>:<value> ...
"""
import json, os, sys, re

BASE = 'kernels/submit-lf/base-0948.ipynb.bak'
META = 'kernels/submit-lf/kernel-metadata.json'

def build(slug, var, value):
    nb = json.load(open(BASE)); meta = json.load(open(META))
    ci = [i for i, c in enumerate(nb['cells']) if f'os.environ["{var}"]' in ''.join(c['source'])]
    assert len(ci) == 1, f'{var}: {ci}'
    cs = ''.join(nb['cells'][ci[0]]['source'])
    old = re.search(rf'os\.environ\["{var}"\]\s*=\s*[\'"]([^\'"]*)[\'"]', cs)
    assert old, var
    cs2 = cs.replace(old.group(0), f'os.environ["{var}"] = "{value}"', 1)
    assert cs2 != cs and cs2.count(f'os.environ["{var}"]') == 1
    nb['cells'][ci[0]]['source'] = cs2.splitlines(keepends=True)
    # the notebook has a configuration-drift guard that pins expected values; keep it in sync
    gi = [i for i, c in enumerate(nb['cells']) if '_EXPECTED_NUMERIC' in ''.join(c['source'])]
    assert len(gi) == 1, gi
    gs = ''.join(nb['cells'][gi[0]]['source'])
    gm = re.search(rf'("{var}":\s*)([0-9.]+)', gs)
    if gm:
        gs2 = gs.replace(gm.group(0), gm.group(1) + str(float(value)), 1)
        assert gs2 != gs
        nb['cells'][gi[0]]['source'] = gs2.splitlines(keepends=True)
        print(f'  guard {var}: {gm.group(2)} -> {float(value)}')
    else:
        gt = re.search(rf'("{var}":\s*)"([^"]*)"', gs)
        if gt:
            gs2 = gs.replace(gt.group(0), f'"{var}": "{value}"', 1); assert gs2 != gs
            nb['cells'][gi[0]]['source'] = gs2.splitlines(keepends=True)
            print(f'  guard {var}: {gt.group(2)} -> {value}')
    meta.update({'id': f'abhijithneilabraham/{slug}', 'title': slug, 'code_file': f'{slug}.ipynb'}); meta.pop('id_no', None)
    meta['dataset_sources'] = [d for d in meta['dataset_sources'] if 'bhpepper' not in d]
    d = f'kernels/det-sweep/{slug}'; os.makedirs(d, exist_ok=True)
    json.dump(nb, open(f'{d}/{slug}.ipynb', 'w'), indent=1); json.dump(meta, open(f'{d}/kernel-metadata.json', 'w'), indent=2)
    print(f'{slug}: {var} {old.group(1)} -> {value}')

for a in sys.argv[1:]:
    slug, rest = a.split('=', 1); var, value = rest.split(':', 1)
    build(slug, var, value)

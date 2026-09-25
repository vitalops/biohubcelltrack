"""Merge our validated gate v3m fork-pruner into the public 0.951 base (DivNet + global-shift relink).
Source of helpers: kernels/submit-lf/biohub-submit-full.ipynb (0.948 code, cell with mitosis_gate_prob).
"""
import json, re, sys

BASE = 'kernels/submit-g/base-kunaldesale-public.ipynb.bak'
OURS = 'kernels/submit-lf/biohub-submit-full.ipynb'
OUT = 'kernels/submit-h/biohub-submit-h.ipynb'

ours = json.load(open(OURS)); base = json.load(open(BASE))
o = ''.join(ours['cells'][5]['source'])
bi = [i for i, c in enumerate(base['cells']) if 'DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()' in ''.join(c['source'])]
assert len(bi) == 1, bi
b = ''.join(base['cells'][bi[0]]['source'])
assert 'mitosis_gate_prob' not in b, 'gate already merged'
import os as _os; _os.makedirs('kernels/submit-h', exist_ok=True)

def block(text, start_marker, end_marker):
    i = text.index(start_marker); j = text.index(end_marker, i)
    return text[i:j]

# 1. helper block: everything from the gate loader through prune_forks_v3/add_divisions_v3
helpers = block(o, '\ndef _load_mitosis_gate' if '\ndef _load_mitosis_gate' in o else '\ndef mitosis_gate_prob',
                '\nDEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()')
anchor_h = '\nDEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()'
assert b.count(anchor_h) == 1, b.count(anchor_h)
b = b.replace(anchor_h, '\n' + helpers.strip('\n') + '\n' + anchor_h, 1)

# 2. the prune call, spliced before the short-track filter exactly as in our 0.948 kernel
call = '    edges = prune_forks_v3(nodes_by_id, edges, stats, dataset=dataset, frame_cache=None)\n'
assert call in o and o.count(call) == 1
anchor_c = '    nodes_by_id, edges = filter_short_track_components'
assert b.count(anchor_c) == 1, b.count(anchor_c)
b = b.replace(anchor_c, call + anchor_c, 1)

compile(b, 'merged', 'exec')
base['cells'][bi[0]]['source'] = b.splitlines(keepends=True)

# 3. env + dataset wiring for the gate artifacts
cfg = [i for i, c in enumerate(base['cells']) if 'BIOHUB_DET_THRESHOLD' in ''.join(c['source'])][0]
cs = ''.join(base['cells'][cfg]['source'])
add = '\n'.join([
    'os.environ["BIOHUB_MITOSIS_GATE_MIN_PROB"] = "0.5"',
    'os.environ["BIOHUB_MITOSIS_PRUNE_MAX_PROB"] = "0"',
    'os.environ["BIOHUB_PRUNE3_MIN_PROB"] = "0.5"',
    'os.environ["BIOHUB_DIV3_MIN_PROB"] = "0"',
]) + '\n'
base['cells'][cfg]['source'] = (cs.rstrip('\n') + '\n' + add).splitlines(keepends=True)
json.dump(base, open(OUT, 'w'), indent=1)

meta = json.load(open('kernels/submit-g/kernel-metadata.json'))
meta.update({'id': 'abhijithneilabraham/biohub-submit-h', 'title': 'biohub-submit-h', 'code_file': 'biohub-submit-h.ipynb'}); meta.pop('id_no', None)
for d in ('abhijithneilabraham/biohub-mitosis-gate-v1',):
    if d not in meta['dataset_sources']: meta['dataset_sources'].append(d)
meta['kernel_sources'] = ['abhijithneilabraham/biohub-gate3-merge']
json.dump(meta, open('kernels/submit-h/kernel-metadata.json', 'w'), indent=2)

chk = ''.join(json.load(open(OUT))['cells'][bi[0]]['source'])
print('merged: mitosis_gate_prob', chk.count('def mitosis_gate_prob'), '| prune_forks_v3', chk.count('def prune_forks_v3'),
      '| prune call', chk.count('prune_forks_v3(') - 1, '| divnet kept', chk.count('load_divnet'))

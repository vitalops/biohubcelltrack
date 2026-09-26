"""Division-RECALL experiments, measured on 20 train movies (no clicks, P100).

Analysis (biohub-a953) showed divJ = 0.167 with 23 of 29 GT divisions MISSED, while the metric
weights divisions at 0.1 — worth +0.033 at divJ 0.5. The 0.953 pipeline rejects division candidates
with a stack of geometric gates (logs show divergence_rejected in the hundreds-to-thousands per
movie) and we additionally PRUNE forks with gate v3m. Both cut recall.

Base for these variants = kernels/submit-i (0.953 + DivNet scoring at the safe-division admission
site), so DivNet — a trained 3D mitosis CNN — can act as the PROPOSER's gate instead of geometry.

Variants:
  g0  : geometric gates off (divergence + mutual-NN + symmetry), DivNet decides (prob >= 0.50)
  g0d : same but DivNet stricter (0.70), to trade recall against the 7 FP we already carry
  geo : geometric gates off, DivNet OFF — isolates how much the gates alone were costing
"""
import json, os, sys

BASE = 'kernels/submit-i/biohub-submit-i.ipynb'
W = 'kernels/worker-prune3/biohub-w-prune3.ipynb'

VARIANTS = {
    'g0':  {'BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE': '0', 'BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN': '0',
            'BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU': '2.0', 'BIOHUB_SAFE_DIV_FRAME_FRAC_CAP': '0.02',
            'BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP': '0.01', 'DIVNET_MIN_PROB': '0.50'},
    'g0d': {'BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE': '0', 'BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN': '0',
            'BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU': '2.0', 'BIOHUB_SAFE_DIV_FRAME_FRAC_CAP': '0.02',
            'BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP': '0.01', 'DIVNET_MIN_PROB': '0.70'},
    'geo': {'BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE': '0', 'BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN': '0',
            'BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU': '2.0', 'BIOHUB_SAFE_DIV_FRAME_FRAC_CAP': '0.02',
            'BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP': '0.01', 'DIVNET_VERIFY': 'False'},
}

def build(var):
    nb = json.load(open(BASE)); w = json.load(open(W))
    cells = [w['cells'][0]] + nb['cells'] + [w['cells'][8], w['cells'][9]]
    nb2 = {'cells': cells, 'metadata': nb.get('metadata', {}), 'nbformat': nb.get('nbformat', 4), 'nbformat_minor': nb.get('nbformat_minor', 5)}
    ci = [i for i, c in enumerate(nb2['cells']) if 'BIOHUB_DET_THRESHOLD' in ''.join(c['source'])][0]
    cs = ''.join(nb2['cells'][ci]['source']).rstrip('\n')
    cs += '\n\n# ---- division-recall variant: ' + var + ' ----\n'
    cs += 'os.environ["BIOHUB_VALIDATOR_ENABLE"] = "1"\nos.environ["BIOHUB_VALIDATOR_N_PER_TYPE"] = "10"\nos.environ["BIOHUB_PPSWEEP_ENABLE"] = "0"\n'
    for k, v in VARIANTS[var].items():
        if k.startswith('BIOHUB_'):
            cs += f'os.environ["{k}"] = "{v}"\n'
        else:
            cs += f'{k} = {v}\n'          # DIVNET_* are plain globals read via globals().get
    cs += 'print("DIV-RECALL variant ' + var + ' config applied")\n'
    nb2['cells'][ci]['source'] = cs.splitlines(keepends=True)
    slug = f'biohub-divr-{var}'
    d = f'kernels/div-recall/{slug}'; os.makedirs(d, exist_ok=True)
    json.dump(nb2, open(f'{d}/{slug}.ipynb', 'w'), indent=1)
    m = json.load(open('kernels/submit-i/kernel-metadata.json'))
    m.update({'id': f'abhijithneilabraham/{slug}', 'title': slug, 'code_file': f'{slug}.ipynb', 'enable_gpu': True, 'keywords': ['gpu']})
    m.pop('id_no', None)
    for ds in ('subinium/pytorch251-cu118-p100-wheelhouse-public',):
        if ds not in m['dataset_sources']: m['dataset_sources'].append(ds)
    json.dump(m, open(f'{d}/kernel-metadata.json', 'w'), indent=2)
    print(slug, '| cells', len(nb2['cells']), '|', VARIANTS[var])

if __name__ == '__main__':
    for v in (sys.argv[1:] or list(VARIANTS)): build(v)

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
    # DivNet reads raw frames through read_test_frame(), which only looks under TEST_DIR. On the
    # validator path the movies live in train/, so every call raised, divnet_score_division swallowed
    # it and returned None, and NOTHING was ever vetoed (geo and g0 produced byte-identical numbers).
    pi = [i for i, c in enumerate(nb2['cells']) if 'def read_test_frame' in ''.join(c['source'])]
    assert len(pi) == 1, pi
    ps = ''.join(nb2['cells'][pi[0]]['source'])
    old_rd = '    zarr_path = TEST_DIR / f"{dataset}.zarr"\n'
    assert ps.count(old_rd) == 1, ps.count(old_rd)
    new_rd = ('    zarr_path = TEST_DIR / f"{dataset}.zarr"\n'
              '    if not (zarr_path / "0" / "zarr.json").is_file():\n'
              '        _alt = (COMP_DIR / "train") / f"{dataset}.zarr"\n'
              '        if (_alt / "0" / "zarr.json").is_file():\n'
              '            zarr_path = _alt\n')
    ps = ps.replace(old_rd, new_rd, 1)
    # make the veto observable: print the counters the hook maintains
    # surface the swallowed exception: divnet_score_division catches everything and returns None,
    # which is why scored=0 twice in a row told us nothing about WHY.
    old_exc = """    except Exception as e:
        return None"""
    if ps.count(old_exc) == 1:
        ps = ps.replace(old_exc, """    except Exception as e:
        global _DIVNET_ERR_SHOWN
        if not globals().get('_DIVNET_ERR_SHOWN'):
            _DIVNET_ERR_SHOWN = True
            import traceback
            print('DIVNET_SCORE_FAILED:', type(e).__name__, e, flush=True)
            traceback.print_exc()
        return None""", 1)
        ps = '_DIVNET_ERR_SHOWN = False\n' + ps
    # print the veto counters right before the short-track filter (anchor is stable in this base)
    old_pr = '    nodes_by_id, edges = filter_short_track_components(nodes_by_id, edges, stats)'
    assert ps.count(old_pr) == 1, ps.count(old_pr)
    ps = ps.replace(old_pr, '    print(f"  [{dataset}] divnet scored={stats.get(\'divnet_scored\', 0)} vetoed={stats.get(\'divnet_vetoed_divisions\', 0)}", flush=True)\n' + old_pr, 1)
    compile(ps, 'rd', 'exec')
    nb2['cells'][pi[0]]['source'] = ps.splitlines(keepends=True)
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

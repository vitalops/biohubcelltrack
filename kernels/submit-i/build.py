"""Port the DivNet 3D mitosis verifier (from the public a951 kernel) onto OUR 0.948 base.
Single delta: divisions admitted by safe-division repair must also pass DivNet (prob >= DIVNET_MIN_PROB).
Everything else stays exactly as the 0.948 submission (gate v1 + gate v3m fork prune, our relink config).
"""
import json, re, os

OURS = 'kernels/submit-g/base-kunaldesale-public.ipynb.bak'
PUB = 'kernels/submit-c/base-0951-public.ipynb.bak'
OUTDIR = 'kernels/submit-i'; SLUG = 'biohub-submit-i'

pub = json.load(open(PUB)); ours = json.load(open(OURS))
pi = [i for i, c in enumerate(pub['cells']) if 'load_divnet_mitosis_model' in ''.join(c['source'])][0]
ps = ''.join(pub['cells'][pi]['source'])
oi = [i for i, c in enumerate(ours['cells']) if 'DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()' in ''.join(c['source'])][0]
os_ = ''.join(ours['cells'][oi]['source'])
assert 'divnet' not in os_.lower(), 'already ported'

# 1. lift the whole DivNet module (definition -> end of divnet_score_division)
start = ps.index('# DivNet 3D-CNN Mitosis Verification Module')
end_marker = '\ndef '
m = re.search(r'\ndef divnet_score_division\(', ps[start:])
after = ps.index('\n\n\n', start + m.start() + 100)
module = ps[start:after]
assert 'DIVNET_BUNDLE = load_divnet_mitosis_model()' in module and 'def divnet_score_division' in module

# 2. veto hook at the SAFE-DIVISION ADMISSION site (the path our config actually uses;
#    the public kernel's geometry-filter path is disabled in our config, so hooking there is a no-op).
ADMIT_ANCHOR = """            candidate = nodes_by_id[candidate_id]
            added.append({"""
assert os_.count(ADMIT_ANCHOR) == 1, os_.count(ADMIT_ANCHOR)
veto = """            candidate = nodes_by_id[candidate_id]
            if globals().get("DIVNET_VERIFY", True) and globals().get("DIVNET_BUNDLE") is not None:
                _existing = out_by_source.get(source_id, [])
                _sib_edge = _existing[0] if _existing else None
                _sib_id = _sib_edge.get("target_id") if isinstance(_sib_edge, dict) else _sib_edge
                _sib = nodes_by_id.get(int(_sib_id)) if _sib_id is not None else None
                if _sib is not None:
                    _p = divnet_score_division(
                        dataset, int(nodes_by_id[source_id]["t"]), nodes_by_id[source_id],
                        _sib, candidate, globals().get("DIVNET_BUNDLE"), frame_cache,
                    )
                    if _p is not None:
                        stats["divnet_scored"] = stats.get("divnet_scored", 0) + 1
                        if _p < float(globals().get("DIVNET_MIN_PROB", 0.50)):
                            stats["divnet_vetoed_divisions"] = stats.get("divnet_vetoed_divisions", 0) + 1
                            continue
            added.append({"""
os_new = os_.replace(ADMIT_ANCHOR, veto, 1)

# 3. DivNet module must be defined before the pipeline runs: put it just before the DeepCenter detector load
anchor_mod = '\nDEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()'
assert os_new.count(anchor_mod) == 1
os_new = os_new.replace(anchor_mod, '\n' + module.strip('\n') + '\n' + anchor_mod, 1)
compile(os_new, 'submit-d', 'exec')
ours['cells'][oi]['source'] = os_new.splitlines(keepends=True)

# 4. config
ci = [i for i, c in enumerate(ours['cells']) if 'BIOHUB_DET_THRESHOLD' in ''.join(c['source'])][0]
cs = ''.join(ours['cells'][ci]['source'])
cs = cs.rstrip('\n') + '\nDIVNET_VERIFY = True\nDIVNET_MIN_PROB = 0.50\nos.environ["BIOHUB_DIVNET_VERIFY"] = "1"\nos.environ["BIOHUB_DIV_MIN_PROB"] = "0.50"\n'
ours['cells'][ci]['source'] = cs.splitlines(keepends=True)

os.makedirs(OUTDIR, exist_ok=True)
json.dump(ours, open(f'{OUTDIR}/{SLUG}.ipynb', 'w'), indent=1)
meta = json.load(open('kernels/submit-g/kernel-metadata.json'))
meta.update({'id': f'abhijithneilabraham/{SLUG}', 'title': SLUG, 'code_file': f'{SLUG}.ipynb'}); meta.pop('id_no', None)
meta['dataset_sources'] = [d for d in meta['dataset_sources'] if 'bhpepper' not in d]
if 'giorgosi/biohub-divnet-v2' not in meta['dataset_sources']: meta['dataset_sources'].append('giorgosi/biohub-divnet-v2')
json.dump(meta, open(f'{OUTDIR}/kernel-metadata.json', 'w'), indent=2)

chk = ''.join(json.load(open(f'{OUTDIR}/{SLUG}.ipynb'))['cells'][oi]['source'])
print('divnet defs', chk.count('def divnet_score_division'), '| bundle', chk.count('DIVNET_BUNDLE = load_divnet_mitosis_model()'),
      '| veto hook', chk.count('divnet_vetoed_divisions'), '| gate kept', chk.count('def prune_forks_v3'),
      '| order ok', chk.index('DIVNET_BUNDLE =') < chk.index('write_test_submission') if 'write_test_submission' in chk else 'n/a')

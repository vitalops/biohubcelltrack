"""Our addition to the CASMI26 0.336 base: CROSS-SPECTRUM AGREEMENT reranking.

The base fuses a molecule's several spectra *before* ranking: `_logits_from` averages the
per-spectrum fingerprint logits (plus a merged-peak view), so by the time candidates are scored
there is a single evidence vector and the multi-spectrum structure is gone. But the test set has
1213 spectra for 400 molecules — ~3 independent measurements each, often at different collision
energies.

Signal we add: a TRUE structure should score well against *every* spectrum of its molecule, while a
decoy typically fits one spectrum (one fragmentation regime) and not the others. So per candidate we
compute the Bayes fingerprint score against each spectrum separately and summarise the agreement:
  agree_mean : mean over spectra of the rank-normalised per-spectrum score
  agree_min  : worst-case support across spectra
  agree_top  : fraction of spectra ranking the candidate in the top 5
These cannot be added to the shipped GBM (it is fitted from rank_train.npz, 25 precomputed columns,
and we cannot regenerate those rows), so they blend post-hoc with the GBM probability in rank space:
  final = rank(p_gbm) + L_MEAN*agree_mean + L_MIN*agree_min + L_TOP*agree_top
Molecules with a single spectrum are untouched by construction (all three terms reduce to the GBM's
own ordering), so the change is a no-op exactly where it carries no information.
"""
import json, os, sys

BASE = 'kernels/casmi-a/base-public-0336.ipynb.bak'
OUT = 'kernels/casmi-b'
SLUG = 'casmi-b'

# 1. expose per-spectrum logits: return them alongside the fused vector
PATCH_LOGITS = (
    "    # average over the ensemble, then over the molecule's spectra\n"
    "    return np.mean([n(*args).float().mean(0).cpu().numpy() for n in nets], axis=0)",
    "    # average over the ensemble, then over the molecule's spectra\n"
    "    _per_net = [n(*args).float().cpu().numpy() for n in nets]          # each (B, nbits)\n"
    "    _per_spec = np.mean(_per_net, axis=0)                              # (B, nbits) per spectrum\n"
    "    globals()['_LAST_PER_SPECTRUM_LOGITS'] = _per_spec                 # ours: kept for agreement\n"
    "    return _per_spec.mean(0)")

# 2. the agreement features + blend, inserted just before the ranked list is cut
PATCH_RANK = (
    "            p   = RANKER.predict_proba(X)[:, 1]\n"
    "            order = np.argsort(-p)[:CFG.TOPN]",
    """            p   = RANKER.predict_proba(X)[:, 1]

            # ---- OURS: cross-spectrum agreement reranking -------------------------------
            _zs = globals().get('_LAST_PER_SPECTRUM_LOGITS')
            _blend = np.zeros(len(p), np.float32)
            if _zs is not None and getattr(_zs, 'ndim', 0) == 2 and _zs.shape[0] > 1:
                _cf = cfp.astype(np.float32)
                _len = np.sqrt(np.maximum(_cf.sum(1), 1.0))                 # length correction
                _S = (_cf @ _zs.T) / _len[:, None]                          # (n_cand, n_spectra)
                _R = np.empty_like(_S)
                for _j in range(_S.shape[1]):                               # rank-normalise per spectrum
                    _o = np.argsort(np.argsort(_S[:, _j])).astype(np.float32)
                    _R[:, _j] = _o / max(len(_o) - 1, 1)
                _agree_mean = _R.mean(1)
                _agree_min  = _R.min(1)
                _k = min(5, _S.shape[0])
                _agree_top = np.mean([(np.argsort(-_S[:, _j])[:_k, None] == np.arange(len(p))[None, :]).any(0)
                                      for _j in range(_S.shape[1])], axis=0).astype(np.float32)
                _blend = (L_MEAN * _agree_mean + L_MIN * _agree_min + L_TOP * _agree_top).astype(np.float32)
                _n_agree_used += 1
            _pr = np.argsort(np.argsort(p)).astype(np.float32) / max(len(p) - 1, 1)
            order = np.argsort(-(_pr + _blend))[:CFG.TOPN]
            # -----------------------------------------------------------------------------""")

def build(l_mean, l_min, l_top, slug=SLUG, outdir=OUT):
    nb = json.load(open(BASE))
    cells = nb['cells']
    def patch(old, new, label):
        hits = [i for i, c in enumerate(cells) if c['cell_type'] == 'code' and old in ''.join(c['source'])]
        assert len(hits) == 1, f'{label}: {len(hits)} anchors'
        s = ''.join(cells[hits[0]]['source']).replace(old, new, 1)
        compile(s, label, 'exec')
        cells[hits[0]]['source'] = s.splitlines(keepends=True)
    patch(*PATCH_LOGITS, 'per-spectrum-logits')
    patch(*PATCH_RANK, 'agreement-rerank')
    # constants + a counter so the log proves the branch fired
    ci = [i for i, c in enumerate(cells) if c['cell_type'] == 'code' and 'class CFG:' in ''.join(c['source'])][0]
    s = ''.join(cells[ci]['source']).rstrip('\n')
    s += (f"\n\n# ---- OURS: cross-spectrum agreement blend weights ----\n"
          f"L_MEAN = {l_mean}\nL_MIN = {l_min}\nL_TOP = {l_top}\n_n_agree_used = 0\n")
    compile(s, 'cfg', 'exec'); cells[ci]['source'] = s.splitlines(keepends=True)
    # report how many molecules actually used it
    di = [i for i, c in enumerate(cells) if c['cell_type'] == 'code' and "wrote submission.csv" in ''.join(c['source'])][0]
    s = ''.join(cells[di]['source'])
    old = "print(f'\\nwrote submission.csv  {submission.shape}   total {time.time()-T0:.0f}s')"
    assert s.count(old) == 1, s.count(old)
    s = s.replace(old, old + "\nprint(f'AGREEMENT RERANK used on {_n_agree_used}/{len(mols)} molecules  "
                             "(L_MEAN={L_MEAN}, L_MIN={L_MIN}, L_TOP={L_TOP})')", 1)
    compile(s, 'diag', 'exec'); cells[di]['source'] = s.splitlines(keepends=True)
    os.makedirs(outdir, exist_ok=True)
    json.dump(nb, open(f'{outdir}/{slug}.ipynb', 'w'), indent=1)
    m = json.load(open('kernels/casmi-a/kernel-metadata.json'))
    m.update({'id': f'abhijithneilabraham/{slug}', 'title': slug, 'code_file': f'{slug}.ipynb'}); m.pop('id_no', None)
    json.dump(m, open(f'{outdir}/kernel-metadata.json', 'w'), indent=2)
    print(f'{slug}: L_MEAN={l_mean} L_MIN={l_min} L_TOP={l_top}')

if __name__ == '__main__':
    a = [float(x) for x in sys.argv[1:4]] if len(sys.argv) > 3 else [0.25, 0.10, 0.05]
    build(*a, slug=(sys.argv[4] if len(sys.argv) > 4 else SLUG), outdir=(f'kernels/{sys.argv[4]}' if len(sys.argv) > 4 else OUT))

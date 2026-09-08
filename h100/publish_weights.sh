#!/usr/bin/env bash
# Pull a trained checkpoint off the box, back it up (repo + Kaggle dataset), and point the
# submission kernel's swap cell at its sha.  Usage: publish_weights.sh <tag> [best|last]
set -euo pipefail
TAG=$1; WHICH=${2:-best}
W=/home/abhijith/.local/share/.ctw; BOX=abhijith@89.169.102.55
REPO=$(git rev-parse --show-toplevel); cd "$REPO"
mkdir -p weights/$TAG
SRC=${SRC:-$W/wts/$TAG}
scp -q $BOX:$SRC/edge_predictor_${WHICH}.pth weights/$TAG/edge_predictor_best.pth
scp -q $BOX:$SRC/config.json weights/$TAG/config.json
scp -q $BOX:$SRC/epochs_done.txt weights/$TAG/epochs_done.txt 2>/dev/null || true
SHA=$(shasum -a 256 weights/$TAG/edge_predictor_best.pth | cut -d' ' -f1)
echo "sha256: $SHA"
# Kaggle dataset (flat layout: files at root so the swap cell's recursive glob finds them)
DS=biohub-h100-$TAG-v1; DSD=$(mktemp -d)
cp weights/$TAG/edge_predictor_best.pth weights/$TAG/config.json "$DSD"/
printf '{"title":"%s","id":"abhijithneilabraham/%s","licenses":[{"name":"CC0-1.0"}]}\n' "$DS" "$DS" > "$DSD/dataset-metadata.json"
kaggle datasets create -p "$DSD" 2>&1 | tail -1
# point the submission kernel at the new primary
python3 - "$SHA" <<'PY'
import json, sys, re
sha = sys.argv[1]; p = 'kernels/submit-full/biohub-submit-full.ipynb'
nb = json.load(open(p)); n = 0
for c in nb['cells']:
    if c['cell_type'] != 'code': continue
    s = ''.join(c['source'])
    if 'SOUP_SHA = "' in s:
        s = re.sub(r'SOUP_SHA = "[0-9a-f]{64}"', f'SOUP_SHA = "{sha}"', s)
        c['source'] = s.splitlines(keepends=True); n += 1
assert n == 1, n
json.dump(nb, open(p, 'w')); print('submit-full swap cell -> new sha')
PY
python3 - "$DS" <<'PY'
import json, sys
p='kernels/submit-full/kernel-metadata.json'; m=json.load(open(p)); ds=f"abhijithneilabraham/{sys.argv[1]}"
m['dataset_sources']=[d for d in m['dataset_sources'] if 'biohub-h100-' not in d]+[ds]
m['enable_gpu']=False   # API auto-run dies fast on CPU (no quota burn); user re-selects T4 x2 on save anyway
json.dump(m, open(p,'w'), indent=2); print('metadata:', ds)
PY
git add weights/$TAG kernels/submit-full && git commit -q -m "Publish H100 weights $TAG (sha ${SHA:0:12}) -> dataset $DS, submit kernel swap" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019WWt6n9ecgVpr8QzccUzZv" && git push -q origin main && echo "repo synced"
echo "NEXT: wait ~1 min for dataset processing, then: kaggle kernels push -p kernels/submit-full"

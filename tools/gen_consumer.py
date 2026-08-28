#!/usr/bin/env python3
"""Generate + push a CPU consumer kernel from a worker kernel + env overrides.

Usage: gen_consumer.py <kernel_slug_suffix> <worker_kernel_ref> '<json_env_overrides>' [--push]
Example:
  gen_consumer.py v92 abhijithneilabraham/biohub-v91-merge '{"BIOHUB_OUTPUT_MIN_TRACK_LEN":"7"}' --push
"""
import json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
suffix, worker, env_json = sys.argv[1], sys.argv[2], sys.argv[3]
push = "--push" in sys.argv
overrides = json.loads(env_json)

base = json.load(open(HERE / "push/cpupost91/biohub-cpu-post91.ipynb"))
nb = json.loads(json.dumps(base))

# cell 1 = config cell; apply overrides by replacing or appending os.environ lines
cfg = "".join(nb["cells"][1]["source"])
for k, v in overrides.items():
    pat = re.compile(r'os\.environ\["%s"\]\s*=\s*[\'"][^\'"]*[\'"]' % re.escape(k))
    line = f'os.environ["{k}"] = "{v}"'
    if pat.search(cfg):
        cfg = pat.sub(line, cfg, count=1)
    else:
        anchor = "print(\"BIOHUB_PRESET"
        cfg = cfg.replace(anchor, line + "\n" + anchor)
nb["cells"][1]["source"] = cfg.splitlines(keepends=True)

slug = f"biohub-cpu-{suffix}"
outdir = HERE / "push" / f"cpu_{suffix}"
outdir.mkdir(parents=True, exist_ok=True)
json.dump(nb, open(outdir / f"{slug}.ipynb", "w"))

meta = json.load(open(HERE / "push/cpupost91/kernel-metadata.json"))
meta["id"] = f"abhijithneilabraham/{slug}"
meta["title"] = slug
meta["code_file"] = f"{slug}.ipynb"
meta["kernel_sources"] = [worker]
json.dump(meta, open(outdir / "kernel-metadata.json", "w"), indent=2)
print("built", outdir)
for k, v in overrides.items():
    print("  ", k, "=", v)
if push:
    r = subprocess.run(["kaggle", "kernels", "push", "-p", str(outdir)], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())

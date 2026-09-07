#!/usr/bin/env python
"""Patch the support-pack train script for H100 finetuning.

Adds: os import, div-loss weight env, warm-start (BIOHUB_WARM_START), seed env,
per-epoch last checkpoint + time cap, TF32/cudnn.benchmark, and a per-epoch
yield check that stops training (after saving) if a foreign GPU process appears.
Usage: patch_train.py <repo_dir>
"""
import sys
from pathlib import Path

REPO = Path(sys.argv[1])
script = REPO / "scripts/train_unet_transformer.py"
src = script.read_text()
patches = []

def rep(old, new):
    global src
    assert old in src, f"anchor missing: {old[:80]!r}"
    src = src.replace(old, new)
    patches.append(old.strip().splitlines()[0][:60])

rep("import argparse\n", "import argparse\nimport os\nimport subprocess\n")

rep("import tracksdata as td",
"""import tracksdata as td

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def _foreign_gpu_procs() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except Exception:
        return []
    me = str(os.getpid())
    return [l.strip() for l in out.splitlines() if l.strip() and not l.strip().startswith(me + ",")]""")

n = src.count("weight[div_rows] = 1.0")
assert n >= 1
src = src.replace("weight[div_rows] = 1.0",
                  'weight[div_rows] = float(os.environ.get("BIOHUB_DIV_LOSS_WEIGHT", "1.0"))')
patches.append(f"div weight x{n}")

rep("""    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=unet_out_channels,
        pos_feat_dim=pos_feat_dim,
    ).to(device)
""",
"""    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=unet_out_channels,
        pos_feat_dim=pos_feat_dim,
    ).to(device)
    _ws = os.environ.get("BIOHUB_WARM_START", "")
    if _ws:
        _state = torch.load(_ws, map_location="cpu", weights_only=True)
        _missing, _unexpected = model.load_state_dict(_state, strict=False)
        print(f"Warm start from {_ws}: missing={len(_missing)} unexpected={len(_unexpected)}", flush=True)
""")

rep("""    unet = TemporalUNet3D(""",
"""    _seed_env = os.environ.get("BIOHUB_TRAIN_SEED", "")
    if _seed_env:
        torch.manual_seed(int(_seed_env)); np.random.seed(int(_seed_env))
        print(f"Train seed: {_seed_env}", flush=True)
    unet = TemporalUNet3D(""")

rep("""    for epoch in pbar:
        t0 = time.monotonic()""",
"""    _train_start = time.monotonic()
    for epoch in pbar:
        if os.environ.get("BIOHUB_YIELD_CHECK", "1") != "0":
            _foreign = _foreign_gpu_procs()
            if _foreign:
                print(f"YIELD: foreign GPU process present before epoch {epoch}: {_foreign}", flush=True)
                break
        t0 = time.monotonic()""")

rep("""            f"train={train_time:.1f}s test={test_time:.1f}s",
            flush=True,
        )
""",
"""            f"train={train_time:.1f}s test={test_time:.1f}s",
            flush=True,
        )
        torch.save(
            {k.replace("unet.module.", "unet.", 1): v for k, v in model.state_dict().items()},
            output_dir / "edge_predictor_last.pth",
        )
        with open(output_dir / "epochs_done.txt", "a") as _f:
            _f.write(f"{epoch} {score:.5f} {test_acc:.5f} {test_recall:.5f}\\n")
        _cap_h = float(os.environ.get("BIOHUB_TRAIN_TIME_CAP_H", "0"))
        if _cap_h > 0 and (time.monotonic() - _train_start) > _cap_h * 3600:
            print(f"Time cap {_cap_h}h reached after epoch {epoch}; stopping.", flush=True)
            break
""")

script.write_text(src)
print("patches applied:")
for p in patches:
    print(" -", p)

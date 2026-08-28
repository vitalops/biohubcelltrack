#!/usr/bin/env python
"""Run one postproc config against the v91 validator geffs and score with the official-metric reimpl.

Usage: run_config.py <config.json> <out.json>
config.json: {"env": {...BIOHUB_* overrides...}}
Frame-dependent stages are disabled locally (no video zarrs): DeepCenter gap veto, synthetic gap refine.
"""
import json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent

cfg = json.loads(Path(sys.argv[1]).read_text())
out_path = Path(sys.argv[2]).resolve()

os.chdir(HERE)

ns: dict = {"__name__": "__main__"}

# 1) base env from the v91 config cell
exec(compile((HERE / "cell_config.py").read_text(), "cell_config", "exec"), ns)
# 2) local constraints: no frames available
os.environ["BIOHUB_USE_DEEPCENTER_VETO"] = "0"
os.environ["BIOHUB_REQUIRE_DEEPCENTER_VETO"] = "0"
os.environ["BIOHUB_DEEPCENTER_GAP_VETO"] = "0"
os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "0"
os.environ["BIOHUB_GAP_REFINE_SYNTHETIC"] = "0"
os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "0"
# 3) sweep overrides
for k, v in cfg.get("env", {}).items():
    os.environ[k] = str(v)

# 4) constants + postproc defs
exec(compile((HERE / "cell_paths.py").read_text(), "cell_paths", "exec"), ns)
# local layout overrides
ns["REPO_DIR"] = HERE / os.environ.get("HARNESS_REPO_DIR", "tracking_repo")
ns["WORKING_DIR"] = HERE / "work"
ns["WORKING_DIR"].mkdir(exist_ok=True)
ns["TRAIN_DIR"] = HERE / "gt"
ns["TEST_DIR"] = HERE / "gt"  # only used for frame reads, which are disabled
exec(compile((HERE / "cell_postdefs.py").read_text(), "cell_postdefs", "exec"), ns)
ns["DEEPCENTER_VETO_DETECTOR"] = None

# 5) validator constants the scorer cell expects
ns["VALIDATOR_ENABLE"] = True
ns["val_stems"] = ["44b6_12dfb391", "44b6_267148e4", "6bba_062c8d37", "6bba_07e24132"]
ns["VALIDATOR_MATCH_RADIUS_UM"] = 7.0
ns["VALIDATOR_NODE_COUNT_PENALTY_A"] = 0.1
ns["VALIDATOR_DIVISION_WEIGHT"] = 0.1
ns["VALIDATOR_STATS_PATH"] = ns["WORKING_DIR"] / "validator_results.csv"

# 6) run the scorer cell (its script loops over val stems, applies filter_output_graph, scores)
exec(compile((HERE / "cell_score.py").read_text(), "cell_score", "exec"), ns)

summary = ns["validator_summary_rows"][0] if ns["validator_summary_rows"] else {}
rows = ns["validator_sample_rows"]
out_path.write_text(json.dumps({"summary": summary, "rows": rows, "env": cfg.get("env", {})}, default=str))
print("PROXY", summary.get("proxy_score"))

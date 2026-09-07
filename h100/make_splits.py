#!/usr/bin/env python
"""Write ft_splits.json: train = all train stems minus leaked test stems minus 4 val stems."""
import json, sys
from pathlib import Path
TRAIN_DIR = Path(sys.argv[1]); OUT = Path(sys.argv[2])
TEST_STEMS = {"44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"}
VAL_STEMS = {"44b6_12dfb391", "44b6_267148e4", "6bba_062c8d37", "6bba_07e24132"}
geff = {p.name[:-5] for p in TRAIN_DIR.iterdir() if p.name.endswith(".geff")}
zarr = {p.name[:-5] for p in TRAIN_DIR.iterdir() if p.name.endswith(".zarr")}
stems = sorted(geff & zarr)
train = [s for s in stems if s not in TEST_STEMS and s not in VAL_STEMS]
val = [s for s in stems if s in VAL_STEMS]
print(f"total={len(stems)} train={len(train)} val={len(val)} excluded_test={len([s for s in stems if s in TEST_STEMS])}")
OUT.write_text(json.dumps([{"split": 0, "train": train, "test": val}], indent=1))

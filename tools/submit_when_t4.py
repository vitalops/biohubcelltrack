#!/usr/bin/env python
"""Submit the latest completed submit-full version if its run used T4."""
import json, sys, subprocess, os, re, tempfile

KERNEL = "abhijithneilabraham/biohub-submit-full"
COMP = "biohub-cell-tracking-during-development"
MSG = sys.argv[1] if len(sys.argv) > 1 else "full pipeline DET0.965+comboD (proxy 0.9424) on T4"
VERSION = sys.argv[2] if len(sys.argv) > 2 else None

with tempfile.TemporaryDirectory() as td:
    subprocess.run(["kaggle", "kernels", "output", KERNEL, "-p", td, "--force"],
                   capture_output=True, text=True)
    logp = os.path.join(td, "biohub-submit-full.log")
    if not os.path.exists(logp):
        print("NO LOG YET"); sys.exit(2)
    entries = json.load(open(logp))
    text = "".join(e.get("data", "") for e in entries if isinstance(e, dict))

gpu_lines = [l for l in text.split("\n") if l.startswith("GPU:")]
print("GPU line:", gpu_lines[:1])
if not any("T4" in l for l in gpu_lines):
    print("NOT T4 — holding submission"); sys.exit(3)
if "submission.csv" not in text and "our submission" not in text:
    finals = [l for l in text.split("\n") if "FINAL:" in l]
    if not finals:
        print("RUN LOOKS INCOMPLETE — holding"); sys.exit(4)

if VERSION is None:
    print("VERSION required for submit"); sys.exit(5)

import kagglesdk.kaggle_http_client as khc
orig = khc.KaggleHttpClient._prepare_response
def patched(self, response_type, http_response):
    if http_response.status_code >= 400:
        print("STATUS", http_response.status_code)
        print("BODY:", http_response.text[:600])
    return orig(self, response_type, http_response)
khc.KaggleHttpClient._prepare_response = patched
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
r = api.competition_submit_code(file_name="submission.csv", kernel=KERNEL,
                                kernel_version=VERSION, message=MSG, competition=COMP)
print("SUBMITTED:", r)

#!/usr/bin/env python
"""Verify the latest completed run of a kernel ran on T4, then submit it. Usage: verify_submit.py <kernel_slug> <message> <version1,version2,...>"""
import json, sys
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
import kagglesdk.kaggle_http_client as khc
orig = khc.KaggleHttpClient._prepare_response
def patched(self, response_type, http_response):
    if http_response.status_code >= 400:
        print("HTTP", http_response.status_code, http_response.text[:200])
    return orig(self, response_type, http_response)
khc.KaggleHttpClient._prepare_response = patched
slug, msg, versions = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
COMP = "biohub-cell-tracking-during-development"
api = KaggleApi(); api.authenticate()
with api.build_kaggle_client() as kg:
    req = ApiListKernelSessionOutputRequest(); req.user_name = "abhijithneilabraham"; req.kernel_slug = slug
    resp = kg.kernels.kernels_api_client.list_kernel_session_output(req)
text = "".join(e.get("data", "") for e in json.loads(resp.log) if isinstance(e, dict))
gpu = [l for l in text.split("\n") if l.startswith("GPU:") or l.startswith("CUDA device:")][:1]
swap = [l for l in text.split("\n") if "PRIMARY = SOUP" in l or "PRIMARY SWAPPED" in l][:1]
print("GPU:", gpu, "| swap:", swap)
if not any("T4" in g for g in gpu):
    print("NOT T4 -> not submitting"); sys.exit(3)
for v in versions:
    try:
        r = api.competition_submit_code(file_name="submission.csv", kernel=f"abhijithneilabraham/{slug}", kernel_version=v, message=msg, competition=COMP)
        print("SUBMITTED version", v, r); break
    except Exception as e:
        print("version", v, "failed:", str(e)[:100])
else:
    sys.exit(4)
s = api.competition_submissions(COMP)[0]
print("SERVER:", str(s.date)[:16], "|", s.description[:60], "|", s.status)

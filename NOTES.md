# Biohub Cell Tracking — Experiment Ledger

Competition: `biohub-cell-tracking-during-development` (deadline 2026-09-29). Account: neilan.

## Leaderboard trajectory
| Date | Config | LB | Notes |
|---|---|---|---|
| Jul 12 | July pipeline (CSV era) | 0.898 | rank ~1240 |
| Aug 26 | merge + comboD + DET 0.965 | 0.918 | first T4 notebook submission |
| Aug 27 | v91 merge (harmonic core), no comboD | 0.924 | validator ordering inverted vs LB |
| Aug 27 | ritesh-exact reproduction | 0.926 | anchor base |
| Aug 28 | ritesh + mitosis gate (prob ≥ 0.5) | 0.935 | top-50; +0.009 from our learned division gate |
| Aug 30 | ritesh + gate + prune 0.15 | 0.935 | prune arm neutral |
| Aug 31 | **ritesh + gate 0.5 + wide radii 12/15** | **0.941** | **rank 32; gate precision unlocks wider candidate recall** |
| Sep 8 | 0.941 recipe + H100 soupft epoch-0 weights (warm-start from 3-way soup, lr 3e-5, batch 32, TF32) | **0.932** | ref 56092611; REGRESSION: finetuned detector is more conservative → at DET 0.96875 it finds 4–40% fewer cells on public test (44b6_0b24845f: 11.9k vs 19k nodes) → edge FNs. Validator (+0.015) was fooled: its adjusted-jaccard factor >1 when T_pred<T_true rewards under-detection |
| Sep 8 | 0.941 recipe + H100 soupft **epoch-5** as primary | **0.930** | ref 56095685; confirms detection-head degradation (ep0 0.932). Next: H100 weights as SECONDARY seed (S) and HYBRID (primary unet+detect_head + ft transformer, sha a04d9a8f) |
| Sep 9 | 0.941 recipe, original primary + **H100 ep5 as SECONDARY seed** | pending | ref 56105539 (v91-merge v7); detection preserved (25.1k/20.2k/5.9k/69.0k nodes); hybrid transplant cancelled (validated 0.9158 < 0.9202) |

## Key mechanics (hard-won)
- Submissions are **notebook-only** and **rerun on a hidden test set**; precomputed outputs die on rerun.
- **P100 is banned** for submissions; API pushes always get P100. Only web-editor "Save & Run All" runs on the UI-selected T4 x2. Flow: API-push config → one UI save → auto-verify GPU from log → submit via API.
- P100 training quirks: SDPA mem-efficient kernel grid.z cap → force math SDPA + chunk per-voxel attention (8192) below sm_75.
- 4-video local validator: useful for big deltas only; it inverted 0.924 vs 0.926, under-called the gate (+0.001 → +0.009 LB) and over-called the H100 finetune (+0.015 → −0.009 LB). Its adjusted-jaccard REWARDS under-detection (T_pred < T_true ⇒ factor > 1) — always compare T_pred/node counts against the baseline, never trust the proxy alone for model swaps.
- DET_THRESHOLD does NOT recover a degraded detector (ep5 @0.94 → fewer nodes than @0.96875): the finetune softened detection heatmaps (merged peaks in dense regions) while improving association. Use finetuned weights as SECONDARY seed or as a HYBRID (primary UNet/detection + finetuned association head); never as primary. One LB-tested delta at a time.
- Max 2 concurrent GPU sessions; a third UI save cancels running ones.

## Our contributions (beyond public notebooks)
1. **Mitosis gate** (`kernels/mitosis-gate/`): GBM over 15 geometric+intensity features, mined from 148 GT divisions + 6.7k hard negatives across 195 train videos (CV AUC 0.977). Gates safe-division admissions in postproc → +0.009 LB.
2. **Division-upweighted finetune** of the edge predictor (7 epochs; worse as primary swap, reserved as ensemble seed).
3. **Big-model training** (`kernels/train-big-model/`): 48/96/192 TemporalUNet3D from scratch, session-chained checkpoints. In progress.
4. **Local tuning harness** (`harness/`): official-metric scorer + postproc replay on held-out prediction graphs, ~67s/config.
5. **Worker/consumer split + tools** (`tools/`): offline iteration machinery; submission verify+submit automation.

## Layout
- `kernels/submit-full/` — the submission kernel (currently: ritesh base + gate + prune arm)
- `kernels/v91-merge/` — harmonic-merge variant kernel
- `kernels/train-big-model/` — training notebook builder (`build_train_nb.py` → ipynb)
- `kernels/mitosis-gate/` — division-event miner + gate trainer
- `kernels/worker-*/` — validator probe workers
- `harness/` — local sweep/scoring
- `tools/` — consumer generator, T4-verify submitter
- `solution/`, `training/` — July-era artifacts

## Next
- **Frozen-detector finetune (next training design):** freeze `unet`+`detect_head`, train only `transformer` (580k) → detection identical to 0.941 baseline by construction; precompute per-node UNet features once → transformer epochs take seconds (fits Kaggle quota). Motivation: H100 finetune drifted the UNet (detection −4–40% nodes) and the hybrid transplant (primary unet+det + ft transformer) validated WORSE (adj 0.9158 vs 0.9202) — the ft transformer only works on its own UNet features.
- Wide-radii + gate submission (code verified; awaiting a correctly-configured T4 save — NOTE: every API push resets the kernel accelerator, re-select T4 x2 on every save)
- Three-seed blend: third seed trained (warm-start seed 424242, 4 epochs, val acc*recall 0.9797) → dataset `abhijithneilabraham/biohub-edge-thirdseed-424242-v1` (weights.tar); integration = extend the dual-seed logit blend in the inference cell
- Gate v2: richer features (DeepCenter scores, edge probs), threshold sweep
- Prune arm tested: neutral at 0.15 (LB 0.935 = unchanged); big-model scratch training abandoned (4.2h/epoch uneconomical)

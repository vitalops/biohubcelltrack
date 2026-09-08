# Trained weights (backup)

Small files (~8 MB each) kept in-repo because the training box is ephemeral.

| File | Origin | Notes |
|---|---|---|
| `soupft/edge_predictor_ep0.pth` (5a285eb3…) | H100 finetune from 3-way soup, epoch 0 | validator proxy 0.9533; submitted ref 56092611 |
| `soupft/edge_predictor_ep5.pth` = `edge_predictor_best.pth` (f6bbdfe8…) | epoch 5, best per-epoch proxy 0.9803 | submitted via submit-full v22 |
| `soupft/edge_predictor_ep6.pth` = `edge_predictor_last.pth` | epoch 6 (final) | resume point for further training |

Kaggle datasets mirror: `abhijithneilabraham/biohub-mitosis-gate-v1`, `biohub-edge-ft-div-weights-v1`, `biohub-edge-thirdseed-424242-v1`, soup via kernel `biohub-soup`.

# grid/

Preregistered weight-decay x task-size grid at ARC's Phase-2 architecture (1024 x 16). Predictions: `PREREG.md`
(committed before any grid net was trained). `run_grid.py` trains one weight-decay row with the batched trainer and
analyses every net (`analyze_net.py`: remnant decomposition, random-remnant ablation, aligned-slice search, Monte Carlo
truth); outputs are one JSON per net plus the seed-0 weights of each cell. Resumable: finished nets are skipped.
Pilot scripts and results live in `arc-whitebox-canary/phase2/harness/mzc_*`.

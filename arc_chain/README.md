# arc_chain/ — the frozen ARC chain on trained weights

*Written: 2026-10-28 18:42 UTC.*

Chain-side analysis for the ARC Phase 2 write-up `WRITEUP_POST_2026-10-28.md` (forum post text). The nets, trainer
and grid preregistrations live in `../grid/` and `../train/`; this directory holds the scripts that run the frozen
#331937 estimator against Monte Carlo truth on those nets, the extraction tests, and their result files.

- `prereg/` — A1, A1b (synthetic aligned update) and I1–I3 (weights-only extraction) preregistrations, each committed
  before its run; commits and SHA-256 prefixes are in the write-up's §8.
- `results/` — one JSON per analysis (grid 1 P1–P5 incl. the amended-inclusion file, grid 2 Q1–Q5, grid 3 H1–H5,
  grid 4 K1–K4, pilot anatomy / decomposition / ablation / slice search / wd sweep, A1/A1b, I1–I3 summaries).
- `mzc_grid*_prereg_analysis.py`, `mzc_grid_q5_chain.py` — the preregistered analyses; `mzc_*` others — pilot scripts;
  `a1_synthetic_*.py`, `i[123]_*.py` — A1/A1b and I1–I3.

Inputs: per-net JSONs and seed-0 weights from `grid/run_grid.py` (default `~/rosetta_data/mzc_grid*`), and the frozen
estimator (the archive graded as #331937, flopscope 0.12.3), passed by path where a script needs it. Scripts that import
the estimator take its location as an argument or `sys.path` entry; nothing here modifies it.

# MZC Phase-2 sweep (1024x16) — overnight laptop GPU scope

*Written: 2026-09-06 07:11 UTC*

Originally filed as Hopper task `t7617dddf` on 2026-09-05 (on this project's own Hopper
instance) by `claude:mzc-phase2-scope`. Duplicated here as a plain markdown file because
the agent doing the implementation is on a **different Hopper instance/board** and
couldn't see the original task.

## Goal

Extend Model_Zoo_Cartography (currently 1,569 MLPs / 69 families, all trained at ARC
Phase-1 architecture: width 256, depth 32) with a parallel corpus at ARC's actual
Phase-2 spec: **width 1024, depth 16** (resolved 2026-08-23). No such sweep, manifest,
or naming convention exists yet anywhere in the repo (confirmed via grep as of
2026-09-05).

## Hardware

NVIDIA RTX 500 Ada laptop GPU, 4 GiB VRAM, WSL2, ~250GB root disk. Same box already
used to build the Phase-1 MZC corpus (`train/*.py`) AND, as of 2026-09-05, marked
ACTIVE for the arc-whitebox-canary corrector campaign (session logged that day, ~7h of
the 4GB GPU already used per `arc-whitebox-canary/phase2/CORRECTOR_CAMPAIGN.md`).

> **BLOCKING:** verify via `nvidia-smi` / process check that the corrector campaign has
> actually finished before starting this sweep — 4GB can't run both jobs at once. Don't
> trust the doc's "ACTIVE" status without checking live state.

## Biggest technical risk: VRAM, not wall-clock

w=512 already needed batch size dropped 32→16 to fit 4GB
(`notes/2026-08-13_next-steps.md`). w=1024 roughly 4x's per-layer params vs 512 and
~2x's activation memory → batch size will likely need to drop to single digits (B=2–8),
or need fp16/bf16. **Do not queue a multi-hour unattended run before smoke-testing
this** — an OOM 3h into an 8h overnight wastes the whole night.

## Compute cost estimate

Baseline batched trainer ~64s/net at w256/d32 (`train/train_mlp_batched.py`,
`FINDINGS.md:308-309`). Per-net cost scales roughly as depth × width²:
(16/32) × (1024/256)² = 8× baseline FLOPs, before batch-size shrinkage erodes batching
efficiency further. Expect 2–6 min/net pending smoke test.

## Grid

No single manifest exists — Phase-1's 69 families are hardcoded across:
- `train/phase_a.py` — 18 GMM configs, C ∈ {2,3,5,8,10,15,20,25,32,40,50}, separation ∈
  {1.5,2,3,4.5,6}, wd ∈ {0.3,1.0}
- `train/phase_b.py` — ~26 width/depth/C configs, mnist/fashion
- `train/weekend_sweep.py` — C ∈ {32..72} fill-ins

A full 1:1 replica at 1024×16 with 16–32 seeds/config, at ~8×+ cost/net, is **not a
one-night job** — the original build took 3–5 overnights at 1× cost; full Phase-2
parity is more like 1–2+ weeks of overnights. Scope the first night as an MVP, not full
parity.

## Recommended scope for first overnight (MVP)

1. Smoke test at 1024×16: B=1 first (confirm no OOM), then ramp B (2, 4, 8...) to find
   max batch fitting in 4GB with headroom, `--tf32` on. Record real wall-clock/net.
2. If it fits: run a reduced core grid mirroring `phase_a.py`'s GMM backbone (C sweep,
   separation=3.0, wd=0.3, task=gmm) at 8 seeds/config (not 16–32), sized to whatever
   night remains after the smoke test + after the corrector campaign frees the GPU.
3. HF upload: no versioning/subdir convention exists yet (flat `corpus/<run_id>/`
   namespace). Propose `run_id` prefix `p2_w1024_d16_<descriptor>` →
   `corpus/p2_w1024_d16_.../` in the same `james-ra-henry/MZC-Corpus` dataset. Use
   `--upload --prune` (`train/corpus_io.py`) given the laptop's small disk.
4. Leave width/depth/task/wd breadth sweeps (phase_b/weekend_sweep equivalents) for
   later nights once real per-net cost from steps 1–2 is known.

## Other gaps to close before an unattended run

- No pinned Python/dependency versions anywhere (`requirements.txt` has no version
  pins, no environment.yml/pyproject.toml, no Python-version file). Worth freezing
  what's actually installed on the laptop before an unattended run.
- Unconfirmed whether `train/train_mlp_batched.py` has per-net failure isolation
  (skip-and-continue on a single net's OOM/NaN). Phase A logging "zero failures"
  doesn't prove retry logic exists, only that nothing crashed that run. Check before
  trusting this unattended overnight.
- MNIST/FashionMNIST auto-download via torchvision to `~/rosetta_data/mnist`
  (`train/mnist_task.py`) — likely already cached on this box from the Phase-1 build,
  not guaranteed on a clean checkout.

Full investigation detail (exact CLI args, per-config GPU-minute estimates) came from
a session on Rosetta_Program's Hopper board, task `t7617dddf`; re-derive from
`Model_Zoo_Cartography/{README.md,FINDINGS.md,notes/,train/}` if more detail is needed.

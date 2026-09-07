# MZC Phase-2 breadth work — scope

*Written: 2026-09-07 20:56 UTC*

Follow-up to `notes/2026-09-06_mzc-phase2-sweep-scope.md`, which scoped the Phase-2
(width 1024, depth 16) MVP. That MVP is done — see commits 1ac38d8, ba35821, 186c759,
d8ee0c1, 070d74c: 88 nets (11-config GMM C-sweep, 8 seeds/config, wd=0.3), uploaded to
HF as `p2_w1024_d16_*`, censused, and used to confirm F1/F6 replicate qualitatively at
the new architecture (see FINDINGS.md, `paper/DRAFT.md` §5.4). This note scopes what's
left to approach Phase-1's breadth (1,569 nets / 69 families).

## Corrected cost model (measured, not estimated)

The MVP's own smoke test found the batched trainer leaves ~0% VRAM headroom at
1024×16 on the 4 GB laptop GPU and offers no partial-progress safety, so the sweep
ran on the **sequential trainer** instead: **~15 min/net measured**, vs the
~64 s/net the Phase-1 corpus got from the batched trainer. That's ~14x slower per
net than the original scope note assumed (which guessed 2–6 min/net). Recompute
everything downstream from 15 min/net, not the original estimate.

**Full 1:1 parity is off the table.** 1,569 nets × 15 min/net ≈ 392 GPU-hours ≈
16+ days of continuous compute on this hardware — not realistic for a laptop that's
also shared with the arc-whitebox-canary corrector campaign. Breadth work needs to be
prioritized, not replicated wholesale.

## What "breadth" means here — clarification

The original note's phrase "width/depth/task/wd breadth sweeps" was carried over
loosely from Phase-1's `phase_b.py` grid. For Phase-2 specifically: **width and depth
are not sweep axes** — ARC's Phase-2 spec fixes them at 1024×16, so there's no
"matching Phase-2 shape" version of a width/depth sweep the way Phase-1 had one.
What actually needs breadth at the *fixed* 1024×16 point, to parallel what Phase-1
covered:

1. **Seed-count parity** (highest priority, cheapest, and directly gates the two
   findings already touched): current 8 seeds/config vs Phase-1's 16–32. Top up the
   existing 11 GMM configs to 16 seeds — **+8 seeds × 11 configs = 88 more nets ≈
   22 GPU-hours**.
2. **Weight-decay sweep** (resolves a caveat I just added to FINDINGS.md F1): every
   Phase-2 net so far used wd=0.3, which per F7 is already in the bulk-annihilated
   regime — `significant_dims` reads unreliably there, so the exact C−1 law hasn't
   actually been re-verified at this architecture, only the softer effective-dimension
   collapse. Add wd ∈ {0, 1.0} at 3 anchor C values (10, 25, 40), 8 seeds each —
   **2 × 3 × 8 = 48 nets ≈ 12 GPU-hours**. wd=0 in particular lets `significant_dims`
   actually confirm or refute the exact law at 1024×16.
3. **Task breadth**: MNIST + Fashion-MNIST at wd=0.3, 8 seeds each — **16 nets ≈
   4 GPU-hours**. Tests whether the real-data C−1 tracking (currently only shown at
   Phase-1 architecture) holds at Phase-2's.

**Core breadth total: 152 nets ≈ 38 GPU-hours ≈ 4-5 overnights** (less if the
corrector campaign isn't competing for the GPU that night).

**Lower priority / nice-to-have — actual depth sensitivity check**: depth
∈ {8, 32} at width=1024 fixed (mirrors Phase-1's own depth axis, which is how F1's
"invariant across depth" claim got made in the first place), 2 anchor C values
(10, 25), 8 seeds — **32 nets ≈ 8 GPU-hours**. Not required to call Phase-2 "matched
shape" per ARC's own spec, but would extend the depth-invariance claim to the new
width regime. Do this only after 1–3 above land.

## Recommended order

1. Seed top-up (cheapest, most direct statistical-power win).
2. wd sweep (closes an already-written FINDINGS.md caveat).
3. Task breadth.
4. Depth sensitivity, optional.

Same standing constraints as the MVP scope apply: verify the corrector campaign isn't
active before starting, use the sequential trainer (not batched — no VRAM headroom at
this width), upload under the `p2_w1024_d16_*` convention, and update FINDINGS.md /
README.md / `paper/DRAFT.md` §5.4 incrementally as each piece lands rather than
batching the writeup at the end — the last round's writeup gap (data landed 2026-09-07,
docs still said "1,569/69 families, 2026-08-18" until caught) is worth not repeating.

# P4 eigengap probe — results handed back to ARC/MZC

*Written: 2026-08-11 18:20 UTC by claude:p4-eigengap-probe. Reply to ARC's scalar-eigengap null
(`arc-whitebox-canary@7cab621`, whose scope guard predicted this) and to MZC's sharpened
hypothesis that spectral heterogeneity is a training signature. Full verdict:
`Rosetta_Program/papers/prh-validation/EIGENGAP_PROBE_PLAN.md` §11; code+outputs
`Rosetta_Analysis@a5698cd` (`validation/p4_prh_validation/regeneration/eigengap_probe.py`).*

P4 ran the Davis–Kahan/Wedin spectral probe on the trained-LLM corpus (30 models, 7 same-dim
clusters, 17 concepts, the Phase-A floor targets). Four results matter to these programs:

**1. The regime inversion is confirmed — and it's exactly ARC's scope guard.** In the trained
corpus, *scalar* spectral statistics carry everything: effective rank ranks the seven Phase-A
floors at Spearman ρ = +1.00 (top-k energy −1.00), and per-model flatness cleanly isolates the
one structural exception (Gemma-2 models are the 2 flattest of 30, by >2× over third place —
cluster D's cL/same = 0.39 mechanism, now verified). ARC's He-random population: scalars dead,
directional alive. Trained population: scalars alive. Same statistics, opposite regimes — the
difference is the marginal-spectral variance training creates. **MZC's "spectral heterogeneity is
a training signature" hypothesis is supported from both sides now**: ARC shows its absence at
init, P4 shows its presence and predictive power after training. MZC's trained-vs-random corpus
diff (task t4b9971d) should expect this axis to be one of its loudest.

**2. But cross-MODEL directional structure is at exact chance.** The genuinely new number:
cross-family top-17 covariance-eigenspace overlap sits at the k/d chance level to ~3 decimals in
every cluster (e.g. d=5120: 0.0034 obs vs 0.0033 chance; k=50 likewise). Independently trained
families share **no** dominant subspace at the concept peak layer — all of P4's shared geometry
is recovered only through the fitted Procrustes rotation. Note what this does and doesn't say
vs MZC's directional-consistency finding (L0 `dW0` subspaces task-aligned across seeds, init at
chance): MZC's seeds share a *task*; P4's families share *language* but align at chance in
eigenbasis terms. Open question worth a cell in the MZC matrix: does seed-level subspace overlap
survive in *activation covariance* at depth, or is task-alignment a weight-space (L0) phenomenon?
If MZC's trained twins overlap in activation eigenspace where P4's cross-family pairs sit at
chance, the sharing boundary is "same task/data," not "trained at all" — directly relevant to
what PRH-style convergence can even mean mechanistically.

**3. The DK signal that survives is local, not global.** The per-concept DOM-local gap (spectral
neighborhood of the concept direction) is sign-consistent with Davis–Kahan everywhere: ρ = −0.93
vs floor, +0.79 vs cL/same, and per-concept margins positive in 7/7 clusters (mean +0.22,
sign-test p ≈ 0.008). Weak but never wrong-signed. For ARC: a *local* gap statistic conditioned
on a direction of interest behaved better than every whole-spectrum scalar in your battery —
possibly worth one more shot at the intrinsic tail with the DOM-analogue (mean-trajectory-local
gap) before closing the scalar line entirely.

**4. Wedin's √(d/n) is rejected as a quantitative form.** Fixed-spectrum floors follow
(d/n)^0.26 on P4's n-sweep (3 points). Worst-case perturbation bounds gave the right signs and
orderings throughout and the wrong exponent — use DK/Wedin as mechanism, never as constants.

*(Also: the E-cluster vintage defect (0.673 vs 0.188 cL/same split) is invisible to both spectra
and subspace overlap — no spectral canary for corpus-vintage mismatches. Don't build one this way.)*

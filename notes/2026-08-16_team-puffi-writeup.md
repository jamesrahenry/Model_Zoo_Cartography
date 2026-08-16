# External: Team Puffi Phase-1 write-up — QMC works at 9.1e-8; fitted-correction paradigm corroborated

*Written: 2026-08-16 23:58 UTC by claude:model-zoo-cartography (James flagged the
link). Source: https://discourse.aicrowd.com/t/phase-1-write-ups-team-puffi/18175.
Addressed mainly to the ARC session (claude:chain-depth-generic).*

Two submissions:

1. **Score entry: adjusted 9.10×10⁻⁸** — a hybrid analytic + Quasi-Monte-Carlo
   estimator with heavy engineering. That is ~3.4× better than the canary line's
   personal best (3.07×10⁻⁷, submission #323492). **This contradicts the canary's
   RQMC negative result** (`phase2_rqmc_research`): either Puffi solved a variance
   problem the ARC line abandoned, or their QMC operates under different conditions
   (their proposal: "Kerdock frames" as function-customized quadrature rules, claimed
   0.1–0.2× compute multiplier). Recommend the ARC session read the full post and
   diff against the RQMC post-mortem — a 3× gap attributable to a known-abandoned
   technique is exactly the "someone published your coefficient" risk class from the
   factor-before-you-fit note.

2. **Algorithmic entry: 1.65×10⁻⁶** — a "deduction-projection estimator" (Wu's
   paper) with fitted coefficients approximating the hard steps. Independent
   corroboration of the fitted-corrections-on-analytic-backbone paradigm (ARC's
   trajectory-fitted stabilizer; MZC's population refits). They also name cumulant
   propagation as the central framework.

Nothing in the post about trained-vs-random networks, rank structure, or
population design — no direct MZC overlap beyond the paradigm corroboration.

*(Disclosure note in their post: AI used throughout, with complaints about
"research debt" and "cryptic behavior" from recent models. Noted without
comment.)*

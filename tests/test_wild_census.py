"""Pin census/wild_census.py::matrix_census — the §3.8 / §5.5 instrument.

The entry-shuffled empirical null is the foundation of the wild-model line
(and of the next paper), and it carries two documented behaviors this file
locks in: the orientation rule (always census the smaller Gram side — the
first artifact the shuffle null ever caught) and the `bulk_regime` flag that
gates whether MP counts are interpretable (F7).
"""
from __future__ import annotations

import numpy as np

from wild_census import matrix_census

RNG = lambda s: np.random.default_rng(s)  # noqa: E731


def bulk(n=384, d=128, scale=0.05, seed=0):
    return RNG(seed).normal(0.0, scale, (n, d))


def spiked(n=384, d=128, k=3, spike=6.0, bulk_scale=0.05, seed=0):
    """iid bulk + k planted rank-one spikes well above the MP edge."""
    rng = RNG(seed)
    W = rng.normal(0.0, bulk_scale, (n, d))
    for i in range(k):
        u = rng.standard_normal(n) / np.sqrt(n)
        v = rng.standard_normal(d) / np.sqrt(d)
        W += spike * bulk_scale * np.sqrt(n) * np.outer(u, v)
    return W


def test_iid_matrix_reads_null_against_its_own_shuffle():
    c = matrix_census(bulk(), RNG(1))
    assert c["sig_scaled"] <= 2
    assert c["sig_shuffled_null"] <= 2
    # shuffling an iid matrix changes nothing statistically
    assert 0.8 <= c["scale_vs_shuffled"] <= 1.25
    assert c["bulk_regime"] == "intact"


def test_planted_spikes_are_counted_and_shuffling_destroys_them():
    c = matrix_census(spiked(k=3), RNG(2))
    assert c["sig_scaled"] >= 3
    assert c["sig_shuffled_null"] <= 2
    # structure concentrates variance: effective dim below the shuffled copy
    assert c["eff_dim"] < c["eff_dim_shuffled"]
    assert c["bulk_regime"] == "intact"


def test_orientation_rule_makes_census_transpose_invariant():
    """Regression for the documented pythia mlp_in artifact: a wide matrix
    must be censused on its smaller Gram side, so W and W.T read the same."""
    W = spiked(n=384, d=128, k=2, seed=3)
    a = matrix_census(W, RNG(4))
    b = matrix_census(W.T, RNG(5))
    assert a["shape"] == b["shape"]
    assert a["sig_scaled"] == b["sig_scaled"]
    assert a["sig_shuffled_null"] == b["sig_shuffled_null"]


def test_depleted_bulk_is_flagged_not_miscounted():
    """F7's regime gate: spikes riding a near-annihilated bulk. The trained
    matrix's median-MP scale estimate collapses relative to its shuffle
    (which smears spike energy into an iid-like bulk), so the flag must read
    'depleted' — the signal that MP counts are no longer interpretable."""
    c = matrix_census(spiked(k=3, spike=600.0, bulk_scale=0.001, seed=6),
                      RNG(7))
    assert c["scale_vs_shuffled"] < 0.25
    assert c["bulk_regime"] == "depleted"

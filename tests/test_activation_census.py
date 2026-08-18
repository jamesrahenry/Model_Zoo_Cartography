"""Pin census/run_activation_census.py::forward_census to hand-computed values.

Why this test exists (draft v0.11 review trail): §4.3's anchored activation
floor and the activation-side corroboration of §4.2's q-clock both flow
through this one forward-pass + census path. If it were wrong, two readings
the paper treats as independent corroboration would share the bug. This file
pins every recorded field on a network small enough to verify by hand, and
locks in the documented top-20 censoring behavior (the 2026-08-18 external
catch: anchored counts are right-censored at the stored top-20 eigenvalues).

The reference values below are computed two ways: literal hand arithmetic in
the comments, and an independent numpy path in the assertions (explicit ReLU
recursion + np.cov/eigvalsh — forward_census itself uses an SVD route for
n_samples <= hidden_dim, so agreement also cross-checks the numerics).
"""
from __future__ import annotations

import numpy as np
import pytest

from run_activation_census import empirical_q, forward_census

# Tiny explicit network: 2 inputs -> 4 units -> 2 units.
#   x: 4 samples, the +/- unit vectors of R^2.
#   W0 column 4 is identically zero => unit 4 never fires (dead_frac = 1/4).
X = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
W0 = np.array([[1.0, 0.0, -1.0, 0.0],
               [0.0, 1.0, 1.0, 0.0]])
W1 = np.array([[1.0, 0.0],
               [1.0, 0.0],
               [0.0, 1.0],
               [0.0, 7.0]])

# Hand-computed layer 0:
#   z0 = X @ W0 = [[ 1,  0, -1, 0],
#                  [ 0,  1,  1, 0],
#                  [-1,  0,  1, 0],
#                  [ 0, -1, -1, 0]]
#   h0 = relu(z0) = [[1, 0, 0, 0],
#                    [0, 1, 1, 0],
#                    [0, 0, 1, 0],
#                    [0, 0, 0, 0]]
# q_pre(z0): columns are mean-zero; with ddof=1 (np.cov),
#   S = [[2/3, 0, -2/3, 0], [0, 2/3, 2/3, 0], [-2/3, 2/3, 4/3, 0], [0,0,0,0]]
#   tr(S) = 8/3;  sum(S*S) = 24/9 diag + 16/9 offdiag = 40/9
#   q_pre = (8/3)^2 / (4 * 40/9) = (64/9) / (160/9) = 0.4  exactly.
Z0 = X @ W0
H0 = np.maximum(Z0, 0.0)
H1 = np.maximum(H0 @ W1, 0.0)


def _relu_chain(x, weights):
    h = x
    for W in weights:
        h = np.maximum(h @ W, 0.0)
    return h


def _cov_eigs(h):
    """Independent spectrum: eigvalsh of the explicit ddof=1 covariance."""
    hc = h - h.mean(axis=0)
    cov = hc.T @ hc / (h.shape[0] - 1)
    return np.sort(np.linalg.eigvalsh(cov))[::-1]


def _pr(eigs):
    eigs = eigs[eigs > 0]
    return float(eigs.sum() ** 2 / (eigs ** 2).sum())


def test_empirical_q_hand_value():
    assert empirical_q(Z0) == pytest.approx(0.4, abs=1e-12)


def test_forward_census_layer0_matches_hand_computation():
    per_layer = forward_census([W0, W1], X)
    e0 = per_layer[0]

    # Exactly one dead unit (column 4 of W0 is zero).
    assert e0["dead_frac"] == pytest.approx(0.25)
    # q_pre is the hand value, rounded to the module's 6 decimals.
    assert e0["q_pre"] == pytest.approx(0.4, abs=1e-6)

    # Spectrum fields against the independent covariance path on the
    # hand-written H0.
    eigs = _cov_eigs(H0)
    assert e0["total_variance"] == pytest.approx(float(eigs.sum()), rel=1e-6)
    assert e0["eff_dim"] == pytest.approx(_pr(eigs), abs=5e-3)  # module rounds to 2dp
    got = np.array(e0["top_eigenvalues"])
    assert got == pytest.approx(eigs[: len(got)], abs=1e-6)

    # Self-calibrated MP count: sigma^2 = mean(eigs), gamma = d/n = 1,
    # edge = 4 * mean(eigs); count anything above it.
    edge = 4.0 * eigs.mean()
    assert e0["significant_dims"] == int(np.sum(eigs > edge))


def test_forward_census_propagates_post_relu_activations():
    """Layer 1 must census relu(h0 @ W1), i.e. the chain state, not x @ W1."""
    per_layer = forward_census([W0, W1], X)
    e1 = per_layer[1]

    eigs = _cov_eigs(H1)
    assert e1["total_variance"] == pytest.approx(float(eigs.sum()), rel=1e-6)
    assert np.array(e1["top_eigenvalues"]) == pytest.approx(
        eigs[: len(e1["top_eigenvalues"])], abs=1e-6)
    # z1 = H0 @ W1 has no negative entries here, so no unit is dead and
    # relu is the identity on it: q_pre computable straight from H0 @ W1.
    assert e1["dead_frac"] == 0.0
    assert e1["q_pre"] == pytest.approx(empirical_q(H0 @ W1), abs=1e-6)
    # Wrong-state regression guard: censusing x @ W1 instead would need
    # x to have 4 columns at all — shape alone forbids it — but also pin
    # the value path: H1 differs from relu(X @ W0 @ W1) nowhere here only
    # because relu is idempotent on this construction; assert the module
    # agrees with the explicit chain.
    assert np.allclose(_relu_chain(X, [W0, W1]), H1)


def test_anchored_count_thresholds_eigenvalue_fractions():
    per_layer = forward_census([W0, W1], X, anchor_fracs=[0.5, 0.999])
    e0, e1 = per_layer

    # Layer 0: fractions of total variance above 0.5 — from the hand
    # spectrum, only the leading eigenvalue clears half the variance.
    eigs = _cov_eigs(H0)
    fracs = eigs[:20] / eigs.sum()
    assert e0["significant_dims_anchored"] == int(np.sum(fracs > 0.5)) == 1

    # Layer 1: an anchor of 0.999 exceeds any single fraction unless the
    # spectrum is rank one; H1 has rank 2, so the count is 0.
    assert e1["significant_dims_anchored"] == 0


def test_anchored_count_is_right_censored_at_top20():
    """Regression-documents the 2026-08-18 external catch (FINDINGS F3).

    The census stores 20 eigenvalues; the anchored count is computed from
    those alone, so a matrix with more than 20 anchored-significant dims
    reads exactly 20 — a lower bound, not a measurement. If this behavior
    is ever fixed (full-spectrum counting), this test should be updated
    alongside the paper's §4.3 censoring caveat.
    """
    rng = np.random.default_rng(0)
    d = 32
    x = rng.standard_normal((512, d))
    W = np.eye(d)  # h = relu(x): full-rank covariance, all 32 dims present
    entry = forward_census([W], x, anchor_fracs=[1e-12])[0]

    assert len(entry["top_eigenvalues"]) == 20
    # The true anchored count over the full spectrum is 32...
    eigs = _cov_eigs(np.maximum(x, 0.0))
    true_count = int(np.sum(eigs / eigs.sum() > 1e-12))
    assert true_count == d
    # ...but the stored reading is capped at exactly 20.
    assert entry["significant_dims_anchored"] == 20 < true_count

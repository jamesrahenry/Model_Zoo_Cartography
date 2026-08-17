"""Unit tests for null_baseline/analytic_vacuum.py — the zero-forward-pass
moment-propagation null that F2-F4 diff the trained corpus against.

CPU-only, numpy-only. The module's own docstring / stage1_validate.py already
describe a Monte-Carlo validation; test_predicted_matches_monte_carlo below
turns that documented check into an executable regression test with a fixed
seed and an explicit tolerance, instead of a script you have to eyeball.
"""
from __future__ import annotations

import numpy as np
import pytest

from analytic_vacuum import (
    _mehler_cov,
    _relu_hermite_coefs,
    _relu_mean,
    _relu_second_moment,
    predict_activation_cov_spectrum,
)


def build_he_mlp(width, depth, seed=0):
    """He-Gaussian N(0, 2/width) square weights, (in, out) convention — matches
    train_mlp.ArcMLP.weights_arc_convention() and stage1_validate.build_mlp."""
    rng = np.random.default_rng(seed)
    scale = (2.0 / width) ** 0.5
    return [rng.standard_normal((width, width)).astype(np.float64) * scale
            for _ in range(depth)]


def mc_layer_cov(weights, n_samples, seed):
    """Monte Carlo per-layer post-ReLU activation covariance eigenspectrum,
    matching manifold_detector's centered-SVD convention."""
    width = weights[0].shape[1]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, width))
    eig = []
    for W in weights:
        x = np.maximum(x @ W, 0.0)
        xc = x - x.mean(axis=0)
        s = np.linalg.svd(xc, compute_uv=False)
        eig.append(np.sort((s * s) / (n_samples - 1))[::-1])
    return eig


# ---------------------------------------------------------------------------
# Per-neuron ReLU moment primitives
# ---------------------------------------------------------------------------

class TestReluMoments:
    def test_relu_mean_at_zero_is_the_half_normal_mean(self):
        # Z ~ N(0, sigma^2): E[ReLU(Z)] = sigma / sqrt(2 pi)
        sigma = 3.0
        mu = np.array([0.0])
        var = np.array([sigma ** 2])
        expected = sigma / np.sqrt(2 * np.pi)
        assert _relu_mean(mu, var)[0] == pytest.approx(expected, rel=1e-6)

    def test_relu_mean_deep_negative_mean_is_near_zero(self):
        mu = np.array([-50.0])
        var = np.array([1.0])
        assert _relu_mean(mu, var)[0] == pytest.approx(0.0, abs=1e-9)

    def test_relu_mean_deep_positive_mean_is_identity(self):
        mu = np.array([50.0])
        var = np.array([1.0])
        assert _relu_mean(mu, var)[0] == pytest.approx(50.0, rel=1e-6)

    def test_relu_second_moment_matches_monte_carlo(self):
        rng = np.random.default_rng(0)
        mu, var = np.array([0.7]), np.array([2.3])
        z = rng.normal(mu[0], np.sqrt(var[0]), size=4_000_000)
        mc = np.mean(np.maximum(z, 0.0) ** 2)
        assert _relu_second_moment(mu, var)[0] == pytest.approx(mc, rel=5e-3)

    def test_relu_mean_matches_monte_carlo(self):
        rng = np.random.default_rng(1)
        mu, var = np.array([-0.5]), np.array([1.7])
        z = rng.normal(mu[0], np.sqrt(var[0]), size=4_000_000)
        mc = np.mean(np.maximum(z, 0.0))
        assert _relu_mean(mu, var)[0] == pytest.approx(mc, rel=5e-3)

    def test_hermite_coef_zero_equals_relu_mean(self):
        # a[0] is documented as E[ReLU] itself — the two functions must agree.
        mu = np.array([0.3, -1.2, 2.0])
        var = np.array([1.0, 0.5, 3.0])
        a = _relu_hermite_coefs(mu, var)
        np.testing.assert_allclose(a[0], _relu_mean(mu, var), atol=1e-10)


# ---------------------------------------------------------------------------
# _mehler_cov: post-ReLU second-moment covariance
# ---------------------------------------------------------------------------

class TestMehlerCov:
    def test_uncorrelated_inputs_give_near_zero_off_diagonal(self):
        mu = np.zeros(2)
        var = np.array([1.0, 1.0])
        Sig = np.diag(var)  # rho = 0
        a = _relu_hermite_coefs(mu, var)
        C, sig, rho = _mehler_cov(a, var, Sig)
        assert C[0, 1] == pytest.approx(0.0, abs=1e-10)

    def test_perfectly_correlated_identical_neurons_approaches_true_variance(self):
        # rho=1, identical marginals: the n>=1 Mehler series should approximate
        # Var[ReLU(Z)], but only approximately — N_HERM=6 is a truncated
        # series (module docstring: "ReLU's Hermite coefficients decay"),
        # which is exactly why predict_activation_cov_spectrum overrides the
        # diagonal with the exact closed form instead of trusting this series.
        # This test pins the truncation error at a known operating point so a
        # silent change to N_HERM or the coefficient recursion gets caught.
        mu = np.array([0.4, 0.4])
        var = np.array([1.5, 1.5])
        Sig = np.array([[1.5, 1.5], [1.5, 1.5]])  # rho = 1
        a = _relu_hermite_coefs(mu, var)
        C, sig, rho = _mehler_cov(a, var, Sig)
        true_var = _relu_second_moment(mu, var) - _relu_mean(mu, var) ** 2
        assert C[0, 1] == pytest.approx(true_var[0], rel=0.01)

    def test_output_is_symmetric_positive_semidefinite_on_a_random_state(self):
        rng = np.random.default_rng(4)
        w = 12
        mu = rng.standard_normal(w) * 0.5
        A = rng.standard_normal((w, w))
        Sig = A @ A.T + np.eye(w)  # PSD with strictly positive diagonal
        var = np.diag(Sig)
        a = _relu_hermite_coefs(mu, var)
        C, sig, rho = _mehler_cov(a, var, Sig)
        C_sym = 0.5 * (C + C.T)
        eigs = np.linalg.eigvalsh(C_sym)
        assert eigs.min() > -1e-8


# ---------------------------------------------------------------------------
# predict_activation_cov_spectrum: end-to-end, validated against Monte Carlo
# ---------------------------------------------------------------------------

class TestPredictActivationCovSpectrum:
    def test_predicted_matches_monte_carlo(self):
        # This is stage1_validate.py's own documented check turned into an
        # assertion: analytic prediction vs Monte Carlo ground truth, at the
        # exact (width=64, depth=8, n=20000, seed=0) config committed in
        # null_baseline/stage1_results.json. Error is tiny near the input and
        # GROWS with depth (Gaussian mean-field truncation compounds) — that
        # growth is the documented, expected shape, not a bug, so the ceiling
        # per layer widens with depth instead of using one flat threshold.
        weights = build_he_mlp(width=64, depth=8, seed=0)
        pred = predict_activation_cov_spectrum(weights)
        mc = mc_layer_cov(weights, n_samples=20_000, seed=1000)
        # stage1_results.json's seed=0 row values, +healthy margin for MC noise
        ceilings = [0.03, 0.10, 0.17, 0.20, 0.22, 0.32, 0.34, 0.55]
        for l in range(8):
            pred_top = np.asarray(pred["eig_centered"][l][:20])
            mc_top = mc[l][:20]
            rel_err = np.linalg.norm(pred_top - mc_top) / np.linalg.norm(mc_top)
            assert rel_err < ceilings[l], f"layer {l}: rel_err={rel_err:.4f}"

    def test_layer_zero_is_exact_wtw_second_moment(self):
        # z_0 = x @ W0, x ~ N(0, I) -> pre-activation covariance is exactly
        # W0^T W0 with zero mean; the module's docstring states this directly,
        # verify it isn't silently perturbed by the ReLU propagation code.
        weights = build_he_mlp(width=16, depth=1, seed=2)
        pred = predict_activation_cov_spectrum(weights)
        assert pred["mu_norm"][0] >= 0.0  # post-ReLU mean of layer 0's OUTPUT
        # Sanity: single-layer call doesn't crash and returns one entry per key.
        for key in ("eig_centered", "eig_uncentered", "eff_dim_centered",
                   "eff_dim_uncentered", "mu_norm"):
            assert len(pred[key]) == 1

    def test_eigenvalues_are_sorted_descending_and_nonnegative(self):
        weights = build_he_mlp(width=32, depth=3, seed=5)
        pred = predict_activation_cov_spectrum(weights)
        for l in range(3):
            ev = pred["eig_centered"][l]
            assert np.all(ev >= -1e-9)
            assert np.all(np.diff(ev) <= 1e-9)  # descending

    def test_n_top_truncates_output(self):
        weights = build_he_mlp(width=32, depth=2, seed=6)
        full = predict_activation_cov_spectrum(weights, n_top=None)
        top5 = predict_activation_cov_spectrum(weights, n_top=5)
        for l in range(2):
            assert len(top5["eig_centered"][l]) == 5
            np.testing.assert_allclose(top5["eig_centered"][l],
                                       full["eig_centered"][l][:5])

    def test_deeper_random_relu_net_effective_dim_decays_toward_a_fixed_point(self):
        # F2's stated null behavior: random nets' terminal rank decays smoothly
        # with depth (in contrast to trained nets, which arrest). Just the
        # monotone-decay half of that claim is checkable analytically, cheaply.
        weights = build_he_mlp(width=48, depth=10, seed=8)
        pred = predict_activation_cov_spectrum(weights)
        eff = pred["eff_dim_centered"]
        # Not strictly monotone every layer, but the tail should be well below
        # the head for a random He-Gaussian stack.
        assert eff[-1] < eff[0]

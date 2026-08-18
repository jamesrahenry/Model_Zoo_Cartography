"""Unit tests for null_baseline/chain_state_keyed.py — the state-keyed
stabilizer machinery behind F4 (population-refit corrections to the analytic
null). Only the pure-numpy chain math is exercised here (step_np, state_basis,
hermite_coeffs, corrections, load_ref_coefs) — the fitting/eval CLI at the
bottom of the module needs the HF parquet atlas and is out of scope.

Requires pyarrow to be importable (the module does `import pyarrow.parquet`
at module scope even though these tests never touch parquet data).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyarrow")

import chain_state_keyed as csk
from analytic_vacuum import _relu_mean as vacuum_relu_mean


# ---------------------------------------------------------------------------
# phi / Phi — standard normal pdf/cdf
# ---------------------------------------------------------------------------

class TestNormalPrimitives:
    def test_phi_matches_known_values(self):
        assert csk.phi(np.array([0.0]))[0] == pytest.approx(1.0 / np.sqrt(2 * np.pi), rel=1e-9)

    def test_Phi_matches_known_values(self):
        # Phi is an Abramowitz-Stegun erf approximation — bound its error.
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        from math import erf
        exact = np.array([0.5 * (1 + erf(xi / np.sqrt(2))) for xi in x])
        np.testing.assert_allclose(csk.Phi(x), exact, atol=1e-6)

    def test_Phi_is_monotone_and_bounded(self):
        x = np.linspace(-8, 8, 200)
        vals = csk.Phi(x)
        assert np.all(np.diff(vals) >= -1e-12)
        assert vals.min() >= 0.0 and vals.max() <= 1.0


# ---------------------------------------------------------------------------
# hermite_coeffs: cross-checked against analytic_vacuum's independent
# reimplementation of the same ReLU-Hermite moment math.
# ---------------------------------------------------------------------------

class TestHermiteCoeffs:
    def test_coef_zero_equals_relu_mean(self):
        a = np.array([0.5, -1.5, 2.0, 0.0])
        coefs = csk.hermite_coeffs(a, K=6)
        expected = csk.phi(a) + a * csk.Phi(a)  # E[ReLU] at unit variance (a = alpha)
        np.testing.assert_allclose(coefs[0], expected, atol=1e-10)

    def test_coef_one_equals_Phi(self):
        a = np.array([0.5, -1.5, 2.0, 0.0])
        coefs = csk.hermite_coeffs(a, K=6)
        np.testing.assert_allclose(coefs[1], csk.Phi(a), atol=1e-10)

    def test_agrees_with_analytic_vacuum_independent_implementation(self):
        # analytic_vacuum.py and chain_state_keyed.py are two separately
        # vendored (different source repos, per PROVENANCE.md) reimplementations
        # of the same ReLU-Hermite moment expansion. At unit pre-activation
        # variance (sigma=1, so alpha=mu) their zeroth coefficients — both
        # documented as E[ReLU(Z)] — must agree; a mismatch would mean one of
        # the two vendored copies has drifted from the shared underlying math.
        mu = np.array([-2.0, -0.3, 0.0, 0.7, 3.0])
        var = np.ones_like(mu)
        csk_coef0 = csk.hermite_coeffs(mu, K=6)[0]
        vacuum_mean = vacuum_relu_mean(mu, var)
        # chain_state_keyed.Phi is an Abramowitz-Stegun erf approximation
        # (~1e-7 max error) vs analytic_vacuum's exact math.erf — tolerance
        # reflects that known approximation error, not exact equality.
        np.testing.assert_allclose(csk_coef0, vacuum_mean, atol=1e-6)


# ---------------------------------------------------------------------------
# step_np: one uncorrected chain step (mean-field moment propagation)
# ---------------------------------------------------------------------------

class TestStepNp:
    def test_matches_monte_carlo_on_a_random_he_gaussian_layer(self):
        rng = np.random.default_rng(0)
        w = 48
        W = rng.standard_normal((w, w)) * np.sqrt(2.0 / w)
        m_act, S_act = np.zeros(w), np.eye(w)  # x ~ N(0, I)
        m_post, S_post, mu, sig, Spre, a, rho = csk.step_np(m_act, S_act, W)

        n = 200_000
        x = rng.standard_normal((n, w))
        z = x @ W
        relu = np.maximum(z, 0.0)
        mc_mean = relu.mean(axis=0)
        mc_cov = np.cov(relu, rowvar=False)

        np.testing.assert_allclose(m_post, mc_mean, atol=0.02)
        # diagonal (variance) comparison — the off-diagonal Mehler series is
        # covered by analytic_vacuum's own MC test; here just check the shape
        # and that variances land in the right ballpark.
        np.testing.assert_allclose(np.diag(S_post), np.diag(mc_cov), rtol=0.15)

    def test_output_shapes_and_psd_like_diagonal(self):
        rng = np.random.default_rng(1)
        w = 10
        W = rng.standard_normal((w, w)) * 0.1
        m_post, S_post, mu, sig, Spre, a, rho = csk.step_np(np.zeros(w), np.eye(w), W)
        assert m_post.shape == (w,)
        assert S_post.shape == (w, w)
        assert np.all(np.diag(S_post) > 0)

    def test_zero_weights_collapse_to_zero_mean_and_relu_of_zero_variance(self):
        w = 5
        W = np.zeros((w, w))
        m_post, S_post, mu, sig, Spre, a, rho = csk.step_np(np.zeros(w), np.eye(w), W)
        # pre-activation is deterministically 0 -> ReLU(0) = 0 exactly
        np.testing.assert_allclose(m_post, np.zeros(w), atol=1e-8)
        np.testing.assert_allclose(np.diag(S_post), np.zeros(w), atol=1e-8)


# ---------------------------------------------------------------------------
# state_basis: the rank-collapse clock q and friends
# ---------------------------------------------------------------------------

class TestStateBasis:
    def test_q_is_one_for_isotropic_covariance(self):
        # q = PR(Spre)/w; an isotropic (identity-proportional) covariance has
        # participation ratio == w exactly, so q == 1 (the documented "(0,1]" max).
        w = 20
        Spre = np.eye(w) * 3.7
        a = np.zeros(w)
        rho = np.eye(w)
        g = csk.state_basis(Spre, a, rho)
        assert g[1] == pytest.approx(1.0, abs=1e-9)  # g[1] is q per state_basis's own ordering

    def test_q_shrinks_toward_zero_for_rank_one_covariance(self):
        w = 20
        v = np.zeros(w); v[0] = 1.0
        Spre = np.outer(v, v)  # rank 1 -> PR = 1 -> q = 1/w
        a = np.zeros(w)
        rho = np.eye(w)
        g = csk.state_basis(Spre, a, rho)
        assert g[1] == pytest.approx(1.0 / w, rel=1e-6)

    def test_basis_vector_length_and_bias_term(self):
        w = 8
        Spre = np.eye(w)
        a = np.array([0.1] * w)
        rho = np.eye(w)
        g = csk.state_basis(Spre, a, rho)
        assert g.shape == (csk.N_BASIS,)
        assert g[0] == 1.0  # documented bias/intercept slot

    def test_abar_and_asd_are_clipped(self):
        w = 6
        Spre = np.eye(w)
        a = np.full(w, 1000.0)  # way outside the documented clip range
        rho = np.eye(w)
        g = csk.state_basis(Spre, a, rho)
        abar = g[3]
        assert abar == pytest.approx(8.0)  # clipped to the documented [-8, 8]


# ---------------------------------------------------------------------------
# corrections / zero_theta: the zero-theta chain must equal the uncorrected chain
# ---------------------------------------------------------------------------

class TestCorrections:
    def test_zero_theta_applies_no_correction(self):
        rng = np.random.default_rng(2)
        w = 10
        theta = csk.zero_theta()
        g = rng.standard_normal(csk.N_BASIS)
        Fm = rng.standard_normal((w, csk.NMF))
        Fv = rng.standard_normal((w, csk.NVF))
        Fo = [rng.standard_normal((w, w)), rng.standard_normal((w, w))]
        dm, dv, dS = csk.corrections(theta, g, Fm, Fv, Fo)
        np.testing.assert_allclose(dm, np.zeros(w))
        np.testing.assert_allclose(dv, np.zeros(w))
        np.testing.assert_allclose(dS, np.zeros((w, w)))

    def test_zero_theta_shapes(self):
        Tm, Tv, To = csk.zero_theta()
        assert Tm.shape == (csk.NMF, csk.N_BASIS)
        assert Tv.shape == (csk.NVF, csk.N_BASIS)
        assert To.shape == (csk.NOF, csk.N_BASIS)


# ---------------------------------------------------------------------------
# load_ref_coefs: the frozen, committed stabilizer coefficients must parse
# ---------------------------------------------------------------------------

class TestRefCoefs:
    def test_loads_32_layers_of_the_documented_shapes(self):
        ref = csk.load_ref_coefs()
        assert len(ref) == 32
        for Am, Av, Ao in ref:
            assert Am.shape == (csk.NMF,)
            assert Av.shape == (csk.NVF,)
            assert Ao.shape == (csk.NOF,)

    def test_roll_chain_ref_runs_on_a_random_net(self):
        rng = np.random.default_rng(3)
        w, depth = csk.W_DIM, 4
        W = rng.standard_normal((depth, w, w)) * np.sqrt(2.0 / w)
        probe = csk.probe_cumulants(W, n_probe=256, seed=42)
        ref = csk.load_ref_coefs()
        means = csk.roll_chain_ref(W, probe, ref)
        assert len(means) == depth
        for m in means:
            assert m.shape == (w,)
            assert np.all(np.isfinite(m))
            assert np.all(m >= 0.0)  # post-ReLU means clipped nonnegative

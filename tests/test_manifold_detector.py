"""Unit tests for census/manifold_detector.py — the MP-floor / significant-dims
/ participation-ratio machinery that FINDINGS.md's F1/F3/F7 all sit on top of.

CPU-only, numpy-only, no corpus/network access required.
"""
from __future__ import annotations

import numpy as np
import pytest

from manifold_detector import (
    _layer_census,
    _mp_median_factor,
    _mp_upper_edge,
    _participation_ratio,
    estimate_mp_variance,
    layer_manifold_census,
)


# ---------------------------------------------------------------------------
# _mp_upper_edge
# ---------------------------------------------------------------------------

class TestMPUpperEdge:
    def test_square_aspect_ratio_is_four_sigma_sq(self):
        # gamma = n_features/n_samples = 1 -> lambda+ = sigma^2 * (1+1)^2 = 4 sigma^2
        assert _mp_upper_edge(n_samples=1000, n_features=1000, variance=1.0) == pytest.approx(4.0)

    def test_scales_linearly_with_variance(self):
        edge1 = _mp_upper_edge(500, 200, variance=1.0)
        edge2 = _mp_upper_edge(500, 200, variance=3.5)
        assert edge2 == pytest.approx(3.5 * edge1)

    def test_wider_aspect_ratio_raises_the_edge(self):
        # more features per sample (gamma up) -> higher noise floor
        narrow = _mp_upper_edge(n_samples=1000, n_features=100)
        wide = _mp_upper_edge(n_samples=1000, n_features=900)
        assert wide > narrow

    def test_more_samples_lowers_the_edge(self):
        few = _mp_upper_edge(n_samples=100, n_features=100)
        many = _mp_upper_edge(n_samples=10_000_000, n_features=100)
        assert many < few
        # as n_samples -> infinity at fixed n_features, edge -> variance
        assert many == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# _mp_median_factor / estimate_mp_variance
# ---------------------------------------------------------------------------

class TestMPMedianFactor:
    def test_gamma_one_matches_montecarlo_square_gaussian_spectrum(self):
        # At gamma=1 (square case) the quadrature result should match the
        # empirical median eigenvalue of an actual square Gaussian sample
        # covariance matrix (a genuinely MP-distributed spectrum at sigma=1).
        factor = _mp_median_factor(1.0)
        assert 0.0 < factor < 4.0  # support is [(1-1)^2, (1+1)^2] = [0, 4]
        rng = np.random.default_rng(2)
        dim = 600
        X = rng.standard_normal((dim, dim))
        cov = (X.T @ X) / dim
        mc_median = float(np.median(np.linalg.eigvalsh(cov)))
        assert factor == pytest.approx(mc_median, rel=0.1)

    def test_small_gamma_median_near_one(self):
        # As gamma -> 0 the MP law concentrates at x=1 (no aspect-ratio spread).
        assert _mp_median_factor(1e-6) == pytest.approx(1.0, abs=0.01)

    def test_monotone_in_gamma(self):
        vals = [_mp_median_factor(g) for g in (0.01, 0.25, 0.5, 0.75, 1.0)]
        assert all(a >= b for a, b in zip(vals, vals[1:])), (
            "MP median should not increase as the bulk spreads with gamma")


class TestEstimateMPVariance:
    def test_recovers_known_scale_on_synthetic_mp_bulk(self):
        # Generate an actual random Gaussian data matrix, its sample-covariance
        # eigenvalues really do follow MP at the given aspect ratio, so the
        # median-based estimator should recover the true per-feature variance
        # to within the documented ~1.00-1.11x validity band.
        rng = np.random.default_rng(0)
        n_samples, n_features, true_var = 4000, 800, 2.25
        X = rng.standard_normal((n_samples, n_features)) * np.sqrt(true_var)
        cov = np.cov(X, rowvar=False)
        eigs = np.linalg.eigvalsh(cov)
        est = estimate_mp_variance(eigs, n_samples, n_features)
        # Documented validity band is ~1.00-1.11x on synthetics; allow a little
        # extra slack in both directions for this specific random draw.
        assert 0.90 * true_var <= est <= 1.20 * true_var

    def test_empty_or_allzero_eigs_returns_one(self):
        assert estimate_mp_variance(np.zeros(10), 100, 10) == 1.0
        assert estimate_mp_variance(np.array([]), 100, 10) == 1.0

    def test_ignores_negative_eigenvalues(self):
        # Tiny numerical negatives should not crash or skew the estimate.
        eigs = np.array([-1e-10, -1e-12, 1.0, 1.0, 1.0])
        est = estimate_mp_variance(eigs, n_samples=1000, n_features=5)
        assert np.isfinite(est) and est > 0


# ---------------------------------------------------------------------------
# _participation_ratio
# ---------------------------------------------------------------------------

class TestParticipationRatio:
    def test_single_dominant_direction_is_one(self):
        assert _participation_ratio(np.array([10.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0)

    def test_uniform_spread_equals_dimension_count(self):
        eigs = np.ones(16)
        assert _participation_ratio(eigs) == pytest.approx(16.0)

    def test_empty_input_is_zero(self):
        assert _participation_ratio(np.array([])) == 0.0

    def test_all_zero_input_is_zero(self):
        assert _participation_ratio(np.zeros(5)) == 0.0

    def test_bounded_by_count_of_positive_eigenvalues(self):
        rng = np.random.default_rng(1)
        eigs = np.abs(rng.standard_normal(50))
        pr = _participation_ratio(eigs)
        assert 1.0 <= pr <= 50.0


# ---------------------------------------------------------------------------
# _layer_census / layer_manifold_census — the full per-layer pipeline
# ---------------------------------------------------------------------------

def _low_rank_plus_noise(rng, n_samples, hidden_dim, rank, signal_scale, noise_scale):
    """Synthetic activations with a known planted rank and a known noise floor
    — the ground truth this instrument is meant to recover."""
    basis = np.linalg.qr(rng.standard_normal((hidden_dim, rank)))[0]
    coeffs = rng.standard_normal((n_samples, rank)) * signal_scale
    signal = coeffs @ basis.T
    noise = rng.standard_normal((n_samples, hidden_dim)) * noise_scale
    return signal + noise


class TestLayerCensusRecoversPlantedRank:
    def test_significant_dims_matches_planted_rank_with_explicit_noise_floor(self):
        rng = np.random.default_rng(42)
        n_samples, hidden_dim, rank = 2000, 128, 5
        noise_scale = 1.0
        acts = _low_rank_plus_noise(rng, n_samples, hidden_dim, rank,
                                    signal_scale=8.0, noise_scale=noise_scale)
        result = _layer_census(acts, layer=0, concept_directions={},
                               noise_variance=noise_scale ** 2)
        # Large-margin planted signal (8x noise) over a clean noise floor should
        # recover the exact rank, with a little room for the SVD splitting a
        # nearly-degenerate signal eigenvalue at this sample size.
        assert result.significant_dims == rank

    def test_self_calibrated_floor_underestimates_pure_noise_by_construction(self):
        # With noise_variance=None, sigma^2 self-calibrates to the OBSERVED
        # mean eigenvalue of this very matrix, so on pure noise the edge sits
        # near the bulk itself and significant_dims should read ~0 — the
        # documented motivation (PROVENANCE.md) for the explicit-variance path.
        rng = np.random.default_rng(3)
        acts = rng.standard_normal((3000, 64))
        result = _layer_census(acts, layer=0, concept_directions={}, noise_variance=None)
        assert result.significant_dims <= 2

    def test_center_flag_changes_result_on_data_with_a_mean_shift(self):
        rng = np.random.default_rng(5)
        acts = rng.standard_normal((1000, 32)) + 5.0  # constant offset every column
        centered = _layer_census(acts, layer=0, concept_directions={},
                                 noise_variance=1.0, center=True)
        uncentered = _layer_census(acts, layer=0, concept_directions={},
                                   noise_variance=1.0, center=False)
        # Uncentered second-moment picks up the shared mean direction as a
        # spike; centered removes it. They must disagree here or `center` is
        # a no-op (regression guard for the PROVENANCE.md-documented adaptation).
        assert uncentered.significant_dims >= centered.significant_dims + 1
        assert uncentered.top_eigenvalues[0] > centered.top_eigenvalues[0]

    def test_uniformly_rescaled_weights_are_scale_sensitive_under_fixed_floor(self):
        # PROVENANCE.md's own motivating check: a uniformly x2 weight matrix
        # must NOT still read as low-rank once the fixed analytic null is used
        # (that was the whole point of adding noise_variance).
        rng = np.random.default_rng(7)
        w = rng.standard_normal((500, 128)) * np.sqrt(2.0 / 128)  # He-Gaussian null
        w_scaled = w * 2.0
        base = _layer_census(w, layer=0, concept_directions={},
                             noise_variance=2.0 / 128, center=False)
        scaled = _layer_census(w_scaled, layer=0, concept_directions={},
                               noise_variance=2.0 / 128, center=False)
        assert scaled.significant_dims > base.significant_dims

    def test_effective_dim_matches_participation_ratio_of_the_spectrum(self):
        rng = np.random.default_rng(9)
        acts = rng.standard_normal((500, 40))
        result = _layer_census(acts, layer=0, concept_directions={})
        assert result.effective_dim == pytest.approx(
            _participation_ratio(np.array(result.top_eigenvalues + [0.0] * 0)),
            rel=0.2,  # top_eigenvalues is truncated to n_top=50 >= 40 here, so exact-ish
        )

    def test_svd_and_eigh_paths_agree_on_identical_data(self):
        # n_samples <= hidden_dim takes the SVD branch (economy SVD of the
        # centered data); n_samples > hidden_dim takes the eigh branch (eigh
        # of np.cov). They must compute the SAME spectrum for the same data —
        # replicate the eigh-branch math by hand on data that (by construction)
        # goes through the SVD branch inside _layer_census, and compare.
        rng = np.random.default_rng(11)
        hidden_dim, n_samples = 20, 15  # n <= d -> SVD branch
        acts = rng.standard_normal((n_samples, hidden_dim))
        result = _layer_census(acts, layer=0, concept_directions={},
                               n_top_eigenvalues=n_samples)
        acts_c = acts - acts.mean(axis=0)
        cov = np.cov(acts_c, rowvar=False)
        eigh_vals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eigh_vals = np.maximum(eigh_vals, 0)[:n_samples]
        np.testing.assert_allclose(result.top_eigenvalues, eigh_vals, atol=1e-8)


class TestConceptCoverage:
    def test_activations_confined_to_concept_subspace_are_fully_covered(self):
        rng = np.random.default_rng(13)
        hidden_dim = 10
        concept_a = np.zeros(hidden_dim); concept_a[0] = 1.0
        concept_b = np.zeros(hidden_dim); concept_b[1] = 1.0
        coeffs = rng.standard_normal((500, 2))
        acts = np.zeros((500, hidden_dim))
        acts[:, 0] = coeffs[:, 0]
        acts[:, 1] = coeffs[:, 1]
        result = _layer_census(acts, layer=0,
                               concept_directions={"a": concept_a, "b": concept_b})
        assert result.concept_coverage == pytest.approx(1.0, abs=1e-6)
        assert result.concept_dims == 2
        assert result.residual_variance == pytest.approx(0.0, abs=1e-8)

    def test_activations_orthogonal_to_concept_subspace_have_zero_coverage(self):
        rng = np.random.default_rng(15)
        hidden_dim = 10
        concept_a = np.zeros(hidden_dim); concept_a[0] = 1.0
        acts = rng.standard_normal((500, hidden_dim))
        acts[:, 0] = 0.0  # exactly zero along the concept direction
        result = _layer_census(acts, layer=0, concept_directions={"a": concept_a})
        assert result.concept_coverage == pytest.approx(0.0, abs=1e-6)

    def test_no_concepts_makes_residual_equal_full_spectrum(self):
        rng = np.random.default_rng(17)
        acts = rng.standard_normal((200, 12))
        result = _layer_census(acts, layer=0, concept_directions={})
        assert result.concept_coverage == 0.0
        assert result.concept_dims == 0
        assert result.residual_dim == result.effective_dim
        assert result.residual_significant == result.significant_dims
        assert result.residual_variance == result.total_variance


class TestLayerManifoldCensus:
    def test_raises_on_empty_input(self):
        with pytest.raises(ValueError):
            layer_manifold_census([])

    def test_wraps_one_result_per_layer_in_order(self):
        rng = np.random.default_rng(19)
        acts = [rng.standard_normal((100, 16)) for _ in range(4)]
        census = layer_manifold_census(acts)
        assert census.n_layers == 4
        assert [lr.layer for lr in census.layers] == [0, 1, 2, 3]

    def test_store_directions_toggle(self):
        rng = np.random.default_rng(21)
        acts = [rng.standard_normal((100, 16))]
        with_dirs = layer_manifold_census(acts, store_directions=True)
        without_dirs = layer_manifold_census(acts, store_directions=False)
        assert with_dirs.layers[0].top_directions is not None
        assert without_dirs.layers[0].top_directions is None

    def test_summary_arrays_shapes(self):
        rng = np.random.default_rng(23)
        acts = [rng.standard_normal((50, 8)) for _ in range(3)]
        census = layer_manifold_census(acts)
        arrays = census.summary_arrays()
        for key in ("effective_dim", "significant_dims", "concept_coverage",
                   "residual_dim", "residual_significant", "depth_pct"):
            assert arrays[key].shape == (3,)
        np.testing.assert_allclose(arrays["depth_pct"], [0.0, 100 / 3, 200 / 3])

"""Unit tests for the pure-numpy subspace-comparison helpers underneath F5
("sharing is coordinate-bound; the depth code is one code up to rotation"):
census/directional_consistency.py, census/eigenspace_overlap.py, and
census/procrustes_overlap.py. Each of these scripts' `main()` reads from the
corpus/ directory (HF-backed, not present in this checkout), but the actual
geometry — overlap of subspaces, Procrustes recovery — lives in small,
corpus-free helper functions that are tested directly here.
"""
from __future__ import annotations

import numpy as np
import pytest

from directional_consistency import overlap, top_left_svs
from eigenspace_overlap import layer_eigvecs, pairwise_overlap
from procrustes_overlap import forward_all_layers, pair_layer_metrics, top_eigvecs


# ---------------------------------------------------------------------------
# directional_consistency.overlap / top_left_svs
# ---------------------------------------------------------------------------

class TestOverlap:
    def test_identical_subspace_overlap_is_one(self):
        rng = np.random.default_rng(0)
        U = np.linalg.qr(rng.standard_normal((20, 5)))[0]
        assert overlap(U, U) == pytest.approx(1.0, abs=1e-10)

    def test_orthogonal_complement_overlap_is_zero(self):
        rng = np.random.default_rng(1)
        Q = np.linalg.qr(rng.standard_normal((20, 20)))[0]
        A, B = Q[:, :5], Q[:, 5:10]
        assert overlap(A, B) == pytest.approx(0.0, abs=1e-10)

    def test_rotation_invariant(self):
        rng = np.random.default_rng(2)
        A = np.linalg.qr(rng.standard_normal((15, 4)))[0]
        B = np.linalg.qr(rng.standard_normal((15, 4)))[0]
        R = np.linalg.qr(rng.standard_normal((4, 4)))[0]  # orthogonal
        assert overlap(A, B) == pytest.approx(overlap(A, B @ R), abs=1e-10)

    def test_random_subspaces_land_near_chance(self):
        # E[overlap] for two independent random k-dim subspaces of R^d is k/d.
        rng = np.random.default_rng(3)
        d, k, trials = 64, 4, 300
        vals = []
        for _ in range(trials):
            A = np.linalg.qr(rng.standard_normal((d, k)))[0]
            B = np.linalg.qr(rng.standard_normal((d, k)))[0]
            vals.append(overlap(A, B))
        assert np.mean(vals) == pytest.approx(k / d, abs=0.03)


class TestTopLeftSVs:
    def test_output_is_orthonormal(self):
        rng = np.random.default_rng(4)
        M = rng.standard_normal((30, 10))
        U = top_left_svs(M, k=4)
        assert U.shape == (30, 4)
        np.testing.assert_allclose(U.T @ U, np.eye(4), atol=1e-8)

    def test_recovers_the_planted_column_space(self):
        rng = np.random.default_rng(5)
        basis = np.linalg.qr(rng.standard_normal((25, 3)))[0]
        M = basis @ (np.diag([10.0, 5.0, 2.0]) @ rng.standard_normal((3, 8)))
        U = top_left_svs(M, k=3)
        assert overlap(U, basis) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# eigenspace_overlap.layer_eigvecs / pairwise_overlap
# ---------------------------------------------------------------------------

class TestLayerEigvecs:
    def test_output_shape_per_layer(self):
        rng = np.random.default_rng(6)
        width, k = 16, 3
        weights = [rng.standard_normal((width, width)) * 0.1 for _ in range(4)]
        x = rng.standard_normal((500, width))
        out = layer_eigvecs(weights, x, k)
        assert len(out) == 4
        for U in out:
            assert U.shape == (width, k)
            np.testing.assert_allclose(U.T @ U, np.eye(k), atol=1e-6)

    def test_pairwise_overlap_of_identical_nets_is_one(self):
        rng = np.random.default_rng(7)
        width, k = 16, 3
        weights = [rng.standard_normal((width, width)) * 0.2 for _ in range(3)]
        x = rng.standard_normal((500, width))
        units = [layer_eigvecs(weights, x, k) for _ in range(2)]  # same net twice
        per_layer = pairwise_overlap(units, k)
        assert len(per_layer) == 3
        for v in per_layer:
            assert v == pytest.approx(1.0, abs=1e-6)

    def test_pairwise_overlap_of_unrelated_nets_is_near_chance(self):
        rng = np.random.default_rng(8)
        width, k = 64, 2
        x = rng.standard_normal((4000, width))
        nets = [[rng.standard_normal((width, width)) * (1.0 / np.sqrt(width))
                for _ in range(1)] for _ in range(6)]
        units = [layer_eigvecs(w, x, k) for w in nets]
        per_layer = pairwise_overlap(units, k)
        assert per_layer[0] == pytest.approx(k / width, abs=0.05)


# ---------------------------------------------------------------------------
# procrustes_overlap: fitted rotation recovers a KNOWN planted rotation
# ---------------------------------------------------------------------------

class TestProcrustesRecovery:
    def test_recovers_a_known_rotation_between_twin_activations(self):
        # Construct two "nets" whose activations are literally the same
        # underlying signal viewed through a fixed random rotation, plus
        # independent noise — the textbook case orthogonal Procrustes exists
        # to solve. The RECOVERED overlap on held-out data should be near 1,
        # far above the RAW overlap (which sees them as unrelated bases).
        rng = np.random.default_rng(9)
        n, width, k = 4000, 20, 4
        signal = rng.standard_normal((n, width))
        R_true = np.linalg.qr(rng.standard_normal((width, width)))[0]
        a_i = signal + 0.01 * rng.standard_normal((n, width))
        a_j = signal @ R_true + 0.01 * rng.standard_normal((n, width))

        raw, rec = pair_layer_metrics(a_i, a_j, k)
        assert rec > 0.9
        assert rec > raw + 0.3

    def test_unrelated_activations_recover_near_chance(self):
        rng = np.random.default_rng(10)
        n, width, k = 4000, 32, 3
        a_i = rng.standard_normal((n, width))
        a_j = rng.standard_normal((n, width))
        raw, rec = pair_layer_metrics(a_i, a_j, k)
        # independent noise: even the FIT-half rotation can't manufacture
        # shared structure that isn't there on held-out data
        assert rec < 0.5

    def test_top_eigvecs_orthonormal(self):
        rng = np.random.default_rng(11)
        a = rng.standard_normal((300, 12))
        U = top_eigvecs(a, k=5)
        assert U.shape == (12, 5)
        np.testing.assert_allclose(U.T @ U, np.eye(5), atol=1e-8)


class TestForwardAllLayers:
    def test_relu_and_shapes(self):
        rng = np.random.default_rng(12)
        width = 8
        weights = [rng.standard_normal((width, width)) for _ in range(3)]
        x = rng.standard_normal((10, width))
        acts = forward_all_layers(weights, x)
        assert len(acts) == 3
        for a in acts:
            assert a.shape == (10, width)
            assert np.all(a >= 0.0)  # post-ReLU

"""Unit tests for train/gmm_task.py — the GMM task family whose entire point
is that its aggregate distribution matches the null baseline's N(0, I) premise
EXACTLY (README: "so the null's premise holds by construction"). These tests
check that construction claim directly, plus the Bayes-accuracy and
class-count-vs-dim guard.
"""
from __future__ import annotations

import numpy as np
import pytest

from gmm_task import make_gmm_task


class TestAggregateDistributionMatchesNullPremise:
    def test_aggregate_mean_is_zero(self):
        task = make_gmm_task(dim=32, n_classes=6, separation=3.0, seed=0)
        rng = np.random.default_rng(1)
        x, _ = task.sample(300_000, rng)
        assert np.abs(x.mean(axis=0)).max() < 0.02

    def test_aggregate_covariance_is_identity(self):
        task = make_gmm_task(dim=32, n_classes=6, separation=3.0, seed=0)
        rng = np.random.default_rng(1)
        x, _ = task.sample(300_000, rng)
        cov = np.cov(x.astype(np.float64), rowvar=False)
        np.testing.assert_allclose(np.diag(cov), np.ones(32), atol=0.05)
        off_diag = cov - np.diag(np.diag(cov))
        assert np.abs(off_diag).max() < 0.05

    def test_class_means_are_exactly_mean_centered(self):
        task = make_gmm_task(dim=16, n_classes=5, separation=2.0, seed=3)
        np.testing.assert_allclose(task.means.mean(axis=0), np.zeros(16), atol=1e-10)


class TestSampling:
    def test_sample_shapes_and_dtypes(self):
        task = make_gmm_task(dim=10, n_classes=4, separation=2.0, seed=0)
        x, y = task.sample(500, np.random.default_rng(0))
        assert x.shape == (500, 10)
        assert y.shape == (500,)
        assert x.dtype == np.float32
        assert y.dtype == np.int64
        assert set(np.unique(y)).issubset(set(range(4)))

    def test_labels_are_uniformly_distributed(self):
        task = make_gmm_task(dim=10, n_classes=4, separation=2.0, seed=0)
        _, y = task.sample(200_000, np.random.default_rng(0))
        counts = np.bincount(y, minlength=4)
        assert np.all(np.abs(counts / 200_000 - 0.25) < 0.01)

    def test_same_rng_seed_is_deterministic(self):
        task = make_gmm_task(dim=8, n_classes=3, separation=2.0, seed=0)
        x1, y1 = task.sample(100, np.random.default_rng(42))
        x2, y2 = task.sample(100, np.random.default_rng(42))
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)


class TestSeparationDial:
    def test_larger_separation_increases_effective_separation(self):
        near = make_gmm_task(dim=16, n_classes=5, separation=1.0, seed=0)
        far = make_gmm_task(dim=16, n_classes=5, separation=6.0, seed=0)
        assert far.effective_separation > near.effective_separation

    def test_larger_separation_increases_bayes_accuracy(self):
        near = make_gmm_task(dim=16, n_classes=5, separation=1.0, seed=0)
        far = make_gmm_task(dim=16, n_classes=5, separation=6.0, seed=0)
        assert far.bayes_accuracy > near.bayes_accuracy

    def test_bayes_accuracy_is_a_valid_probability_above_chance(self):
        task = make_gmm_task(dim=16, n_classes=5, separation=3.0, seed=0)
        chance = 1.0 / task.n_classes
        assert chance <= task.bayes_accuracy <= 1.0


class TestValidation:
    def test_more_classes_than_dims_raises(self):
        with pytest.raises(ValueError):
            make_gmm_task(dim=4, n_classes=10, separation=3.0, seed=0)

    def test_classes_equal_to_dims_is_allowed(self):
        task = make_gmm_task(dim=5, n_classes=5, separation=3.0, seed=0)
        assert task.means.shape == (5, 5)


class TestDescribe:
    def test_describe_contains_the_documented_keys(self):
        task = make_gmm_task(dim=16, n_classes=5, separation=3.0, seed=7)
        d = task.describe()
        assert d["family"] == "gmm"
        assert d["n_classes"] == 5
        assert d["task_seed"] == 7
        assert d["aggregate_distribution"] == "exact mean-0 cov-I (whitened)"

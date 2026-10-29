"""mzc_toolkit on small synthetic nets (n = 256, depth 6): He random vs shrunken-remnant-plus-shared-update."""
import numpy as np
import pytest
from mzc_toolkit import (alpha_hat, triage, decompose, rebuild_with_slice, fidelity_curve, he_reference, load_layers)

N, D, R = 256, 6, 8


def he_net(seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((N, N)) * np.sqrt(2.0 / N) for _ in range(D)]


def trained_like(seed, alpha=0.3):
    """alpha * W0 + sqrt(2) * Q R_l Q^T with one subspace Q shared by all layers (the structure seen in trained nets)."""
    rng = np.random.default_rng(seed); W0 = he_net(seed); Q = np.linalg.qr(rng.standard_normal((N, R)))[0]
    Ws = [alpha * w + np.sqrt(2) * Q @ np.linalg.qr(rng.standard_normal((R, R)))[0] @ Q.T for w in W0]
    return Ws, W0


def test_he_reference_is_cached_and_sane():
    med, mx = he_reference(N); assert he_reference(N) == (med, mx); assert 1.0 < med < 1.3 and 2.0 < mx < 3.2


def test_alpha_hat_on_random_net_is_one():
    for W in he_net(1): assert abs(alpha_hat(W) - 1.0) < 0.03


def test_triage_random_vs_trained_like():
    r = triage(he_net(2), k=R); assert not r["structure_detected"] and r["chain_over_chance"] < 5
    Ws, _ = trained_like(3); t = triage(Ws, k=R)
    assert t["structure_detected"], t["chain_over_chance"]
    assert abs(t["alpha_hat_trusted_mean"] - 0.3) < 0.03
    assert min(t["outliers"]) >= R // 2


def test_decompose_recovers_alpha_and_low_rank_update():
    Ws, W0 = trained_like(4); d = decompose(Ws, W0)
    for o in d["layers"]:
        assert abs(o["alpha"] - 0.3) < 0.02 and o["delta_pr"] < 3 * R
    assert d["rho"] > 0.5


def test_slice_rebuild_is_identity_at_full_rank_and_lossy_at_zero():
    Ws, W0 = trained_like(5)
    for W, V in zip(Ws, rebuild_with_slice(Ws, W0, N)): assert np.allclose(W, V, atol=1e-8)
    X = np.random.default_rng(6).standard_normal((512, N))
    eps = fidelity_curve(Ws, lambda k: rebuild_with_slice(Ws, W0, k), [0, R, N], X)
    assert eps[N] < 1e-12 and eps[0] > eps[N] and eps[R] <= eps[0] + 1e-9


def test_load_layers_requires_keys(tmp_path):
    p = tmp_path / "net.npz"; np.savez(p, w0=np.eye(3), w1=np.eye(3), init_w0=np.eye(3))
    assert len(load_layers(p)) == 2 and len(load_layers(p, "init_w")) == 1
    with pytest.raises(KeyError): load_layers(p, "nope")

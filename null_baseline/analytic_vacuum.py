"""AMC Plan B — the analytic vacuum (weights -> activation covariance spectrum).

*Written: 2026-06-15 UTC*

PURE NUMPY, CPU-ONLY. No torch math, no GPU, no forward passes for the predictor.
(torch is imported only by the Stage-2 script to READ weights on CPU.)

`predict_activation_cov_spectrum(weights)` takes a list of square ReLU-MLP weight
matrices (forward convention z_{l+1} = relu(x_l) @ W_l, matching
starterkit/local_engine.build_mlp / forward_monte_carlo) and analytically propagates
the per-neuron pre-activation mean (mu) and covariance (Sig) layer by layer with ZERO
forward passes, returning the per-layer activation-covariance eigenspectrum.

The covariance recursion is the Gaussian mean-field / arccosine-kernel result for ReLU
(Cho&Saul 2009; Poole/Schoenholz 2016), implemented here via the exact Mehler/Hermite
series ported from starterkit/estimator_v66.py (mehler_cov + relu_quantities). The
post-ReLU SECOND-MOMENT covariance is propagated; mean is tracked so we can report the
CENTERED covariance spectrum (which is what the AMC census computes: it centers the
activations before the SVD).

Recursion (matching local_engine forward `x = relu(x @ W)`):
    Sig_1 = W_0^T W_0                 (pre-activation 2nd moment for x ~ N(0, I))
    mu_1  = 0
    for each layer l:
        a   = post-ReLU activation a_l = relu(z_l),  z_l ~ N(mu_l, Sig_l)
        Sig_a = mehler 2nd-moment covariance of a_l   (uncentered)
        mu_a  = E[a_l]   (per neuron, closed form)
        z_{l+1} = a_l @ W_l   ->  mu_{l+1} = W_l^T mu_a ;  Sig_{l+1} = W_l^T Sig_a W_l
The reported spectrum at layer l is eig( Sig_a - mu_a mu_a^T )  (CENTERED, matches census)
or eig(Sig_a) (uncentered) — both returned.

NOTE on scope vs estimator_v66: that estimator carries 3rd/4th cumulants (Edgeworth,
Stein skew) to nail the MEAN to the ARC FLOP budget. For the COVARIANCE SPECTRUM the
leading Gaussian-mean-field term is the dominant physics and is what we validate here;
higher cumulant corrections to the covariance are a known refinement and are reported
as part of the residual if Stage-1 error is non-negligible.
"""
from __future__ import annotations

import math

import numpy as np

SQRT2PI = math.sqrt(2.0 * math.pi)

# Mehler series order. ReLU's Hermite coefficients decay; N_HERM=6 matches estimator_v66.
N_HERM = 6
_FACT = [float(math.factorial(n)) for n in range(N_HERM + 3)]
_FACTN = np.array([_FACT[n] for n in range(1, N_HERM + 1)])  # 1!..6!


def _norm_cdf(x):
    # Phi(x) = 0.5*(1+erf(x/sqrt2))
    from math import erf
    vec = np.vectorize(erf)
    return 0.5 * (1.0 + vec(x / math.sqrt(2.0)))


def _norm_pdf(x):
    return np.exp(-0.5 * x * x) / SQRT2PI


def _relu_hermite_coefs(mu, var):
    """Probabilists'-Hermite coefficients a_n = (1/n!) sig^n E[ReLU^{(n)}(Z)],
    Z ~ N(mu, var), per neuron. Returns array (N_HERM+1, W).

    a_n = sig^n / n! * E[d^n ReLU(Z)]  with the closed-form ReLU Wick coefficients
    (port of estimator_v66.relu_quantities.w1):
        E[d^0 ReLU] = sig*phi + mu*Phi
        E[d^1 ReLU] = Phi
        E[d^k ReLU] = (-1)^{k-2} sig^{-(k-1)} He_{k-2}(alpha) phi   (k>=2)
    a_0 = E[ReLU] = the per-neuron MEAN.
    """
    sig = np.sqrt(np.maximum(var, 1e-20))
    alpha = mu / sig
    phi = _norm_pdf(alpha)
    Phi = _norm_cdf(alpha)

    # probabilists' Hermite He_0..He_{N_HERM-2} at alpha
    He = [np.ones_like(alpha), alpha]
    for n in range(1, N_HERM):
        He.append(alpha * He[n] - n * He[n - 1])

    sigpow = [np.ones_like(sig)]
    for _ in range(N_HERM):
        sigpow.append(sigpow[-1] * sig)

    def w1(k):  # E[d^k ReLU]
        if k == 0:
            return sig * phi + mu * Phi
        if k == 1:
            return Phi
        return ((-1.0) ** (k - 2)) / sigpow[k - 1] * He[k - 2] * phi

    a = np.stack([sigpow[n] / _FACT[n] * w1(n) for n in range(N_HERM + 1)], axis=0)
    return a  # (N_HERM+1, W); a[0] = E[ReLU] per neuron


def _mehler_cov(a, var, Sig):
    """Post-ReLU 2nd-moment covariance via Mehler series (port of
    estimator_v66.mehler_cov). a: (N_HERM+1, W) Hermite coefs; var: (W,) marginal
    pre-activation variances; Sig: (W,W) pre-activation covariance.

    C_ij = sum_{n>=1} n! a_n^i a_n^j rho_ij^n   (off-diagonal),  C_ii = E[ReLU(Z_i)^2].
    rho_ij = Sig_ij / (sig_i sig_j).  The diagonal is set to the exact 2nd moment
    E[ReLU^2] rather than the truncated series (matches estimator_v66 fill_diagonal,
    but with the true 2nd moment so the UNcentered cov is exact on the diagonal).
    """
    sig = np.sqrt(np.maximum(var, 1e-20))
    rho = np.clip(Sig / np.outer(sig, sig), -0.99999, 0.99999)
    # rho^1 .. rho^N
    RP = [rho]
    for _ in range(N_HERM - 1):
        RP.append(RP[-1] * rho)
    RP = np.stack(RP, axis=0)  # (N, W, W)
    a1 = a[1:]  # (N, W)
    # The Mehler series with n>=1 is exactly the CENTERED cross-covariance
    # Cov[ReLU(Z_i),ReLU(Z_j)] = sum_{n>=1} n! a_n^i a_n^j rho_ij^n
    # (the n=0 term a_0^i a_0^j = E[ReLU_i]E[ReLU_j] is the mean outer product, omitted).
    C = np.einsum("ni,nj,nij->ij", a1 * _FACTN[:, None], a1, RP)
    return C, sig, rho


def _relu_second_moment(mu, var):
    """E[ReLU(Z)^2], Z~N(mu,var), per neuron (closed form)."""
    sig = np.sqrt(np.maximum(var, 1e-20))
    alpha = mu / sig
    phi = _norm_pdf(alpha)
    Phi = _norm_cdf(alpha)
    return (mu * mu + var) * Phi + mu * sig * phi


def _relu_mean(mu, var):
    sig = np.sqrt(np.maximum(var, 1e-20))
    alpha = mu / sig
    return sig * _norm_pdf(alpha) + mu * _norm_cdf(alpha)


def predict_activation_cov_spectrum(weights, n_top=None, return_full=True):
    """Analytic per-layer post-activation covariance eigenspectrum from weights alone.

    weights: list of (W,W) numpy arrays. Forward convention z_{l+1} = relu(x) @ W_l,
             input x ~ N(0, I_W). (Same as starterkit local_engine.)
    n_top:   if set, return only the top-n eigenvalues per layer.
    return_full: also return participation ratio / effective dim per layer.

    Returns dict:
      'eig_centered'   list per layer of eigenvalues of (Sig_a - mu_a mu_a^T) (desc)
      'eig_uncentered' list per layer of eigenvalues of Sig_a (desc)
      'eff_dim_centered'   participation ratio of centered spectrum per layer
      'eff_dim_uncentered' participation ratio of uncentered spectrum per layer
      'mu_norm'        ||mu_a|| per layer (mean magnitude diagnostic)
    Layer index l corresponds to the post-activation of MLP layer l (a_l = relu(z_l)),
    matching forward_monte_carlo which records relu output of each weight matrix.
    """
    W0 = np.asarray(weights[0], dtype=np.float64)
    width = W0.shape[1]
    # pre-activation of layer 0: z_0 = x @ W0, x~N(0,I) -> Sig = W0^T W0, mu = 0
    Sig = W0.T @ W0
    mu = np.zeros(width, dtype=np.float64)

    out = dict(eig_centered=[], eig_uncentered=[],
               eff_dim_centered=[], eff_dim_uncentered=[], mu_norm=[])

    n_layers = len(weights)
    for l in range(n_layers):
        var = np.maximum(np.diag(Sig), 1e-20)
        a = _relu_hermite_coefs(mu, var)
        mu_a = _relu_mean(mu, var)                       # E[a_l]
        C, sig, rho = _mehler_cov(a, var, Sig)           # CENTERED cov (n>=1 series)
        # exact CENTERED diagonal: Var[ReLU] = E[ReLU^2] - E[ReLU]^2 (replaces the
        # truncated series value on the diagonal so the centered cov is exact there).
        var_a = _relu_second_moment(mu, var) - mu_a * mu_a
        np.fill_diagonal(C, var_a)
        Sig_a_c = 0.5 * (C + C.T)                        # CENTERED post-ReLU cov
        Sig_a = Sig_a_c + np.outer(mu_a, mu_a)           # UNcentered (for propagation)

        ev_c = np.linalg.eigvalsh(Sig_a_c)[::-1]
        ev_c = np.clip(ev_c, 0.0, None)                  # tiny negatives -> 0
        ev_u = np.linalg.eigvalsh(0.5 * (Sig_a + Sig_a.T))[::-1]
        ev_u = np.clip(ev_u, 0.0, None)

        out['eig_centered'].append(ev_c if n_top is None else ev_c[:n_top])
        out['eig_uncentered'].append(ev_u if n_top is None else ev_u[:n_top])
        out['eff_dim_centered'].append(_participation_ratio(ev_c))
        out['eff_dim_uncentered'].append(_participation_ratio(ev_u))
        out['mu_norm'].append(float(np.linalg.norm(mu_a)))

        if l == n_layers - 1:
            break
        # propagate to next pre-activation z_{l+1} = a_l @ weights[l+1]:
        #   mean  mu_{l+1} = weights[l+1]^T E[a]
        #   CENTERED cov Sig_{l+1} = weights[l+1]^T Cov[a] weights[l+1]
        # `Sig` is the CENTERED pre-activation covariance (mean tracked separately in
        # `mu`); var = diag(Sig) is the centered marginal variance and rho uses it,
        # matching estimator_v66.v6_layer_means.
        Wn = np.asarray(weights[l + 1], dtype=np.float64)
        mu = Wn.T @ mu_a
        Sig = Wn.T @ Sig_a_c @ Wn
        Sig = 0.5 * (Sig + Sig.T)
    return out


def _participation_ratio(eigs):
    """(sum lambda)^2 / sum(lambda^2). Matches caz/manifold_detector._participation_ratio."""
    eigs = np.asarray(eigs, dtype=np.float64)
    eigs = eigs[eigs > 0]
    if eigs.size == 0:
        return 0.0
    s = eigs.sum()
    s2 = (eigs * eigs).sum()
    return float(s * s / s2) if s2 > 0 else 0.0

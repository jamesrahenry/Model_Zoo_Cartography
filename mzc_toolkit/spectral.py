"""Stage 0: weights-only instruments."""
import numpy as np

_REF = {}


def he_reference(n, seed=5):
    """(median, max) singular value of a He-Gaussian n x n matrix (entry std sqrt(2/n)); cached."""
    if n not in _REF:
        s = np.linalg.svd(np.random.default_rng(seed).standard_normal((n, n)) * np.sqrt(2.0 / n), compute_uv=False)
        _REF[n] = (float(np.median(s)), float(s.max()))
    return _REF[n]


def alpha_hat(W):
    """Surviving random-bulk amplitude: median singular value of W over that of a He matrix. Measured median error 0.004 at layers >= 2
    (1024 x 16 GMM nets); unreliable at layers 0-1, which training rewrites."""
    s = np.linalg.svd(np.asarray(W, dtype=np.float64), compute_uv=False)
    return float(np.median(s) / he_reference(W.shape[0])[0])


def layer_subspaces(Ws, k=16):
    """Top-k left (input-side) and right (output-side) singular vectors of every layer, columns orthonormal."""
    U, V = [], []
    for W in Ws:
        u, _, vh = np.linalg.svd(np.asarray(W, dtype=np.float64))
        U.append(u[:, :k]); V.append(vh[:k].T)
    return U, V


def chain_score(U, V):
    """Mean squared cosine between one layer's output subspace and the next layer's input subspace (chance = k/n)."""
    k = U[0].shape[1]
    return float(np.mean([np.sum((U[l + 1].T @ V[l]) ** 2) / k for l in range(len(U) - 1)]))


def outlier_count(W, ahat=None, factor=1.05):
    """Singular values of W above factor * alpha_hat * (He max singular value): directions poking out of the random bulk."""
    n = W.shape[0]; ahat = alpha_hat(W) if ahat is None else ahat
    s = np.linalg.svd(np.asarray(W, dtype=np.float64), compute_uv=False)
    return int((s > factor * ahat * he_reference(n)[1]).sum())


def triage(Ws, k=16, detect_factor=10.0):
    """One-call Stage 0 (one SVD per layer). Returns alpha_hat and outlier counts per layer, the chain score, and a plain verdict.
    structure_detected means chain_score >= detect_factor * (k/n); validated at n = 1024 (92% of trained nets at factor 10, inits <= 0.016)."""
    n = Ws[0].shape[0]; med0, max0 = he_reference(n)
    ah, outl, U, V = [], [], [], []
    for W in Ws:
        u, s, vh = np.linalg.svd(np.asarray(W, dtype=np.float64))
        a = float(np.median(s) / med0); ah.append(a); outl.append(int((s > 1.05 * a * max0).sum())); U.append(u[:, :k]); V.append(vh[:k].T)
    S = chain_score(U, V); chance = k / n
    return dict(n=n, depth=len(Ws), k=k, alpha_hat=ah, alpha_hat_trusted_mean=float(np.mean(ah[2:])) if len(ah) > 2 else None,
                outliers=outl, chain_score=S, chance=chance, chain_over_chance=S / chance, structure_detected=bool(S >= detect_factor * chance),
                note="alpha_hat is unreliable at layers 0-1; the chain score says that and roughly where structure exists, not what it is")


def chained_subspaces(Ws, k=16):
    """Weights-only cross-layer subspaces. V_l: top-k right singular vectors of the prefix product W_0...W_l (output-side directions that receive the most
    accumulated gain); U_l: top-k left singular vectors of the suffix product W_l...W_{L-1} (input-side directions the remaining layers amplify most);
    products are rescaled after every multiplication. In 64 low-/no-decay nets that depend on their random draw, rebuilding each layer from the part of W touching these
    subspaces (rest replaced by a fresh He draw at alpha_hat) kept 95% of accuracy at k = 16 in 77% of nets (per-layer singular subspaces: 5%; random subspace: median k 1024),
    but only 14% reached the fidelity of a cumulant estimator, and the subspaces do not coincide with the update's own (capture 0.47)."""
    D = len(Ws); Wd = [np.asarray(W, dtype=np.float64) for W in Ws]
    V, U = [], [None] * D; P = None
    for l in range(D):
        P = Wd[l] if P is None else P @ Wd[l]; P = P / np.linalg.norm(P)
        V.append(np.linalg.svd(P)[2][:k].T)
    S = None
    for l in reversed(range(D)):
        S = Wd[l] if S is None else Wd[l] @ S; S = S / np.linalg.norm(S)
        U[l] = np.linalg.svd(S)[0][:, :k]
    return U, V

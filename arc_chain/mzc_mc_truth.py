"""GPU Monte Carlo per-layer means (and sampling variance) of a bias-free ReLU MLP under N(0,I) input.

Written 2026-10-20.  Oracle diagnostic only.  Convention matches MZC/ARC: h = relu(h @ W) at every layer.
usage: mzc_mc_truth.py NPZ KEY_PREFIX(w|init_w) N_SAMPLES OUT.npz [--seed S]
Saves mean (16,1024) float64 and se2 (16,1024) = sample variance / N (variance of the estimated mean).
"""
import sys, numpy as np, torch
npz, pre, N, out = sys.argv[1], sys.argv[2], int(float(sys.argv[3])), sys.argv[4]
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
torch.backends.cuda.matmul.allow_tf32 = False
z = np.load(npz); dev = "cuda"
Ws = [torch.tensor(z[f"{pre}{i}"], device=dev) for i in range(16)]
g = torch.Generator(device=dev); g.manual_seed(seed)
B = 32768
s1 = torch.zeros(16, 1024, dtype=torch.float64, device=dev); s2 = torch.zeros_like(s1)
done = 0
while done < N:
    b = min(B, N - done)
    h = torch.randn(b, 1024, device=dev, generator=g)
    for l, W in enumerate(Ws):
        h = torch.relu(h @ W)
        s1[l] += h.sum(0, dtype=torch.float64); s2[l] += (h.double() ** 2).sum(0)
    done += b
mean = s1 / N; var = s2 / N - mean ** 2
np.savez(out, mean=mean.cpu().numpy(), se2=(var / N).cpu().numpy(), N=N)

#!/usr/bin/env python3
"""E9b: Poisson denoising by Haar soft-thresholding, direct vs Anscombe domain.

Threshold-based denoisers need a single global noise level; Poisson noise is
heteroscedastic (var = mean), so the VST pipeline should genuinely win here.
"""
import math
import random

random.seed(11)


def poisson(lam):
    if lam < 30:
        L = math.exp(-lam)
        k, p = 0, 1.0
        while True:
            p *= random.random()
            if p <= L:
                return k
            k += 1
    # normal approximation for large lambda
    return max(0, round(random.gauss(lam, math.sqrt(lam))))


def haar_fwd(sig, levels):
    a = list(sig)
    details = []
    for _ in range(levels):
        n = len(a) // 2
        approx = [(a[2 * i] + a[2 * i + 1]) / math.sqrt(2) for i in range(n)]
        det = [(a[2 * i] - a[2 * i + 1]) / math.sqrt(2) for i in range(n)]
        details.append(det)
        a = approx
    return a, details


def haar_inv(approx, details):
    a = list(approx)
    for det in reversed(details):
        out = []
        for x, d in zip(a, det):
            out.append((x + d) / math.sqrt(2))
            out.append((x - d) / math.sqrt(2))
        a = out
    return a


def soft(v, t):
    return math.copysign(max(abs(v) - t, 0.0), v)


def denoise(sig, sigma, levels=6):
    n = len(sig)
    approx, details = haar_fwd(sig, levels)
    t = sigma * math.sqrt(2.0 * math.log(n))
    details = [[soft(v, t) for v in det] for det in details]
    return haar_inv(approx, details)


N = 4096
lam = [2.0 + 98.0 * math.sin(2 * math.pi * i / 512.0) ** 2 for i in range(N)]
obs = [float(poisson(l)) for l in lam]

# --- direct domain: single global sigma estimated by MAD of finest details
_, det = haar_fwd(obs, 1)
fin = sorted(abs(v) for v in det[0])
sigma_hat = fin[len(fin) // 2] / 0.6745
den_direct = denoise(obs, sigma_hat)

# --- Anscombe domain: sigma = 1 by construction
ans = [2.0 * math.sqrt(v + 0.375) for v in obs]
den_ans = denoise(ans, 1.0)
inv_unb = [(v / 2.0) ** 2 - 0.125 for v in den_ans]


def mse(est):
    return sum((e - l) ** 2 for e, l in zip(est, lam)) / N


print(f"lambda range: [{min(lam):.1f}, {max(lam):.1f}]  (heteroscedastic, var = mean)")
print(f"MSE raw counts                         = {mse(obs):9.4f}")
print(f"MSE Haar soft-threshold, direct domain = {mse(den_direct):9.4f}  (global sigma_hat = {sigma_hat:.2f})")
print(f"MSE Anscombe + Haar + unbiased inverse = {mse(inv_unb):9.4f}  (sigma = 1 exactly)")

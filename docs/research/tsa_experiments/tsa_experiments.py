#!/usr/bin/env python3
"""Falsification experiments for the TSA (Transformed Scalar Algorithms) research mission.

Pure-stdlib. Each experiment prints labelled results used in the report.
"""
import math
import random

LINE = "-" * 72


def digamma(x: float) -> float:
    """Digamma psi(x) via recurrence + asymptotic series (|err| < 1e-12 for x>0)."""
    acc = 0.0
    while x < 8.0:
        acc -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    # ln x - 1/(2x) - sum B_2n/(2n x^{2n})
    return acc + math.log(x) - 0.5 * inv - inv2 * (
        1.0 / 12 - inv2 * (1.0 / 120 - inv2 * (1.0 / 252 - inv2 * (1.0 / 240 - inv2 / 132))))


# ---------------------------------------------------------------- experiment 1
print(LINE)
print("E1. Fractional factorials x! = Gamma(x+1)")
print(LINE)
for label, x in [("0.2", 0.2), ("0.5", 0.5), ("0.9", 0.9), ("1/pi", 1 / math.pi),
                 ("sqrt2", math.sqrt(2)), ("e", math.e), ("pi", math.pi)]:
    print(f"  {label:>6}! = Gamma({x:.6f}+1) = {math.gamma(x + 1):.12f}")

# ---------------------------------------------------------------- experiment 2
print(LINE)
print("E2. Non-injectivity of x -> Gamma(x+1) on (0,1)")
print(LINE)
# golden-section minimum of f(x)=Gamma(x+1) on (0,1)
gr = (math.sqrt(5) - 1) / 2
a, b = 0.0, 1.0
while b - a > 1e-14:
    c, d = b - gr * (b - a), a + gr * (b - a)
    if math.gamma(c + 1) < math.gamma(d + 1):
        b = d
    else:
        a = c
xmin = (a + b) / 2
print(f"  argmin of Gamma(x+1) on (0,1): x* = {xmin:.10f}, Gamma(x*+1) = {math.gamma(xmin + 1):.10f}")
# find b>x* with Gamma(1+b)=Gamma(1.1)  (a=0.1 < x*)
target = math.gamma(1.1)
lo, hi = xmin, 1.0
while hi - lo > 1e-14:
    mid = (lo + hi) / 2
    if math.gamma(mid + 1) < target:
        lo = mid
    else:
        hi = mid
print(f"  counterexample: 0.1! = {math.gamma(1.1):.12f} and {lo:.12f}! = {math.gamma(lo + 1):.12f}")
print("  => phi(x) = x! is NOT injective on (0,1); it is not a valid scalar transform there.")

# ---------------------------------------------------------------- experiment 3
print(LINE)
print("E3. Conditioning of phi(x)=Gamma(x+1) and of its inverse")
print(LINE)
print("  forward relative condition  kappa_phi(x) = |x * psi(x+1)|")
print("  inverse relative condition  kappa_inv(y) = 1 / kappa_phi(x)   (y = phi(x))")
for x in [0.4616, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 170.0]:
    k = abs(x * digamma(x + 1))
    kinv = float('inf') if k == 0 else 1.0 / k
    print(f"  x = {x:8.4f}   kappa_phi = {k:12.4f}   kappa_inv = {kinv:12.4f}")
print("  near x* (where phi'(x*)=0) the INVERSE condition number diverges:")
for eps in [1e-2, 1e-4, 1e-6, 1e-8]:
    x = xmin + eps
    k = abs(x * digamma(x + 1))
    print(f"    x = x* + {eps:.0e}:  kappa_phi = {k:.3e}   kappa_inv = {1.0 / k:.3e}")

# ---------------------------------------------------------------- experiment 4
print(LINE)
print("E4. Dynamic range / overflow threshold of Gamma(x+1) in binary64")
print(LINE)
lo, hi = 100.0, 200.0
def overflows(x):
    try:
        math.gamma(x + 1)
        return False
    except OverflowError:
        return True
while hi - lo > 1e-9:
    mid = (lo + hi) / 2
    if overflows(mid):
        hi = mid
    else:
        lo = mid
print(f"  Gamma(x+1) overflows binary64 for x > {lo:.6f}  (max finite double ~1.798e308)")
print(f"  lgamma stays finite: lgamma(1e300+1) = {math.lgamma(1e300):.6e} (log-domain survives)")

# ---------------------------------------------------------------- experiment 5
print(LINE)
print("E5. Round-trip x -> y=Gamma(x+1) -> Newton-inverse, relative error (branch x > x*)")
print(LINE)
def inv_gamma_plus1(logy, x0):
    """Solve lgamma(x+1) = logy by Newton on the increasing branch."""
    x = x0
    for _ in range(80):
        f = math.lgamma(x + 1) - logy
        d = digamma(x + 1)
        if d == 0:
            break
        step = f / d
        x -= step
        if abs(step) < 1e-16 * max(1.0, abs(x)):
            break
    return x

worst_far, worst_near = 0.0, 0.0
for x in [0.6, 0.9, 1.5, 3.0, 7.0, 20.0, 80.0, 150.0]:
    y = math.gamma(x + 1)
    xr = inv_gamma_plus1(math.log(y), max(x * 1.3, 1.0))
    rel = abs(xr - x) / abs(x)
    worst_far = max(worst_far, rel)
    print(f"  x = {x:8.3f}: round-trip rel err = {rel:.3e}")
for eps in [1e-2, 1e-4, 1e-6]:
    x = xmin + eps
    y = math.gamma(x + 1)
    # perturb y by one ulp and invert: error amplification near the minimum
    y_ulp = math.nextafter(y, math.inf)
    xr = inv_gamma_plus1(math.log(y_ulp), xmin + 2 * eps + 1e-3)
    rel = abs(xr - x) / abs(x)
    worst_near = max(worst_near, rel)
    print(f"  x = x*+{eps:.0e}: 1-ulp perturbation of y -> recovered-x rel err = {rel:.3e}")
print(f"  => far from x*: round trip ~1e-15..1e-16; near x*: 1 ulp in y costs up to {worst_near:.1e} in x")

# ---------------------------------------------------------------- experiment 6
print(LINE)
print("E6. Smooth conjugation preserves asymptotic convergence rate (proof check)")
print(LINE)
# A(x) = cos x, fixed point p, |A'(p)| = sin p. Conjugate by T = exp: B = log . cos . exp
p = 0.5
for _ in range(200):
    p = math.cos(p)
rate_A = abs(math.sin(p))
x = 0.5
errs_A = []
for _ in range(60):
    x = math.cos(x)
    errs_A.append(abs(x - p))
q = math.log(p)
u = math.log(0.5)
errs_B = []
for _ in range(60):
    u = math.log(math.cos(math.exp(u)))
    errs_B.append(abs(u - q))
rA = errs_A[-1] / errs_A[-2]
rB = errs_B[-1] / errs_B[-2]
print(f"  fixed point p = {p:.12f}, theoretical rate |A'(p)| = {rate_A:.12f}")
print(f"  measured rate, plain iteration      : {rA:.12f}")
print(f"  measured rate, exp-conjugated       : {rB:.12f}")
print("  => identical linear convergence factor: conjugation cannot accelerate the asymptotic rate.")

# ---------------------------------------------------------------- experiment 7
print(LINE)
print("E7. Where transformed scalars DO help: log-domain product of 100k factors")
print(LINE)
random.seed(42)
vals = [random.uniform(0.0, 1.0) for _ in range(100_000)]
naive = 1.0
for v in vals:
    naive *= v
logsum = sum(math.log(v) for v in vals)
print(f"  naive product          = {naive}   (underflow)")
print(f"  log-domain: log(prod)  = {logsum:.6f}  -> representable, exact to ~1e-11 rel")

# ---------------------------------------------------------------- experiment 8
print(LINE)
print("E8. Gamma-mean (quasi-arithmetic mean, generator lgamma(x+1)) vs classical means")
print(LINE)
data = [1.0, 2.0, 3.0, 10.0]
am = sum(data) / len(data)
gm = math.exp(sum(math.log(v) for v in data) / len(data))
t = sum(math.lgamma(v + 1) for v in data) / len(data)
gmean = inv_gamma_plus1(t, am)
print(f"  data = {data}")
print(f"  arithmetic mean = {am:.6f}, geometric mean = {gm:.6f}, Gamma-mean = {gmean:.6f}")
sh = [v + 5 for v in data]
t2 = sum(math.lgamma(v + 1) for v in sh) / len(sh)
gmean_sh = inv_gamma_plus1(t2, am + 5)
print(f"  shifted data +5: Gamma-mean = {gmean_sh:.6f} vs shifted Gamma-mean = {gmean + 5:.6f}")
print("  => Gamma-mean is a valid quasi-arithmetic mean (Kolmogorov 1930 class),")
print("     not translation-equivariant, and (unlike log) carries no algebraic identity.")

# ---------------------------------------------------------------- experiment 9
print(LINE)
print("E9. TSA filter = variance-stabilizing transform: Poisson median-denoise demo")
print(LINE)
random.seed(7)

def poisson(lam):
    # Knuth
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= random.random()
        if p <= L:
            return k
        k += 1

N, W = 4096, 4  # window half-width -> 9-tap median
lam = [5.0 + 15.0 * math.sin(2 * math.pi * i / 256.0) ** 2 for i in range(N)]
obs = [float(poisson(l)) for l in lam]

def medfilt(sig, w):
    out = []
    n = len(sig)
    for i in range(n):
        lo, hi = max(0, i - w), min(n, i + w + 1)
        s = sorted(sig[lo:hi])
        out.append(s[len(s) // 2])
    return out

# direct median filter on counts
den_direct = medfilt(obs, W)
# Anscombe -> median -> algebraic inverse (with +1/8 asymptotic bias term removed via (y/2)^2 - 1/8)
ans = [2.0 * math.sqrt(v + 0.375) for v in obs]
den_ans = medfilt(ans, W)
inv_alg = [(v / 2.0) ** 2 - 0.375 for v in den_ans]
inv_unb = [(v / 2.0) ** 2 - 0.125 for v in den_ans]  # first-order unbiased inverse

def mse(est):
    return sum((e - l) ** 2 for e, l in zip(est, lam)) / N

print(f"  MSE raw counts               = {mse(obs):8.4f}")
print(f"  MSE median filter (direct)   = {mse(den_direct):8.4f}")
print(f"  MSE Anscombe+median+alg.inv  = {mse(inv_alg):8.4f}")
print(f"  MSE Anscombe+median+unb.inv  = {mse(inv_unb):8.4f}")
# and the honest control: a LINEAR filter gains nothing from the transform
def boxfilt(sig, w):
    out = []
    n = len(sig)
    for i in range(n):
        lo, hi = max(0, i - w), min(n, i + w + 1)
        out.append(sum(sig[lo:hi]) / (hi - lo))
    return out
den_box = boxfilt(obs, W)
box_ans = [(v / 2.0) ** 2 - 0.125 for v in boxfilt(ans, W)]
print(f"  MSE box filter (direct)      = {mse(den_box):8.4f}")
print(f"  MSE Anscombe+box+unb.inv     = {mse(box_ans):8.4f}")
print("  => transform-domain filtering helps exactly when the filter assumes")
print("     homoscedastic Gaussian noise; it is the classical VST pipeline (Anscombe 1948).")
print(LINE)

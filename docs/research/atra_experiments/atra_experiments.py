#!/usr/bin/env python3
"""ATRA falsification/grounding experiments X1-X4 (pure stdlib, fixed seeds).

X1  Branch A / Stage 1: property table of candidate scalar transforms.
X2  Branch C / Stage 2: summation shootout — representation vs compensation.
X3  Branch A+C / Stage 2: adaptive companding for quantization vs Lloyd-Max.
X4  Branch E / Stage 2: stochastic rounding vs round-to-nearest stagnation.
"""
import bisect
import math
import random
import time

EPS = 2.0 ** -52
LINE = "-" * 78


# ================================================================ X1
print(LINE)
print("X1. Scalar transform property table (Stage 1)")
print(LINE)

def logspace(a, b, n):
    la, lb = math.log10(a), math.log10(b)
    return [10 ** (la + (lb - la) * i / (n - 1)) for i in range(n)]

def rt_ulps(f, finv, xs):
    worst = 0.0
    for x in xs:
        try:
            r = finv(f(x))
        except (ValueError, OverflowError):
            return float("inf")
        if x != 0:
            worst = max(worst, abs(r - x) / abs(x) / EPS)
        else:
            worst = max(worst, abs(r) / EPS)
    return worst

MU = 255.0
LN1PMU = math.log1p(MU)

def yj(x, lam=0.5):
    if x >= 0:
        return (math.pow(x + 1, lam) - 1) / lam
    return -(math.pow(1 - x, 2 - lam) - 1) / (2 - lam)

def yj_inv(y, lam=0.5):
    if y >= 0:
        return math.pow(lam * y + 1, 1 / lam) - 1
    return 1 - math.pow(1 - (2 - lam) * y, 1 / (2 - lam))

TRANSFORMS = [
    ("log/exp", "x>0", math.log, math.exp,
     lambda x: 1 / x, logspace(1e-300, 1e300, 601)),
    ("log1p/expm1", "x>-1", math.log1p, math.expm1,
     lambda x: 1 / (1 + x), [-1 + 1e-15] + logspace(1e-15, 1e300, 401)),
    ("asinh (signed log)", "R", math.asinh, math.sinh,
     lambda x: 1 / math.sqrt(1 + x * x),
     [-v for v in logspace(1e-300, 1e300, 200)] + logspace(1e-300, 1e300, 200)),
    ("sqrt/square", "x>=0", math.sqrt, lambda y: y * y,
     lambda x: 0.5 / math.sqrt(x) if x > 0 else float("inf"), logspace(1e-300, 1e300, 401)),
    ("Anscombe", "x>=0", lambda x: 2 * math.sqrt(x + 0.375),
     lambda y: (y / 2) ** 2 - 0.375,
     lambda x: 1 / math.sqrt(x + 0.375), logspace(1e-6, 1e12, 401)),
    ("Box-Cox l=0.5", "x>0", lambda x: (math.sqrt(x) - 1) / 0.5,
     lambda y: (0.5 * y + 1) ** 2,
     lambda x: 0.5 / math.sqrt(x) * 2, logspace(1e-10, 1e10, 401)),
    ("Yeo-Johnson l=0.5", "R", yj, yj_inv,
     lambda x: math.pow(x + 1, -0.5) if x >= 0 else math.pow(1 - x, 0.5),
     [-v for v in logspace(1e-10, 1e10, 200)] + logspace(1e-10, 1e10, 200)),
    ("Box-Cox l=0.5 expm1", "x>0",
     lambda x: math.expm1(0.5 * math.log(x)) / 0.5,
     lambda y: math.exp(math.log1p(0.5 * y) / 0.5),
     lambda x: 0.5 / math.sqrt(x) * 2, logspace(1e-10, 1e10, 401)),
    ("logit/sigmoid", "(0,1)", lambda x: math.log(x / (1 - x)),
     lambda y: 1 / (1 + math.exp(-y)),
     lambda x: 1 / (x * (1 - x)), [1e-16, 1e-8, 1e-3, 0.1, 0.5, 0.9, 0.999, 1 - 1e-8, 1 - 1e-16]),
    ("mu-law (mu=255)", "[-1,1]",
     lambda x: math.copysign(math.log1p(MU * abs(x)) / LN1PMU, x),
     lambda y: math.copysign((math.pow(1 + MU, abs(y)) - 1) / MU, y),
     lambda x: MU / ((1 + MU * abs(x)) * LN1PMU),
     [i / 500 - 1 for i in range(1001)] + [1e-12, -1e-12]),
    ("mu-law expm1-inverse", "[-1,1]",
     lambda x: math.copysign(math.log1p(MU * abs(x)) / LN1PMU, x),
     lambda y: math.copysign(math.expm1(abs(y) * LN1PMU) / MU, y),
     lambda x: MU / ((1 + MU * abs(x)) * LN1PMU),
     [i / 500 - 1 for i in range(1001)] + [1e-12, -1e-12]),
    ("tanh/atanh", "R", math.tanh, math.atanh,
     lambda x: 1 - math.tanh(x) ** 2, [i / 10 for i in range(1, 180)]),
    ("softsign", "R", lambda x: x / (1 + abs(x)),
     lambda y: y / (1 - abs(y)),
     lambda x: 1 / (1 + abs(x)) ** 2,
     [-v for v in logspace(1e-10, 1e12, 200)] + logspace(1e-10, 1e12, 200)),
]

print(f"{'transform':<20} {'domain':<8} {'max RT err (ulps)':>18} {'min d/dx':>10} {'max d/dx':>10}")
for name, dom, f, finv, deriv, xs in TRANSFORMS:
    ulps = rt_ulps(f, finv, xs)
    ds = [deriv(x) for x in xs if math.isfinite(deriv(x))]
    print(f"{name:<20} {dom:<8} {ulps:>18.1f} {min(ds):>10.2e} {max(ds):>10.2e}")

# saturation thresholds (FP-effective loss of injectivity)
lo, hi = 1.0, 40.0
while hi - lo > 1e-9:
    mid = (lo + hi) / 2
    if math.tanh(mid) == 1.0:
        hi = mid
    else:
        lo = mid
print(f"\n  tanh saturates to exactly 1.0 in binary64 for x >= {hi:.6f}")
print(f"  => tanh/atanh loses injectivity in FP beyond that point (atanh(1.0) = inf).")
print(f"  softsign(1e12) = {1e12/(1+1e12):.17f} (still < 1: slower saturation, but")
print(f"  round-trip already fails: inverse of it = {1e12/(1+1e12)/(1-1e12/(1+1e12)):.4e})")

# ================================================================ X2
print(LINE)
print("X2. Summation: representation change vs compensated algorithms (Stage 2)")
print(LINE)

def kahan(xs):
    s = c = 0.0
    for x in xs:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s

def neumaier(xs):
    s = c = 0.0
    for x in xs:
        t = s + x
        if abs(s) >= abs(x):
            c += (s - t) + x
        else:
            c += (x - t) + s
        s = t
    return s + c

def pairwise(xs):
    n = len(xs)
    if n <= 8:
        s = 0.0
        for x in xs:
            s += x
        return s
    return pairwise(xs[: n // 2]) + pairwise(xs[n // 2:])

def logaddexp(a, b):
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))

print("(a) canonical Kahan-killer  [1.0, 1e100, 1.0, -1e100]  (true sum = 2.0):")
data = [1.0, 1e100, 1.0, -1e100]
print(f"    naive = {sum(data)},  Kahan = {kahan(data)},  Neumaier = {neumaier(data)},  fsum = {math.fsum(data)}")

print("(b) positive terms, huge dynamic range: x_i = 10^U, U~Unif(-30,30), n = 100000")
random.seed(20260716)
xs = [10 ** random.uniform(-30, 30) for _ in range(100_000)]
ref = math.fsum(xs)
t0 = time.perf_counter(); s_naive = sum(xs); t_naive = time.perf_counter() - t0
t0 = time.perf_counter(); s_neu = neumaier(xs); t_neu = time.perf_counter() - t0
s_pair = pairwise(xs)
s_sorted = sum(sorted(xs))
t0 = time.perf_counter()
acc = math.log(xs[0])
for v in xs[1:]:
    acc = logaddexp(acc, math.log(v))
t_log = time.perf_counter() - t0
s_log = math.exp(acc)
for lbl, v in [("naive", s_naive), ("sorted-ascending", s_sorted), ("pairwise", s_pair),
               ("Kahan", kahan(xs)), ("Neumaier", s_neu), ("log-domain (LSE)", s_log)]:
    print(f"    {lbl:<18} rel err = {abs(v - ref) / ref:.2e}")
print(f"    time: naive {t_naive*1e3:.0f} ms, Neumaier {t_neu*1e3:.0f} ms, log-domain {t_log*1e3:.0f} ms")

print("(c) signed cancellation: pairs (a_i, -a_i) + small signal, n = 60000")
random.seed(7)
big = [10 ** random.uniform(10, 16) for _ in range(20_000)]
small = [random.uniform(-1, 1) for _ in range(20_000)]
xs = []
for b, d in zip(big, small):
    xs.extend((b, -b, d))
random.shuffle(xs)
ref = math.fsum(xs)
print(f"    true sum = {ref:.6f}")
for lbl, v in [("naive", sum(xs)), ("pairwise", pairwise(xs)),
               ("Kahan", kahan(xs)), ("Neumaier", neumaier(xs))]:
    err = abs(v - ref) / abs(ref)
    print(f"    {lbl:<18} rel err = {err:.2e}")
print("    log-domain: UNDEFINED for signed terms (needs sign-tracked LSE at ~2x cost,")
print("    and cancellation then happens in the exp/expm1 stage instead - not removed).")

# ================================================================ X3
print(LINE)
print("X3. Quantization to L=256 levels of lognormal(0, sigma=2) data (Stage 2)")
print("    adaptive representation selection vs fixed transforms vs Lloyd-Max oracle")
print(LINE)

L = 256

def quant_uniform_in(f, finv, dev, test):
    lo = min(f(v) for v in dev)
    hi = max(f(v) for v in dev)
    step = (hi - lo) / L
    out = []
    for v in test:
        idx = min(L - 1, max(0, int((f(v) - lo) / step)))
        out.append(finv(lo + (idx + 0.5) * step))
    return out

def codebook_of(f, finv, dev):
    lo = min(f(v) for v in dev)
    hi = max(f(v) for v in dev)
    step = (hi - lo) / L
    return [finv(lo + (i + 0.5) * step) for i in range(L)]

def lloyd_max(dev, init_cents, iters=500):
    """Lloyd-Max refinement from a given codebook (monotone MSE descent,
    so the result is at least as good as the init on dev)."""
    data = sorted(dev)
    n = len(data)
    prefix = [0.0]
    for v in data:
        prefix.append(prefix[-1] + v)
    cents = sorted(init_cents)
    for _ in range(iters):
        bounds = [(cents[i] + cents[i + 1]) / 2 for i in range(L - 1)]
        idx = [bisect.bisect_left(data, b) for b in bounds]
        idx = [0] + idx + [n]
        new = []
        for i in range(L):
            a, b = idx[i], idx[i + 1]
            new.append((prefix[b] - prefix[a]) / (b - a) if b > a else cents[i])
        if new == cents:
            break
        cents = new
    bounds = [(cents[i] + cents[i + 1]) / 2 for i in range(L - 1)]
    return cents, bounds

def sqnr_db(test, est):
    num = sum(v * v for v in test)
    den = sum((v - e) ** 2 for v, e in zip(test, est))
    return 10 * math.log10(num / den)

results = {}
for seed in (1, 2, 3):
    random.seed(seed)
    data = [math.exp(random.gauss(0, 1.5)) for _ in range(65_536)]
    dev, test = data[:32_768], data[32_768:]
    # finite-support quantizer: overload region set at dev 0.05%/99.95% quantiles
    # (standard granular-vs-overload design; both dev and test are clipped so all
    # methods share the same loading factor)
    sd = sorted(dev)
    lo_q, hi_q = sd[len(sd) // 2000], sd[-1 - len(sd) // 2000]
    dev = [min(max(v, lo_q), hi_q) for v in dev]
    test = [min(max(v, lo_q), hi_q) for v in test]
    xmax = max(dev)

    r = {}
    r["uniform (direct)"] = sqnr_db(test, quant_uniform_in(lambda x: x, lambda y: y, dev, test))
    r["log-companding"] = sqnr_db(test, quant_uniform_in(math.log, math.exp, dev, test))
    r["mu-law (mu=255)"] = sqnr_db(test, quant_uniform_in(
        lambda x: math.log1p(MU * x / xmax) / LN1PMU,
        lambda y: (math.pow(1 + MU, y) - 1) / MU * xmax, dev, test))
    # adaptive Box-Cox: select lambda on dev, evaluate on test (per section 13 protocol)
    best_lam, best = None, -1e9
    for lam in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]:
        f = (lambda x, l=lam: (math.pow(x, l) - 1) / l)
        finv = (lambda y, l=lam: math.pow(max(l * y + 1, 1e-300), 1 / l))
        s = sqnr_db(dev, quant_uniform_in(f, finv, dev, dev))
        if s > best:
            best, best_lam = s, lam
    f = (lambda x, l=best_lam: (math.pow(x, l) - 1) / l)
    finv = (lambda y, l=best_lam: math.pow(max(l * y + 1, 1e-300), 1 / l))
    r[f"Box-Cox (dev-selected l={best_lam})"] = sqnr_db(test, quant_uniform_in(f, finv, dev, test))
    cents, bounds = lloyd_max(dev, codebook_of(f, finv, dev))
    est = [cents[bisect.bisect_left(bounds, v)] for v in test]
    r["Lloyd-Max (compander init)"] = sqnr_db(test, est)
    for k, v in r.items():
        results.setdefault(k, []).append(v)

print(f"{'quantizer':<32} {'test SQNR dB (mean +/- sd over 3 seeds)'}")
for k, vs in results.items():
    m = sum(vs) / len(vs)
    sd = (sum((v - m) ** 2 for v in vs) / (len(vs) - 1)) ** 0.5
    print(f"  {k:<30} {m:8.2f} +/- {sd:.2f}")

# ================================================================ X4
print(LINE)
print("X4. Stochastic rounding vs round-to-nearest: stagnation (Stage 2 / Branch E)")
print(LINE)

P = 10  # simulated significand bits

def round_p(v, mode, rng):
    if v == 0:
        return 0.0
    m, e = math.frexp(v)          # v = m * 2^e, m in [0.5, 1)
    scale = 2.0 ** (e - 1 - P)    # grid spacing at this binade
    q = v / scale
    lo = math.floor(q)
    frac = q - lo
    if mode == "rn":
        lo = round(q)             # round-half-even
    elif frac > 0 and rng.random() < frac:
        lo += 1
    return lo * scale

N_STEPS, INC, TRUE = 200_000, 1e-4, 20.0
s = 0.0
for _ in range(N_STEPS):
    s = round_p(s + INC, "rn", None)
print(f"  precision p = {P} bits, {N_STEPS} increments of {INC}, true sum = {TRUE}")
print(f"  round-to-nearest: final = {s:.6f}  (stagnates once inc < ulp(s)/2 ~ s*2^-{P+1})")
finals = []
for seed in range(5):
    rng = random.Random(1000 + seed)
    s = 0.0
    for _ in range(N_STEPS):
        s = round_p(s + INC, "sr", rng)
    finals.append(s)
m = sum(finals) / len(finals)
sd = (sum((v - m) ** 2 for v in finals) / (len(finals) - 1)) ** 0.5
print(f"  stochastic rounding: final = {m:.4f} +/- {sd:.4f} (5 seeds) — unbiased, no stagnation")
print(LINE)

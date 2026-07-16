#!/usr/bin/env python3
"""CANR experiments Y1-Y6 (pure stdlib; Decimal prec-60 multiprecision reference;
fixed seeds; dev/held-out splits where selection occurs).

Y1  Box-Cox independent verification vs Decimal-60 reference (mission section 9).
Y2  mu-law independent verification vs Decimal-60 reference.
Y3  H3: joint representation+operator selection (log domain needs LSE).
Y4  H4: reproducibility across permutations (order-invariance of reductions).
Y5  Reduction design study: accuracy table on canonical datasets.
Y6  H2: certificate-driven selection prototype (Higham bounds as certificates).
"""
import math
import random
import time
from decimal import Decimal, getcontext
from fractions import Fraction

getcontext().prec = 60
LINE = "-" * 78
U = 2.0 ** -53  # unit roundoff, binary64


def d(x):
    return Decimal(x)


def dexp(x):
    return x.exp()


def dln(x):
    return x.ln()


def ulps_vs_ref(val_f64, ref_dec):
    ref_f = float(ref_dec)
    u = math.ulp(ref_f) if ref_f != 0 else math.ulp(0.0)
    return float(abs(d(val_f64) - ref_dec) / d(u))


# ================================================================ Y1
print(LINE)
print("Y1. Box-Cox verification vs Decimal(prec=60) reference  phi(x)=(x^l - 1)/l")
print(LINE)

def bc_ref(x, lam):
    return (dexp(d(lam) * dln(d(x))) - 1) / d(lam)

def pow_ref(x, lam):
    return dexp(d(lam) * dln(d(x)))

def logspace(a, b, n):
    la, lb = math.log10(a), math.log10(b)
    return [10 ** (la + (lb - la) * i / (n - 1)) for i in range(n)]

XS = logspace(1e-12, 1e12, 25)
print("Certificate per point: RT_ulps <= 8*(k_rt+1); k_rt*u >= 0.5 => 'invalid region'")
print(f"{'lambda':>7} | {'fwd ULP pow':>12} {'fwd ULP expm1':>14} | "
      f"{'RT ULP shifted':>14} {'RT ULP x^l':>11} | {'max k_rt':>10} {'cert':>12}")
for lam in (0.1, 0.25, 0.5, 0.75, 1.5):
    fwd_pow = fwd_e1 = rt_sh = rt_st = krt_max = 0.0
    viol = 0
    invalid_predicted = invalid_observed = 0
    for x in XS:
        ref = bc_ref(x, lam)
        v_pow = (x ** lam - 1.0) / lam
        v_e1 = math.expm1(lam * math.log(x)) / lam
        fwd_pow = max(fwd_pow, ulps_vs_ref(v_pow, ref))
        fwd_e1 = max(fwd_e1, ulps_vs_ref(v_e1, ref))
        krt = abs((x ** lam - 1.0) / (lam * x ** lam))
        krt_max = max(krt_max, krt)
        if krt * U >= 0.5:
            invalid_predicted += 1
        try:
            r_e1 = math.exp(math.log1p(lam * v_e1) / lam)
            e_sh = abs(r_e1 - x) / math.ulp(x)
        except ValueError:
            e_sh = float("inf")
        if not math.isfinite(e_sh) or abs((r_e1 if math.isfinite(e_sh) else 0) - x) / abs(x) > 0.5:
            invalid_observed += 1
        else:
            rt_sh = max(rt_sh, e_sh)
            if e_sh > 8.0 * (krt + 1.0):
                viol += 1
        v_st = x ** lam                       # unshifted storage: k_rt == 1/lam, flat
        r_st = v_st ** (1.0 / lam)
        rt_st = max(rt_st, abs(r_st - x) / math.ulp(x))
    cert = (f"{25-invalid_predicted}/25 ok, {viol} viol"
            f"{', inv:' + str(invalid_predicted) + '=' + str(invalid_observed) if invalid_predicted else ''}")
    print(f"{lam:>7} | {fwd_pow:>12.1f} {fwd_e1:>14.1f} | "
          f"{rt_sh:>14.1f} {rt_st:>11.1f} | {krt_max:>10.3g} {cert:>12}")
print("  (viol = points where certificate under-predicted the valid-region RT error;")
print("   inv:a=b means a points predicted invalid (k_rt*u>=0.5), b observed total loss)")

print("\nDownstream operator test (mission 9: measure before preferring x^l storage):")
print("Box-Cox quasi-arithmetic mean, l=0.25, n=4096, 3 seeds; rel err of decoded mean")
lam = 0.25
for scale, label in ((1.0, "standard scale (lognormal s=1.5)"),
                     (1e-8, "tiny scale (x * 1e-8)")):
    errs_sh, errs_un = [], []
    for seed in (1, 2, 3):
        random.seed(seed)
        xs = [scale * math.exp(random.gauss(0, 1.5)) for _ in range(4096)]
        # (i) shifted storage: mean of (x^l-1)/l  -> decode
        t = math.fsum((x ** lam - 1.0) / lam for x in xs) / len(xs)
        dec_sh = (lam * t + 1.0) ** (1.0 / lam)
        # (ii) unshifted storage: mean of x^l -> decode (shift applied symbolically)
        m = math.fsum(x ** lam for x in xs) / len(xs)
        dec_un = m ** (1.0 / lam)
        # Decimal reference
        acc = Decimal(0)
        for x in xs:
            acc += pow_ref(x, lam)
        mref = acc / len(xs)
        dec_ref = dexp(dln(mref) / d(lam))
        errs_sh.append(float(abs(d(dec_sh) - dec_ref) / dec_ref))
        errs_un.append(float(abs(d(dec_un) - dec_ref) / dec_ref))
    print(f"  {label:<38} shifted: {max(errs_sh):.2e}   unshifted: {max(errs_un):.2e}")

# ================================================================ Y2
print(LINE)
print("Y2. mu-law verification vs Decimal reference (mu=255)")
print(LINE)
MU = 255.0
LN1PMU = math.log1p(MU)
DMU, DLN = d(MU), dln(d(1) + d(MU))
worst_naive = worst_e1 = worst_fwd = 0.0
for mag in (1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 0.1, 0.5, 0.99):
    for s in (1.0, -1.0):
        x = s * mag
        y = math.copysign(math.log1p(MU * abs(x)) / LN1PMU, x)
        yref = (dln(d(1) + DMU * d(abs(x))) / DLN) * (1 if s > 0 else -1)
        worst_fwd = max(worst_fwd, ulps_vs_ref(y, yref))
        inv_naive = math.copysign((math.pow(1 + MU, abs(y)) - 1) / MU, y)
        inv_e1 = math.copysign(math.expm1(abs(y) * LN1PMU) / MU, y)
        worst_naive = max(worst_naive, abs(inv_naive - x) / math.ulp(x))
        worst_e1 = max(worst_e1, abs(inv_e1 - x) / math.ulp(x))
print(f"  encode fwd err vs reference : {worst_fwd:.1f} ulps")
print(f"  round trip, naive pow-1 inverse : {worst_naive:.3g} ulps")
print(f"  round trip, expm1 inverse       : {worst_e1:.3g} ulps")

# ================================================================ Y3
print(LINE)
print("Y3. H3: representation alone vs representation + matched operator")
print(LINE)
random.seed(42)
vals = [random.uniform(0.0, 1.0) for _ in range(100_000)]
p = 1.0
for v in vals:
    p *= v
logs = [math.log(v) for v in vals]
print(f"(a) product of 1e5 U(0,1) factors: direct = {p} (underflow);"
      f" log-repr + sum = {math.fsum(logs):.4f} (exact ~1e-11)")
lp = [-800.0 + random.uniform(0, 2) for _ in range(100_000)]
naive_op = math.fsum(math.exp(v) for v in lp)          # exp-then-sum
mx = max(lp)
lse = mx + math.log(math.fsum(math.exp(v - mx) for v in lp))
print(f"(b) sum of 1e5 probs given as logs near -800:")
print(f"    log-repr + WRONG operator (exp,sum) = {naive_op}  (underflow -> 0)")
print(f"    log-repr + LSE operator             = {lse:.6f}  (correct log-sum)")
logits = [800.0, 799.0, 795.0]
try:
    sm = [math.exp(v) for v in logits]
    tot = sum(sm)
    naive_sm = [v / tot for v in sm]
except OverflowError:
    naive_sm = "OverflowError"
mx = max(logits)
sh = [math.exp(v - mx) for v in logits]
tot = sum(sh)
print(f"(c) softmax(logits max=800): naive = {naive_sm}; max-shifted = "
      f"{[round(v / tot, 6) for v in sh]}")

# ================================================================ Y4
print(LINE)
print("Y4. H4: reproducibility across summation orders (20k terms, 10 permutations)")
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

def klein(xs):
    s = cs = ccs = 0.0
    for x in xs:
        t = s + x
        if abs(s) >= abs(x):
            c = (s - t) + x
        else:
            c = (x - t) + s
        s = t
        t2 = cs + c
        if abs(cs) >= abs(c):
            cc = (cs - t2) + c
        else:
            cc = (c - t2) + cs
        cs = t2
        ccs += cc
    return s + cs + ccs

def pairwise(xs):
    n = len(xs)
    if n <= 8:
        s = 0.0
        for x in xs:
            s += x
        return s
    return pairwise(xs[: n // 2]) + pairwise(xs[n // 2:])

def superacc(xs, guard_bits=200):
    """Exact-to-truncation integer superaccumulator; order-invariant."""
    emax = max(math.frexp(x)[1] for x in xs if x != 0.0)
    e0 = emax - guard_bits
    acc = 0
    for x in xs:
        if x == 0.0:
            continue
        m, e = math.frexp(x)
        mi = int(m * (1 << 53))          # exact 53-bit integer mantissa
        sh = (e - 53) - e0
        acc += mi << sh if sh >= 0 else mi >> (-sh)
    return math.ldexp(float(acc), e0) if abs(acc) < (1 << 1020) else acc * (2.0 ** e0)

random.seed(20260716)
base = [math.copysign(10 ** random.uniform(-10, 10), random.uniform(-1, 1)) for _ in range(20_000)]
exact = sum(Fraction(x) for x in base)
exact_f = float(exact)
methods = [("naive", sum), ("pairwise(order-tree)", pairwise), ("Kahan", kahan),
           ("Neumaier", neumaier), ("Klein", klein), ("superaccumulator", superacc)]
print(f"{'method':<22} {'distinct results/10 perms':>26} {'max rel err vs exact':>22}")
for name, fn in methods:
    outs = set()
    worst = 0.0
    for pidx in range(10):
        rng = random.Random(500 + pidx)
        xs = base[:]
        rng.shuffle(xs)
        v = fn(xs)
        outs.add(v)
        worst = max(worst, abs(v - exact_f) / abs(exact_f))
    print(f"{name:<22} {len(outs):>26} {worst:>22.2e}")

# ================================================================ Y5
print(LINE)
print("Y5. Reduction design study: accuracy on canonical datasets")
print(LINE)
random.seed(7)
big = [10 ** random.uniform(10, 16) for _ in range(20_000)]
small = [random.uniform(-1, 1) for _ in range(20_000)]
cancel = []
for b, dd in zip(big, small):
    cancel.extend((b, -b, dd))
random.shuffle(cancel)
random.seed(20260716)
wide = [10 ** random.uniform(-30, 30) for _ in range(100_000)]

for label, data in (("wide-range positive (n=1e5)", wide),
                    ("signed cancellation (n=6e4)", cancel)):
    ref = float(sum(Fraction(x) for x in data))
    print(f"  dataset: {label}, exact = {ref:.6e}")
    for name, fn in [("naive", sum), ("pairwise", pairwise), ("Kahan", kahan),
                     ("Neumaier", neumaier), ("Klein", klein),
                     ("superaccumulator", superacc)]:
        t0 = time.perf_counter()
        v = fn(data)
        dt = time.perf_counter() - t0
        print(f"    {name:<18} rel err = {abs(v - ref) / abs(ref):.2e}   ({dt*1e3:6.1f} ms indicative)")

# ================================================================ Y6
print(LINE)
print("Y6. H2: certificate-driven selection prototype (summation)")
print(LINE)
print("Certificates (Higham 2002): |err| <= K(method) * C_sum,  C_sum = sum|x|/|sum x|")
print("Cost model: flops/element (Rust-relevant), not Python wall time.")

CAND = [  # (name, fn, K(n) coefficient of C_sum, flops/element)
    ("naive", sum, lambda n: n * U, 1),
    ("pairwise", pairwise, lambda n: math.ceil(math.log2(n)) * U * 1.1, 1),
    ("Neumaier", neumaier, lambda n: 2 * U + n * n * U * U, 7),
    ("superaccumulator", superacc, lambda n: 2 ** -140, 40),
]

def make_workload(kind, seed, n=30_000):
    rng = random.Random(seed)
    if kind == "benign uniform":
        return [rng.uniform(0.5, 1.5) for _ in range(n)]
    if kind == "wide range":
        return [10 ** rng.uniform(-25, 25) for _ in range(n)]
    out = []
    for _ in range(n // 3):
        b = 10 ** rng.uniform(8, 14)
        out.extend((b, -b, rng.uniform(-1, 1)))
    rng.shuffle(out)
    return out

for kind, tau in (("benign uniform", 1e-9), ("wide range", 1e-13), ("cancellation", 1e-8)):
    dev = make_workload(kind, seed=1)
    n = len(dev)
    ref_dev = float(sum(Fraction(x) for x in dev))
    csum = math.fsum(abs(x) for x in dev) / abs(ref_dev)
    chosen = None
    for name, fn, K, flops in CAND:                     # candidates sorted by cost
        if K(n) * csum <= tau:
            chosen = (name, fn, flops)
            break
    if chosen is None:
        chosen = ("superaccumulator", superacc, 40)
    # held-out evaluation, 3 fresh seeds
    worst = 0.0
    for seed in (11, 12, 13):
        test = make_workload(kind, seed=seed)
        ref = float(sum(Fraction(x) for x in test))
        worst = max(worst, abs(chosen[1](test) - ref) / abs(ref))
    print(f"  workload {kind:<16} tau={tau:.0e}  C_sum={csum:9.3g}  ->  {chosen[0]:<16}"
          f" ({chosen[2]} flops/elt)  held-out err = {worst:.2e}  {'PASS' if worst <= tau else 'FAIL'}")
print(LINE)

#!/usr/bin/env python3
"""ATRA experiment X5 (Branch F / Stage 5): quaternion rotation averaging.

Compares three estimators of a mean rotation from noisy unit-quaternion samples:
  1. componentwise arithmetic mean + renormalization (hemisphere-aligned),
  2. chordal L2 mean (eigenvector of sum q q^T; Markley et al. 2007),
  3. intrinsic (Karcher) mean via iterated log/exp averaging.
Pure stdlib; fixed seeds; mean +/- sd over trials.
"""
import math
import random

def qmul(p, q):
    a, b, c, d = p
    e, f, g, h = q
    return (a*e - b*f - c*g - d*h,
            a*f + b*e + c*h - d*g,
            a*g - b*h + c*e + d*f,
            a*h + b*g - c*f + d*e)

def qconj(q):
    return (q[0], -q[1], -q[2], -q[3])

def qnorm(q):
    n = math.sqrt(sum(v*v for v in q))
    return tuple(v/n for v in q)

def qexp(v):
    """exp of pure quaternion (0, v)."""
    t = math.sqrt(sum(x*x for x in v))
    if t < 1e-300:
        return (1.0, 0.0, 0.0, 0.0)
    s = math.sin(t) / t
    return (math.cos(t), v[0]*s, v[1]*s, v[2]*s)

def qlog(q):
    """log of unit quaternion -> pure vector part."""
    w = max(-1.0, min(1.0, q[0]))
    vnorm = math.sqrt(q[1]**2 + q[2]**2 + q[3]**2)
    if vnorm < 1e-300:
        return (0.0, 0.0, 0.0)
    t = math.atan2(vnorm, w)
    return (q[1]/vnorm*t, q[2]/vnorm*t, q[3]/vnorm*t)

def rot_err_deg(p, q):
    d = abs(sum(a*b for a, b in zip(p, q)))
    return 2 * math.acos(min(1.0, d)) * 180 / math.pi

def rand_unit_quat(rng):
    q = (rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))
    return qnorm(q)

def est_componentwise(qs):
    ref = qs[0]
    acc = [0.0]*4
    for q in qs:
        s = 1.0 if sum(a*b for a, b in zip(q, ref)) >= 0 else -1.0
        for i in range(4):
            acc[i] += s*q[i]
    return qnorm(acc)

def est_chordal(qs):
    # power iteration on M = sum q q^T (sign-invariant)
    M = [[0.0]*4 for _ in range(4)]
    for q in qs:
        for i in range(4):
            for j in range(4):
                M[i][j] += q[i]*q[j]
    v = (1.0, 0.0, 0.0, 0.0)
    for _ in range(200):
        w = tuple(sum(M[i][j]*v[j] for j in range(4)) for i in range(4))
        v = qnorm(w)
    return v

def est_karcher(qs, iters=20):
    m = est_componentwise(qs)   # warm start
    for _ in range(iters):
        acc = [0.0, 0.0, 0.0]
        for q in qs:
            s = 1.0 if sum(a*b for a, b in zip(q, m)) >= 0 else -1.0
            d = qmul(qconj(m), tuple(s*v for v in q))
            l = qlog(qnorm(d))
            for i in range(3):
                acc[i] += l[i]
        step = tuple(a/len(qs) for a in acc)
        if max(abs(a) for a in step) < 1e-14:
            break
        m = qnorm(qmul(m, qexp(step)))
    return m

N_OBS, N_TRIALS = 100, 20
print(f"{N_OBS} noisy observations per trial, {N_TRIALS} trials; error to true rotation (deg)")
print(f"{'sigma(rad)':<11} {'componentwise+renorm':>22} {'chordal (Markley)':>19} {'Karcher (log/exp)':>19}")
for sigma in (0.2, 0.8, 1.5):
    errs = {"cw": [], "ch": [], "ka": []}
    for trial in range(N_TRIALS):
        rng = random.Random(9000 + trial)
        q0 = rand_unit_quat(rng)
        qs = []
        for _ in range(N_OBS):
            axis = qnorm((0.0, rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)))[1:]
            ang = rng.gauss(0, sigma)
            qs.append(qnorm(qmul(q0, qexp(tuple(0.5*ang*a for a in axis)))))
        errs["cw"].append(rot_err_deg(est_componentwise(qs), q0))
        errs["ch"].append(rot_err_deg(est_chordal(qs), q0))
        errs["ka"].append(rot_err_deg(est_karcher(qs), q0))
    def ms(v):
        m = sum(v)/len(v)
        sd = (sum((x-m)**2 for x in v)/(len(v)-1))**0.5
        return f"{m:6.3f} +/- {sd:5.3f}"
    print(f"{sigma:<11} {ms(errs['cw']):>22} {ms(errs['ch']):>19} {ms(errs['ka']):>19}")
print("\nNote: all three estimators are standard attitude-estimation methods;")
print("the question is whether nonlinear (polar/log) representations beat the")
print("linear chordal mean — Markley et al. (2007) already answer this domain.")

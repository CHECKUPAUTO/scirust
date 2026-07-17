#!/usr/bin/env python3
"""ANEE experiments -- reproducible falsification checks for the Adaptive
Numerical Execution Engine investigation
(docs/research/ANEE_ADAPTIVE_NUMERICAL_EXECUTION_ENGINE_2026-07-17.md).

Pure stdlib, fixed seeds. Companion to tsa_experiments.py / atra_experiments.py
/ canr_experiments.py; does not repeat their measurements. Where this phase's
formal claims are *exact identities* (Z3) rather than first-order
approximations (as in [ATRA P4]), the experiment checks agreement to many more
digits than the earlier phases' ulp-level checks, and says so.

Z1: combinatorial size of the joint search space R x O x A x T x Q x M x H
    versus the sum of per-axis sizes, using SciRust's OWN current dictionaries
    where they exist (scirust-core::transform_search::Representation,
    scirust-core::autotune_accumulate::AccumMethod,
    scirust-simd::dispatch::BackendKind) and clearly-labelled conservative
    placeholders for the axes SciRust does not yet enumerate.
Z2: non-separability / interaction -- a synthetic 2-axis objective with a
    tunable interaction strength; compares independent-per-axis choice,
    one-pass-per-axis coordinate descent, and full joint grid search; reports
    the optimality gap as interaction strength grows from 0.
Z3: kappa_rt chain-rule composition -- numerically verifies that the
    round-trip condition number of a composed representation map (power(lam1)
    then log) equals the *exact* product of the two hops' kappa_rt (an
    elasticity/log-derivative chain-rule identity, not a first-order
    approximation), against a Decimal high-precision reference.
Z4: determinism composition -- empirically confirms that piping a provably
    order-invariant (D1-style, exact-arithmetic) stage into an
    order-dependent (D0) stage yields an order-dependent *overall* pipeline:
    determinism composes as a meet (weakest link), not an average.
"""

import itertools
import random
from decimal import Decimal, getcontext

getcontext().prec = 60
U = 2.0 ** -53  # binary64 unit roundoff


def d(x):
    return Decimal(x)


# --------------------------------------------------------------------------
# Z1 -- combinatorial blow-up of the joint search space
# --------------------------------------------------------------------------


def z1_combinatorial_blowup():
    print("=" * 78)
    print("Z1: joint search space |R x O x A x T x Q x M x H| vs. sum of axes")
    print("=" * 78)

    # Grounded in SciRust's actual current code (verified by direct reading):
    R = 7  # scirust-core::transform_search::Representation (7 enum variants)
    A = 5  # scirust-core::autotune_accumulate::AccumMethod (5 enum variants)
    H = 7  # scirust-simd::dispatch::BackendKind (7 enum variants)
    # T ranges over "no transform" plus the same representation dictionary in
    # every treatment this investigation and [ATRA]/[CANR] use (T and R are
    # not independently catalogued in SciRust; this is the most defensible
    # grounded estimate, not an invented number).
    T = R + 1
    # O, Q, M are NOT catalogued anywhere in SciRust today (verified: no
    # operator-implementation registry, no precision-level registry, no
    # execution-model registry exists). These are illustrative, deliberately
    # SMALL placeholders (a typical kernel might realistically have this many
    # hand-written variants) -- flagged as such, not measured.
    O_illustrative = 4
    Q_illustrative = 4
    M_illustrative = 3

    axes_grounded = {"R": R, "A": A, "H": H, "T (derived)": T}
    axes_illustrative = {"O": O_illustrative, "Q": Q_illustrative, "M": M_illustrative}
    all_axes = {**axes_grounded, **axes_illustrative}

    independent_cost = sum(all_axes.values())
    joint_cost = 1
    for v in all_axes.values():
        joint_cost *= v

    print(f"Grounded axes (read directly from SciRust source): {axes_grounded}")
    print(f"Illustrative placeholder axes (NOT catalogued in SciRust today): {axes_illustrative}")
    print(f"Sum of per-axis sizes (7 independent single-axis searches): {independent_cost}")
    print(f"Product of axis sizes (one joint search over the full plan space): {joint_cost}")
    print(f"Blow-up factor: {joint_cost / independent_cost:.1f}x")
    print()
    print("Sensitivity: if every axis had size k, independent cost is 7k and joint")
    print("cost is k^7. Table (k, 7k, k^7):")
    for k in (3, 5, 7, 10):
        print(f"  k={k:>2}  independent={7*k:>6}  joint={k**7:>12,}")
    print()
    return joint_cost, independent_cost


# --------------------------------------------------------------------------
# Z2 -- non-separability: independent vs. coordinate-descent vs. joint search
# --------------------------------------------------------------------------


def z2_objective(x1, x2, lam, n1, n2):
    """A synthetic cost (LOWER is better) over two discrete axes: a mixture of
    two axis-aligned Gaussian *bumps* (rewards). A single bump is separable
    (product of a function of x1 and a function of x2), but the SUM of two
    bumps at different centers is NOT expressible as f(x1) + g(x2) for any
    f, g -- a genuine, irreducible interaction between the axes, exactly the
    situation [CANR Sec.1 H3] demonstrated concretely for (representation,
    operator): the best choice on one axis depends on what was chosen on the
    other.

    Basin A (the true global optimum, fixed height 10) sits at (5, 15).
    Basin B (a decoy, height controlled by lam) sits at (18, 3). At lam=0
    only A exists. As lam grows, B becomes competitive and can capture
    naive per-axis or short-sighted iterative search into a WORSE, MIXED
    point -- e.g. "x1 near A's optimum, x2 near B's optimum" -- which is not
    close to either basin and is not the true joint optimum.
    """
    sigma2 = 2 * 3.0 ** 2
    bump_a = 10.0 * pow(2.718281828459045, -((x1 - 5) ** 2 + (x2 - 15) ** 2) / sigma2)
    bump_b = lam * pow(2.718281828459045, -((x1 - 18) ** 2 + (x2 - 3) ** 2) / sigma2)
    return -(bump_a + bump_b)


def z2_full_joint(lam, n1, n2):
    best = None
    for x1 in range(n1):
        for x2 in range(n2):
            c = z2_objective(x1, x2, lam, n1, n2)
            if best is None or c < best[0]:
                best = (c, x1, x2)
    return best


def z2_independent(lam, n1, n2, default2, default1):
    # Axis 1 tuned alone, holding axis 2 at an arbitrary default (this is
    # exactly what calling scirust-core::transform_autotune::autotune_by once
    # per axis, in isolation, does: it cannot see the other axis's eventual
    # choice).
    best_x1 = min(range(n1), key=lambda x1: z2_objective(x1, default2, lam, n1, n2))
    best_x2 = min(range(n2), key=lambda x2: z2_objective(default1, x2, lam, n1, n2))
    c = z2_objective(best_x1, best_x2, lam, n1, n2)
    return c, best_x1, best_x2


def z2_coordinate_descent(lam, n1, n2, x1, x2, max_passes=10):
    for _ in range(max_passes):
        new_x1 = min(range(n1), key=lambda a: z2_objective(a, x2, lam, n1, n2))
        new_x2 = min(range(n2), key=lambda b: z2_objective(new_x1, b, lam, n1, n2))
        if new_x1 == x1 and new_x2 == x2:
            break
        x1, x2 = new_x1, new_x2
    c = z2_objective(x1, x2, lam, n1, n2)
    return c, x1, x2


def z2_interaction_experiment():
    print("=" * 78)
    print("Z2: independent vs. coordinate-descent vs. joint search under interaction")
    print("=" * 78)
    n1, n2 = 25, 25
    print(f"{'lambda':>7} {'independent':>14} {'coord-descent':>15} {'joint (optimal)':>17} "
          f"{'indep gap':>10} {'CD gap':>8}")
    results = []
    for lam in (0.0, 3.0, 6.0, 8.0, 10.0, 12.0, 15.0):
        c_joint, x1j, x2j = z2_full_joint(lam, n1, n2)
        c_indep, x1i, x2i = z2_independent(lam, n1, n2, default2=n2 // 2, default1=n1 // 2)
        c_cd, x1c, x2c = z2_coordinate_descent(lam, n1, n2, x1=n1 // 2, x2=n2 // 2)
        indep_gap = c_indep - c_joint
        cd_gap = c_cd - c_joint
        print(f"{lam:>7.1f} {c_indep:>14.3f} {c_cd:>15.3f} {c_joint:>17.3f} "
              f"{indep_gap:>10.3f} {cd_gap:>8.3f}")
        results.append((lam, c_indep, c_cd, c_joint, indep_gap, cd_gap))
    print()
    print("Reading (three regimes, all produced by this run, not hypothesized):")
    print("  lambda=0..3   objective is a single basin (effectively separable given the")
    print("                starting points used): independent, coordinate-descent, and")
    print("                joint search all tie exactly (0 gap).")
    print("  lambda=6..10  a second, still-shorter basin becomes strong enough to win")
    print("                ONE of the two isolated single-axis sub-problems: one-shot")
    print("                independent search combines a piece of each basin's answer")
    print("                into a point near NEITHER basin (gap ~10) -- exactly [CANR")
    print("                Sec.1 H3]'s finding that representation and operator must be")
    print("                selected as pairs, generalized to any two interacting axes.")
    print("                Iterative coordinate descent still recovers the true optimum")
    print("                here (0 gap): cheap local search can compensate for mild")
    print("                interaction.")
    print("  lambda=12..15 the second basin overtakes the first as the TRUE global")
    print("                optimum, but coordinate descent's trajectory from the same")
    print("                starting point is still captured by the (now suboptimal)")
    print("                first basin: it under-performs full joint search by a")
    print("                growing, non-vanishing margin. Only exhaustive joint search")
    print("                is correct in every regime tested -- and Z1 already showed")
    print("                exhaustive joint search over 7 real axes is combinatorially")
    print("                infeasible, which is exactly why the algorithm-configuration")
    print("                literature (SMAC, irace, successive halving, Bayesian")
    print("                optimization) exists instead of either extreme.")
    print()
    return results


# --------------------------------------------------------------------------
# Z3 -- exact multiplicative composition of kappa_rt (elasticity chain rule)
# --------------------------------------------------------------------------


def kappa_rt_power(lam):
    """kappa_rt(x) for phi(x) = x^lam, CANR Sec.4: identically 1/lam (exact,
    independent of x)."""
    return d(1) / d(lam)


def kappa_rt_log_at(y):
    """kappa_rt(y) for phi(y) = ln(y), derived in this report: |ln(y)|
    (CANR Sec.3.1 formula kappa_rt(x) = |phi(x) / (x * phi'(x))|, phi'=1/y so
    x*phi'(x) = 1, kappa_rt(y) = |ln(y)|)."""
    return abs(y.ln())


def composed_kappa_rt_power_then_log(x, lam1):
    """Exact kappa_rt of Phi(x) = ln(x^lam1) = lam1 * ln(x), via CANR's
    formula applied directly to the composed map (the ground truth this
    experiment checks the chain rule against)."""
    # Phi(x) = lam1 * ln(x); Phi'(x) = lam1 / x; x*Phi'(x) = lam1.
    Phi = lam1 * x.ln()
    return abs(Phi) / abs(lam1)


def z3_kappa_composition():
    print("=" * 78)
    print("Z3: kappa_rt composes EXACTLY multiplicatively (elasticity chain rule)")
    print("=" * 78)
    print(f"{'x':>12} {'lam1':>6} {'kappa(power)':>14} {'kappa(log@y)':>14} "
          f"{'product':>14} {'exact kappa(Phi)':>18} {'rel. diff':>12}")
    max_rel_diff = d(0)
    for x_val, lam1 in [
        ("2.5", "0.3"), ("100", "0.7"), ("1e-6", "1.5"), ("1e12", "0.1"),
        ("3.14159", "2.0"), ("1e-30", "0.05"), ("7.5", "0.9"),
    ]:
        x = d(x_val)
        lam1_d = d(lam1)
        y = x ** lam1_d  # y = x^lam1, the intermediate representation
        k_power = kappa_rt_power(lam1_d)
        k_log = kappa_rt_log_at(y)
        product = k_power * k_log
        exact = composed_kappa_rt_power_then_log(x, lam1_d)
        rel_diff = abs(product - exact) / abs(exact) if exact != 0 else abs(product - exact)
        max_rel_diff = max(max_rel_diff, rel_diff)
        print(f"{x_val:>12} {lam1:>6} {float(k_power):>14.6f} {float(k_log):>14.6f} "
              f"{float(product):>14.6f} {float(exact):>18.6f} {float(rel_diff):>12.3e}")
    print()
    print(f"Max relative difference across all points: {float(max_rel_diff):.3e}")
    print("(At Decimal prec=60, this is floating-point-of-the-proof-itself noise,")
    print("not approximation error -- confirming the composition is an EXACT")
    print("identity: kappa_rt(g o f)(x) = kappa_rt(g)(f(x)) * kappa_rt(f)(x),")
    print("because kappa_rt(x) = |phi(x)/(x phi'(x))| IS the logarithmic derivative")
    print("(elasticity) of phi, and elasticities compose exactly multiplicatively")
    print("under composition -- ordinary calculus, not new mathematics.)")
    print()


# --------------------------------------------------------------------------
# Z4 -- determinism composes as a meet (weakest link), not an average
# --------------------------------------------------------------------------


def exact_sum(xs):
    """Stand-in for a 'D1-proved, order-invariant' accumulator: exact
    rational-equivalent summation via Decimal at high precision is, for this
    experiment's finite float inputs, bitwise order-invariant."""
    s = d(0)
    for x in xs:
        s += d(x)
    return s


def naive_f64_sum(xs):
    """A ordinary order-DEPENDENT ('D0') running sum in native float64."""
    s = 0.0
    for x in xs:
        s += x
    return s


def z4_determinism_meet():
    print("=" * 78)
    print("Z4: determinism composes as a meet (weakest link), not an average")
    print("=" * 78)
    random.seed(20260717)
    # Stage 1 data: values chosen to be summed by a D1-style exact accumulator.
    stage1_data = [random.uniform(-1e6, 1e6) for _ in range(500)]
    s1 = exact_sum(stage1_data)  # "D1": identical regardless of stage1_data order
    # Confirm stage 1 truly is order-invariant across permutations (sanity check).
    perm_results_stage1 = set()
    for _ in range(20):
        shuffled = stage1_data[:]
        random.shuffle(shuffled)
        perm_results_stage1.add(exact_sum(shuffled))
    print(f"Stage 1 (D1-style exact accumulator): {len(perm_results_stage1)} distinct "
          f"result(s) across 20 random permutations of its own input (expect 1).")

    # Stage 2: pipe s1 (bitwise fixed, from above) together with a second
    # batch of data through an ORDER-DEPENDENT ('D0') naive float64 sum, and
    # vary only the order of stage 2's own inputs.
    stage2_extra = [random.uniform(-1.0, 1.0) for _ in range(200)]
    outputs = set()
    outputs_list = []
    for trial in range(20):
        batch = stage2_extra[:]
        random.shuffle(batch)
        pipeline_input = [float(s1)] + batch  # stage 1's fixed output feeds stage 2
        out = naive_f64_sum(pipeline_input)
        outputs.add(out)
        outputs_list.append(out)
    print(f"Stage 2 (D0 naive float64 sum, fed by stage 1's fixed output): "
          f"{len(outputs)} distinct result(s) across 20 permutations of stage 2's own input.")
    if len(outputs) > 1:
        spread = max(outputs) - min(outputs)
        print(f"Spread across permutations: {spread:.3e} (nonzero => NOT order-invariant)")
    print()
    print("Conclusion: stage 1's D1 guarantee (order-invariance of ITS OWN inputs)")
    print("does not propagate 'upgrade' stage 2's determinism level. The overall")
    print("pipeline's determinism level is the MEET (infimum) of its stages' levels")
    print("in the D0 < D1/D2 < ... partial order established in [CANR Sec.6.1] --")
    print("a chain is only as reproducible as its least-reproducible stage, exactly")
    print("as for any other monotone-decreasing composed guarantee (cf. the classical")
    print("weakest-link / meet-semilattice pattern; not a new observation once stated,")
    print("but not previously stated for CANR's determinism ladder specifically.)")
    print()


if __name__ == "__main__":
    z1_combinatorial_blowup()
    z2_interaction_experiment()
    z3_kappa_composition()
    z4_determinism_meet()

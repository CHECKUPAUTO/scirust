//! Radar binary (M-of-N) integration — the coincidence-detection decision rule.
//!
//! After matched filtering and single-pulse thresholding ([`super::cfar`],
//! [`super::swerling`]), each of `N` pulses on a target either exceeds the
//! threshold (a *hit*) or not. **Binary integration** — also called *M-of-N* or
//! *coincidence* detection — declares a target only when at least `M` of the `N`
//! pulses are hits. It is the cheapest non-coherent integrator (a counter and a
//! comparator, no summation of amplitudes) and, because the per-pulse hit is a
//! Bernoulli trial, the post-integration probabilities follow the **binomial**
//! distribution in closed form.
//!
//! Choosing `M > 1` slashes the false-alarm rate: an isolated noise spike rarely
//! repeats, so `P_fa` collapses from the single-pulse value to
//! `Σ_{k=M}^{N} C(N,k)·P_fa^k·(1−P_fa)^{N−k}` — e.g. a 3-of-5 rule at a
//! single-pulse `P_fa = 10⁻²` reaches ~`10⁻⁵`. A real target hits often enough
//! that a modest `M` keeps the probability of detection high, so the near-optimal
//! rule `M ≈ 1.5·√N` trades a small detection loss for a large false-alarm gain.
//! This module gives the binomial building blocks, the integrated `P_fa` / `P_d`,
//! and that optimal-`M` rule. Dependency-free.

/// The **binomial probability mass** `P(X = k)` for `X ~ Binomial(n, p)`:
/// `C(n,k)·p^k·(1−p)^{n−k}`, the probability of exactly `k` hits in `n`
/// independent pulses each with hit probability `p`. The coefficient `C(n,k)` is
/// formed by a divide-as-you-go integer/float product, so no factorial ever
/// overflows. Returns `0.0` when `k > n`; `p` is clamped to `[0, 1]`.
pub fn binomial_pmf(k: u32, n: u32, p: f64) -> f64 {
    if k > n
    {
        return 0.0;
    }
    let p = p.clamp(0.0, 1.0);
    let q = 1.0 - p;
    // C(n, k) via the symmetric multiplicative form, dividing as we multiply so
    // the running product stays near the final (small) coefficient — no overflow.
    let kk = k.min(n - k);
    let mut coeff = 1.0_f64;
    for i in 0..kk
    {
        coeff = coeff * (n - i) as f64 / (i + 1) as f64;
    }
    coeff * p.powi(k as i32) * q.powi((n - k) as i32)
}

/// The **binomial survival function** `P(X ≥ m)` for `X ~ Binomial(n, p)`:
/// `Σ_{k=m}^{n} C(n,k)·p^k·(1−p)^{n−k}`, the probability that at least `m` of `n`
/// pulses are hits — the M-of-N decision probability. Returns `0.0` when `m > n`
/// (the event is impossible) and `1.0` when `m = 0`. `p` is clamped to `[0, 1]`.
pub fn binomial_sf_ge(m: u32, n: u32, p: f64) -> f64 {
    if m > n
    {
        return 0.0;
    }
    let mut sum = 0.0;
    for k in m..=n
    {
        sum += binomial_pmf(k, n, p);
    }
    sum.clamp(0.0, 1.0)
}

/// The **integrated false-alarm probability** of an M-of-N rule:
/// `P_fa = P(X ≥ m)` for `X ~ Binomial(n, pfa_single)`. Requiring `m` coincident
/// threshold crossings out of `n` pulses drives the false-alarm rate far below the
/// single-pulse `pfa_single` (a 3-of-5 rule at `10⁻²` reaches ~`10⁻⁵`). Thin
/// wrapper over [`binomial_sf_ge`].
pub fn integrated_pfa(m: u32, n: u32, pfa_single: f64) -> f64 {
    binomial_sf_ge(m, n, pfa_single)
}

/// The **integrated probability of detection** of an M-of-N rule:
/// `P_d = P(X ≥ m)` for `X ~ Binomial(n, pd_single)`. A real target hits often
/// enough that a modest `m` keeps `P_d` high while [`integrated_pfa`] plunges.
/// Thin wrapper over [`binomial_sf_ge`].
pub fn integrated_pd(m: u32, n: u32, pd_single: f64) -> f64 {
    binomial_sf_ge(m, n, pd_single)
}

/// The classic near-optimal M-of-N threshold `M ≈ round(1.5·√N)`, clamped to
/// `1..=n`. This rule sits close to the `M` that minimises the required
/// single-pulse SNR across the useful `P_d`/`P_fa` range, balancing the
/// false-alarm reduction of a larger `M` against its detection loss. Returns `0`
/// for the degenerate `n = 0`.
pub fn optimal_m(n: u32) -> u32 {
    if n == 0
    {
        return 0;
    }
    let m = (1.5 * (n as f64).sqrt()).round() as u32;
    m.clamp(1, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pmf_is_a_normalised_distribution() {
        // Σ_{k=0}^{n} P(X = k) = 1 for any (n, p) — conservation of probability.
        for &(n, p) in &[(1_u32, 0.5_f64), (5, 0.01), (8, 0.7), (12, 0.999)]
        {
            let total: f64 = (0..=n).map(|k| binomial_pmf(k, n, p)).sum();
            assert!((total - 1.0).abs() < 1e-12, "n={n} p={p}: Σ={total}");
        }
    }

    #[test]
    fn pmf_matches_exact_closed_forms() {
        // Symmetric fair-coin case: P(X = k) = C(n,k)/2^n.
        assert!((binomial_pmf(2, 4, 0.5) - 6.0 / 16.0).abs() < 1e-15);
        assert!((binomial_pmf(0, 5, 0.5) - 1.0 / 32.0).abs() < 1e-15);
        // Endpoints reduce to pure powers: P(0) = (1−p)^n, P(n) = p^n.
        assert!((binomial_pmf(0, 5, 0.3) - 0.7_f64.powi(5)).abs() < 1e-15);
        assert!((binomial_pmf(5, 5, 0.3) - 0.3_f64.powi(5)).abs() < 1e-15);
        // A larger-N coefficient exercises the overflow-safe product: C(30,15).
        let expect = 155_117_520.0 / 2.0_f64.powi(30);
        assert!((binomial_pmf(15, 30, 0.5) - expect).abs() < 1e-12);
    }

    #[test]
    fn sf_ge_matches_the_at_least_one_closed_form() {
        // P(X ≥ 1) = 1 − (1−p)^n, the probability of at least one hit.
        for &(n, p) in &[(5_u32, 0.2_f64), (8, 0.05), (10, 0.6)]
        {
            let expect = 1.0 - (1.0 - p).powi(n as i32);
            assert!(
                (binomial_sf_ge(1, n, p) - expect).abs() < 1e-12,
                "n={n} p={p}"
            );
        }
        // P(X ≥ 0) = 1 (certain); complement identity P(X≥m) = 1 − P(X≤m−1).
        assert!((binomial_sf_ge(0, 7, 0.3) - 1.0).abs() < 1e-12);
        let below: f64 = (0..3).map(|k| binomial_pmf(k, 7, 0.3)).sum();
        assert!((binomial_sf_ge(3, 7, 0.3) - (1.0 - below)).abs() < 1e-12);
    }

    #[test]
    fn integration_slashes_the_false_alarm_rate() {
        // 3-of-5 at a single-pulse P_fa = 1e-2 lands near 1e-5 — the textbook
        // false-alarm reduction — matched here against the hand-summed binomial.
        let pfa_single = 1e-2_f64;
        let expect = 10.0 * pfa_single.powi(3) * (1.0 - pfa_single).powi(2)
            + 5.0 * pfa_single.powi(4) * (1.0 - pfa_single)
            + pfa_single.powi(5);
        let got = integrated_pfa(3, 5, pfa_single);
        assert!((got - expect).abs() < 1e-15, "got={got} expect={expect}");
        assert!(got < 1e-4 && got > 1e-6, "3-of-5 P_fa = {got}");
        // Orders of magnitude below the single-pulse rate.
        assert!(got < pfa_single / 1000.0);
    }

    #[test]
    fn detection_stays_high_for_small_m() {
        // A moderate single-pulse P_d with a small M keeps integrated P_d high.
        // 1-of-3 at pd_single = 0.7: P_d = 1 − 0.3^3 = 0.973.
        let pd = integrated_pd(1, 3, 0.7);
        assert!((pd - (1.0 - 0.3_f64.powi(3))).abs() < 1e-12);
        assert!(pd > 0.97);
        // 2-of-3 stays comfortably high as well.
        assert!(integrated_pd(2, 3, 0.7) > 0.78);
    }

    #[test]
    fn sf_ge_is_monotone_decreasing_in_m() {
        // Raising the coincidence count M can only lower the decision probability,
        // for both the target (P_d) and noise (P_fa) branches.
        let (n, pd, pfa) = (8_u32, 0.6_f64, 0.1);
        for m in 1..n
        {
            assert!(integrated_pd(m + 1, n, pd) <= integrated_pd(m, n, pd));
            assert!(integrated_pfa(m + 1, n, pfa) <= integrated_pfa(m, n, pfa));
        }
    }

    #[test]
    fn optimal_m_follows_the_root_n_rule() {
        // round(1.5·√N), clamped into 1..=N.
        assert_eq!(optimal_m(4), 3); // round(3.0)
        assert_eq!(optimal_m(16), 6); // round(6.0)
        assert_eq!(optimal_m(1), 1); // round(1.5)=2 clamped down to 1
        for n in 1..=64_u32
        {
            let m = optimal_m(n);
            assert!((1..=n).contains(&m), "n={n}: M={m} out of range");
            let target = 1.5 * (n as f64).sqrt();
            assert!((m as f64 - target).abs() <= 1.0, "n={n}: M={m} vs {target}");
        }
        assert_eq!(optimal_m(0), 0); // degenerate guard
    }

    #[test]
    fn degenerate_inputs_are_guarded() {
        // Impossible event: needing more hits than pulses.
        assert_eq!(binomial_pmf(6, 5, 0.5), 0.0);
        assert_eq!(binomial_sf_ge(6, 5, 0.5), 0.0);
        // Probabilities outside [0, 1] are clamped, never NaN.
        assert_eq!(binomial_pmf(0, 4, -1.0), 1.0); // clamps to p = 0
        assert_eq!(binomial_pmf(4, 4, 2.0), 1.0); // clamps to p = 1
        assert!(binomial_sf_ge(2, 5, 3.0).is_finite());
        // Certain-hit / never-hit endpoints.
        assert_eq!(integrated_pd(5, 5, 1.0), 1.0);
        assert_eq!(integrated_pfa(1, 5, 0.0), 0.0);
    }
}

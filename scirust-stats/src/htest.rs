//! Classical hypothesis tests. Each returns a [`TestResult`] with the test
//! statistic, its reference degrees of freedom, and a p-value computed from the
//! corresponding distribution in [`crate::dist`] — so every p-value traces back
//! to the audited `scirust-special` numeric base.

use crate::describe::{mean, variance};
use crate::discrete::DiscreteDistribution;
use crate::dist::{ChiSquared, Distribution, FisherF, StudentT};

/// Outcome of a hypothesis test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TestResult {
    /// The test statistic (t, F, or χ²).
    pub statistic: f64,
    /// Reference degrees of freedom (for F, this is the numerator dof).
    pub df: f64,
    /// The p-value.
    pub p_value: f64,
}

/// Which tail(s) the alternative hypothesis occupies for a t-test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tail {
    /// H₁: mean ≠ reference (default).
    TwoSided,
    /// H₁: mean > reference.
    Greater,
    /// H₁: mean < reference.
    Less,
}

fn t_p_value(t: f64, df: f64, tail: Tail) -> f64 {
    let dist = StudentT::new(df);
    match tail
    {
        Tail::TwoSided => 2.0 * dist.sf(t.abs()),
        Tail::Greater => dist.sf(t),
        Tail::Less => dist.cdf(t),
    }
}

/// One-sample t-test of `H₀: mean(data) = mu0`.
///
/// Returns `None` if fewer than two samples (variance undefined).
pub fn t_test_one_sample(data: &[f64], mu0: f64, tail: Tail) -> Option<TestResult> {
    let n = data.len();
    if n < 2
    {
        return None;
    }
    let m = mean(data);
    let s = variance(data).sqrt();
    let df = n as f64 - 1.0;
    let t = (m - mu0) / (s / (n as f64).sqrt());
    Some(TestResult {
        statistic: t,
        df,
        p_value: t_p_value(t, df, tail),
    })
}

/// Two-sample t-test of `H₀: mean(a) = mean(b)`.
///
/// `equal_var = true` uses the pooled-variance Student test; `false` uses
/// Welch's unequal-variance test (the safer default in practice). Returns `None`
/// if either group has fewer than two samples.
pub fn t_test_two_sample(a: &[f64], b: &[f64], equal_var: bool, tail: Tail) -> Option<TestResult> {
    let (n1, n2) = (a.len(), b.len());
    if n1 < 2 || n2 < 2
    {
        return None;
    }
    let (m1, m2) = (mean(a), mean(b));
    let (v1, v2) = (variance(a), variance(b));
    let (n1f, n2f) = (n1 as f64, n2 as f64);

    let (t, df) = if equal_var
    {
        let sp2 = ((n1f - 1.0) * v1 + (n2f - 1.0) * v2) / (n1f + n2f - 2.0);
        let se = (sp2 * (1.0 / n1f + 1.0 / n2f)).sqrt();
        ((m1 - m2) / se, n1f + n2f - 2.0)
    }
    else
    {
        let se = (v1 / n1f + v2 / n2f).sqrt();
        // Welch–Satterthwaite degrees of freedom.
        let num = (v1 / n1f + v2 / n2f).powi(2);
        let den = (v1 / n1f).powi(2) / (n1f - 1.0) + (v2 / n2f).powi(2) / (n2f - 1.0);
        ((m1 - m2) / se, num / den)
    };
    Some(TestResult {
        statistic: t,
        df,
        p_value: t_p_value(t, df, tail),
    })
}

/// One-way ANOVA of `H₀: all group means are equal`.
///
/// `groups` is a slice of samples per group. Returns `None` if fewer than two
/// groups or fewer than one degree of freedom within groups.
pub fn one_way_anova(groups: &[&[f64]]) -> Option<TestResult> {
    // Empty samples contain no observations and therefore are not factor
    // levels. Counting them in k corrupts both numerator and denominator dof.
    let groups: Vec<&[f64]> = groups.iter().copied().filter(|g| !g.is_empty()).collect();
    let k = groups.len();
    if k < 2
    {
        return None;
    }
    let n_total: usize = groups.iter().map(|g| g.len()).sum();
    if n_total <= k
    {
        return None; // no within-group degrees of freedom
    }
    let grand = mean(
        &groups
            .iter()
            .flat_map(|g| g.iter().copied())
            .collect::<Vec<_>>(),
    );
    let mut ss_between = 0.0;
    let mut ss_within = 0.0;
    for g in groups
    {
        let gm = mean(g);
        ss_between += g.len() as f64 * (gm - grand).powi(2);
        for &x in g.iter()
        {
            ss_within += (x - gm).powi(2);
        }
    }
    let df1 = (k - 1) as f64;
    let df2 = (n_total - k) as f64;
    let ms_between = ss_between / df1;
    let ms_within = ss_within / df2;
    let f = ms_between / ms_within;
    let p = FisherF::new(df1, df2).sf(f);
    Some(TestResult {
        statistic: f,
        df: df1,
        p_value: p,
    })
}

/// Pearson's χ² goodness-of-fit test.
///
/// `observed` and `expected` counts must be the same non-empty length with all
/// `expected > 0`. `ddof` is the number of parameters estimated from the data
/// (subtracted from `k − 1` degrees of freedom; pass 0 for a fully-specified
/// model). Returns `None` on a shape/positivity violation.
pub fn chi_square_gof(observed: &[f64], expected: &[f64], ddof: usize) -> Option<TestResult> {
    if observed.len() != expected.len() || observed.is_empty()
    {
        return None;
    }
    if expected.iter().any(|&e| e <= 0.0)
    {
        return None;
    }
    let chi2: f64 = observed
        .iter()
        .zip(expected.iter())
        .map(|(&o, &e)| (o - e) * (o - e) / e)
        .sum();
    let df = observed.len() as f64 - 1.0 - ddof as f64;
    if df <= 0.0
    {
        return None;
    }
    let p = ChiSquared::new(df).sf(chi2);
    Some(TestResult {
        statistic: chi2,
        df,
        p_value: p,
    })
}

/// Pearson's χ² goodness-of-fit test for a **discrete distribution**.
///
/// `observed[i]` is the count of observations equal to the value `i`, except
/// the final entry `observed[L−1]`, which counts the whole upper tail
/// (value `≥ L−1`). Expected counts are taken from `dist` — `N·pmf(i)` for the
/// exact bins and `N·sf(L−2)` for the tail bin — so they sum to `N` exactly.
///
/// Adjacent bins are pooled until every pooled expected count reaches
/// `min_expected` (Cochran's rule of thumb is `5`), which also absorbs the
/// zero-probability leading bins of supports that start at 1 (e.g.
/// [`crate::discrete::Geometric`]). `ddof` is the number of parameters
/// estimated from the data — pass `1` when the distribution was fitted with
/// `fit_mom` on one parameter, `0` when fully specified — and is subtracted
/// from `(pooled bins − 1)` degrees of freedom.
///
/// Returns `None` if there is no data, fewer than two bins survive pooling, or
/// the residual degrees of freedom are non-positive.
pub fn chi2_gof_discrete<D: DiscreteDistribution>(
    observed: &[u64],
    dist: &D,
    ddof: usize,
    min_expected: f64,
) -> Option<TestResult> {
    let l = observed.len();
    let n: f64 = observed.iter().map(|&c| c as f64).sum();
    if l < 2 || n <= 0.0
    {
        return None;
    }
    // Expected counts: exact bins 0..L−1, tail bin = N·P(X ≥ L−1).
    let mut expected = vec![0.0_f64; l];
    for (i, e) in expected.iter_mut().enumerate().take(l - 1)
    {
        *e = n * dist.pmf(i as u64);
    }
    expected[l - 1] = n * dist.sf((l - 2) as u64);

    // Pool adjacent bins so each pooled expected ≥ min_expected (> 0 floor so
    // zero-probability bins always merge forward).
    let floor = min_expected.max(1e-12);
    let (mut po, mut pe) = (Vec::new(), Vec::new());
    let (mut co, mut ce) = (0.0_f64, 0.0_f64);
    for i in 0..l
    {
        co += observed[i] as f64;
        ce += expected[i];
        if ce >= floor
        {
            po.push(co);
            pe.push(ce);
            co = 0.0;
            ce = 0.0;
        }
    }
    // Fold any small trailing remainder into the last pooled bin.
    if ce > 0.0 || co > 0.0
    {
        if let (Some(lo), Some(le)) = (po.last_mut(), pe.last_mut())
        {
            *lo += co;
            *le += ce;
        }
        else
        {
            po.push(co);
            pe.push(ce);
        }
    }
    if po.len() < 2
    {
        return None;
    }
    chi_square_gof(&po, &pe, ddof)
}

/// One-sample Kolmogorov–Smirnov test that `data` is drawn from `dist`.
///
/// Returns the D statistic and the asymptotic p-value (Kolmogorov distribution).
/// Returns `None` for an empty sample.
pub fn ks_test_one_sample<D: Distribution>(data: &[f64], dist: &D) -> Option<TestResult> {
    let n = data.len();
    if n == 0
    {
        return None;
    }
    let mut v: Vec<f64> = data.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let nf = n as f64;
    let mut d = 0.0_f64;
    for (i, &x) in v.iter().enumerate()
    {
        let f = dist.cdf(x);
        let d_plus = (i as f64 + 1.0) / nf - f;
        let d_minus = f - i as f64 / nf;
        d = d.max(d_plus).max(d_minus);
    }
    // Asymptotic p-value with the Stephens small-sample correction.
    let en = nf.sqrt();
    let lambda = (en + 0.12 + 0.11 / en) * d;
    Some(TestResult {
        statistic: d,
        df: nf,
        p_value: kolmogorov_q(lambda),
    })
}

/// The complementary Kolmogorov distribution
/// `Q(λ) = 2 Σ_{j≥1} (−1)^{j−1} e^{−2 j² λ²}`, clamped to `[0, 1]`.
fn kolmogorov_q(lambda: f64) -> f64 {
    if lambda.is_nan()
    {
        return f64::NAN;
    }
    if lambda <= 0.0
    {
        return 1.0;
    }
    if lambda.is_infinite()
    {
        return 0.0;
    }

    // The alternating expansion below is catastrophically slow and suffers
    // cancellation near zero. Jacobi's theta transformation gives the CDF
    // there as a rapidly convergent sum of strictly positive terms; Q = 1-CDF.
    if lambda < 1.18
    {
        let scale = std::f64::consts::PI.sqrt() * 2.0_f64.sqrt() / lambda;
        let exponent_scale = -std::f64::consts::PI.powi(2) / (8.0 * lambda * lambda);
        let mut cdf_sum = 0.0;
        for j in 1..=100
        {
            let odd = (2 * j - 1) as f64;
            let term = (exponent_scale * odd * odd).exp();
            cdf_sum += term;
            if term <= f64::EPSILON * cdf_sum.max(1.0)
            {
                break;
            }
        }
        return (1.0 - scale * cdf_sum).clamp(0.0, 1.0);
    }

    let mut sum = 0.0;
    let a = -2.0 * lambda * lambda;
    for j in 1..=10_000
    {
        let jf = j as f64;
        let term = (a * jf * jf).exp();
        sum += if j % 2 == 1 { term } else { -term };
        if term <= f64::EPSILON * sum.abs().max(1.0)
        {
            break;
        }
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::Normal;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn one_sample_t_matches_reference() {
        // Data with mean 5, tested against μ0 = 4.
        let d = [5.1, 4.9, 5.3, 4.7, 5.0, 5.2, 4.8, 5.0];
        let r = t_test_one_sample(&d, 4.0, Tail::TwoSided).unwrap();
        assert!(r.statistic > 0.0);
        assert!(r.df == 7.0);
        // Strongly significant (mean clearly ≠ 4); the exact value is ≈ 2.1e-6.
        assert!(r.p_value < 1e-5, "p = {}", r.p_value);
        // A one-sample test of the data against its own mean gives t ≈ 0, p ≈ 1.
        let r0 = t_test_one_sample(&d, mean(&d), Tail::TwoSided).unwrap();
        assert!(close(r0.statistic, 0.0, 1e-9));
        assert!(close(r0.p_value, 1.0, 1e-9));
    }

    #[test]
    fn two_sample_welch_and_pooled() {
        let a = [23.0, 21.0, 25.0, 22.0, 24.0, 20.0];
        let b = [27.0, 29.0, 26.0, 28.0, 30.0, 25.0];
        let welch = t_test_two_sample(&a, &b, false, Tail::TwoSided).unwrap();
        let pooled = t_test_two_sample(&a, &b, true, Tail::TwoSided).unwrap();
        // Both should find a highly significant difference.
        assert!(welch.p_value < 0.001);
        assert!(pooled.p_value < 0.001);
        // Welch dof ≤ pooled dof (n1+n2−2 = 10).
        assert!(welch.df <= 10.0 + 1e-9);
        // Identical groups → t = 0, p = 1.
        let same = t_test_two_sample(&a, &a, true, Tail::TwoSided).unwrap();
        assert!(close(same.statistic, 0.0, 1e-9));
        assert!(close(same.p_value, 1.0, 1e-9));
    }

    #[test]
    fn anova_detects_and_rejects() {
        // Three clearly-separated groups → tiny p.
        let g1 = [1.0, 2.0, 1.5, 1.8, 2.2];
        let g2 = [5.0, 6.0, 5.5, 5.8, 6.2];
        let g3 = [9.0, 10.0, 9.5, 9.8, 10.2];
        let r = one_way_anova(&[&g1, &g2, &g3]).unwrap();
        assert!(r.p_value < 1e-6, "p = {}", r.p_value);
        assert!(r.df == 2.0);
        // Three copies of the same group → F ≈ 0, p ≈ 1.
        let r2 = one_way_anova(&[&g1, &g1, &g1]).unwrap();
        assert!(r2.statistic.abs() < 1e-9);
        assert!(close(r2.p_value, 1.0, 1e-9));
    }

    #[test]
    fn anova_ignores_empty_groups_when_computing_degrees_of_freedom() {
        let g1 = [1.0, 2.0];
        let g2 = [3.0, 4.0];
        let empty = [];
        let result = one_way_anova(&[&g1, &g2, &empty]).unwrap();
        assert!(close(result.statistic, 8.0, 1e-12));
        assert_eq!(result.df, 1.0);
    }

    #[test]
    fn chi_square_gof_fair_die() {
        // A near-fair die: expect ~1/6 each of 60 rolls.
        let observed = [9.0, 11.0, 10.0, 8.0, 12.0, 10.0];
        let expected = [10.0; 6];
        let r = chi_square_gof(&observed, &expected, 0).unwrap();
        assert!(r.df == 5.0);
        assert!(r.p_value > 0.5); // consistent with fair
        // A loaded die → significant.
        let loaded = [30.0, 6.0, 6.0, 6.0, 6.0, 6.0];
        let r2 = chi_square_gof(&loaded, &expected, 0).unwrap();
        assert!(r2.p_value < 1e-6);
    }

    #[test]
    fn ks_accepts_matching_and_rejects_mismatched() {
        // Deterministic sample from N(0,1) should NOT be rejected against N(0,1).
        let mut rng = crate::rng::SplitMix64::new(2024);
        let n = Normal::standard();
        let sample: Vec<f64> = (0..2000).map(|_| n.sample(&mut rng)).collect();
        let r = ks_test_one_sample(&sample, &n).unwrap();
        assert!(r.p_value > 0.05, "same dist p = {}", r.p_value);
        // The same sample vs a shifted normal IS rejected.
        let shifted = Normal::new(1.0, 1.0);
        let r2 = ks_test_one_sample(&sample, &shifted).unwrap();
        assert!(r2.p_value < 1e-6, "shifted p = {}", r2.p_value);
    }

    #[test]
    fn kolmogorov_q_is_stable_for_small_lambda() {
        let q = kolmogorov_q(0.001);
        assert_eq!(q, 1.0);
        // Reference values from the Kolmogorov limiting distribution.
        assert!(close(kolmogorov_q(0.5), 0.963_945_243_664_875, 1e-13));
        assert!(close(kolmogorov_q(1.0), 0.269_999_671_677_354_56, 1e-13));
    }

    #[test]
    fn discrete_gof_poisson_matches_scipy() {
        use crate::discrete::Poisson;
        // Binned counts 0,1,2,3,4,≥5 (N = 100); a Poisson(1.98) fit.
        let observed = [10u64, 28, 32, 18, 8, 4];
        let dist = Poisson::new(1.98);
        // One parameter estimated ⇒ ddof = 1; every expected ≥ 5 ⇒ no pooling.
        let r = chi2_gof_discrete(&observed, &dist, 1, 5.0).unwrap();
        assert!(
            (r.statistic - 2.279_187_103).abs() < 1e-6,
            "chi2 = {}",
            r.statistic
        );
        assert!((r.df - 4.0).abs() < 1e-12);
        assert!(
            (r.p_value - 0.684_560_899).abs() < 1e-6,
            "p = {}",
            r.p_value
        );
    }

    #[test]
    fn discrete_gof_rejects_bad_fit_and_pools() {
        use crate::discrete::{Geometric, Poisson};
        // Data clearly not Poisson(1.0): far too much mass in the tail.
        let observed = [5u64, 5, 5, 5, 80];
        let bad = Poisson::new(1.0);
        let r = chi2_gof_discrete(&observed, &bad, 1, 5.0).unwrap();
        assert!(r.p_value < 1e-6, "bad fit not rejected, p = {}", r.p_value);
        // Geometric support starts at 1: bin 0 has zero probability and must
        // pool forward rather than break the test (expected > 0 everywhere).
        let obs_g = [0u64, 50, 25, 13, 12];
        let g = Geometric::new(0.5);
        let rg = chi2_gof_discrete(&obs_g, &g, 1, 5.0);
        assert!(rg.is_some());
        // Degenerate: a single bin cannot yield a test.
        assert!(chi2_gof_discrete(&[10], &bad, 0, 5.0).is_none());
        assert!(chi2_gof_discrete(&[0, 0, 0], &bad, 0, 5.0).is_none());
    }
}

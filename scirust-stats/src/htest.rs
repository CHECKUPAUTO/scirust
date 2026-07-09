//! Classical hypothesis tests. Each returns a [`TestResult`] with the test
//! statistic, its reference degrees of freedom, and a p-value computed from the
//! corresponding distribution in [`crate::dist`] — so every p-value traces back
//! to the audited `scirust-special` numeric base.

use crate::describe::{mean, variance};
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
        if g.is_empty()
        {
            continue;
        }
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
    if lambda <= 0.0
    {
        return 1.0;
    }
    let mut sum = 0.0;
    let a = -2.0 * lambda * lambda;
    let mut term_prev = 0.0;
    for j in 1..=100
    {
        let jf = j as f64;
        let term = (a * jf * jf).exp();
        sum += if j % 2 == 1 { term } else { -term };
        // Converged when two successive terms are negligible.
        if term < 1e-12 && term_prev < 1e-12
        {
            break;
        }
        term_prev = term;
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
}

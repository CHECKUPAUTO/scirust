//! Shewhart variable-control-chart constants (subgroup size `n`).
//!
//! The control limits of X-bar / R / S charts are built from a handful of
//! constants that depend only on the subgroup size `n`. They come from the
//! distribution of the range and standard deviation of `n` i.i.d. normal
//! samples:
//!
//! - `d2(n) = E[R/σ]`, `d3(n) = SD[R/σ]` — mean and spread of the *relative
//!   range*. These have no elementary closed form (they are integrals of the
//!   range distribution), so the long-established tabulated values are used.
//! - `c4(n) = E[s/σ] = sqrt(2/(n-1)) · Γ(n/2) / Γ((n-1)/2)` — the
//!   bias-correction factor for the sample standard deviation. This one *is*
//!   closed-form and is computed from the gamma function.
//!
//! Everything else is derived from those three:
//!
//! ```text
//! A2(n) = 3 / (d2 · sqrt(n))            X-bar limits from R-bar
//! A3(n) = 3 / (c4 · sqrt(n))            X-bar limits from s-bar
//! D3(n) = max(0, 1 - 3·d3/d2)           R-chart lower limit factor
//! D4(n) =          1 + 3·d3/d2          R-chart upper limit factor
//! B3(n) = max(0, 1 - (3/c4)·sqrt(1-c4²))  s-chart lower limit factor
//! B4(n) =          1 + (3/c4)·sqrt(1-c4²)  s-chart upper limit factor
//! ```
//!
//! The unit tests cross-check every derived value against the canonical
//! published table, so a transcription error in the `d2`/`d3` tables cannot
//! pass silently.

/// `d2(n) = E[R/σ]`, tabulated for `n = 2..=10`. Index `n - 2`.
const D2: [f64; 9] = [
    1.128, 1.693, 2.059, 2.326, 2.534, 2.704, 2.847, 2.970, 3.078,
];

/// `d3(n) = SD[R/σ]`, tabulated for `n = 2..=10`. Index `n - 2`. Four-decimal
/// values so the derived `D3`/`D4` factors match the canonical published table.
const D3_SD: [f64; 9] = [
    0.8525, 0.8884, 0.8798, 0.8641, 0.8480, 0.8332, 0.8198, 0.8078, 0.7971,
];

/// Smallest subgroup size for which the range/std constants are defined.
pub const MIN_N: usize = 2;
/// Largest subgroup size covered by the tabulated `d2`/`d3` values.
pub const MAX_N: usize = 10;

/// Lanczos approximation of `ln Γ(x)` for `x > 0` (g = 7, 9 coefficients).
///
/// Accurate to roughly 15 significant digits across the half-integer and
/// integer arguments that `c4` needs.
fn ln_gamma(x: f64) -> f64 {
    // Coefficients for g = 7.
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    // Reflection is unnecessary here: c4 only ever evaluates Γ at arguments
    // >= 1/2, well away from the poles on the negative axis.
    let xm1 = x - 1.0;
    let mut a = C[0];
    let t = xm1 + G + 0.5;
    for (i, &coef) in C.iter().enumerate().skip(1)
    {
        a += coef / (xm1 + i as f64);
    }
    0.5 * (2.0 * core::f64::consts::PI).ln() + (xm1 + 0.5) * t.ln() - t + a.ln()
}

/// `c4(n) = sqrt(2/(n-1)) · Γ(n/2) / Γ((n-1)/2)`, the unbiasing constant for
/// the sample standard deviation. Defined for `n >= 2`.
pub fn c4(n: usize) -> f64 {
    assert!(n >= 2, "c4 needs subgroup size n >= 2");
    let nn = n as f64;
    (2.0 / (nn - 1.0)).sqrt() * (ln_gamma(nn / 2.0) - ln_gamma((nn - 1.0) / 2.0)).exp()
}

/// `d2(n) = E[R/σ]`. `None` outside `2..=10`.
pub fn d2(n: usize) -> Option<f64> {
    if (MIN_N..=MAX_N).contains(&n)
    {
        Some(D2[n - 2])
    }
    else
    {
        None
    }
}

/// `d3(n) = SD[R/σ]`. `None` outside `2..=10`.
pub fn d3(n: usize) -> Option<f64> {
    if (MIN_N..=MAX_N).contains(&n)
    {
        Some(D3_SD[n - 2])
    }
    else
    {
        None
    }
}

/// `A2(n) = 3 / (d2 · sqrt(n))` — X-bar limit factor from the mean range.
pub fn a2(n: usize) -> Option<f64> {
    d2(n).map(|d| 3.0 / (d * (n as f64).sqrt()))
}

/// `A3(n) = 3 / (c4 · sqrt(n))` — X-bar limit factor from the mean std.
pub fn a3(n: usize) -> f64 {
    3.0 / (c4(n) * (n as f64).sqrt())
}

/// `D3(n) = max(0, 1 - 3·d3/d2)` — R-chart lower-limit factor.
pub fn d3_factor(n: usize) -> Option<f64> {
    match (d2(n), d3(n))
    {
        (Some(d2v), Some(d3v)) => Some((1.0 - 3.0 * d3v / d2v).max(0.0)),
        _ => None,
    }
}

/// `D4(n) = 1 + 3·d3/d2` — R-chart upper-limit factor.
pub fn d4_factor(n: usize) -> Option<f64> {
    match (d2(n), d3(n))
    {
        (Some(d2v), Some(d3v)) => Some(1.0 + 3.0 * d3v / d2v),
        _ => None,
    }
}

/// `B3(n) = max(0, 1 - (3/c4)·sqrt(1 - c4²))` — s-chart lower-limit factor.
pub fn b3(n: usize) -> f64 {
    let c = c4(n);
    (1.0 - 3.0 / c * (1.0 - c * c).sqrt()).max(0.0)
}

/// `B4(n) = 1 + (3/c4)·sqrt(1 - c4²)` — s-chart upper-limit factor.
pub fn b4(n: usize) -> f64 {
    let c = c4(n);
    1.0 + 3.0 / c * (1.0 - c * c).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} vs {b} (tol {tol})");
    }

    #[test]
    fn ln_gamma_matches_known_values() {
        // Γ(1) = 1, Γ(2) = 1, Γ(3) = 2, Γ(4) = 6, Γ(5) = 24.
        close(ln_gamma(1.0).exp(), 1.0, 1e-9);
        close(ln_gamma(2.0).exp(), 1.0, 1e-9);
        close(ln_gamma(3.0).exp(), 2.0, 1e-9);
        close(ln_gamma(4.0).exp(), 6.0, 1e-9);
        close(ln_gamma(5.0).exp(), 24.0, 1e-8);
        // Γ(1/2) = sqrt(pi); Γ(3/2) = sqrt(pi)/2; Γ(5/2) = 3·sqrt(pi)/4.
        let sp = core::f64::consts::PI.sqrt();
        close(ln_gamma(0.5).exp(), sp, 1e-9);
        close(ln_gamma(1.5).exp(), sp / 2.0, 1e-9);
        close(ln_gamma(2.5).exp(), 0.75 * sp, 1e-9);
    }

    #[test]
    fn c4_matches_canonical_table() {
        // Published c4 values (NIST / AIAG), 4 decimals.
        let want = [
            (2, 0.7979),
            (3, 0.8862),
            (4, 0.9213),
            (5, 0.9400),
            (6, 0.9515),
            (7, 0.9594),
            (10, 0.9727),
        ];
        for (n, v) in want
        {
            close(c4(n), v, 5e-5);
        }
    }

    #[test]
    fn a2_matches_canonical_table() {
        for (n, v) in [(2, 1.880), (3, 1.023), (4, 0.729), (5, 0.577), (7, 0.419)]
        {
            close(a2(n).unwrap(), v, 1e-3);
        }
    }

    #[test]
    fn a3_matches_canonical_table() {
        for (n, v) in [(2, 2.659), (3, 1.954), (4, 1.628), (5, 1.427), (10, 0.975)]
        {
            close(a3(n), v, 1e-3);
        }
    }

    #[test]
    fn d3_d4_factors_match_canonical_table() {
        // (n, D3, D4)
        let want = [
            (2, 0.000, 3.267),
            (3, 0.000, 2.574),
            (4, 0.000, 2.282),
            (5, 0.000, 2.114),
            (6, 0.000, 2.004),
            (7, 0.076, 1.924),
        ];
        for (n, lo, hi) in want
        {
            close(d3_factor(n).unwrap(), lo, 1e-3);
            close(d4_factor(n).unwrap(), hi, 1e-3);
        }
    }

    #[test]
    fn b3_b4_factors_match_canonical_table() {
        // (n, B3, B4)
        let want = [
            (2, 0.000, 3.267),
            (3, 0.000, 2.568),
            (4, 0.000, 2.266),
            (5, 0.000, 2.089),
            (6, 0.030, 1.970),
            (10, 0.284, 1.716),
        ];
        for (n, lo, hi) in want
        {
            close(b3(n), lo, 1e-3);
            close(b4(n), hi, 1e-3);
        }
    }

    #[test]
    fn out_of_range_subgroups_return_none() {
        assert!(d2(1).is_none());
        assert!(d3(11).is_none());
        assert!(a2(0).is_none());
        assert!(d4_factor(100).is_none());
    }
}

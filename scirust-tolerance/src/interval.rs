//! Statistical tolerance intervals (normal theory, ISO 16269-6).
//!
//! A capability index answers "how far is the spread from the limits"; a
//! **tolerance interval** answers the complementary question a customer asks:
//! *from this sample of `n` parts, give me limits that contain at least a
//! proportion `p` of the whole population, with confidence `1−α`.* The interval
//! is `x̄ ± k·s`, and the whole subtlety is the factor `k`, which must inflate
//! the sample spread to account for the finite sample.
//!
//! - [`tolerance_factor_two_sided`] — the two-sided factor by the Howe (1969)
//!   approximation `k = z_{(1+p)/2}·√(ν(1+1/n)/χ²_{ν,α})`.
//! - [`tolerance_factor_one_sided`] — the one-sided factor by the Natrella
//!   approximation (an accurate closed form for the non-central-`t` exact).
//! - [`tolerance_interval`] / [`ToleranceInterval::covers_spec`] — the interval
//!   itself and whether it fits inside a `[LSL, USL]` specification (a
//!   sample-based, confidence-bounded conformance statement, unlike a bare
//!   `x̄ ± 3s`).
//!
//! Both factors tend to the normal quantile (`z_{(1+p)/2}`, `z_p`) as `n → ∞`,
//! and are validated in `fuzz_crosscheck` against the Monte-Carlo coverage
//! probability that *defines* them.

use crate::special::{chi2_quantile, inv_normal_cdf};
use serde::{Deserialize, Serialize};

fn valid(n: usize, p: f64, conf: f64) -> bool {
    n >= 2 && p > 0.0 && p < 1.0 && conf > 0.0 && conf < 1.0
}

/// Two-sided normal tolerance factor `k` such that `x̄ ± k·s` contains at least
/// proportion `p` of the population with confidence `conf`, by the Howe (1969)
/// approximation. `None` for `n < 2` or `p`/`conf` outside `(0, 1)`.
pub fn tolerance_factor_two_sided(n: usize, p: f64, conf: f64) -> Option<f64> {
    if !valid(n, p, conf)
    {
        return None;
    }
    let nu = (n - 1) as f64;
    let z = inv_normal_cdf(0.5 * (1.0 + p));
    // Lower α = 1 − conf quantile of χ²_ν.
    let chi = chi2_quantile(nu, 1.0 - conf);
    if chi <= 0.0
    {
        return None;
    }
    Some(z * (nu * (1.0 + 1.0 / n as f64) / chi).sqrt())
}

/// One-sided normal tolerance factor `k` such that `x̄ + k·s` (upper) or
/// `x̄ − k·s` (lower) bounds proportion `p` of the population with confidence
/// `conf`, by the Natrella closed-form approximation of the exact non-central-`t`
/// factor. `None` for invalid inputs.
pub fn tolerance_factor_one_sided(n: usize, p: f64, conf: f64) -> Option<f64> {
    if !valid(n, p, conf)
    {
        return None;
    }
    let zp = inv_normal_cdf(p);
    let zg = inv_normal_cdf(conf);
    let a = 1.0 - zg * zg / (2.0 * (n - 1) as f64);
    let b = zp * zp - zg * zg / n as f64;
    if a <= 0.0
    {
        return None;
    }
    let disc = zp * zp - a * b;
    if disc < 0.0
    {
        return None;
    }
    Some((zp + disc.sqrt()) / a)
}

/// A two-sided statistical tolerance interval `[lower, upper] = x̄ ± k·s`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ToleranceInterval {
    /// Lower tolerance limit `x̄ − k·s`.
    pub lower: f64,
    /// Upper tolerance limit `x̄ + k·s`.
    pub upper: f64,
    /// The tolerance factor `k` used.
    pub k: f64,
}

impl ToleranceInterval {
    /// Whether the interval lies inside a `[lsl, usl]` specification — a
    /// confidence-bounded statement that at least `p` of the population conforms.
    pub fn covers_spec(&self, lsl: f64, usl: f64) -> bool {
        self.lower >= lsl && self.upper <= usl
    }

    /// Width `upper − lower`.
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }
}

/// Two-sided tolerance interval `x̄ ± k·s` from a sample's `mean` and standard
/// deviation `sd` (unbiased, `n − 1`), sample size `n`, coverage `p` and
/// confidence `conf`. `None` for invalid inputs.
pub fn tolerance_interval(
    mean: f64,
    sd: f64,
    n: usize,
    p: f64,
    conf: f64,
) -> Option<ToleranceInterval> {
    let k = tolerance_factor_two_sided(n, p, conf)?;
    Some(ToleranceInterval {
        lower: mean - k * sd,
        upper: mean + k * sd,
        k,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn factors_exceed_the_normal_quantile_and_shrink_with_n() {
        // k > z always (finite-sample inflation), and decreases toward z as n
        // grows — but only slowly (∝ 1 − 1.645·√(2/ν)), so it converges to z
        // only at very large n.
        let z2 = inv_normal_cdf(0.5 * (1.0 + 0.95));
        let k_small = tolerance_factor_two_sided(5, 0.95, 0.95).unwrap();
        let k_big = tolerance_factor_two_sided(500, 0.95, 0.95).unwrap();
        assert!(k_small > k_big);
        assert!(k_big > z2);
        let k_huge = tolerance_factor_two_sided(100_000, 0.95, 0.95).unwrap();
        assert_relative_eq!(k_huge, z2, epsilon = 0.02);
    }

    #[test]
    fn one_sided_factor_tends_to_zp() {
        let zp = inv_normal_cdf(0.90);
        let k = tolerance_factor_one_sided(100_000, 0.90, 0.95).unwrap();
        assert_relative_eq!(k, zp, epsilon = 0.02);
        // One-sided is smaller than two-sided at the same coverage.
        assert!(
            tolerance_factor_one_sided(20, 0.95, 0.95).unwrap()
                < tolerance_factor_two_sided(20, 0.95, 0.95).unwrap()
        );
    }

    #[test]
    fn interval_and_spec_coverage() {
        let ti = tolerance_interval(10.0, 0.5, 30, 0.99, 0.95).unwrap();
        assert!(ti.upper > ti.lower);
        assert_relative_eq!(ti.lower, 10.0 - ti.k * 0.5, epsilon = 1e-12);
        assert!(ti.covers_spec(5.0, 15.0));
        assert!(!ti.covers_spec(9.9, 10.1));
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert!(tolerance_factor_two_sided(1, 0.9, 0.9).is_none());
        assert!(tolerance_factor_two_sided(10, 1.0, 0.9).is_none());
        assert!(tolerance_factor_one_sided(10, 0.9, 0.0).is_none());
    }
}

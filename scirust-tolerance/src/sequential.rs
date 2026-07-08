//! Multi-stage acceptance sampling — double sampling and Wald's sequential test.
//!
//! [`crate::attributes`] sentences a lot from a *single* sample. Two refinements
//! reach the same protection with a smaller **average** sample by letting an
//! early clear-cut result stop inspection:
//!
//! - **Double sampling** — a first sample of `n1` accepts on `d1 ≤ c1`, rejects
//!   on `d1 ≥ r1`, and only for the ambiguous middle draws a second sample `n2`,
//!   accepting on `d1 + d2 ≤ c2`. A very good or very bad lot is settled on the
//!   first sample.
//! - **Sequential** (Wald's SPRT) — inspect one item at a time and, after each,
//!   accept / reject / continue against two straight boundary lines; the test
//!   ends as soon as the running defect count crosses one. Its average sample
//!   number is the smallest of the three schemes for the same risks.
//!
//! ## Operating characteristic
//!
//! For a large lot the defect counts are binomial, so the double-sampling
//! probability of acceptance at fraction defective `p` is
//!
//! ```text
//! Pa(p) = P(d1 ≤ c1) + Σ_{k=c1+1}^{r1−1} P(d1 = k)·P(d2 ≤ c2 − k) ,
//! ```
//!
//! and the average sample number `ASN(p) = n1 + n2·P(c1 < d1 < r1)`.
//!
//! ## Sequential boundaries
//!
//! With producer/consumer qualities `p0 < p1` and risks `α, β`, the accept and
//! reject lines are `d = s·n ∓ h`, where
//!
//! ```text
//! k  = ln[ p1(1−p0) / (p0(1−p1)) ] ,     s = ln[(1−p0)/(1−p1)] / k ,
//! h₁ = ln[(1−α)/β] / k ,                 h₂ = ln[(1−β)/α] / k .
//! ```
//!
//! Accept when the cumulative defects `d ≤ s·n − h₁`, reject when `d ≥ s·n + h₂`,
//! else continue — Wald's sequential probability ratio test.

use serde::{Deserialize, Serialize};

/// Binomial pmf `C(n,k) pᵏ (1−p)ⁿ⁻ᵏ`, evaluated by a stable ratio recurrence.
fn binomial_pmf(n: usize, k: usize, p: f64) -> f64 {
    if k > n
    {
        return 0.0;
    }
    if p <= 0.0
    {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    if p >= 1.0
    {
        return if k == n { 1.0 } else { 0.0 };
    }
    let mut term = (1.0 - p).powi(n as i32);
    let ratio = p / (1.0 - p);
    for i in 1..=k
    {
        term *= (n - i + 1) as f64 / i as f64 * ratio;
    }
    term
}

/// Binomial cdf `P(D ≤ c)`.
fn binomial_cdf(n: usize, c: usize, p: f64) -> f64 {
    let c = c.min(n);
    if p <= 0.0
    {
        return 1.0;
    }
    if p >= 1.0
    {
        return if c >= n { 1.0 } else { 0.0 };
    }
    let mut term = (1.0 - p).powi(n as i32);
    let mut cdf = term;
    let ratio = p / (1.0 - p);
    for d in 1..=c
    {
        term *= (n - d + 1) as f64 / d as f64 * ratio;
        cdf += term;
    }
    cdf.min(1.0)
}

/// A double-sampling attributes plan: first sample `n1` with acceptance `c1` and
/// rejection `r1` (`c1 < r1`); if the first count falls between, a second sample
/// `n2` accepts on a combined count `≤ c2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoubleSamplingPlan {
    /// First sample size `n1`.
    pub n1: usize,
    /// First acceptance number `c1` (accept if `d1 ≤ c1`).
    pub c1: usize,
    /// First rejection number `r1` (reject if `d1 ≥ r1`); requires `r1 > c1 + 1`
    /// to have an undecided region.
    pub r1: usize,
    /// Second sample size `n2`.
    pub n2: usize,
    /// Combined acceptance number `c2` (accept if `d1 + d2 ≤ c2`).
    pub c2: usize,
}

impl DoubleSamplingPlan {
    /// Build a plan from explicit parameters.
    pub fn new(n1: usize, c1: usize, r1: usize, n2: usize, c2: usize) -> Self {
        DoubleSamplingPlan { n1, c1, r1, n2, c2 }
    }

    /// Probability of accepting a lot of fraction defective `p`.
    pub fn probability_of_acceptance(&self, p: f64) -> f64 {
        // Accept outright on the first sample.
        let mut pa = binomial_cdf(self.n1, self.c1, p);
        // Undecided first counts k in (c1, r1): need d1+d2 ≤ c2.
        let lo = self.c1 + 1;
        let hi = self.r1.saturating_sub(1);
        for k in lo..=hi
        {
            if k > self.c2
            {
                continue; // already over the combined limit even with d2 = 0
            }
            let p1 = binomial_pmf(self.n1, k, p);
            pa += p1 * binomial_cdf(self.n2, self.c2 - k, p);
        }
        pa.min(1.0)
    }

    /// Average sample number `ASN(p) = n1 + n2·P(c1 < d1 < r1)` — the expected
    /// items inspected, which drops toward `n1` for very good or very bad lots.
    pub fn average_sample_number(&self, p: f64) -> f64 {
        let p_second = (binomial_cdf(self.n1, self.r1.saturating_sub(1), p)
            - binomial_cdf(self.n1, self.c1, p))
        .clamp(0.0, 1.0);
        self.n1 as f64 + self.n2 as f64 * p_second
    }
}

/// A sequential (SPRT) plan: the two straight boundary lines `d = slope·n ∓ h`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SequentialPlan {
    /// Common slope `s` of both boundary lines.
    pub slope: f64,
    /// Accept-line intercept magnitude `h₁` (accept when `d ≤ slope·n − h₁`).
    pub accept_intercept: f64,
    /// Reject-line intercept `h₂` (reject when `d ≥ slope·n + h₂`).
    pub reject_intercept: f64,
}

/// Verdict of a sequential test after `n` items with `d` defects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SequentialVerdict {
    /// Accept the lot.
    Accept,
    /// Reject the lot.
    Reject,
    /// Inspect another item.
    Continue,
}

impl SequentialPlan {
    /// Acceptance boundary `slope·n − h₁`: accept when defects are at or below it.
    pub fn accept_boundary(&self, n: usize) -> f64 {
        self.slope * n as f64 - self.accept_intercept
    }

    /// Rejection boundary `slope·n + h₂`: reject when defects are at or above it.
    pub fn reject_boundary(&self, n: usize) -> f64 {
        self.slope * n as f64 + self.reject_intercept
    }

    /// Verdict after inspecting `n` items and finding `d` defects.
    pub fn verdict(&self, n: usize, d: usize) -> SequentialVerdict {
        let d = d as f64;
        if d >= self.reject_boundary(n)
        {
            SequentialVerdict::Reject
        }
        else if d <= self.accept_boundary(n)
        {
            SequentialVerdict::Accept
        }
        else
        {
            SequentialVerdict::Continue
        }
    }
}

/// Design a sequential (SPRT) plan from the producer quality `aql` (with risk
/// `alpha`) and consumer quality `rql` (with risk `beta`), `0 < aql < rql < 1`.
/// Returns `None` on out-of-range inputs.
pub fn design_sequential_plan(aql: f64, rql: f64, alpha: f64, beta: f64) -> Option<SequentialPlan> {
    if !(aql > 0.0 && aql < rql && rql < 1.0)
        || !(0.0..1.0).contains(&alpha)
        || !(0.0..1.0).contains(&beta)
        || alpha <= 0.0
        || beta <= 0.0
    {
        return None;
    }
    let k = (rql * (1.0 - aql) / (aql * (1.0 - rql))).ln();
    if k <= 0.0
    {
        return None;
    }
    let slope = ((1.0 - aql) / (1.0 - rql)).ln() / k;
    let accept_intercept = ((1.0 - alpha) / beta).ln() / k;
    let reject_intercept = ((1.0 - beta) / alpha).ln() / k;
    Some(SequentialPlan {
        slope,
        accept_intercept,
        reject_intercept,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn double_plan_oc_endpoints() {
        let plan = DoubleSamplingPlan::new(50, 1, 4, 50, 5);
        // Perfect lot always accepted; all-defective always rejected.
        assert_relative_eq!(plan.probability_of_acceptance(0.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(plan.probability_of_acceptance(1.0), 0.0, epsilon = 1e-12);
        // OC decreases in p.
        assert!(plan.probability_of_acceptance(0.02) > plan.probability_of_acceptance(0.10));
    }

    #[test]
    fn double_plan_asn_peaks_in_the_middle() {
        let plan = DoubleSamplingPlan::new(50, 1, 4, 50, 5);
        // A decisive lot rarely needs the second sample ⇒ ASN near n1.
        let good = plan.average_sample_number(0.001);
        let mid = plan.average_sample_number(0.05);
        assert!(good < 51.0);
        // The ambiguous middle needs the second sample more often ⇒ larger ASN.
        assert!(mid > good);
        assert!(mid <= 100.0);
    }

    #[test]
    fn double_reduces_to_single_when_no_middle() {
        // r1 = c1 + 1 ⇒ no undecided region ⇒ pure single sampling on (n1, c1).
        let plan = DoubleSamplingPlan::new(80, 2, 3, 80, 9);
        for &p in &[0.01, 0.03, 0.08]
        {
            let single = binomial_cdf(80, 2, p);
            assert_relative_eq!(plan.probability_of_acceptance(p), single, epsilon = 1e-12);
            // Second sample never taken ⇒ ASN = n1.
            assert_relative_eq!(plan.average_sample_number(p), 80.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn sequential_boundaries_are_ordered_and_parallel() {
        let plan = design_sequential_plan(0.01, 0.08, 0.05, 0.10).unwrap();
        // Reject line sits above the accept line at every n (gap = h1 + h2).
        for n in [1usize, 10, 50, 200]
        {
            assert!(plan.reject_boundary(n) > plan.accept_boundary(n));
        }
        let gap0 = plan.reject_boundary(0) - plan.accept_boundary(0);
        let gap1 = plan.reject_boundary(100) - plan.accept_boundary(100);
        assert_relative_eq!(gap0, gap1, epsilon = 1e-9); // parallel
        assert!(plan.slope > 0.0 && plan.slope < 1.0);
    }

    #[test]
    fn sequential_verdict_transitions() {
        let plan = design_sequential_plan(0.01, 0.08, 0.05, 0.10).unwrap();
        // Early on with no defects we cannot yet accept (need n past h1/slope).
        assert_eq!(plan.verdict(1, 0), SequentialVerdict::Continue);
        // Enough clean items ⇒ accept.
        let n_accept = (plan.accept_intercept / plan.slope).ceil() as usize + 1;
        assert_eq!(plan.verdict(n_accept, 0), SequentialVerdict::Accept);
        // A burst of defects early ⇒ reject.
        assert_eq!(plan.verdict(5, 10), SequentialVerdict::Reject);
    }

    #[test]
    fn rejects_bad_sequential_inputs() {
        assert!(design_sequential_plan(0.08, 0.01, 0.05, 0.10).is_none()); // aql≥rql
        assert!(design_sequential_plan(0.01, 0.08, 0.0, 0.10).is_none());
    }
}

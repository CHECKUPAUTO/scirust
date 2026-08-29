//! Foundational non-parametric survival analysis for right-censored data.
//!
//! This module implements the product-limit estimator of Kaplan & Meier
//! (1958), the cumulative-hazard estimator of Nelson (1972), and the
//! two-sample log-rank statistic described by Mantel (1966).  It intentionally
//! stops at these foundations: regression models such as Cox proportional
//! hazards require additional modelling and validation abstractions.
//!
//! Ties follow the usual right-censoring convention: subjects censored at a
//! time are still in the risk set for events occurring at that same time.

use crate::dist::{ChiSquared, Distribution};
use core::fmt;

/// A single non-negative finite time-to-event observation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RightCensoredObservation {
    time: f64,
    event: bool,
}

impl RightCensoredObservation {
    /// Construct an observation. `event == true` denotes an observed event;
    /// `false` denotes right censoring.
    pub fn new(time: f64, event: bool) -> Result<Self, SurvivalError> {
        if !time.is_finite() || time < 0.0 {
            return Err(SurvivalError::InvalidTime(time));
        }
        Ok(Self { time, event })
    }

    /// Observation time.
    #[must_use]
    pub fn time(self) -> f64 {
        self.time
    }

    /// Whether the event was observed (`false` means right-censored).
    #[must_use]
    pub fn event_observed(self) -> bool {
        self.event
    }
}

/// Error returned by survival-analysis routines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SurvivalError {
    /// A time is negative, NaN, or infinite.
    InvalidTime(f64),
    /// At least one observation is required.
    EmptySample,
    /// The two-sample log-rank test requires both groups to be non-empty.
    EmptyGroup,
    /// The log-rank variance is zero, so no finite test statistic is defined.
    ZeroVariance,
}

impl fmt::Display for SurvivalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTime(t) => write!(f, "survival time must be finite and >= 0, got {t}"),
            Self::EmptySample => write!(f, "survival sample must not be empty"),
            Self::EmptyGroup => write!(f, "log-rank test requires two non-empty groups"),
            Self::ZeroVariance => write!(f, "log-rank statistic has zero variance"),
        }
    }
}

impl std::error::Error for SurvivalError {}

/// One row of a Kaplan-Meier product-limit estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KaplanMeierPoint {
    /// Unique observation time represented by this row.
    pub time: f64,
    /// Number at risk immediately before this time.
    pub at_risk: usize,
    /// Observed events at this time.
    pub events: usize,
    /// Right-censored observations at this time.
    pub censored: usize,
    /// Product-limit survival estimate after events at this time.
    pub survival: f64,
}

/// One row of a Nelson-Aalen cumulative-hazard estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NelsonAalenPoint {
    /// Unique observation time represented by this row.
    pub time: f64,
    /// Number at risk immediately before this time.
    pub at_risk: usize,
    /// Observed events at this time.
    pub events: usize,
    /// Right-censored observations at this time.
    pub censored: usize,
    /// Cumulative hazard after events at this time.
    pub cumulative_hazard: f64,
}

/// Result of a two-sample Mantel log-rank test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogRankResult {
    /// Total observed events in group A over pooled event times.
    pub observed_a: f64,
    /// Expected events in group A under equal hazards.
    pub expected_a: f64,
    /// Hypergeometric variance of `observed_a - expected_a`.
    pub variance: f64,
    /// One-degree-of-freedom chi-square statistic.
    pub chi_square: f64,
    /// Upper-tail probability from a chi-square distribution with one degree
    /// of freedom.
    pub p_value: f64,
}

fn sorted_observations(
    observations: &[RightCensoredObservation],
) -> Result<Vec<RightCensoredObservation>, SurvivalError> {
    if observations.is_empty() {
        return Err(SurvivalError::EmptySample);
    }
    for obs in observations {
        if !obs.time.is_finite() || obs.time < 0.0 {
            return Err(SurvivalError::InvalidTime(obs.time));
        }
    }
    let mut sorted = observations.to_vec();
    sorted.sort_by(|a, b| a.time.total_cmp(&b.time));
    Ok(sorted)
}

/// Compute the Kaplan-Meier product-limit estimator.
///
/// At each unique time `t_i`, with `n_i` subjects at risk and `d_i` observed
/// events, the survival estimate is updated as
/// `S(t_i) = S(t_i-) * (1 - d_i / n_i)`. Censoring at `t_i` is applied only
/// after that event update.
pub fn kaplan_meier(
    observations: &[RightCensoredObservation],
) -> Result<Vec<KaplanMeierPoint>, SurvivalError> {
    let sorted = sorted_observations(observations)?;
    let mut at_risk = sorted.len();
    let mut survival = 1.0;
    let mut out = Vec::new();
    let mut i = 0;

    while i < sorted.len() {
        let time = sorted[i].time;
        let mut events = 0usize;
        let mut censored = 0usize;
        let mut j = i;
        while j < sorted.len() && sorted[j].time == time {
            if sorted[j].event {
                events += 1;
            } else {
                censored += 1;
            }
            j += 1;
        }

        if events > 0 {
            survival *= 1.0 - events as f64 / at_risk as f64;
        }
        out.push(KaplanMeierPoint {
            time,
            at_risk,
            events,
            censored,
            survival,
        });
        at_risk -= events + censored;
        i = j;
    }
    Ok(out)
}

/// Compute the Nelson-Aalen cumulative-hazard estimator.
///
/// At each unique time, the hazard increment is `d_i / n_i`, where `d_i`
/// is the number of events and `n_i` the number at risk immediately before
/// that time.
pub fn nelson_aalen(
    observations: &[RightCensoredObservation],
) -> Result<Vec<NelsonAalenPoint>, SurvivalError> {
    let sorted = sorted_observations(observations)?;
    let mut at_risk = sorted.len();
    let mut cumulative_hazard = 0.0;
    let mut out = Vec::new();
    let mut i = 0;

    while i < sorted.len() {
        let time = sorted[i].time;
        let mut events = 0usize;
        let mut censored = 0usize;
        let mut j = i;
        while j < sorted.len() && sorted[j].time == time {
            if sorted[j].event {
                events += 1;
            } else {
                censored += 1;
            }
            j += 1;
        }

        if events > 0 {
            cumulative_hazard += events as f64 / at_risk as f64;
        }
        out.push(NelsonAalenPoint {
            time,
            at_risk,
            events,
            censored,
            cumulative_hazard,
        });
        at_risk -= events + censored;
        i = j;
    }
    Ok(out)
}

/// Compare two right-censored samples with the unweighted two-sample log-rank
/// test.
///
/// For each pooled event time, the expected number of group-A events is
/// `d_i * n_ai / n_i`. The variance uses the finite-population
/// hypergeometric correction
/// `n_ai*n_bi*d_i*(n_i-d_i) / (n_i^2*(n_i-1))`.
pub fn log_rank(
    group_a: &[RightCensoredObservation],
    group_b: &[RightCensoredObservation],
) -> Result<LogRankResult, SurvivalError> {
    if group_a.is_empty() || group_b.is_empty() {
        return Err(SurvivalError::EmptyGroup);
    }
    for obs in group_a.iter().chain(group_b) {
        if !obs.time.is_finite() || obs.time < 0.0 {
            return Err(SurvivalError::InvalidTime(obs.time));
        }
    }

    let mut event_times: Vec<f64> = group_a
        .iter()
        .chain(group_b)
        .filter(|obs| obs.event)
        .map(|obs| obs.time)
        .collect();
    event_times.sort_by(f64::total_cmp);
    event_times.dedup();

    let mut observed_a = 0.0;
    let mut expected_a = 0.0;
    let mut variance = 0.0;

    for time in event_times {
        let n_a = group_a.iter().filter(|obs| obs.time >= time).count();
        let n_b = group_b.iter().filter(|obs| obs.time >= time).count();
        let d_a = group_a
            .iter()
            .filter(|obs| obs.time == time && obs.event)
            .count();
        let d_b = group_b
            .iter()
            .filter(|obs| obs.time == time && obs.event)
            .count();
        let n = n_a + n_b;
        let d = d_a + d_b;
        if n == 0 || d == 0 {
            continue;
        }

        observed_a += d_a as f64;
        expected_a += d as f64 * n_a as f64 / n as f64;
        if n > 1 {
            variance += (n_a * n_b * d * (n - d)) as f64
                / ((n * n * (n - 1)) as f64);
        }
    }

    if variance <= 0.0 {
        return Err(SurvivalError::ZeroVariance);
    }
    let delta = observed_a - expected_a;
    let chi_square = delta * delta / variance;
    let p_value = ChiSquared::new(1.0).sf(chi_square);
    Ok(LogRankResult {
        observed_a,
        expected_a,
        variance,
        chi_square,
        p_value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(time: f64) -> RightCensoredObservation {
        RightCensoredObservation::new(time, true).unwrap()
    }

    fn c(time: f64) -> RightCensoredObservation {
        RightCensoredObservation::new(time, false).unwrap()
    }

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} != {b} within {tol}");
    }

    #[test]
    fn rejects_invalid_times_and_empty_samples() {
        assert!(matches!(
            RightCensoredObservation::new(-1.0, true),
            Err(SurvivalError::InvalidTime(_))
        ));
        assert!(matches!(
            RightCensoredObservation::new(f64::NAN, false),
            Err(SurvivalError::InvalidTime(_))
        ));
        assert_eq!(kaplan_meier(&[]), Err(SurvivalError::EmptySample));
        assert_eq!(nelson_aalen(&[]), Err(SurvivalError::EmptySample));
    }

    #[test]
    fn kaplan_meier_matches_hand_product_limit_with_ties() {
        // Risk sets and event/censor counts are:
        // t=1: n=5,d=1,c=0 => S=4/5
        // t=2: n=4,d=1,c=1 => S=(4/5)(3/4)=3/5
        // t=3: n=2,d=1,c=0 => S=(3/5)(1/2)=3/10
        // t=4: n=1,d=0,c=1 => S remains 3/10.
        let sample = [e(1.0), e(2.0), c(2.0), e(3.0), c(4.0)];
        let km = kaplan_meier(&sample).unwrap();
        assert_eq!(km.len(), 4);
        assert_eq!((km[1].at_risk, km[1].events, km[1].censored), (4, 1, 1));
        close(km[0].survival, 0.8, 1e-15);
        close(km[1].survival, 0.6, 1e-15);
        close(km[2].survival, 0.3, 1e-15);
        close(km[3].survival, 0.3, 1e-15);
    }

    #[test]
    fn nelson_aalen_matches_hand_cumulative_hazard() {
        let sample = [e(1.0), e(2.0), c(2.0), e(3.0), c(4.0)];
        let na = nelson_aalen(&sample).unwrap();
        // H = 1/5 + 1/4 + 1/2 = 0.95.
        close(na[0].cumulative_hazard, 0.2, 1e-15);
        close(na[1].cumulative_hazard, 0.45, 1e-15);
        close(na[2].cumulative_hazard, 0.95, 1e-15);
        close(na[3].cumulative_hazard, 0.95, 1e-15);
    }

    #[test]
    fn censor_at_event_time_remains_in_risk_set() {
        let sample = [e(1.0), c(1.0), e(2.0)];
        let km = kaplan_meier(&sample).unwrap();
        close(km[0].survival, 2.0 / 3.0, 1e-15);
        assert_eq!((km[0].at_risk, km[0].events, km[0].censored), (3, 1, 1));
    }

    #[test]
    fn all_censored_curve_stays_at_one_and_hazard_zero() {
        let sample = [c(1.0), c(2.0), c(3.0)];
        let km = kaplan_meier(&sample).unwrap();
        let na = nelson_aalen(&sample).unwrap();
        assert!(km.iter().all(|p| p.survival == 1.0));
        assert!(na.iter().all(|p| p.cumulative_hazard == 0.0));
    }

    #[test]
    fn log_rank_matches_hand_hypergeometric_calculation() {
        let a = [e(1.0), e(2.0), e(3.0)];
        let b = [e(4.0), e(5.0), e(6.0)];
        let r = log_rank(&a, &b).unwrap();
        // First three event times contribute E_A = 1/2 + 2/5 + 1/4 = 1.15
        // and V = 1/4 + 6/25 + 3/16 = 0.6775. Later times have n_A=0.
        close(r.observed_a, 3.0, 1e-15);
        close(r.expected_a, 1.15, 1e-15);
        close(r.variance, 0.6775, 1e-15);
        close(r.chi_square, 5.051_660_516_605_166, 1e-12);
        assert!(r.p_value > 0.02 && r.p_value < 0.03);
    }

    #[test]
    fn log_rank_reports_degenerate_comparison() {
        let a = [c(1.0), c(2.0)];
        let b = [c(1.5), c(2.5)];
        assert_eq!(log_rank(&a, &b), Err(SurvivalError::ZeroVariance));
        assert_eq!(log_rank(&[], &b), Err(SurvivalError::EmptyGroup));
    }
}

//! [`Estimate`] — an EIG estimate that carries its own uncertainty — and
//! [`Cost`], the experiment cost model.

use serde::{Deserialize, Serialize};
use sos_core::DeterminismLevel;
use sos_core::canonical::{Canonical, CanonicalEncoder};

/// Millibits per bit — the fixed-point unit for expected information gain. EIG is
/// carried in **millibits** (`1 bit == 1000`) so planning is integer-exact and
/// portable (the kernel encoder is float-free; SDE §05).
pub const MILLIBITS_PER_BIT: i64 = 1000;

/// An expected-information-gain estimate that **carries its own uncertainty**
/// (SDE §05.4). EIG is a nested expectation — expensive and biased if estimated
/// naively — so the planner never treats it as exact: an estimate is a *point*
/// value, a *standard error*, and the [`DeterminismLevel`] it was computed at
/// (`L3` closed-form, `L2`/`L1` Monte-Carlo). All in millibits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Estimate {
    /// The EIG point estimate, in millibits.
    pub bits_milli: i64,
    /// The estimate's standard error, in millibits (non-negative).
    pub se_milli: i64,
    /// The determinism level the estimator realized.
    pub level: DeterminismLevel,
}

impl Estimate {
    /// An estimate of `bits_milli` millibits with standard error `se_milli`
    /// (clamped non-negative) at determinism `level`.
    #[must_use]
    pub fn new(bits_milli: i64, se_milli: i64, level: DeterminismLevel) -> Self {
        Self {
            bits_milli,
            se_milli: se_milli.max(0),
            level,
        }
    }

    /// An exact, zero-error estimate (e.g. a closed-form GP EIG), `L3`.
    #[must_use]
    pub fn exact(bits_milli: i64) -> Self {
        Self::new(bits_milli, 0, DeterminismLevel::L3)
    }

    /// The conservative lower bound on the EIG: `point − standard error`. A
    /// planner that wants to avoid over-claiming ranks on this.
    #[must_use]
    pub fn lower_bound(&self) -> i64 {
        self.bits_milli.saturating_sub(self.se_milli)
    }

    /// Whether the estimate is **significantly informative** — its point value
    /// exceeds its own noise (`bits_milli > se_milli`). A `0.02 ± 0.03` bit
    /// estimate is *not* significant.
    #[must_use]
    pub fn is_significant(&self) -> bool {
        self.bits_milli > self.se_milli
    }
}

impl Canonical for Estimate {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.i64(self.bits_milli);
        enc.i64(self.se_milli);
        enc.value(&self.level);
    }
}

/// The cost of running an experiment (SDE §05.3): compute, wall-time, samples,
/// and risk, in caller-defined non-negative units. Kept structured so a policy
/// can weight the components; [`Cost::total`] is the default scalarization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Cost {
    /// Compute cost (e.g. core-seconds).
    pub compute: i64,
    /// Wall-time cost.
    pub time: i64,
    /// Sample / material cost.
    pub samples: i64,
    /// Risk cost (a caller-scaled penalty for hazardous or irreversible steps).
    pub risk: i64,
}

impl Cost {
    /// Construct a cost from its components (each clamped non-negative).
    #[must_use]
    pub fn new(compute: i64, time: i64, samples: i64, risk: i64) -> Self {
        Self {
            compute: compute.max(0),
            time: time.max(0),
            samples: samples.max(0),
            risk: risk.max(0),
        }
    }

    /// The scalar total cost — the saturating sum of the components.
    #[must_use]
    pub fn total(&self) -> i64 {
        self.compute
            .saturating_add(self.time)
            .saturating_add(self.samples)
            .saturating_add(self.risk)
    }

    /// Whether the experiment is free (zero total cost).
    #[must_use]
    pub fn is_free(&self) -> bool {
        self.total() == 0
    }
}

impl Canonical for Cost {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.i64(self.compute);
        enc.i64(self.time);
        enc.i64(self.samples);
        enc.i64(self.risk);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn significance_and_lower_bound() {
        // 0.9 ± 0.1 bits is significant; 0.02 ± 0.03 is not.
        let good = Estimate::new(900, 100, DeterminismLevel::L2);
        assert!(good.is_significant());
        assert_eq!(good.lower_bound(), 800);

        let noise = Estimate::new(20, 30, DeterminismLevel::L1);
        assert!(!noise.is_significant());
        assert_eq!(noise.lower_bound(), -10); // 20 − 30, honestly below zero
    }

    #[test]
    fn cost_total_and_free() {
        assert_eq!(Cost::new(1, 2, 3, 4).total(), 10);
        assert!(Cost::default().is_free());
        // Negatives are clamped away.
        assert_eq!(Cost::new(-5, 0, 0, 0).total(), 0);
    }
}

//! [`UtilityPolicy`] — the explicit, versioned "value of information" policy.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};

use crate::estimate::{Cost, Estimate};

/// Fixed-point scale for utility, so `EIG / cost` stays integer-exact.
pub const UTILITY_SCALE: i64 = 1_000;

/// A utility that marks a candidate as **excluded** (it violates a hard
/// constraint, e.g. an over-budget design). Ranks strictly below any real
/// utility.
pub const EXCLUDED: i64 = i64::MIN;

/// How the planner turns an [`Estimate`] and a [`Cost`] into the utility it
/// maximizes (SDE §05.3). "Most informative" is domain-relative, so utility is a
/// **pluggable, versioned policy** — never a hardcoded ratio (Invariant VI).
///
/// The two policies here cover the resource-bounded and fixed-budget cases; the
/// heavier value-of-information policies named in the RFC (`knowledge_gradient`,
/// `min_max_regret`) need the decision/​belief model and are deferred to their
/// backend per Invariant VIII.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum UtilityPolicy {
    /// `U = EIG / cost` — bits per unit cost. The default for resource-bounded
    /// labs. A free experiment is costed as one unit so utility stays finite.
    EigPerCost,
    /// `U = EIG` subject to `cost ≤ budget` — a fixed per-experiment budget. Any
    /// design over budget is [`EXCLUDED`].
    EigBudgeted {
        /// The per-experiment cost ceiling.
        budget: i64,
    },
}

impl UtilityPolicy {
    /// A short, stable code for display and canonical hashing.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self
        {
            Self::EigPerCost => "eig-per-cost",
            Self::EigBudgeted { .. } => "eig-budgeted",
        }
    }

    /// The fixed-point utility of a candidate under this policy. [`EXCLUDED`]
    /// ([`i64::MIN`]) means the candidate violates a hard constraint. Uses
    /// saturating arithmetic, so no weight or estimate can overflow-panic.
    #[must_use]
    pub fn utility(self, eig: &Estimate, cost: &Cost) -> i64 {
        match self
        {
            Self::EigPerCost =>
            {
                // Free experiments are costed as one unit so utility is finite.
                let denom = cost.total().max(1);
                eig.bits_milli.saturating_mul(UTILITY_SCALE) / denom
            },
            Self::EigBudgeted { budget } =>
            {
                if cost.total() > budget
                {
                    EXCLUDED
                }
                else
                {
                    eig.bits_milli
                }
            },
        }
    }
}

impl Canonical for UtilityPolicy {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(self.code());
        if let Self::EigBudgeted { budget } = self
        {
            enc.i64(*budget);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sos_core::DeterminismLevel;

    fn eig(bits_milli: i64) -> Estimate {
        Estimate::new(bits_milli, 0, DeterminismLevel::L2)
    }

    #[test]
    fn eig_per_cost_is_bits_per_unit_cost() {
        // 0.9 bits at cost 2 ⇒ U = 900 * 1000 / 2 = 450_000.
        assert_eq!(
            UtilityPolicy::EigPerCost.utility(&eig(900), &Cost::new(2, 0, 0, 0)),
            450_000
        );
        // A free experiment is costed as one unit (finite utility).
        assert_eq!(
            UtilityPolicy::EigPerCost.utility(&eig(50), &Cost::default()),
            50_000
        );
    }

    #[test]
    fn eig_budgeted_excludes_over_budget_designs() {
        let p = UtilityPolicy::EigBudgeted { budget: 5 };
        assert_eq!(p.utility(&eig(900), &Cost::new(3, 0, 0, 0)), 900); // affordable ⇒ pure EIG
        assert_eq!(p.utility(&eig(900), &Cost::new(6, 0, 0, 0)), EXCLUDED); // over budget
    }

    #[test]
    fn canonical_distinguishes_policies_and_budgets() {
        assert_ne!(
            UtilityPolicy::EigPerCost.canonical_bytes(),
            UtilityPolicy::EigBudgeted { budget: 5 }.canonical_bytes()
        );
        assert_ne!(
            UtilityPolicy::EigBudgeted { budget: 5 }.canonical_bytes(),
            UtilityPolicy::EigBudgeted { budget: 6 }.canonical_bytes()
        );
    }
}

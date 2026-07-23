//! Composable [`StoppingRule`]s and the [`StopSignals`] they evaluate against.

use serde::{Deserialize, Serialize};

/// The signals a [`StoppingRule`] reads — the current state of a study's
/// belief, information frontier, and budget. All fractional quantities are
/// fixed-point (`1.0 == 1000`), so evaluation is deterministic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StopSignals {
    /// The largest posterior mass on any single hypothesis, in milli (`1.0 == 1000`).
    pub max_posterior_mass_milli: i64,
    /// The best available candidate's EIG, in millibits.
    pub best_eig_milli: i64,
    /// Budget spent so far (experiments / compute / cost, caller-defined units).
    pub budget_spent: i64,
    /// The budget cap in the same units.
    pub budget_cap: i64,
}

/// An explicit, composable stopping rule for the discovery loop (SDE §04.4). The
/// interesting one is [`EigFloor`](StoppingRule::EigFloor) — the study stops not
/// when it is bored but when *no available experiment can teach it more than
/// `ε`*, and it can say so with a number.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StoppingRule {
    /// Stop once one hypothesis has captured at least `threshold_milli` posterior
    /// mass (`1.0 == 1000`).
    PosteriorMass {
        /// The mass threshold, in milli.
        threshold_milli: i64,
    },
    /// Stop when the best available EIG falls **below** `epsilon_milli` —
    /// information is exhausted.
    EigFloor {
        /// The floor `ε`, in millibits.
        epsilon_milli: i64,
    },
    /// Stop once the budget spent reaches the cap.
    BudgetExhausted,
    /// Stop if **any** sub-rule fires.
    Any(Vec<StoppingRule>),
    /// Stop only if **all** sub-rules fire.
    All(Vec<StoppingRule>),
}

impl StoppingRule {
    /// Evaluate the rule against the current `signals` — `true` means *stop*.
    #[must_use]
    pub fn evaluate(&self, signals: &StopSignals) -> bool {
        match self
        {
            Self::PosteriorMass { threshold_milli } =>
            {
                signals.max_posterior_mass_milli >= *threshold_milli
            },
            Self::EigFloor { epsilon_milli } => signals.best_eig_milli < *epsilon_milli,
            Self::BudgetExhausted => signals.budget_spent >= signals.budget_cap,
            Self::Any(rules) => rules.iter().any(|r| r.evaluate(signals)),
            Self::All(rules) => rules.iter().all(|r| r.evaluate(signals)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn individual_rules() {
        let s = StopSignals {
            max_posterior_mass_milli: 990,
            best_eig_milli: 5,
            budget_spent: 20,
            budget_cap: 20,
        };
        assert!(
            StoppingRule::PosteriorMass {
                threshold_milli: 990
            }
            .evaluate(&s)
        );
        assert!(
            !StoppingRule::PosteriorMass {
                threshold_milli: 995
            }
            .evaluate(&s)
        );
        assert!(StoppingRule::EigFloor { epsilon_milli: 10 }.evaluate(&s)); // 5 < 10 ⇒ exhausted
        assert!(!StoppingRule::EigFloor { epsilon_milli: 4 }.evaluate(&s));
        assert!(StoppingRule::BudgetExhausted.evaluate(&s)); // 20 >= 20
    }

    #[test]
    fn any_and_all_compose() {
        let s = StopSignals {
            max_posterior_mass_milli: 500,
            best_eig_milli: 100,
            budget_spent: 5,
            budget_cap: 20,
        };
        // Nothing individually fires here.
        let mass = StoppingRule::PosteriorMass {
            threshold_milli: 990,
        };
        let eig = StoppingRule::EigFloor { epsilon_milli: 50 };
        assert!(!StoppingRule::Any(vec![mass.clone(), eig.clone()]).evaluate(&s));
        assert!(!StoppingRule::All(vec![mass.clone(), eig.clone()]).evaluate(&s));
        // Add a firing rule.
        let low_eig = StoppingRule::EigFloor { epsilon_milli: 200 }; // 100 < 200 ⇒ fires
        assert!(StoppingRule::Any(vec![mass, low_eig.clone()]).evaluate(&s));
        assert!(!StoppingRule::All(vec![eig, low_eig]).evaluate(&s)); // one fires, one doesn't
    }
}

//! [`Derivation`] and [`DerivationStep`] — the explanation every conclusion
//! carries.
//!
//! A `Derivation` is a content-addressed `Object<Derivation>` (it implements
//! [`sos_core::Body`]), so an explanation can be stored, cited, and
//! independently re-verified — the audit trail that makes SOS's reasoning
//! trustworthy (RFC-0002 §05.4).

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Body, ObjectId};

use crate::soundness::Soundness;

/// One step of a derivation: a rule applied to some premises to reach an
/// intermediate conclusion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivationStep {
    /// The inference rule or technique applied, e.g. `"direct-edge"` or
    /// `"transitivity(specializes)"`.
    pub rule: String,
    /// The objects (knowledge nodes / edges) this step consumed.
    pub premises: Vec<ObjectId>,
    /// A stable, human-readable statement of what the step concluded.
    pub conclusion: String,
}

impl DerivationStep {
    /// Construct a derivation step.
    #[must_use]
    pub fn new(
        rule: impl Into<String>,
        premises: Vec<ObjectId>,
        conclusion: impl Into<String>,
    ) -> Self {
        Self {
            rule: rule.into(),
            premises,
            conclusion: conclusion.into(),
        }
    }
}

impl Canonical for DerivationStep {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.rule);
        enc.seq(&self.premises);
        enc.str(&self.conclusion);
    }
}

/// A complete derivation: the goal, the ordered steps that establish (or fail to
/// establish) it, the leaf premises it rests on, and its [`Soundness`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Derivation {
    /// A stable statement of the goal being derived.
    pub goal: String,
    /// The ordered inference steps.
    pub steps: Vec<DerivationStep>,
    /// The leaf objects (edges / nodes) the whole derivation cites.
    pub premises: Vec<ObjectId>,
    /// How strong the derivation is.
    pub soundness: Soundness,
}

impl Derivation {
    /// Construct a derivation.
    #[must_use]
    pub fn new(
        goal: impl Into<String>,
        steps: Vec<DerivationStep>,
        premises: Vec<ObjectId>,
        soundness: Soundness,
    ) -> Self {
        Self {
            goal: goal.into(),
            steps,
            premises,
            soundness,
        }
    }

    /// An empty derivation for an undetermined goal — no steps, `Check` soundness
    /// (deterministic "not found", not a disproof).
    #[must_use]
    pub fn undetermined(goal: impl Into<String>) -> Self {
        Self::new(goal, Vec::new(), Vec::new(), Soundness::Check)
    }
}

impl Canonical for Derivation {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.goal);
        enc.seq(&self.steps);
        enc.seq(&self.premises);
        enc.value(&self.soundness);
    }
}

impl Body for Derivation {
    const KIND: &'static str = "Derivation";
    const SCHEMA_VERSION: u32 = 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivation_is_canonical_and_content_sensitive() {
        let d1 = Derivation::new(
            "x specializes z",
            vec![DerivationStep::new(
                "direct-edge",
                vec![],
                "x specializes y",
            )],
            vec![],
            Soundness::Proof,
        );
        assert_eq!(d1.canonical_bytes(), d1.clone().canonical_bytes());
        let mut d2 = d1.clone();
        d2.soundness = Soundness::Check;
        assert_ne!(d1.canonical_bytes(), d2.canonical_bytes());
    }

    #[test]
    fn undetermined_is_a_check() {
        let d = Derivation::undetermined("a implies b");
        assert_eq!(d.soundness, Soundness::Check);
        assert!(d.steps.is_empty());
    }
}

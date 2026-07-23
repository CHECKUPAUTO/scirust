//! [`Soundness`] and [`Verdict`] — honest labels on a reasoning result.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};

/// How strong a conclusion is (RFC-0002 §05.5). A reasoning engine that
/// overstates its certainty is worse than none, so every conclusion declares
/// which of these it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Soundness {
    /// **Sound**: within its inference rules, if it says proven, it is (e.g.
    /// transitivity of a transitive relation; a direct asserted edge).
    Proof,
    /// **Deterministic but incomplete**: evidence, not a theorem (e.g. a bounded
    /// search that did not find a derivation, or a heuristic check). A `Check`
    /// must never be presented as a `Proof`.
    Check,
}

impl Soundness {
    /// A short, stable code (`"proof"` / `"check"`).
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self
        {
            Self::Proof => "proof",
            Self::Check => "check",
        }
    }
}

impl Canonical for Soundness {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(self.code());
    }
}

/// The outcome of a reasoning query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Verdict {
    /// The goal is derivable (with a sound derivation).
    Proven,
    /// The goal is refuted (its negation is derivable / it is excluded).
    Refuted,
    /// Neither proven nor refuted from the available knowledge.
    Undetermined,
}

impl Verdict {
    /// A short, stable code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self
        {
            Self::Proven => "proven",
            Self::Refuted => "refuted",
            Self::Undetermined => "undetermined",
        }
    }
}

impl Canonical for Verdict {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(self.code());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_and_canonical_are_stable() {
        assert_eq!(Soundness::Proof.code(), "proof");
        assert_eq!(Verdict::Undetermined.code(), "undetermined");
        assert_ne!(
            Soundness::Proof.canonical_bytes(),
            Soundness::Check.canonical_bytes()
        );
        assert_ne!(
            Verdict::Proven.canonical_bytes(),
            Verdict::Refuted.canonical_bytes()
        );
    }

    #[test]
    fn serde_roundtrips() {
        for v in [Verdict::Proven, Verdict::Refuted, Verdict::Undetermined]
        {
            let j = serde_json::to_string(&v).unwrap();
            assert_eq!(serde_json::from_str::<Verdict>(&j).unwrap(), v);
        }
    }
}

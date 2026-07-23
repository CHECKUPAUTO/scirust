//! [`Relation`] — the typed vocabulary of knowledge-graph edges.

use core::fmt;

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};

/// A directed, typed relationship between two knowledge objects.
///
/// The fixed variants are the core cross-domain vocabulary (RFC-0002 §04.2);
/// [`Relation::Custom`] carries a domain-specific relation named by string. Each
/// relation has a stable [`Relation::code`] used for canonical hashing and
/// display; a `Custom` code is prefixed (`custom:…`) so it can never collide
/// with a fixed relation's code.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Relation {
    /// `A is-a B` — instance/kind membership.
    IsA,
    /// `A generalizes B` — A is a more general form of B.
    Generalizes,
    /// `A specializes B` — A is a special case of B.
    Specializes,
    /// `A derives-from B` — A was derived from B.
    DerivesFrom,
    /// `A implies B` — A logically entails B.
    Implies,
    /// `A equivalent-to B` — A and B are equivalent.
    EquivalentTo,
    /// `A contradicts B` — A and B cannot both hold.
    Contradicts,
    /// `A supported-by B` — evidence B supports A.
    SupportedBy,
    /// `A refuted-by B` — evidence B refutes A.
    RefutedBy,
    /// `A constrained-by B` — constraint B restricts A.
    ConstrainedBy,
    /// `A has-dimension B` — A has dimensional signature B.
    HasDimension,
    /// `A measures B` — A is a measurement of B.
    Measures,
    /// `A cites B` — A references B.
    Cites,
    /// `A instance-of B` — A is an instance of type B.
    InstanceOf,
    /// `A supersedes B` — A replaces B.
    Supersedes,
    /// `A analogous-to B` — A is structurally analogous to B (cross-domain).
    AnalogousTo,
    /// `A limit-of B` — A is a limiting case of B.
    LimitOf,
    /// A domain-specific relation, named by string.
    Custom(String),
}

impl Relation {
    /// The stable code for this relation, used for canonical hashing and
    /// display. `Custom(s)` yields `"custom:{s}"`, which never collides with a
    /// fixed relation's code.
    #[must_use]
    pub fn code(&self) -> String {
        match self
        {
            Self::IsA => "is-a".into(),
            Self::Generalizes => "generalizes".into(),
            Self::Specializes => "specializes".into(),
            Self::DerivesFrom => "derives-from".into(),
            Self::Implies => "implies".into(),
            Self::EquivalentTo => "equivalent-to".into(),
            Self::Contradicts => "contradicts".into(),
            Self::SupportedBy => "supported-by".into(),
            Self::RefutedBy => "refuted-by".into(),
            Self::ConstrainedBy => "constrained-by".into(),
            Self::HasDimension => "has-dimension".into(),
            Self::Measures => "measures".into(),
            Self::Cites => "cites".into(),
            Self::InstanceOf => "instance-of".into(),
            Self::Supersedes => "supersedes".into(),
            Self::AnalogousTo => "analogous-to".into(),
            Self::LimitOf => "limit-of".into(),
            Self::Custom(s) => format!("custom:{s}"),
        }
    }
}

impl Canonical for Relation {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.code());
    }
}

impl fmt::Display for Relation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.code())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_stable_and_distinct() {
        assert_eq!(Relation::AnalogousTo.code(), "analogous-to");
        assert_ne!(Relation::IsA.code(), Relation::InstanceOf.code());
    }

    #[test]
    fn custom_cannot_collide_with_fixed() {
        // A Custom relation whose inner string equals a fixed code still gets a
        // distinct canonical code, so the two never hash the same.
        let custom = Relation::Custom("is-a".into());
        assert_eq!(custom.code(), "custom:is-a");
        assert_ne!(custom.canonical_bytes(), Relation::IsA.canonical_bytes());
    }

    #[test]
    fn canonical_and_serde_roundtrip() {
        for r in [Relation::Contradicts, Relation::Custom("catalyzes".into())]
        {
            assert_eq!(r.canonical_bytes(), r.clone().canonical_bytes());
            let json = serde_json::to_string(&r).unwrap();
            let back: Relation = serde_json::from_str(&json).unwrap();
            assert_eq!(r, back);
        }
    }
}

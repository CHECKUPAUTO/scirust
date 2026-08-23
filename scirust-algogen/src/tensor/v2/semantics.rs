//! Numerical semantic regimes for the V2 scientific IR.
//!
//! The interpreter always evaluates operations with Rust's native IEEE-754
//! operators at the declared dtype. A regime does **not** replace execution
//! semantics; it states which equivalence assumptions canonicalization and
//! search-time rewrites may rely on.

use serde::{Deserialize, Serialize};

/// Equivalence contract attached to every research program.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NumericalSemantics {
    /// Preserve IEEE evaluation semantics, including infinities, NaNs, signed
    /// zero, operation order, and the distinction between fused and unfused
    /// arithmetic. Only structural rewrites are permitted.
    #[default]
    StrictIeee,
    /// Rewrites may assume every value in the active program is finite.
    /// Signed zero remains observable and reassociation remains forbidden.
    FiniteOnly,
    /// Explicit research escape hatch for real-algebraic hypotheses. Results
    /// need not be IEEE-equivalent after a rewrite. This regime is never the
    /// default and must remain visible in canonical identity and archives.
    RealAlgebraicExperimental,
}

impl NumericalSemantics {
    /// Stable canonical-encoding tag. Tags are append-only.
    pub const fn tag(self) -> u8 {
        match self
        {
            Self::StrictIeee => 0,
            Self::FiniteOnly => 1,
            Self::RealAlgebraicExperimental => 2,
        }
    }

    /// Decode a canonical tag.
    pub const fn from_tag(tag: u8) -> Option<Self> {
        match tag
        {
            0 => Some(Self::StrictIeee),
            1 => Some(Self::FiniteOnly),
            2 => Some(Self::RealAlgebraicExperimental),
            _ => None,
        }
    }

    /// Whether finite-domain IEEE identities and commutative normalization
    /// are admitted. This never admits reassociation or FMA contraction.
    pub const fn admits_finite_rewrites(self) -> bool {
        matches!(self, Self::FiniteOnly | Self::RealAlgebraicExperimental)
    }

    /// Whether explicitly real-algebraic (not generally IEEE-equivalent)
    /// rules may run.
    pub const fn admits_real_algebra(self) -> bool {
        matches!(self, Self::RealAlgebraicExperimental)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_round_trip_and_default_is_strict() {
        for regime in [
            NumericalSemantics::StrictIeee,
            NumericalSemantics::FiniteOnly,
            NumericalSemantics::RealAlgebraicExperimental,
        ]
        {
            assert_eq!(NumericalSemantics::from_tag(regime.tag()), Some(regime));
        }
        assert_eq!(NumericalSemantics::from_tag(3), None);
        assert_eq!(
            NumericalSemantics::default(),
            NumericalSemantics::StrictIeee
        );
    }
}

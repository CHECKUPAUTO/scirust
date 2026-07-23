//! [`Strategy`] — the deterministic lens that surfaced a question.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};

/// Which deterministic scanner ("lens") raised a question (RFC-0002 §06.2).
///
/// Each strategy is a pure, deterministic scan of the knowledge graph. The set
/// is `#[non_exhaustive]` because further lenses — maximal-information-gain
/// (via `sos-planner`), cross-domain analogy (via `scirust-graph`),
/// unexplored-parameters (via `scirust-symbolic`) — are added as the backends
/// that power them attach (Invariant VIII); this crate ships the lenses it can
/// fully implement over the graph alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Strategy {
    /// Unresolved contradictions in the graph (found via the Reasoning Engine).
    ContradictionHunt,
    /// Weakly-connected / isolated nodes that few relations situate.
    UnderConnected,
    /// Claims that are refuted yet carry no recorded support.
    WeaklySupported,
}

impl Strategy {
    /// A short, stable code used for canonical hashing and display.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self
        {
            Self::ContradictionHunt => "contradiction-hunt",
            Self::UnderConnected => "under-connected",
            Self::WeaklySupported => "weakly-supported",
        }
    }
}

impl Canonical for Strategy {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(self.code());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_stable_and_distinct() {
        assert_eq!(Strategy::ContradictionHunt.code(), "contradiction-hunt");
        assert_ne!(
            Strategy::UnderConnected.canonical_bytes(),
            Strategy::WeaklySupported.canonical_bytes()
        );
    }

    #[test]
    fn serde_roundtrips() {
        for s in [
            Strategy::ContradictionHunt,
            Strategy::UnderConnected,
            Strategy::WeaklySupported,
        ]
        {
            let j = serde_json::to_string(&s).unwrap();
            assert_eq!(serde_json::from_str::<Strategy>(&j).unwrap(), s);
        }
    }
}

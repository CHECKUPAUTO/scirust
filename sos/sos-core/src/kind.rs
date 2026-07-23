//! [`Kind`] — an object's type name plus the schema version of its shape.
//!
//! `Kind` is part of every object's hashed identity and of its
//! domain-separation prefix, so two objects of different kinds (or different
//! schema versions of the same kind) can never share an id even if their bodies
//! encode identically. Bumping `schema_version` is therefore a breaking change
//! by construction — it changes every id — which is exactly the RFC-0002 §09.5
//! versioning rule made mechanical.

use serde::{Deserialize, Serialize};

use crate::canonical::{Canonical, CanonicalEncoder};

/// The type name and schema version of an [`crate::Object`]'s body.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Kind {
    /// Stable type name, e.g. `"Hypothesis"`. Conventionally UpperCamelCase and
    /// unique across the object catalog.
    pub name: String,
    /// Schema version of the body's *shape*. Incremented (breaking) whenever the
    /// canonical encoding of the body changes.
    pub schema_version: u32,
}

impl Kind {
    /// Construct a kind from a name and schema version.
    #[must_use]
    pub fn new(name: impl Into<String>, schema_version: u32) -> Self {
        Self {
            name: name.into(),
            schema_version,
        }
    }

    /// The domain-separation prefix used when hashing an object of this kind,
    /// e.g. `b"sos-obj:Hypothesis:v1"`. Distinct kinds/versions get distinct
    /// prefixes, so their ids live in disjoint spaces.
    #[must_use]
    pub fn domain(&self) -> Vec<u8> {
        format!("sos-obj:{}:v{}", self.name, self.schema_version).into_bytes()
    }
}

impl Canonical for Kind {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.name);
        enc.u64(u64::from(self.schema_version));
    }
}

impl core::fmt::Display for Kind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}@v{}", self.name, self.schema_version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_is_kind_and_version_separated() {
        let a = Kind::new("Law", 1);
        let b = Kind::new("Law", 2);
        let c = Kind::new("Theory", 1);
        assert_ne!(a.domain(), b.domain());
        assert_ne!(a.domain(), c.domain());
        assert_eq!(a.domain(), b"sos-obj:Law:v1");
    }

    #[test]
    fn canonical_distinguishes_version() {
        assert_ne!(
            Kind::new("Law", 1).canonical_bytes(),
            Kind::new("Law", 2).canonical_bytes()
        );
    }
}

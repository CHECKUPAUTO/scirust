//! [`Contradiction`] — a recorded incompatibility between two objects.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Author, Body, Object, ObjectId};

/// A detected incompatibility: `left` and `right` cannot both hold, for the
/// stated `reason`. Endpoints are stored in sorted order so the symmetric pair
/// `(a, b)` and `(b, a)` yield the same contradiction (dedup).
///
/// A `Contradiction` is a content-addressed `Object<Contradiction>` — a
/// first-class node, never a silent deletion (RFC-0002 §05.3).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Contradiction {
    /// The lesser of the two conflicting object ids.
    pub left: ObjectId,
    /// The greater of the two conflicting object ids.
    pub right: ObjectId,
    /// A stable, human-readable reason, e.g. `"asserted-contradiction"` or
    /// `"mutual-supersession"`.
    pub reason: String,
}

impl Contradiction {
    /// Construct a contradiction, normalising the endpoint order so the pair is
    /// symmetric (`a↔b` == `b↔a`).
    #[must_use]
    pub fn new(a: ObjectId, b: ObjectId, reason: impl Into<String>) -> Self {
        let (left, right) = if a <= b { (a, b) } else { (b, a) };
        Self {
            left,
            right,
            reason: reason.into(),
        }
    }
}

impl Canonical for Contradiction {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.left);
        enc.value(&self.right);
        enc.str(&self.reason);
    }
}

impl Body for Contradiction {
    const KIND: &'static str = "Contradiction";
    const SCHEMA_VERSION: u32 = 1;
}

/// Seal a [`Contradiction`] as a storable `Object<Contradiction>`.
#[must_use]
pub fn seal_contradiction(contradiction: Contradiction, author: Author) -> Object<Contradiction> {
    Object::builder(contradiction).author(author).seal()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sos_core::HashAlgo;

    fn oid(tag: &[u8]) -> ObjectId {
        ObjectId::compute(HashAlgo::default(), b"sos-obj:N:v1", tag)
    }

    #[test]
    fn endpoints_are_normalised_symmetric() {
        let a = oid(b"a");
        let b = oid(b"b");
        let c1 = Contradiction::new(a, b, "r");
        let c2 = Contradiction::new(b, a, "r");
        assert_eq!(c1, c2); // symmetric
        assert!(c1.left <= c1.right);
    }

    #[test]
    fn seals_to_a_verifiable_object() {
        let c = Contradiction::new(oid(b"a"), oid(b"b"), "asserted-contradiction");
        let obj = seal_contradiction(c, Author::engine("sos-reasoning"));
        assert!(obj.verify_id());
        assert_eq!(obj.kind.name, "Contradiction");
    }
}

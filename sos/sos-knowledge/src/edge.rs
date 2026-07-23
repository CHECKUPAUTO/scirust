//! [`Edge`] — a first-class, typed relation between two knowledge objects.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Author, Body, Object, ObjectId};

use crate::relation::Relation;

/// A directed, typed edge: `from --relation--> to`.
///
/// An edge is a body type, so an asserted relationship is sealed as an ordinary
/// content-addressed `Object<Edge>` — hashed, versioned, and provenance-bound
/// like any other object. Two identical assertions therefore share an id
/// (dedup), and an edge can be cited, superseded, or refuted like any node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    /// The source object.
    pub from: ObjectId,
    /// The target object.
    pub to: ObjectId,
    /// The typed relation from `from` to `to`.
    pub relation: Relation,
}

impl Edge {
    /// Construct an edge.
    #[must_use]
    pub fn new(from: ObjectId, to: ObjectId, relation: Relation) -> Self {
        Self { from, to, relation }
    }
}

impl Canonical for Edge {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.from);
        enc.value(&self.to);
        enc.value(&self.relation);
    }
}

impl Body for Edge {
    const KIND: &'static str = "Edge";
    const SCHEMA_VERSION: u32 = 1;
}

/// Build and seal an [`Edge`] as an `Object<Edge>`, ready to store.
///
/// The edge is authored by `author` with no provenance parents: its identity
/// derives from its endpoints and relation (in the body), and it is an
/// *assertion* rather than something derived from other objects. Store it with
/// [`sos_store::TypedStore::put_object`] like any object.
#[must_use]
pub fn seal_edge(from: ObjectId, to: ObjectId, relation: Relation, author: Author) -> Object<Edge> {
    Object::builder(Edge::new(from, to, relation))
        .author(author)
        .seal()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sos_core::HashAlgo;

    fn oid(tag: &[u8]) -> ObjectId {
        ObjectId::compute(HashAlgo::default(), b"sos-obj:T:v1", tag)
    }

    #[test]
    fn edge_kind_is_stable() {
        assert_eq!(Edge::KIND, "Edge");
        assert_eq!(Edge::SCHEMA_VERSION, 1);
    }

    #[test]
    fn identical_edges_share_an_id() {
        let (a, b) = (oid(b"a"), oid(b"b"));
        let e1 = seal_edge(a, b, Relation::IsA, Author::human("x"));
        let e2 = seal_edge(a, b, Relation::IsA, Author::human("x"));
        assert_eq!(e1.id, e2.id);
        assert!(e1.verify_id());
    }

    #[test]
    fn endpoints_and_relation_affect_identity() {
        let (a, b) = (oid(b"a"), oid(b"b"));
        let base = seal_edge(a, b, Relation::IsA, Author::human("x"));
        let swapped = seal_edge(b, a, Relation::IsA, Author::human("x"));
        let other_rel = seal_edge(a, b, Relation::Cites, Author::human("x"));
        assert_ne!(base.id, swapped.id); // direction matters
        assert_ne!(base.id, other_rel.id); // relation matters
    }
}

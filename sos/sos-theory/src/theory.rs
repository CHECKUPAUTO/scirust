//! [`Theory`] — a first-class, immutable, evolving scientific theory.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Author, Body, Object, ObjectId};

use crate::scope::Scope;

/// A scientific theory as a **first-class, immutable, evolving object** — not a
/// status flag on a hypothesis (RFC-0002 §07.3).
///
/// Every field is an [`ObjectId`] into the knowledge graph, so a `Theory` is a
/// **view over provenance**, not a document. Crucially, `contradicting` evidence
/// is a first-class field: a theory that hides its anomalies is dishonest, so
/// SOS keeps them, and "what does this theory fail to explain?" is always
/// answerable. A theory is never mutated — it **evolves** by
/// [revision](Theory::revise) into a *new* node that cites its parent.
///
/// Build one with [`Theory::builder`]; it normalizes the id-list fields
/// (sorted + deduplicated) so identical theories are content-addressed alike.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Theory {
    /// Knowledge nodes taken as given.
    pub axioms: Vec<ObjectId>,
    /// Explicit, defeasible premises.
    pub assumptions: Vec<ObjectId>,
    /// The `Law`/`Equation` nodes this theory asserts.
    pub equations: Vec<ObjectId>,
    /// Where it claims to hold (and, by exclusion, where it does not).
    pub domain_of_validity: Scope,
    /// Evidence **for** the theory.
    pub supporting: Vec<ObjectId>,
    /// Evidence **against** — retained, never hidden.
    pub contradicting: Vec<ObjectId>,
    /// A `Confidence` object (posterior / Bayes factors), once the statistics
    /// engine has estimated it. `None` until then — never a fabricated value.
    pub confidence: Option<ObjectId>,
    /// Papers / prior theories cited.
    pub citations: Vec<ObjectId>,
    /// The parent theory this one supersedes, if it is a revision.
    pub revises: Option<ObjectId>,
    /// Rival theories over the same phenomenon (which coexist — the engine does
    /// not force a single winner).
    pub competitors: Vec<ObjectId>,
}

impl Theory {
    /// Start building a theory over the given domain of validity.
    #[must_use]
    pub fn builder(domain_of_validity: Scope) -> TheoryBuilder {
        TheoryBuilder {
            inner: Theory {
                domain_of_validity,
                ..Theory::default()
            },
        }
    }

    /// Normalize the id-list fields in place: sort and deduplicate, so a theory's
    /// content address does not depend on insertion order.
    fn normalize(&mut self) {
        for v in [
            &mut self.axioms,
            &mut self.assumptions,
            &mut self.equations,
            &mut self.supporting,
            &mut self.contradicting,
            &mut self.citations,
            &mut self.competitors,
        ]
        {
            v.sort_unstable();
            v.dedup();
        }
    }

    /// Build a **successor** that revises `self` (whose id is `parent_id`),
    /// forced by `forced_by` — the `Evidence`/`Contradiction` objects that
    /// motivated the revision (RFC-0002 §07.3).
    ///
    /// The successor:
    /// * points back to the parent via [`revises`](Theory::revises);
    /// * **inherits** the parent's axioms, assumptions, equations, domain, and
    ///   citations (the caller then adjusts equations / narrows the domain as the
    ///   science requires — e.g. Newtonian mechanics as the low-velocity limit);
    /// * **retains all** of the parent's supporting *and* contradicting evidence,
    ///   and records `forced_by` among the contradicting evidence to be addressed
    ///   — anomalies are never dropped;
    /// * resets [`confidence`](Theory::confidence) to `None`, since a new theory's
    ///   confidence must be **re-estimated**, never inherited.
    ///
    /// The parent is untouched and remains a valid, queryable node.
    #[must_use]
    pub fn revise(&self, parent_id: ObjectId, forced_by: &[ObjectId]) -> Theory {
        let mut contradicting = self.contradicting.clone();
        contradicting.extend_from_slice(forced_by);
        let mut successor = Theory {
            axioms: self.axioms.clone(),
            assumptions: self.assumptions.clone(),
            equations: self.equations.clone(),
            domain_of_validity: self.domain_of_validity.clone(),
            supporting: self.supporting.clone(),
            contradicting,
            confidence: None,
            citations: self.citations.clone(),
            revises: Some(parent_id),
            competitors: self.competitors.clone(),
        };
        successor.normalize();
        successor
    }
}

impl Canonical for Theory {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.seq(&self.axioms);
        enc.seq(&self.assumptions);
        enc.seq(&self.equations);
        enc.value(&self.domain_of_validity);
        enc.seq(&self.supporting);
        enc.seq(&self.contradicting);
        enc.option(&self.confidence);
        enc.seq(&self.citations);
        enc.option(&self.revises);
        enc.seq(&self.competitors);
    }
}

impl Body for Theory {
    const KIND: &'static str = "Theory";
    const SCHEMA_VERSION: u32 = 1;
}

/// A builder for [`Theory`] that normalizes on [`build`](TheoryBuilder::build).
#[derive(Debug, Clone)]
pub struct TheoryBuilder {
    inner: Theory,
}

impl TheoryBuilder {
    /// Set the axioms.
    #[must_use]
    pub fn axioms(mut self, ids: Vec<ObjectId>) -> Self {
        self.inner.axioms = ids;
        self
    }

    /// Set the assumptions.
    #[must_use]
    pub fn assumptions(mut self, ids: Vec<ObjectId>) -> Self {
        self.inner.assumptions = ids;
        self
    }

    /// Set the asserted equations / laws.
    #[must_use]
    pub fn equations(mut self, ids: Vec<ObjectId>) -> Self {
        self.inner.equations = ids;
        self
    }

    /// Set the supporting evidence.
    #[must_use]
    pub fn supporting(mut self, ids: Vec<ObjectId>) -> Self {
        self.inner.supporting = ids;
        self
    }

    /// Set the contradicting evidence (retained anomalies).
    #[must_use]
    pub fn contradicting(mut self, ids: Vec<ObjectId>) -> Self {
        self.inner.contradicting = ids;
        self
    }

    /// Set the citations.
    #[must_use]
    pub fn citations(mut self, ids: Vec<ObjectId>) -> Self {
        self.inner.citations = ids;
        self
    }

    /// Set the competing theories.
    #[must_use]
    pub fn competitors(mut self, ids: Vec<ObjectId>) -> Self {
        self.inner.competitors = ids;
        self
    }

    /// Set the confidence object.
    #[must_use]
    pub fn confidence(mut self, id: ObjectId) -> Self {
        self.inner.confidence = Some(id);
        self
    }

    /// Mark this theory as a revision of `parent_id`.
    #[must_use]
    pub fn revises(mut self, parent_id: ObjectId) -> Self {
        self.inner.revises = Some(parent_id);
        self
    }

    /// Finish, normalizing the id-list fields.
    #[must_use]
    pub fn build(mut self) -> Theory {
        self.inner.normalize();
        self.inner
    }
}

/// Seal a [`Theory`] as a storable `Object<Theory>`.
#[must_use]
pub fn seal_theory(theory: Theory, author: Author) -> Object<Theory> {
    Object::builder(theory).author(author).seal()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sos_core::HashAlgo;

    fn oid(tag: &[u8]) -> ObjectId {
        ObjectId::compute(HashAlgo::default(), b"sos-obj:N:v1", tag)
    }

    #[test]
    fn builder_normalizes_id_lists() {
        let t = Theory::builder(Scope::universal())
            .equations(vec![oid(b"b"), oid(b"a"), oid(b"a")])
            .build();
        // sorted + deduplicated
        assert_eq!(t.equations, {
            let mut v = vec![oid(b"a"), oid(b"b")];
            v.sort_unstable();
            v
        });
    }

    #[test]
    fn revision_retains_anomalies_and_links_parent() {
        let parent_id = oid(b"parent");
        let parent = Theory::builder(Scope::from_predicates(["low-velocity"]))
            .supporting(vec![oid(b"ev1")])
            .contradicting(vec![oid(b"anomaly1")])
            .build();

        let forcing = oid(b"mercury-perihelion");
        let child = parent.revise(parent_id, &[forcing]);

        assert_eq!(child.revises, Some(parent_id));
        // Parent's anomaly is retained AND the forcing evidence is recorded.
        assert!(child.contradicting.contains(&oid(b"anomaly1")));
        assert!(child.contradicting.contains(&forcing));
        // Supporting evidence carried forward; confidence reset for re-estimation.
        assert!(child.supporting.contains(&oid(b"ev1")));
        assert_eq!(child.confidence, None);
        // The parent object is unchanged.
        assert_eq!(parent.revises, None);
    }

    #[test]
    fn seals_to_a_verifiable_object() {
        let t = Theory::builder(Scope::universal())
            .equations(vec![oid(b"e")])
            .build();
        let obj = seal_theory(t, Author::engine("sos-theory"));
        assert!(obj.verify_id());
        assert_eq!(obj.kind.name, "Theory");
    }
}

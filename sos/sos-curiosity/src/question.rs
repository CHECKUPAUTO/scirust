//! [`ScientificQuestion`] — a generated question, grounded in real graph nodes.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Author, Body, Object, ObjectId};

use crate::strategy::Strategy;

/// A scientific question the Curiosity Engine raises: *what to investigate*,
/// grounded in the graph nodes it concerns.
///
/// A `ScientificQuestion` is a content-addressed `Object<ScientificQuestion>`,
/// so the research agenda is itself auditable and citable (RFC-0002 §06.1). Its
/// `subject` cites the **real** nodes the question is about — a question that
/// grounds in nothing is never emitted (RFC-0002 §06.4). Subjects are stored in
/// sorted order so the identity is independent of discovery order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScientificQuestion {
    /// The graph node ids this question concerns (sorted, non-empty).
    pub subject: Vec<ObjectId>,
    /// A stable, human-readable statement of the question.
    pub prompt: String,
    /// The deterministic lens that raised it.
    pub strategy: Strategy,
}

impl ScientificQuestion {
    /// Construct a question, sorting `subject` so the identity is order-stable.
    #[must_use]
    pub fn new(mut subject: Vec<ObjectId>, prompt: impl Into<String>, strategy: Strategy) -> Self {
        subject.sort_unstable();
        subject.dedup();
        Self {
            subject,
            prompt: prompt.into(),
            strategy,
        }
    }

    /// Whether the question is grounded — it cites at least one graph node.
    #[must_use]
    pub fn is_grounded(&self) -> bool {
        !self.subject.is_empty()
    }
}

impl Canonical for ScientificQuestion {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.seq(&self.subject);
        enc.str(&self.prompt);
        enc.value(&self.strategy);
    }
}

impl Body for ScientificQuestion {
    const KIND: &'static str = "ScientificQuestion";
    const SCHEMA_VERSION: u32 = 1;
}

/// Seal a [`ScientificQuestion`] as a storable `Object<ScientificQuestion>`.
#[must_use]
pub fn seal_question(question: ScientificQuestion, author: Author) -> Object<ScientificQuestion> {
    Object::builder(question).author(author).seal()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sos_core::HashAlgo;

    fn oid(tag: &[u8]) -> ObjectId {
        ObjectId::compute(HashAlgo::default(), b"sos-obj:N:v1", tag)
    }

    #[test]
    fn subject_is_order_stable() {
        let (a, b) = (oid(b"a"), oid(b"b"));
        let q1 = ScientificQuestion::new(vec![a, b], "p", Strategy::ContradictionHunt);
        let q2 = ScientificQuestion::new(vec![b, a], "p", Strategy::ContradictionHunt);
        assert_eq!(q1, q2); // identity independent of input order
        assert!(q1.subject.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn seals_to_a_verifiable_object() {
        let q = ScientificQuestion::new(vec![oid(b"a")], "why?", Strategy::UnderConnected);
        assert!(q.is_grounded());
        let obj = seal_question(q, Author::engine("sos-curiosity"));
        assert!(obj.verify_id());
        assert_eq!(obj.kind.name, "ScientificQuestion");
    }
}

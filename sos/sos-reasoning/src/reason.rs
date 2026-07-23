//! The [`Reasoner`], the [`Reason`] trait, and [`Conclusion`].

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sos_core::{DeterminismLevel, ObjectId};
use sos_knowledge::{Knowledge, KnowledgeGraph, Relation};

use crate::contradiction::Contradiction;
use crate::derivation::{Derivation, DerivationStep};
use crate::soundness::{Soundness, Verdict};

/// A reasoning result: the [`Verdict`], the [`Derivation`] that explains it, and
/// the [`DeterminismLevel`] it was computed at (always `L3` here — the reasoning
/// is exact/symbolic).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Conclusion {
    /// Whether the goal was proven, refuted, or left undetermined.
    pub verdict: Verdict,
    /// The explanation — always present.
    pub derivation: Derivation,
    /// The determinism level of this conclusion.
    pub level: DeterminismLevel,
}

/// Whether a relation is **transitive**, so that a chain `a R b R c` soundly
/// entails `a R c`. Only transitive relations license transitive-closure
/// entailment; for the rest, only a directly-asserted edge counts.
#[must_use]
pub fn is_transitive(relation: &Relation) -> bool {
    matches!(
        relation,
        Relation::IsA
            | Relation::Specializes
            | Relation::Generalizes
            | Relation::DerivesFrom
            | Relation::Implies
            | Relation::Supersedes
            | Relation::LimitOf
            | Relation::EquivalentTo
    )
}

/// The read-side Reasoning Engine syscall: derive conclusions from the knowledge
/// graph, each with an explanation.
pub trait Reason {
    /// Decide whether `from --relation--> to` is derivable, returning a
    /// [`Conclusion`] whose [`Derivation`] cites the exact edges used.
    ///
    /// * A **direct** asserted edge ⇒ `Proven`, `Proof`.
    /// * A **chain** of a [`transitive`](is_transitive) relation ⇒ `Proven`,
    ///   `Proof` (transitivity is sound).
    /// * Otherwise ⇒ `Undetermined` (a `Check`-level "not found", never a
    ///   disproof).
    fn entails(&self, from: ObjectId, relation: &Relation, to: ObjectId) -> Conclusion;

    /// All contradictions in the graph: asserted `contradicts` edges and
    /// mutual-`supersedes` cycles, deduplicated and sorted.
    fn contradictions(&self) -> Vec<Contradiction>;
}

/// A deterministic reasoner over a [`KnowledgeGraph`].
#[derive(Debug, Clone, Copy)]
pub struct Reasoner<'g> {
    graph: &'g KnowledgeGraph,
}

impl<'g> Reasoner<'g> {
    /// Create a reasoner over a built knowledge graph.
    #[must_use]
    pub fn new(graph: &'g KnowledgeGraph) -> Self {
        Self { graph }
    }

    /// The id of the edge object asserting `from --relation--> to`, if present.
    fn edge_id(&self, from: ObjectId, relation: &Relation, to: ObjectId) -> Option<ObjectId> {
        self.graph
            .edges()
            .iter()
            .find(|e| e.from == from && e.to == to && &e.relation == relation)
            .map(|e| e.id)
    }
}

impl Reason for Reasoner<'_> {
    fn entails(&self, from: ObjectId, relation: &Relation, to: ObjectId) -> Conclusion {
        let goal = format!("{from} {} {to}", relation.code());

        // 1. A directly-asserted edge is a one-step proof.
        if let Some(eid) = self.edge_id(from, relation, to)
        {
            let step = DerivationStep::new(
                "direct-edge",
                vec![eid],
                format!("{from} {} {to} (asserted)", relation.code()),
            );
            return Conclusion {
                verdict: Verdict::Proven,
                derivation: Derivation::new(goal, vec![step], vec![eid], Soundness::Proof),
                level: DeterminismLevel::L3,
            };
        }

        // 2. For a transitive relation, a chain of edges is a sound proof.
        if is_transitive(relation)
        {
            if let Some(path) = self.graph.path(from, to, Some(relation))
            {
                let mut steps = Vec::new();
                let mut premises = Vec::new();
                for pair in path.windows(2)
                {
                    let (u, v) = (pair[0], pair[1]);
                    if let Some(eid) = self.edge_id(u, relation, v)
                    {
                        premises.push(eid);
                        steps.push(DerivationStep::new(
                            format!("transitivity({})", relation.code()),
                            vec![eid],
                            format!("{u} {} {v}", relation.code()),
                        ));
                    }
                }
                if !steps.is_empty()
                {
                    return Conclusion {
                        verdict: Verdict::Proven,
                        derivation: Derivation::new(goal, steps, premises, Soundness::Proof),
                        level: DeterminismLevel::L3,
                    };
                }
            }
        }

        // 3. Not derivable from the available knowledge.
        Conclusion {
            verdict: Verdict::Undetermined,
            derivation: Derivation::undetermined(goal),
            level: DeterminismLevel::L3,
        }
    }

    fn contradictions(&self) -> Vec<Contradiction> {
        let mut out: BTreeSet<Contradiction> = BTreeSet::new();
        for e in self.graph.edges()
        {
            match &e.relation
            {
                Relation::Contradicts =>
                {
                    out.insert(Contradiction::new(e.from, e.to, "asserted-contradiction"));
                },
                // A supersedes B and B supersedes A is an incoherent pair.
                Relation::Supersedes
                    if self
                        .graph
                        .related(e.to, e.from)
                        .contains(&Relation::Supersedes) =>
                {
                    out.insert(Contradiction::new(e.from, e.to, "mutual-supersession"));
                },
                _ =>
                {},
            }
        }
        out.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transitive_set_is_as_declared() {
        assert!(is_transitive(&Relation::Specializes));
        assert!(is_transitive(&Relation::Implies));
        assert!(!is_transitive(&Relation::Cites));
        assert!(!is_transitive(&Relation::AnalogousTo));
        assert!(!is_transitive(&Relation::Contradicts));
    }
}

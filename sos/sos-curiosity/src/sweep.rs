//! The sweep: [`Budget`], [`ScoredQuestion`], the [`BeCurious`] trait, and the
//! [`Curiosity`] engine with its deterministic scanners.

use serde::{Deserialize, Serialize};
use sos_core::ObjectId;
use sos_core::canonical::Canonical;
use sos_knowledge::{Knowledge, KnowledgeGraph, Relation};
use sos_reasoning::{Derivation, DerivationStep, Reason, Reasoner, Soundness};

use crate::policy::{CuriosityPolicy, Features, Priority};
use crate::question::ScientificQuestion;
use crate::strategy::Strategy;

/// Nodes with degree at or below this are treated as under-connected.
const UNDER_CONNECTED_MAX_DEGREE: usize = 1;

/// A bound on a single curiosity sweep — the OS idle loop is perpetual, but any
/// one sweep is finite (RFC-0002 §06.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    /// The maximum number of ranked questions a sweep returns.
    pub max_questions: usize,
}

impl Budget {
    /// A budget admitting at most `max_questions` questions.
    #[must_use]
    pub fn new(max_questions: usize) -> Self {
        Self { max_questions }
    }
}

impl Default for Budget {
    /// A modest default cap of 32 questions per sweep.
    fn default() -> Self {
        Self { max_questions: 32 }
    }
}

/// A ranked question with its score and the derivation explaining why it is
/// worth asking. Like every SOS conclusion, the question ships an explanation —
/// never "the engine felt this was interesting" (RFC-0002 §06.1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoredQuestion {
    /// The generated question.
    pub question: ScientificQuestion,
    /// Its auditable priority.
    pub priority: Priority,
    /// Why it is worth asking — grounded in real graph structure.
    pub derivation: Derivation,
}

/// The Curiosity Engine syscall (RFC-0002 §06.1): scan the knowledge graph and
/// emit ranked [`ScoredQuestion`]s, highest priority first.
pub trait BeCurious {
    /// Run every scanner under `policy`, rank the merged results deterministically,
    /// and return at most `budget.max_questions` of them.
    fn sweep(&self, policy: &CuriosityPolicy, budget: &Budget) -> Vec<ScoredQuestion>;
}

/// A deterministic curiosity engine over a [`KnowledgeGraph`].
#[derive(Debug, Clone, Copy)]
pub struct Curiosity<'g> {
    graph: &'g KnowledgeGraph,
}

/// A pre-scoring candidate: a question, its explanation, and its raw features.
struct Candidate {
    question: ScientificQuestion,
    derivation: Derivation,
    features: Features,
}

impl<'g> Curiosity<'g> {
    /// Create a curiosity engine over a built knowledge graph.
    #[must_use]
    pub fn new(graph: &'g KnowledgeGraph) -> Self {
        Self { graph }
    }

    /// A node's undirected degree: distinct incident out- plus in-edges.
    fn degree(&self, id: ObjectId) -> usize {
        self.graph.out_edges(id).len() + self.graph.in_edges(id).len()
    }

    /// Contradiction lens: every unresolved [`sos_reasoning::Contradiction`]
    /// becomes a "how is this resolved?" question.
    fn scan_contradictions(&self) -> Vec<Candidate> {
        let reasoner = Reasoner::new(self.graph);
        reasoner
            .contradictions()
            .into_iter()
            .map(|c| {
                let degree = self.degree(c.left).min(self.degree(c.right));
                let step = DerivationStep::new(
                    "contradiction-scan",
                    vec![c.left, c.right],
                    format!("{} and {} are in conflict ({})", c.left, c.right, c.reason),
                );
                let derivation = Derivation::new(
                    format!("resolve the conflict between {} and {}", c.left, c.right),
                    vec![step],
                    vec![c.left, c.right],
                    Soundness::Check,
                );
                let question = ScientificQuestion::new(
                    vec![c.left, c.right],
                    format!(
                        "How can the contradiction between {} and {} be resolved?",
                        c.left, c.right
                    ),
                    Strategy::ContradictionHunt,
                );
                Candidate {
                    question,
                    derivation,
                    features: Features::from_structure(degree, 2, true),
                }
            })
            .collect()
    }

    /// Connectivity lens: nodes with degree at or below
    /// [`UNDER_CONNECTED_MAX_DEGREE`] are candidates for further linking.
    fn scan_under_connected(&self) -> Vec<Candidate> {
        let mut out = Vec::new();
        for n in self.graph.nodes()
        {
            let d = self.degree(n);
            if d > UNDER_CONNECTED_MAX_DEGREE
            {
                continue;
            }
            let step =
                DerivationStep::new("connectivity-scan", vec![n], format!("{n} has degree {d}"));
            let derivation = Derivation::new(
                format!("situate {n} in the graph"),
                vec![step],
                vec![n],
                Soundness::Check,
            );
            let question = ScientificQuestion::new(
                vec![n],
                format!(
                    "Node {n} is weakly connected (degree {d}); what relations situate it in the graph?"
                ),
                Strategy::UnderConnected,
            );
            out.push(Candidate {
                question,
                derivation,
                features: Features::from_structure(d, 1, false),
            });
        }
        out
    }

    /// Support lens: a claim that is `refuted-by` something yet is `supported-by`
    /// nothing is a standing tension worth resolving.
    fn scan_weakly_supported(&self) -> Vec<Candidate> {
        let mut out = Vec::new();
        for n in self.graph.nodes()
        {
            let refuters = self.graph.neighbors(n, &Relation::RefutedBy);
            let supports = self.graph.neighbors(n, &Relation::SupportedBy);
            if refuters.is_empty() || !supports.is_empty()
            {
                continue;
            }
            let mut premises = vec![n];
            premises.extend(refuters.iter().copied());
            let step = DerivationStep::new(
                "support-scan",
                premises.clone(),
                format!(
                    "{n} is refuted by {} node(s) and supported by none",
                    refuters.len()
                ),
            );
            let derivation = Derivation::new(
                format!("substantiate or retract {n}"),
                vec![step],
                premises,
                Soundness::Check,
            );
            let question = ScientificQuestion::new(
                vec![n],
                format!(
                    "Claim {n} is refuted (by {}) with no recorded support; retract it or find supporting evidence?",
                    refuters.len()
                ),
                Strategy::WeaklySupported,
            );
            out.push(Candidate {
                question,
                derivation,
                features: Features::from_structure(self.degree(n), 1, false),
            });
        }
        out
    }
}

impl BeCurious for Curiosity<'_> {
    fn sweep(&self, policy: &CuriosityPolicy, budget: &Budget) -> Vec<ScoredQuestion> {
        let mut candidates = self.scan_contradictions();
        candidates.append(&mut self.scan_under_connected());
        candidates.append(&mut self.scan_weakly_supported());

        let mut scored: Vec<ScoredQuestion> = candidates
            .into_iter()
            .map(|c| ScoredQuestion {
                priority: policy.score(c.features),
                question: c.question,
                derivation: c.derivation,
            })
            .collect();

        // Deterministic ranking: highest total first; ties broken by the
        // question's canonical bytes (stable and author-independent).
        scored.sort_by(|a, b| {
            b.priority.total.cmp(&a.priority.total).then_with(|| {
                a.question
                    .canonical_bytes()
                    .cmp(&b.question.canonical_bytes())
            })
        });
        scored.truncate(budget.max_questions);
        scored
    }
}

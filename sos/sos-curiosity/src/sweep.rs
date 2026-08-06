//! The sweep: [`Budget`], [`ScoredQuestion`], the [`BeCurious`] trait, and the
//! [`Curiosity`] engine with its deterministic scanners.

use serde::{Deserialize, Serialize};
use sos_core::ObjectId;
use sos_core::canonical::Canonical;
use sos_knowledge::{Knowledge, KnowledgeGraph, Relation};
use sos_planner::Candidate as DesignCandidate;
use sos_reasoning::{Derivation, DerivationStep, Reason, Reasoner, Soundness};

use crate::policy::{CuriosityPolicy, Features, Priority};
use crate::question::ScientificQuestion;
use crate::strategy::Strategy;

/// Nodes with degree at or below this are treated as under-connected.
const UNDER_CONNECTED_MAX_DEGREE: usize = 1;

/// The default EIG saturation for [`Curiosity::with_designs`]: **2 bits**, in
/// millibits. Two bits distinguishes four equally-likely hypotheses — a
/// substantial result for one experiment — so it is a defensible default
/// "as interesting as it gets" reference. Callers who know their domain's
/// scale should say so explicitly rather than inherit this.
pub const DEFAULT_EIG_SATURATION_MILLI: i64 = 2_000;

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

/// A deterministic curiosity engine over a [`KnowledgeGraph`], optionally
/// also over planner-supplied designs ([`Curiosity::with_designs`]).
#[derive(Debug, Clone, Copy)]
pub struct Curiosity<'g> {
    graph: &'g KnowledgeGraph,
    designs: &'g [DesignCandidate],
    eig_saturation_milli: i64,
}

/// A pre-scoring candidate: a question, its explanation, and its raw features.
struct Candidate {
    question: ScientificQuestion,
    derivation: Derivation,
    features: Features,
}

impl<'g> Curiosity<'g> {
    /// Create a curiosity engine over a built knowledge graph. Without
    /// [`with_designs`](Curiosity::with_designs) the sweep runs the three purely
    /// structural lenses.
    #[must_use]
    pub fn new(graph: &'g KnowledgeGraph) -> Self {
        Self {
            graph,
            designs: &[],
            eig_saturation_milli: DEFAULT_EIG_SATURATION_MILLI,
        }
    }

    /// Additionally consider `designs` — candidate experiments the **Planning
    /// Engine** has estimated the expected information gain of — enabling the
    /// [`Strategy::MaxInfoGain`] lens.
    ///
    /// This is the sanctioned `sos-curiosity` → `sos-planner` composition edge
    /// (RFC-0002 §11.5 rule 3): curiosity does not *compute* EIG (that is the
    /// planner's job, per Invariant VIII) — it asks the questions that a real
    /// EIG estimate justifies, scored by the very same [`CuriosityPolicy`] as
    /// every structural question, so the two are directly comparable.
    ///
    /// `eig_saturation_milli` is the EIG (in millibits) at which the
    /// information-gain feature reads its maximum; see
    /// [`Features::from_design`] for why that reference is explicit and
    /// caller-owned. [`DEFAULT_EIG_SATURATION_MILLI`] is a reasonable default.
    #[must_use]
    pub fn with_designs(
        mut self,
        designs: &'g [DesignCandidate],
        eig_saturation_milli: i64,
    ) -> Self {
        self.designs = designs;
        self.eig_saturation_milli = eig_saturation_milli;
        self
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

    /// Information lens: a planner-supplied design whose expected information
    /// gain justifies running it becomes a "run this experiment?" question.
    ///
    /// A design whose EIG is **not significant** — its estimate does not exceed
    /// its own standard error ([`sos_planner::Estimate::is_significant`]) — is
    /// skipped entirely rather than emitted with a near-zero score. That
    /// mirrors the planner's own admission rule (its `clears_floor` requires
    /// significance too), and keeps the agenda free of questions whose premise
    /// is "this might teach us nothing".
    fn scan_max_info_gain(&self) -> Vec<Candidate> {
        self.designs
            .iter()
            .filter(|d| d.eig.is_significant())
            .map(|d| {
                let lower = d.eig.lower_bound().max(0);
                let step = DerivationStep::new(
                    "information-gain-scan",
                    vec![d.experiment],
                    format!(
                        "design {} is estimated to yield {} ± {} millibits at cost {} \
                         (defensible lower bound {lower})",
                        d.experiment,
                        d.eig.bits_milli,
                        d.eig.se_milli,
                        d.cost.total()
                    ),
                );
                let derivation = Derivation::new(
                    format!("run {} to gain information", d.experiment),
                    vec![step],
                    vec![d.experiment],
                    Soundness::Check,
                );
                let question = ScientificQuestion::new(
                    vec![d.experiment],
                    format!(
                        "Design {} is estimated to yield {} millibits of information \
                         (± {}) at cost {}; should it be run next?",
                        d.experiment,
                        d.eig.bits_milli,
                        d.eig.se_milli,
                        d.cost.total()
                    ),
                    Strategy::MaxInfoGain,
                );
                Candidate {
                    question,
                    derivation,
                    features: Features::from_design(
                        self.degree(d.experiment),
                        &d.eig,
                        &d.cost,
                        self.eig_saturation_milli,
                    ),
                }
            })
            .collect()
    }
}

impl BeCurious for Curiosity<'_> {
    fn sweep(&self, policy: &CuriosityPolicy, budget: &Budget) -> Vec<ScoredQuestion> {
        let mut candidates = self.scan_contradictions();
        candidates.append(&mut self.scan_under_connected());
        candidates.append(&mut self.scan_weakly_supported());
        candidates.append(&mut self.scan_max_info_gain());

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

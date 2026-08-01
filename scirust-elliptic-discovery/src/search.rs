//! Fixed-budget gate pipeline for candidate falsification and scale validation.

use core::fmt;

use crate::{
    Classification, ClassificationStatus, Corpus, CorpusKind, Counterexample, ExperimentManifest,
    LocalResearchCase, Relation, classify, first_point_counterexample, generate_relations,
};

const HOLDOUT_SEED_DOMAIN: u64 = 0x484f_4c44_4f55_5421;
const SCALE_SEED_DOMAIN: u64 = 0x5343_414c_4521_2121;

/// Hard upper bounds keep the finite grammar and corpus work explicit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SearchPlan
{
    seed: u64,
    expression_depth: u8,
    max_scalar: u64,
    candidate_budget: u32,
    tuple_budget_per_candidate: u64,
    sampled_curves_per_prime: u32,
}

impl SearchPlan
{
    pub const MAX_EXPRESSION_DEPTH: u8 = 4;
    pub const MAX_SCALAR: u64 = 32;
    pub const MAX_CANDIDATES: u32 = 10_000;
    pub const MAX_TUPLES_PER_CANDIDATE: u64 = 10_000_000;
    pub const MAX_CURVES_PER_PRIME: u32 = 64;

    /// Validates every finite search bound before corpus construction.
    pub fn new(
        seed: u64,
        expression_depth: u8,
        max_scalar: u64,
        candidate_budget: u32,
        tuple_budget_per_candidate: u64,
        sampled_curves_per_prime: u32,
    ) -> Result<Self, SearchError>
    {
        if expression_depth > Self::MAX_EXPRESSION_DEPTH
        {
            return Err(SearchError::ExpressionDepth);
        }
        if max_scalar > Self::MAX_SCALAR
        {
            return Err(SearchError::Scalar);
        }
        if candidate_budget == 0 || candidate_budget > Self::MAX_CANDIDATES
        {
            return Err(SearchError::CandidateBudget);
        }
        if tuple_budget_per_candidate == 0
            || tuple_budget_per_candidate > Self::MAX_TUPLES_PER_CANDIDATE
        {
            return Err(SearchError::TupleBudget);
        }
        if sampled_curves_per_prime == 0
            || sampled_curves_per_prime > Self::MAX_CURVES_PER_PRIME
        {
            return Err(SearchError::CurveBudget);
        }
        Ok(Self {
            seed,
            expression_depth,
            max_scalar,
            candidate_budget,
            tuple_budget_per_candidate,
            sampled_curves_per_prime,
        })
    }
}

/// Invalid or excessive search configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchError
{
    ExpressionDepth,
    Scalar,
    CandidateBudget,
    TupleBudget,
    CurveBudget,
}

impl fmt::Display for SearchError
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result
    {
        match self
        {
            Self::ExpressionDepth => write!(formatter, "expression depth exceeds the fixed bound"),
            Self::Scalar => write!(formatter, "scalar exceeds the fixed bound"),
            Self::CandidateBudget => write!(formatter, "candidate budget is zero or excessive"),
            Self::TupleBudget => write!(formatter, "tuple budget is zero or excessive"),
            Self::CurveBudget => write!(formatter, "curve budget is zero or excessive"),
        }
    }
}

impl std::error::Error for SearchError {}

/// Three disjoint deterministic datasets required by gates G2, G3 and G6.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResearchCorpora
{
    exhaustive_small: Corpus,
    independent_holdout: Corpus,
    scale_ladder: Corpus,
}

impl ResearchCorpora
{
    /// Builds all partitions before candidate generation.
    pub fn generate(plan: SearchPlan) -> Self
    {
        let exhaustive_small = corpus(
            plan.seed,
            CorpusKind::ExhaustiveSmall,
            1,
            plan.tuple_budget_per_candidate,
        );
        let independent_holdout = corpus(
            plan.seed ^ HOLDOUT_SEED_DOMAIN,
            CorpusKind::IndependentHoldout,
            plan.sampled_curves_per_prime,
            plan.tuple_budget_per_candidate,
        );
        let scale_ladder = corpus(
            plan.seed ^ SCALE_SEED_DOMAIN,
            CorpusKind::ScaleLadder,
            plan.sampled_curves_per_prime,
            plan.tuple_budget_per_candidate,
        );
        Self {
            exhaustive_small,
            independent_holdout,
            scale_ladder,
        }
    }

    pub const fn exhaustive_small(&self) -> &Corpus
    {
        &self.exhaustive_small
    }

    pub const fn independent_holdout(&self) -> &Corpus
    {
        &self.independent_holdout
    }

    pub const fn scale_ladder(&self) -> &Corpus
    {
        &self.scale_ladder
    }
}

fn corpus(seed: u64, kind: CorpusKind, curves: u32, tuples: u64) -> Corpus
{
    let research_case = LocalResearchCase::new(seed, kind, curves, tuples)
        .expect("validated search plan creates a valid local case");
    Corpus::generate(ExperimentManifest::new(research_case))
}

/// Outcome of one mandatory dataset gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GateState
{
    Passed,
    Refuted,
    InsufficientCoverage,
}

/// Exact coverage record for one gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GateReport
{
    corpus: CorpusKind,
    evaluated_tuples: u64,
    required_tuples: u64,
    state: GateState,
}

impl GateReport
{
    pub const fn corpus(self) -> CorpusKind
    {
        self.corpus
    }

    pub const fn evaluated_tuples(self) -> u64
    {
        self.evaluated_tuples
    }

    pub const fn required_tuples(self) -> u64
    {
        self.required_tuples
    }

    pub const fn state(self) -> GateState
    {
        self.state
    }
}

/// Complete automated evaluation of one candidate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateEvaluation
{
    relation: Relation,
    classification: Classification,
    gates: Vec<GateReport>,
    counterexample: Option<Counterexample>,
}

impl CandidateEvaluation
{
    pub const fn relation(&self) -> &Relation
    {
        &self.relation
    }

    pub const fn classification(&self) -> Classification
    {
        self.classification
    }

    pub fn gates(&self) -> &[GateReport]
    {
        &self.gates
    }

    pub const fn counterexample(&self) -> Option<&Counterexample>
    {
        self.counterexample.as_ref()
    }
}

/// Runs G2 through G6 in order. No passing unknown relation skips the scale gate.
pub fn evaluate_candidate(
    relation: Relation,
    corpora: &ResearchCorpora,
    tuple_budget: u64,
) -> CandidateEvaluation
{
    let signature = relation.signature();
    let mut gates = Vec::new();

    let (g2, counterexample) = evaluate_gate(
        &relation,
        corpora.exhaustive_small(),
        tuple_budget,
        "G2.ExhaustiveSmall",
    );
    gates.push(g2);
    if g2.state != GateState::Passed
    {
        return finish_early(relation, signature, gates, counterexample, g2.state);
    }

    let (g3, counterexample) = evaluate_gate(
        &relation,
        corpora.independent_holdout(),
        tuple_budget,
        "G3.IndependentHoldout",
    );
    gates.push(g3);
    if g3.state != GateState::Passed
    {
        return finish_early(relation, signature, gates, counterexample, g3.state);
    }

    let catalog_classification = classify(signature, false);
    if matches!(
        catalog_classification.status(),
        ClassificationStatus::Known | ClassificationStatus::RepresentationArtifact
    )
    {
        return CandidateEvaluation {
            relation,
            classification: catalog_classification,
            gates,
            counterexample: None,
        };
    }

    let (g6, counterexample) = evaluate_gate(
        &relation,
        corpora.scale_ladder(),
        tuple_budget,
        "G6.ScaleLadder",
    );
    gates.push(g6);
    if g6.state != GateState::Passed
    {
        return finish_early(relation, signature, gates, counterexample, g6.state);
    }

    CandidateEvaluation {
        relation,
        classification: catalog_classification,
        gates,
        counterexample: None,
    }
}

fn finish_early(
    relation: Relation,
    signature: crate::RelationSignature,
    gates: Vec<GateReport>,
    counterexample: Option<Counterexample>,
    state: GateState,
) -> CandidateEvaluation
{
    let classification = match state
    {
        GateState::Refuted => classify(signature, true),
        GateState::InsufficientCoverage | GateState::Passed => Classification::inconclusive(),
    };
    CandidateEvaluation {
        relation,
        classification,
        gates,
        counterexample,
    }
}

fn evaluate_gate(
    relation: &Relation,
    corpus: &Corpus,
    tuple_budget: u64,
    relation_id: &str,
) -> (GateReport, Option<Counterexample>)
{
    let required_tuples = corpus.total_points();
    let mut evaluated_tuples = 0u64;
    let counterexample = first_point_counterexample(corpus, relation_id, |curve, point| {
        if evaluated_tuples == tuple_budget
        {
            return true;
        }
        evaluated_tuples += 1;
        relation.evaluate(curve, point).unwrap_or(false)
    });
    let state = if counterexample.is_some()
    {
        GateState::Refuted
    }
    else if evaluated_tuples < required_tuples
    {
        GateState::InsufficientCoverage
    }
    else
    {
        GateState::Passed
    };
    (
        GateReport {
            corpus: corpus.manifest().research_case().corpus(),
            evaluated_tuples,
            required_tuples,
            state,
        },
        counterexample,
    )
}

/// Generates and evaluates the deterministic candidate prefix.
pub fn run_search(plan: SearchPlan, corpora: &ResearchCorpora) -> Vec<CandidateEvaluation>
{
    generate_relations(
        plan.expression_depth,
        plan.max_scalar,
        usize::try_from(plan.candidate_budget).expect("candidate budget fits in usize"),
    )
    .into_iter()
    .map(|relation| {
        evaluate_candidate(relation, corpora, plan.tuple_budget_per_candidate)
    })
    .collect()
}

#[cfg(test)]
mod tests
{
    use super::*;
    use crate::PointExpression;

    fn plan() -> SearchPlan
    {
        SearchPlan::new(7, 2, 3, 10, 1_000_000, 1).expect("bounded plan")
    }

    #[test]
    fn false_training_relation_is_refuted_at_g2()
    {
        let corpora = ResearchCorpora::generate(plan());
        let result = evaluate_candidate(
            Relation::CurveAEquals(0),
            &corpora,
            plan().tuple_budget_per_candidate,
        );
        assert_eq!(result.classification().status(), ClassificationStatus::Refuted);
        assert_eq!(result.gates()[0].state(), GateState::Refuted);
        assert!(result.counterexample().is_some());
    }

    #[test]
    fn known_relation_passes_independent_data_then_hits_catalog()
    {
        let corpora = ResearchCorpora::generate(plan());
        let input = PointExpression::Input;
        let relation = Relation::PointEqual(
            PointExpression::Negate(Box::new(PointExpression::Negate(Box::new(
                input.clone(),
            )))),
            input,
        );
        let result = evaluate_candidate(
            relation,
            &corpora,
            plan().tuple_budget_per_candidate,
        );
        assert_eq!(result.classification().status(), ClassificationStatus::Known);
        assert_eq!(result.gates().len(), 2);
        assert!(result.gates().iter().all(|gate| gate.state() == GateState::Passed));
    }

    #[test]
    fn insufficient_budget_is_never_a_pass()
    {
        let corpora = ResearchCorpora::generate(plan());
        let relation = Relation::PointEqual(PointExpression::Input, PointExpression::Input);
        let result = evaluate_candidate(relation, &corpora, 1);
        assert_eq!(
            result.classification().status(),
            ClassificationStatus::Inconclusive
        );
        assert_eq!(result.gates()[0].state(), GateState::InsufficientCoverage);
    }
}

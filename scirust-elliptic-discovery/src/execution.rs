//! Canonical execution receipts and strict replay for local-only searches.

use crate::canonical::{CanonicalEncoder, sha256};
use crate::{
    CandidateEvaluation, CatalogFamily, ClassificationStatus, CorpusKind, Counterexample,
    GateState, PointExpression, Relation, ResearchCorpora, SearchPlan, run_search,
};

const STATUSES: [ClassificationStatus; 6] = [
    ClassificationStatus::Refuted,
    ClassificationStatus::Known,
    ClassificationStatus::RepresentationArtifact,
    ClassificationStatus::NeedsLiteratureReview,
    ClassificationStatus::Inconclusive,
    ClassificationStatus::CandidateUnclassified,
];

/// Exact aggregate over every automated classification in one run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionSummary {
    status_counts: [u32; 6],
    counterexamples: u32,
}

impl ExecutionSummary {
    /// Number of candidates carrying a given conservative status.
    pub const fn count(self, status: ClassificationStatus) -> u32 {
        self.status_counts[status_index(status)]
    }

    /// Number of candidates evaluated by the run.
    pub fn candidate_count(self) -> u32 {
        self.status_counts.iter().sum()
    }

    /// Number of candidates with a recorded first counterexample.
    pub const fn counterexample_count(self) -> u32 {
        self.counterexamples
    }
}

/// Canonical result of a complete, bounded, local search.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionReceipt {
    plan: SearchPlan,
    corpus_fingerprints: [[u8; 32]; 3],
    candidate_fingerprints: Vec<[u8; 32]>,
    summary: ExecutionSummary,
}

impl ExecutionReceipt {
    pub const SCHEMA_VERSION: u32 = 2;

    /// Validated search plan which can be replayed without external input.
    pub const fn plan(&self) -> SearchPlan {
        self.plan
    }

    /// Corpus fingerprints ordered as exhaustive, holdout, then scale ladder.
    pub const fn corpus_fingerprints(&self) -> &[[u8; 32]; 3] {
        &self.corpus_fingerprints
    }

    /// Ordered exact fingerprints of all candidate evaluations.
    pub fn candidate_fingerprints(&self) -> &[[u8; 32]] {
        &self.candidate_fingerprints
    }

    /// Conservative aggregate of the run.
    pub const fn summary(&self) -> ExecutionSummary {
        self.summary
    }

    /// Stable binary representation of the complete receipt.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoder =
            CanonicalEncoder::with_domain(b"SCIRUST-ELLIPTIC-DISCOVERY/EXECUTION-RECEIPT/V2");
        encoder.u32(Self::SCHEMA_VERSION);
        encoder.bytes(&self.plan.canonical_bytes());
        for (kind, fingerprint) in [
            CorpusKind::ExhaustiveSmall,
            CorpusKind::IndependentHoldout,
            CorpusKind::ScaleLadder,
        ]
        .into_iter()
        .zip(self.corpus_fingerprints)
        {
            encoder.u8(kind.tag());
            encoder.bytes(&fingerprint);
        }
        encoder.u64(
            u64::try_from(self.candidate_fingerprints.len())
                .expect("candidate fingerprint count fits in u64"),
        );
        for fingerprint in &self.candidate_fingerprints
        {
            encoder.bytes(fingerprint);
        }
        for status in STATUSES
        {
            encoder.u8(status_tag(status));
            encoder.u32(self.summary.count(status));
        }
        encoder.u32(self.summary.counterexample_count());
        encoder.finish()
    }

    /// Integrity fingerprint of all receipt fields.
    pub fn fingerprint(&self) -> [u8; 32] {
        sha256(&self.canonical_bytes())
    }
}

/// Result of recomputing a receipt from its closed local plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayReport {
    expected_fingerprint: [u8; 32],
    observed: ExecutionReceipt,
    matches: bool,
}

impl ReplayReport {
    /// Fingerprint supplied by the receipt being audited.
    pub const fn expected_fingerprint(&self) -> [u8; 32] {
        self.expected_fingerprint
    }

    /// Newly computed receipt retained even when replay diverges.
    pub const fn observed(&self) -> &ExecutionReceipt {
        &self.observed
    }

    /// Whether every canonical receipt byte matched.
    pub const fn matches(&self) -> bool {
        self.matches
    }
}

/// Runs the complete deterministic search using only built-in local corpora.
pub fn execute_local(plan: SearchPlan) -> ExecutionReceipt {
    let corpora = ResearchCorpora::generate(plan);
    let candidates = run_search(plan, &corpora);
    let corpus_fingerprints = [
        corpora.exhaustive_small().fingerprint(),
        corpora.independent_holdout().fingerprint(),
        corpora.scale_ladder().fingerprint(),
    ];
    let candidate_fingerprints = candidates.iter().map(candidate_fingerprint).collect();
    let summary = summarize(&candidates);
    ExecutionReceipt {
        plan,
        corpus_fingerprints,
        candidate_fingerprints,
        summary,
    }
}

/// Recomputes a receipt and reports any byte-level divergence.
pub fn replay_local(expected: &ExecutionReceipt) -> ReplayReport {
    let expected_bytes = expected.canonical_bytes();
    let expected_fingerprint = sha256(&expected_bytes);
    let observed = execute_local(expected.plan);
    let matches = expected_bytes == observed.canonical_bytes();
    ReplayReport {
        expected_fingerprint,
        observed,
        matches,
    }
}

fn summarize(candidates: &[CandidateEvaluation]) -> ExecutionSummary {
    let mut status_counts = [0u32; 6];
    let mut counterexamples = 0u32;
    for candidate in candidates
    {
        let index = status_index(candidate.classification().status());
        status_counts[index] = status_counts[index]
            .checked_add(1)
            .expect("candidate budget keeps status counts in u32");
        if candidate.counterexample().is_some()
        {
            counterexamples = counterexamples
                .checked_add(1)
                .expect("candidate budget keeps counterexample count in u32");
        }
    }
    ExecutionSummary {
        status_counts,
        counterexamples,
    }
}

fn candidate_fingerprint(candidate: &CandidateEvaluation) -> [u8; 32] {
    let mut encoder =
        CanonicalEncoder::with_domain(b"SCIRUST-ELLIPTIC-DISCOVERY/CANDIDATE-EVALUATION/V2");
    encode_relation(&mut encoder, candidate.relation());
    let classification = candidate.classification();
    encoder.u8(status_tag(classification.status()));
    match classification.catalog()
    {
        Some(entry) =>
        {
            encoder.u8(1);
            encoder.bytes(entry.id.as_bytes());
            encoder.u8(catalog_family_tag(entry.family));
            encoder.u8(u8::from(entry.conditional));
            encoder.u8(u8::from(entry.representation_artifact));
            encoder.bytes(entry.reference.as_bytes());
        },
        None => encoder.u8(0),
    }
    encoder.u64(u64::try_from(candidate.gates().len()).expect("gate count fits in u64"));
    for gate in candidate.gates()
    {
        encoder.u8(gate.corpus().tag());
        encoder.u64(gate.evaluated_tuples());
        encoder.u64(gate.required_tuples());
        encoder.u8(gate_state_tag(gate.state()));
    }
    match candidate.counterexample()
    {
        Some(counterexample) =>
        {
            encoder.u8(1);
            encode_counterexample(&mut encoder, counterexample);
        },
        None => encoder.u8(0),
    }
    sha256(&encoder.finish())
}

fn encode_relation(encoder: &mut CanonicalEncoder, relation: &Relation) {
    match relation
    {
        Relation::PointEqual(left, right) =>
        {
            encoder.u8(0);
            encode_point_expression(encoder, left);
            encode_point_expression(encoder, right);
        },
        Relation::IsInfinity(point) =>
        {
            encoder.u8(1);
            encode_point_expression(encoder, point);
        },
        Relation::CurveAEquals(value) =>
        {
            encoder.u8(2);
            encoder.u64(*value);
        },
        Relation::CurveJEquals(value) =>
        {
            encoder.u8(3);
            encoder.u64(*value);
        },
    }
}

fn encode_point_expression(encoder: &mut CanonicalEncoder, expression: &PointExpression) {
    match expression
    {
        PointExpression::Input => encoder.u8(0),
        PointExpression::Identity => encoder.u8(1),
        PointExpression::Negate(point) =>
        {
            encoder.u8(2);
            encode_point_expression(encoder, point);
        },
        PointExpression::Double(point) =>
        {
            encoder.u8(3);
            encode_point_expression(encoder, point);
        },
        PointExpression::ScalarMultiply { scalar, point } =>
        {
            encoder.u8(4);
            encoder.u64(*scalar);
            encode_point_expression(encoder, point);
        },
        PointExpression::Add(left, right) =>
        {
            encoder.u8(5);
            encode_point_expression(encoder, left);
            encode_point_expression(encoder, right);
        },
    }
}

fn encode_counterexample(encoder: &mut CanonicalEncoder, counterexample: &Counterexample) {
    encoder.bytes(counterexample.relation_id().as_bytes());
    let (prime, a, b) = counterexample.curve_key();
    encoder.u64(prime);
    encoder.u64(a);
    encoder.u64(b);
    encoder.u64(counterexample.point_index());
    match counterexample.point().affine_coordinates()
    {
        Some((x, y)) =>
        {
            encoder.u8(1);
            encoder.u64(x);
            encoder.u64(y);
        },
        None => encoder.u8(0),
    }
}

const fn status_index(status: ClassificationStatus) -> usize {
    match status
    {
        ClassificationStatus::Refuted => 0,
        ClassificationStatus::Known => 1,
        ClassificationStatus::RepresentationArtifact => 2,
        ClassificationStatus::NeedsLiteratureReview => 3,
        ClassificationStatus::Inconclusive => 4,
        ClassificationStatus::CandidateUnclassified => 5,
    }
}

const fn status_tag(status: ClassificationStatus) -> u8 {
    match status
    {
        ClassificationStatus::Refuted => 0,
        ClassificationStatus::Known => 1,
        ClassificationStatus::RepresentationArtifact => 2,
        ClassificationStatus::NeedsLiteratureReview => 3,
        ClassificationStatus::Inconclusive => 4,
        ClassificationStatus::CandidateUnclassified => 5,
    }
}

const fn gate_state_tag(state: GateState) -> u8 {
    match state
    {
        GateState::Passed => 0,
        GateState::Refuted => 1,
        GateState::InsufficientCoverage => 2,
    }
}

const fn catalog_family_tag(family: CatalogFamily) -> u8 {
    match family
    {
        CatalogFamily::NegationAndIdentity => 0,
        CatalogFamily::GroupLinearity => 1,
        CatalogFamily::JZeroAutomorphism => 2,
        CatalogFamily::CubeRootsOfUnity => 3,
        CatalogFamily::J1728Automorphism => 4,
        CatalogFamily::GlvEndomorphism => 5,
        CatalogFamily::CoordinateChange => 6,
        CatalogFamily::EncodingSymmetry => 7,
        CatalogFamily::TwistAndJClass => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(seed: u64) -> SearchPlan {
        SearchPlan::new(seed, 1, 2, 3, 1, 1).expect("bounded local plan")
    }

    #[test]
    fn repeated_executions_are_byte_identical() {
        let left = execute_local(plan(17));
        let right = execute_local(plan(17));
        assert_eq!(left.canonical_bytes(), right.canonical_bytes());
        assert_eq!(left.fingerprint(), right.fingerprint());
        assert_eq!(left.summary().candidate_count(), 3);
    }

    #[test]
    fn different_seeds_change_the_receipt() {
        assert_ne!(
            execute_local(plan(17)).fingerprint(),
            execute_local(plan(18)).fingerprint()
        );
    }

    #[test]
    fn intact_receipt_replays_exactly() {
        let receipt = execute_local(plan(19));
        let replay = replay_local(&receipt);
        assert!(replay.matches());
        assert_eq!(replay.expected_fingerprint(), receipt.fingerprint());
        assert_eq!(replay.observed(), &receipt);
    }

    #[test]
    fn altered_receipt_is_detected() {
        let mut receipt = execute_local(plan(23));
        receipt.candidate_fingerprints[0][0] ^= 1;
        let replay = replay_local(&receipt);
        assert!(!replay.matches());
        assert_ne!(replay.observed(), &receipt);
    }

    #[test]
    fn relation_encoding_is_structural_and_variant_complete() {
        let input = PointExpression::Input;
        let expressions = [
            input.clone(),
            PointExpression::Identity,
            PointExpression::Negate(Box::new(input.clone())),
            PointExpression::Double(Box::new(input.clone())),
            PointExpression::ScalarMultiply {
                scalar: 3,
                point: Box::new(input.clone()),
            },
            PointExpression::Add(Box::new(input.clone()), Box::new(input)),
        ];
        let relations = [
            Relation::PointEqual(expressions[0].clone(), expressions[1].clone()),
            Relation::IsInfinity(expressions[2].clone()),
            Relation::CurveAEquals(3),
            Relation::CurveJEquals(0),
        ];
        let mut fingerprints = std::collections::BTreeSet::new();
        for relation in relations
        {
            let mut encoder = CanonicalEncoder::with_domain(b"RELATION-ENCODING-TEST/V1");
            encode_relation(&mut encoder, &relation);
            fingerprints.insert(sha256(&encoder.finish()));
        }
        assert_eq!(fingerprints.len(), 4);
    }
}

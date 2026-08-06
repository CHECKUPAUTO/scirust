//! End-to-end orchestration for complete, local-only discovery campaigns.

use core::fmt::Write;

use crate::canonical::{CanonicalEncoder, hex, sha256};
use crate::execution::{
    candidate_fingerprint, encode_classification, encode_counterexample, receipt_from_results,
};
use crate::{
    ClassificationStatus, ControlId, ControlResult, ExecutionReceipt, LiteratureReview,
    ResearchCorpora, ReviewReport, ReviewedCandidate, SearchPlan, attempt_justification,
    review_candidate, run_control, run_search,
};

/// Canonical order of every control required by the research contract.
pub const MANDATORY_CONTROLS: [ControlId; 6] = [
    ControlId::TrueNegation,
    ControlId::FalseNegationKeepsY,
    ControlId::FalseDoublingSign,
    ControlId::JZeroClaimedUniversal,
    ControlId::EncodingSignClaimedNovel,
    ControlId::OverfitAZero,
];

const REPORT_STATUSES: [ClassificationStatus; 6] = [
    ClassificationStatus::Refuted,
    ClassificationStatus::Known,
    ClassificationStatus::RepresentationArtifact,
    ClassificationStatus::NeedsLiteratureReview,
    ClassificationStatus::Inconclusive,
    ClassificationStatus::CandidateUnclassified,
];

/// Complete result of one deterministic campaign over built-in toy corpora.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CampaignRun {
    receipt: ExecutionReceipt,
    controls: Vec<ControlResult>,
    candidates: Vec<ReviewedCandidate>,
}

impl CampaignRun {
    pub const SCHEMA_VERSION: u32 = 1;

    /// Validated local plan which fully determines this campaign.
    pub const fn plan(&self) -> SearchPlan {
        self.receipt.plan()
    }

    /// Phase-6-compatible receipt built from these exact candidate evaluations.
    pub const fn receipt(&self) -> &ExecutionReceipt {
        &self.receipt
    }

    /// Mandatory controls in their canonical order.
    pub fn controls(&self) -> &[ControlResult] {
        &self.controls
    }

    /// Evaluated, justified candidates with explicitly pending human reviews.
    pub fn candidates(&self) -> &[ReviewedCandidate] {
        &self.candidates
    }

    /// Whether every mandatory control has its exact expected outcome.
    pub fn controls_valid(&self) -> bool {
        self.controls.len() == MANDATORY_CONTROLS.len()
            && self
                .controls
                .iter()
                .zip(MANDATORY_CONTROLS)
                .all(|(result, id)| control_matches_expectation(result, id))
    }

    /// Stable binary representation of the complete campaign evidence.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoder = CanonicalEncoder::with_domain(b"SCIRUST-ELLIPTIC-DISCOVERY/CAMPAIGN/V1");
        encoder.u32(Self::SCHEMA_VERSION);
        encoder.bytes(&self.receipt.canonical_bytes());

        encoder.u64(u64::try_from(self.controls.len()).expect("control count fits in u64"));
        for control in &self.controls
        {
            encode_control(&mut encoder, control);
        }

        encoder.u64(u64::try_from(self.candidates.len()).expect("candidate count fits in u64"));
        for reviewed in &self.candidates
        {
            encoder.bytes(&candidate_fingerprint(reviewed.candidate()));
            encoder.bytes(&ReviewReport::new(reviewed).canonical_bytes());
        }
        encoder.finish()
    }

    /// SHA-256 integrity fingerprint of every campaign field.
    pub fn fingerprint(&self) -> [u8; 32] {
        sha256(&self.canonical_bytes())
    }
}

/// Result of recomputing a complete campaign from its closed local plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CampaignReplayReport {
    expected_fingerprint: [u8; 32],
    observed: CampaignRun,
    matches: bool,
}

impl CampaignReplayReport {
    /// Fingerprint supplied by the campaign being audited.
    pub const fn expected_fingerprint(&self) -> [u8; 32] {
        self.expected_fingerprint
    }

    /// Newly computed campaign retained even when replay diverges.
    pub const fn observed(&self) -> &CampaignRun {
        &self.observed
    }

    /// Whether every canonical campaign byte matched.
    pub const fn matches(&self) -> bool {
        self.matches
    }
}

/// Runs controls, search, justification and pending review as one local campaign.
pub fn execute_campaign(plan: SearchPlan) -> CampaignRun {
    let corpora = ResearchCorpora::generate(plan);
    let controls = MANDATORY_CONTROLS
        .into_iter()
        .map(|id| run_control(corpora.exhaustive_small(), id))
        .collect();
    let evaluations = run_search(plan, &corpora);
    let receipt = receipt_from_results(plan, &corpora, &evaluations);
    let candidates = evaluations
        .into_iter()
        .map(|candidate| {
            let justification = attempt_justification(&candidate);
            review_candidate(candidate, justification, LiteratureReview::pending())
        })
        .collect();
    CampaignRun {
        receipt,
        controls,
        candidates,
    }
}

/// Recomputes an entire campaign and reports any byte-level divergence.
pub fn replay_campaign(expected: &CampaignRun) -> CampaignReplayReport {
    let expected_bytes = expected.canonical_bytes();
    let expected_fingerprint = sha256(&expected_bytes);
    let observed = execute_campaign(expected.plan());
    let matches = expected_bytes == observed.canonical_bytes();
    CampaignReplayReport {
        expected_fingerprint,
        observed,
        matches,
    }
}

/// Deterministic Markdown view of a complete campaign.
pub struct CampaignReport<'a> {
    campaign: &'a CampaignRun,
}

impl<'a> CampaignReport<'a> {
    pub const fn new(campaign: &'a CampaignRun) -> Self {
        Self { campaign }
    }

    /// Stable UTF-8 report. The binary campaign remains the source of authority.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut output = String::new();
        writeln!(output, "# Elliptic discovery local campaign").expect("String write");
        writeln!(output).expect("String write");
        writeln!(
            output,
            "- Plan SHA-256: `{}`",
            hex(&self.campaign.plan().fingerprint())
        )
        .expect("String write");
        writeln!(
            output,
            "- Execution receipt SHA-256: `{}`",
            hex(&self.campaign.receipt.fingerprint())
        )
        .expect("String write");
        writeln!(
            output,
            "- Campaign SHA-256: `{}`",
            hex(&self.campaign.fingerprint())
        )
        .expect("String write");
        writeln!(
            output,
            "- Mandatory controls valid: {}",
            self.campaign.controls_valid()
        )
        .expect("String write");

        writeln!(output, "\n## Mandatory controls\n").expect("String write");
        for control in &self.campaign.controls
        {
            write_control(&mut output, control);
        }

        writeln!(output, "\n## Automated summary\n").expect("String write");
        let summary = self.campaign.receipt.summary();
        writeln!(output, "- Candidates: {}", summary.candidate_count()).expect("String write");
        writeln!(
            output,
            "- Counterexamples: {}",
            summary.counterexample_count()
        )
        .expect("String write");
        for status in REPORT_STATUSES
        {
            writeln!(output, "- {status:?}: {}", summary.count(status)).expect("String write");
        }

        writeln!(output, "\n## Candidate records\n").expect("String write");
        for (index, reviewed) in self.campaign.candidates.iter().enumerate()
        {
            writeln!(output, "### Candidate {index}\n").expect("String write");
            let report = ReviewReport::new(reviewed).canonical_bytes();
            let report = core::str::from_utf8(&report).expect("review reports are UTF-8");
            output.push_str(report);
            if !report.ends_with('\n')
            {
                output.push('\n');
            }
        }

        writeln!(output, "\n## Interpretation\n").expect("String write");
        writeln!(
            output,
            "This automated campaign records hypotheses only. It does not claim novelty or discovery."
        )
        .expect("String write");
        output.into_bytes()
    }

    /// Integrity fingerprint of the readable campaign report.
    pub fn fingerprint_hex(&self) -> String {
        hex(&sha256(&self.canonical_bytes()))
    }
}

fn control_matches_expectation(result: &ControlResult, id: ControlId) -> bool {
    let (expected_status, expects_counterexample) = match id
    {
        ControlId::TrueNegation => (ClassificationStatus::Known, false),
        ControlId::FalseNegationKeepsY
        | ControlId::FalseDoublingSign
        | ControlId::JZeroClaimedUniversal
        | ControlId::OverfitAZero => (ClassificationStatus::Refuted, true),
        ControlId::EncodingSignClaimedNovel =>
        {
            (ClassificationStatus::RepresentationArtifact, false)
        },
    };
    result.id() == id
        && result.status() == expected_status
        && result.counterexample().is_some() == expects_counterexample
}

fn encode_control(encoder: &mut CanonicalEncoder, control: &ControlResult) {
    encoder.u8(control_id_tag(control.id()));
    encode_classification(encoder, control.classification());
    match control.counterexample()
    {
        Some(counterexample) =>
        {
            encoder.u8(1);
            encode_counterexample(encoder, counterexample);
        },
        None => encoder.u8(0),
    }
}

const fn control_id_tag(id: ControlId) -> u8 {
    match id
    {
        ControlId::TrueNegation => 0,
        ControlId::FalseNegationKeepsY => 1,
        ControlId::FalseDoublingSign => 2,
        ControlId::JZeroClaimedUniversal => 3,
        ControlId::EncodingSignClaimedNovel => 4,
        ControlId::OverfitAZero => 5,
    }
}

fn write_control(output: &mut String, control: &ControlResult) {
    write!(output, "- {:?}: {:?}", control.id(), control.status()).expect("String write");
    if let Some(counterexample) = control.counterexample()
    {
        let (prime, a, b) = counterexample.curve_key();
        write!(
            output,
            "; first counterexample p={prime}, a={a}, b={b}, point_index={}",
            counterexample.point_index()
        )
        .expect("String write");
    }
    writeln!(output).expect("String write");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LiteratureDecision, execute_local};

    fn plan(seed: u64) -> SearchPlan {
        SearchPlan::new(seed, 2, 3, 8, 1_000_000, 1).expect("bounded campaign plan")
    }

    #[test]
    fn repeated_campaigns_are_byte_identical() {
        let left = execute_campaign(plan(31));
        let right = execute_campaign(plan(31));
        assert_eq!(left.canonical_bytes(), right.canonical_bytes());
        assert_eq!(left.fingerprint(), right.fingerprint());
        assert_eq!(left.candidates().len(), 8);
    }

    #[test]
    fn campaign_runs_all_controls_and_reuses_exact_search_results() {
        let campaign = execute_campaign(plan(37));
        assert!(campaign.controls_valid());
        assert_eq!(campaign.controls().len(), MANDATORY_CONTROLS.len());
        assert_eq!(campaign.receipt(), &execute_local(plan(37)));
        assert_eq!(
            campaign.candidates().len(),
            usize::try_from(campaign.receipt().summary().candidate_count())
                .expect("candidate count fits in usize")
        );
        assert!(campaign.candidates().iter().all(|reviewed| {
            reviewed.literature_review().decision() == LiteratureDecision::Pending
                && reviewed.final_status() != ClassificationStatus::CandidateUnclassified
        }));
    }

    #[test]
    fn different_seeds_change_campaign_identity() {
        assert_ne!(
            execute_campaign(plan(41)).fingerprint(),
            execute_campaign(plan(42)).fingerprint()
        );
    }

    #[test]
    fn campaign_replay_detects_removed_candidate() {
        let intact = execute_campaign(plan(43));
        assert!(replay_campaign(&intact).matches());

        let mut altered = intact;
        altered.candidates.pop();
        let replay = replay_campaign(&altered);
        assert!(!replay.matches());
        assert_eq!(replay.observed().candidates().len(), 8);
    }

    #[test]
    fn readable_report_contains_every_evidence_boundary() {
        let campaign = execute_campaign(plan(47));
        let report = CampaignReport::new(&campaign);
        let repeated = CampaignReport::new(&campaign);
        assert_eq!(report.canonical_bytes(), repeated.canonical_bytes());
        let text = String::from_utf8(report.canonical_bytes()).expect("UTF-8 campaign report");
        assert!(text.contains("## Mandatory controls"));
        assert!(text.contains("## Automated summary"));
        assert!(text.contains("## Candidate records"));
        assert!(text.contains("## Coverage gates"));
        assert!(text.contains("## Exact justification"));
        assert!(text.contains("does not claim novelty or discovery"));
    }
}

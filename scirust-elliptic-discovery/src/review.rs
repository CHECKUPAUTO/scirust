//! Human literature-review boundary and readable research reports.

use core::fmt::{self, Write};
use std::collections::BTreeSet;

use crate::canonical::{hex, sha256};
use crate::{
    CandidateEvaluation, ClassificationStatus, GateState, Justification, ProofCertificate,
};

/// Human decision recorded after an independent literature search.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LiteratureDecision {
    Pending,
    Known,
    NoConflictFound,
}

/// Auditable literature-review record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LiteratureReview {
    decision: LiteratureDecision,
    reviewer: String,
    sources: BTreeSet<String>,
}

impl LiteratureReview {
    /// Creates an explicitly pending review.
    pub fn pending() -> Self {
        Self {
            decision: LiteratureDecision::Pending,
            reviewer: String::new(),
            sources: BTreeSet::new(),
        }
    }

    /// Records a completed review. Reviewer and sources must be nonempty.
    pub fn completed(
        decision: LiteratureDecision,
        reviewer: impl Into<String>,
        sources: impl IntoIterator<Item = String>,
    ) -> Result<Self, ReviewError> {
        if decision == LiteratureDecision::Pending
        {
            return Err(ReviewError::PendingCannotBeCompleted);
        }
        let reviewer = reviewer.into();
        if reviewer.trim().is_empty()
        {
            return Err(ReviewError::MissingReviewer);
        }
        let sources: BTreeSet<_> = sources
            .into_iter()
            .filter(|source| !source.trim().is_empty())
            .collect();
        if sources.is_empty()
        {
            return Err(ReviewError::MissingSources);
        }
        Ok(Self {
            decision,
            reviewer,
            sources,
        })
    }
}

/// Invalid human-review record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReviewError {
    PendingCannotBeCompleted,
    MissingReviewer,
    MissingSources,
}

impl fmt::Display for ReviewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::PendingCannotBeCompleted =>
            {
                write!(formatter, "pending is not a completed literature decision")
            },
            Self::MissingReviewer => write!(formatter, "completed review requires a reviewer"),
            Self::MissingSources => write!(formatter, "completed review requires sources"),
        }
    }
}

impl std::error::Error for ReviewError {}

/// Final conservative record. CandidateUnclassified is still not a discovery claim.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReviewedCandidate {
    candidate: CandidateEvaluation,
    justification: Justification,
    literature_review: LiteratureReview,
    final_status: ClassificationStatus,
}

impl ReviewedCandidate {
    pub const fn candidate(&self) -> &CandidateEvaluation {
        &self.candidate
    }

    pub const fn justification(&self) -> &Justification {
        &self.justification
    }

    pub const fn literature_review(&self) -> &LiteratureReview {
        &self.literature_review
    }

    pub const fn final_status(&self) -> ClassificationStatus {
        self.final_status
    }
}

/// Applies the human-review boundary without ever creating a novelty status.
pub fn review_candidate(
    candidate: CandidateEvaluation,
    justification: Justification,
    literature_review: LiteratureReview,
) -> ReviewedCandidate {
    let automated = candidate.classification().status();
    let final_status = match automated
    {
        ClassificationStatus::Refuted
        | ClassificationStatus::Known
        | ClassificationStatus::RepresentationArtifact
        | ClassificationStatus::Inconclusive => automated,
        ClassificationStatus::NeedsLiteratureReview
        | ClassificationStatus::CandidateUnclassified => match literature_review.decision
        {
            LiteratureDecision::Pending => ClassificationStatus::NeedsLiteratureReview,
            LiteratureDecision::Known => ClassificationStatus::Known,
            LiteratureDecision::NoConflictFound
                if proof_was_attempted(&justification)
                    && candidate
                        .gates()
                        .iter()
                        .all(|gate| gate.state() == GateState::Passed) =>
            {
                ClassificationStatus::CandidateUnclassified
            },
            LiteratureDecision::NoConflictFound => ClassificationStatus::Inconclusive,
        },
    };
    ReviewedCandidate {
        candidate,
        justification,
        literature_review,
        final_status,
    }
}

fn proof_was_attempted(justification: &Justification) -> bool {
    matches!(
        justification,
        Justification::Catalog(_) | Justification::Proved(_) | Justification::NoCertificate { .. }
    )
}

/// Stable readable report separating evidence categories.
pub struct ReviewReport<'a> {
    reviewed: &'a ReviewedCandidate,
}

impl<'a> ReviewReport<'a> {
    pub const fn new(reviewed: &'a ReviewedCandidate) -> Self {
        Self { reviewed }
    }

    /// Deterministic Markdown report.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut output = String::new();
        writeln!(output, "# Elliptic discovery candidate review").expect("String write");
        writeln!(output).expect("String write");
        writeln!(output, "- Final status: '{:?}'", self.reviewed.final_status)
            .expect("String write");
        writeln!(
            output,
            "- Automated status: '{:?}'",
            self.reviewed.candidate.classification().status()
        )
        .expect("String write");
        writeln!(
            output,
            "- Relation: '{:?}'",
            self.reviewed.candidate.relation()
        )
        .expect("String write");

        writeln!(output, "\n## Coverage gates\n").expect("String write");
        for gate in self.reviewed.candidate.gates()
        {
            writeln!(
                output,
                "- {}: {:?}; evaluated {}/{} tuples",
                gate.corpus().name(),
                gate.state(),
                gate.evaluated_tuples(),
                gate.required_tuples()
            )
            .expect("String write");
        }

        writeln!(output, "\n## Counterexample\n").expect("String write");
        match self.reviewed.candidate.counterexample()
        {
            Some(counterexample) =>
            {
                let (prime, a, b) = counterexample.curve_key();
                writeln!(
                    output,
                    "First canonical counterexample: p={prime}, a={a}, b={b}, point_index={}.",
                    counterexample.point_index()
                )
                .expect("String write");
            },
            None => writeln!(
                output,
                "No counterexample was found within declared coverage."
            )
            .expect("String write"),
        }

        writeln!(output, "\n## Known-property catalog\n").expect("String write");
        match self.reviewed.candidate.classification().catalog()
        {
            Some(entry) => writeln!(
                output,
                "Matched '{}'. Reference: {}.",
                entry.id, entry.reference
            )
            .expect("String write"),
            None => writeln!(output, "No automated catalog match.").expect("String write"),
        }

        writeln!(output, "\n## Exact justification\n").expect("String write");
        write_justification(&mut output, &self.reviewed.justification);

        writeln!(output, "\n## Literature review\n").expect("String write");
        writeln!(
            output,
            "- Decision: '{:?}'",
            self.reviewed.literature_review.decision
        )
        .expect("String write");
        if !self.reviewed.literature_review.reviewer.is_empty()
        {
            writeln!(
                output,
                "- Reviewer: {}",
                self.reviewed.literature_review.reviewer
            )
            .expect("String write");
        }
        for source in &self.reviewed.literature_review.sources
        {
            writeln!(output, "- Source: {source}").expect("String write");
        }

        writeln!(output, "\n## Interpretation\n").expect("String write");
        writeln!(
            output,
            "CandidateUnclassified, when present, denotes a hypothesis requiring independent \
             mathematical scrutiny. It is not a claim of novelty or discovery."
        )
        .expect("String write");
        output.into_bytes()
    }

    /// Integrity fingerprint for the readable report.
    pub fn fingerprint_hex(&self) -> String {
        hex(&sha256(&self.canonical_bytes()))
    }
}

fn write_justification(output: &mut String, justification: &Justification) {
    match justification
    {
        Justification::Catalog(entry) =>
        {
            writeln!(output, "Catalog justification: '{}'.", entry.id).expect("String write");
        },
        Justification::Proved(ProofCertificate::GroupModuleIdentity { coefficient }) =>
        {
            writeln!(
                output,
                "Verified group-module identity with coefficient {coefficient}."
            )
            .expect("String write");
        },
        Justification::Proved(ProofCertificate::GroupIdentity { coefficient }) =>
        {
            writeln!(
                output,
                "Verified normalization to the identity with coefficient {coefficient}."
            )
            .expect("String write");
        },
        Justification::Proved(ProofCertificate::PolynomialIdentity(certificate)) =>
        {
            writeln!(
                output,
                "Verified polynomial identity over F_{}: {}.",
                certificate.prime().value(),
                certificate.verify()
            )
            .expect("String write");
        },
        Justification::NoCertificate { reason } =>
        {
            writeln!(output, "No exact certificate: {reason}.").expect("String write");
        },
        Justification::NotEligible { reason } =>
        {
            writeln!(output, "Proof attempt not eligible: {reason}.").expect("String write");
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        PointExpression, Relation, ResearchCorpora, SearchPlan, attempt_justification,
        evaluate_candidate,
    };

    #[test]
    fn candidate_status_requires_completed_review_and_proof_attempt() {
        let plan = SearchPlan::new(11, 1, 2, 1, 1_000_000, 1).expect("bounded plan");
        let corpora = ResearchCorpora::generate(plan);
        let candidate = evaluate_candidate(
            Relation::PointEqual(PointExpression::Input, PointExpression::Input),
            &corpora,
            plan.tuple_budget_per_candidate(),
        );
        let justification = attempt_justification(&candidate);
        let pending = review_candidate(
            candidate.clone(),
            justification.clone(),
            LiteratureReview::pending(),
        );
        assert_eq!(
            pending.final_status(),
            ClassificationStatus::NeedsLiteratureReview
        );

        let review = LiteratureReview::completed(
            LiteratureDecision::NoConflictFound,
            "independent reviewer",
            ["Silverman, Arithmetic of Elliptic Curves".to_string()],
        )
        .expect("complete review");
        let reviewed = review_candidate(candidate, justification, review);
        assert_eq!(
            reviewed.final_status(),
            ClassificationStatus::CandidateUnclassified
        );
        let report = ReviewReport::new(&reviewed);
        let text = String::from_utf8(report.canonical_bytes()).expect("UTF-8 report");
        assert!(text.contains("## Counterexample"));
        assert!(text.contains("## Known-property catalog"));
        assert!(text.contains("## Exact justification"));
        assert!(text.contains("not a claim of novelty or discovery"));
    }
}

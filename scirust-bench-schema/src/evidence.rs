//! Scientific evidence classification for benchmark and research artifacts.
//!
//! This module classifies what a piece of evidence *is* separately from what
//! happened to the evaluated claim. That distinction prevents a rejected or
//! inconclusive hypothesis from disappearing merely because it did not support
//! the hoped-for result.

use serde::{Deserialize, Serialize};

/// Scientific status of an evidence item.
///
/// These variants deliberately distinguish mathematical, numerical, empirical
/// and model-level claims. They are not an ordinal scale and must not be
/// collapsed into a boolean such as `validated`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScientificEvidenceKind {
    /// Direct measurement or experiment against an external/real system.
    EmpiricalValidation,
    /// A computed result whose meaning depends on a declared numerical approximation.
    NumericalApproximation,
    /// A result established exactly from its declared mathematical assumptions.
    ExactMathematicalResult,
    /// A model introduced to describe observed behavior without claiming fundamental truth.
    PhenomenologicalModel,
    /// A proposed model or mechanism not yet independently validated.
    SpeculativeModel,
    /// A predeclared criterion whose purpose is to reject or falsify a candidate claim/model.
    RejectionCriterion,
}

/// Outcome of evaluating a claim or criterion with one evidence item.
///
/// `Rejects` is first-class evidence: it means the evaluated claim or candidate
/// failed under the recorded evidence. It never means the record should be
/// discarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceDisposition {
    /// The evidence supports the specifically stated claim under its declared conditions.
    Supports,
    /// The evidence rejects the specifically stated claim/candidate under its declared conditions.
    Rejects,
    /// The evidence does not resolve the claim either way.
    Inconclusive,
    /// The evidence item records a definition/result for which support-vs-rejection is not applicable.
    NotApplicable,
}

/// Machine-readable scientific interpretation attached to an artifact/measurement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScientificEvidence {
    /// Scientific kind of this evidence item.
    pub kind: ScientificEvidenceKind,
    /// Outcome with respect to the stated claim or criterion.
    pub disposition: EvidenceDisposition,
    /// Non-empty statement identifying the claim, result, hypothesis or criterion being classified.
    pub statement: String,
}

impl ScientificEvidence {
    /// Construct a scientific evidence classification.
    ///
    /// Empty or whitespace-only statements are rejected so a negative or
    /// inconclusive result cannot be stored without saying what was evaluated.
    pub fn new(
        kind: ScientificEvidenceKind,
        disposition: EvidenceDisposition,
        statement: impl Into<String>,
    ) -> Result<Self, String> {
        let statement = statement.into();
        if statement.trim().is_empty()
        {
            return Err("scientific evidence statement must not be empty".to_owned());
        }
        Ok(Self {
            kind,
            disposition,
            statement,
        })
    }

    /// Return whether this evidence explicitly records a rejected claim/candidate.
    #[must_use]
    pub const fn is_negative_result(&self) -> bool {
        matches!(self.disposition, EvidenceDisposition::Rejects)
    }
}

#[cfg(test)]
mod tests {
    use super::{EvidenceDisposition, ScientificEvidence, ScientificEvidenceKind};

    #[test]
    fn every_required_kind_round_trips_through_json() {
        let kinds = [
            ScientificEvidenceKind::EmpiricalValidation,
            ScientificEvidenceKind::NumericalApproximation,
            ScientificEvidenceKind::ExactMathematicalResult,
            ScientificEvidenceKind::PhenomenologicalModel,
            ScientificEvidenceKind::SpeculativeModel,
            ScientificEvidenceKind::RejectionCriterion,
        ];

        for kind in kinds
        {
            let evidence = ScientificEvidence::new(
                kind,
                EvidenceDisposition::NotApplicable,
                "fixture statement",
            )
            .unwrap();
            let json = serde_json::to_string(&evidence).unwrap();
            let decoded: ScientificEvidence = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, evidence);
        }
    }

    #[test]
    fn rejected_hypothesis_is_retained_as_first_class_evidence() {
        let evidence = ScientificEvidence::new(
            ScientificEvidenceKind::NumericalApproximation,
            EvidenceDisposition::Rejects,
            "bounded history improves endpoint error over complete history",
        )
        .unwrap();

        assert!(evidence.is_negative_result());
        let json = serde_json::to_string(&evidence).unwrap();
        assert!(json.contains("\"disposition\":\"rejects\""));
        assert!(json.contains("bounded history improves endpoint error"));
    }

    #[test]
    fn empty_statement_fails_closed() {
        assert!(
            ScientificEvidence::new(
                ScientificEvidenceKind::SpeculativeModel,
                EvidenceDisposition::Inconclusive,
                "   ",
            )
            .is_err()
        );
    }
}

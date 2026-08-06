use crate::{FeatureHypothesis, ProposalReview, RankedHypothesis};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalOrigin {
    SciAgent,
    CognoModel,
    ImportedLocalData,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustClass {
    ValidatedLocalData,
    UntrustedModelData,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuarantineState {
    Proposed,
    DevelopmentValidated,
    ValidationValidated,
    ConfirmationEligible,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceReference {
    pub id: String,
    pub sha256: String,
    pub split: String,
}

impl EvidenceReference {
    fn validate(&self) -> Result<(), String> {
        if self.id.trim().is_empty() {
            return Err("evidence id must not be empty".to_string());
        }
        if self.sha256.len() != 64 || !self.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!("evidence `{}` has an invalid sha256", self.id));
        }
        if self.split.trim().is_empty() {
            return Err(format!("evidence `{}` has an empty split", self.id));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognoFeatureProposal {
    pub schema_version: u16,
    pub proposal_id: String,
    pub origin: ProposalOrigin,
    pub trust_class: TrustClass,
    pub confidence_bps: u16,
    pub hypothesis: FeatureHypothesis,
    pub evidence: Vec<EvidenceReference>,
    pub quarantine_state: QuarantineState,
    pub capability_ids: Vec<String>,
}

impl CognoFeatureProposal {
    pub fn validate(&self, maximum_evidence: usize) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err(format!(
                "unsupported COGNO feature proposal schema {}",
                self.schema_version
            ));
        }
        if self.proposal_id.trim().is_empty() {
            return Err("proposal_id must not be empty".to_string());
        }
        if self.confidence_bps > 10_000 {
            return Err(format!(
                "proposal `{}` confidence exceeds 10_000 bps",
                self.proposal_id
            ));
        }
        if self.trust_class != TrustClass::UntrustedModelData {
            return Err(format!(
                "proposal `{}` must enter as untrusted model data",
                self.proposal_id
            ));
        }
        if self.quarantine_state != QuarantineState::Proposed {
            return Err(format!(
                "proposal `{}` must enter quarantine in proposed state",
                self.proposal_id
            ));
        }
        if !self.capability_ids.is_empty() {
            return Err(format!(
                "proposal `{}` must not request runtime capabilities",
                self.proposal_id
            ));
        }
        if self.evidence.len() > maximum_evidence {
            return Err(format!(
                "proposal `{}` exceeds the evidence bound",
                self.proposal_id
            ));
        }
        let mut evidence_ids = BTreeSet::new();
        for evidence in &self.evidence {
            evidence.validate()?;
            if !evidence_ids.insert(evidence.id.as_str()) {
                return Err(format!(
                    "proposal `{}` contains duplicate evidence id `{}`",
                    self.proposal_id, evidence.id
                ));
            }
        }
        self.hypothesis.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognoProposalBatch {
    pub schema_version: u16,
    pub experiment_id: String,
    pub proposals: Vec<CognoFeatureProposal>,
    pub maximum_evidence_per_proposal: usize,
}

impl CognoProposalBatch {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err(format!(
                "unsupported COGNO proposal batch schema {}",
                self.schema_version
            ));
        }
        if self.experiment_id.trim().is_empty() {
            return Err("experiment_id must not be empty".to_string());
        }
        if self.maximum_evidence_per_proposal == 0 {
            return Err("maximum_evidence_per_proposal must be positive".to_string());
        }
        let mut proposal_ids = BTreeSet::new();
        for proposal in &self.proposals {
            proposal.validate(self.maximum_evidence_per_proposal)?;
            if !proposal_ids.insert(proposal.proposal_id.as_str()) {
                return Err(format!(
                    "duplicate proposal id `{}`",
                    proposal.proposal_id
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognoAdoptionRecord {
    pub schema_version: u16,
    pub proposal_id: String,
    pub hypothesis_id: String,
    pub decision: QuarantineState,
    pub deterministic_score: f64,
    pub reasons: Vec<String>,
}

pub fn build_cogno_adoption_records(
    batch: &CognoProposalBatch,
    review: &ProposalReview,
) -> Result<Vec<CognoAdoptionRecord>, String> {
    batch.validate()?;
    let accepted: std::collections::BTreeMap<&str, &RankedHypothesis> = review
        .accepted
        .iter()
        .map(|item| (item.hypothesis.id.as_str(), item))
        .collect();
    let rejected: std::collections::BTreeMap<&str, &str> = review
        .rejected
        .iter()
        .map(|item| (item.id.as_str(), item.reason.as_str()))
        .collect();

    let mut records = Vec::with_capacity(batch.proposals.len());
    for proposal in &batch.proposals {
        if let Some(item) = accepted.get(proposal.hypothesis.id.as_str()) {
            records.push(CognoAdoptionRecord {
                schema_version: 1,
                proposal_id: proposal.proposal_id.clone(),
                hypothesis_id: proposal.hypothesis.id.clone(),
                decision: QuarantineState::DevelopmentValidated,
                deterministic_score: item.score,
                reasons: vec![
                    "structurally valid".to_string(),
                    "runtime safe".to_string(),
                    "within declared budget".to_string(),
                    "accepted for development evaluation only".to_string(),
                ],
            });
        } else {
            records.push(CognoAdoptionRecord {
                schema_version: 1,
                proposal_id: proposal.proposal_id.clone(),
                hypothesis_id: proposal.hypothesis.id.clone(),
                decision: QuarantineState::Rejected,
                deterministic_score: 0.0,
                reasons: vec![rejected
                    .get(proposal.hypothesis.id.as_str())
                    .copied()
                    .unwrap_or("proposal not present in review")
                    .to_string()],
            });
        }
    }
    records.sort_by(|left, right| left.proposal_id.cmp(&right.proposal_id));
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FeatureFamily, RuntimeCost, TemporalAvailability};

    fn hypothesis(id: &str) -> FeatureHypothesis {
        FeatureHypothesis {
            id: id.to_string(),
            name: id.to_string(),
            family: FeatureFamily::Stability,
            expression: "abs(drift)".to_string(),
            required_signals: vec!["drift".to_string()],
            temporal_availability: TemporalAvailability::CurrentDecision,
            runtime_cost: RuntimeCost::constant(1),
            rationale: "runtime instability".to_string(),
            expected_failure_mode: "prediction divergence".to_string(),
            ablation_group: "cogno".to_string(),
            deterministic: true,
        }
    }

    #[test]
    fn rejects_capability_requests() {
        let proposal = CognoFeatureProposal {
            schema_version: 1,
            proposal_id: "p1".to_string(),
            origin: ProposalOrigin::CognoModel,
            trust_class: TrustClass::UntrustedModelData,
            confidence_bps: 8_000,
            hypothesis: hypothesis("h1"),
            evidence: Vec::new(),
            quarantine_state: QuarantineState::Proposed,
            capability_ids: vec!["shell".to_string()],
        };
        assert!(proposal.validate(4).is_err());
    }

    #[test]
    fn accepts_bounded_quarantined_model_proposal() {
        let proposal = CognoFeatureProposal {
            schema_version: 1,
            proposal_id: "p1".to_string(),
            origin: ProposalOrigin::CognoModel,
            trust_class: TrustClass::UntrustedModelData,
            confidence_bps: 8_000,
            hypothesis: hypothesis("h1"),
            evidence: vec![EvidenceReference {
                id: "dataset".to_string(),
                sha256: "0".repeat(64),
                split: "development_train".to_string(),
            }],
            quarantine_state: QuarantineState::Proposed,
            capability_ids: Vec::new(),
        };
        assert!(proposal.validate(4).is_ok());
    }
}

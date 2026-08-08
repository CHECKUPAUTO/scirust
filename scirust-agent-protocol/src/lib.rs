#![forbid(unsafe_code)]

mod execution_attestation;

pub use execution_attestation::{
    EXECUTION_PROFILE_SCHEMA_VERSION, ExecutionArchitecture, ExecutionArchitectureFamily,
    ExecutionAttestation, ExecutionAttestationError, ExecutionBackendKind, ExecutionProfile,
    ExecutionReproducibility, Sha256Digest,
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

pub const SCHEMA_VERSION: u16 = 1;
pub const MAX_CONFIDENCE_BPS: u16 = 10_000;
pub const MAX_RECIPIENTS: usize = 16;
pub const MAX_EVIDENCE_REFERENCES: usize = 32;
pub const MAX_REQUESTED_CAPABILITIES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentKind {
    Human,
    ExternalModel,
    SciAgent,
    CognoObserver,
    CognoCommunicator,
    RuntimeDiscovery,
    DeterministicKernel,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AgentIdentity {
    pub kind: AgentKind,
    pub id: String,
}

impl AgentIdentity {
    fn validate(&self) -> Result<(), ProtocolError> {
        if self.id.trim().is_empty()
        {
            return Err(ProtocolError::EmptyAgentId);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageKind {
    Question,
    Hypothesis,
    Critique,
    Counterexample,
    ExperimentRequest,
    ExperimentResult,
    PreferenceObservation,
    Contradiction,
    Explanation,
    Decision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustClass {
    HumanProvided,
    ValidatedLocalData,
    UntrustedModelOutput,
    DeterministicKernelOutput,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceReference {
    pub id: String,
    pub sha256: String,
    pub split: String,
}

impl EvidenceReference {
    fn validate(&self) -> Result<(), ProtocolError> {
        if self.id.trim().is_empty()
        {
            return Err(ProtocolError::InvalidEvidence(
                "empty evidence id".to_string(),
            ));
        }
        if self.sha256.len() != 64 || !self.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(ProtocolError::InvalidEvidence(format!(
                "evidence `{}` has an invalid sha256",
                self.id
            )));
        }
        if self.split.trim().is_empty()
        {
            return Err(ProtocolError::InvalidEvidence(format!(
                "evidence `{}` has an empty split",
                self.id
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityRequest {
    pub capability_id: String,
    pub rationale: String,
}

impl CapabilityRequest {
    fn validate(&self) -> Result<(), ProtocolError> {
        if self.capability_id.trim().is_empty()
        {
            return Err(ProtocolError::InvalidCapability(
                "empty capability id".to_string(),
            ));
        }
        if self.rationale.trim().is_empty()
        {
            return Err(ProtocolError::InvalidCapability(format!(
                "capability `{}` has an empty rationale",
                self.capability_id
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentMessage {
    pub schema_version: u16,
    pub message_id: String,
    pub conversation_id: String,
    pub parent_message_id: Option<String>,
    pub sender: AgentIdentity,
    pub recipients: Vec<AgentIdentity>,
    pub message_kind: MessageKind,
    pub trust_class: TrustClass,
    pub confidence_bps: u16,
    pub evidence: Vec<EvidenceReference>,
    pub payload: Value,
    pub requested_capabilities: Vec<CapabilityRequest>,
}

impl AgentMessage {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.schema_version != SCHEMA_VERSION
        {
            return Err(ProtocolError::UnsupportedSchema(self.schema_version));
        }
        if self.message_id.trim().is_empty()
        {
            return Err(ProtocolError::EmptyMessageId);
        }
        if self.conversation_id.trim().is_empty()
        {
            return Err(ProtocolError::EmptyConversationId);
        }
        if self.confidence_bps > MAX_CONFIDENCE_BPS
        {
            return Err(ProtocolError::ConfidenceOutOfRange(self.confidence_bps));
        }
        self.sender.validate()?;
        if self.recipients.is_empty() || self.recipients.len() > MAX_RECIPIENTS
        {
            return Err(ProtocolError::InvalidRecipientCount(self.recipients.len()));
        }
        let mut recipient_ids = BTreeSet::new();
        for recipient in &self.recipients
        {
            recipient.validate()?;
            let key = (recipient.kind, recipient.id.as_str());
            if !recipient_ids.insert(key)
            {
                return Err(ProtocolError::DuplicateRecipient(recipient.id.clone()));
            }
        }
        if self.evidence.len() > MAX_EVIDENCE_REFERENCES
        {
            return Err(ProtocolError::TooManyEvidenceReferences(
                self.evidence.len(),
            ));
        }
        let mut evidence_ids = BTreeSet::new();
        for evidence in &self.evidence
        {
            evidence.validate()?;
            if !evidence_ids.insert(evidence.id.as_str())
            {
                return Err(ProtocolError::DuplicateEvidence(evidence.id.clone()));
            }
        }
        if self.requested_capabilities.len() > MAX_REQUESTED_CAPABILITIES
        {
            return Err(ProtocolError::TooManyCapabilities(
                self.requested_capabilities.len(),
            ));
        }
        let mut capability_ids = BTreeSet::new();
        for capability in &self.requested_capabilities
        {
            capability.validate()?;
            if !capability_ids.insert(capability.capability_id.as_str())
            {
                return Err(ProtocolError::DuplicateCapability(
                    capability.capability_id.clone(),
                ));
            }
        }
        self.validate_sender_policy()
    }

    fn validate_sender_policy(&self) -> Result<(), ProtocolError> {
        match self.sender.kind
        {
            AgentKind::CognoObserver =>
            {
                if !matches!(
                    self.message_kind,
                    MessageKind::PreferenceObservation | MessageKind::Contradiction
                )
                {
                    return Err(ProtocolError::CognoObserverMayNotSpeak);
                }
                if !self.requested_capabilities.is_empty()
                {
                    return Err(ProtocolError::CognoObserverMayNotRequestCapabilities);
                }
                if self
                    .recipients
                    .iter()
                    .any(|recipient| !matches!(recipient.kind, AgentKind::DeterministicKernel))
                {
                    return Err(ProtocolError::CognoObserverMustRemainInternal);
                }
            },
            AgentKind::ExternalModel | AgentKind::SciAgent | AgentKind::CognoCommunicator =>
            {
                if self.trust_class != TrustClass::UntrustedModelOutput
                {
                    return Err(ProtocolError::ModelOutputMustRemainUntrusted);
                }
            },
            AgentKind::RuntimeDiscovery | AgentKind::DeterministicKernel =>
            {
                if self.trust_class != TrustClass::DeterministicKernelOutput
                {
                    return Err(ProtocolError::KernelOutputTrustMismatch);
                }
            },
            AgentKind::Human =>
            {},
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    UnsupportedSchema(u16),
    EmptyMessageId,
    EmptyConversationId,
    EmptyAgentId,
    ConfidenceOutOfRange(u16),
    InvalidRecipientCount(usize),
    DuplicateRecipient(String),
    InvalidEvidence(String),
    DuplicateEvidence(String),
    TooManyEvidenceReferences(usize),
    InvalidCapability(String),
    DuplicateCapability(String),
    TooManyCapabilities(usize),
    CognoObserverMayNotSpeak,
    CognoObserverMayNotRequestCapabilities,
    CognoObserverMustRemainInternal,
    ModelOutputMustRemainUntrusted,
    KernelOutputTrustMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn identity(kind: AgentKind, id: &str) -> AgentIdentity {
        AgentIdentity {
            kind,
            id: id.to_string(),
        }
    }

    fn message(sender: AgentKind, kind: MessageKind) -> AgentMessage {
        AgentMessage {
            schema_version: SCHEMA_VERSION,
            message_id: "m-1".to_string(),
            conversation_id: "c-1".to_string(),
            parent_message_id: None,
            sender: identity(sender, "sender"),
            recipients: vec![identity(AgentKind::DeterministicKernel, "kernel")],
            message_kind: kind,
            trust_class: if matches!(
                sender,
                AgentKind::RuntimeDiscovery | AgentKind::DeterministicKernel
            )
            {
                TrustClass::DeterministicKernelOutput
            }
            else if sender == AgentKind::Human
            {
                TrustClass::HumanProvided
            }
            else
            {
                TrustClass::UntrustedModelOutput
            },
            confidence_bps: 5_000,
            evidence: Vec::new(),
            payload: json!({"text": "test"}),
            requested_capabilities: Vec::new(),
        }
    }

    #[test]
    fn sciagent_can_send_hypotheses() {
        assert!(
            message(AgentKind::SciAgent, MessageKind::Hypothesis)
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn cogno_observer_cannot_send_explanations() {
        assert_eq!(
            message(AgentKind::CognoObserver, MessageKind::Explanation).validate(),
            Err(ProtocolError::CognoObserverMayNotSpeak)
        );
    }

    #[test]
    fn cogno_observer_can_record_internal_contradictions() {
        assert!(
            message(AgentKind::CognoObserver, MessageKind::Contradiction)
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn model_output_never_claims_kernel_trust() {
        let mut value = message(AgentKind::ExternalModel, MessageKind::Critique);
        value.trust_class = TrustClass::DeterministicKernelOutput;
        assert_eq!(
            value.validate(),
            Err(ProtocolError::ModelOutputMustRemainUntrusted)
        );
    }
}

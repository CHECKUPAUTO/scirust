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
    /// Optional semantic execution attestation produced by a trusted local
    /// runtime/discrete kernel path. The field is additive within schema v1:
    /// historical messages deserialize with `None`, and `None` is omitted when
    /// serializing so their wire shape remains unchanged.
    ///
    /// A message carrying this field cannot be accepted through [`Self::validate`]
    /// alone because `sender` is self-declared wire data. Callers must use
    /// [`Self::validate_with_authenticated_sender`] after their transport/runtime
    /// has authenticated the sender identity independently of this message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_attestation: Option<ExecutionAttestation>,
}

impl AgentMessage {
    /// Validate a message when no independently authenticated sender identity is
    /// available.
    ///
    /// Execution attestations fail closed on this path even when the wire-level
    /// `sender.kind` claims to be a deterministic runtime. A self-declared sender
    /// must never bootstrap deterministic-kernel trust.
    pub fn validate(&self) -> Result<(), ProtocolError> {
        self.validate_base()?;
        self.validate_execution_attestation(None)
    }

    /// Validate a message against a sender identity authenticated by the calling
    /// transport/runtime boundary.
    ///
    /// The authenticated identity must exactly match the serialized sender. An
    /// attached execution attestation is then permitted only for authenticated
    /// `RuntimeDiscovery` or `DeterministicKernel` identities and its fingerprint
    /// is recomputed before acceptance.
    pub fn validate_with_authenticated_sender(
        &self,
        authenticated_sender: &AgentIdentity,
    ) -> Result<(), ProtocolError> {
        self.validate_base()?;
        if authenticated_sender != &self.sender
        {
            return Err(ProtocolError::AuthenticatedSenderMismatch);
        }
        self.validate_execution_attestation(Some(authenticated_sender))
    }

    fn validate_base(&self) -> Result<(), ProtocolError> {
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

    fn validate_execution_attestation(
        &self,
        authenticated_sender: Option<&AgentIdentity>,
    ) -> Result<(), ProtocolError> {
        let Some(attestation) = &self.execution_attestation
        else
        {
            return Ok(());
        };
        let Some(authenticated_sender) = authenticated_sender
        else
        {
            return Err(ProtocolError::ExecutionAttestationRequiresAuthenticatedSender);
        };

        if !matches!(
            authenticated_sender.kind,
            AgentKind::RuntimeDiscovery | AgentKind::DeterministicKernel
        )
        {
            return Err(ProtocolError::ExecutionAttestationSenderMismatch(
                authenticated_sender.kind,
            ));
        }

        attestation
            .verify()
            .map_err(ProtocolError::InvalidExecutionAttestation)
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
    AuthenticatedSenderMismatch,
    ExecutionAttestationRequiresAuthenticatedSender,
    ExecutionAttestationSenderMismatch(AgentKind),
    InvalidExecutionAttestation(ExecutionAttestationError),
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
            execution_attestation: None,
        }
    }

    fn digest(byte: u8) -> Sha256Digest {
        Sha256Digest::parse(format!("{byte:02x}").repeat(32)).unwrap()
    }

    fn execution_attestation() -> ExecutionAttestation {
        ExecutionAttestation::new(ExecutionProfile {
            schema_version: EXECUTION_PROFILE_SCHEMA_VERSION,
            backend: ExecutionBackendKind::Cuda,
            device_ordinal: 0,
            architecture: ExecutionArchitecture {
                family: ExecutionArchitectureFamily::NvidiaGpu,
                name: Some("sm_110".to_string()),
            },
            capability_profile_sha256: digest(0x11),
            topology_profile_sha256: digest(0x22),
            memory_budget_bytes: Some(1024),
            numeric_mode: "bf16-fp32-accum-v1".to_string(),
            reproducibility: ExecutionReproducibility::NumericallyEquivalent,
            kernel_semantic_version: "route-b-v1".to_string(),
            sampler_semantic_version: Some("sampler-v1".to_string()),
            model_sha256: digest(0x33),
            tokenizer_sha256: digest(0x44),
        })
        .unwrap()
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

    #[test]
    fn schema_v1_messages_without_attestation_keep_their_wire_shape() {
        let value = message(AgentKind::SciAgent, MessageKind::Hypothesis);
        let encoded = serde_json::to_string(&value).unwrap();
        assert!(!encoded.contains("execution_attestation"));

        let decoded: AgentMessage = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, value);
        assert_eq!(decoded.execution_attestation, None);
    }

    #[test]
    fn self_declared_runtime_identity_cannot_bootstrap_attestation_trust() {
        let mut value = message(
            AgentKind::DeterministicKernel,
            MessageKind::ExperimentResult,
        );
        value.execution_attestation = Some(execution_attestation());

        assert_eq!(
            value.validate(),
            Err(ProtocolError::ExecutionAttestationRequiresAuthenticatedSender)
        );
    }

    #[test]
    fn authenticated_deterministic_kernel_can_attach_verified_attestation() {
        let mut value = message(
            AgentKind::DeterministicKernel,
            MessageKind::ExperimentResult,
        );
        value.execution_attestation = Some(execution_attestation());
        let authenticated_sender = value.sender.clone();

        assert_eq!(
            value.validate_with_authenticated_sender(&authenticated_sender),
            Ok(())
        );
    }

    #[test]
    fn authenticated_sender_must_match_serialized_sender() {
        let mut value = message(
            AgentKind::DeterministicKernel,
            MessageKind::ExperimentResult,
        );
        value.execution_attestation = Some(execution_attestation());
        let authenticated_sender = identity(AgentKind::DeterministicKernel, "different-kernel");

        assert_eq!(
            value.validate_with_authenticated_sender(&authenticated_sender),
            Err(ProtocolError::AuthenticatedSenderMismatch)
        );
    }

    #[test]
    fn authenticated_model_sender_cannot_attach_kernel_attestation() {
        let mut value = message(AgentKind::SciAgent, MessageKind::Hypothesis);
        value.execution_attestation = Some(execution_attestation());
        let authenticated_sender = value.sender.clone();

        assert_eq!(
            value.validate_with_authenticated_sender(&authenticated_sender),
            Err(ProtocolError::ExecutionAttestationSenderMismatch(
                AgentKind::SciAgent
            ))
        );
    }

    #[test]
    fn tampered_execution_attestation_fails_closed_with_authenticated_sender() {
        let mut attestation = execution_attestation();
        attestation.profile.numeric_mode = "fp32".to_string();
        let mut value = message(AgentKind::RuntimeDiscovery, MessageKind::ExperimentResult);
        value.execution_attestation = Some(attestation);
        let authenticated_sender = value.sender.clone();

        assert_eq!(
            value.validate_with_authenticated_sender(&authenticated_sender),
            Err(ProtocolError::InvalidExecutionAttestation(
                ExecutionAttestationError::DigestMismatch
            ))
        );
    }
}

use scirust_agent_protocol::{
    AgentIdentity, AgentKind, AgentMessage, ExecutionAttestation, MessageKind, ProtocolError,
    SCHEMA_VERSION, TrustClass,
};
use serde_json::{Value, json};

pub const COGNO_SCIENTIFIC_EXCHANGE_PAYLOAD_SCHEMA_VERSION: u16 = 1;
pub const MAX_COGNO_SCIENTIFIC_EXCHANGE_CANONICAL_PAYLOAD_BYTES: usize = 64 * 1024;

/// Deterministic scientific verdict transported to COGNO.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeScientificExchangeVerdict {
    Confirmed,
    Contradicted,
    Inconclusive,
}

impl RuntimeScientificExchangeVerdict {
    const fn wire_name(self) -> &'static str {
        match self
        {
            Self::Confirmed => "confirmed",
            Self::Contradicted => "contradicted",
            Self::Inconclusive => "inconclusive",
        }
    }
}

/// Versioned scientific-exchange payload emitted by an authenticated
/// [`RuntimeEndpoint`] for COGNO.
///
/// There is deliberately no origin/authority field. COGNO assigns deterministic
/// provenance only after independently authenticating the sender and verifying
/// the attached execution profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeScientificExchangePayload {
    schema_version: u16,
    observation_id: u64,
    preference_id: u64,
    evidence_id: u64,
    verdict: RuntimeScientificExchangeVerdict,
    confidence_bps: u16,
    canonical_payload: Vec<u8>,
}

impl RuntimeScientificExchangePayload {
    pub fn new(
        observation_id: u64,
        preference_id: u64,
        evidence_id: u64,
        verdict: RuntimeScientificExchangeVerdict,
        confidence_bps: u16,
        canonical_payload: Vec<u8>,
    ) -> Result<Self, RuntimeScientificExchangePayloadError> {
        let payload = Self {
            schema_version: COGNO_SCIENTIFIC_EXCHANGE_PAYLOAD_SCHEMA_VERSION,
            observation_id,
            preference_id,
            evidence_id,
            verdict,
            confidence_bps,
            canonical_payload,
        };
        payload.validate()?;
        Ok(payload)
    }

    pub fn validate(&self) -> Result<(), RuntimeScientificExchangePayloadError> {
        if self.schema_version != COGNO_SCIENTIFIC_EXCHANGE_PAYLOAD_SCHEMA_VERSION
        {
            return Err(RuntimeScientificExchangePayloadError::UnsupportedSchema(
                self.schema_version,
            ));
        }
        if self.observation_id == 0
        {
            return Err(RuntimeScientificExchangePayloadError::ZeroObservationId);
        }
        if self.preference_id == 0
        {
            return Err(RuntimeScientificExchangePayloadError::ZeroPreferenceId);
        }
        if self.evidence_id == 0
        {
            return Err(RuntimeScientificExchangePayloadError::ZeroEvidenceId);
        }
        if self.confidence_bps > 10_000
        {
            return Err(RuntimeScientificExchangePayloadError::ConfidenceOutOfRange(
                self.confidence_bps,
            ));
        }
        if self.canonical_payload.is_empty()
        {
            return Err(RuntimeScientificExchangePayloadError::EmptyCanonicalPayload);
        }
        if self.canonical_payload.len() > MAX_COGNO_SCIENTIFIC_EXCHANGE_CANONICAL_PAYLOAD_BYTES
        {
            return Err(RuntimeScientificExchangePayloadError::CanonicalPayloadTooLarge);
        }
        Ok(())
    }

    fn wire_value(&self) -> Value {
        json!({
            "schema_version": self.schema_version,
            "observation_id": self.observation_id,
            "preference_id": self.preference_id,
            "evidence_id": self.evidence_id,
            "verdict": self.verdict.wire_name(),
            "confidence_bps": self.confidence_bps,
            "canonical_payload": self.canonical_payload,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeScientificExchangePayloadError {
    UnsupportedSchema(u16),
    ZeroObservationId,
    ZeroPreferenceId,
    ZeroEvidenceId,
    ConfidenceOutOfRange(u16),
    EmptyCanonicalPayload,
    CanonicalPayloadTooLarge,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeScientificExchangeMessageError {
    InvalidCognoRecipient,
    Payload(RuntimeScientificExchangePayloadError),
    Protocol(ProtocolError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SciAgentEndpoint {
    identity: AgentIdentity,
}

impl SciAgentEndpoint {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            identity: AgentIdentity {
                kind: AgentKind::SciAgent,
                id: id.into(),
            },
        }
    }

    pub fn identity(&self) -> &AgentIdentity {
        &self.identity
    }

    pub fn validate_incoming(&self, message: &AgentMessage) -> Result<(), ProtocolError> {
        message.validate()?;
        if message
            .recipients
            .iter()
            .any(|recipient| recipient == &self.identity)
        {
            Ok(())
        }
        else
        {
            Err(ProtocolError::InvalidRecipientCount(0))
        }
    }

    pub fn hypothesis_message(
        &self,
        message_id: impl Into<String>,
        conversation_id: impl Into<String>,
        recipient: AgentIdentity,
        confidence_bps: u16,
        hypothesis: Value,
    ) -> Result<AgentMessage, ProtocolError> {
        let message = AgentMessage {
            schema_version: SCHEMA_VERSION,
            message_id: message_id.into(),
            conversation_id: conversation_id.into(),
            parent_message_id: None,
            sender: self.identity.clone(),
            recipients: vec![recipient],
            message_kind: MessageKind::Hypothesis,
            trust_class: TrustClass::UntrustedModelOutput,
            confidence_bps,
            evidence: Vec::new(),
            payload: json!({
                "hypothesis": hypothesis,
                "authority": "proposal_only",
            }),
            requested_capabilities: Vec::new(),
            execution_attestation: None,
        };
        message.validate()?;
        Ok(message)
    }

    pub fn critique_message(
        &self,
        message_id: impl Into<String>,
        conversation_id: impl Into<String>,
        parent_message_id: impl Into<String>,
        recipient: AgentIdentity,
        confidence_bps: u16,
        critique: Value,
    ) -> Result<AgentMessage, ProtocolError> {
        let message = AgentMessage {
            schema_version: SCHEMA_VERSION,
            message_id: message_id.into(),
            conversation_id: conversation_id.into(),
            parent_message_id: Some(parent_message_id.into()),
            sender: self.identity.clone(),
            recipients: vec![recipient],
            message_kind: MessageKind::Critique,
            trust_class: TrustClass::UntrustedModelOutput,
            confidence_bps,
            evidence: Vec::new(),
            payload: json!({
                "critique": critique,
                "authority": "proposal_only",
            }),
            requested_capabilities: Vec::new(),
            execution_attestation: None,
        };
        message.validate()?;
        Ok(message)
    }
}

/// Local endpoint whose identity is established by the runtime boundary rather
/// than by model-produced wire data.
///
/// Constructors deliberately expose only the two protocol identities allowed to
/// carry execution attestations. Messages are validated with that independently
/// held identity before they leave the endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeEndpoint {
    identity: AgentIdentity,
}

impl RuntimeEndpoint {
    #[must_use]
    pub fn runtime_discovery(id: impl Into<String>) -> Self {
        Self {
            identity: AgentIdentity {
                kind: AgentKind::RuntimeDiscovery,
                id: id.into(),
            },
        }
    }

    #[must_use]
    pub fn deterministic_kernel(id: impl Into<String>) -> Self {
        Self {
            identity: AgentIdentity {
                kind: AgentKind::DeterministicKernel,
                id: id.into(),
            },
        }
    }

    #[must_use]
    pub fn identity(&self) -> &AgentIdentity {
        &self.identity
    }

    pub fn experiment_result_message(
        &self,
        message_id: impl Into<String>,
        conversation_id: impl Into<String>,
        parent_message_id: Option<String>,
        recipient: AgentIdentity,
        payload: Value,
        execution_attestation: ExecutionAttestation,
    ) -> Result<AgentMessage, ProtocolError> {
        let message = AgentMessage {
            schema_version: SCHEMA_VERSION,
            message_id: message_id.into(),
            conversation_id: conversation_id.into(),
            parent_message_id,
            sender: self.identity.clone(),
            recipients: vec![recipient],
            message_kind: MessageKind::ExperimentResult,
            trust_class: TrustClass::DeterministicKernelOutput,
            confidence_bps: 10_000,
            evidence: Vec::new(),
            payload,
            requested_capabilities: Vec::new(),
            execution_attestation: Some(execution_attestation),
        };
        message.validate_with_authenticated_sender(&self.identity)?;
        Ok(message)
    }

    pub fn scientific_exchange_result_message(
        &self,
        message_id: impl Into<String>,
        conversation_id: impl Into<String>,
        parent_message_id: Option<String>,
        recipient: AgentIdentity,
        payload: RuntimeScientificExchangePayload,
        execution_attestation: ExecutionAttestation,
    ) -> Result<AgentMessage, RuntimeScientificExchangeMessageError> {
        if recipient.kind != AgentKind::CognoCommunicator
        {
            return Err(RuntimeScientificExchangeMessageError::InvalidCognoRecipient);
        }
        payload
            .validate()
            .map_err(RuntimeScientificExchangeMessageError::Payload)?;
        self.experiment_result_message(
            message_id,
            conversation_id,
            parent_message_id,
            recipient,
            payload.wire_value(),
            execution_attestation,
        )
        .map_err(RuntimeScientificExchangeMessageError::Protocol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_agent_protocol::{
        EXECUTION_PROFILE_SCHEMA_VERSION, ExecutionArchitecture, ExecutionArchitectureFamily,
        ExecutionBackendKind, ExecutionProfile, ExecutionReproducibility, Sha256Digest,
    };

    fn digest(byte: u8) -> Sha256Digest {
        Sha256Digest::parse(format!("{byte:02x}").repeat(32)).unwrap()
    }

    fn attestation() -> ExecutionAttestation {
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
            kernel_semantic_version: "cuda-decode-v1".to_string(),
            sampler_semantic_version: Some("greedy-v1".to_string()),
            model_sha256: digest(0x33),
            tokenizer_sha256: digest(0x44),
        })
        .unwrap()
    }

    #[test]
    fn emits_valid_untrusted_hypothesis() {
        let endpoint = SciAgentEndpoint::new("sciagent-local");
        let message = endpoint
            .hypothesis_message(
                "m-1",
                "c-1",
                AgentIdentity {
                    kind: AgentKind::RuntimeDiscovery,
                    id: "runtime".to_string(),
                },
                7_500,
                json!({"expression": "drift * head_std"}),
            )
            .unwrap();
        assert_eq!(message.message_kind, MessageKind::Hypothesis);
        assert_eq!(message.trust_class, TrustClass::UntrustedModelOutput);
        assert_eq!(message.execution_attestation, None);
    }

    #[test]
    fn runtime_endpoint_emits_authenticated_attested_result() {
        let endpoint = RuntimeEndpoint::deterministic_kernel("cuda-decode-local");
        let message = endpoint
            .experiment_result_message(
                "m-runtime-1",
                "c-1",
                Some("request-1".to_string()),
                AgentIdentity {
                    kind: AgentKind::CognoCommunicator,
                    id: "cogno".to_string(),
                },
                json!({"tokens": [1, 2, 3]}),
                attestation(),
            )
            .unwrap();

        assert_eq!(message.sender, *endpoint.identity());
        assert_eq!(message.trust_class, TrustClass::DeterministicKernelOutput);
        assert!(message.execution_attestation.is_some());
        assert_eq!(
            message.validate_with_authenticated_sender(endpoint.identity()),
            Ok(())
        );
        assert!(message.validate().is_err());
    }

    #[test]
    fn runtime_endpoint_emits_cogno_scientific_exchange_without_origin_field() {
        let endpoint = RuntimeEndpoint::deterministic_kernel("cuda-decode-local");
        let payload = RuntimeScientificExchangePayload::new(
            17,
            42,
            117,
            RuntimeScientificExchangeVerdict::Confirmed,
            8_500,
            b"deterministic".to_vec(),
        )
        .unwrap();
        let message = endpoint
            .scientific_exchange_result_message(
                "m-runtime-1",
                "c-1",
                Some("request-1".to_string()),
                AgentIdentity {
                    kind: AgentKind::CognoCommunicator,
                    id: "cogno".to_string(),
                },
                payload,
                attestation(),
            )
            .unwrap();

        assert_eq!(message.payload["schema_version"], 1);
        assert_eq!(message.payload["observation_id"], 17);
        assert_eq!(message.payload["preference_id"], 42);
        assert_eq!(message.payload["evidence_id"], 117);
        assert_eq!(message.payload["verdict"], "confirmed");
        assert_eq!(message.payload["confidence_bps"], 8_500);
        assert!(message.payload.get("origin").is_none());
        assert!(message.payload.get("authority").is_none());
        assert_eq!(
            message.validate_with_authenticated_sender(endpoint.identity()),
            Ok(())
        );
        assert!(message.validate().is_err());
    }

    #[test]
    fn scientific_exchange_builder_requires_cogno_recipient() {
        let endpoint = RuntimeEndpoint::deterministic_kernel("cuda-decode-local");
        let payload = RuntimeScientificExchangePayload::new(
            17,
            42,
            117,
            RuntimeScientificExchangeVerdict::Confirmed,
            8_500,
            b"deterministic".to_vec(),
        )
        .unwrap();
        assert_eq!(
            endpoint.scientific_exchange_result_message(
                "m-runtime-1",
                "c-1",
                None,
                AgentIdentity {
                    kind: AgentKind::DeterministicKernel,
                    id: "other".to_string(),
                },
                payload,
                attestation(),
            ),
            Err(RuntimeScientificExchangeMessageError::InvalidCognoRecipient)
        );
    }

    #[test]
    fn scientific_exchange_payload_is_bounded() {
        assert_eq!(
            RuntimeScientificExchangePayload::new(
                17,
                42,
                117,
                RuntimeScientificExchangeVerdict::Confirmed,
                10_001,
                b"deterministic".to_vec(),
            ),
            Err(RuntimeScientificExchangePayloadError::ConfidenceOutOfRange(
                10_001
            ))
        );
        assert_eq!(
            RuntimeScientificExchangePayload::new(
                17,
                42,
                117,
                RuntimeScientificExchangeVerdict::Confirmed,
                8_500,
                vec![0; MAX_COGNO_SCIENTIFIC_EXCHANGE_CANONICAL_PAYLOAD_BYTES + 1],
            ),
            Err(RuntimeScientificExchangePayloadError::CanonicalPayloadTooLarge)
        );
    }
}

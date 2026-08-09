use scirust_agent_protocol::{
    AgentIdentity, AgentKind, AgentMessage, ExecutionAttestation, MessageKind, ProtocolError,
    SCHEMA_VERSION, TrustClass,
};
use serde_json::{Value, json};

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
}

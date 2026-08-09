use scirust_agent_protocol::{
    AgentIdentity, AgentKind, AgentMessage, MessageKind, ProtocolError, SCHEMA_VERSION, TrustClass,
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

#[cfg(test)]
mod tests {
    use super::*;

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
}

use scirust_agent_protocol::{
    AgentMessage, MessageKind, ProtocolError, SCHEMA_VERSION, TrustClass,
};
use scirust_sciagent::SciAgentEndpoint;
use serde_json::json;
use std::io::{self, BufRead, Write};

const MAX_JSONL_LINE_BYTES: usize = 1_048_576;

fn response_kind(incoming: MessageKind) -> MessageKind {
    match incoming
    {
        MessageKind::Hypothesis
        | MessageKind::Critique
        | MessageKind::Counterexample
        | MessageKind::ExperimentResult
        | MessageKind::Contradiction => MessageKind::Critique,
        MessageKind::Question
        | MessageKind::ExperimentRequest
        | MessageKind::PreferenceObservation
        | MessageKind::Explanation
        | MessageKind::Decision => MessageKind::Explanation,
    }
}

fn build_response(
    endpoint: &SciAgentEndpoint,
    incoming: &AgentMessage,
    ordinal: u64,
) -> Result<AgentMessage, ProtocolError> {
    let message = AgentMessage {
        schema_version: SCHEMA_VERSION,
        message_id: format!("{}-reply-{ordinal}", incoming.message_id),
        conversation_id: incoming.conversation_id.clone(),
        parent_message_id: Some(incoming.message_id.clone()),
        sender: endpoint.identity().clone(),
        recipients: vec![incoming.sender.clone()],
        message_kind: response_kind(incoming.message_kind),
        trust_class: TrustClass::UntrustedModelOutput,
        confidence_bps: 5_000,
        evidence: incoming.evidence.clone(),
        payload: json!({
            "status": "accepted_for_sciagent_processing",
            "incoming_kind": incoming.message_kind,
            "incoming_payload": incoming.payload.clone(),
            "authority": "proposal_only",
            "next_stage": "inference_or_deterministic_tool_routing",
        }),
        requested_capabilities: Vec::new(),
    };
    message.validate()?;
    Ok(message)
}

fn process_line(
    endpoint: &SciAgentEndpoint,
    line: &str,
    line_number: usize,
) -> Result<AgentMessage, String> {
    if line.len() > MAX_JSONL_LINE_BYTES
    {
        return Err(format!(
            "line {line_number} exceeds {MAX_JSONL_LINE_BYTES} bytes"
        ));
    }
    let incoming: AgentMessage = serde_json::from_str(line)
        .map_err(|error| format!("invalid JSON on line {line_number}: {error}"))?;
    endpoint
        .validate_incoming(&incoming)
        .map_err(|error| format!("invalid message on line {line_number}: {error:?}"))?;
    build_response(endpoint, &incoming, line_number as u64)
        .map_err(|error| format!("cannot build response for line {line_number}: {error:?}"))
}

fn run() -> Result<(), String> {
    let endpoint = SciAgentEndpoint::new("sciagent-jsonl");
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut output = stdout.lock();

    for (index, line_result) in stdin.lock().lines().enumerate()
    {
        let line_number = index + 1;
        let line =
            line_result.map_err(|error| format!("cannot read line {line_number}: {error}"))?;
        if line.trim().is_empty()
        {
            continue;
        }
        let response = process_line(&endpoint, &line, line_number)?;
        serde_json::to_writer(&mut output, &response)
            .map_err(|error| format!("cannot encode response for line {line_number}: {error}"))?;
        output
            .write_all(b"\n")
            .map_err(|error| format!("cannot terminate response line {line_number}: {error}"))?;
        output
            .flush()
            .map_err(|error| format!("cannot flush response line {line_number}: {error}"))?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run()
    {
        eprintln!("sciagent-dialogue: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_agent_protocol::{AgentIdentity, AgentKind};

    fn incoming_for(recipient: AgentIdentity) -> AgentMessage {
        AgentMessage {
            schema_version: SCHEMA_VERSION,
            message_id: "question-1".to_string(),
            conversation_id: "conversation-1".to_string(),
            parent_message_id: None,
            sender: AgentIdentity {
                kind: AgentKind::ExternalModel,
                id: "external-model".to_string(),
            },
            recipients: vec![recipient],
            message_kind: MessageKind::Question,
            trust_class: TrustClass::UntrustedModelOutput,
            confidence_bps: 5_000,
            evidence: Vec::new(),
            payload: json!({"question": "Which runtime feature should be tested next?"}),
            requested_capabilities: Vec::new(),
        }
    }

    #[test]
    fn response_preserves_conversation_and_parent() {
        let endpoint = SciAgentEndpoint::new("sciagent-jsonl");
        let incoming = incoming_for(endpoint.identity().clone());
        let response = build_response(&endpoint, &incoming, 1).unwrap();
        assert_eq!(response.conversation_id, incoming.conversation_id);
        assert_eq!(response.parent_message_id.as_deref(), Some("question-1"));
        assert_eq!(response.message_kind, MessageKind::Explanation);
        assert_eq!(response.trust_class, TrustClass::UntrustedModelOutput);
    }

    #[test]
    fn rejects_message_for_another_agent() {
        let endpoint = SciAgentEndpoint::new("sciagent-jsonl");
        let incoming = incoming_for(AgentIdentity {
            kind: AgentKind::SciAgent,
            id: "different-sciagent".to_string(),
        });
        let line = serde_json::to_string(&incoming).unwrap();
        assert!(process_line(&endpoint, &line, 1).is_err());
    }

    #[test]
    fn rejects_oversized_line_before_parsing() {
        let endpoint = SciAgentEndpoint::new("sciagent-jsonl");
        let line = "x".repeat(MAX_JSONL_LINE_BYTES + 1);
        assert!(process_line(&endpoint, &line, 1).is_err());
    }
}

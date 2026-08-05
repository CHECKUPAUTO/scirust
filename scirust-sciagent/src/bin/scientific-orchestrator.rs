use scirust_agent_protocol::{AgentKind, AgentMessage, MessageKind};
use serde::Serialize;
use std::io::{self, BufRead, Write};

const MAX_JSONL_LINE_BYTES: usize = 1_048_576;

#[derive(Debug, Serialize)]
struct CognoObservationEnvelope {
    schema_version: u16,
    observation_id: String,
    conversation_id: String,
    message_id: String,
    parent_message_id: Option<String>,
    sender_class: &'static str,
    message_kind: &'static str,
    evidence_id: u64,
    canonical_payload: Vec<u8>,
    authority: &'static str,
    permitted_action: &'static str,
}

fn sender_class(kind: AgentKind) -> &'static str {
    match kind {
        AgentKind::Human => "human",
        AgentKind::ExternalModel => "external_model",
        AgentKind::SciAgent => "sciagent",
        AgentKind::RuntimeDiscovery => "runtime_discovery",
        AgentKind::DeterministicKernel => "deterministic_kernel",
        AgentKind::CognoObserver => "cogno_observer",
        AgentKind::CognoCommunicator => "cogno_communicator",
    }
}

fn message_kind(kind: MessageKind) -> &'static str {
    match kind {
        MessageKind::Question => "question",
        MessageKind::Hypothesis => "hypothesis",
        MessageKind::Critique => "critique",
        MessageKind::Counterexample => "counterexample",
        MessageKind::ExperimentRequest => "experiment_request",
        MessageKind::ExperimentResult => "experiment_result",
        MessageKind::PreferenceObservation => "preference_observation",
        MessageKind::Contradiction => "contradiction",
        MessageKind::Explanation => "explanation",
        MessageKind::Decision => "decision",
    }
}

fn observation_from_message(
    message: &AgentMessage,
    ordinal: u64,
) -> Result<CognoObservationEnvelope, String> {
    message
        .validate()
        .map_err(|error| format!("invalid agent message: {error:?}"))?;
    let canonical_payload = serde_json::to_vec(message)
        .map_err(|error| format!("cannot canonicalize agent message: {error}"))?;
    Ok(CognoObservationEnvelope {
        schema_version: 1,
        observation_id: format!("{}-observation-{ordinal}", message.message_id),
        conversation_id: message.conversation_id.clone(),
        message_id: message.message_id.clone(),
        parent_message_id: message.parent_message_id.clone(),
        sender_class: sender_class(message.sender.kind),
        message_kind: message_kind(message.message_kind),
        evidence_id: ordinal,
        canonical_payload,
        authority: "observation_only",
        permitted_action: "append_to_cogno_journal_after_runtime_sha256",
    })
}

fn process_line(line: &str, ordinal: u64) -> Result<CognoObservationEnvelope, String> {
    if line.len() > MAX_JSONL_LINE_BYTES {
        return Err(format!(
            "line exceeds maximum of {MAX_JSONL_LINE_BYTES} bytes"
        ));
    }
    let message: AgentMessage =
        serde_json::from_str(line).map_err(|error| format!("invalid JSON: {error}"))?;
    observation_from_message(&message, ordinal)
}

fn run() -> Result<(), String> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    for (index, line_result) in stdin.lock().lines().enumerate() {
        let line_number = index + 1;
        let line = line_result.map_err(|error| format!("cannot read line {line_number}: {error}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let observation = process_line(&line, line_number as u64)
            .map_err(|error| format!("line {line_number}: {error}"))?;
        serde_json::to_writer(&mut output, &observation)
            .map_err(|error| format!("cannot encode observation: {error}"))?;
        output
            .write_all(b"\n")
            .map_err(|error| format!("cannot terminate observation: {error}"))?;
    }
    output
        .flush()
        .map_err(|error| format!("cannot flush observations: {error}"))
}

fn main() {
    if let Err(error) = run() {
        eprintln!("scientific-orchestrator: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_agent_protocol::{
        AgentIdentity, SCHEMA_VERSION, TrustClass,
    };
    use serde_json::json;

    fn message() -> AgentMessage {
        AgentMessage {
            schema_version: SCHEMA_VERSION,
            message_id: "m1".to_string(),
            conversation_id: "c1".to_string(),
            parent_message_id: None,
            sender: AgentIdentity {
                kind: AgentKind::SciAgent,
                id: "sciagent".to_string(),
            },
            recipients: vec![AgentIdentity {
                kind: AgentKind::RuntimeDiscovery,
                id: "runtime".to_string(),
            }],
            message_kind: MessageKind::Hypothesis,
            trust_class: TrustClass::UntrustedModelOutput,
            confidence_bps: 5_000,
            evidence: Vec::new(),
            payload: json!({"expression": "logit_entropy"}),
            requested_capabilities: Vec::new(),
        }
    }

    #[test]
    fn creates_observation_only_envelope() {
        let observation = observation_from_message(&message(), 1).expect("observation");
        assert_eq!(observation.authority, "observation_only");
        assert_eq!(
            observation.permitted_action,
            "append_to_cogno_journal_after_runtime_sha256"
        );
        assert_eq!(observation.message_kind, "hypothesis");
    }

    #[test]
    fn rejects_invalid_agent_message() {
        let mut invalid = message();
        invalid.confidence_bps = 10_001;
        assert!(observation_from_message(&invalid, 1).is_err());
    }
}

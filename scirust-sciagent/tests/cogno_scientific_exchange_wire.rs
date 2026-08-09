use scirust_agent_protocol::{
    AgentIdentity, AgentKind, EXECUTION_PROFILE_SCHEMA_VERSION, ExecutionArchitecture,
    ExecutionArchitectureFamily, ExecutionAttestation, ExecutionBackendKind, ExecutionProfile,
    ExecutionReproducibility, Sha256Digest,
};
use scirust_sciagent::{
    RuntimeEndpoint, RuntimeScientificExchangePayload, RuntimeScientificExchangeVerdict,
};

const GOLDEN_EXECUTION_PROFILE_SHA256: &str =
    "c984c0151e84300875c2aead5764d018f9ef5d09d218ab8f8f1ea9ab7157bec8";

fn digest(byte: u8) -> Sha256Digest {
    Sha256Digest::parse(format!("{byte:02x}").repeat(32)).expect("fixture digest")
}

fn golden_attestation() -> ExecutionAttestation {
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
    .expect("golden execution attestation")
}

#[test]
fn cogno_exchange_wire_matches_cross_repo_golden_profile() {
    let attestation = golden_attestation();
    assert_eq!(
        attestation.profile_sha256.as_str(),
        GOLDEN_EXECUTION_PROFILE_SHA256
    );

    let endpoint = RuntimeEndpoint::deterministic_kernel("cuda-decode-local");
    let payload = RuntimeScientificExchangePayload::new(
        17,
        42,
        117,
        RuntimeScientificExchangeVerdict::Confirmed,
        8_500,
        b"deterministic".to_vec(),
    )
    .expect("scientific payload");
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
            attestation,
        )
        .expect("COGNO scientific result");

    let wire = serde_json::to_value(&message).expect("wire json");
    assert_eq!(wire["sender"]["kind"], "deterministic_kernel");
    assert_eq!(wire["recipients"][0]["kind"], "cogno_communicator");
    assert_eq!(wire["message_kind"], "experiment_result");
    assert_eq!(wire["trust_class"], "deterministic_kernel_output");
    assert_eq!(wire["confidence_bps"], 10_000);
    assert_eq!(wire["payload"]["schema_version"], 1);
    assert_eq!(wire["payload"]["observation_id"], 17);
    assert_eq!(wire["payload"]["preference_id"], 42);
    assert_eq!(wire["payload"]["evidence_id"], 117);
    assert_eq!(wire["payload"]["verdict"], "confirmed");
    assert_eq!(wire["payload"]["confidence_bps"], 8_500);
    assert!(wire["payload"].get("origin").is_none());
    assert!(wire["payload"].get("authority").is_none());
    assert_eq!(
        wire["execution_attestation"]["profile_sha256"],
        GOLDEN_EXECUTION_PROFILE_SHA256
    );
    assert_eq!(
        message.validate_with_authenticated_sender(endpoint.identity()),
        Ok(())
    );
    assert!(message.validate().is_err());
}

use scirust_agent_protocol::{
    EXECUTION_PROFILE_SCHEMA_VERSION, ExecutionArchitecture, ExecutionArchitectureFamily,
    ExecutionAttestation, ExecutionAttestationError, ExecutionBackendKind, ExecutionProfile,
    ExecutionReproducibility, Sha256Digest,
};

fn hash(byte: u8) -> Sha256Digest {
    Sha256Digest::parse(format!("{byte:02x}").repeat(32)).unwrap()
}

fn profile() -> ExecutionProfile {
    ExecutionProfile {
        schema_version: EXECUTION_PROFILE_SCHEMA_VERSION,
        backend: ExecutionBackendKind::Cuda,
        device_ordinal: 0,
        architecture: ExecutionArchitecture {
            family: ExecutionArchitectureFamily::NvidiaGpu,
            name: Some("sm_110".to_string()),
        },
        capability_profile_sha256: hash(0x11),
        topology_profile_sha256: hash(0x22),
        numeric_mode: "bf16_tensor_core".to_string(),
        reproducibility: ExecutionReproducibility::Deterministic,
        kernel_semantic_version: "sciagent.decode.v1".to_string(),
        sampler_semantic_version: Some("resident_sampler.v1".to_string()),
        model_sha256: hash(0x33),
        tokenizer_sha256: hash(0x44),
    }
}

#[test]
fn v1_profile_has_a_stable_golden_fingerprint() {
    assert_eq!(
        profile().fingerprint().unwrap().as_str(),
        "872aff14c03ea57da47ca2554981bebc4bd2f81695c4ba649b1fbc88b521d2ea"
    );
}

#[test]
fn wire_round_trip_preserves_the_verified_attestation() {
    let attestation = ExecutionAttestation::new(profile()).unwrap();
    let json = serde_json::to_vec(&attestation).unwrap();
    let decoded: ExecutionAttestation = serde_json::from_slice(&json).unwrap();

    assert_eq!(decoded.verify(), Ok(()));
    assert_eq!(decoded, attestation);
}

#[test]
fn tampered_wire_profile_fails_verification() {
    let attestation = ExecutionAttestation::new(profile()).unwrap();
    let mut value = serde_json::to_value(attestation).unwrap();
    value["profile"]["numeric_mode"] = serde_json::json!("f32");
    let decoded: ExecutionAttestation = serde_json::from_value(value).unwrap();

    assert_eq!(
        decoded.verify(),
        Err(ExecutionAttestationError::DigestMismatch)
    );
}

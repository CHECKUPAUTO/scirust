use ccos_core::distributed_event_log::DistributedEventLog;

#[test]
fn replay_is_deterministic_under_load() {
    let mut log = DistributedEventLog::new();

    // Load 500 events
    for i in 0..500 {
        log.append(
            format!("event_payload_number_{}", i),
            format!("source_module_{}", i % 10),
        );
    }

    let replay1 = log.replay();
    let replay2 = log.replay();
    let replay3 = log.replay();

    // All replays must be bitwise identical
    assert_eq!(replay1.len(), replay2.len());
    assert_eq!(replay2.len(), replay3.len());
    assert_eq!(replay1.len(), 500);

    for i in 0..500 {
        assert_eq!(
            replay1[i].id, replay2[i].id,
            "Replay mismatch at index {}: run1 id={}, run2 id={}",
            i, replay1[i].id, replay2[i].id
        );
        assert_eq!(
            replay1[i].payload, replay2[i].payload,
            "Replay mismatch at index {}: payload differs",
            i
        );
        assert_eq!(
            replay1[i].timestamp, replay2[i].timestamp,
            "Replay mismatch at index {}: timestamp differs",
            i
        );

        assert_eq!(
            replay2[i].id, replay3[i].id,
            "Replay mismatch between run2 and run3 at index {}",
            i
        );
        assert_eq!(replay2[i].payload, replay3[i].payload);
        assert_eq!(replay2[i].timestamp, replay3[i].timestamp);
    }
}

#[test]
fn hash_chain_verifiable() {
    let mut log = DistributedEventLog::new();

    for i in 0..100 {
        log.append(format!("critical_operation_{}", i), "kernel".into());
    }

    let report = log.verify_integrity();
    assert!(
        report.valid,
        "Hash chain must be valid: {:?}",
        report.errors
    );
    assert_eq!(report.total_events, 100);
    assert_eq!(report.verified_events, 100);
}

#[test]
fn hash_chain_prevents_mutation() {
    let mut log = DistributedEventLog::new();
    log.append("immutable_record".into(), "audit".into());

    // Tamper with the event payload
    log.events[0].payload = "MUTATED_RECORD".into();

    let report = log.verify_integrity();
    assert!(!report.valid, "Tampered log must fail integrity check");
    assert!(
        report.errors.iter().any(|e| e.contains("Hash mismatch")),
        "Error must indicate hash mismatch: {:?}",
        report.errors
    );
}

#[test]
fn chain_linkage_preserved_under_load() {
    let mut log = DistributedEventLog::new();

    for i in 0..1000 {
        log.append(format!("load_test_event_{}", i), "stress".into());
    }

    // Verify chain is unbroken: each link references previous hash
    for i in 1..log.hash_chain.len() {
        assert_eq!(
            log.hash_chain[i].previous_hash,
            log.hash_chain[i - 1].hash,
            "Chain broken between event {} and {}",
            log.hash_chain[i - 1].event_id,
            log.hash_chain[i].event_id
        );
    }
}

#[test]
fn export_and_verify_external() {
    let mut log = DistributedEventLog::new();
    log.append("export_test_1".into(), "exporter".into());
    log.append("export_test_2".into(), "exporter".into());

    let exported = log.export_chain();

    // External verifier can recompute hashes
    for (i, link) in exported.iter().enumerate() {
        let expected_prev = if i == 0 {
            "GENESIS".to_string()
        } else {
            exported[i - 1].hash.clone()
        };
        assert_eq!(
            link.previous_hash, expected_prev,
            "Exported chain has broken linkage at {}",
            link.event_id
        );
    }
}

#[test]
fn empty_log_is_valid() {
    let log = DistributedEventLog::new();
    let report = log.verify_integrity();
    assert!(report.valid);
    assert_eq!(report.total_events, 0);
}

#[test]
fn replay_empty_log() {
    let log = DistributedEventLog::new();
    let replay = log.replay();
    assert!(replay.is_empty());
}

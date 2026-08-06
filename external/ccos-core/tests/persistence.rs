//! CCOS v0.3 — Persistent state engine integration tests: a full
//! create → shutdown → reload → compare cycle, plus a corrupted-log chaos case.

use ccos_core::distributed_event_log::DistributedEventLog;
use ccos_core::event_log::{EventLog, EventPayload, EventType};
use ccos_core::incremental::IncrementalGraphEngine;
use ccos_core::memory::MemoryGraph;
use ccos_core::persistence::{PersistenceError, PersistentRuntime, RuntimeState};
use std::path::PathBuf;

fn temp_dir(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "ccos_persist_int_{}_{}_{}",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    p
}

fn build_state() -> RuntimeState {
    let mut graph = MemoryGraph::new(0.2, 1_000_000);
    let mut engine = IncrementalGraphEngine::new();
    let mut event_log = EventLog::new("persist".into());
    let mut dist_log = DistributedEventLog::new();
    for i in 0..12 {
        let src = format!("mod m{i};\npub fn f{i}() {{}}\nstruct S{i};\n");
        engine.process_delta(&format!("src/f{i}.rs"), None, &src, &mut graph);
        event_log.append(
            EventType::CycleEnd,
            EventPayload::CycleEvent {
                cycle_number: i,
                action: "ingest".into(),
            },
        );
        dist_log.append(format!("file_{i}"), "kernel".into());
    }
    RuntimeState::new(graph, event_log, dist_log)
}

#[test]
fn full_shutdown_and_reboot_preserves_state() {
    let dir = temp_dir("reboot");
    let runtime = PersistentRuntime::new(&dir);

    // 1. Create + 2. persist + simulate shutdown.
    let before = build_state();
    let snapshot = (
        before.graph.node_count(),
        before.graph.edge_count(),
        before.event_log.event_count(),
        before.dist_log.event_count(),
    );
    runtime.save_state(&before).unwrap();
    drop(before);
    assert!(runtime.exists());

    // 3. Reload on a brand-new handle (as after a reboot).
    let after = PersistentRuntime::new(&dir).restore_runtime().unwrap();

    // 4. Compare — state must be identical and verified.
    assert_eq!(after.graph.node_count(), snapshot.0);
    assert_eq!(after.graph.edge_count(), snapshot.1);
    assert_eq!(after.event_log.event_count(), snapshot.2);
    assert_eq!(after.dist_log.event_count(), snapshot.3);
    assert!(after.dist_log.verify_integrity().valid);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn chaos_corrupted_log_rejected_on_restore() {
    let dir = temp_dir("corrupt");
    let runtime = PersistentRuntime::new(&dir);
    let mut state = build_state();
    // Corrupt a hash-chain link, then save.
    state.dist_log.hash_chain[3].hash = "00".repeat(32);
    runtime.save_state(&state).unwrap();

    // Restore must reject the tampered state cleanly (no panic).
    let result = runtime.restore_runtime();
    assert!(matches!(result, Err(PersistenceError::Integrity(_))));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn chaos_truncated_file_errors_cleanly() {
    let dir = temp_dir("truncated");
    let runtime = PersistentRuntime::new(&dir);
    runtime.save_state(&build_state()).unwrap();
    // Corrupt the graph snapshot with non-JSON garbage.
    std::fs::write(runtime.graph_path(), "{ this is not json").unwrap();

    let result = runtime.load_state();
    assert!(matches!(result, Err(PersistenceError::Serde(_))));

    std::fs::remove_dir_all(&dir).ok();
}

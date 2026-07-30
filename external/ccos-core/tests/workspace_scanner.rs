//! CCOS v0.3 — Real workspace scanner integration tests, including the
//! canonical "modify one file → only that file is reparsed" scenario and a
//! file-vanishes-mid-scan chaos case.
//!
//! The workspace scanner is async (`tokio::fs`), so these tests live behind the
//! `llm` feature with the module.
#![cfg(feature = "llm")]

use ccos_core::incremental::IncrementalGraphEngine;
use ccos_core::memory::MemoryGraph;
use ccos_core::workspace::WorkspaceScanner;
use std::path::PathBuf;

fn temp_workspace(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "ccos_wsint_{}_{}_{}",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&p).unwrap();
    std::fs::write(p.join("main.rs"), "fn main() {}").unwrap();
    std::fs::write(p.join("lib.rs"), "pub mod module;\npub fn lib_fn() {}").unwrap();
    std::fs::write(p.join("module.rs"), "pub fn a() {}").unwrap();
    p
}

#[tokio::test]
async fn modifying_one_file_reparses_only_that_file() {
    let dir = temp_workspace("modify");
    let mut scanner = WorkspaceScanner::new(dir.to_string_lossy().to_string());
    let mut engine = IncrementalGraphEngine::new();
    let mut graph = MemoryGraph::new(0.2, 1_000_000);

    // Initial sync: 3 files ingested.
    let d0 = scanner.sync(&mut engine, &mut graph).await.unwrap();
    assert_eq!(d0.added.len(), 3);
    let mutations0 = engine.total_mutations();

    // Modify only module.rs.
    std::fs::write(
        dir.join("module.rs"),
        "pub fn a() {}\npub fn b() {}\nstruct M;",
    )
    .unwrap();
    let d1 = scanner.sync(&mut engine, &mut graph).await.unwrap();

    assert_eq!(d1.modified.len(), 1);
    assert!(d1.modified[0].ends_with("module.rs"));
    assert!(d1.added.is_empty() && d1.removed.is_empty());
    assert_eq!(
        engine.total_mutations() - mutations0,
        1,
        "exactly one file reparsed (O(Δ))"
    );
    assert_eq!(graph.prune_dangling_edges(), 0);

    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn chaos_file_deleted_is_handled_gracefully() {
    let dir = temp_workspace("chaos_delete");
    let mut scanner = WorkspaceScanner::new(dir.to_string_lossy().to_string());
    let mut engine = IncrementalGraphEngine::new();
    let mut graph = MemoryGraph::new(0.2, 1_000_000);
    scanner.sync(&mut engine, &mut graph).await.unwrap();
    let before = graph.node_count();

    // A file disappears between syncs — must not crash, must evict its nodes.
    std::fs::remove_file(dir.join("module.rs")).unwrap();
    let d = scanner.sync(&mut engine, &mut graph).await.unwrap();
    assert_eq!(d.removed.len(), 1);
    assert!(graph.node_count() < before);
    assert_eq!(
        graph.prune_dangling_edges(),
        0,
        "no dangling edges after deletion"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn scanning_missing_directory_does_not_crash() {
    let mut scanner = WorkspaceScanner::new("/nonexistent/ccos/path/xyz");
    // Empty/absent tree yields an empty delta rather than a panic.
    let delta = scanner.scan_workspace().await.unwrap();
    assert!(delta.is_empty());
}

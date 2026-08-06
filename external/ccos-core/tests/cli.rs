//! End-to-end tests that drive the `ccos` binary as a black box.
//!
//! Gated on `llm` because the binary declares `required-features = ["llm"]`, so
//! it is only built (and `CARGO_BIN_EXE_ccos` only set) under `--features llm`.
//! Under plain `cargo test` this file compiles to nothing, like
//! `workspace_scanner.rs`.
#![cfg(feature = "llm")]

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

fn ccos() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ccos"))
}

static CNT: AtomicU32 = AtomicU32::new(0);
fn tmp(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "ccos_cli_{}_{}_{}",
        tag,
        std::process::id(),
        CNT.fetch_add(1, Ordering::Relaxed)
    ));
    p
}

#[test]
fn version_and_help_exit_zero() {
    let v = ccos().arg("version").output().unwrap();
    assert!(v.status.success());
    assert!(String::from_utf8_lossy(&v.stdout).contains("ccos"));

    let h = ccos().arg("--help").output().unwrap();
    assert!(h.status.success());
    assert!(String::from_utf8_lossy(&h.stdout).contains("USAGE"));
}

#[test]
fn unknown_command_exits_two() {
    let out = ccos().arg("frobnicate").output().unwrap();
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn analyze_verify_replay_roundtrip() {
    let dir = tmp("roundtrip");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("a.rs"),
        "use crate::b;\npub fn a() -> i32 { b::x() }\n",
    )
    .unwrap();
    std::fs::write(dir.join("b.rs"), "pub fn x() -> i32 { 1 }\n").unwrap();
    let snap = dir.join("run.json");

    let a = ccos()
        .args([
            "analyze",
            dir.to_str().unwrap(),
            "--out",
            snap.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        a.status.success(),
        "analyze failed: {}",
        String::from_utf8_lossy(&a.stderr)
    );
    assert!(snap.exists(), "analyze must write the snapshot");

    let v = ccos()
        .args(["verify", snap.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        v.status.success(),
        "verify must accept a freshly written snapshot"
    );

    let r = ccos()
        .args(["replay", snap.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(r.status.success(), "replay must succeed on the snapshot");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn sanitize_strict_flags_a_bidi_override() {
    let dir = tmp("sanitize");
    std::fs::create_dir_all(&dir).unwrap();
    let clean = dir.join("clean.rs");
    std::fs::write(&clean, "pub fn ok() -> i32 { 1 }\n").unwrap();
    let evil = dir.join("evil.rs");
    // A real RLO (U+202E) bidi override — the Trojan-Source vector.
    std::fs::write(&evil, "let admin = false;\u{202E} // grant\n").unwrap();

    let c = ccos()
        .args(["sanitize", "--strict", clean.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(c.status.code(), Some(0), "a clean file is not dangerous");

    let e = ccos()
        .args(["sanitize", "--strict", evil.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(
        e.status.code(),
        Some(1),
        "a bidi override must exit non-zero under --strict"
    );
    assert!(
        String::from_utf8_lossy(&e.stdout).contains("RLO"),
        "the surfaced literal must name the override"
    );

    std::fs::remove_dir_all(&dir).ok();
}

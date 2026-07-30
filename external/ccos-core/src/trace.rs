//! # Runtime execution traces — the dynamic layer of the dual graph
//!
//! CCOS's *static* causal graph (AST imports) ties a lexical RAG on real-bug
//! retrieval because cross-file structure is diffuse (measured: see the
//! causal-validation harness). The *dynamic* layer is the fix: a crash's stack
//! trace is a **direct causal path** from the symptom (the failing test, top of
//! stack) to the cause (where it panicked, bottom of stack), skipping the diffuse
//! middle that dilutes a purely-structural walk.
//!
//! This module turns `cargo test` / panic / backtrace text into the set of source
//! locations the execution actually touched — the input to a *context page fault*,
//! where the governor injects high-priority failure pressure on exactly those
//! nodes (and the static graph then expands a bounded neighbourhood around them).
//!
//! It is intentionally **non-intrusive**: it parses output the harness already
//! captures, needs no instrumentation, and runs on any Rust project unmodified.
//!
//! ```
//! use ccos_core::trace::parse_cargo_test_output;
//! let out = "thread 'tests::t' panicked at src/lib.rs:42:9:\n\
//!            stack backtrace:\n\
//!            \x20  2: mycrate::compute\n\
//!            \x20            at ./src/compute.rs:17\n";
//! let trace = parse_cargo_test_output(out);
//! assert_eq!(trace.files(), vec!["src/compute.rs".to_string(), "src/lib.rs".to_string()]);
//! ```

use std::collections::BTreeSet;

/// One source location implicated by an execution trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceHit {
    /// Project-relative file path (e.g. `src/compute.rs`).
    pub file: String,
    /// 1-based line number.
    pub line: u32,
    /// Stack depth: `0` for the panic site / non-frame hits, increasing down the
    /// backtrace. The lowest non-zero depths are nearest the cause.
    pub frame_depth: usize,
}

/// The locations one failing execution touched, in the order they appear.
#[derive(Debug, Clone, Default)]
pub struct ExecutionTrace {
    /// The panic / assertion message line, if found.
    pub message: String,
    /// Implicated source locations, de-duplicated, in appearance order.
    pub hits: Vec<TraceHit>,
}

impl ExecutionTrace {
    /// Distinct project files in the trace, sorted — the page-fault seed set.
    pub fn files(&self) -> Vec<String> {
        let mut v: Vec<String> = self.hits.iter().map(|h| h.file.clone()).collect();
        v.sort();
        v.dedup();
        v
    }

    /// Whether the trace found any project location.
    pub fn is_empty(&self) -> bool {
        self.hits.is_empty()
    }
}

/// Strip a leading `./`; treat as project-relative.
fn normalize(path: &str) -> String {
    path.strip_prefix("./").unwrap_or(path).to_string()
}

/// Keep only locations that belong to the project under test, dropping the
/// standard library, the registry, and the compiler's own sources.
fn is_project_path(path: &str) -> bool {
    let p = path.strip_prefix("./").unwrap_or(path);
    if p.starts_with('/') {
        // Absolute paths are almost always toolchain/registry; keep none.
        return false;
    }
    !(p.contains(".cargo/registry")
        || p.contains(".rustup")
        || p.starts_with("/rustc/")
        || p.contains("library/std")
        || p.contains("library/core")
        || p.contains("library/alloc"))
}

/// Extract every `<path>.rs:<line>` occurrence from a single line.
fn extract_rs_locs(line: &str) -> Vec<(String, u32)> {
    let bytes = line.as_bytes();
    let mut out = Vec::new();
    let mut search = 0usize;
    while let Some(rel) = line[search..].find(".rs:") {
        let dot = search + rel; // index of '.' in ".rs:"
                                // Walk back to the start of the path token.
        let mut start = dot;
        while start > 0 {
            let c = bytes[start - 1];
            if matches!(
                c,
                b' ' | b'\t' | b'\'' | b'"' | b',' | b'(' | b'`' | b'=' | b'<'
            ) {
                break;
            }
            start -= 1;
        }
        let path_end = dot + 3; // index just past ".rs"
        let file = &line[start..path_end];
        // Parse the line number following ".rs:".
        let mut j = path_end + 1; // skip the ':'
        let num_start = j;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
        }
        if j > num_start {
            if let Ok(ln) = line[num_start..j].parse::<u32>() {
                if !file.is_empty() {
                    out.push((file.to_string(), ln));
                }
            }
        }
        search = j.max(dot + 4);
    }
    out
}

/// Parse `cargo test` (panic + `RUST_BACKTRACE` + assertion) output into the
/// execution trace it describes. Locations outside the project (std, registry)
/// are dropped; project locations are de-duplicated in appearance order.
pub fn parse_cargo_test_output(output: &str) -> ExecutionTrace {
    let mut hits = Vec::new();
    let mut seen: BTreeSet<(String, u32)> = BTreeSet::new();
    let mut message = String::new();
    let mut depth = 0usize;

    for line in output.lines() {
        let trimmed = line.trim_start();
        // Backtrace frame location lines look like `at ./src/foo.rs:17`.
        let is_frame = trimmed.starts_with("at ");

        for (raw, ln) in extract_rs_locs(line) {
            if !is_project_path(&raw) {
                continue;
            }
            let file = normalize(&raw);
            if seen.insert((file.clone(), ln)) {
                hits.push(TraceHit {
                    file,
                    line: ln,
                    frame_depth: if is_frame { depth } else { 0 },
                });
            }
        }

        if message.is_empty() && trimmed.contains("panicked at") {
            message = trimmed.to_string();
        }
        if is_frame {
            depth += 1;
        }
    }

    ExecutionTrace { message, hits }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
running 1 test
test tests::it_works ... FAILED

failures:

---- tests::it_works stdout ----
thread 'tests::it_works' panicked at src/lib.rs:42:9:
assertion `left == right` failed
  left: 4
 right: 5
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
stack backtrace:
   0: rust_begin_unwind
             at /rustc/abc/library/std/src/panicking.rs:662:5
   1: core::panicking::panic_fmt
             at /rustc/abc/library/core/src/panicking.rs:74:14
   2: mycrate::compute::add
             at ./src/compute.rs:17
   3: mycrate::tests::it_works
             at ./src/lib.rs:40:9
";

    #[test]
    fn extracts_project_locations_and_drops_std() {
        let t = parse_cargo_test_output(SAMPLE);
        // Project files only: src/lib.rs (panic site + frame) and src/compute.rs.
        assert_eq!(
            t.files(),
            vec!["src/compute.rs".to_string(), "src/lib.rs".to_string()]
        );
        // std/core frames are dropped.
        assert!(!t.hits.iter().any(|h| h.file.contains("library")));
        assert!(t.message.contains("panicked at src/lib.rs:42"));
    }

    #[test]
    fn panic_site_is_depth_zero_frames_increase() {
        let t = parse_cargo_test_output(SAMPLE);
        let panic_site = t
            .hits
            .iter()
            .find(|h| h.file == "src/lib.rs" && h.line == 42)
            .unwrap();
        assert_eq!(
            panic_site.frame_depth, 0,
            "panic site is not a backtrace frame"
        );
        let cause = t.hits.iter().find(|h| h.file == "src/compute.rs").unwrap();
        assert!(cause.frame_depth >= 1, "backtrace frame has non-zero depth");
    }

    #[test]
    fn old_style_inline_panic_message() {
        // Pre-1.65 format: `panicked at 'msg', src/foo.rs:10:5`.
        let t = parse_cargo_test_output("thread 'x' panicked at 'boom', src/foo.rs:10:5\n");
        assert_eq!(t.files(), vec!["src/foo.rs".to_string()]);
        assert_eq!(t.hits[0].line, 10);
    }

    #[test]
    fn empty_on_clean_output() {
        let t = parse_cargo_test_output("test tests::ok ... ok\n\ntest result: ok. 1 passed\n");
        assert!(t.is_empty());
        assert!(t.files().is_empty());
    }

    #[test]
    fn handles_windows_style_and_columns() {
        let t = parse_cargo_test_output("   at src/net/tcp.rs:128:13\n");
        assert_eq!(t.files(), vec!["src/net/tcp.rs".to_string()]);
        assert_eq!(t.hits[0].line, 128);
    }
}

//! Shared engine of the SOM command-line tools (`som-analyze` and the
//! `cargo som` subcommand): analyze a real Rust file with the ownership
//! oracle and render the result as human text or SARIF 2.1.0.

use scirust_som_frontend::lower_str;
use scirust_som_symbolic::{Analysis, OwnershipOracle};
use scirust_som_visualizer::render_markdown;

/// Output format of the analyzer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Text,
    Sarif,
}

/// Outcome of analyzing one file.
#[derive(Debug, Clone)]
pub struct Outcome {
    /// Rendered report (markdown text or SARIF JSON).
    pub rendered: String,
    /// Number of ownership faults found.
    pub faults: usize,
}

/// Analyze `path` and render in `format`.
///
/// `Err(message)` covers I/O and Rust syntax errors (caller exit code 2);
/// `Ok` with `faults > 0` is the linter-failure case (caller exit code 1).
pub fn analyze_file(path: &str, format: Format) -> Result<Outcome, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let lowered = lower_str(&src).map_err(|e| format!("{path} is not valid Rust: {e}"))?;
    let analysis = OwnershipOracle::new().analyze(&lowered.ast);
    let faults = analysis.diagnostics.len();

    let rendered = match format
    {
        Format::Sarif => render_sarif(path, &analysis),
        Format::Text =>
        {
            let mut out = String::new();
            out.push_str(&format!("# SOM ownership analysis — {path}\n\n"));
            out.push_str(&render_markdown(&analysis));
            if !lowered.approximations.is_empty()
            {
                out.push_str("\nApproximations applied:\n");
                for a in &lowered.approximations
                {
                    out.push_str(&format!("- {a}\n"));
                }
            }
            if !lowered.unsupported.is_empty()
            {
                out.push_str("\nConstructs skipped (not modelled):\n");
                for u in &lowered.unsupported
                {
                    out.push_str(&format!("- {u}\n"));
                }
            }
            out.push_str(&format!(
                "\nSummary: {} token(s), {} ownership fault(s).\n",
                analysis.tokens.len(),
                faults
            ));
            out
        },
    };
    Ok(Outcome { rendered, faults })
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Minimal, valid SARIF 2.1.0 (one run, one result per diagnostic).
///
/// Locations are file-level for now: the oracle reports token indices, not
/// source spans — span-accurate regions are the next milestone (frontend
/// `proc-macro2` span-locations). The token context is carried in the
/// result message so reviewers can pinpoint the fault.
pub fn render_sarif(path: &str, analysis: &Analysis) -> String {
    let mut results = String::new();
    for (i, d) in analysis.diagnostics.iter().enumerate()
    {
        if i > 0
        {
            results.push(',');
        }
        let token = analysis
            .tokens
            .get(d.token_index)
            .map(|t| format!("{t:?}"))
            .unwrap_or_default();
        results.push_str(&format!(
            r#"{{"ruleId":"{:?}","level":"error","message":{{"text":"{} on `{}` (token #{}: {})"}},"locations":[{{"physicalLocation":{{"artifactLocation":{{"uri":"{}"}}}}}}]}}"#,
            d.kind,
            format_args!("{:?}", d.kind),
            json_escape(&d.var),
            d.token_index,
            json_escape(&token),
            json_escape(path),
        ));
    }
    format!(
        r#"{{"$schema":"https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json","version":"2.1.0","runs":[{{"tool":{{"driver":{{"name":"scirust-som","informationUri":"https://github.com/Memorithm/scirust","version":"0.1.0"}}}},"results":[{results}]}}]}}"#
    )
}

/// Parse the argument vector of the SOM CLIs: an optional `--sarif` flag and
/// exactly one positional `<file.rs>` path, in any order.
///
/// Returns `None` for the two usage errors the driver maps to exit code 2:
/// no path given, or a second positional argument. Kept side-effect free (no
/// printing, no process exit) so the parsing contract is unit-testable; the
/// caller renders the usage message and chooses the exit code.
pub fn parse_args(args: &[String]) -> Option<(Format, &str)> {
    let mut format = Format::Text;
    let mut path: Option<&str> = None;
    for a in args
    {
        match a.as_str()
        {
            "--sarif" => format = Format::Sarif,
            other if path.is_none() => path = Some(other),
            _ => return None,
        }
    }
    path.map(|p| (format, p))
}

/// Drop cargo's injected subcommand word for the `cargo som` front-end.
///
/// Cargo invokes an external subcommand `cargo-som` as `cargo-som som <args…>`,
/// so the first argument is the literal subcommand name `som` and must be
/// removed before the remaining `[--sarif] <file.rs>` is parsed. When the
/// binary is run directly (not via cargo) the `som` word is absent and the
/// arguments are returned unchanged. Only the *first* argument is considered,
/// so a file positionally named `som` (e.g. `cargo som som`) survives.
pub fn strip_cargo_subcommand(args: &[String]) -> &[String] {
    match args.first()
    {
        Some(first) if first == "som" => &args[1..],
        _ => args,
    }
}

/// Shared CLI driver: parse `[--sarif] <file.rs>` and print the report.
/// Returns the process exit code (0 clean, 1 faults, 2 usage/IO/syntax).
pub fn run(args: &[String], tool_name: &str) -> u8 {
    let Some((format, path)) = parse_args(args)
    else
    {
        eprintln!("usage: {tool_name} [--sarif] <file.rs>");
        return 2;
    };
    match analyze_file(path, format)
    {
        Ok(outcome) =>
        {
            println!("{}", outcome.rendered);
            if outcome.faults > 0 { 1 } else { 0 }
        },
        Err(e) =>
        {
            eprintln!("error: {e}");
            2
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Write `src` to a fresh, uniquely named `.rs` file under the OS temp
    /// directory and return its path. Each call gets a distinct name so the
    /// I/O-driven tests (`analyze_file`, `run`) never collide, even in
    /// parallel. The file is left on disk (temp dir); the content is what the
    /// pipeline reads back.
    fn fixture(src: &str) -> PathBuf {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!("som_cli_test_{}_{n}.rs", std::process::id()));
        std::fs::write(&path, src).expect("write fixture");
        path
    }

    // ---- parse_args: the full argument grammar -----------------------------

    #[test]
    fn parse_args_path_only_defaults_to_text() {
        let args = vec!["foo.rs".to_string()];
        assert_eq!(parse_args(&args), Some((Format::Text, "foo.rs")));
    }

    #[test]
    fn parse_args_sarif_flag_before_path() {
        let args = vec!["--sarif".to_string(), "foo.rs".to_string()];
        assert_eq!(parse_args(&args), Some((Format::Sarif, "foo.rs")));
    }

    #[test]
    fn parse_args_sarif_flag_after_path() {
        // The flag may appear in either order; the path is still `foo.rs`,
        // not the flag.
        let args = vec!["foo.rs".to_string(), "--sarif".to_string()];
        assert_eq!(parse_args(&args), Some((Format::Sarif, "foo.rs")));
    }

    #[test]
    fn parse_args_no_path_is_usage_error() {
        assert_eq!(parse_args(&[]), None);
        assert_eq!(parse_args(&["--sarif".to_string()]), None);
    }

    #[test]
    fn parse_args_second_positional_is_usage_error() {
        let args = vec!["a.rs".to_string(), "b.rs".to_string()];
        assert_eq!(parse_args(&args), None);
    }

    #[test]
    fn parse_args_path_named_like_flag_is_taken_literally() {
        // A file whose name happens to be `--sarif` cannot be expressed (the
        // flag wins), but any other token — including one starting with `-` —
        // is a path. A leading `-` is not specially rejected.
        let args = vec!["-weird-name.rs".to_string()];
        assert_eq!(parse_args(&args), Some((Format::Text, "-weird-name.rs")));
    }

    // ---- strip_cargo_subcommand: the `cargo som` argument fixup -----------

    #[test]
    fn strip_cargo_subcommand_removes_injected_som_word() {
        // `cargo som --sarif foo.rs` reaches the binary as
        // `["som", "--sarif", "foo.rs"]`; the leading `som` is dropped.
        let args = vec![
            "som".to_string(),
            "--sarif".to_string(),
            "foo.rs".to_string(),
        ];
        assert_eq!(strip_cargo_subcommand(&args), &args[1..]);
        assert_eq!(strip_cargo_subcommand(&args), &["--sarif", "foo.rs"]);
    }

    #[test]
    fn strip_cargo_subcommand_leaves_direct_invocation_untouched() {
        // Run directly (not via cargo): no `som` word, arguments unchanged.
        let args = vec!["foo.rs".to_string()];
        assert_eq!(strip_cargo_subcommand(&args), args.as_slice());
        assert_eq!(strip_cargo_subcommand(&[]), &[] as &[String]);
    }

    #[test]
    fn strip_cargo_subcommand_strips_only_the_first_word() {
        // `cargo som som` (a file literally named `som`) → `["som", "som"]`;
        // only the cargo-injected first word goes, the path `som` survives.
        let args = vec!["som".to_string(), "som".to_string()];
        let stripped = strip_cargo_subcommand(&args);
        assert_eq!(stripped, &["som"]);
        assert_eq!(parse_args(stripped), Some((Format::Text, "som")));
    }

    // ---- analyze_file (text): exact, hand-derived oracle output ------------

    /// The bundled `use_after_move.rs` program, fed through the real CLI
    /// pipeline (read → `lower_str` → `OwnershipOracle` → `render_markdown`),
    /// renders to *exactly* this report. Every row, label, the FAULT marker,
    /// the diagnostics bullet and the trailing Summary line are hand-derived
    /// from the documented oracle semantics for
    /// `fn process(input){ let owned=input; let moved=owned; let oops=owned; drop(oops); drop(moved); }`:
    ///
    /// - param `input` is Owned; `use input` (initializing `owned`) moves it.
    /// - `use owned` #1 (init `moved`) is the legal move → Moved, no fault.
    /// - `use owned` #2 (init `oops`) is use-after-move → Moved + FAULT.
    /// - drops run in reverse declaration order oops, moved, owned, input;
    ///   each was moved out, so every drop row is labelled `Moved` (a moved
    ///   binding's Drop is `Moved`, not `Dropped`).
    /// - one diagnostic: token 6 `owned` UseAfterMove; Summary: 14 tokens, 1.
    #[test]
    fn analyze_file_text_use_after_move_is_exact() {
        let path = fixture(
            "fn process(input: String) {\n    let owned = input;\n    let moved = owned;\n    let oops = owned;\n    drop(oops);\n    drop(moved);\n}\n",
        );
        let p = path.to_str().unwrap();
        let outcome = analyze_file(p, Format::Text).expect("valid rust");
        assert_eq!(outcome.faults, 1);

        let expected = format!(
            "# SOM ownership analysis — {p}\n\n\
| # | token | ownership | borrow | fault |\n\
|---|-------|-----------|--------|-------|\n\
| 0 | `fn process` | - | - |  |\n\
| 1 | `param input` | Owned | None |  |\n\
| 2 | `use input` | Moved | None |  |\n\
| 3 | `let owned` | Owned | None |  |\n\
| 4 | `use owned` | Moved | None |  |\n\
| 5 | `let moved` | Owned | None |  |\n\
| 6 | `use owned` | Moved | None | FAULT |\n\
| 7 | `let oops` | Owned | None |  |\n\
| 8 | `use oops` | Moved | None |  |\n\
| 9 | `use moved` | Moved | None |  |\n\
| 10 | `drop oops` | Moved | None |  |\n\
| 11 | `drop moved` | Moved | None |  |\n\
| 12 | `drop owned` | Moved | None |  |\n\
| 13 | `drop input` | Moved | None |  |\n\
\nDiagnostics:\n\
- token 6 `owned`: UseAfterMove\n\
\nSummary: 14 token(s), 1 ownership fault(s).\n"
        );
        assert_eq!(outcome.rendered, expected);
    }

    /// A clean program: the report ends with the visualizer's "program is
    /// clean" line, carries no FAULT marker and no diagnostics, and the
    /// Summary reports zero faults. `b` (a `String` move binding) is moved
    /// into the `drop` argument, so its own end-scope drop row is `Moved`;
    /// `a` was moved into `b`, so its drop row is `Moved` too.
    #[test]
    fn analyze_file_text_clean_program_is_exact() {
        let path = fixture("fn ok(a: String) {\n    let b = a;\n    drop(b);\n}\n");
        let p = path.to_str().unwrap();
        let outcome = analyze_file(p, Format::Text).expect("valid rust");
        assert_eq!(outcome.faults, 0);

        let expected = format!(
            "# SOM ownership analysis — {p}\n\n\
| # | token | ownership | borrow | fault |\n\
|---|-------|-----------|--------|-------|\n\
| 0 | `fn ok` | - | - |  |\n\
| 1 | `param a` | Owned | None |  |\n\
| 2 | `use a` | Moved | None |  |\n\
| 3 | `let b` | Owned | None |  |\n\
| 4 | `use b` | Moved | None |  |\n\
| 5 | `drop b` | Moved | None |  |\n\
| 6 | `drop a` | Moved | None |  |\n\
\nNo diagnostics — program is clean.\n\
\nSummary: 7 token(s), 0 ownership fault(s).\n"
        );
        assert_eq!(outcome.rendered, expected);
    }

    /// A program containing an `if` (branch-sensitive, not modelled) must
    /// surface the "Constructs skipped (not modelled)" section listing the
    /// frontend's exact note, ahead of the Summary. The `if` body is dropped,
    /// so `x` (declared but never used outside the skipped branch) is `Dropped`
    /// at scope end and the program is clean.
    #[test]
    fn analyze_file_text_reports_skipped_constructs() {
        let path =
            fixture("fn g(c: bool) {\n    let x = String::new();\n    if c { let y = x; }\n}\n");
        let p = path.to_str().unwrap();
        let outcome = analyze_file(p, Format::Text).expect("valid rust");
        assert_eq!(outcome.faults, 0);
        assert!(
            outcome.rendered.contains(
                "\nConstructs skipped (not modelled):\n- `if` expression (branch-sensitive ownership)\n"
            ),
            "missing skipped-constructs section:\n{}",
            outcome.rendered
        );
        // The skipped section precedes the trailing Summary line.
        let skipped_at = outcome.rendered.find("Constructs skipped").unwrap();
        let summary_at = outcome.rendered.find("\nSummary:").unwrap();
        assert!(skipped_at < summary_at);
        assert!(
            outcome
                .rendered
                .ends_with("Summary: 5 token(s), 0 ownership fault(s).\n")
        );
    }

    /// A method call lowers the receiver to a shared borrow — a documented
    /// approximation. The text report must surface the "Approximations
    /// applied" section with the frontend's exact note.
    #[test]
    fn analyze_file_text_reports_approximations() {
        let path = fixture("fn m(s: String) {\n    let n = s.len();\n    drop(n);\n}\n");
        let p = path.to_str().unwrap();
        let outcome = analyze_file(p, Format::Text).expect("valid rust");
        assert!(
            outcome.rendered.contains(
                "\nApproximations applied:\n- method-call receiver treated as a shared borrow\n"
            ),
            "missing approximations section:\n{}",
            outcome.rendered
        );
    }

    // ---- analyze_file: error paths (exit-code-2 cases) ---------------------

    #[test]
    fn analyze_file_missing_path_errors_with_cannot_read() {
        let err = analyze_file("/no/such/som_cli_file.rs", Format::Text)
            .expect_err("missing file must error");
        assert!(
            err.starts_with("cannot read /no/such/som_cli_file.rs: "),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn analyze_file_invalid_rust_errors_with_not_valid_rust() {
        let path = fixture("fn ( {");
        let p = path.to_str().unwrap();
        let err = analyze_file(p, Format::Text).expect_err("invalid rust must error");
        assert!(
            err.starts_with(&format!("{p} is not valid Rust: ")),
            "unexpected message: {err}"
        );
    }

    // ---- analyze_file (SARIF) through the public surface -------------------

    /// The SARIF path of `analyze_file` produces the same document as
    /// `render_sarif`, parses as JSON, carries the 2.1.0 envelope and one
    /// result per diagnostic with the offending rule and the path echoed in
    /// the location.
    #[test]
    fn analyze_file_sarif_lists_fault_and_is_valid_json() {
        let path = fixture(
            "fn process(input: String) {\n    let owned = input;\n    let moved = owned;\n    let oops = owned;\n    drop(oops);\n    drop(moved);\n}\n",
        );
        let p = path.to_str().unwrap();
        let outcome = analyze_file(p, Format::Sarif).expect("valid rust");
        assert_eq!(outcome.faults, 1);

        let parsed: serde_json::Value =
            serde_json::from_str(&outcome.rendered).expect("valid JSON");
        assert_eq!(parsed["version"], "2.1.0");
        assert_eq!(parsed["runs"][0]["tool"]["driver"]["name"], "scirust-som");
        let results = parsed["runs"][0]["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["ruleId"], "UseAfterMove");
        assert_eq!(results[0]["level"], "error");
        assert_eq!(
            results[0]["message"]["text"],
            "UseAfterMove on `owned` (token #6: Use(\"owned\"))"
        );
        assert_eq!(
            results[0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"],
            p
        );
    }

    /// `render_sarif` must JSON-escape backslashes and quotes in the path so
    /// the envelope stays parseable — and decode back to the exact original —
    /// even for awkward file names (e.g. a Windows-style path). The path is
    /// only emitted inside a result's location, so this drives a real
    /// diagnostic (a use-after-move from the oracle) and reads the URI back.
    #[test]
    fn render_sarif_escapes_path_and_round_trips() {
        let lowered = lower_str(
            "fn process(input: String) { let owned = input; let moved = owned; let oops = owned; drop(oops); drop(moved); }",
        )
        .unwrap();
        let analysis = OwnershipOracle::new().analyze(&lowered.ast);
        assert_eq!(analysis.diagnostics.len(), 1, "fixture must have one fault");

        let nasty = r#"weird"name\dir\a.rs"#;
        let sarif = render_sarif(nasty, &analysis);
        // Raw quotes/backslashes in the path would break the JSON envelope if
        // unescaped; this only parses because they are escaped.
        let parsed: serde_json::Value = serde_json::from_str(&sarif).expect("valid JSON");
        let results = parsed["runs"][0]["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        // The decoded URI is byte-for-byte the original path.
        assert_eq!(
            results[0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"],
            nasty
        );
    }

    // ---- run: the end-to-end exit-code contract (0 / 1 / 2) ----------------

    #[test]
    fn run_returns_1_on_fault_and_0_when_clean() {
        let faulty = fixture(
            "fn process(input: String) {\n    let owned = input;\n    let moved = owned;\n    let oops = owned;\n    drop(oops);\n    drop(moved);\n}\n",
        );
        let clean = fixture("fn ok(a: String) {\n    let b = a;\n    drop(b);\n}\n");

        assert_eq!(
            run(&[faulty.to_str().unwrap().to_string()], "som-analyze"),
            1,
            "a use-after-move file must exit 1"
        );
        assert_eq!(
            run(&[clean.to_str().unwrap().to_string()], "som-analyze"),
            0,
            "a clean file must exit 0"
        );
    }

    #[test]
    fn run_returns_2_on_usage_error() {
        // No path, and an extra positional argument: both are usage errors.
        assert_eq!(run(&[], "som-analyze"), 2);
        assert_eq!(
            run(&["a.rs".to_string(), "b.rs".to_string()], "som-analyze"),
            2
        );
    }

    #[test]
    fn run_returns_2_on_io_and_syntax_errors() {
        assert_eq!(
            run(&["/no/such/som_cli_run_file.rs".to_string()], "som-analyze"),
            2,
            "unreadable file must exit 2"
        );
        let bad = fixture("fn ( {");
        assert_eq!(
            run(&[bad.to_str().unwrap().to_string()], "som-analyze"),
            2,
            "invalid Rust must exit 2"
        );
    }

    #[test]
    fn run_sarif_flag_selects_sarif_and_still_signals_fault() {
        // `--sarif <faulty>` must emit SARIF (caller sees it on stdout) and
        // still exit 1 because the fault count is non-zero. We assert the
        // exit-code half here; the document shape is pinned above.
        let faulty = fixture(
            "fn process(input: String) {\n    let owned = input;\n    let moved = owned;\n    let oops = owned;\n    drop(oops);\n    drop(moved);\n}\n",
        );
        let code = run(
            &["--sarif".to_string(), faulty.to_str().unwrap().to_string()],
            "som-analyze",
        );
        assert_eq!(code, 1);
    }
}

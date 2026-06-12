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
        r#"{{"$schema":"https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json","version":"2.1.0","runs":[{{"tool":{{"driver":{{"name":"scirust-som","informationUri":"https://github.com/CHECKUPAUTO/scirust","version":"0.1.0"}}}},"results":[{results}]}}]}}"#
    )
}

/// Shared CLI driver: parse `[--sarif] <file.rs>` and print the report.
/// Returns the process exit code (0 clean, 1 faults, 2 usage/IO/syntax).
pub fn run(args: &[String], tool_name: &str) -> u8 {
    let mut format = Format::Text;
    let mut path: Option<&String> = None;
    for a in args
    {
        match a.as_str()
        {
            "--sarif" => format = Format::Sarif,
            _ if path.is_none() => path = Some(a),
            _ =>
            {
                eprintln!("usage: {tool_name} [--sarif] <file.rs>");
                return 2;
            },
        }
    }
    let Some(path) = path
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

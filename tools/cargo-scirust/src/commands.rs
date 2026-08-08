use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use serde_json::json;

use crate::workspace::{Impact, Workspace};
use crate::{AppError, AppResult};

pub fn affected(ws: &Workspace, args: &[String]) -> AppResult<()> {
    let opts = ImpactOptions::parse(args)?;
    let impact = ws.impact(opts.base.as_deref(), opts.head.as_deref())?;
    print_impact(&impact, opts.json)
}

pub fn check(ws: &Workspace, args: &[String]) -> AppResult<()> {
    let mut base = None;
    let mut head = None;
    let mut dry_run = false;
    let mut all = false;
    let mut full = false;
    let mut fmt = true;
    let mut clippy = true;
    let mut tests = true;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--base" => base = Some(value_after(args, &mut i, "--base")?),
            "--head" => head = Some(value_after(args, &mut i, "--head")?),
            "--dry-run" => dry_run = true,
            "--all" => all = true,
            "--full" => full = true,
            "--no-fmt" => fmt = false,
            "--no-clippy" => clippy = false,
            "--no-test" | "--no-tests" => tests = false,
            "-h" | "--help" => {
                println!(
                    "cargo scirust check [--base REF] [--head REF] [--all] [--full] [--dry-run] [--no-fmt] [--no-clippy] [--no-test]"
                );
                println!(
                    "  default: fmt + clippy + tests only for transitively affected workspace crates"
                );
                println!("  --full: additionally run the workspace MSRV (Rust 1.89) cargo check");
                return Ok(());
            },
            other => return Err(AppError::message(format!("unknown check option: {other}"))),
        }
        i += 1;
    }

    let impact = ws.impact(base.as_deref(), head.as_deref())?;
    let packages = if all {
        ws.package_names()
    } else {
        impact.affected.clone()
    };

    println!("SciRust check plan");
    println!("  base: {}", impact.base);
    if let Some(head) = &impact.head {
        println!("  head: {head}");
    }
    println!("  changed files: {}", impact.changed_files.len());
    println!("  selected crates: {}", packages.len());

    if fmt {
        run_command(
            &ws.root,
            cargo_args(["fmt", "--all", "--", "--check"]),
            dry_run,
        )?;
    }

    if packages.is_empty() {
        println!("No workspace crate is affected; package gates skipped.");
        return Ok(());
    }

    if clippy {
        let mut command = vec!["clippy".to_string()];
        push_packages(&mut command, &packages);
        command.extend([
            "--all-targets".into(),
            "--".into(),
            "-D".into(),
            "warnings".into(),
        ]);
        run_command(&ws.root, cargo_vec(command), dry_run)?;
    }

    if tests {
        let mut command = vec!["test".to_string()];
        push_packages(&mut command, &packages);
        run_command(&ws.root, cargo_vec(command), dry_run)?;
    }

    if full {
        let mut command = vec!["+1.89.0".to_string(), "check".to_string()];
        push_packages(&mut command, &packages);
        command.push("--all-targets".into());
        run_command(&ws.root, cargo_vec(command), dry_run)?;
    }

    Ok(())
}

pub fn features(ws: &Workspace, args: &[String]) -> AppResult<()> {
    if args.iter().any(|arg| arg == "-h" || arg == "--help") || args.is_empty() {
        println!("cargo scirust features <package> [--cover pairwise] [--execute] [--max N]");
        println!(
            "Lists features or generates deterministic no-default single/pair feature checks."
        );
        return Ok(());
    }

    let package_name = &args[0];
    let package = ws
        .package(package_name)
        .ok_or_else(|| AppError::message(format!("unknown workspace package: {package_name}")))?;
    let mut cover = None;
    let mut execute = false;
    let mut max_cases = 64usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--cover" => cover = Some(value_after(args, &mut i, "--cover")?),
            "--execute" => execute = true,
            "--max" => {
                max_cases = value_after(args, &mut i, "--max")?
                    .parse()
                    .map_err(|_| AppError::message("--max expects a positive integer"))?;
            },
            other => {
                return Err(AppError::message(format!(
                    "unknown features option: {other}"
                )));
            },
        }
        i += 1;
    }

    let features: Vec<String> = package
        .features
        .keys()
        .filter(|name| name.as_str() != "default")
        .cloned()
        .collect();

    if cover.is_none() {
        println!("Features for {} ({}):", package.name, features.len());
        for feature in &features {
            let expands = package
                .features
                .get(feature)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if expands.is_empty() {
                println!("  {feature}");
            } else {
                println!("  {feature} -> {}", expands.join(", "));
            }
        }
        return Ok(());
    }

    if cover.as_deref() != Some("pairwise") {
        return Err(AppError::message(
            "only --cover pairwise is currently supported",
        ));
    }

    let mut cases = Vec::<Vec<String>>::new();
    cases.push(Vec::new());
    for feature in &features {
        cases.push(vec![feature.clone()]);
    }
    for left in 0..features.len() {
        for right in left + 1..features.len() {
            cases.push(vec![features[left].clone(), features[right].clone()]);
        }
    }

    if cases.len() > max_cases {
        return Err(AppError::message(format!(
            "pairwise plan has {} cases, exceeding --max {max_cases}; raise --max deliberately",
            cases.len()
        )));
    }

    println!(
        "Pairwise feature plan for {}: {} cases",
        package.name,
        cases.len()
    );
    for (index, case) in cases.iter().enumerate() {
        let label = if case.is_empty() {
            "<no features>".to_string()
        } else {
            case.join(",")
        };
        println!("  {:>3}. {label}", index + 1);
        if execute {
            let mut command = vec![
                "check".to_string(),
                "-p".to_string(),
                package.name.clone(),
                "--no-default-features".to_string(),
            ];
            if !case.is_empty() {
                command.push("--features".to_string());
                command.push(case.join(","));
            }
            run_command(&ws.root, cargo_vec(command), false)?;
        }
    }
    Ok(())
}

pub fn bench(ws: &Workspace, args: &[String]) -> AppResult<()> {
    let mut base = None;
    let mut head = None;
    let mut dry_run = false;
    let mut all = false;
    let mut package = None;
    let mut passthrough = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--base" => base = Some(value_after(args, &mut i, "--base")?),
            "--head" => head = Some(value_after(args, &mut i, "--head")?),
            "--dry-run" => dry_run = true,
            "--all" => all = true,
            "--package" | "-p" => package = Some(value_after(args, &mut i, "--package")?),
            "--" => {
                passthrough.extend_from_slice(&args[i + 1..]);
                break;
            },
            "-h" | "--help" => {
                println!(
                    "cargo scirust bench [--base REF] [--head REF] [--all] [-p PACKAGE] [--dry-run] [-- <cargo bench args>]"
                );
                return Ok(());
            },
            other => return Err(AppError::message(format!("unknown bench option: {other}"))),
        }
        i += 1;
    }

    let impact = ws.impact(base.as_deref(), head.as_deref())?;
    let packages = if let Some(package) = package {
        if ws.package(&package).is_none() {
            return Err(AppError::message(format!(
                "unknown workspace package: {package}"
            )));
        }
        vec![package]
    } else if all {
        ws.package_names()
    } else {
        impact.affected
    };

    if packages.is_empty() {
        println!("No affected workspace crate; no benchmark command needed.");
        return Ok(());
    }

    let mut command = vec!["bench".to_string()];
    push_packages(&mut command, &packages);
    command.extend(passthrough);
    run_command(&ws.root, cargo_vec(command), dry_run)
}

pub fn parity(ws: &Workspace, args: &[String]) -> AppResult<()> {
    let mut left = None;
    let mut right = None;
    let mut ignore_stderr = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--left" => left = Some(value_after(args, &mut i, "--left")?),
            "--right" => right = Some(value_after(args, &mut i, "--right")?),
            "--ignore-stderr" => ignore_stderr = true,
            "-h" | "--help" => {
                println!(
                    "cargo scirust parity --left \"COMMAND\" --right \"COMMAND\" [--ignore-stderr]"
                );
                println!(
                    "Runs both commands from the SciRust root and requires equal exit code/stdout[/stderr]."
                );
                return Ok(());
            },
            other => return Err(AppError::message(format!("unknown parity option: {other}"))),
        }
        i += 1;
    }

    let left = left.ok_or_else(|| AppError::message("parity requires --left \"COMMAND\""))?;
    let right = right.ok_or_else(|| AppError::message("parity requires --right \"COMMAND\""))?;

    println!("Parity left : {left}");
    let left_out = shell_capture(&ws.root, &left, None)?;
    println!("Parity right: {right}");
    let right_out = shell_capture(&ws.root, &right, None)?;

    let exit_equal = left_out.status.code() == right_out.status.code();
    let stdout_equal =
        normalize_newlines(&left_out.stdout) == normalize_newlines(&right_out.stdout);
    let stderr_equal = ignore_stderr
        || normalize_newlines(&left_out.stderr) == normalize_newlines(&right_out.stderr);

    println!("  exit:   {}", verdict(exit_equal));
    println!(
        "  stdout: {}  left={} right={}",
        verdict(stdout_equal),
        fingerprint(&left_out.stdout),
        fingerprint(&right_out.stdout)
    );
    if !ignore_stderr {
        println!(
            "  stderr: {}  left={} right={}",
            verdict(stderr_equal),
            fingerprint(&left_out.stderr),
            fingerprint(&right_out.stderr)
        );
    }

    if exit_equal && stdout_equal && stderr_equal {
        println!("PARITY PASS");
        Ok(())
    } else {
        Err(AppError::message("PARITY FAIL"))
    }
}

pub fn determinism(ws: &Workspace, args: &[String]) -> AppResult<()> {
    let mut repeat = 3usize;
    let mut command_start = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--repeat" => {
                repeat = value_after(args, &mut i, "--repeat")?
                    .parse()
                    .map_err(|_| AppError::message("--repeat expects an integer >= 2"))?;
            },
            "--" => {
                command_start = Some(i + 1);
                break;
            },
            "-h" | "--help" => {
                println!("cargo scirust determinism [--repeat N] -- PROGRAM [ARGS...]");
                println!(
                    "Compares exact exit code/stdout/stderr and emits stable FNV-1a fingerprints."
                );
                return Ok(());
            },
            other => {
                return Err(AppError::message(format!(
                    "unknown determinism option before --: {other}"
                )));
            },
        }
        i += 1;
    }

    if repeat < 2 {
        return Err(AppError::message("--repeat must be at least 2"));
    }
    let start = command_start
        .ok_or_else(|| AppError::message("determinism requires `-- PROGRAM [ARGS...]`"))?;
    let command = &args[start..];
    if command.is_empty() {
        return Err(AppError::message("determinism command is empty"));
    }

    let mut baseline: Option<Output> = None;
    for run in 1..=repeat {
        let output = direct_capture(&ws.root, command, Some(run))?;
        println!(
            "  run {run}/{repeat}: exit={:?} stdout={} stderr={}",
            output.status.code(),
            fingerprint(&output.stdout),
            fingerprint(&output.stderr)
        );
        if let Some(first) = &baseline {
            if first.status.code() != output.status.code()
                || normalize_newlines(&first.stdout) != normalize_newlines(&output.stdout)
                || normalize_newlines(&first.stderr) != normalize_newlines(&output.stderr)
            {
                return Err(AppError::message(format!("DETERMINISM FAIL at run {run}")));
            }
        } else {
            baseline = Some(output);
        }
    }
    println!("DETERMINISM PASS ({repeat} exact runs)");
    Ok(())
}

pub fn cost(ws: &Workspace, args: &[String]) -> AppResult<()> {
    let mut path = None;
    let mut package = None;
    let mut json_output = false;
    let mut limit = 30usize;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--path" => path = Some(PathBuf::from(value_after(args, &mut i, "--path")?)),
            "--package" | "-p" => package = Some(value_after(args, &mut i, "--package")?),
            "--json" => json_output = true,
            "--limit" => {
                limit = value_after(args, &mut i, "--limit")?
                    .parse()
                    .map_err(|_| AppError::message("--limit expects an integer"))?;
            },
            "-h" | "--help" => {
                println!("cargo scirust cost [--path PATH | -p PACKAGE] [--json] [--limit N]");
                println!(
                    "Static heuristic: reports copy/allocation/transfer/synchronization indicators; it does not pretend to be a profiler."
                );
                return Ok(());
            },
            other => return Err(AppError::message(format!("unknown cost option: {other}"))),
        }
        i += 1;
    }

    let root = match (path, package) {
        (Some(_), Some(_)) => return Err(AppError::message("use only one of --path or --package")),
        (Some(path), None) => {
            if path.is_absolute() {
                path
            } else {
                ws.root.join(path)
            }
        },
        (None, Some(name)) => ws
            .package(&name)
            .ok_or_else(|| AppError::message(format!("unknown workspace package: {name}")))?
            .dir
            .clone(),
        (None, None) => ws.root.clone(),
    };

    let findings = scan_cost(&root)?;
    let mut counts = BTreeMap::<&'static str, usize>::new();
    for finding in &findings {
        *counts.entry(finding.kind).or_default() += 1;
    }

    if json_output {
        let details: Vec<_> = findings
            .iter()
            .take(limit)
            .map(|finding| json!({"kind": finding.kind, "path": finding.path, "line": finding.line, "snippet": finding.snippet}))
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "root": root,
                "heuristic": true,
                "total_indicators": findings.len(),
                "counts": counts,
                "findings": details,
            }))
            .map_err(|err| AppError::message(err.to_string()))?
        );
        return Ok(());
    }

    println!("Static cost indicators under {}", root.display());
    println!("  NOTE: indicators are source heuristics, not measured runtime costs.");
    for (kind, count) in &counts {
        println!("  {kind:<24} {count:>6}");
    }
    println!("  total                    {:>6}", findings.len());
    for finding in findings.iter().take(limit) {
        println!(
            "  {}:{} [{}] {}",
            finding.path.display(),
            finding.line,
            finding.kind,
            finding.snippet.trim()
        );
    }
    if findings.len() > limit {
        println!(
            "  ... {} more (raise --limit to display)",
            findings.len() - limit
        );
    }
    Ok(())
}

pub fn calibrate(_ws: &Workspace, args: &[String]) -> AppResult<()> {
    let mut pieces = None;
    let mut lengths = None;
    let mut json_output = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--pieces" => pieces = Some(PathBuf::from(value_after(args, &mut i, "--pieces")?)),
            "--lengths" => lengths = Some(PathBuf::from(value_after(args, &mut i, "--lengths")?)),
            "--json" => json_output = true,
            "-h" | "--help" => {
                println!("cargo scirust calibrate (--pieces FILE | --lengths FILE) [--json]");
                println!(
                    "Derives S/M/L/XL/XXL/XXXL byte-size classes from observed tokenizer pieces without changing BPE semantics."
                );
                println!("  --pieces: one decoded token piece per line; byte length is measured");
                println!("  --lengths: one positive byte length per line");
                return Ok(());
            },
            other => {
                return Err(AppError::message(format!(
                    "unknown calibrate option: {other}"
                )));
            },
        }
        i += 1;
    }

    let mut values = match (pieces, lengths) {
        (Some(_), Some(_)) | (None, None) => {
            return Err(AppError::message(
                "calibrate requires exactly one of --pieces or --lengths",
            ));
        },
        (Some(path), None) => read_piece_lengths(&path)?,
        (None, Some(path)) => read_numeric_lengths(&path)?,
    };
    if values.is_empty() {
        return Err(AppError::message(
            "calibration input contains no positive piece lengths",
        ));
    }
    values.sort_unstable();

    // Five equal-frequency cuts produce six deterministic classes. The cuts are
    // learned only from observed piece sizes; token identities and merge ranks
    // remain untouched.
    let cuts =
        [1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0, 4.0 / 6.0, 5.0 / 6.0].map(|q| nearest_rank(&values, q));
    let sum: usize = values.iter().sum();
    let mean = sum as f64 / values.len() as f64;

    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "count": values.len(),
                "min_bytes": values[0],
                "max_bytes": values[values.len() - 1],
                "mean_bytes": mean,
                "boundaries": {
                    "S_max": cuts[0], "M_max": cuts[1], "L_max": cuts[2],
                    "XL_max": cuts[3], "XXL_max": cuts[4], "XXXL": format!(">{}", cuts[4])
                },
                "method": "equal-frequency-nearest-rank-v1"
            }))
            .map_err(|err| AppError::message(err.to_string()))?
        );
    } else {
        println!("ElasticTokenizer size calibration (equal-frequency nearest-rank v1)");
        println!("  observations: {}", values.len());
        println!(
            "  bytes: min={} mean={mean:.3} max={}",
            values[0],
            values[values.len() - 1]
        );
        println!("  S     <= {} bytes", cuts[0]);
        println!("  M     <= {} bytes", cuts[1]);
        println!("  L     <= {} bytes", cuts[2]);
        println!("  XL    <= {} bytes", cuts[3]);
        println!("  XXL   <= {} bytes", cuts[4]);
        println!("  XXXL   > {} bytes", cuts[4]);
        println!("  BPE merge ranks/token identities are not modified.");
    }
    Ok(())
}

#[derive(Default)]
struct ImpactOptions {
    base: Option<String>,
    head: Option<String>,
    json: bool,
}

impl ImpactOptions {
    fn parse(args: &[String]) -> AppResult<Self> {
        let mut options = Self::default();
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--base" => options.base = Some(value_after(args, &mut i, "--base")?),
                "--head" => options.head = Some(value_after(args, &mut i, "--head")?),
                "--json" => options.json = true,
                "-h" | "--help" => {
                    println!("cargo scirust affected [--base REF] [--head REF] [--json]");
                    return Ok(options);
                },
                other => {
                    return Err(AppError::message(format!(
                        "unknown affected option: {other}"
                    )));
                },
            }
            i += 1;
        }
        Ok(options)
    }
}

fn print_impact(impact: &Impact, json_output: bool) -> AppResult<()> {
    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "base": impact.base,
                "head": impact.head,
                "global_change": impact.global_change,
                "changed_files": impact.changed_files,
                "direct": impact.direct,
                "affected": impact.affected,
            }))
            .map_err(|err| AppError::message(err.to_string()))?
        );
        return Ok(());
    }

    println!("SciRust affected analysis");
    println!("  base: {}", impact.base);
    if let Some(head) = &impact.head {
        println!("  head: {head}");
    }
    println!("  changed files: {}", impact.changed_files.len());
    println!("  global workspace change: {}", impact.global_change);
    println!("Directly affected ({}):", impact.direct.len());
    for package in &impact.direct {
        println!("  {package}");
    }
    println!("Transitively affected ({}):", impact.affected.len());
    for package in &impact.affected {
        println!("  {package}");
    }
    Ok(())
}

fn push_packages(command: &mut Vec<String>, packages: &[String]) {
    for package in packages {
        command.push("-p".to_string());
        command.push(package.clone());
    }
}

fn cargo_args<const N: usize>(args: [&str; N]) -> Vec<String> {
    let mut command = vec!["cargo".to_string()];
    command.extend(args.into_iter().map(str::to_string));
    command
}

fn cargo_vec(args: Vec<String>) -> Vec<String> {
    let mut command = vec!["cargo".to_string()];
    command.extend(args);
    command
}

fn run_command(root: &Path, command: Vec<String>, dry_run: bool) -> AppResult<()> {
    println!("$ {}", printable_command(&command));
    if dry_run {
        return Ok(());
    }
    let (program, args) = command
        .split_first()
        .ok_or_else(|| AppError::message("internal error: empty command"))?;
    let status = Command::new(program)
        .current_dir(root)
        .args(args)
        .status()
        .map_err(AppError::io)?;
    if status.success() {
        Ok(())
    } else {
        Err(AppError::message(format!(
            "command failed with status {:?}: {}",
            status.code(),
            printable_command(&command)
        )))
    }
}

fn printable_command(command: &[String]) -> String {
    command
        .iter()
        .map(|arg| {
            if arg
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "-_=./,:+".contains(c))
            {
                arg.clone()
            } else {
                format!("{:?}", arg)
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn value_after(args: &[String], i: &mut usize, flag: &str) -> AppResult<String> {
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| AppError::message(format!("{flag} requires a value")))
}

fn shell_capture(root: &Path, command: &str, run: Option<usize>) -> AppResult<Output> {
    #[cfg(windows)]
    let mut child = {
        let mut cmd = Command::new("cmd");
        cmd.args(["/C", command]);
        cmd
    };
    #[cfg(not(windows))]
    let mut child = {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", command]);
        cmd
    };
    child.current_dir(root).stdin(Stdio::null());
    if let Some(run) = run {
        child.env("SCIRUST_DETERMINISM_RUN", run.to_string());
    }
    child.output().map_err(AppError::io)
}

fn direct_capture(root: &Path, command: &[String], run: Option<usize>) -> AppResult<Output> {
    let (program, args) = command
        .split_first()
        .ok_or_else(|| AppError::message("empty command"))?;
    let mut child = Command::new(program);
    child.current_dir(root).args(args).stdin(Stdio::null());
    if let Some(run) = run {
        child.env("SCIRUST_DETERMINISM_RUN", run.to_string());
    }
    child.output().map_err(AppError::io)
}

fn normalize_newlines(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\r' && bytes.get(i + 1) == Some(&b'\n') {
            out.push(b'\n');
            i += 2;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    out
}

fn fingerprint(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

fn verdict(ok: bool) -> &'static str {
    if ok { "PASS" } else { "FAIL" }
}

#[derive(Debug)]
struct CostFinding {
    kind: &'static str,
    path: PathBuf,
    line: usize,
    snippet: String,
}

fn scan_cost(root: &Path) -> AppResult<Vec<CostFinding>> {
    let mut files = Vec::new();
    collect_rust_files(root, &mut files)?;
    files.sort();
    let indicators: &[(&str, &str)] = &[
        ("clone-copy-candidate", ".clone()"),
        ("owned-copy-candidate", ".to_owned()"),
        ("vec-copy-allocation", ".to_vec()"),
        ("vec-allocation", "Vec::with_capacity("),
        ("vec-allocation", "Vec::new("),
        ("collect-vec-allocation", "collect::<Vec"),
        ("contiguous-materialize", "contiguous("),
        ("gpu-upload-indicator", "write_buffer("),
        ("gpu-readback-indicator", "map_async("),
        ("gpu-readback-indicator", "read_buffer("),
        ("host-sync-indicator", "device.poll("),
        ("host-sync-indicator", ".wait("),
    ];

    let mut findings = Vec::new();
    for file in files {
        let text = match fs::read_to_string(&file) {
            Ok(text) => text,
            Err(_) => continue,
        };
        for (line_index, line) in text.lines().enumerate() {
            for &(kind, needle) in indicators {
                if line.contains(needle) {
                    findings.push(CostFinding {
                        kind,
                        path: file.clone(),
                        line: line_index + 1,
                        snippet: line.to_string(),
                    });
                }
            }
        }
    }
    Ok(findings)
}

fn collect_rust_files(path: &Path, output: &mut Vec<PathBuf>) -> AppResult<()> {
    if path.is_file() {
        if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            output.push(path.to_path_buf());
        }
        return Ok(());
    }
    let entries = fs::read_dir(path).map_err(AppError::io)?;
    for entry in entries {
        let entry = entry.map_err(AppError::io)?;
        let path = entry.path();
        let name = entry.file_name();
        if path.is_dir() {
            if matches!(
                name.to_str(),
                Some("target" | ".git" | "data" | "node_modules")
            ) {
                continue;
            }
            collect_rust_files(&path, output)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            output.push(path);
        }
    }
    Ok(())
}

fn read_piece_lengths(path: &Path) -> AppResult<Vec<usize>> {
    let text = fs::read_to_string(path).map_err(AppError::io)?;
    Ok(text
        .split_terminator('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line).len())
        .filter(|&length| length > 0)
        .collect())
}

fn read_numeric_lengths(path: &Path) -> AppResult<Vec<usize>> {
    let text = fs::read_to_string(path).map_err(AppError::io)?;
    let mut out = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: usize = line.parse().map_err(|_| {
            AppError::message(format!(
                "invalid positive integer at {}:{}",
                path.display(),
                index + 1
            ))
        })?;
        if value > 0 {
            out.push(value);
        }
    }
    Ok(out)
}

fn nearest_rank(sorted: &[usize], q: f64) -> usize {
    let rank = (q * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_is_stable() {
        assert_eq!(fingerprint(b"hello"), "fnv1a64:a430d84680aabd0b");
    }

    #[test]
    fn newline_normalization_is_cross_platform_only() {
        assert_eq!(normalize_newlines(b"a\r\nb\n"), b"a\nb\n");
        assert_eq!(normalize_newlines(b"a\rb"), b"a\rb");
    }

    #[test]
    fn nearest_rank_is_deterministic() {
        let values = [1, 2, 3, 4, 5, 6];
        assert_eq!(nearest_rank(&values, 1.0 / 6.0), 1);
        assert_eq!(nearest_rank(&values, 0.5), 3);
        assert_eq!(nearest_rank(&values, 5.0 / 6.0), 5);
    }
}

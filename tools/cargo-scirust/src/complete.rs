use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::Instant;

use serde_json::json;

use crate::workspace::{Impact, Workspace};
use crate::{AppError, AppResult};

pub fn affected(ws: &Workspace, args: &[String]) -> AppResult<()> {
    let mut base = None;
    let mut head = None;
    let mut json_output = false;
    let mut names_only = false;
    let mut direct_only = false;
    let mut fail_if_empty = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--base" => base = Some(value_after(args, &mut i, "--base")?),
            "--head" => head = Some(value_after(args, &mut i, "--head")?),
            "--json" => json_output = true,
            "--names-only" => names_only = true,
            "--direct-only" => direct_only = true,
            "--fail-if-empty" => fail_if_empty = true,
            "-h" | "--help" => {
                println!("cargo scirust affected [--base REF] [--head REF] [--json] [--names-only] [--direct-only] [--fail-if-empty]");
                return Ok(());
            },
            other => return Err(AppError::message(format!("unknown affected option: {other}"))),
        }
        i += 1;
    }

    if json_output && names_only {
        return Err(AppError::message("--json and --names-only are mutually exclusive"));
    }

    let impact = ws.impact(base.as_deref(), head.as_deref())?;
    let selected = if direct_only {
        &impact.direct
    } else {
        &impact.affected
    };
    if fail_if_empty && selected.is_empty() {
        return Err(AppError::message("no affected workspace crate"));
    }

    if names_only {
        for package in selected {
            println!("{package}");
        }
        return Ok(());
    }
    print_impact(&impact, json_output, direct_only)
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
    let mut locked = true;
    let mut all_features = false;

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
            "--unlocked" => locked = false,
            "--all-features" => all_features = true,
            "-h" | "--help" => {
                println!("cargo scirust check [--base REF] [--head REF] [--all] [--full] [--dry-run] [--all-features] [--unlocked] [--no-fmt] [--no-clippy] [--no-test]");
                println!("  default: fmt + locked clippy + locked tests for the transitive affected closure");
                println!("  --full: additionally check selected crates on Rust 1.89");
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
    println!("  Cargo.lock enforced: {locked}");

    if fmt {
        run_command(
            &ws.root,
            vec![
                "cargo".into(),
                "fmt".into(),
                "--all".into(),
                "--".into(),
                "--check".into(),
            ],
            dry_run,
        )?;
    }

    if packages.is_empty() {
        println!("No workspace crate is affected; package gates skipped.");
        return Ok(());
    }

    if clippy {
        let mut command = vec!["cargo".into(), "clippy".into()];
        if locked {
            command.push("--locked".into());
        }
        push_packages(&mut command, &packages);
        command.push("--all-targets".into());
        if all_features {
            command.push("--all-features".into());
        }
        command.extend(["--".into(), "-D".into(), "warnings".into()]);
        run_command(&ws.root, command, dry_run)?;
    }

    if tests {
        let mut command = vec!["cargo".into(), "test".into()];
        if locked {
            command.push("--locked".into());
        }
        push_packages(&mut command, &packages);
        if all_features {
            command.push("--all-features".into());
        }
        run_command(&ws.root, command, dry_run)?;
    }

    if full {
        let mut command = vec!["cargo".into(), "+1.89.0".into(), "check".into()];
        if locked {
            command.push("--locked".into());
        }
        push_packages(&mut command, &packages);
        command.push("--all-targets".into());
        if all_features {
            command.push("--all-features".into());
        }
        run_command(&ws.root, command, dry_run)?;
    }

    Ok(())
}

#[derive(Debug)]
struct FeatureCase {
    features: Vec<String>,
}

#[derive(Debug)]
struct FeatureResult {
    features: Vec<String>,
    success: bool,
    exit_code: Option<i32>,
    diagnostic: String,
}

pub fn features(ws: &Workspace, args: &[String]) -> AppResult<()> {
    if args.is_empty() || args.iter().any(|arg| matches!(arg.as_str(), "-h" | "--help")) {
        println!("cargo scirust features <package> [--cover pairwise] [--execute] [--max N] [--json] [--allow-incompatible] [--include-default]");
        println!("Executes the full baseline/single/pair matrix and classifies incompatible pairs instead of stopping at the first failure.");
        return Ok(());
    }

    let package_name = &args[0];
    let package = ws
        .package(package_name)
        .ok_or_else(|| AppError::message(format!("unknown workspace package: {package_name}")))?;
    let mut cover = None;
    let mut execute = false;
    let mut max_cases = 64usize;
    let mut json_output = false;
    let mut allow_incompatible = false;
    let mut include_default = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--cover" => cover = Some(value_after(args, &mut i, "--cover")?),
            "--execute" => execute = true,
            "--max" => {
                max_cases = parse_positive_usize(&value_after(args, &mut i, "--max")?, "--max")?;
            },
            "--json" => json_output = true,
            "--allow-incompatible" => allow_incompatible = true,
            "--include-default" => include_default = true,
            other => return Err(AppError::message(format!("unknown features option: {other}"))),
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
        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "package": package.name,
                    "features": package.features,
                }))
                .map_err(|error| AppError::message(error.to_string()))?
            );
        } else {
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
        }
        return Ok(());
    }

    if cover.as_deref() != Some("pairwise") {
        return Err(AppError::message("only --cover pairwise is supported"));
    }

    let mut cases = Vec::new();
    cases.push(FeatureCase { features: Vec::new() });
    for feature in &features {
        cases.push(FeatureCase {
            features: vec![feature.clone()],
        });
    }
    for left in 0..features.len() {
        for right in left + 1..features.len() {
            cases.push(FeatureCase {
                features: vec![features[left].clone(), features[right].clone()],
            });
        }
    }

    if cases.len() > max_cases {
        return Err(AppError::message(format!(
            "pairwise plan has {} cases, exceeding --max {max_cases}; raise --max deliberately",
            cases.len()
        )));
    }

    if !execute {
        if json_output {
            let plan = cases
                .iter()
                .map(|case| case.features.clone())
                .collect::<Vec<_>>();
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "package": package.name,
                    "cases": plan,
                    "count": cases.len(),
                }))
                .map_err(|error| AppError::message(error.to_string()))?
            );
        } else {
            println!("Pairwise feature plan for {}: {} cases", package.name, cases.len());
            for (index, case) in cases.iter().enumerate() {
                println!("  {:>3}. {}", index + 1, feature_label(&case.features));
            }
        }
        return Ok(());
    }

    let mut results = Vec::with_capacity(cases.len());
    for (index, case) in cases.iter().enumerate() {
        if !json_output {
            println!("  {:>3}/{} {}", index + 1, cases.len(), feature_label(&case.features));
        }
        let mut command = vec![
            "cargo".to_string(),
            "check".to_string(),
            "--locked".to_string(),
            "-p".to_string(),
            package.name.clone(),
            "--all-targets".to_string(),
        ];
        if !include_default {
            command.push("--no-default-features".to_string());
        }
        if !case.features.is_empty() {
            command.push("--features".to_string());
            command.push(case.features.join(","));
        }
        let output = capture_command(&ws.root, &command)?;
        results.push(FeatureResult {
            features: case.features.clone(),
            success: output.status.success(),
            exit_code: output.status.code(),
            diagnostic: diagnostic_tail(&output.stderr, 6),
        });
    }

    let singles: BTreeMap<&str, bool> = results
        .iter()
        .filter(|result| result.features.len() == 1)
        .map(|result| (result.features[0].as_str(), result.success))
        .collect();
    let incompatible_pairs = results
        .iter()
        .filter(|result| result.features.len() == 2 && !result.success)
        .filter(|result| {
            singles.get(result.features[0].as_str()) == Some(&true)
                && singles.get(result.features[1].as_str()) == Some(&true)
        })
        .map(|result| result.features.clone())
        .collect::<Vec<_>>();
    let failed = results.iter().filter(|result| !result.success).count();

    if json_output {
        let rendered = results
            .iter()
            .map(|result| {
                json!({
                    "features": result.features,
                    "success": result.success,
                    "exit_code": result.exit_code,
                    "diagnostic": result.diagnostic,
                })
            })
            .collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "package": package.name,
                "cases": rendered,
                "failed_cases": failed,
                "incompatible_pairs": incompatible_pairs,
            }))
            .map_err(|error| AppError::message(error.to_string()))?
        );
    } else {
        println!("Feature matrix complete: {} pass, {} fail", results.len() - failed, failed);
        if incompatible_pairs.is_empty() {
            println!("  no pair-specific incompatibility detected");
        } else {
            println!("  pair-specific incompatibilities:");
            for pair in &incompatible_pairs {
                println!("    {} + {}", pair[0], pair[1]);
            }
        }
        for result in results.iter().filter(|result| !result.success) {
            println!("  FAIL {} (exit {:?})", feature_label(&result.features), result.exit_code);
            if !result.diagnostic.is_empty() {
                println!("{}", indent(&result.diagnostic, "      "));
            }
        }
    }

    if failed > 0 && !allow_incompatible {
        Err(AppError::message(format!(
            "feature matrix has {failed} failing case(s); use --allow-incompatible only when failures are expected"
        )))
    } else {
        Ok(())
    }
}

pub fn bench(ws: &Workspace, args: &[String]) -> AppResult<()> {
    let mut base = None;
    let mut head = None;
    let mut dry_run = false;
    let mut all = false;
    let mut package = None;
    let mut repeat = 1usize;
    let mut passthrough = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--base" => base = Some(value_after(args, &mut i, "--base")?),
            "--head" => head = Some(value_after(args, &mut i, "--head")?),
            "--dry-run" => dry_run = true,
            "--all" => all = true,
            "--package" | "-p" => package = Some(value_after(args, &mut i, "--package")?),
            "--repeat" => {
                repeat = parse_positive_usize(&value_after(args, &mut i, "--repeat")?, "--repeat")?;
            },
            "--" => {
                passthrough.extend_from_slice(&args[i + 1..]);
                break;
            },
            "-h" | "--help" => {
                println!("cargo scirust bench [--base REF] [--head REF] [--all] [-p PACKAGE] [--repeat N] [--dry-run] [-- <cargo bench args>]");
                return Ok(());
            },
            other => return Err(AppError::message(format!("unknown bench option: {other}"))),
        }
        i += 1;
    }

    let impact = ws.impact(base.as_deref(), head.as_deref())?;
    let packages = if let Some(package) = package {
        if ws.package(&package).is_none() {
            return Err(AppError::message(format!("unknown workspace package: {package}")));
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

    let mut command = vec!["cargo".to_string(), "bench".to_string(), "--locked".to_string()];
    push_packages(&mut command, &packages);
    command.extend(passthrough);

    if dry_run {
        println!("$ {}", printable_command(&command));
        return Ok(());
    }

    let mut timings = Vec::with_capacity(repeat);
    for run in 1..=repeat {
        println!("Benchmark run {run}/{repeat}");
        timings.push(run_timed_command(&ws.root, &command, true)?);
    }
    if repeat > 1 {
        print_timing_summary("cargo bench wall time", &timings);
    }
    Ok(())
}

pub fn parity(ws: &Workspace, args: &[String]) -> AppResult<()> {
    let mut left = None;
    let mut right = None;
    let mut ignore_stderr = false;
    let mut allow_failure = false;
    let mut repeat = 1usize;
    let mut json_output = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--left" => left = Some(value_after(args, &mut i, "--left")?),
            "--right" => right = Some(value_after(args, &mut i, "--right")?),
            "--ignore-stderr" => ignore_stderr = true,
            "--allow-failure" => allow_failure = true,
            "--repeat" => {
                repeat = parse_positive_usize(&value_after(args, &mut i, "--repeat")?, "--repeat")?;
            },
            "--json" => json_output = true,
            "-h" | "--help" => {
                println!("cargo scirust parity --left \"COMMAND\" --right \"COMMAND\" [--repeat N] [--ignore-stderr] [--allow-failure] [--json]");
                println!("By default both commands must succeed as well as produce identical normalized output.");
                return Ok(());
            },
            other => return Err(AppError::message(format!("unknown parity option: {other}"))),
        }
        i += 1;
    }

    let left = left.ok_or_else(|| AppError::message("parity requires --left \"COMMAND\""))?;
    let right = right.ok_or_else(|| AppError::message("parity requires --right \"COMMAND\""))?;
    let mut reports = Vec::with_capacity(repeat);
    let mut all_pass = true;

    for run in 1..=repeat {
        let left_out = shell_capture(&ws.root, &left, Some(run))?;
        let right_out = shell_capture(&ws.root, &right, Some(run))?;
        let left_stdout = normalize_newlines(&left_out.stdout);
        let right_stdout = normalize_newlines(&right_out.stdout);
        let left_stderr = normalize_newlines(&left_out.stderr);
        let right_stderr = normalize_newlines(&right_out.stderr);
        let exit_equal = left_out.status.code() == right_out.status.code();
        let stdout_equal = left_stdout == right_stdout;
        let stderr_equal = ignore_stderr || left_stderr == right_stderr;
        let success_ok = allow_failure || (left_out.status.success() && right_out.status.success());
        let pass = exit_equal && stdout_equal && stderr_equal && success_ok;
        all_pass &= pass;

        let stdout_diff = first_difference(&left_stdout, &right_stdout);
        let stderr_diff = if ignore_stderr {
            None
        } else {
            first_difference(&left_stderr, &right_stderr)
        };
        reports.push(json!({
            "run": run,
            "pass": pass,
            "left_exit": left_out.status.code(),
            "right_exit": right_out.status.code(),
            "exit_equal": exit_equal,
            "success_required": !allow_failure,
            "stdout_equal": stdout_equal,
            "stderr_equal": stderr_equal,
            "left_stdout_fingerprint": fingerprint(&left_out.stdout),
            "right_stdout_fingerprint": fingerprint(&right_out.stdout),
            "left_stderr_fingerprint": fingerprint(&left_out.stderr),
            "right_stderr_fingerprint": fingerprint(&right_out.stderr),
            "stdout_first_difference": stdout_diff,
            "stderr_first_difference": stderr_diff,
        }));

        if !json_output {
            println!("Parity run {run}/{repeat}: {}", verdict(pass));
            println!("  exit:   {} ({:?} / {:?})", verdict(exit_equal && success_ok), left_out.status.code(), right_out.status.code());
            println!("  stdout: {}  left={} right={}", verdict(stdout_equal), fingerprint(&left_out.stdout), fingerprint(&right_out.stdout));
            if let Some(offset) = stdout_diff {
                println!("          first difference: byte {offset}, line {}", line_at_offset(&left_stdout, offset));
            }
            if !ignore_stderr {
                println!("  stderr: {}  left={} right={}", verdict(stderr_equal), fingerprint(&left_out.stderr), fingerprint(&right_out.stderr));
                if let Some(offset) = stderr_diff {
                    println!("          first difference: byte {offset}, line {}", line_at_offset(&left_stderr, offset));
                }
            }
        }
    }

    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "left": left,
                "right": right,
                "repeat": repeat,
                "pass": all_pass,
                "runs": reports,
            }))
            .map_err(|error| AppError::message(error.to_string()))?
        );
    } else if all_pass {
        println!("PARITY PASS ({repeat} run(s))");
    }

    if all_pass {
        Ok(())
    } else {
        Err(AppError::message("PARITY FAIL"))
    }
}

pub fn determinism(ws: &Workspace, args: &[String]) -> AppResult<()> {
    let mut repeat = 3usize;
    let mut command_start = None;
    let mut ignore_stderr = false;
    let mut allow_failure = false;
    let mut json_output = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--repeat" => {
                repeat = parse_positive_usize(&value_after(args, &mut i, "--repeat")?, "--repeat")?;
            },
            "--ignore-stderr" => ignore_stderr = true,
            "--allow-failure" => allow_failure = true,
            "--json" => json_output = true,
            "--" => {
                command_start = Some(i + 1);
                break;
            },
            "-h" | "--help" => {
                println!("cargo scirust determinism [--repeat N] [--ignore-stderr] [--allow-failure] [--json] -- PROGRAM [ARGS...]");
                println!("By default every run must exit successfully and reproduce normalized stdout/stderr exactly.");
                return Ok(());
            },
            other => return Err(AppError::message(format!("unknown determinism option before --: {other}"))),
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
    let mut reports = Vec::with_capacity(repeat);
    let mut all_pass = true;
    for run in 1..=repeat {
        let output = direct_capture(&ws.root, command, Some(run))?;
        let success_ok = allow_failure || output.status.success();
        let mut pass = success_ok;
        let mut stdout_diff = None;
        let mut stderr_diff = None;
        if let Some(first) = &baseline {
            let first_stdout = normalize_newlines(&first.stdout);
            let output_stdout = normalize_newlines(&output.stdout);
            let first_stderr = normalize_newlines(&first.stderr);
            let output_stderr = normalize_newlines(&output.stderr);
            stdout_diff = first_difference(&first_stdout, &output_stdout);
            stderr_diff = if ignore_stderr {
                None
            } else {
                first_difference(&first_stderr, &output_stderr)
            };
            pass &= first.status.code() == output.status.code();
            pass &= stdout_diff.is_none();
            pass &= ignore_stderr || stderr_diff.is_none();
        }
        all_pass &= pass;

        reports.push(json!({
            "run": run,
            "pass": pass,
            "exit": output.status.code(),
            "stdout_fingerprint": fingerprint(&output.stdout),
            "stderr_fingerprint": fingerprint(&output.stderr),
            "stdout_first_difference_from_run1": stdout_diff,
            "stderr_first_difference_from_run1": stderr_diff,
        }));

        if !json_output {
            println!("  run {run}/{repeat}: {} exit={:?} stdout={} stderr={}", verdict(pass), output.status.code(), fingerprint(&output.stdout), fingerprint(&output.stderr));
            if let Some(offset) = stdout_diff {
                println!("    stdout first differs at normalized byte {offset}");
            }
            if let Some(offset) = stderr_diff {
                println!("    stderr first differs at normalized byte {offset}");
            }
        }
        if baseline.is_none() {
            baseline = Some(output);
        }
    }

    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "command": command,
                "repeat": repeat,
                "pass": all_pass,
                "runs": reports,
            }))
            .map_err(|error| AppError::message(error.to_string()))?
        );
    } else if all_pass {
        println!("DETERMINISM PASS ({repeat} exact runs)");
    }

    if all_pass {
        Ok(())
    } else {
        Err(AppError::message("DETERMINISM FAIL"))
    }
}

#[derive(Debug)]
struct CostFinding {
    kind: &'static str,
    path: PathBuf,
    line: usize,
    snippet: String,
}

pub fn cost(ws: &Workspace, args: &[String]) -> AppResult<()> {
    let mut path = None;
    let mut package = None;
    let mut json_output = false;
    let mut limit = 30usize;
    let mut no_static = false;
    let mut measured_runs = 0usize;
    let mut warmup_runs = 1usize;
    let mut inherit_io = false;
    let mut command_start = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--path" => path = Some(PathBuf::from(value_after(args, &mut i, "--path")?)),
            "--package" | "-p" => package = Some(value_after(args, &mut i, "--package")?),
            "--json" => json_output = true,
            "--limit" => {
                limit = parse_positive_usize(&value_after(args, &mut i, "--limit")?, "--limit")?;
            },
            "--no-static" => no_static = true,
            "--measure" => {
                measured_runs = parse_positive_usize(&value_after(args, &mut i, "--measure")?, "--measure")?;
            },
            "--warmup" => {
                warmup_runs = value_after(args, &mut i, "--warmup")?
                    .parse::<usize>()
                    .map_err(|_| AppError::message("--warmup expects a non-negative integer"))?;
            },
            "--inherit-io" => inherit_io = true,
            "--" => {
                command_start = Some(i + 1);
                break;
            },
            "-h" | "--help" => {
                println!("cargo scirust cost [--path PATH | -p PACKAGE] [--limit N] [--json] [--no-static] [--measure N] [--warmup N] [--inherit-io] [-- PROGRAM ARGS...]");
                println!("Without --measure this is a static source heuristic. With --measure it additionally reports real wall-clock samples for the supplied command.");
                return Ok(());
            },
            other => return Err(AppError::message(format!("unknown cost option: {other}"))),
        }
        i += 1;
    }

    let root = match (path, package) {
        (Some(_), Some(_)) => return Err(AppError::message("use only one of --path or --package")),
        (Some(path), None) => {
            if path.is_absolute() { path } else { ws.root.join(path) }
        },
        (None, Some(name)) => ws
            .package(&name)
            .ok_or_else(|| AppError::message(format!("unknown workspace package: {name}")))?
            .dir
            .clone(),
        (None, None) => ws.root.clone(),
    };

    let findings = if no_static { Vec::new() } else { scan_cost(&root)? };
    let mut counts = BTreeMap::<&'static str, usize>::new();
    for finding in &findings {
        *counts.entry(finding.kind).or_default() += 1;
    }

    let mut timings = Vec::new();
    let command = command_start.map(|start| args[start..].to_vec());
    if measured_runs > 0 {
        let command = command
            .as_ref()
            .ok_or_else(|| AppError::message("--measure requires `-- PROGRAM [ARGS...]`"))?;
        if command.is_empty() {
            return Err(AppError::message("measured command is empty"));
        }
        for _ in 0..warmup_runs {
            run_timed_command(&ws.root, command, inherit_io)?;
        }
        for run in 1..=measured_runs {
            let elapsed = run_timed_command(&ws.root, command, inherit_io)?;
            if !json_output {
                println!("  measured run {run}/{measured_runs}: {:.3} ms", elapsed as f64 / 1_000_000.0);
            }
            timings.push(elapsed);
        }
    } else if command.is_some() {
        return Err(AppError::message("a command after `--` requires --measure N"));
    }

    if json_output {
        let details = findings
            .iter()
            .take(limit)
            .map(|finding| json!({
                "kind": finding.kind,
                "path": finding.path,
                "line": finding.line,
                "snippet": finding.snippet,
            }))
            .collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "root": root,
                "static_heuristic": !no_static,
                "total_indicators": findings.len(),
                "counts": counts,
                "findings": details,
                "measurement": timing_json(&timings, command.as_deref()),
            }))
            .map_err(|error| AppError::message(error.to_string()))?
        );
        return Ok(());
    }

    if !no_static {
        println!("Static cost indicators under {}", root.display());
        println!("  NOTE: indicators are source heuristics, not measured runtime costs.");
        for (kind, count) in &counts {
            println!("  {kind:<24} {count:>6}");
        }
        println!("  total                    {:>6}", findings.len());
        for finding in findings.iter().take(limit) {
            println!("  {}:{} [{}] {}", finding.path.display(), finding.line, finding.kind, finding.snippet.trim());
        }
        if findings.len() > limit {
            println!("  ... {} more (raise --limit to display)", findings.len() - limit);
        }
    }
    if !timings.is_empty() {
        print_timing_summary("Measured command wall time", &timings);
    }
    Ok(())
}

pub fn calibrate(ws: &Workspace, args: &[String]) -> AppResult<()> {
    let mut pieces = None;
    let mut lengths = None;
    let mut tokenizer = None;
    let mut inputs = Vec::<PathBuf>::new();
    let mut output = None;
    let mut probe_lengths = None;
    let mut cases_per_length = None;
    let mut warmup_runs = None;
    let mut measured_runs = None;
    let mut extension = None;
    let mut device = None;
    let mut recursive = false;
    let mut debug = false;
    let mut dry_run = false;
    let mut json_output = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--pieces" => pieces = Some(PathBuf::from(value_after(args, &mut i, "--pieces")?)),
            "--lengths" => lengths = Some(PathBuf::from(value_after(args, &mut i, "--lengths")?)),
            "--tokenizer" => tokenizer = Some(PathBuf::from(value_after(args, &mut i, "--tokenizer")?)),
            "--input" => inputs.push(PathBuf::from(value_after(args, &mut i, "--input")?)),
            "--output" => output = Some(PathBuf::from(value_after(args, &mut i, "--output")?)),
            "--probe-lengths" => probe_lengths = Some(value_after(args, &mut i, "--probe-lengths")?),
            "--cases-per-length" => cases_per_length = Some(value_after(args, &mut i, "--cases-per-length")?),
            "--warmup-runs" => warmup_runs = Some(value_after(args, &mut i, "--warmup-runs")?),
            "--measured-runs" => measured_runs = Some(value_after(args, &mut i, "--measured-runs")?),
            "--extension" => extension = Some(value_after(args, &mut i, "--extension")?),
            "--device" => device = Some(value_after(args, &mut i, "--device")?),
            "--recursive" => recursive = true,
            "--debug" => debug = true,
            "--dry-run" => dry_run = true,
            "--json" => json_output = true,
            "-h" | "--help" => {
                println!("cargo scirust calibrate --tokenizer FILE --input PATH [--input PATH...] --output PROFILE.json [--recursive] [--probe-lengths CSV] [--cases-per-length N] [--warmup-runs N] [--measured-runs N] [--extension CSV] [--device NAME] [--debug] [--dry-run]");
                println!("cargo scirust calibrate (--pieces FILE | --lengths FILE) [--json]");
                println!("The tokenizer mode runs SciAgent's semantics-gated ElasticTokenizer autotuner. The pieces/lengths mode is distribution-only and does not select kernels.");
                return Ok(());
            },
            other => return Err(AppError::message(format!("unknown calibrate option: {other}"))),
        }
        i += 1;
    }

    if tokenizer.is_some() || !inputs.is_empty() || output.is_some() {
        if pieces.is_some() || lengths.is_some() || json_output {
            return Err(AppError::message("tokenizer autotune mode cannot be combined with --pieces, --lengths, or --json"));
        }
        let tokenizer = tokenizer.ok_or_else(|| AppError::message("autotune mode requires --tokenizer FILE"))?;
        let output = output.ok_or_else(|| AppError::message("autotune mode requires --output PROFILE.json"))?;
        if inputs.is_empty() {
            return Err(AppError::message("autotune mode requires at least one --input PATH"));
        }

        let mut command = vec![
            "cargo".into(),
            "run".into(),
            "--locked".into(),
        ];
        if !debug {
            command.push("--release".into());
        }
        command.extend([
            "-p".into(),
            "scirust-sciagent".into(),
            "--bin".into(),
            "tokenizer-autotune".into(),
            "--".into(),
            "--tokenizer".into(),
            path_arg(&tokenizer),
        ]);
        for input in &inputs {
            command.push("--input".into());
            command.push(path_arg(input));
        }
        command.push("--output".into());
        command.push(path_arg(&output));
        push_optional_arg(&mut command, "--probe-lengths", probe_lengths);
        push_optional_arg(&mut command, "--cases-per-length", cases_per_length);
        push_optional_arg(&mut command, "--warmup-runs", warmup_runs);
        push_optional_arg(&mut command, "--measured-runs", measured_runs);
        push_optional_arg(&mut command, "--extension", extension);
        push_optional_arg(&mut command, "--device", device);
        if recursive {
            command.push("--recursive".into());
        }
        println!("ElasticTokenizer full autotune ({})", if debug { "debug" } else { "release" });
        return run_command(&ws.root, command, dry_run);
    }

    let mut values = match (pieces, lengths) {
        (Some(_), Some(_)) | (None, None) => {
            return Err(AppError::message("distribution-only calibration requires exactly one of --pieces or --lengths"));
        },
        (Some(path), None) => read_piece_lengths(&path)?,
        (None, Some(path)) => read_numeric_lengths(&path)?,
    };
    if values.is_empty() {
        return Err(AppError::message("calibration input contains no positive piece lengths"));
    }
    values.sort_unstable();
    let cuts = strict_distribution_cuts(&values)?;
    let sum: u128 = values.iter().map(|&value| value as u128).sum();
    let mean = sum as f64 / values.len() as f64;

    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "count": values.len(),
                "distinct_lengths": values.iter().copied().collect::<BTreeSet<_>>().len(),
                "min_bytes": values[0],
                "max_bytes": values[values.len() - 1],
                "mean_bytes": mean,
                "boundaries": {
                    "S_max": cuts[0], "M_max": cuts[1], "L_max": cuts[2],
                    "XL_max": cuts[3], "XXL_max": cuts[4], "XXXL": format!(">{}", cuts[4])
                },
                "method": "distinct-quantile-midpoint-v2",
                "kernel_selection": false,
            }))
            .map_err(|error| AppError::message(error.to_string()))?
        );
    } else {
        println!("ElasticTokenizer distribution-only calibration (distinct-quantile-midpoint-v2)");
        println!("  observations: {}", values.len());
        println!("  bytes: min={} mean={mean:.3} max={}", values[0], values[values.len() - 1]);
        println!("  S     <= {} bytes", cuts[0]);
        println!("  M     <= {} bytes", cuts[1]);
        println!("  L     <= {} bytes", cuts[2]);
        println!("  XL    <= {} bytes", cuts[3]);
        println!("  XXL   <= {} bytes", cuts[4]);
        println!("  XXXL   > {} bytes", cuts[4]);
        println!("  NOTE: this mode does not select BPE kernels; use --tokenizer/--input/--output for full autotune.");
    }
    Ok(())
}

fn print_impact(impact: &Impact, json_output: bool, direct_only: bool) -> AppResult<()> {
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
                "selected": if direct_only { &impact.direct } else { &impact.affected },
            }))
            .map_err(|error| AppError::message(error.to_string()))?
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
    if direct_only {
        println!("Directly affected ({}):", impact.direct.len());
        for package in &impact.direct {
            println!("  {package}");
        }
    } else {
        println!("Directly affected ({}):", impact.direct.len());
        for package in &impact.direct {
            println!("  {package}");
        }
        println!("Transitively affected ({}):", impact.affected.len());
        for package in &impact.affected {
            println!("  {package}");
        }
    }
    Ok(())
}

fn push_packages(command: &mut Vec<String>, packages: &[String]) {
    for package in packages {
        command.push("-p".to_string());
        command.push(package.clone());
    }
}

fn push_optional_arg(command: &mut Vec<String>, flag: &str, value: Option<String>) {
    if let Some(value) = value {
        command.push(flag.to_string());
        command.push(value);
    }
}

fn path_arg(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn value_after(args: &[String], i: &mut usize, flag: &str) -> AppResult<String> {
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| AppError::message(format!("{flag} requires a value")))
}

fn parse_positive_usize(value: &str, flag: &str) -> AppResult<usize> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| AppError::message(format!("{flag} expects a positive integer")))?;
    if parsed == 0 {
        return Err(AppError::message(format!("{flag} expects a positive integer")));
    }
    Ok(parsed)
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

fn capture_command(root: &Path, command: &[String]) -> AppResult<Output> {
    let (program, args) = command
        .split_first()
        .ok_or_else(|| AppError::message("internal error: empty command"))?;
    Command::new(program)
        .current_dir(root)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .map_err(AppError::io)
}

fn run_timed_command(root: &Path, command: &[String], inherit_io: bool) -> AppResult<u64> {
    let (program, args) = command
        .split_first()
        .ok_or_else(|| AppError::message("empty measured command"))?;
    let mut child = Command::new(program);
    child.current_dir(root).args(args).stdin(Stdio::null());
    if !inherit_io {
        child.stdout(Stdio::null()).stderr(Stdio::null());
    }
    let start = Instant::now();
    let status = child.status().map_err(AppError::io)?;
    let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    if !status.success() {
        return Err(AppError::message(format!(
            "measured command failed with status {:?}: {}",
            status.code(),
            printable_command(command)
        )));
    }
    Ok(elapsed)
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
        child.env("SCIRUST_PARITY_RUN", run.to_string());
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
    let normalized = normalize_newlines(bytes);
    let mut hash = 0xcbf29ce484222325u64;
    for byte in normalized {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

fn verdict(ok: bool) -> &'static str {
    if ok { "PASS" } else { "FAIL" }
}

fn first_difference(left: &[u8], right: &[u8]) -> Option<usize> {
    let common = left.len().min(right.len());
    for index in 0..common {
        if left[index] != right[index] {
            return Some(index);
        }
    }
    (left.len() != right.len()).then_some(common)
}

fn line_at_offset(bytes: &[u8], offset: usize) -> usize {
    1 + bytes[..offset.min(bytes.len())]
        .iter()
        .filter(|&&byte| byte == b'\n')
        .count()
}

fn feature_label(features: &[String]) -> String {
    if features.is_empty() {
        "<no features>".to_string()
    } else {
        features.join(",")
    }
}

fn diagnostic_tail(bytes: &[u8], lines: usize) -> String {
    let text = String::from_utf8_lossy(bytes);
    let all = text.lines().collect::<Vec<_>>();
    let start = all.len().saturating_sub(lines);
    all[start..].join("\n")
}

fn indent(text: &str, prefix: &str) -> String {
    text.lines()
        .map(|line| format!("{prefix}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
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
            if matches!(name.to_str(), Some("target" | ".git" | "data" | "node_modules")) {
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
        let value = line.parse::<usize>().map_err(|_| {
            AppError::message(format!("invalid positive integer at {}:{}", path.display(), index + 1))
        })?;
        if value > 0 {
            out.push(value);
        }
    }
    Ok(out)
}

fn strict_distribution_cuts(sorted: &[usize]) -> AppResult<[usize; 5]> {
    let distinct = sorted.iter().copied().collect::<BTreeSet<_>>().into_iter().collect::<Vec<_>>();
    if distinct.len() < 6 {
        return Err(AppError::message(format!(
            "six strict ElasticTokenizer classes require at least six distinct positive lengths; found {}",
            distinct.len()
        )));
    }
    let n = distinct.len();
    let mut cuts = [0usize; 5];
    for k in 1..=5 {
        let mut split = (k * n).div_ceil(6);
        split = split.max(k).min(n - (6 - k));
        let left = distinct[split - 1];
        let right = distinct[split];
        cuts[k - 1] = left + (right - left) / 2;
    }
    if !cuts.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err(AppError::message("failed to derive strictly increasing ElasticTokenizer thresholds"));
    }
    Ok(cuts)
}

fn timing_json(timings: &[u64], command: Option<&[String]>) -> serde_json::Value {
    if timings.is_empty() {
        return serde_json::Value::Null;
    }
    let mut sorted = timings.to_vec();
    sorted.sort_unstable();
    let sum: u128 = sorted.iter().map(|&value| u128::from(value)).sum();
    json!({
        "command": command,
        "runs": sorted.len(),
        "min_nanos": sorted[0],
        "median_nanos": integer_median(&sorted),
        "max_nanos": sorted[sorted.len() - 1],
        "mean_nanos": sum as f64 / sorted.len() as f64,
    })
}

fn print_timing_summary(label: &str, timings: &[u64]) {
    let mut sorted = timings.to_vec();
    sorted.sort_unstable();
    let sum: u128 = sorted.iter().map(|&value| u128::from(value)).sum();
    let mean = sum as f64 / sorted.len() as f64;
    println!("{label} ({} runs)", sorted.len());
    println!("  min:    {:.3} ms", sorted[0] as f64 / 1_000_000.0);
    println!("  median: {:.3} ms", integer_median(&sorted) as f64 / 1_000_000.0);
    println!("  mean:   {:.3} ms", mean / 1_000_000.0);
    println!("  max:    {:.3} ms", sorted[sorted.len() - 1] as f64 / 1_000_000.0);
}

fn integer_median(sorted: &[u64]) -> u64 {
    debug_assert!(!sorted.is_empty());
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[middle]
    } else {
        let low = sorted[middle - 1];
        let high = sorted[middle];
        low + (high - low) / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_is_stable_and_normalizes_crlf() {
        assert_eq!(fingerprint(b"hello"), "fnv1a64:a430d84680aabd0b");
        assert_eq!(fingerprint(b"a\r\nb\r\n"), fingerprint(b"a\nb\n"));
    }

    #[test]
    fn first_difference_handles_content_and_length_changes() {
        assert_eq!(first_difference(b"abc", b"axc"), Some(1));
        assert_eq!(first_difference(b"abc", b"abcx"), Some(3));
        assert_eq!(first_difference(b"abc", b"abc"), None);
    }

    #[test]
    fn strict_distribution_cuts_are_monotonic() {
        let values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let cuts = strict_distribution_cuts(&values).unwrap();
        assert!(cuts.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(cuts, [2, 4, 6, 8, 10]);
    }

    #[test]
    fn strict_distribution_cuts_reject_low_diversity() {
        assert!(strict_distribution_cuts(&[1, 1, 2, 2, 3, 3]).is_err());
    }

    #[test]
    fn median_does_not_overflow() {
        assert_eq!(integer_median(&[10, 20]), 15);
        assert_eq!(integer_median(&[u64::MAX - 1, u64::MAX]), u64::MAX - 1);
    }
}

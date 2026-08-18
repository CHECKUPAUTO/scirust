from pathlib import Path

path = Path("scirust-sciagent/src/optimization_agent/mod.rs")
text = path.read_text()


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one occurrence, found {count}: {old[:80]!r}")
    text = text.replace(old, new, 1)


replace_once("use std::fs;", "use std::fs::{self, File};")
replace_once("use std::process::Command;", "use std::process::{Command, Stdio};")

replace_once(
    """pub enum OptimizationDecision {
    Promote,
    RewriteForCorrectness,
    RewriteForPerformance,
    BudgetExhausted,
}
""",
    """pub enum OptimizationDecision {
    Promote,
    RetryGeneration,
    RewriteForCompilation,
    RewriteForCorrectness,
    RewriteForPerformance,
    BudgetExhausted,
}
""",
)

replace_once(
    """pub struct IterationRecord {
    pub iteration: usize,
    pub verification: VerificationMeasurement,
    pub timing: TimingMeasurement,
    pub speedup: f64,
    pub correctness_gate: bool,
    pub performance_gate: bool,
    pub decision: OptimizationDecision,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationReport {
""",
    """pub struct IterationRecord {
    pub iteration: usize,
    pub verification: VerificationMeasurement,
    pub timing: TimingMeasurement,
    pub speedup: f64,
    pub correctness_gate: bool,
    pub performance_gate: bool,
    pub decision: OptimizationDecision,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationFailure {
    pub iteration: usize,
    pub stage: String,
    pub message: String,
    pub log_path: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationReport {
""",
)

replace_once(
    """    pub target_speedup: f64,
    pub iterations: Vec<IterationRecord>,
    pub final_decision: OptimizationDecision,
""",
    """    pub target_speedup: f64,
    pub iterations: Vec<IterationRecord>,
    pub failures: Vec<OptimizationFailure>,
    pub final_decision: OptimizationDecision,
""",
)

replace_once(
    """    baseline_median_ns: f64,
    previous_iterations: Vec<IterationRecord>,
}
""",
    """    baseline_median_ns: f64,
    previous_iterations: Vec<IterationRecord>,
    failures: Vec<OptimizationFailure>,
}
""",
)

replace_once(
    """            target_speedup: task.budget.min_speedup,
            iterations: Vec::new(),
            final_decision: OptimizationDecision::BudgetExhausted,
""",
    """            target_speedup: task.budget.min_speedup,
            iterations: Vec::new(),
            failures: Vec::new(),
            final_decision: OptimizationDecision::BudgetExhausted,
""",
)

replace_once(
    """                    baseline_median_ns: baseline.median_ns,
                    previous_iterations: report.iterations.clone(),
                },
""",
    """                    baseline_median_ns: baseline.median_ns,
                    previous_iterations: report.iterations.clone(),
                    failures: report.failures.clone(),
                },
""",
)

start_marker = """            self.execute_stage(
                if iteration == 1
"""
end_marker = """            let decision = if correctness_gate && performance_gate
"""
start = text.index(start_marker)
end = text.index(end_marker, start)
replacement = """            let (generation_stage, generation_command) = if iteration == 1 {
                ("generate", &task.commands.generate)
            } else {
                (
                    "rewrite",
                    task.commands
                        .rewrite
                        .as_ref()
                        .unwrap_or(&task.commands.generate),
                )
            };
            if !self.execute_or_record(
                generation_stage,
                generation_command,
                task,
                iteration,
                &workspace,
                &run_dir,
                OptimizationDecision::RetryGeneration,
                &mut report,
            )? {
                continue;
            }
            if !self.execute_or_record(
                "compile",
                &task.commands.compile,
                task,
                iteration,
                &workspace,
                &run_dir,
                OptimizationDecision::RewriteForCompilation,
                &mut report,
            )? {
                continue;
            }
            if !self.execute_or_record(
                "verify",
                &task.commands.verify,
                task,
                iteration,
                &workspace,
                &run_dir,
                OptimizationDecision::RewriteForCorrectness,
                &mut report,
            )? {
                continue;
            }
            let verification: VerificationMeasurement = read_json(&verify_path)?;
            verification.validate()?;
            let correctness_gate = verification.passed
                && verification
                    .max_abs_error
                    .is_none_or(|value| value <= task.budget.max_abs_error)
                && verification
                    .max_rel_error
                    .is_none_or(|value| value <= task.budget.max_rel_error);
            if !correctness_gate {
                report.failures.push(OptimizationFailure {
                    iteration,
                    stage: "verify".to_string(),
                    message: format!(
                        "verification rejected candidate: passed={}, max_abs_error={:?}, max_rel_error={:?}",
                        verification.passed,
                        verification.max_abs_error,
                        verification.max_rel_error,
                    ),
                    log_path: stage_log_path(&run_dir, iteration, "verify")
                        .to_string_lossy()
                        .into_owned(),
                });
                report.final_decision = OptimizationDecision::RewriteForCorrectness;
                write_json(&run_dir.join("report.json"), &report)?;
                continue;
            }
            if !self.execute_or_record(
                "benchmark",
                &task.commands.benchmark,
                task,
                iteration,
                &workspace,
                &run_dir,
                OptimizationDecision::RewriteForPerformance,
                &mut report,
            )? {
                continue;
            }
            let timing: TimingMeasurement = read_json(&candidate_path)?;
            timing.validate("candidate")?;

            let speedup = baseline.median_ns / timing.median_ns;
            let performance_gate = speedup >= task.budget.min_speedup;
"""
text = text[:start] + replacement + text[end:]

replace_once(
    """                if let Some(profile) = &task.commands.profile
                {
                    self.execute_stage("profile", profile, task, iteration, &workspace, &run_dir)?;
                }
""",
    """                if let Some(profile) = &task.commands.profile {
                    let _ = self.execute_or_record(
                        "profile",
                        profile,
                        task,
                        iteration,
                        &workspace,
                        &run_dir,
                        OptimizationDecision::RewriteForPerformance,
                        &mut report,
                    )?;
                }
""",
)

method_marker = """    fn execute_stage(
"""
method_pos = text.index(method_marker)
helper = """    fn execute_or_record(
        &self,
        stage: &str,
        spec: &CommandSpec,
        task: &OptimizationTask,
        iteration: usize,
        workspace: &Path,
        run_dir: &Path,
        decision: OptimizationDecision,
        report: &mut OptimizationReport,
    ) -> Result<bool, OptimizationError> {
        match self.execute_stage(stage, spec, task, iteration, workspace, run_dir) {
            Ok(()) => Ok(true),
            Err(error) => {
                report.failures.push(OptimizationFailure {
                    iteration,
                    stage: stage.to_string(),
                    message: error.to_string(),
                    log_path: stage_log_path(run_dir, iteration, stage)
                        .to_string_lossy()
                        .into_owned(),
                });
                report.final_decision = decision;
                write_json(&run_dir.join("report.json"), report)?;
                Ok(false)
            }
        }
    }

"""
text = text[:method_pos] + helper + text[method_pos:]

spawn_marker = """        let mut child = command
            .spawn()
"""
replace_once(
    spawn_marker,
    """        let log_path = stage_log_path(run_dir, iteration, stage);
        let stdout = File::create(&log_path).map_err(|source| OptimizationError::Io {
            path: log_path.clone(),
            source,
        })?;
        let stderr = stdout.try_clone().map_err(|source| OptimizationError::Io {
            path: log_path.clone(),
            source,
        })?;
        command.stdout(Stdio::from(stdout)).stderr(Stdio::from(stderr));

        let mut child = command
            .spawn()
""",
)

canonical_marker = """fn canonicalize_existing(path: &Path) -> Result<PathBuf, OptimizationError> {
"""
canonical_pos = text.index(canonical_marker)
stage_path_fn = """fn stage_log_path(run_dir: &Path, iteration: usize, stage: &str) -> PathBuf {
    run_dir.join(format!("{iteration:02}-{stage}.log"))
}

"""
text = text[:canonical_pos] + stage_path_fn + text[canonical_pos:]

path.write_text(text)

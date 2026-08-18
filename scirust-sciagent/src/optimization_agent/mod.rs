use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

pub const OPTIMIZATION_SKILL: &str = include_str!("../../SKILL_OPTIMIZATION.md");
const REPORT_SCHEMA_VERSION: u32 = 1;
const CONTEXT_SCHEMA_VERSION: u32 = 1;
const SECRET_ENV_VARS: &[&str] = &[
    "SCIRUST_DISCOVERY_KEY",
    "SCIRUST_EXCHANGE_SECRET",
    "SCIRUST_WALLET_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OptimizationBackend {
    Cpu,
    Simd,
    Sve,
    Wgpu,
    Cuda,
}

impl fmt::Display for OptimizationBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self
        {
            Self::Cpu => "cpu",
            Self::Simd => "simd",
            Self::Sve => "sve",
            Self::Wgpu => "wgpu",
            Self::Cuda => "cuda",
        };
        f.write_str(text)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandSpec {
    pub program: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub timeout_secs: Option<u64>,
}

impl CommandSpec {
    fn validate(&self, name: &str) -> Result<(), OptimizationError> {
        if self.program.trim().is_empty()
        {
            return Err(OptimizationError::InvalidTask(format!(
                "{name}.program must not be empty"
            )));
        }
        if self.timeout_secs == Some(0)
        {
            return Err(OptimizationError::InvalidTask(format!(
                "{name}.timeout_secs must be greater than zero"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationBudget {
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
    #[serde(default = "default_min_speedup")]
    pub min_speedup: f64,
    #[serde(default = "default_max_abs_error")]
    pub max_abs_error: f64,
    #[serde(default = "default_max_rel_error")]
    pub max_rel_error: f64,
    #[serde(default = "default_timeout_secs")]
    pub command_timeout_secs: u64,
}

impl Default for OptimizationBudget {
    fn default() -> Self {
        Self {
            max_iterations: default_max_iterations(),
            min_speedup: default_min_speedup(),
            max_abs_error: default_max_abs_error(),
            max_rel_error: default_max_rel_error(),
            command_timeout_secs: default_timeout_secs(),
        }
    }
}

fn default_max_iterations() -> usize {
    8
}
fn default_min_speedup() -> f64 {
    1.05
}
fn default_max_abs_error() -> f64 {
    1.0e-6
}
fn default_max_rel_error() -> f64 {
    1.0e-6
}
fn default_timeout_secs() -> u64 {
    1800
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationCommands {
    pub baseline: CommandSpec,
    pub generate: CommandSpec,
    pub compile: CommandSpec,
    pub verify: CommandSpec,
    pub benchmark: CommandSpec,
    #[serde(default)]
    pub profile: Option<CommandSpec>,
    #[serde(default)]
    pub rewrite: Option<CommandSpec>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationTask {
    pub id: String,
    pub goal: String,
    pub crate_name: String,
    pub backend: OptimizationBackend,
    #[serde(default)]
    pub allowed_paths: Vec<String>,
    #[serde(default)]
    pub budget: OptimizationBudget,
    pub commands: OptimizationCommands,
}

impl OptimizationTask {
    pub fn validate(&self) -> Result<(), OptimizationError> {
        if self.id.is_empty()
            || !self
                .id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        {
            return Err(OptimizationError::InvalidTask(
                "id must contain only ASCII letters, digits, '-' or '_'".to_string(),
            ));
        }
        if self.goal.trim().is_empty()
        {
            return Err(OptimizationError::InvalidTask(
                "goal must not be empty".to_string(),
            ));
        }
        if self.crate_name.trim().is_empty()
        {
            return Err(OptimizationError::InvalidTask(
                "crate_name must not be empty".to_string(),
            ));
        }
        if self.allowed_paths.is_empty()
        {
            return Err(OptimizationError::InvalidTask(
                "allowed_paths must name at least one path".to_string(),
            ));
        }
        if self
            .allowed_paths
            .iter()
            .any(|path| path.is_empty() || Path::new(path).is_absolute() || path.contains(".."))
        {
            return Err(OptimizationError::InvalidTask(
                "allowed_paths must be non-empty workspace-relative paths without '..'".to_string(),
            ));
        }
        if self.budget.max_iterations == 0
        {
            return Err(OptimizationError::InvalidTask(
                "budget.max_iterations must be greater than zero".to_string(),
            ));
        }
        if !self.budget.min_speedup.is_finite() || self.budget.min_speedup < 1.0
        {
            return Err(OptimizationError::InvalidTask(
                "budget.min_speedup must be finite and >= 1.0".to_string(),
            ));
        }
        for (name, value) in [
            ("max_abs_error", self.budget.max_abs_error),
            ("max_rel_error", self.budget.max_rel_error),
        ]
        {
            if !value.is_finite() || value < 0.0
            {
                return Err(OptimizationError::InvalidTask(format!(
                    "budget.{name} must be finite and >= 0"
                )));
            }
        }
        if self.budget.command_timeout_secs == 0
        {
            return Err(OptimizationError::InvalidTask(
                "budget.command_timeout_secs must be greater than zero".to_string(),
            ));
        }

        self.commands.baseline.validate("commands.baseline")?;
        self.commands.generate.validate("commands.generate")?;
        self.commands.compile.validate("commands.compile")?;
        self.commands.verify.validate("commands.verify")?;
        self.commands.benchmark.validate("commands.benchmark")?;
        if let Some(command) = &self.commands.profile
        {
            command.validate("commands.profile")?;
        }
        if let Some(command) = &self.commands.rewrite
        {
            command.validate("commands.rewrite")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimingMeasurement {
    pub median_ns: f64,
}

impl TimingMeasurement {
    fn validate(&self, label: &str) -> Result<(), OptimizationError> {
        if !self.median_ns.is_finite() || self.median_ns <= 0.0
        {
            return Err(OptimizationError::InvalidMetrics(format!(
                "{label}.median_ns must be finite and > 0"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationMeasurement {
    pub passed: bool,
    #[serde(default)]
    pub max_abs_error: Option<f64>,
    #[serde(default)]
    pub max_rel_error: Option<f64>,
    #[serde(default)]
    pub notes: Option<String>,
}

impl VerificationMeasurement {
    fn validate(&self) -> Result<(), OptimizationError> {
        for (name, value) in [
            ("max_abs_error", self.max_abs_error),
            ("max_rel_error", self.max_rel_error),
        ]
        {
            if let Some(value) = value
            {
                if !value.is_finite() || value < 0.0
                {
                    return Err(OptimizationError::InvalidMetrics(format!(
                        "verification.{name} must be finite and >= 0"
                    )));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OptimizationDecision {
    Promote,
    RetryGeneration,
    RewriteForCompilation,
    RewriteForCorrectness,
    RewriteForPerformance,
    BudgetExhausted,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IterationRecord {
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
    pub schema_version: u32,
    pub task_id: String,
    pub backend: OptimizationBackend,
    pub baseline: TimingMeasurement,
    pub target_speedup: f64,
    pub iterations: Vec<IterationRecord>,
    pub failures: Vec<OptimizationFailure>,
    pub final_decision: OptimizationDecision,
    pub best_verified_speedup: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct OptimizationContext {
    schema_version: u32,
    task_id: String,
    crate_name: String,
    backend: OptimizationBackend,
    goal: String,
    allowed_paths: Vec<String>,
    iteration: usize,
    target_speedup: f64,
    max_abs_error: f64,
    max_rel_error: f64,
    baseline_median_ns: f64,
    previous_iterations: Vec<IterationRecord>,
    failures: Vec<OptimizationFailure>,
}

#[derive(Debug)]
pub enum OptimizationError {
    InvalidTask(String),
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    CommandSpawn {
        stage: String,
        source: std::io::Error,
    },
    CommandPoll {
        stage: String,
        source: std::io::Error,
    },
    CommandFailed {
        stage: String,
        code: Option<i32>,
    },
    CommandTimedOut {
        stage: String,
        timeout_secs: u64,
    },
    InvalidMetrics(String),
}

impl fmt::Display for OptimizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidTask(message) => write!(f, "invalid optimization task: {message}"),
            Self::Io { path, source } => write!(f, "I/O error at {}: {source}", path.display()),
            Self::Json { path, source } =>
            {
                write!(f, "invalid JSON at {}: {source}", path.display())
            },
            Self::CommandSpawn { stage, source } =>
            {
                write!(f, "cannot start stage `{stage}`: {source}")
            },
            Self::CommandPoll { stage, source } =>
            {
                write!(f, "cannot poll stage `{stage}`: {source}")
            },
            Self::CommandFailed { stage, code } =>
            {
                write!(f, "stage `{stage}` failed with exit code {code:?}")
            },
            Self::CommandTimedOut {
                stage,
                timeout_secs,
            } => write!(f, "stage `{stage}` timed out after {timeout_secs}s"),
            Self::InvalidMetrics(message) => write!(f, "invalid optimization metrics: {message}"),
        }
    }
}

impl std::error::Error for OptimizationError {}

#[derive(Debug, Clone)]
pub struct OptimizationRunner {
    workspace: PathBuf,
    run_root: PathBuf,
}

impl OptimizationRunner {
    pub fn new(workspace: impl Into<PathBuf>, run_root: impl Into<PathBuf>) -> Self {
        Self {
            workspace: workspace.into(),
            run_root: run_root.into(),
        }
    }

    pub fn run(&self, task: &OptimizationTask) -> Result<OptimizationReport, OptimizationError> {
        task.validate()?;
        let workspace = canonicalize_existing(&self.workspace)?;
        let run_root = if self.run_root.is_absolute()
        {
            self.run_root.clone()
        }
        else
        {
            workspace.join(&self.run_root)
        };
        fs::create_dir_all(&run_root).map_err(|source| OptimizationError::Io {
            path: run_root.clone(),
            source,
        })?;
        let run_dir = run_root.join(&task.id);
        fs::create_dir_all(&run_dir).map_err(|source| OptimizationError::Io {
            path: run_dir.clone(),
            source,
        })?;

        let baseline_path = run_dir.join("baseline.json");
        remove_if_exists(&baseline_path)?;
        self.execute_stage(
            "baseline",
            &task.commands.baseline,
            task,
            0,
            &workspace,
            &run_dir,
        )?;
        let baseline: TimingMeasurement = read_json(&baseline_path)?;
        baseline.validate("baseline")?;

        let mut report = OptimizationReport {
            schema_version: REPORT_SCHEMA_VERSION,
            task_id: task.id.clone(),
            backend: task.backend,
            baseline: baseline.clone(),
            target_speedup: task.budget.min_speedup,
            iterations: Vec::new(),
            failures: Vec::new(),
            final_decision: OptimizationDecision::BudgetExhausted,
            best_verified_speedup: None,
        };

        for iteration in 1..=task.budget.max_iterations
        {
            let context_path = run_dir.join("context.json");
            let verify_path = run_dir.join("verify.json");
            let candidate_path = run_dir.join("candidate.json");
            write_json(
                &context_path,
                &OptimizationContext {
                    schema_version: CONTEXT_SCHEMA_VERSION,
                    task_id: task.id.clone(),
                    crate_name: task.crate_name.clone(),
                    backend: task.backend,
                    goal: task.goal.clone(),
                    allowed_paths: task.allowed_paths.clone(),
                    iteration,
                    target_speedup: task.budget.min_speedup,
                    max_abs_error: task.budget.max_abs_error,
                    max_rel_error: task.budget.max_rel_error,
                    baseline_median_ns: baseline.median_ns,
                    previous_iterations: report.iterations.clone(),
                    failures: report.failures.clone(),
                },
            )?;
            remove_if_exists(&verify_path)?;
            remove_if_exists(&candidate_path)?;

            let (generation_stage, generation_command) = if iteration == 1
            {
                ("generate", &task.commands.generate)
            }
            else
            {
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
            )?
            {
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
            )?
            {
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
            )?
            {
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
            if !correctness_gate
            {
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
            )?
            {
                continue;
            }
            let timing: TimingMeasurement = read_json(&candidate_path)?;
            timing.validate("candidate")?;

            let speedup = baseline.median_ns / timing.median_ns;
            let performance_gate = speedup >= task.budget.min_speedup;
            let decision = if correctness_gate && performance_gate
            {
                OptimizationDecision::Promote
            }
            else if !correctness_gate
            {
                OptimizationDecision::RewriteForCorrectness
            }
            else
            {
                OptimizationDecision::RewriteForPerformance
            };

            if correctness_gate
            {
                report.best_verified_speedup = Some(
                    report
                        .best_verified_speedup
                        .map_or(speedup, |best| best.max(speedup)),
                );
            }

            report.iterations.push(IterationRecord {
                iteration,
                verification,
                timing,
                speedup,
                correctness_gate,
                performance_gate,
                decision,
            });
            report.final_decision = decision;
            write_json(&run_dir.join("report.json"), &report)?;

            if decision == OptimizationDecision::Promote
            {
                return Ok(report);
            }
            if decision == OptimizationDecision::RewriteForPerformance
            {
                if let Some(profile) = &task.commands.profile
                {
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
            }
        }

        report.final_decision = OptimizationDecision::BudgetExhausted;
        write_json(&run_dir.join("report.json"), &report)?;
        Ok(report)
    }

    fn execute_or_record(
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
        match self.execute_stage(stage, spec, task, iteration, workspace, run_dir)
        {
            Ok(()) => Ok(true),
            Err(error) =>
            {
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
            },
        }
    }

    fn execute_stage(
        &self,
        stage: &str,
        spec: &CommandSpec,
        task: &OptimizationTask,
        iteration: usize,
        workspace: &Path,
        run_dir: &Path,
    ) -> Result<(), OptimizationError> {
        let timeout_secs = spec
            .timeout_secs
            .unwrap_or(task.budget.command_timeout_secs);
        let mut command = Command::new(&spec.program);
        command.args(&spec.args).current_dir(workspace);
        for secret in SECRET_ENV_VARS
        {
            command.env_remove(secret);
        }
        for (key, value) in &spec.env
        {
            command.env(key, value);
        }
        command.env("SCIAGENT_OPT_TASK_ID", &task.id);
        command.env("SCIAGENT_OPT_BACKEND", task.backend.to_string());
        command.env("SCIAGENT_OPT_GOAL", &task.goal);
        command.env("SCIAGENT_OPT_ITERATION", iteration.to_string());
        command.env("SCIAGENT_OPT_RUN_DIR", run_dir);
        command.env("SCIAGENT_OPT_CONTEXT", run_dir.join("context.json"));
        command.env(
            "SCIAGENT_OPT_BASELINE_METRICS",
            run_dir.join("baseline.json"),
        );
        command.env("SCIAGENT_OPT_VERIFY_METRICS", run_dir.join("verify.json"));
        command.env(
            "SCIAGENT_OPT_CANDIDATE_METRICS",
            run_dir.join("candidate.json"),
        );
        command.env(
            "SCIAGENT_OPT_SKILL_PATH",
            Path::new(env!("CARGO_MANIFEST_DIR")).join("SKILL_OPTIMIZATION.md"),
        );

        let log_path = stage_log_path(run_dir, iteration, stage);
        let stdout = File::create(&log_path).map_err(|source| OptimizationError::Io {
            path: log_path.clone(),
            source,
        })?;
        let stderr = stdout.try_clone().map_err(|source| OptimizationError::Io {
            path: log_path.clone(),
            source,
        })?;
        command
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr));

        let mut child = command
            .spawn()
            .map_err(|source| OptimizationError::CommandSpawn {
                stage: stage.to_string(),
                source,
            })?;
        let deadline = Instant::now() + Duration::from_secs(timeout_secs);
        loop
        {
            match child
                .try_wait()
                .map_err(|source| OptimizationError::CommandPoll {
                    stage: stage.to_string(),
                    source,
                })?
            {
                Some(status) =>
                {
                    if status.success()
                    {
                        return Ok(());
                    }
                    return Err(OptimizationError::CommandFailed {
                        stage: stage.to_string(),
                        code: status.code(),
                    });
                },
                None if Instant::now() >= deadline =>
                {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(OptimizationError::CommandTimedOut {
                        stage: stage.to_string(),
                        timeout_secs,
                    });
                },
                None => thread::sleep(Duration::from_millis(50)),
            }
        }
    }
}

pub fn evaluate_candidate(
    baseline: &TimingMeasurement,
    verification: &VerificationMeasurement,
    candidate: &TimingMeasurement,
    budget: &OptimizationBudget,
) -> Result<IterationRecord, OptimizationError> {
    baseline.validate("baseline")?;
    verification.validate()?;
    candidate.validate("candidate")?;
    let speedup = baseline.median_ns / candidate.median_ns;
    let correctness_gate = verification.passed
        && verification
            .max_abs_error
            .is_none_or(|value| value <= budget.max_abs_error)
        && verification
            .max_rel_error
            .is_none_or(|value| value <= budget.max_rel_error);
    let performance_gate = speedup >= budget.min_speedup;
    let decision = if correctness_gate && performance_gate
    {
        OptimizationDecision::Promote
    }
    else if !correctness_gate
    {
        OptimizationDecision::RewriteForCorrectness
    }
    else
    {
        OptimizationDecision::RewriteForPerformance
    };
    Ok(IterationRecord {
        iteration: 0,
        verification: verification.clone(),
        timing: candidate.clone(),
        speedup,
        correctness_gate,
        performance_gate,
        decision,
    })
}

pub fn load_task(path: impl AsRef<Path>) -> Result<OptimizationTask, OptimizationError> {
    let path = path.as_ref();
    let task: OptimizationTask = read_json(path)?;
    task.validate()?;
    Ok(task)
}

fn stage_log_path(run_dir: &Path, iteration: usize, stage: &str) -> PathBuf {
    run_dir.join(format!("{iteration:02}-{stage}.log"))
}

fn canonicalize_existing(path: &Path) -> Result<PathBuf, OptimizationError> {
    fs::canonicalize(path).map_err(|source| OptimizationError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn remove_if_exists(path: &Path) -> Result<(), OptimizationError> {
    match fs::remove_file(path)
    {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(OptimizationError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, OptimizationError> {
    let bytes = fs::read(path).map_err(|source| OptimizationError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| OptimizationError::Json {
        path: path.to_path_buf(),
        source,
    })
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), OptimizationError> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|source| OptimizationError::Json {
        path: path.to_path_buf(),
        source,
    })?;
    fs::write(path, bytes).map_err(|source| OptimizationError::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correct_fast_candidate_is_promoted() {
        let record = evaluate_candidate(
            &TimingMeasurement { median_ns: 100.0 },
            &VerificationMeasurement {
                passed: true,
                max_abs_error: Some(1.0e-9),
                max_rel_error: Some(2.0e-9),
                notes: None,
            },
            &TimingMeasurement { median_ns: 80.0 },
            &OptimizationBudget::default(),
        )
        .expect("valid metrics");
        assert_eq!(record.decision, OptimizationDecision::Promote);
        assert!((record.speedup - 1.25).abs() < 1.0e-12);
    }

    #[test]
    fn incorrect_candidate_never_promotes_even_if_fast() {
        let record = evaluate_candidate(
            &TimingMeasurement { median_ns: 100.0 },
            &VerificationMeasurement {
                passed: false,
                max_abs_error: Some(0.0),
                max_rel_error: Some(0.0),
                notes: None,
            },
            &TimingMeasurement { median_ns: 10.0 },
            &OptimizationBudget::default(),
        )
        .expect("valid metrics");
        assert_eq!(record.decision, OptimizationDecision::RewriteForCorrectness);
        assert!(!record.correctness_gate);
    }

    #[test]
    fn verified_but_slow_candidate_requests_performance_rewrite() {
        let record = evaluate_candidate(
            &TimingMeasurement { median_ns: 100.0 },
            &VerificationMeasurement {
                passed: true,
                max_abs_error: None,
                max_rel_error: None,
                notes: None,
            },
            &TimingMeasurement { median_ns: 99.0 },
            &OptimizationBudget::default(),
        )
        .expect("valid metrics");
        assert_eq!(record.decision, OptimizationDecision::RewriteForPerformance);
        assert!(record.correctness_gate);
        assert!(!record.performance_gate);
    }

    #[test]
    fn non_finite_timings_fail_closed() {
        let error = evaluate_candidate(
            &TimingMeasurement { median_ns: 100.0 },
            &VerificationMeasurement {
                passed: true,
                max_abs_error: None,
                max_rel_error: None,
                notes: None,
            },
            &TimingMeasurement {
                median_ns: f64::NAN,
            },
            &OptimizationBudget::default(),
        )
        .expect_err("NaN timing must be rejected");
        assert!(error.to_string().contains("candidate.median_ns"));
    }
}

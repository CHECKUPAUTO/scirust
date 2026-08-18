use clap::{Parser, Subcommand};
use scirust_sciagent::optimization_agent::{
    OptimizationBudget, OptimizationDecision, OptimizationRunner, TimingMeasurement,
    VerificationMeasurement, evaluate_candidate, load_task,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "sciagent-optimize",
    about = "Evidence-driven generate/compile/verify/benchmark/profile/rewrite loop for SciRust"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate a task manifest and print the deterministic execution plan.
    Plan {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Execute the optimization loop in the current or selected workspace.
    Run {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long, default_value = ".")]
        workspace: PathBuf,
        #[arg(long, default_value = ".sciagent-opt")]
        run_root: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Evaluate one already-measured candidate against the promotion gates.
    Score {
        #[arg(long)]
        baseline_ns: f64,
        #[arg(long)]
        candidate_ns: f64,
        #[arg(long)]
        verified: bool,
        #[arg(long)]
        max_abs_error: Option<f64>,
        #[arg(long)]
        max_rel_error: Option<f64>,
        #[arg(long, default_value_t = 1.05)]
        min_speedup: f64,
        #[arg(long, default_value_t = 1.0e-6)]
        abs_tolerance: f64,
        #[arg(long, default_value_t = 1.0e-6)]
        rel_tolerance: f64,
        #[arg(long)]
        json: bool,
    },
}

fn main() {
    let cli = Cli::parse();
    if let Err(error) = run(cli)
    {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command
    {
        Command::Plan { manifest, json } =>
        {
            let task = load_task(&manifest)?;
            if json
            {
                println!("{}", serde_json::to_string_pretty(&task)?);
            }
            else
            {
                println!("SCIAGENT optimization task: {}", task.id);
                println!("crate: {}", task.crate_name);
                println!("backend: {}", task.backend);
                println!("goal: {}", task.goal);
                println!("target speedup: {:.4}x", task.budget.min_speedup);
                println!("iteration budget: {}", task.budget.max_iterations);
                println!("allowed paths:");
                for path in &task.allowed_paths
                {
                    println!("  - {path}");
                }
                println!("stages:");
                println!("  0. baseline");
                println!("  1. generate");
                println!("  2. compile");
                println!("  3. verify");
                println!("  4. benchmark");
                if task.commands.profile.is_some()
                {
                    println!("  5. profile on correct-but-slow candidates");
                }
                if task.commands.rewrite.is_some()
                {
                    println!("  6. rewrite from accumulated evidence");
                }
                else
                {
                    println!("  6. regenerate from accumulated evidence");
                }
                println!("  7. promote only when correctness + speed gates pass");
            }
        },
        Command::Run {
            manifest,
            workspace,
            run_root,
            json,
        } =>
        {
            let task = load_task(&manifest)?;
            let runner = OptimizationRunner::new(workspace, run_root);
            let report = runner.run(&task)?;
            if json
            {
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
            else
            {
                println!("task: {}", report.task_id);
                println!("baseline: {:.3} ns", report.baseline.median_ns);
                for record in &report.iterations
                {
                    println!(
                        "iteration {}: {:.4}x, correctness={}, performance={}, decision={:?}",
                        record.iteration,
                        record.speedup,
                        record.correctness_gate,
                        record.performance_gate,
                        record.decision
                    );
                }
                println!("final decision: {:?}", report.final_decision);
                if let Some(best) = report.best_verified_speedup
                {
                    println!("best verified speedup: {best:.4}x");
                }
            }
            if report.final_decision != OptimizationDecision::Promote
            {
                std::process::exit(2);
            }
        },
        Command::Score {
            baseline_ns,
            candidate_ns,
            verified,
            max_abs_error,
            max_rel_error,
            min_speedup,
            abs_tolerance,
            rel_tolerance,
            json,
        } =>
        {
            let budget = OptimizationBudget {
                min_speedup,
                max_abs_error: abs_tolerance,
                max_rel_error: rel_tolerance,
                ..OptimizationBudget::default()
            };
            let record = evaluate_candidate(
                &TimingMeasurement {
                    median_ns: baseline_ns,
                },
                &VerificationMeasurement {
                    passed: verified,
                    max_abs_error,
                    max_rel_error,
                    notes: None,
                },
                &TimingMeasurement {
                    median_ns: candidate_ns,
                },
                &budget,
            )?;
            if json
            {
                println!("{}", serde_json::to_string_pretty(&record)?);
            }
            else
            {
                println!("speedup: {:.6}x", record.speedup);
                println!("correctness gate: {}", record.correctness_gate);
                println!("performance gate: {}", record.performance_gate);
                println!("decision: {:?}", record.decision);
            }
        },
    }
    Ok(())
}

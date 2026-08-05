mod causal;
mod data;
mod policy;
mod sequential;

use causal::{CausalDiagnostic, run_causal_diagnostic};
use data::{
    LabelSummary, PromptSplit, RUNTIME_FEATURE_NAMES, RUNTIME_FEATURES, Standardizer,
    read_candidate_jsonl, sequences_for_indices, split_by_prompt, summarize_labels,
};
use policy::{
    GpRiskReport, NsgaConfig, NsgaReport, RiskPredictions, discover_fail_closed_policy, fit_gp_risk,
};
use sequential::{SequentialRiskReport, fit_sequential_risk};
use serde::Serialize;
use std::fs;
use std::path::Path;

pub use data::CandidateRecord;

#[derive(Debug, Clone)]
pub struct TrajectoryDiscoveryConfig {
    pub seed: u64,
    pub crf_epochs: usize,
    pub crf_learning_rate: f64,
    pub crf_l2_penalty: f64,
    pub nsga_population: usize,
    pub nsga_generations: usize,
    pub minimum_holdout_coverage: f64,
    pub symbolic: bool,
}

impl Default for TrajectoryDiscoveryConfig {
    fn default() -> Self {
        Self {
            seed: 20_260_810,
            crf_epochs: 300,
            crf_learning_rate: 0.03,
            crf_l2_penalty: 0.002,
            nsga_population: 120,
            nsga_generations: 80,
            minimum_holdout_coverage: 0.02,
            symbolic: true,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SplitSummary {
    pub train: LabelSummary,
    pub validation: LabelSummary,
    pub holdout: LabelSummary,
    pub train_prompt_ids: Vec<usize>,
    pub validation_prompt_ids: Vec<usize>,
    pub holdout_prompt_ids: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SymbolicTrajectoryCandidate {
    pub size: usize,
    pub mean_squared_error: f64,
    pub expression: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SymbolicTrajectoryReport {
    pub enabled: bool,
    pub engine: String,
    pub target: String,
    pub candidates: Vec<SymbolicTrajectoryCandidate>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrajectoryDiscoveryReport {
    pub schema_version: u32,
    pub status: String,
    pub source_dataset: String,
    pub source_dataset_sha256: String,
    pub development_split: String,
    pub features: Vec<String>,
    pub strict_unsafe_definition: Vec<String>,
    pub split: SplitSummary,
    pub standardizer: Standardizer,
    pub causal: CausalDiagnostic,
    pub sequential: SequentialRiskReport,
    pub gaussian_process: GpRiskReport,
    pub nsga2: NsgaReport,
    pub symbolic: SymbolicTrajectoryReport,
    pub fail_closed_development_success: bool,
    pub success_criteria: Vec<String>,
    pub scientific_conclusion: String,
    pub evidence_boundary: String,
}

fn symbolic_report(
    rows: &[CandidateRecord],
    training_indices: &[usize],
    standardized: &[[f64; RUNTIME_FEATURES]],
    seed: u64,
    enabled: bool,
) -> SymbolicTrajectoryReport {
    if !enabled
    {
        return SymbolicTrajectoryReport {
            enabled: false,
            engine: "scirust-symreg".to_string(),
            target: "strict trajectory-unsafe indicator".to_string(),
            candidates: Vec::new(),
        };
    }
    let data: Vec<(Vec<f64>, f64)> = training_indices
        .iter()
        .map(|index| {
            (
                standardized[*index].to_vec(),
                if rows[*index].strict_unsafe { 1.0 } else { 0.0 },
            )
        })
        .collect();
    let input_names: Vec<&str> = RUNTIME_FEATURE_NAMES.to_vec();
    let seeds = [seed, seed.wrapping_add(1), seed.wrapping_add(2)];
    let front = scirust_symreg::discover(&data, &input_names, &seeds, 72, 10, 18, 18);
    SymbolicTrajectoryReport {
        enabled: true,
        engine: "scirust-symreg genetic programming with symbolic differentiation".to_string(),
        target: "strict trajectory-unsafe indicator".to_string(),
        candidates: front
            .into_iter()
            .take(16)
            .map(
                |(size, mean_squared_error, expression)| SymbolicTrajectoryCandidate {
                    size,
                    mean_squared_error,
                    expression: expression.to_string(),
                },
            )
            .collect(),
    }
}

fn source_sha256(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("cannot hash trajectory dataset {}: {error}", path.display()))?;
    Ok(scirust_causal::sha256_hex(&bytes))
}

pub fn discover_trajectory_policy(
    dataset_path: &Path,
    config: &TrajectoryDiscoveryConfig,
) -> Result<TrajectoryDiscoveryReport, String> {
    if !config.minimum_holdout_coverage.is_finite()
        || !(0.0..=1.0).contains(&config.minimum_holdout_coverage)
    {
        return Err("minimum holdout coverage must lie in [0,1]".to_string());
    }
    let rows = read_candidate_jsonl(dataset_path)?;
    let split: PromptSplit = split_by_prompt(&rows, config.seed)?;
    let standardizer = Standardizer::fit(&rows, &split.train_rows)?;
    let standardized: Vec<[f64; RUNTIME_FEATURES]> = rows
        .iter()
        .map(|row| standardizer.transform(&row.raw_features))
        .collect();

    let training_sequences = sequences_for_indices(&rows, &split.train_rows);
    let all_indices: Vec<usize> = (0..rows.len()).collect();
    let all_sequences = sequences_for_indices(&rows, &all_indices);
    let sequential_model = fit_sequential_risk(
        &rows,
        standardized.clone(),
        &training_sequences,
        config.crf_epochs,
        config.crf_learning_rate,
        config.crf_l2_penalty,
    )?;
    let crf_probabilities = sequential_model.unsafe_probabilities(&all_sequences, rows.len())?;
    if crf_probabilities
        .iter()
        .any(|probability| !probability.is_finite())
    {
        return Err("CRF did not assign every trajectory candidate a risk".to_string());
    }

    let (gp_report, gp_mean, gp_stddev) = fit_gp_risk(&rows, &standardized, &split.train_rows)?;
    let predictions = RiskPredictions {
        crf_unsafe_probability: crf_probabilities,
        gp_mean,
        gp_stddev,
    };
    let causal = run_causal_diagnostic(&rows, &split.train_rows)?;
    let nsga2 = discover_fail_closed_policy(
        &rows,
        &split.train_rows,
        &split.validation_rows,
        &split.holdout_rows,
        &predictions,
        NsgaConfig {
            seed: config.seed.wrapping_add(100),
            population: config.nsga_population,
            generations: config.nsga_generations,
        },
    )?;
    let symbolic = symbolic_report(
        &rows,
        &split.train_rows,
        &standardized,
        config.seed.wrapping_add(200),
        config.symbolic,
    );

    let development_success = nsga2.validation.strict_unsafe_allowed == 0
        && nsga2.holdout.strict_unsafe_allowed == 0
        && nsga2.validation.quality_regressions_allowed == 0
        && nsga2.holdout.quality_regressions_allowed == 0
        && nsga2.validation.allowed > 0
        && nsga2.holdout.allowed > 0
        && nsga2.holdout.coverage + 1e-15 >= config.minimum_holdout_coverage
        && nsga2.holdout.selected_net_refresh_saving > 0.0;

    Ok(TrajectoryDiscoveryReport {
        schema_version: 1,
        status: "causal_sequential_trajectory_policy_development".to_string(),
        source_dataset: dataset_path.display().to_string(),
        source_dataset_sha256: source_sha256(dataset_path)?,
        development_split: "GSM8K train only".to_string(),
        features: RUNTIME_FEATURE_NAMES
            .iter()
            .map(|name| (*name).to_string())
            .collect(),
        strict_unsafe_definition: vec![
            "final response hash changes".to_string(),
            "numeric prediction changes".to_string(),
            "correctness changes or regresses".to_string(),
            "generation decision count changes".to_string(),
        ],
        split: SplitSummary {
            train: summarize_labels(&rows, &split.train_rows),
            validation: summarize_labels(&rows, &split.validation_rows),
            holdout: summarize_labels(&rows, &split.holdout_rows),
            train_prompt_ids: split.train_prompts,
            validation_prompt_ids: split.validation_prompts,
            holdout_prompt_ids: split.holdout_prompts,
        },
        standardizer,
        causal,
        sequential: sequential_model.report,
        gaussian_process: gp_report,
        nsga2,
        symbolic,
        fail_closed_development_success: development_success,
        success_criteria: vec![
            "zero strict-unsafe skips permitted on validation".to_string(),
            "zero strict-unsafe skips permitted on internal GSM8K-train holdout"
                .to_string(),
            "zero quality regressions permitted on both held-out development splits"
                .to_string(),
            format!(
                "holdout coverage at least {:.4}",
                config.minimum_holdout_coverage
            ),
            "positive net refresh-cost saving on the internal holdout".to_string(),
            "deny all skips when no candidate satisfies every constraint".to_string(),
        ],
        scientific_conclusion: if development_success {
            "A fail-closed trajectory policy survived internal GSM8K-train development splits; it is not independently confirmed and must be frozen before any untouched task evaluation."
                .to_string()
        } else {
            "No admissible trajectory policy satisfied every fail-closed development criterion; the correct runtime action remains always refresh."
                .to_string()
        },
        evidence_boundary: "All labels come from atomic single-skip interventions on GSM8K train. The 120 previously consumed GSM8K test prompts are not read. This phase is development, not confirmation."
            .to_string(),
    })
}

use super::data::{CandidateRecord, RUNTIME_FEATURES};
use scirust_evo::{MoIndividual, Nsga2};
use scirust_gp::{GaussianProcess, Matern52};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct GpRiskReport {
    pub model: String,
    pub kernel: String,
    pub lengthscale: f64,
    pub variance: f64,
    pub noise_variance: f64,
    pub training_points: usize,
    pub log_marginal_likelihood: f64,
}

#[derive(Debug, Clone)]
pub struct RiskPredictions {
    pub crf_unsafe_probability: Vec<f64>,
    pub gp_mean: Vec<f64>,
    pub gp_stddev: Vec<f64>,
}

pub fn fit_gp_risk(
    rows: &[CandidateRecord],
    standardized: &[[f64; RUNTIME_FEATURES]],
    training_indices: &[usize],
) -> Result<(GpRiskReport, Vec<f64>, Vec<f64>), String> {
    if training_indices.is_empty() {
        return Err("cannot fit a Gaussian process on an empty split".to_string());
    }
    if standardized.len() != rows.len() {
        return Err("GP feature table length does not match trajectory rows".to_string());
    }
    let x: Vec<Vec<f64>> = training_indices
        .iter()
        .map(|index| standardized[*index].to_vec())
        .collect();
    let y: Vec<f64> = training_indices
        .iter()
        .map(|index| f64::from(rows[*index].strict_unsafe))
        .collect();
    let kernel = Matern52 {
        lengthscale: 1.5,
        variance: 1.0,
    };
    let mut fitted = None;
    for noise in [0.01, 0.03, 0.05, 0.1, 0.2] {
        if let Ok(gp) = GaussianProcess::fit(&x, &y, kernel, noise) {
            fitted = Some((gp, noise));
            break;
        }
    }
    let (gp, noise_variance) = fitted.ok_or_else(|| {
        "scirust-gp could not factor the trajectory covariance matrix".to_string()
    })?;
    let mut means = Vec::with_capacity(rows.len());
    let mut standard_deviations = Vec::with_capacity(rows.len());
    for features in standardized {
        let (mean, variance) = gp.predict(features);
        if !mean.is_finite() || !variance.is_finite() || variance < 0.0 {
            return Err("scirust-gp produced a non-finite trajectory prediction".to_string());
        }
        means.push(mean.clamp(0.0, 1.0));
        standard_deviations.push(variance.sqrt());
    }
    let report = GpRiskReport {
        model: "exact deterministic Gaussian-process risk regressor".to_string(),
        kernel: "Matern-5/2".to_string(),
        lengthscale: kernel.lengthscale,
        variance: kernel.variance,
        noise_variance,
        training_points: training_indices.len(),
        log_marginal_likelihood: gp.log_marginal_likelihood(),
    };
    Ok((report, means, standard_deviations))
}

#[derive(Debug, Clone, Serialize)]
pub struct PolicyParameters {
    pub maximum_crf_unsafe_probability: f64,
    pub maximum_gp_upper_risk: f64,
    pub gp_uncertainty_multiplier: f64,
    pub minimum_skip_margin: f64,
    pub maximum_candidate_ordinal: usize,
    pub maximum_refresh_votes: usize,
}

impl PolicyParameters {
    pub fn deny_all() -> Self {
        Self {
            maximum_crf_unsafe_probability: 0.0,
            maximum_gp_upper_risk: 0.0,
            gp_uncertainty_multiplier: 4.0,
            minimum_skip_margin: f64::INFINITY,
            maximum_candidate_ordinal: 0,
            maximum_refresh_votes: 0,
        }
    }

    pub fn allows(
        &self,
        row: &CandidateRecord,
        crf_probability: f64,
        gp_mean: f64,
        gp_stddev: f64,
    ) -> bool {
        if self.maximum_candidate_ordinal == 0 {
            return false;
        }
        if !crf_probability.is_finite() || !gp_mean.is_finite() || !gp_stddev.is_finite() {
            return false;
        }
        let upper_risk = gp_mean + self.gp_uncertainty_multiplier * gp_stddev;
        crf_probability <= self.maximum_crf_unsafe_probability
            && upper_risk <= self.maximum_gp_upper_risk
            && row.skip_margin + 1e-15 >= self.minimum_skip_margin
            && row.ordinal <= self.maximum_candidate_ordinal
            && row.votes <= self.maximum_refresh_votes
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PolicyMetrics {
    pub candidates: usize,
    pub strict_unsafe_candidates: usize,
    pub semantic_unsafe_candidates: usize,
    pub trajectory_unsafe_candidates: usize,
    pub allowed: usize,
    pub coverage: f64,
    pub safe_candidates: usize,
    pub safe_allowed: usize,
    pub safe_coverage: f64,
    pub strict_unsafe_allowed: usize,
    pub semantic_unsafe_allowed: usize,
    pub trajectory_unsafe_allowed: usize,
    pub quality_regressions_allowed: usize,
    pub prediction_changes_allowed: usize,
    pub decision_count_changes_allowed: usize,
    pub response_changes_allowed: usize,
    pub selected_positive_refresh_saving: f64,
    pub available_safe_positive_refresh_saving: f64,
    pub safe_refresh_saving_capture: f64,
    pub selected_net_refresh_saving: f64,
    pub mean_selected_latency_improvement: f64,
    pub mean_selected_refresh_cost_improvement: f64,
    pub mean_selected_decision_delta: f64,
}

pub fn evaluate_policy(
    parameters: &PolicyParameters,
    rows: &[CandidateRecord],
    indices: &[usize],
    predictions: &RiskPredictions,
) -> PolicyMetrics {
    let mut allowed = 0usize;
    let mut safe_allowed = 0usize;
    let mut strict_unsafe_allowed = 0usize;
    let mut semantic_unsafe_allowed = 0usize;
    let mut trajectory_unsafe_allowed = 0usize;
    let mut quality_regressions_allowed = 0usize;
    let mut prediction_changes_allowed = 0usize;
    let mut decision_count_changes_allowed = 0usize;
    let mut response_changes_allowed = 0usize;
    let mut selected_positive_refresh_saving = 0.0;
    let mut selected_net_refresh_saving = 0.0;
    let mut available_safe_positive_refresh_saving = 0.0;
    let mut latency_sum = 0.0;
    let mut refresh_improvement_sum = 0.0;
    let mut decision_delta_sum = 0.0;

    let strict_unsafe_candidates = indices
        .iter()
        .filter(|index| rows[**index].strict_unsafe)
        .count();
    let semantic_unsafe_candidates = indices
        .iter()
        .filter(|index| rows[**index].semantic_unsafe)
        .count();
    let trajectory_unsafe_candidates = indices
        .iter()
        .filter(|index| rows[**index].trajectory_unsafe)
        .count();
    let safe_candidates = indices.len() - strict_unsafe_candidates;

    for &index in indices {
        let row = &rows[index];
        if !row.strict_unsafe {
            available_safe_positive_refresh_saving += row.saved_refresh_cost.max(0.0);
        }
        let permit = parameters.allows(
            row,
            predictions.crf_unsafe_probability[index],
            predictions.gp_mean[index],
            predictions.gp_stddev[index],
        );
        if !permit {
            continue;
        }
        allowed += 1;
        if row.strict_unsafe {
            strict_unsafe_allowed += 1;
        } else {
            safe_allowed += 1;
        }
        semantic_unsafe_allowed += usize::from(row.semantic_unsafe);
        trajectory_unsafe_allowed += usize::from(row.trajectory_unsafe);
        quality_regressions_allowed += usize::from(row.quality_regression);
        prediction_changes_allowed += usize::from(row.prediction_changed);
        decision_count_changes_allowed += usize::from(row.decision_count_changed);
        response_changes_allowed += usize::from(row.response_changed);
        selected_positive_refresh_saving += row.saved_refresh_cost.max(0.0);
        selected_net_refresh_saving += row.saved_refresh_cost;
        latency_sum += row.latency_improvement;
        refresh_improvement_sum += row.refresh_cost_improvement;
        decision_delta_sum += row.decision_delta as f64;
    }

    let safe_refresh_saving_capture = if available_safe_positive_refresh_saving > 0.0 {
        selected_positive_refresh_saving / available_safe_positive_refresh_saving
    } else {
        0.0
    };
    PolicyMetrics {
        candidates: indices.len(),
        strict_unsafe_candidates,
        semantic_unsafe_candidates,
        trajectory_unsafe_candidates,
        allowed,
        coverage: allowed as f64 / indices.len().max(1) as f64,
        safe_candidates,
        safe_allowed,
        safe_coverage: safe_allowed as f64 / safe_candidates.max(1) as f64,
        strict_unsafe_allowed,
        semantic_unsafe_allowed,
        trajectory_unsafe_allowed,
        quality_regressions_allowed,
        prediction_changes_allowed,
        decision_count_changes_allowed,
        response_changes_allowed,
        selected_positive_refresh_saving,
        available_safe_positive_refresh_saving,
        safe_refresh_saving_capture,
        selected_net_refresh_saving,
        mean_selected_latency_improvement: latency_sum / allowed.max(1) as f64,
        mean_selected_refresh_cost_improvement: refresh_improvement_sum
            / allowed.max(1) as f64,
        mean_selected_decision_delta: decision_delta_sum / allowed.max(1) as f64,
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponent = value.exp();
        exponent / (1.0 + exponent)
    }
}

fn decode_genome(genome: &[f64], minimum_margin: f64, maximum_margin: f64) -> PolicyParameters {
    let range = (maximum_margin - minimum_margin).max(0.0);
    let maximum_candidate_ordinal = 1 + ((4.0 * sigmoid(genome[4])).floor() as usize).min(3);
    let maximum_refresh_votes = ((3.0 * sigmoid(genome[5])).floor() as usize).min(2);
    PolicyParameters {
        maximum_crf_unsafe_probability: 0.005 + 0.495 * sigmoid(genome[0]),
        maximum_gp_upper_risk: 0.005 + 0.495 * sigmoid(genome[1]),
        gp_uncertainty_multiplier: 0.5 + 3.5 * sigmoid(genome[2]),
        minimum_skip_margin: minimum_margin + range * sigmoid(genome[3]),
        maximum_candidate_ordinal,
        maximum_refresh_votes,
    }
}

fn objectives(metrics: &PolicyMetrics) -> Vec<f64> {
    let false_safe_rate = metrics.strict_unsafe_allowed as f64
        / metrics.strict_unsafe_candidates.max(1) as f64;
    let missed_safe_saving = 1.0 - metrics.safe_refresh_saving_capture.clamp(0.0, 1.0);
    let missed_safe_coverage = 1.0 - metrics.safe_coverage.clamp(0.0, 1.0);
    vec![false_safe_rate, missed_safe_saving, missed_safe_coverage]
}

#[derive(Debug, Clone, Serialize)]
pub struct NsgaReport {
    pub optimizer: String,
    pub seed: u64,
    pub population: usize,
    pub generations: usize,
    pub objectives: Vec<String>,
    pub zero_false_safe_candidates: usize,
    pub selected_from_pareto_rank_one: bool,
    pub selected_parameters: PolicyParameters,
    pub training: PolicyMetrics,
    pub validation: PolicyMetrics,
    pub holdout: PolicyMetrics,
}

pub struct NsgaConfig {
    pub seed: u64,
    pub population: usize,
    pub generations: usize,
}

pub fn discover_fail_closed_policy(
    rows: &[CandidateRecord],
    training_indices: &[usize],
    validation_indices: &[usize],
    holdout_indices: &[usize],
    predictions: &RiskPredictions,
    config: NsgaConfig,
) -> Result<NsgaReport, String> {
    if validation_indices.is_empty() || training_indices.is_empty() || holdout_indices.is_empty() {
        return Err("NSGA-II requires non-empty train, validation, and holdout splits".to_string());
    }
    if config.population < 8 || config.generations == 0 {
        return Err("invalid NSGA-II population or generation count".to_string());
    }
    let minimum_margin = validation_indices
        .iter()
        .map(|index| rows[*index].skip_margin)
        .fold(f64::INFINITY, f64::min);
    let maximum_margin = validation_indices
        .iter()
        .map(|index| rows[*index].skip_margin)
        .fold(f64::NEG_INFINITY, f64::max);
    if !minimum_margin.is_finite() || !maximum_margin.is_finite() {
        return Err("validation skip-margin range is non-finite".to_string());
    }

    let mut optimizer = Nsga2::seeded(config.seed);
    optimizer.pop_size = config.population;
    optimizer.bounds = (-6.0, 6.0);
    optimizer.mutation_rate = 0.18;
    optimizer.crossover_rate = 0.9;
    let mut population = optimizer.init_pop(6);
    for _ in 0..config.generations {
        optimizer.evolve(&mut population, |individuals| {
            individuals
                .iter()
                .map(|individual| {
                    let parameters = decode_genome(
                        &individual.genome,
                        minimum_margin,
                        maximum_margin,
                    );
                    let metrics = evaluate_policy(
                        &parameters,
                        rows,
                        validation_indices,
                        predictions,
                    );
                    objectives(&metrics)
                })
                .collect()
        });
    }

    let mut zero_false_safe = Vec::<(PolicyParameters, PolicyMetrics, bool)>::new();
    for individual in &population {
        let parameters = decode_genome(&individual.genome, minimum_margin, maximum_margin);
        let metrics = evaluate_policy(&parameters, rows, validation_indices, predictions);
        if metrics.strict_unsafe_allowed == 0 && metrics.allowed > 0 {
            zero_false_safe.push((parameters, metrics, individual.rank == 1));
        }
    }
    zero_false_safe.sort_by(|left, right| {
        right
            .1
            .selected_positive_refresh_saving
            .total_cmp(&left.1.selected_positive_refresh_saving)
            .then_with(|| right.1.safe_allowed.cmp(&left.1.safe_allowed))
            .then_with(|| {
                left.0
                    .maximum_gp_upper_risk
                    .total_cmp(&right.0.maximum_gp_upper_risk)
            })
            .then_with(|| {
                left.0
                    .maximum_crf_unsafe_probability
                    .total_cmp(&right.0.maximum_crf_unsafe_probability)
            })
    });
    let zero_false_safe_candidates = zero_false_safe.len();
    let (selected_parameters, validation, rank_one) = zero_false_safe
        .into_iter()
        .next()
        .unwrap_or_else(|| {
            let parameters = PolicyParameters::deny_all();
            let metrics = evaluate_policy(&parameters, rows, validation_indices, predictions);
            (parameters, metrics, false)
        });
    let training = evaluate_policy(&selected_parameters, rows, training_indices, predictions);
    let holdout = evaluate_policy(&selected_parameters, rows, holdout_indices, predictions);

    Ok(NsgaReport {
        optimizer: "scirust-evo NSGA-II".to_string(),
        seed: config.seed,
        population: config.population,
        generations: config.generations,
        objectives: vec![
            "minimize strict-unsafe candidates incorrectly permitted".to_string(),
            "maximize safe positive refresh-cost saving captured".to_string(),
            "maximize safe candidate coverage".to_string(),
        ],
        zero_false_safe_candidates,
        selected_from_pareto_rank_one: rank_one,
        selected_parameters,
        training,
        validation,
        holdout,
    })
}

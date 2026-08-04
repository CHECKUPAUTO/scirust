use crate::model::{
    DiscoveryConfig, DiscoveryResult, FEATURE_NAMES, GammaBaseline, HoldoutComparison,
    LinearPolicy, PolicyMetrics, RobustHoldoutComparison, SymbolicCandidate, TraceRow,
    TrajectoryPolicyMetrics, dot, evaluate_policy, evaluate_policy_by_trajectory,
    nearest_rank_quantile,
};
use scirust_evo::CmaEs;
use std::collections::BTreeMap;

pub fn best_fixed_gamma(rows: &[TraceRow], max_quality_loss: f64) -> GammaBaseline {
    let mut best: Option<GammaBaseline> = None;
    for i in 0..=2_000
    {
        let gamma = i as f64 / 2_000.0;
        let metrics = evaluate_policy(rows, |row| row.similarity < gamma);
        if metrics.quality_loss_fraction <= max_quality_loss + 1e-12
            && best
                .as_ref()
                .is_none_or(|current| metrics.compute_fraction < current.metrics.compute_fraction)
        {
            best = Some(GammaBaseline { gamma, metrics });
        }
    }
    best.unwrap_or_else(|| GammaBaseline {
        gamma: f64::INFINITY,
        metrics: evaluate_policy(rows, |_| true),
    })
}

pub fn best_fixed_gamma_robust(
    rows: &[TraceRow],
    max_quality_loss: f64,
    tail_quality_quantile: f64,
) -> (GammaBaseline, TrajectoryPolicyMetrics) {
    let mut best: Option<(GammaBaseline, TrajectoryPolicyMetrics)> = None;
    for i in 0..=2_000
    {
        let gamma = i as f64 / 2_000.0;
        let metrics = evaluate_policy(rows, |row| row.similarity < gamma);
        let trajectory = evaluate_policy_by_trajectory(rows, tail_quality_quantile, |row| {
            row.similarity < gamma
        });
        let meets_budget = metrics.quality_loss_fraction <= max_quality_loss + 1e-12
            && trajectory.mean_quality_loss_fraction <= max_quality_loss + 1e-12
            && trajectory.tail_quality_loss_fraction <= max_quality_loss + 1e-12;
        if meets_budget
            && best.as_ref().is_none_or(|(current, _)| {
                metrics.compute_fraction < current.metrics.compute_fraction
            })
        {
            best = Some((GammaBaseline { gamma, metrics }, trajectory));
        }
    }
    best.unwrap_or_else(|| {
        (
            GammaBaseline {
                gamma: f64::INFINITY,
                metrics: evaluate_policy(rows, |_| true),
            },
            evaluate_policy_by_trajectory(rows, tail_quality_quantile, |_| true),
        )
    })
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0
    {
        1.0 / (1.0 + (-value).exp())
    }
    else
    {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn build_trajectory_groups(rows: &[TraceRow]) -> Vec<Vec<usize>> {
    let mut grouped: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for (index, row) in rows.iter().enumerate()
    {
        grouped.entry(row.trajectory_id).or_default().push(index);
    }
    grouped.into_values().collect()
}

fn smooth_objective(
    rows: &[TraceRow],
    trajectory_groups: &[Vec<usize>],
    theta: &[f64],
    config: DiscoveryConfig,
) -> f64 {
    let l2 = theta.iter().map(|value| value * value).sum::<f64>();
    if !config.trajectory_balanced
    {
        let total_loss = rows
            .iter()
            .map(|row| row.stale_loss)
            .sum::<f64>()
            .max(f64::EPSILON);
        let total_cost = rows
            .iter()
            .map(|row| row.refresh_cost)
            .sum::<f64>()
            .max(f64::EPSILON);
        let mut incurred_loss = 0.0;
        let mut refresh_cost = 0.0;

        for row in rows
        {
            let features = row.features();
            let score = theta[8]
                + theta[..8]
                    .iter()
                    .zip(features)
                    .map(|(weight, feature)| weight * feature)
                    .sum::<f64>();
            let refresh_probability = sigmoid(score / 0.12);
            incurred_loss += (1.0 - refresh_probability) * row.stale_loss;
            refresh_cost += refresh_probability * row.refresh_cost;
        }

        let quality = incurred_loss / total_loss;
        let compute = refresh_cost / total_cost;
        let excess = (quality - config.max_quality_loss).max(0.0);
        return compute + 500.0 * excess * excess + 0.0005 * l2;
    }

    let mut trajectory_qualities = Vec::with_capacity(trajectory_groups.len());
    let mut mean_compute = 0.0;
    for group in trajectory_groups
    {
        let mut total_loss = 0.0;
        let mut total_cost = 0.0;
        let mut incurred_loss = 0.0;
        let mut refresh_cost = 0.0;
        for &index in group
        {
            let row = rows[index];
            let features = row.features();
            let score = theta[8]
                + theta[..8]
                    .iter()
                    .zip(features)
                    .map(|(weight, feature)| weight * feature)
                    .sum::<f64>();
            let refresh_probability = sigmoid(score / 0.12);
            total_loss += row.stale_loss;
            total_cost += row.refresh_cost;
            incurred_loss += (1.0 - refresh_probability) * row.stale_loss;
            refresh_cost += refresh_probability * row.refresh_cost;
        }
        trajectory_qualities.push(incurred_loss / total_loss.max(f64::EPSILON));
        mean_compute += refresh_cost / total_cost.max(f64::EPSILON);
    }

    let trajectories = trajectory_groups.len().max(1) as f64;
    mean_compute /= trajectories;
    let mean_quality = trajectory_qualities.iter().sum::<f64>() / trajectories;
    let tail_quality =
        nearest_rank_quantile(&mut trajectory_qualities, config.tail_quality_quantile);
    let mean_excess = (mean_quality - config.max_quality_loss).max(0.0);
    let tail_excess = (tail_quality - config.max_quality_loss).max(0.0);

    mean_compute
        + 500.0 * mean_excess * mean_excess
        + 500.0 * config.tail_penalty_weight * tail_excess * tail_excess
        + 0.0005 * l2
}

/// Cheapest hard threshold satisfying the aggregate quality budget for fixed risk weights.
pub fn calibrate_threshold(
    rows: &[TraceRow],
    weights: &[f64; 8],
    max_quality_loss: f64,
) -> Option<(f64, PolicyMetrics)> {
    if rows.is_empty()
    {
        return None;
    }

    let mut ranked: Vec<(f64, usize)> = rows
        .iter()
        .enumerate()
        .map(|(index, row)| (dot(weights, &row.features()), index))
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    let total_loss = rows
        .iter()
        .map(|row| row.stale_loss)
        .sum::<f64>()
        .max(f64::EPSILON);
    let total_cost = rows
        .iter()
        .map(|row| row.refresh_cost)
        .sum::<f64>()
        .max(f64::EPSILON);
    let mut incurred_loss = rows.iter().map(|row| row.stale_loss).sum::<f64>();
    let mut refresh_cost = 0.0;
    let mut refreshes = 0usize;
    let mut cursor = 0usize;

    let no_refresh = PolicyMetrics {
        quality_loss_fraction: incurred_loss / total_loss,
        compute_fraction: 0.0,
        refresh_rate: 0.0,
    };
    if no_refresh.quality_loss_fraction <= max_quality_loss + 1e-12
    {
        return Some((f64::INFINITY, no_refresh));
    }

    while cursor < ranked.len()
    {
        let threshold = ranked[cursor].0;
        let mut end = cursor;
        while end < ranked.len() && ranked[end].0.total_cmp(&threshold).is_eq()
        {
            let row = rows[ranked[end].1];
            incurred_loss -= row.stale_loss;
            refresh_cost += row.refresh_cost;
            refreshes += 1;
            end += 1;
        }
        let metrics = PolicyMetrics {
            quality_loss_fraction: (incurred_loss / total_loss).max(0.0),
            compute_fraction: refresh_cost / total_cost,
            refresh_rate: refreshes as f64 / rows.len() as f64,
        };
        if metrics.quality_loss_fraction <= max_quality_loss + 1e-12
        {
            return Some((threshold, metrics));
        }
        cursor = end;
    }

    Some((
        f64::NEG_INFINITY,
        PolicyMetrics {
            quality_loss_fraction: 0.0,
            compute_fraction: 1.0,
            refresh_rate: 1.0,
        },
    ))
}

pub fn calibrate_threshold_robust(
    rows: &[TraceRow],
    weights: &[f64; 8],
    max_quality_loss: f64,
    tail_quality_quantile: f64,
) -> Option<(f64, PolicyMetrics, TrajectoryPolicyMetrics)> {
    #[derive(Debug)]
    struct TrajectoryAccumulator {
        total_loss: f64,
        total_cost: f64,
        incurred_loss: f64,
        refresh_cost: f64,
        rows: usize,
        refreshes: usize,
    }

    if rows.is_empty()
    {
        return None;
    }

    let mut trajectory_index = BTreeMap::new();
    for row in rows
    {
        let next = trajectory_index.len();
        trajectory_index.entry(row.trajectory_id).or_insert(next);
    }
    let mut accumulators: Vec<TrajectoryAccumulator> = (0..trajectory_index.len())
        .map(|_| TrajectoryAccumulator {
            total_loss: 0.0,
            total_cost: 0.0,
            incurred_loss: 0.0,
            refresh_cost: 0.0,
            rows: 0,
            refreshes: 0,
        })
        .collect();
    let mut row_to_trajectory = Vec::with_capacity(rows.len());
    for row in rows
    {
        let index = trajectory_index[&row.trajectory_id];
        row_to_trajectory.push(index);
        let accumulator = &mut accumulators[index];
        accumulator.total_loss += row.stale_loss;
        accumulator.total_cost += row.refresh_cost;
        accumulator.incurred_loss += row.stale_loss;
        accumulator.rows += 1;
    }

    let total_loss = rows
        .iter()
        .map(|row| row.stale_loss)
        .sum::<f64>()
        .max(f64::EPSILON);
    let total_cost = rows
        .iter()
        .map(|row| row.refresh_cost)
        .sum::<f64>()
        .max(f64::EPSILON);
    let mut incurred_loss = rows.iter().map(|row| row.stale_loss).sum::<f64>();
    let mut refresh_cost = 0.0;
    let mut refreshes = 0usize;

    let trajectory_metrics = |accumulators: &[TrajectoryAccumulator]| {
        let mut qualities = Vec::with_capacity(accumulators.len());
        let mut compute_sum = 0.0;
        let mut refresh_rate_sum = 0.0;
        for accumulator in accumulators
        {
            qualities.push(accumulator.incurred_loss / accumulator.total_loss.max(f64::EPSILON));
            compute_sum += accumulator.refresh_cost / accumulator.total_cost.max(f64::EPSILON);
            refresh_rate_sum += accumulator.refreshes as f64 / accumulator.rows.max(1) as f64;
        }
        let trajectories = qualities.len();
        let mean_quality = qualities.iter().sum::<f64>() / trajectories.max(1) as f64;
        let worst_quality = qualities.iter().copied().fold(0.0, f64::max);
        let tail_quality = nearest_rank_quantile(&mut qualities, tail_quality_quantile);
        TrajectoryPolicyMetrics {
            trajectories,
            mean_quality_loss_fraction: mean_quality,
            tail_quality_loss_fraction: tail_quality,
            worst_quality_loss_fraction: worst_quality,
            mean_compute_fraction: compute_sum / trajectories.max(1) as f64,
            mean_refresh_rate: refresh_rate_sum / trajectories.max(1) as f64,
        }
    };

    let aggregate_metrics =
        |incurred_loss: f64, refresh_cost: f64, refreshes: usize| PolicyMetrics {
            quality_loss_fraction: (incurred_loss / total_loss).max(0.0),
            compute_fraction: refresh_cost / total_cost,
            refresh_rate: refreshes as f64 / rows.len() as f64,
        };

    let meets_budget = |aggregate: PolicyMetrics, trajectory: TrajectoryPolicyMetrics| {
        aggregate.quality_loss_fraction <= max_quality_loss + 1e-12
            && trajectory.mean_quality_loss_fraction <= max_quality_loss + 1e-12
            && trajectory.tail_quality_loss_fraction <= max_quality_loss + 1e-12
    };

    let no_refresh = aggregate_metrics(incurred_loss, refresh_cost, refreshes);
    let no_refresh_trajectory = trajectory_metrics(&accumulators);
    if meets_budget(no_refresh, no_refresh_trajectory)
    {
        return Some((f64::INFINITY, no_refresh, no_refresh_trajectory));
    }

    let mut ranked: Vec<(f64, usize)> = rows
        .iter()
        .enumerate()
        .map(|(index, row)| (dot(weights, &row.features()), index))
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    let mut cursor = 0usize;
    while cursor < ranked.len()
    {
        let threshold = ranked[cursor].0;
        let mut end = cursor;
        while end < ranked.len() && ranked[end].0.total_cmp(&threshold).is_eq()
        {
            let row_index = ranked[end].1;
            let row = rows[row_index];
            let accumulator = &mut accumulators[row_to_trajectory[row_index]];
            incurred_loss -= row.stale_loss;
            refresh_cost += row.refresh_cost;
            refreshes += 1;
            accumulator.incurred_loss -= row.stale_loss;
            accumulator.refresh_cost += row.refresh_cost;
            accumulator.refreshes += 1;
            end += 1;
        }
        let aggregate = aggregate_metrics(incurred_loss, refresh_cost, refreshes);
        let trajectory = trajectory_metrics(&accumulators);
        if meets_budget(aggregate, trajectory)
        {
            return Some((threshold, aggregate, trajectory));
        }
        cursor = end;
    }

    let all_refresh = PolicyMetrics {
        quality_loss_fraction: 0.0,
        compute_fraction: 1.0,
        refresh_rate: 1.0,
    };
    let all_refresh_trajectory = TrajectoryPolicyMetrics {
        trajectories: accumulators.len(),
        mean_quality_loss_fraction: 0.0,
        tail_quality_loss_fraction: 0.0,
        worst_quality_loss_fraction: 0.0,
        mean_compute_fraction: 1.0,
        mean_refresh_rate: 1.0,
    };
    Some((f64::NEG_INFINITY, all_refresh, all_refresh_trajectory))
}

pub fn discover_linear_policy(
    training: &[TraceRow],
    validation: &[TraceRow],
    config: DiscoveryConfig,
) -> Result<DiscoveryResult, String> {
    if training.is_empty() || validation.is_empty()
    {
        return Err("training and validation traces must both be non-empty".into());
    }
    if !(0.0..=1.0).contains(&config.max_quality_loss)
    {
        return Err("max_quality_loss must lie in [0,1]".into());
    }
    if !config.calibration_budget_fraction.is_finite()
        || !(0.0..=1.0).contains(&config.calibration_budget_fraction)
        || config.calibration_budget_fraction == 0.0
    {
        return Err("calibration_budget_fraction must lie in (0,1]".into());
    }
    if !config.tail_quality_quantile.is_finite()
        || !(0.0..=1.0).contains(&config.tail_quality_quantile)
        || config.tail_quality_quantile == 0.0
    {
        return Err("tail_quality_quantile must lie in (0,1]".into());
    }
    if !config.tail_penalty_weight.is_finite() || config.tail_penalty_weight < 0.0
    {
        return Err("tail_penalty_weight must be finite and non-negative".into());
    }
    if config.steps == 0
    {
        return Err("steps must be greater than zero".into());
    }
    if !config.initial_sigma.is_finite()
        || !config.minimum_sigma.is_finite()
        || config.initial_sigma <= 0.0
        || config.minimum_sigma <= 0.0
        || config.minimum_sigma > config.initial_sigma
    {
        return Err("sigma values must be finite, positive, and ordered".into());
    }

    let trajectory_groups = build_trajectory_groups(training);
    let mut optimizer = CmaEs::seeded(9, config.seed);
    optimizer.bounds = (-8.0, 8.0);
    optimizer.sigma = config.initial_sigma;
    let mut theta = vec![0.0; 9];
    let mut best_theta = theta.clone();
    let mut best_fitness = f64::NEG_INFINITY;

    for _ in 0..config.steps
    {
        let population = optimizer.step(&mut theta, |candidate| {
            -smooth_objective(training, &trajectory_groups, candidate, config)
        });
        for individual in population
        {
            if individual.fitness.total_cmp(&best_fitness).is_gt()
            {
                best_fitness = individual.fitness;
                best_theta.clone_from(&individual.genome);
            }
        }
        optimizer.sigma = (optimizer.sigma * 0.995).max(config.minimum_sigma);
    }

    let mut weights = [0.0; 8];
    weights.copy_from_slice(&best_theta[..8]);
    let calibration_budget = config.max_quality_loss * config.calibration_budget_fraction;
    let (threshold, validation_metrics, validation_trajectory) = if config.trajectory_balanced
    {
        calibrate_threshold_robust(
            validation,
            &weights,
            calibration_budget,
            config.tail_quality_quantile,
        )
        .ok_or_else(|| "cannot calibrate an empty validation trace".to_string())?
    }
    else
    {
        let (threshold, metrics) = calibrate_threshold(validation, &weights, calibration_budget)
            .ok_or_else(|| "cannot calibrate an empty validation trace".to_string())?;
        let policy = LinearPolicy { weights, threshold };
        let trajectory =
            evaluate_policy_by_trajectory(validation, config.tail_quality_quantile, |row| {
                policy.refresh(row)
            });
        (threshold, metrics, trajectory)
    };

    Ok(DiscoveryResult {
        policy: LinearPolicy { weights, threshold },
        validation: validation_metrics,
        validation_trajectory,
        optimizer_fitness: best_fitness,
    })
}

pub fn compare_on_holdout(
    policy: &LinearPolicy,
    test: &[TraceRow],
    max_quality_loss: f64,
) -> HoldoutComparison {
    let learned = evaluate_policy(test, |row| policy.refresh(row));
    let fixed_gamma = best_fixed_gamma(test, max_quality_loss);
    let learned_meets_budget = learned.quality_loss_fraction <= max_quality_loss + 1e-12;
    let fixed_gamma_meets_budget =
        fixed_gamma.metrics.quality_loss_fraction <= max_quality_loss + 1e-12;
    let constrained_better = learned_meets_budget
        && fixed_gamma_meets_budget
        && learned.compute_fraction < fixed_gamma.metrics.compute_fraction;
    let relative_compute_improvement = if fixed_gamma.metrics.compute_fraction > 0.0
    {
        (fixed_gamma.metrics.compute_fraction - learned.compute_fraction)
            / fixed_gamma.metrics.compute_fraction
    }
    else
    {
        0.0
    };
    let pareto_dominates = learned.quality_loss_fraction
        <= fixed_gamma.metrics.quality_loss_fraction + 1e-12
        && learned.compute_fraction <= fixed_gamma.metrics.compute_fraction + 1e-12
        && (learned.quality_loss_fraction < fixed_gamma.metrics.quality_loss_fraction - 1e-12
            || learned.compute_fraction < fixed_gamma.metrics.compute_fraction - 1e-12);

    HoldoutComparison {
        quality_budget: max_quality_loss,
        learned,
        fixed_gamma,
        learned_meets_budget,
        fixed_gamma_meets_budget,
        constrained_better,
        relative_compute_improvement,
        pareto_dominates,
    }
}

pub fn compare_on_holdout_robust(
    policy: &LinearPolicy,
    test: &[TraceRow],
    max_quality_loss: f64,
    tail_quality_quantile: f64,
) -> RobustHoldoutComparison {
    let learned = evaluate_policy(test, |row| policy.refresh(row));
    let learned_trajectory =
        evaluate_policy_by_trajectory(test, tail_quality_quantile, |row| policy.refresh(row));
    let (fixed_gamma, fixed_gamma_trajectory) =
        best_fixed_gamma_robust(test, max_quality_loss, tail_quality_quantile);

    let learned_meets_budget = learned.quality_loss_fraction <= max_quality_loss + 1e-12
        && learned_trajectory.mean_quality_loss_fraction <= max_quality_loss + 1e-12
        && learned_trajectory.tail_quality_loss_fraction <= max_quality_loss + 1e-12;
    let fixed_gamma_meets_budget = fixed_gamma.metrics.quality_loss_fraction
        <= max_quality_loss + 1e-12
        && fixed_gamma_trajectory.mean_quality_loss_fraction <= max_quality_loss + 1e-12
        && fixed_gamma_trajectory.tail_quality_loss_fraction <= max_quality_loss + 1e-12;
    let constrained_better = learned_meets_budget
        && fixed_gamma_meets_budget
        && learned.compute_fraction < fixed_gamma.metrics.compute_fraction;
    let relative_compute_improvement = if fixed_gamma.metrics.compute_fraction > 0.0
    {
        (fixed_gamma.metrics.compute_fraction - learned.compute_fraction)
            / fixed_gamma.metrics.compute_fraction
    }
    else
    {
        0.0
    };
    let pareto_dominates = learned_meets_budget
        && fixed_gamma_meets_budget
        && learned.compute_fraction <= fixed_gamma.metrics.compute_fraction + 1e-12
        && learned_trajectory.mean_quality_loss_fraction
            <= fixed_gamma_trajectory.mean_quality_loss_fraction + 1e-12
        && learned_trajectory.tail_quality_loss_fraction
            <= fixed_gamma_trajectory.tail_quality_loss_fraction + 1e-12
        && (learned.compute_fraction < fixed_gamma.metrics.compute_fraction - 1e-12
            || learned_trajectory.mean_quality_loss_fraction
                < fixed_gamma_trajectory.mean_quality_loss_fraction - 1e-12
            || learned_trajectory.tail_quality_loss_fraction
                < fixed_gamma_trajectory.tail_quality_loss_fraction - 1e-12);

    RobustHoldoutComparison {
        quality_budget: max_quality_loss,
        tail_quality_quantile,
        learned,
        learned_trajectory,
        fixed_gamma,
        fixed_gamma_trajectory,
        learned_meets_budget,
        fixed_gamma_meets_budget,
        constrained_better,
        relative_compute_improvement,
        pareto_dominates,
    }
}

pub fn discover_symbolic_surrogate(
    rows: &[TraceRow],
    seeds: &[u64],
    population: usize,
    generations: usize,
    inner_iterations: usize,
    max_size: usize,
) -> Vec<SymbolicCandidate> {
    let data: Vec<(Vec<f64>, f64)> = rows
        .iter()
        .map(|row| {
            let target = row.stale_loss / row.refresh_cost.max(1e-6);
            (row.features().to_vec(), target)
        })
        .collect();
    scirust_symreg::discover(
        &data,
        &FEATURE_NAMES,
        seeds,
        population,
        generations,
        inner_iterations,
        max_size,
    )
    .into_iter()
    .map(|(size, mse, expression)| SymbolicCandidate {
        size,
        mse,
        expression: expression.to_string(),
    })
    .collect()
}

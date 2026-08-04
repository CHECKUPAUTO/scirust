use crate::model::{
    DiscoveryConfig, DiscoveryResult, FEATURE_NAMES, GammaBaseline, HoldoutComparison,
    LinearPolicy, PolicyMetrics, SymbolicCandidate, TraceRow, dot, evaluate_policy,
};
use scirust_evo::CmaEs;

pub fn best_fixed_gamma(rows: &[TraceRow], max_quality_loss: f64) -> GammaBaseline {
    let mut best: Option<GammaBaseline> = None;
    for i in 0..=2_000 {
        let gamma = i as f64 / 2_000.0;
        let metrics = evaluate_policy(rows, |row| row.similarity < gamma);
        if metrics.quality_loss_fraction <= max_quality_loss + 1e-12
            && best.as_ref().is_none_or(|current| {
                metrics.compute_fraction < current.metrics.compute_fraction
            })
        {
            best = Some(GammaBaseline { gamma, metrics });
        }
    }
    best.unwrap_or_else(|| GammaBaseline {
        gamma: f64::INFINITY,
        metrics: evaluate_policy(rows, |_| true),
    })
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn smooth_objective(rows: &[TraceRow], theta: &[f64], max_quality_loss: f64) -> f64 {
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

    for row in rows {
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
    let excess = (quality - max_quality_loss).max(0.0);
    let l2 = theta.iter().map(|value| value * value).sum::<f64>();
    compute + 500.0 * excess * excess + 0.0005 * l2
}

/// Cheapest hard threshold satisfying the quality budget for fixed risk weights.
pub fn calibrate_threshold(
    rows: &[TraceRow],
    weights: &[f64; 8],
    max_quality_loss: f64,
) -> Option<(f64, PolicyMetrics)> {
    if rows.is_empty() {
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
    if no_refresh.quality_loss_fraction <= max_quality_loss + 1e-12 {
        return Some((f64::INFINITY, no_refresh));
    }

    while cursor < ranked.len() {
        let threshold = ranked[cursor].0;
        let mut end = cursor;
        while end < ranked.len() && ranked[end].0.total_cmp(&threshold).is_eq() {
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
        if metrics.quality_loss_fraction <= max_quality_loss + 1e-12 {
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

pub fn discover_linear_policy(
    training: &[TraceRow],
    validation: &[TraceRow],
    config: DiscoveryConfig,
) -> Result<DiscoveryResult, String> {
    if training.is_empty() || validation.is_empty() {
        return Err("training and validation traces must both be non-empty".into());
    }
    if !(0.0..=1.0).contains(&config.max_quality_loss) {
        return Err("max_quality_loss must lie in [0,1]".into());
    }
    if config.steps == 0 {
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

    let mut optimizer = CmaEs::seeded(9, config.seed);
    optimizer.bounds = (-8.0, 8.0);
    optimizer.sigma = config.initial_sigma;
    let mut theta = vec![0.0; 9];
    let mut best_theta = theta.clone();
    let mut best_fitness = f64::NEG_INFINITY;

    for _ in 0..config.steps {
        let population = optimizer.step(&mut theta, |candidate| {
            -smooth_objective(training, candidate, config.max_quality_loss)
        });
        for individual in population {
            if individual.fitness.total_cmp(&best_fitness).is_gt() {
                best_fitness = individual.fitness;
                best_theta.clone_from(&individual.genome);
            }
        }
        optimizer.sigma = (optimizer.sigma * 0.995).max(config.minimum_sigma);
    }

    let mut weights = [0.0; 8];
    weights.copy_from_slice(&best_theta[..8]);
    let (threshold, validation_metrics) =
        calibrate_threshold(validation, &weights, config.max_quality_loss)
            .ok_or_else(|| "cannot calibrate an empty validation trace".to_string())?;

    Ok(DiscoveryResult {
        policy: LinearPolicy { weights, threshold },
        validation: validation_metrics,
        optimizer_fitness: best_fitness,
    })
}

pub fn compare_on_holdout(policy: &LinearPolicy, test: &[TraceRow]) -> HoldoutComparison {
    let learned = evaluate_policy(test, |row| policy.refresh(row));
    let fixed_gamma = best_fixed_gamma(test, learned.quality_loss_fraction);
    let relative_compute_improvement = if fixed_gamma.metrics.compute_fraction > 0.0 {
        (fixed_gamma.metrics.compute_fraction - learned.compute_fraction)
            / fixed_gamma.metrics.compute_fraction
    } else {
        0.0
    };
    let pareto_dominates = learned.quality_loss_fraction
        <= fixed_gamma.metrics.quality_loss_fraction + 1e-12
        && learned.compute_fraction <= fixed_gamma.metrics.compute_fraction + 1e-12
        && (learned.quality_loss_fraction < fixed_gamma.metrics.quality_loss_fraction - 1e-12
            || learned.compute_fraction < fixed_gamma.metrics.compute_fraction - 1e-12);

    HoldoutComparison {
        learned,
        fixed_gamma,
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

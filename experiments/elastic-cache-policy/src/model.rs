use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraceRow {
    pub trajectory_id: u64,
    pub step: u32,
    pub layer_id: u32,
    pub similarity: f64,
    pub similarity_delta: f64,
    pub head_variance: f64,
    pub cache_age: f64,
    pub attention_mass: f64,
    pub layer_fraction: f64,
    pub refresh_cost: f64,
    pub stale_loss: f64,
}

pub type TraceSplit = (Vec<TraceRow>, Vec<TraceRow>, Vec<TraceRow>);

pub const FEATURE_NAMES: [&str; 8] = [
    "drift",
    "worsening",
    "head_std",
    "cache_age",
    "untracked_mass",
    "layer_fraction",
    "drift_age",
    "refresh_cost",
];

impl TraceRow {
    pub fn validate(&self) -> Result<(), String> {
        let bounded = [
            ("similarity", self.similarity),
            ("head_variance", self.head_variance),
            ("cache_age", self.cache_age),
            ("attention_mass", self.attention_mass),
            ("layer_fraction", self.layer_fraction),
            ("refresh_cost", self.refresh_cost),
            ("stale_loss", self.stale_loss),
        ];
        for (name, value) in bounded
        {
            if !value.is_finite() || !(0.0..=1.0).contains(&value)
            {
                return Err(format!("{name} must be finite and in [0,1], got {value}"));
            }
        }
        if !self.similarity_delta.is_finite() || !(-1.0..=1.0).contains(&self.similarity_delta)
        {
            return Err(format!(
                "similarity_delta must be finite and in [-1,1], got {}",
                self.similarity_delta
            ));
        }
        Ok(())
    }

    #[inline]
    pub fn features(&self) -> [f64; 8] {
        let drift = 1.0 - self.similarity;
        let worsening = (-self.similarity_delta).max(0.0);
        [
            drift,
            worsening,
            self.head_variance.sqrt(),
            self.cache_age,
            1.0 - self.attention_mass,
            self.layer_fraction,
            drift * self.cache_age,
            self.refresh_cost,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolicyMetrics {
    /// Fraction of total counterfactual stale loss left uncorrected.
    pub quality_loss_fraction: f64,
    /// Fraction of full-refresh-equivalent compute spent by the policy.
    pub compute_fraction: f64,
    pub refresh_rate: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectoryPolicyMetrics {
    pub trajectories: usize,
    pub mean_quality_loss_fraction: f64,
    pub tail_quality_loss_fraction: f64,
    pub worst_quality_loss_fraction: f64,
    pub mean_compute_fraction: f64,
    pub mean_refresh_rate: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearPolicy {
    pub weights: [f64; 8],
    pub threshold: f64,
}

impl LinearPolicy {
    #[inline]
    pub fn risk(&self, row: &TraceRow) -> f64 {
        dot(&self.weights, &row.features())
    }

    #[inline]
    pub fn refresh(&self, row: &TraceRow) -> bool {
        self.risk(row) >= self.threshold
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DiscoveryConfig {
    pub seed: u64,
    pub steps: usize,
    pub max_quality_loss: f64,
    /// Fraction of the final quality budget used to calibrate the validation threshold.
    pub calibration_budget_fraction: f64,
    /// Equalize trajectory influence and constrain a high quantile of per-trajectory loss.
    pub trajectory_balanced: bool,
    /// Nearest-rank quantile used by robust training, calibration, and holdout reporting.
    pub tail_quality_quantile: f64,
    /// Multiplier applied to the tail-loss penalty in the smooth CMA-ES objective.
    pub tail_penalty_weight: f64,
    pub initial_sigma: f64,
    pub minimum_sigma: f64,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            seed: 20_260_804,
            steps: 600,
            max_quality_loss: 0.05,
            calibration_budget_fraction: 0.8,
            trajectory_balanced: false,
            tail_quality_quantile: 0.9,
            tail_penalty_weight: 4.0,
            initial_sigma: 0.8,
            minimum_sigma: 0.05,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiscoveryResult {
    pub policy: LinearPolicy,
    pub validation: PolicyMetrics,
    pub validation_trajectory: TrajectoryPolicyMetrics,
    pub optimizer_fitness: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GammaBaseline {
    pub gamma: f64,
    pub metrics: PolicyMetrics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HoldoutComparison {
    pub quality_budget: f64,
    pub learned: PolicyMetrics,
    pub fixed_gamma: GammaBaseline,
    pub learned_meets_budget: bool,
    pub fixed_gamma_meets_budget: bool,
    pub constrained_better: bool,
    pub relative_compute_improvement: f64,
    pub pareto_dominates: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustHoldoutComparison {
    pub quality_budget: f64,
    pub tail_quality_quantile: f64,
    pub learned: PolicyMetrics,
    pub learned_trajectory: TrajectoryPolicyMetrics,
    pub fixed_gamma: GammaBaseline,
    pub fixed_gamma_trajectory: TrajectoryPolicyMetrics,
    pub learned_meets_budget: bool,
    pub fixed_gamma_meets_budget: bool,
    pub constrained_better: bool,
    pub relative_compute_improvement: f64,
    pub pareto_dominates: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SymbolicCandidate {
    pub size: usize,
    pub mse: f64,
    pub expression: String,
}

#[inline]
pub(crate) fn dot(a: &[f64; 8], b: &[f64; 8]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(crate) fn nearest_rank_quantile(values: &mut [f64], quantile: f64) -> f64 {
    if values.is_empty()
    {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let bounded = quantile.clamp(0.0, 1.0);
    let rank = (bounded * values.len() as f64).ceil() as usize;
    values[rank.saturating_sub(1).min(values.len() - 1)]
}

pub fn evaluate_policy<F>(rows: &[TraceRow], refresh: F) -> PolicyMetrics
where
    F: Fn(&TraceRow) -> bool,
{
    if rows.is_empty()
    {
        return PolicyMetrics {
            quality_loss_fraction: 0.0,
            compute_fraction: 0.0,
            refresh_rate: 0.0,
        };
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
    let mut incurred_loss = 0.0;
    let mut refresh_cost = 0.0;
    let mut refreshes = 0usize;

    for row in rows
    {
        if refresh(row)
        {
            refresh_cost += row.refresh_cost;
            refreshes += 1;
        }
        else
        {
            incurred_loss += row.stale_loss;
        }
    }

    PolicyMetrics {
        quality_loss_fraction: incurred_loss / total_loss,
        compute_fraction: refresh_cost / total_cost,
        refresh_rate: refreshes as f64 / rows.len() as f64,
    }
}

pub fn evaluate_policy_by_trajectory<F>(
    rows: &[TraceRow],
    tail_quality_quantile: f64,
    refresh: F,
) -> TrajectoryPolicyMetrics
where
    F: Fn(&TraceRow) -> bool,
{
    #[derive(Default)]
    struct Accumulator {
        total_loss: f64,
        total_cost: f64,
        incurred_loss: f64,
        refresh_cost: f64,
        rows: usize,
        refreshes: usize,
    }

    if rows.is_empty()
    {
        return TrajectoryPolicyMetrics {
            trajectories: 0,
            mean_quality_loss_fraction: 0.0,
            tail_quality_loss_fraction: 0.0,
            worst_quality_loss_fraction: 0.0,
            mean_compute_fraction: 0.0,
            mean_refresh_rate: 0.0,
        };
    }

    let mut by_trajectory: BTreeMap<u64, Accumulator> = BTreeMap::new();
    for row in rows
    {
        let accumulator = by_trajectory.entry(row.trajectory_id).or_default();
        accumulator.total_loss += row.stale_loss;
        accumulator.total_cost += row.refresh_cost;
        accumulator.rows += 1;
        if refresh(row)
        {
            accumulator.refresh_cost += row.refresh_cost;
            accumulator.refreshes += 1;
        }
        else
        {
            accumulator.incurred_loss += row.stale_loss;
        }
    }

    let mut qualities = Vec::with_capacity(by_trajectory.len());
    let mut compute_sum = 0.0;
    let mut refresh_rate_sum = 0.0;
    for accumulator in by_trajectory.values()
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
}

pub fn split_by_trajectory_fold(
    rows: &[TraceRow],
    folds: u64,
    test_fold: u64,
) -> Result<TraceSplit, String> {
    if folds < 3
    {
        return Err("folds must be at least 3".into());
    }
    if test_fold >= folds
    {
        return Err(format!(
            "test_fold must be smaller than folds, got test_fold={test_fold} folds={folds}"
        ));
    }

    let validation_fold = (test_fold + folds - 1) % folds;
    let mut training = Vec::new();
    let mut validation = Vec::new();
    let mut test = Vec::new();
    for row in rows
    {
        let fold = row.trajectory_id % folds;
        if fold == test_fold
        {
            test.push(*row);
        }
        else if fold == validation_fold
        {
            validation.push(*row);
        }
        else
        {
            training.push(*row);
        }
    }
    Ok((training, validation, test))
}

pub fn split_by_trajectory(rows: &[TraceRow]) -> TraceSplit {
    split_by_trajectory_fold(rows, 5, 4).expect("the default five-fold split is valid")
}

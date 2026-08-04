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
        for (name, value) in bounded {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(format!("{name} must be finite and in [0,1], got {value}"));
            }
        }
        if !self.similarity_delta.is_finite()
            || !(-1.0..=1.0).contains(&self.similarity_delta)
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
    pub initial_sigma: f64,
    pub minimum_sigma: f64,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            seed: 20_260_804,
            steps: 600,
            max_quality_loss: 0.05,
            initial_sigma: 0.8,
            minimum_sigma: 0.05,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiscoveryResult {
    pub policy: LinearPolicy,
    pub validation: PolicyMetrics,
    pub optimizer_fitness: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GammaBaseline {
    pub gamma: f64,
    pub metrics: PolicyMetrics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HoldoutComparison {
    pub learned: PolicyMetrics,
    pub fixed_gamma: GammaBaseline,
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

pub fn evaluate_policy<F>(rows: &[TraceRow], refresh: F) -> PolicyMetrics
where
    F: Fn(&TraceRow) -> bool,
{
    if rows.is_empty() {
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

    for row in rows {
        if refresh(row) {
            refresh_cost += row.refresh_cost;
            refreshes += 1;
        } else {
            incurred_loss += row.stale_loss;
        }
    }

    PolicyMetrics {
        quality_loss_fraction: incurred_loss / total_loss,
        compute_fraction: refresh_cost / total_cost,
        refresh_rate: refreshes as f64 / rows.len() as f64,
    }
}

pub fn split_by_trajectory(
    rows: &[TraceRow],
) -> (Vec<TraceRow>, Vec<TraceRow>, Vec<TraceRow>) {
    let mut training = Vec::new();
    let mut validation = Vec::new();
    let mut test = Vec::new();
    for row in rows {
        match row.trajectory_id % 5 {
            0..=2 => training.push(*row),
            3 => validation.push(*row),
            _ => test.push(*row),
        }
    }
    (training, validation, test)
}

//! Phase 3 strict-budget planning for the Elastic Latent KV experiment.
//!
//! Phase 3 enumerates deterministic key/value rank pairs, accounts for
//! persistent coefficient and basis storage, evaluates reconstruction and
//! attention quality, builds a Pareto frontier and selects one plan under a
//! strict byte budget. It deliberately does not introduce per-token ranks,
//! quantization, residual channels or production integration.

use crate::phase2::{
    OrthonormalBasis, ProjectedAttentionInput, ProjectedAttentionMetrics, ProjectionError,
    ReconstructionMetrics, evaluate_projected_attention, reconstruction_metrics,
};
use core::{cmp::Ordering, fmt};

const F32_BYTES: u64 = 4;
/// Errors returned by deterministic Phase 3 budget planning.
#[derive(Debug, Clone, PartialEq)]
pub enum BudgetError {
    /// A required count, dimension or rank was zero.
    ZeroField {
        /// Name of the zero-valued field.
        field: &'static str,
    },
    /// A flat buffer has an unexpected number of elements.
    InvalidBufferLength {
        /// Human-readable buffer name.
        name: &'static str,
        /// Required number of elements.
        expected: usize,
        /// Supplied number of elements.
        actual: usize,
    },
    /// A rank exceeds the dense dimension or token count.
    InvalidRank {
        /// Requested rank.
        rank: usize,
        /// Maximum accepted rank.
        maximum: usize,
    },
    /// A numeric target is non-finite or outside its accepted range.
    InvalidThreshold {
        /// Human-readable target name.
        name: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// The strict budget cannot store even the rank-one/rank-one candidate.
    BudgetBelowMinimum {
        /// Supplied persistent-storage budget.
        budget_bytes: u64,
        /// Minimum bytes required by rank-one key and value bases.
        minimum_bytes: u64,
    },
    /// An integer accounting operation overflowed.
    ArithmeticOverflow,
    /// A Phase 2 projection operation failed.
    Projection(ProjectionError),
}

impl fmt::Display for BudgetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::ZeroField { field } => write!(formatter, "{field} must be non-zero"),
            Self::InvalidBufferLength {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "{name} length mismatch: expected {expected}, received {actual}"
            ),
            Self::InvalidRank { rank, maximum } =>
            {
                write!(formatter, "rank {rank} exceeds maximum {maximum}")
            },
            Self::InvalidThreshold { name, value } => write!(
                formatter,
                "{name} must be finite and inside its accepted range, received {value}"
            ),
            Self::BudgetBelowMinimum {
                budget_bytes,
                minimum_bytes,
            } => write!(
                formatter,
                "persistent budget {budget_bytes} is below minimum {minimum_bytes}"
            ),
            Self::ArithmeticOverflow => write!(formatter, "budget arithmetic overflow"),
            Self::Projection(error) => write!(formatter, "projection error: {error}"),
        }
    }
}

impl std::error::Error for BudgetError {}

impl From<ProjectionError> for BudgetError {
    fn from(error: ProjectionError) -> Self {
        Self::Projection(error)
    }
}

/// Closed-form persistent-storage accounting for one key/value rank pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StorageAccounting {
    /// Dense key/value payload bytes for the same token count and dimension.
    pub dense_bytes: u64,
    /// Latent per-token key/value coefficient bytes.
    pub coefficient_bytes: u64,
    /// Shared dense-to-latent key/value basis bytes.
    pub basis_bytes: u64,
    /// Total persistent latent bytes: coefficients plus bases.
    pub total_bytes: u64,
    /// Dense bytes not consumed by the latent representation.
    pub savings_bytes: u64,
    /// Dense bytes divided by latent bytes.
    pub compression_ratio: f64,
}

/// Computes exact persistent-storage bytes for one fixed-rank representation.
pub fn storage_accounting(
    token_count: usize,
    dimension: usize,
    key_rank: usize,
    value_rank: usize,
) -> Result<StorageAccounting, BudgetError> {
    require_non_zero("token_count", token_count)?;
    require_non_zero("dimension", dimension)?;
    require_non_zero("key_rank", key_rank)?;
    require_non_zero("value_rank", value_rank)?;

    for rank in [key_rank, value_rank]
    {
        if rank > dimension
        {
            return Err(BudgetError::InvalidRank {
                rank,
                maximum: dimension,
            });
        }
    }

    let tokens = to_u64(token_count)?;
    let dense_dimension = to_u64(dimension)?;
    let rank_sum = to_u64(
        key_rank
            .checked_add(value_rank)
            .ok_or(BudgetError::ArithmeticOverflow)?,
    )?;

    let dense_scalars = checked_mul(checked_mul(tokens, dense_dimension)?, 2)?;
    let coefficient_scalars = checked_mul(tokens, rank_sum)?;
    let basis_scalars = checked_mul(dense_dimension, rank_sum)?;

    let dense_bytes = checked_mul(dense_scalars, F32_BYTES)?;
    let coefficient_bytes = checked_mul(coefficient_scalars, F32_BYTES)?;
    let basis_bytes = checked_mul(basis_scalars, F32_BYTES)?;
    let total_bytes = coefficient_bytes
        .checked_add(basis_bytes)
        .ok_or(BudgetError::ArithmeticOverflow)?;

    Ok(StorageAccounting {
        dense_bytes,
        coefficient_bytes,
        basis_bytes,
        total_bytes,
        savings_bytes: dense_bytes.saturating_sub(total_bytes),
        compression_ratio: dense_bytes as f64 / total_bytes as f64,
    })
}

/// One fully evaluated rank-pair candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetCandidate {
    /// Key basis rank.
    pub key_rank: usize,
    /// Value basis rank.
    pub value_rank: usize,
    /// Persistent-storage accounting.
    pub storage: StorageAccounting,
    /// Key reconstruction quality.
    pub key_reconstruction: ReconstructionMetrics,
    /// Value reconstruction quality.
    pub value_reconstruction: ReconstructionMetrics,
    /// Dense-versus-latent attention quality.
    pub attention: ProjectedAttentionMetrics,
    /// Maximum of the three error-to-target ratios.
    pub worst_target_ratio: f64,
    /// Whether all declared reconstruction and attention targets are met.
    pub quality_guard_met: bool,
}

/// Complete result of strict-budget deterministic rank planning.
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetPlan {
    /// Strict persistent-storage budget supplied to the planner.
    pub budget_bytes: u64,
    /// Bytes required by the minimum rank-one/rank-one candidate.
    pub minimum_candidate_bytes: u64,
    /// Total rank pairs evaluated before budget filtering.
    pub evaluated_pairs: usize,
    /// Rank pairs whose persistent storage fits the strict budget.
    pub budget_feasible_pairs: usize,
    /// Budget-feasible pairs satisfying every declared quality target.
    pub quality_feasible_pairs: usize,
    /// Deterministically selected candidate.
    pub selected: BudgetCandidate,
    /// Budget-feasible non-dominated candidates sorted by storage and ranks.
    pub pareto_frontier: Vec<BudgetCandidate>,
}

/// Borrowed inputs for one strict-budget planning operation.
#[derive(Debug, Clone, Copy)]
pub struct BudgetPlannerInput<'a> {
    /// Dense keys in row-major `[token_count, dimension]` order.
    pub keys: &'a [f32],
    /// Dense values in row-major `[token_count, dimension]` order.
    pub values: &'a [f32],
    /// Number of key/value tokens.
    pub token_count: usize,
    /// Dense queries in row-major `[query_count, dimension]` order.
    pub queries: &'a [f32],
    /// Number of attention queries.
    pub query_count: usize,
    /// Dense head dimension.
    pub dimension: usize,
    /// Largest explored key rank.
    pub maximum_key_rank: usize,
    /// Largest explored value rank.
    pub maximum_value_rank: usize,
    /// Strict persistent-storage budget in bytes.
    pub budget_bytes: u64,
    /// Maximum key reconstruction relative RMS.
    pub key_target_relative_root_mean_square: f64,
    /// Maximum value reconstruction relative RMS.
    pub value_target_relative_root_mean_square: f64,
    /// Maximum dense-versus-latent attention absolute error.
    pub attention_target_max_absolute: f64,
    /// Positive basis-construction norm tolerance.
    pub norm_tolerance: f64,
    /// Positive finite scaled-dot-product attention scale.
    pub scale: f32,
}

/// Enumerates rank pairs, builds the Pareto frontier and selects one plan.
pub fn plan_under_budget(input: BudgetPlannerInput<'_>) -> Result<BudgetPlan, BudgetError> {
    validate_planner_input(&input)?;

    let maximum_key_basis = OrthonormalBasis::from_greedy_samples(
        input.keys,
        input.token_count,
        input.dimension,
        input.maximum_key_rank,
        input.norm_tolerance,
    )?;
    let maximum_value_basis = OrthonormalBasis::from_greedy_samples(
        input.values,
        input.token_count,
        input.dimension,
        input.maximum_value_rank,
        input.norm_tolerance,
    )?;

    let available_key_rank = maximum_key_basis.rank();
    let available_value_rank = maximum_value_basis.rank();
    let minimum_candidate_bytes =
        storage_accounting(input.token_count, input.dimension, 1, 1)?.total_bytes;

    if input.budget_bytes < minimum_candidate_bytes
    {
        return Err(BudgetError::BudgetBelowMinimum {
            budget_bytes: input.budget_bytes,
            minimum_bytes: minimum_candidate_bytes,
        });
    }

    let key_prefixes = prefix_metrics(
        input.keys,
        input.token_count,
        &maximum_key_basis,
        available_key_rank,
    )?;
    let value_prefixes = prefix_metrics(
        input.values,
        input.token_count,
        &maximum_value_basis,
        available_value_rank,
    )?;

    let evaluated_pairs = available_key_rank
        .checked_mul(available_value_rank)
        .ok_or(BudgetError::ArithmeticOverflow)?;
    let mut candidates = Vec::with_capacity(evaluated_pairs);

    for (key_index, (key_basis, key_metrics)) in key_prefixes.iter().enumerate()
    {
        let key_rank = key_index + 1;

        for (value_index, (value_basis, value_metrics)) in value_prefixes.iter().enumerate()
        {
            let value_rank = value_index + 1;
            let storage =
                storage_accounting(input.token_count, input.dimension, key_rank, value_rank)?;

            if storage.total_bytes > input.budget_bytes
            {
                continue;
            }

            let attention = evaluate_projected_attention(ProjectedAttentionInput {
                keys: input.keys,
                values: input.values,
                token_count: input.token_count,
                queries: input.queries,
                query_count: input.query_count,
                key_basis,
                value_basis,
                scale: input.scale,
            })?;

            let key_ratio = target_ratio(
                key_metrics.relative_root_mean_square,
                input.key_target_relative_root_mean_square,
            );
            let value_ratio = target_ratio(
                value_metrics.relative_root_mean_square,
                input.value_target_relative_root_mean_square,
            );
            let attention_ratio =
                target_ratio(attention.max_absolute, input.attention_target_max_absolute);
            let worst_target_ratio = key_ratio.max(value_ratio).max(attention_ratio);
            let quality_guard_met =
                key_ratio <= 1.0 && value_ratio <= 1.0 && attention_ratio <= 1.0;

            candidates.push(BudgetCandidate {
                key_rank,
                value_rank,
                storage,
                key_reconstruction: key_metrics.clone(),
                value_reconstruction: value_metrics.clone(),
                attention,
                worst_target_ratio,
                quality_guard_met,
            });
        }
    }

    let selected = candidates
        .iter()
        .min_by(|left, right| compare_candidates(left, right))
        .cloned()
        .ok_or(BudgetError::BudgetBelowMinimum {
            budget_bytes: input.budget_bytes,
            minimum_bytes: minimum_candidate_bytes,
        })?;
    let quality_feasible_pairs = candidates
        .iter()
        .filter(|candidate| candidate.quality_guard_met)
        .count();
    let pareto_frontier = pareto_frontier(&candidates);

    Ok(BudgetPlan {
        budget_bytes: input.budget_bytes,
        minimum_candidate_bytes,
        evaluated_pairs,
        budget_feasible_pairs: candidates.len(),
        quality_feasible_pairs,
        selected,
        pareto_frontier,
    })
}

fn prefix_metrics(
    samples: &[f32],
    sample_count: usize,
    maximum_basis: &OrthonormalBasis,
    available_rank: usize,
) -> Result<Vec<(OrthonormalBasis, ReconstructionMetrics)>, BudgetError> {
    let mut prefixes = Vec::with_capacity(available_rank);

    for rank in 1..=available_rank
    {
        let basis = maximum_basis.prefix(rank)?;
        let metrics = reconstruction_metrics(samples, sample_count, &basis)?;
        prefixes.push((basis, metrics));
    }

    Ok(prefixes)
}

fn compare_candidates(left: &BudgetCandidate, right: &BudgetCandidate) -> Ordering {
    if left.quality_guard_met != right.quality_guard_met
    {
        return if left.quality_guard_met
        {
            Ordering::Less
        }
        else
        {
            Ordering::Greater
        };
    }

    if left.quality_guard_met
    {
        left.storage
            .total_bytes
            .cmp(&right.storage.total_bytes)
            .then_with(|| {
                left.attention
                    .max_absolute
                    .total_cmp(&right.attention.max_absolute)
            })
            .then_with(|| {
                candidate_reconstruction_max(left).total_cmp(&candidate_reconstruction_max(right))
            })
            .then_with(|| left.key_rank.cmp(&right.key_rank))
            .then_with(|| left.value_rank.cmp(&right.value_rank))
    }
    else
    {
        left.worst_target_ratio
            .total_cmp(&right.worst_target_ratio)
            .then_with(|| {
                left.attention
                    .max_absolute
                    .total_cmp(&right.attention.max_absolute)
            })
            .then_with(|| {
                candidate_reconstruction_max(left).total_cmp(&candidate_reconstruction_max(right))
            })
            .then_with(|| left.storage.total_bytes.cmp(&right.storage.total_bytes))
            .then_with(|| left.key_rank.cmp(&right.key_rank))
            .then_with(|| left.value_rank.cmp(&right.value_rank))
    }
}

fn candidate_reconstruction_max(candidate: &BudgetCandidate) -> f64 {
    candidate
        .key_reconstruction
        .relative_root_mean_square
        .max(candidate.value_reconstruction.relative_root_mean_square)
}

fn pareto_frontier(candidates: &[BudgetCandidate]) -> Vec<BudgetCandidate> {
    let mut frontier = Vec::new();

    for (candidate_index, candidate) in candidates.iter().enumerate()
    {
        let dominated = candidates.iter().enumerate().any(|(other_index, other)| {
            candidate_index != other_index && dominates(other, candidate)
        });

        if !dominated
        {
            frontier.push(candidate.clone());
        }
    }

    frontier.sort_by(|left, right| {
        left.storage
            .total_bytes
            .cmp(&right.storage.total_bytes)
            .then_with(|| left.key_rank.cmp(&right.key_rank))
            .then_with(|| left.value_rank.cmp(&right.value_rank))
    });
    frontier
}

fn dominates(left: &BudgetCandidate, right: &BudgetCandidate) -> bool {
    let no_worse = left.storage.total_bytes <= right.storage.total_bytes
        && left.key_reconstruction.relative_root_mean_square
            <= right.key_reconstruction.relative_root_mean_square
        && left.value_reconstruction.relative_root_mean_square
            <= right.value_reconstruction.relative_root_mean_square
        && left.attention.max_absolute <= right.attention.max_absolute;
    let strictly_better = left.storage.total_bytes < right.storage.total_bytes
        || left.key_reconstruction.relative_root_mean_square
            < right.key_reconstruction.relative_root_mean_square
        || left.value_reconstruction.relative_root_mean_square
            < right.value_reconstruction.relative_root_mean_square
        || left.attention.max_absolute < right.attention.max_absolute;

    no_worse && strictly_better
}

fn validate_planner_input(input: &BudgetPlannerInput<'_>) -> Result<(), BudgetError> {
    for (field, value) in [
        ("token_count", input.token_count),
        ("query_count", input.query_count),
        ("dimension", input.dimension),
        ("maximum_key_rank", input.maximum_key_rank),
        ("maximum_value_rank", input.maximum_value_rank),
    ]
    {
        require_non_zero(field, value)?;
    }

    for rank in [input.maximum_key_rank, input.maximum_value_rank]
    {
        let maximum = input.dimension.min(input.token_count);
        if rank > maximum
        {
            return Err(BudgetError::InvalidRank { rank, maximum });
        }
    }

    let expected_samples = checked_usize_mul(input.token_count, input.dimension)?;
    let expected_queries = checked_usize_mul(input.query_count, input.dimension)?;
    require_buffer_length("keys", input.keys, expected_samples)?;
    require_buffer_length("values", input.values, expected_samples)?;
    require_buffer_length("queries", input.queries, expected_queries)?;

    for (name, value) in [
        (
            "key_target_relative_root_mean_square",
            input.key_target_relative_root_mean_square,
        ),
        (
            "value_target_relative_root_mean_square",
            input.value_target_relative_root_mean_square,
        ),
        (
            "attention_target_max_absolute",
            input.attention_target_max_absolute,
        ),
    ]
    {
        if !value.is_finite() || value < 0.0
        {
            return Err(BudgetError::InvalidThreshold { name, value });
        }
    }

    if !input.norm_tolerance.is_finite() || input.norm_tolerance <= 0.0
    {
        return Err(BudgetError::InvalidThreshold {
            name: "norm_tolerance",
            value: input.norm_tolerance,
        });
    }

    if !input.scale.is_finite() || input.scale <= 0.0
    {
        return Err(BudgetError::InvalidThreshold {
            name: "scale",
            value: f64::from(input.scale),
        });
    }

    Ok(())
}

fn target_ratio(error: f64, target: f64) -> f64 {
    if target == 0.0
    {
        if error == 0.0 { 0.0 } else { f64::INFINITY }
    }
    else
    {
        error / target
    }
}

/// Deterministic Phase 3 budget scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetScenario {
    /// Number of dense key/value tokens.
    pub token_count: usize,
    /// Dense head dimension.
    pub head_dimension: usize,
    /// Number of attention queries.
    pub query_count: usize,
    /// Intrinsic generated key rank.
    pub intrinsic_key_rank: usize,
    /// Intrinsic generated value rank.
    pub intrinsic_value_rank: usize,
    /// Largest explored key rank.
    pub maximum_key_rank: usize,
    /// Largest explored value rank.
    pub maximum_value_rank: usize,
    /// Budget numerator applied to dense persistent bytes.
    pub budget_numerator: u64,
    /// Budget denominator applied to dense persistent bytes.
    pub budget_denominator: u64,
    /// Maximum key reconstruction relative RMS.
    pub key_target_relative_root_mean_square: f64,
    /// Maximum value reconstruction relative RMS.
    pub value_target_relative_root_mean_square: f64,
    /// Maximum attention absolute error.
    pub attention_target_max_absolute: f64,
    /// Additive dense noise amplitude.
    pub noise_amplitude: f32,
    /// Signal coefficient amplitude.
    pub signal_amplitude: f32,
    /// Deterministic seed.
    pub seed: u64,
}

impl BudgetScenario {
    /// Validates structural and numeric scenario constraints.
    pub fn validate(&self) -> Result<(), BudgetError> {
        for (field, value) in [
            ("token_count", self.token_count),
            ("head_dimension", self.head_dimension),
            ("query_count", self.query_count),
            ("intrinsic_key_rank", self.intrinsic_key_rank),
            ("intrinsic_value_rank", self.intrinsic_value_rank),
            ("maximum_key_rank", self.maximum_key_rank),
            ("maximum_value_rank", self.maximum_value_rank),
        ]
        {
            require_non_zero(field, value)?;
        }

        if self.budget_numerator == 0
        {
            return Err(BudgetError::ZeroField {
                field: "budget_numerator",
            });
        }
        if self.budget_denominator == 0
        {
            return Err(BudgetError::ZeroField {
                field: "budget_denominator",
            });
        }

        let maximum = self.head_dimension.min(self.token_count);
        for rank in [
            self.intrinsic_key_rank,
            self.intrinsic_value_rank,
            self.maximum_key_rank,
            self.maximum_value_rank,
        ]
        {
            if rank > maximum
            {
                return Err(BudgetError::InvalidRank { rank, maximum });
            }
        }

        if self.intrinsic_key_rank > self.maximum_key_rank
        {
            return Err(BudgetError::InvalidRank {
                rank: self.intrinsic_key_rank,
                maximum: self.maximum_key_rank,
            });
        }
        if self.intrinsic_value_rank > self.maximum_value_rank
        {
            return Err(BudgetError::InvalidRank {
                rank: self.intrinsic_value_rank,
                maximum: self.maximum_value_rank,
            });
        }

        for (name, value) in [
            (
                "key_target_relative_root_mean_square",
                self.key_target_relative_root_mean_square,
            ),
            (
                "value_target_relative_root_mean_square",
                self.value_target_relative_root_mean_square,
            ),
            (
                "attention_target_max_absolute",
                self.attention_target_max_absolute,
            ),
        ]
        {
            if !value.is_finite() || value < 0.0
            {
                return Err(BudgetError::InvalidThreshold { name, value });
            }
        }

        if !self.noise_amplitude.is_finite() || self.noise_amplitude < 0.0
        {
            return Err(BudgetError::InvalidThreshold {
                name: "noise_amplitude",
                value: f64::from(self.noise_amplitude),
            });
        }
        if !self.signal_amplitude.is_finite() || self.signal_amplitude <= 0.0
        {
            return Err(BudgetError::InvalidThreshold {
                name: "signal_amplitude",
                value: f64::from(self.signal_amplitude),
            });
        }

        Ok(())
    }
}

/// Complete report for one deterministic Phase 3 scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct BudgetScenarioReport {
    /// Original scenario configuration.
    pub scenario: BudgetScenario,
    /// Strict-budget planner result.
    pub plan: BudgetPlan,
}

impl BudgetScenarioReport {
    /// Serializes one stable CSV row.
    #[must_use]
    pub fn to_csv_row(&self) -> String {
        let selected = &self.plan.selected;
        format!(
            concat!(
                "{},{},{},{},{},{},{},{},{},{},{},{:.9e},{:.9e},{:.9e},",
                "{:.9e},{:.9e},{},{},{},{},{:.9e},{},{:.9e},{:.9e},{:.9e},",
                "{:.9e},{},{},{},{},{:016x}"
            ),
            self.scenario.seed,
            self.scenario.token_count,
            self.scenario.head_dimension,
            self.scenario.query_count,
            self.scenario.intrinsic_key_rank,
            self.scenario.intrinsic_value_rank,
            self.scenario.maximum_key_rank,
            self.scenario.maximum_value_rank,
            self.scenario.budget_numerator,
            self.scenario.budget_denominator,
            self.plan.budget_bytes,
            self.scenario.noise_amplitude,
            self.scenario.key_target_relative_root_mean_square,
            self.scenario.value_target_relative_root_mean_square,
            self.scenario.attention_target_max_absolute,
            self.scenario.signal_amplitude,
            selected.key_rank,
            selected.value_rank,
            selected.storage.dense_bytes,
            selected.storage.total_bytes,
            selected.storage.compression_ratio,
            u8::from(selected.quality_guard_met),
            selected.worst_target_ratio,
            selected.key_reconstruction.relative_root_mean_square,
            selected.value_reconstruction.relative_root_mean_square,
            selected.attention.max_absolute,
            self.plan.evaluated_pairs,
            self.plan.budget_feasible_pairs,
            self.plan.quality_feasible_pairs,
            self.plan.pareto_frontier.len(),
            selected.attention.output_fingerprint,
        )
    }
}

/// Stable CSV header for Phase 3 reports.
pub const CSV_HEADER: &str = concat!(
    "seed,token_count,head_dimension,query_count,intrinsic_key_rank,",
    "intrinsic_value_rank,maximum_key_rank,maximum_value_rank,budget_numerator,",
    "budget_denominator,budget_bytes,noise_amplitude,key_target_relative_rms,",
    "value_target_relative_rms,attention_target_max_absolute,signal_amplitude,",
    "selected_key_rank,selected_value_rank,dense_bytes,selected_total_bytes,",
    "compression_ratio,quality_guard_met,worst_target_ratio,",
    "key_reconstruction_relative_rms,value_reconstruction_relative_rms,",
    "attention_max_absolute,evaluated_pairs,budget_feasible_pairs,",
    "quality_feasible_pairs,pareto_frontier_length,output_fingerprint"
);

/// Runs one deterministic Phase 3 budget scenario.
pub fn run_budget_scenario(scenario: &BudgetScenario) -> Result<BudgetScenarioReport, BudgetError> {
    scenario.validate()?;

    let mut rng = DeterministicRng::new(scenario.seed);
    let key_generator = random_basis(
        &mut rng,
        scenario.head_dimension,
        scenario.intrinsic_key_rank,
    )?;
    let value_generator = random_basis(
        &mut rng,
        scenario.head_dimension,
        scenario.intrinsic_value_rank,
    )?;
    let keys = generate_dataset(
        &mut rng,
        scenario.token_count,
        &key_generator,
        scenario.signal_amplitude,
        scenario.noise_amplitude,
    )?;
    let values = generate_dataset(
        &mut rng,
        scenario.token_count,
        &value_generator,
        scenario.signal_amplitude,
        scenario.noise_amplitude,
    )?;
    let queries = random_vector(
        &mut rng,
        checked_usize_mul(scenario.query_count, scenario.head_dimension)?,
        scenario.signal_amplitude,
    );

    let dense_bytes =
        storage_accounting(scenario.token_count, scenario.head_dimension, 1, 1)?.dense_bytes;
    let budget_bytes = dense_bytes
        .checked_mul(scenario.budget_numerator)
        .ok_or(BudgetError::ArithmeticOverflow)?
        / scenario.budget_denominator;
    let scale = (scenario.head_dimension as f32).sqrt().recip();

    let plan = plan_under_budget(BudgetPlannerInput {
        keys: &keys,
        values: &values,
        token_count: scenario.token_count,
        queries: &queries,
        query_count: scenario.query_count,
        dimension: scenario.head_dimension,
        maximum_key_rank: scenario.maximum_key_rank,
        maximum_value_rank: scenario.maximum_value_rank,
        budget_bytes,
        key_target_relative_root_mean_square: scenario.key_target_relative_root_mean_square,
        value_target_relative_root_mean_square: scenario.value_target_relative_root_mean_square,
        attention_target_max_absolute: scenario.attention_target_max_absolute,
        norm_tolerance: 1.0e-10,
        scale,
    })?;

    Ok(BudgetScenarioReport {
        scenario: scenario.clone(),
        plan,
    })
}

/// Returns the deterministic 12-scenario Phase 3 suite.
#[must_use]
pub fn standard_scenarios() -> Vec<BudgetScenario> {
    let dimensions = [8_usize, 16, 32, 64];
    let noise_and_budget = [(0.0_f32, 45_u64), (0.02_f32, 60_u64), (0.05_f32, 75_u64)];
    let mut scenarios = Vec::with_capacity(dimensions.len() * noise_and_budget.len());

    for (dimension_index, dimension) in dimensions.into_iter().enumerate()
    {
        for (variant_index, (noise_amplitude, budget_percent)) in
            noise_and_budget.into_iter().enumerate()
        {
            let intrinsic_key_rank = 1 + ((dimension_index + variant_index) % 4);
            let intrinsic_value_rank = 1 + ((2 * dimension_index + variant_index + 1) % 5);
            let maximum_key_rank = (intrinsic_key_rank + 4).min(dimension);
            let maximum_value_rank = (intrinsic_value_rank + 4).min(dimension);
            let token_count = (maximum_key_rank.max(maximum_value_rank) + 8).max(16);
            let reconstruction_target = if noise_amplitude == 0.0
            {
                1.0e-5
            }
            else if noise_amplitude <= 0.02
            {
                0.08
            }
            else
            {
                0.15
            };
            let attention_target = if noise_amplitude == 0.0
            {
                2.0e-5
            }
            else if noise_amplitude <= 0.02
            {
                0.02
            }
            else
            {
                0.04
            };
            let seed = 0xE1A5_7300_0000_0000_u64
                ^ ((dimension as u64) << 32)
                ^ ((variant_index as u64) << 24)
                ^ ((intrinsic_key_rank as u64) << 16)
                ^ ((intrinsic_value_rank as u64) << 8)
                ^ budget_percent;

            scenarios.push(BudgetScenario {
                token_count,
                head_dimension: dimension,
                query_count: 4,
                intrinsic_key_rank,
                intrinsic_value_rank,
                maximum_key_rank,
                maximum_value_rank,
                budget_numerator: budget_percent,
                budget_denominator: 100,
                key_target_relative_root_mean_square: reconstruction_target,
                value_target_relative_root_mean_square: reconstruction_target,
                attention_target_max_absolute: attention_target,
                noise_amplitude,
                signal_amplitude: 1.0,
                seed,
            });
        }
    }

    scenarios
}

/// Runs the deterministic 12-scenario Phase 3 suite.
pub fn run_standard_suite() -> Result<Vec<BudgetScenarioReport>, BudgetError> {
    standard_scenarios()
        .iter()
        .map(run_budget_scenario)
        .collect()
}

/// Serializes Phase 3 reports as stable newline-terminated CSV.
#[must_use]
pub fn suite_to_csv(reports: &[BudgetScenarioReport]) -> String {
    let mut csv = String::new();
    csv.push_str(CSV_HEADER);
    csv.push('\n');

    for report in reports
    {
        csv.push_str(&report.to_csv_row());
        csv.push('\n');
    }

    csv
}

#[derive(Debug, Clone)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        value.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_symmetric_f32(&mut self, amplitude: f32) -> f32 {
        const MANTISSA_MASK: u64 = (1_u64 << 24) - 1;
        const DENOMINATOR: f32 = MANTISSA_MASK as f32;

        let mantissa = (self.next_u64() >> 40) & MANTISSA_MASK;
        let unit_interval = mantissa as f32 / DENOMINATOR;
        (2.0 * unit_interval - 1.0) * amplitude
    }
}

fn random_basis(
    rng: &mut DeterministicRng,
    dimension: usize,
    rank: usize,
) -> Result<OrthonormalBasis, BudgetError> {
    let raw = random_vector(rng, checked_usize_mul(rank, dimension)?, 1.0);
    Ok(OrthonormalBasis::from_greedy_samples(
        &raw, rank, dimension, rank, 1.0e-10,
    )?)
}

fn generate_dataset(
    rng: &mut DeterministicRng,
    sample_count: usize,
    basis: &OrthonormalBasis,
    signal_amplitude: f32,
    noise_amplitude: f32,
) -> Result<Vec<f32>, BudgetError> {
    let length = checked_usize_mul(sample_count, basis.dimension())?;
    let mut samples = vec![0.0_f32; length];
    let mut coefficients = vec![0.0_f32; basis.rank()];

    for sample in samples.chunks_exact_mut(basis.dimension())
    {
        fill_random_vector(rng, &mut coefficients, signal_amplitude);
        basis.reconstruct_into(&coefficients, sample)?;

        if noise_amplitude > 0.0
        {
            for value in sample
            {
                *value += rng.next_symmetric_f32(noise_amplitude);
            }
        }
    }

    Ok(samples)
}

fn random_vector(rng: &mut DeterministicRng, length: usize, amplitude: f32) -> Vec<f32> {
    let mut vector = vec![0.0_f32; length];
    fill_random_vector(rng, &mut vector, amplitude);
    vector
}

fn fill_random_vector(rng: &mut DeterministicRng, vector: &mut [f32], amplitude: f32) {
    for value in vector
    {
        *value = rng.next_symmetric_f32(amplitude);
    }
}

fn require_non_zero(field: &'static str, value: usize) -> Result<(), BudgetError> {
    if value == 0
    {
        return Err(BudgetError::ZeroField { field });
    }
    Ok(())
}

fn require_buffer_length(
    name: &'static str,
    buffer: &[f32],
    expected: usize,
) -> Result<(), BudgetError> {
    if buffer.len() != expected
    {
        return Err(BudgetError::InvalidBufferLength {
            name,
            expected,
            actual: buffer.len(),
        });
    }
    Ok(())
}

fn checked_usize_mul(left: usize, right: usize) -> Result<usize, BudgetError> {
    left.checked_mul(right)
        .ok_or(BudgetError::ArithmeticOverflow)
}

fn checked_mul(left: u64, right: u64) -> Result<u64, BudgetError> {
    left.checked_mul(right)
        .ok_or(BudgetError::ArithmeticOverflow)
}

fn to_u64(value: usize) -> Result<u64, BudgetError> {
    u64::try_from(value).map_err(|_| BudgetError::ArithmeticOverflow)
}

#[cfg(test)]
mod tests {
    use super::{
        BudgetError, BudgetPlannerInput, BudgetScenario, CSV_HEADER, plan_under_budget,
        run_budget_scenario, standard_scenarios, storage_accounting, suite_to_csv,
    };
    use crate::phase2::OrthonormalBasis;

    fn compact_exact_scenario() -> BudgetScenario {
        BudgetScenario {
            token_count: 16,
            head_dimension: 8,
            query_count: 3,
            intrinsic_key_rank: 2,
            intrinsic_value_rank: 3,
            maximum_key_rank: 6,
            maximum_value_rank: 6,
            budget_numerator: 50,
            budget_denominator: 100,
            key_target_relative_root_mean_square: 1.0e-5,
            value_target_relative_root_mean_square: 1.0e-5,
            attention_target_max_absolute: 2.0e-5,
            noise_amplitude: 0.0,
            signal_amplitude: 1.0,
            seed: 0xE1A5_7300_CAFE_BABE,
        }
    }

    #[test]
    fn storage_accounting_matches_closed_form() {
        let accounting = storage_accounting(16, 8, 2, 3).unwrap();

        assert_eq!(accounting.dense_bytes, 1_024);
        assert_eq!(accounting.coefficient_bytes, 320);
        assert_eq!(accounting.basis_bytes, 160);
        assert_eq!(accounting.total_bytes, 480);
        assert_eq!(accounting.savings_bytes, 544);
        assert_eq!(
            accounting.compression_ratio.to_bits(),
            (1024.0_f64 / 480.0).to_bits()
        );
    }

    #[test]
    fn budget_below_rank_one_pair_is_rejected() {
        let basis = OrthonormalBasis::identity(2, 2).unwrap();
        let keys = [1.0, 0.0, 0.0, 1.0];
        let values = keys;
        let queries = [1.0, 0.0];
        let minimum = storage_accounting(2, 2, 1, 1).unwrap().total_bytes;

        let error = plan_under_budget(BudgetPlannerInput {
            keys: &keys,
            values: &values,
            token_count: 2,
            queries: &queries,
            query_count: 1,
            dimension: 2,
            maximum_key_rank: basis.rank(),
            maximum_value_rank: basis.rank(),
            budget_bytes: minimum - 1,
            key_target_relative_root_mean_square: 1.0,
            value_target_relative_root_mean_square: 1.0,
            attention_target_max_absolute: 1.0,
            norm_tolerance: 1.0e-12,
            scale: 1.0,
        })
        .unwrap_err();

        assert_eq!(
            error,
            BudgetError::BudgetBelowMinimum {
                budget_bytes: minimum - 1,
                minimum_bytes: minimum,
            }
        );
    }

    #[test]
    fn exact_scenario_recovers_intrinsic_ranks_under_budget() {
        let scenario = compact_exact_scenario();
        let report = run_budget_scenario(&scenario).unwrap();

        assert!(report.plan.selected.quality_guard_met);
        assert_eq!(report.plan.selected.key_rank, scenario.intrinsic_key_rank);
        assert_eq!(
            report.plan.selected.value_rank,
            scenario.intrinsic_value_rank
        );
        assert!(report.plan.selected.storage.total_bytes <= report.plan.budget_bytes);
        assert!(report.plan.selected.attention.max_absolute <= 2.0e-5);
    }

    #[test]
    fn tight_budget_exposes_quality_guard_failure() {
        let mut scenario = compact_exact_scenario();
        scenario.budget_numerator = 19;
        let report = run_budget_scenario(&scenario).unwrap();

        assert!(!report.plan.selected.quality_guard_met);
        assert_eq!(report.plan.selected.key_rank, 1);
        assert_eq!(report.plan.selected.value_rank, 1);
        assert_eq!(report.plan.quality_feasible_pairs, 0);
    }

    #[test]
    fn selected_candidate_never_exceeds_strict_budget() {
        for scenario in standard_scenarios()
        {
            let report = run_budget_scenario(&scenario).unwrap();
            assert!(report.plan.selected.storage.total_bytes <= report.plan.budget_bytes);
        }
    }

    #[test]
    fn pareto_frontier_contains_only_non_dominated_candidates() {
        let report = run_budget_scenario(&compact_exact_scenario()).unwrap();
        let frontier = &report.plan.pareto_frontier;

        assert!(!frontier.is_empty());
        for (index, candidate) in frontier.iter().enumerate()
        {
            for (other_index, other) in frontier.iter().enumerate()
            {
                if index == other_index
                {
                    continue;
                }

                let no_worse = other.storage.total_bytes <= candidate.storage.total_bytes
                    && other.key_reconstruction.relative_root_mean_square
                        <= candidate.key_reconstruction.relative_root_mean_square
                    && other.value_reconstruction.relative_root_mean_square
                        <= candidate.value_reconstruction.relative_root_mean_square
                    && other.attention.max_absolute <= candidate.attention.max_absolute;
                let strict = other.storage.total_bytes < candidate.storage.total_bytes
                    || other.key_reconstruction.relative_root_mean_square
                        < candidate.key_reconstruction.relative_root_mean_square
                    || other.value_reconstruction.relative_root_mean_square
                        < candidate.value_reconstruction.relative_root_mean_square
                    || other.attention.max_absolute < candidate.attention.max_absolute;

                assert!(!(no_worse && strict));
            }
        }
    }

    #[test]
    fn scenario_execution_is_bit_deterministic() {
        let scenario = compact_exact_scenario();
        let first = run_budget_scenario(&scenario).unwrap();
        let second = run_budget_scenario(&scenario).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.to_csv_row(), second.to_csv_row());
    }

    #[test]
    fn standard_suite_is_stable_and_complete() {
        let scenarios = standard_scenarios();

        assert_eq!(scenarios.len(), 12);
        assert_eq!(
            scenarios
                .iter()
                .filter(|scenario| scenario.noise_amplitude == 0.0)
                .count(),
            4
        );
        assert!(scenarios.iter().all(|scenario| scenario.validate().is_ok()));
    }

    #[test]
    fn csv_export_has_expected_shape() {
        let reports: Vec<_> = standard_scenarios()
            .iter()
            .map(|scenario| run_budget_scenario(scenario).unwrap())
            .collect();
        let csv = suite_to_csv(&reports);
        let mut lines = csv.lines();

        assert_eq!(lines.next(), Some(CSV_HEADER));
        assert_eq!(lines.count(), reports.len());
        assert_eq!(CSV_HEADER.split(',').count(), 31);
        assert!(
            reports
                .iter()
                .all(|report| report.to_csv_row().split(',').count() == 31)
        );
    }
}

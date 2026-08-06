//! Phase 5 sparse residual correction for the Elastic Latent KV experiment.
//!
//! Phase 5 augments one fixed-rank latent representation with a deterministic
//! fixed-slot sparse residual channel. Key residuals correct attention scores
//! without dense reconstruction, while value residuals are accumulated directly
//! into the dense output. The planner enumerates rank and residual-slot tuples
//! under a strict persistent byte budget and compares them with a zero-residual
//! baseline evaluated on the same inputs.

use crate::phase2::{
    OrthonormalBasis, ProjectedAttentionMetrics, ProjectionError, ReconstructionMetrics,
};
use core::{cmp::Ordering, fmt};

const F32_BYTES: u64 = 4;
const U16_BYTES: u64 = 2;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const ENERGY_FLOOR: f64 = 1.0e-30;
const EMPTY_INDEX: u16 = u16::MAX;

/// Errors returned by deterministic Phase 5 residual planning.
#[derive(Debug, Clone, PartialEq)]
pub enum ResidualError {
    /// A required count, dimension or rank was zero.
    ZeroField {
        /// Human-readable field name.
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
    /// A requested rank exceeds its maximum.
    InvalidRank {
        /// Requested rank.
        rank: usize,
        /// Maximum accepted rank.
        maximum: usize,
    },
    /// A residual slot count exceeds the dense dimension.
    InvalidSlots {
        /// Requested slots per token.
        slots: usize,
        /// Maximum accepted slots per token.
        maximum: usize,
    },
    /// The dense dimension cannot be encoded by the `u16` residual index format.
    DimensionTooLarge {
        /// Supplied dense dimension.
        dimension: usize,
        /// Largest accepted dense dimension.
        maximum: usize,
    },
    /// A numeric target is non-finite or outside its accepted range.
    InvalidThreshold {
        /// Human-readable target name.
        name: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// The strict budget cannot store even the rank-one zero-residual baseline.
    BudgetBelowMinimum {
        /// Supplied strict budget.
        budget_bytes: u64,
        /// Minimum persistent bytes required.
        minimum_bytes: u64,
    },
    /// An integer accounting operation overflowed.
    ArithmeticOverflow,
    /// A Phase 2 projection operation failed.
    Projection(ProjectionError),
}

impl fmt::Display for ResidualError {
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
            Self::InvalidSlots { slots, maximum } =>
            {
                write!(formatter, "residual slots {slots} exceed maximum {maximum}")
            },
            Self::DimensionTooLarge { dimension, maximum } => write!(
                formatter,
                "dimension {dimension} exceeds residual-index maximum {maximum}"
            ),
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
            Self::ArithmeticOverflow => write!(formatter, "residual accounting overflow"),
            Self::Projection(error) => write!(formatter, "projection error: {error}"),
        }
    }
}

impl std::error::Error for ResidualError {}

impl From<ProjectionError> for ResidualError {
    fn from(error: ProjectionError) -> Self {
        Self::Projection(error)
    }
}

/// Exact persistent-storage accounting for one residual-augmented candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualStorageAccounting {
    /// Dense key/value payload bytes for the same token count and dimension.
    pub dense_bytes: u64,
    /// Per-token latent key/value coefficient bytes.
    pub coefficient_bytes: u64,
    /// Shared dense-to-latent key/value basis bytes.
    pub basis_bytes: u64,
    /// Fixed-slot sparse residual index bytes.
    pub residual_index_bytes: u64,
    /// Fixed-slot sparse residual value bytes.
    pub residual_value_bytes: u64,
    /// Total persistent bytes.
    pub total_bytes: u64,
    /// Dense bytes not consumed by the candidate.
    pub savings_bytes: u64,
    /// Dense bytes divided by candidate bytes.
    pub compression_ratio: f64,
}

/// Computes exact persistent bytes for one residual-augmented representation.
pub fn residual_storage_accounting(
    token_count: usize,
    dimension: usize,
    key_rank: usize,
    value_rank: usize,
    key_slots_per_token: usize,
    value_slots_per_token: usize,
) -> Result<ResidualStorageAccounting, ResidualError> {
    require_non_zero("token_count", token_count)?;
    require_non_zero("dimension", dimension)?;
    require_non_zero("key_rank", key_rank)?;
    require_non_zero("value_rank", value_rank)?;

    if key_rank > dimension
    {
        return Err(ResidualError::InvalidRank {
            rank: key_rank,
            maximum: dimension,
        });
    }
    if value_rank > dimension
    {
        return Err(ResidualError::InvalidRank {
            rank: value_rank,
            maximum: dimension,
        });
    }
    if key_slots_per_token > dimension
    {
        return Err(ResidualError::InvalidSlots {
            slots: key_slots_per_token,
            maximum: dimension,
        });
    }
    if value_slots_per_token > dimension
    {
        return Err(ResidualError::InvalidSlots {
            slots: value_slots_per_token,
            maximum: dimension,
        });
    }
    require_encodable_dimension(dimension)?;

    let tokens = to_u64(token_count)?;
    let dense_dimension = to_u64(dimension)?;
    let rank_sum = checked_add(to_u64(key_rank)?, to_u64(value_rank)?)?;
    let slot_sum = checked_add(to_u64(key_slots_per_token)?, to_u64(value_slots_per_token)?)?;

    let dense_bytes = checked_mul(checked_mul(tokens, dense_dimension)?, 2 * F32_BYTES)?;
    let coefficient_bytes = checked_mul(checked_mul(tokens, rank_sum)?, F32_BYTES)?;
    let basis_bytes = checked_mul(checked_mul(dense_dimension, rank_sum)?, F32_BYTES)?;
    let residual_index_bytes = checked_mul(checked_mul(tokens, slot_sum)?, U16_BYTES)?;
    let residual_value_bytes = checked_mul(checked_mul(tokens, slot_sum)?, F32_BYTES)?;
    let total_bytes = checked_add(
        checked_add(coefficient_bytes, basis_bytes)?,
        checked_add(residual_index_bytes, residual_value_bytes)?,
    )?;

    Ok(ResidualStorageAccounting {
        dense_bytes,
        coefficient_bytes,
        basis_bytes,
        residual_index_bytes,
        residual_value_bytes,
        total_bytes,
        savings_bytes: dense_bytes.saturating_sub(total_bytes),
        compression_ratio: dense_bytes as f64 / total_bytes as f64,
    })
}

/// Fixed-slot deterministic sparse residual matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseResidualMatrix {
    token_count: usize,
    dimension: usize,
    slots_per_token: usize,
    indices: Vec<u16>,
    values: Vec<f32>,
}

impl SparseResidualMatrix {
    /// Builds top-magnitude residual slots after projection onto `basis`.
    ///
    /// Ties are resolved by the lowest dense coordinate index. Empty reserved
    /// slots use the sentinel index `u16::MAX` and a zero value.
    pub fn from_samples(
        samples: &[f32],
        token_count: usize,
        basis: &OrthonormalBasis,
        slots_per_token: usize,
    ) -> Result<Self, ResidualError> {
        require_non_zero("token_count", token_count)?;
        require_encodable_dimension(basis.dimension())?;

        if slots_per_token > basis.dimension()
        {
            return Err(ResidualError::InvalidSlots {
                slots: slots_per_token,
                maximum: basis.dimension(),
            });
        }

        let expected = checked_usize_mul(token_count, basis.dimension())?;
        require_buffer_length("samples", samples, expected)?;
        let slot_count = checked_usize_mul(token_count, slots_per_token)?;
        let mut indices = vec![EMPTY_INDEX; slot_count];
        let mut values = vec![0.0_f32; slot_count];

        if slots_per_token == 0
        {
            return Ok(Self {
                token_count,
                dimension: basis.dimension(),
                slots_per_token,
                indices,
                values,
            });
        }

        let mut coefficients = vec![0.0_f32; basis.rank()];
        let mut reconstruction = vec![0.0_f32; basis.dimension()];
        let mut residual = vec![0.0_f32; basis.dimension()];
        let mut selected = vec![false; basis.dimension()];

        for (token_index, sample) in samples.chunks_exact(basis.dimension()).enumerate()
        {
            basis.project_into(sample, &mut coefficients)?;
            basis.reconstruct_into(&coefficients, &mut reconstruction)?;

            for ((destination, reference), candidate) in residual
                .iter_mut()
                .zip(sample.iter())
                .zip(reconstruction.iter())
            {
                *destination = *reference - *candidate;
            }
            selected.fill(false);

            for slot_index in 0..slots_per_token
            {
                let mut best_index = None;
                let mut best_magnitude = 0.0_f32;

                for (coordinate, value) in residual.iter().copied().enumerate()
                {
                    if selected[coordinate]
                    {
                        continue;
                    }
                    let magnitude = value.abs();
                    let replace = magnitude > best_magnitude
                        || (magnitude == best_magnitude
                            && magnitude > 0.0
                            && best_index.is_none_or(|current| coordinate < current));
                    if replace
                    {
                        best_index = Some(coordinate);
                        best_magnitude = magnitude;
                    }
                }

                let Some(coordinate) = best_index
                else
                {
                    break;
                };
                if best_magnitude == 0.0
                {
                    break;
                }

                selected[coordinate] = true;
                let destination = token_index * slots_per_token + slot_index;
                indices[destination] =
                    u16::try_from(coordinate).map_err(|_| ResidualError::ArithmeticOverflow)?;
                values[destination] = residual[coordinate];
            }
        }

        Ok(Self {
            token_count,
            dimension: basis.dimension(),
            slots_per_token,
            indices,
            values,
        })
    }

    /// Returns the number of represented tokens.
    #[must_use]
    pub const fn token_count(&self) -> usize {
        self.token_count
    }

    /// Returns the dense vector dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns reserved residual slots per token.
    #[must_use]
    pub const fn slots_per_token(&self) -> usize {
        self.slots_per_token
    }

    /// Returns the flat fixed-slot index buffer.
    #[must_use]
    pub fn indices(&self) -> &[u16] {
        &self.indices
    }

    /// Returns the flat fixed-slot residual value buffer.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    fn token_range(&self, token_index: usize) -> Result<core::ops::Range<usize>, ResidualError> {
        if token_index >= self.token_count
        {
            return Err(ResidualError::InvalidBufferLength {
                name: "token_index",
                expected: self.token_count,
                actual: token_index.saturating_add(1),
            });
        }
        let start = checked_usize_mul(token_index, self.slots_per_token)?;
        Ok(start..start + self.slots_per_token)
    }

    fn dot(&self, token_index: usize, dense: &[f32]) -> Result<f32, ResidualError> {
        require_buffer_length("dense", dense, self.dimension)?;
        let range = self.token_range(token_index)?;
        let mut sum = 0.0_f32;

        for (&index, &value) in self.indices[range.clone()]
            .iter()
            .zip(self.values[range].iter())
        {
            if index != EMPTY_INDEX
            {
                sum += dense[usize::from(index)] * value;
            }
        }
        Ok(sum)
    }

    fn add_scaled_to(
        &self,
        token_index: usize,
        scale: f32,
        dense: &mut [f32],
    ) -> Result<(), ResidualError> {
        require_buffer_length("dense", dense, self.dimension)?;
        let range = self.token_range(token_index)?;

        for (&index, &value) in self.indices[range.clone()]
            .iter()
            .zip(self.values[range].iter())
        {
            if index != EMPTY_INDEX
            {
                dense[usize::from(index)] += scale * value;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct ResidualRepresentation {
    basis: OrthonormalBasis,
    coefficients: Vec<f32>,
    residuals: SparseResidualMatrix,
}

impl ResidualRepresentation {
    fn from_samples(
        samples: &[f32],
        token_count: usize,
        basis: OrthonormalBasis,
        slots_per_token: usize,
    ) -> Result<Self, ResidualError> {
        let expected = checked_usize_mul(token_count, basis.dimension())?;
        require_buffer_length("samples", samples, expected)?;
        let coefficient_count = checked_usize_mul(token_count, basis.rank())?;
        let mut coefficients = vec![0.0_f32; coefficient_count];

        for (sample, destination) in samples
            .chunks_exact(basis.dimension())
            .zip(coefficients.chunks_exact_mut(basis.rank()))
        {
            basis.project_into(sample, destination)?;
        }

        let residuals =
            SparseResidualMatrix::from_samples(samples, token_count, &basis, slots_per_token)?;

        Ok(Self {
            basis,
            coefficients,
            residuals,
        })
    }

    fn coefficients_for(&self, token_index: usize) -> Result<&[f32], ResidualError> {
        if token_index >= self.residuals.token_count()
        {
            return Err(ResidualError::InvalidBufferLength {
                name: "token_index",
                expected: self.residuals.token_count(),
                actual: token_index.saturating_add(1),
            });
        }
        let start = checked_usize_mul(token_index, self.basis.rank())?;
        Ok(&self.coefficients[start..start + self.basis.rank()])
    }
}

/// Complete reconstruction and attention evaluation for one residual candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualEvaluation {
    /// Key reconstruction metrics.
    pub key_reconstruction: ReconstructionMetrics,
    /// Value reconstruction metrics.
    pub value_reconstruction: ReconstructionMetrics,
    /// Dense-versus-residual-latent attention metrics.
    pub attention: ProjectedAttentionMetrics,
}

/// Inputs for one reconstruction-free sparse-residual evaluation.
#[derive(Debug, Clone, Copy)]
pub struct ResidualEvaluationInput<'a> {
    /// Dense keys in row-major `[token_count, dimension]` order.
    pub keys: &'a [f32],
    /// Dense values in row-major `[token_count, dimension]` order.
    pub values: &'a [f32],
    /// Number of cached key/value vectors.
    pub token_count: usize,
    /// Dense queries in row-major `[query_count, dimension]` order.
    pub queries: &'a [f32],
    /// Number of query vectors.
    pub query_count: usize,
    /// Selected key basis.
    pub key_basis: &'a OrthonormalBasis,
    /// Selected value basis.
    pub value_basis: &'a OrthonormalBasis,
    /// Reserved key residual slots per token.
    pub key_slots_per_token: usize,
    /// Reserved value residual slots per token.
    pub value_slots_per_token: usize,
    /// Positive finite attention scale.
    pub scale: f32,
}

/// Evaluates residual reconstruction and reconstruction-free attention.
pub fn evaluate_sparse_residual_attention(
    input: ResidualEvaluationInput<'_>,
) -> Result<ResidualEvaluation, ResidualError> {
    validate_evaluation_input(input)?;

    let key_representation = ResidualRepresentation::from_samples(
        input.keys,
        input.token_count,
        input.key_basis.clone(),
        input.key_slots_per_token,
    )?;
    let value_representation = ResidualRepresentation::from_samples(
        input.values,
        input.token_count,
        input.value_basis.clone(),
        input.value_slots_per_token,
    )?;

    let key_reconstruction =
        reconstruction_metrics_with_residuals(input.keys, input.token_count, &key_representation)?;
    let value_reconstruction = reconstruction_metrics_with_residuals(
        input.values,
        input.token_count,
        &value_representation,
    )?;
    let attention =
        attention_metrics_with_residuals(input, &key_representation, &value_representation)?;

    Ok(ResidualEvaluation {
        key_reconstruction,
        value_reconstruction,
        attention,
    })
}

/// One rank-and-residual tuple evaluated under a strict budget.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualCandidate {
    /// Key latent rank.
    pub key_rank: usize,
    /// Value latent rank.
    pub value_rank: usize,
    /// Key residual slots per token.
    pub key_slots_per_token: usize,
    /// Value residual slots per token.
    pub value_slots_per_token: usize,
    /// Exact persistent storage accounting.
    pub storage: ResidualStorageAccounting,
    /// Key reconstruction metrics.
    pub key_reconstruction: ReconstructionMetrics,
    /// Value reconstruction metrics.
    pub value_reconstruction: ReconstructionMetrics,
    /// Attention error metrics.
    pub attention: ProjectedAttentionMetrics,
    /// Maximum normalized quality-target ratio.
    pub worst_target_ratio: f64,
    /// Whether all reconstruction and attention targets are satisfied.
    pub quality_guard_met: bool,
}

/// Complete strict-budget residual planning result.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualPlan {
    /// Strict persistent-storage budget.
    pub budget_bytes: u64,
    /// Best zero-residual candidate under the same budget.
    pub baseline: ResidualCandidate,
    /// Selected residual-aware candidate.
    pub selected: ResidualCandidate,
    /// Number of rank-and-slot tuples evaluated.
    pub evaluated_candidates: usize,
    /// Number of candidates fitting the strict budget.
    pub budget_feasible_candidates: usize,
    /// Number of candidates satisfying all quality targets.
    pub quality_feasible_candidates: usize,
    /// Deterministic non-dominated candidates ordered by storage then quality.
    pub pareto_frontier: Vec<ResidualCandidate>,
}

/// Inputs for strict-budget residual planning.
#[derive(Debug, Clone, Copy)]
pub struct ResidualPlannerInput<'a> {
    /// Dense keys in row-major `[token_count, dimension]` order.
    pub keys: &'a [f32],
    /// Dense values in row-major `[token_count, dimension]` order.
    pub values: &'a [f32],
    /// Number of cached key/value vectors.
    pub token_count: usize,
    /// Dense queries in row-major `[query_count, dimension]` order.
    pub queries: &'a [f32],
    /// Number of dense query vectors.
    pub query_count: usize,
    /// Maximum nested key basis.
    pub maximum_key_basis: &'a OrthonormalBasis,
    /// Maximum nested value basis.
    pub maximum_value_basis: &'a OrthonormalBasis,
    /// Maximum key residual slots per token.
    pub maximum_key_slots_per_token: usize,
    /// Maximum value residual slots per token.
    pub maximum_value_slots_per_token: usize,
    /// Strict persistent byte budget.
    pub budget_bytes: u64,
    /// Maximum key reconstruction relative RMS.
    pub key_target_relative_root_mean_square: f64,
    /// Maximum value reconstruction relative RMS.
    pub value_target_relative_root_mean_square: f64,
    /// Maximum dense-versus-candidate attention absolute error.
    pub attention_target_max_absolute: f64,
    /// Positive finite attention scale.
    pub scale: f32,
}

/// Enumerates deterministic rank-and-slot tuples under a strict byte budget.
pub fn plan_with_sparse_residuals(
    input: ResidualPlannerInput<'_>,
) -> Result<ResidualPlan, ResidualError> {
    validate_planner_input(input)?;

    let minimum_bytes = residual_storage_accounting(
        input.token_count,
        input.maximum_key_basis.dimension(),
        1,
        1,
        0,
        0,
    )?
    .total_bytes;
    if input.budget_bytes < minimum_bytes
    {
        return Err(ResidualError::BudgetBelowMinimum {
            budget_bytes: input.budget_bytes,
            minimum_bytes,
        });
    }

    let mut candidates = Vec::new();
    let mut evaluated_candidates = 0_usize;

    for key_rank in 1..=input.maximum_key_basis.rank()
    {
        let key_basis = input.maximum_key_basis.prefix(key_rank)?;
        for value_rank in 1..=input.maximum_value_basis.rank()
        {
            let value_basis = input.maximum_value_basis.prefix(value_rank)?;
            for key_slots in 0..=input.maximum_key_slots_per_token
            {
                for value_slots in 0..=input.maximum_value_slots_per_token
                {
                    evaluated_candidates = evaluated_candidates
                        .checked_add(1)
                        .ok_or(ResidualError::ArithmeticOverflow)?;
                    let storage = residual_storage_accounting(
                        input.token_count,
                        input.maximum_key_basis.dimension(),
                        key_rank,
                        value_rank,
                        key_slots,
                        value_slots,
                    )?;
                    if storage.total_bytes > input.budget_bytes
                    {
                        continue;
                    }

                    let evaluation = evaluate_sparse_residual_attention(ResidualEvaluationInput {
                        keys: input.keys,
                        values: input.values,
                        token_count: input.token_count,
                        queries: input.queries,
                        query_count: input.query_count,
                        key_basis: &key_basis,
                        value_basis: &value_basis,
                        key_slots_per_token: key_slots,
                        value_slots_per_token: value_slots,
                        scale: input.scale,
                    })?;
                    let key_ratio = target_ratio(
                        evaluation.key_reconstruction.relative_root_mean_square,
                        input.key_target_relative_root_mean_square,
                    );
                    let value_ratio = target_ratio(
                        evaluation.value_reconstruction.relative_root_mean_square,
                        input.value_target_relative_root_mean_square,
                    );
                    let attention_ratio = target_ratio(
                        evaluation.attention.max_absolute,
                        input.attention_target_max_absolute,
                    );
                    let worst_target_ratio = key_ratio.max(value_ratio).max(attention_ratio);
                    let quality_guard_met =
                        key_ratio <= 1.0 && value_ratio <= 1.0 && attention_ratio <= 1.0;

                    candidates.push(ResidualCandidate {
                        key_rank,
                        value_rank,
                        key_slots_per_token: key_slots,
                        value_slots_per_token: value_slots,
                        storage,
                        key_reconstruction: evaluation.key_reconstruction,
                        value_reconstruction: evaluation.value_reconstruction,
                        attention: evaluation.attention,
                        worst_target_ratio,
                        quality_guard_met,
                    });
                }
            }
        }
    }

    let baseline = candidates
        .iter()
        .filter(|candidate| {
            candidate.key_slots_per_token == 0 && candidate.value_slots_per_token == 0
        })
        .min_by(|left, right| compare_candidates(left, right))
        .cloned()
        .ok_or(ResidualError::BudgetBelowMinimum {
            budget_bytes: input.budget_bytes,
            minimum_bytes,
        })?;
    let selected = candidates
        .iter()
        .min_by(|left, right| compare_candidates(left, right))
        .cloned()
        .ok_or(ResidualError::BudgetBelowMinimum {
            budget_bytes: input.budget_bytes,
            minimum_bytes,
        })?;
    let quality_feasible_candidates = candidates
        .iter()
        .filter(|candidate| candidate.quality_guard_met)
        .count();
    let pareto_frontier = residual_pareto_frontier(&candidates);

    Ok(ResidualPlan {
        budget_bytes: input.budget_bytes,
        baseline,
        selected,
        evaluated_candidates,
        budget_feasible_candidates: candidates.len(),
        quality_feasible_candidates,
        pareto_frontier,
    })
}

/// Deterministic Phase 5 structured-residual scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualScenario {
    /// Number of dense key/value vectors.
    pub token_count: usize,
    /// Dense head dimension.
    pub head_dimension: usize,
    /// Number of dense queries.
    pub query_count: usize,
    /// Intrinsic low-rank key dimension.
    pub intrinsic_key_rank: usize,
    /// Intrinsic low-rank value dimension.
    pub intrinsic_value_rank: usize,
    /// Sparse key residual coordinates injected per token.
    pub injected_key_slots_per_token: usize,
    /// Sparse value residual coordinates injected per token.
    pub injected_value_slots_per_token: usize,
    /// Maximum key latent rank explored.
    pub maximum_key_rank: usize,
    /// Maximum value latent rank explored.
    pub maximum_value_rank: usize,
    /// Maximum key residual slots explored.
    pub maximum_key_slots_per_token: usize,
    /// Maximum value residual slots explored.
    pub maximum_value_slots_per_token: usize,
    /// Budget numerator applied to dense payload bytes.
    pub budget_numerator: u64,
    /// Budget denominator applied to dense payload bytes.
    pub budget_denominator: u64,
    /// Sparse residual amplitude.
    pub residual_amplitude: f32,
    /// Signal coefficient amplitude.
    pub signal_amplitude: f32,
    /// Key reconstruction relative-RMS target.
    pub key_target_relative_root_mean_square: f64,
    /// Value reconstruction relative-RMS target.
    pub value_target_relative_root_mean_square: f64,
    /// Attention absolute-error target.
    pub attention_target_max_absolute: f64,
    /// Deterministic seed.
    pub seed: u64,
}

impl ResidualScenario {
    /// Validates structural, budget and numeric fields.
    pub fn validate(&self) -> Result<(), ResidualError> {
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
            return Err(ResidualError::ZeroField {
                field: "budget_numerator",
            });
        }
        if self.budget_denominator == 0
        {
            return Err(ResidualError::ZeroField {
                field: "budget_denominator",
            });
        }
        require_encodable_dimension(self.head_dimension)?;

        for rank in [
            self.intrinsic_key_rank,
            self.intrinsic_value_rank,
            self.maximum_key_rank,
            self.maximum_value_rank,
        ]
        {
            if rank > self.head_dimension
            {
                return Err(ResidualError::InvalidRank {
                    rank,
                    maximum: self.head_dimension,
                });
            }
        }
        if self.intrinsic_key_rank > self.maximum_key_rank
        {
            return Err(ResidualError::InvalidRank {
                rank: self.intrinsic_key_rank,
                maximum: self.maximum_key_rank,
            });
        }
        if self.intrinsic_value_rank > self.maximum_value_rank
        {
            return Err(ResidualError::InvalidRank {
                rank: self.intrinsic_value_rank,
                maximum: self.maximum_value_rank,
            });
        }

        for slots in [
            self.injected_key_slots_per_token,
            self.injected_value_slots_per_token,
            self.maximum_key_slots_per_token,
            self.maximum_value_slots_per_token,
        ]
        {
            if slots > self.head_dimension
            {
                return Err(ResidualError::InvalidSlots {
                    slots,
                    maximum: self.head_dimension,
                });
            }
        }
        if self.injected_key_slots_per_token > self.maximum_key_slots_per_token
        {
            return Err(ResidualError::InvalidSlots {
                slots: self.injected_key_slots_per_token,
                maximum: self.maximum_key_slots_per_token,
            });
        }
        if self.injected_value_slots_per_token > self.maximum_value_slots_per_token
        {
            return Err(ResidualError::InvalidSlots {
                slots: self.injected_value_slots_per_token,
                maximum: self.maximum_value_slots_per_token,
            });
        }

        let key_tail = self.head_dimension - self.maximum_key_rank;
        let value_tail = self.head_dimension - self.maximum_value_rank;
        if self.injected_key_slots_per_token > key_tail
        {
            return Err(ResidualError::InvalidSlots {
                slots: self.injected_key_slots_per_token,
                maximum: key_tail,
            });
        }
        if self.injected_value_slots_per_token > value_tail
        {
            return Err(ResidualError::InvalidSlots {
                slots: self.injected_value_slots_per_token,
                maximum: value_tail,
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
                return Err(ResidualError::InvalidThreshold { name, value });
            }
        }
        if !self.residual_amplitude.is_finite() || self.residual_amplitude < 0.0
        {
            return Err(ResidualError::InvalidThreshold {
                name: "residual_amplitude",
                value: f64::from(self.residual_amplitude),
            });
        }
        if !self.signal_amplitude.is_finite() || self.signal_amplitude <= 0.0
        {
            return Err(ResidualError::InvalidThreshold {
                name: "signal_amplitude",
                value: f64::from(self.signal_amplitude),
            });
        }
        Ok(())
    }
}

/// Complete report for one deterministic Phase 5 scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualScenarioReport {
    /// Original scenario configuration.
    pub scenario: ResidualScenario,
    /// Strict-budget residual planner result.
    pub plan: ResidualPlan,
}

impl ResidualScenarioReport {
    /// Returns whether residual slots strictly improved the baseline target ratio.
    #[must_use]
    pub fn residual_improved_quality(&self) -> bool {
        self.plan.selected.worst_target_ratio < self.plan.baseline.worst_target_ratio
    }

    /// Serializes one stable CSV row.
    #[must_use]
    pub fn to_csv_row(&self) -> String {
        let selected = &self.plan.selected;
        let baseline = &self.plan.baseline;
        format!(
            concat!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.9e},",
                "{:.9e},{:.9e},{:.9e},{},{},{},{},{},{},{},{},{:.9e},{:.9e},",
                "{},{},{},{:.9e},{:.9e},{:.9e},{:.9e},{},{},{},{},{:016x}"
            ),
            self.scenario.seed,
            self.scenario.token_count,
            self.scenario.head_dimension,
            self.scenario.query_count,
            self.scenario.intrinsic_key_rank,
            self.scenario.intrinsic_value_rank,
            self.scenario.injected_key_slots_per_token,
            self.scenario.injected_value_slots_per_token,
            self.scenario.maximum_key_rank,
            self.scenario.maximum_value_rank,
            self.scenario.maximum_key_slots_per_token,
            self.scenario.maximum_value_slots_per_token,
            self.scenario.budget_numerator,
            self.scenario.budget_denominator,
            self.plan.budget_bytes,
            self.scenario.residual_amplitude,
            self.scenario.key_target_relative_root_mean_square,
            self.scenario.value_target_relative_root_mean_square,
            self.scenario.attention_target_max_absolute,
            selected.key_rank,
            selected.value_rank,
            selected.key_slots_per_token,
            selected.value_slots_per_token,
            baseline.key_rank,
            baseline.value_rank,
            u8::from(baseline.quality_guard_met),
            u8::from(selected.quality_guard_met),
            baseline.worst_target_ratio,
            selected.worst_target_ratio,
            selected.storage.dense_bytes,
            baseline.storage.total_bytes,
            selected.storage.total_bytes,
            selected.storage.compression_ratio,
            selected.key_reconstruction.relative_root_mean_square,
            selected.value_reconstruction.relative_root_mean_square,
            selected.attention.max_absolute,
            self.plan.evaluated_candidates,
            self.plan.budget_feasible_candidates,
            self.plan.quality_feasible_candidates,
            self.plan.pareto_frontier.len(),
            selected.attention.output_fingerprint,
        )
    }
}

/// Stable CSV header for Phase 5 reports.
pub const CSV_HEADER: &str = concat!(
    "seed,token_count,head_dimension,query_count,intrinsic_key_rank,",
    "intrinsic_value_rank,injected_key_slots,injected_value_slots,",
    "maximum_key_rank,maximum_value_rank,maximum_key_slots,maximum_value_slots,",
    "budget_numerator,budget_denominator,budget_bytes,residual_amplitude,",
    "key_target_relative_rms,value_target_relative_rms,",
    "attention_target_max_absolute,selected_key_rank,selected_value_rank,",
    "selected_key_slots,selected_value_slots,baseline_key_rank,",
    "baseline_value_rank,baseline_quality_guard_met,selected_quality_guard_met,",
    "baseline_worst_target_ratio,selected_worst_target_ratio,dense_bytes,",
    "baseline_total_bytes,selected_total_bytes,compression_ratio,",
    "key_reconstruction_relative_rms,value_reconstruction_relative_rms,",
    "attention_max_absolute,evaluated_candidates,budget_feasible_candidates,",
    "quality_feasible_candidates,pareto_frontier_length,output_fingerprint"
);

/// Runs one deterministic Phase 5 residual scenario.
pub fn run_residual_scenario(
    scenario: &ResidualScenario,
) -> Result<ResidualScenarioReport, ResidualError> {
    scenario.validate()?;

    let maximum_key_basis =
        OrthonormalBasis::identity(scenario.head_dimension, scenario.maximum_key_rank)?;
    let maximum_value_basis =
        OrthonormalBasis::identity(scenario.head_dimension, scenario.maximum_value_rank)?;
    let mut rng = DeterministicRng::new(scenario.seed);
    let keys = generate_structured_dataset(
        &mut rng,
        scenario.token_count,
        scenario.head_dimension,
        scenario.intrinsic_key_rank,
        scenario.maximum_key_rank,
        scenario.injected_key_slots_per_token,
        scenario.signal_amplitude,
        scenario.residual_amplitude,
    )?;
    let values = generate_structured_dataset(
        &mut rng,
        scenario.token_count,
        scenario.head_dimension,
        scenario.intrinsic_value_rank,
        scenario.maximum_value_rank,
        scenario.injected_value_slots_per_token,
        scenario.signal_amplitude,
        scenario.residual_amplitude,
    )?;
    let queries = random_vector(
        &mut rng,
        checked_usize_mul(scenario.query_count, scenario.head_dimension)?,
        scenario.signal_amplitude,
    );
    let dense_bytes =
        residual_storage_accounting(scenario.token_count, scenario.head_dimension, 1, 1, 0, 0)?
            .dense_bytes;
    let budget_bytes = dense_bytes
        .checked_mul(scenario.budget_numerator)
        .ok_or(ResidualError::ArithmeticOverflow)?
        / scenario.budget_denominator;
    let scale = (scenario.head_dimension as f32).sqrt().recip();

    let plan = plan_with_sparse_residuals(ResidualPlannerInput {
        keys: &keys,
        values: &values,
        token_count: scenario.token_count,
        queries: &queries,
        query_count: scenario.query_count,
        maximum_key_basis: &maximum_key_basis,
        maximum_value_basis: &maximum_value_basis,
        maximum_key_slots_per_token: scenario.maximum_key_slots_per_token,
        maximum_value_slots_per_token: scenario.maximum_value_slots_per_token,
        budget_bytes,
        key_target_relative_root_mean_square: scenario.key_target_relative_root_mean_square,
        value_target_relative_root_mean_square: scenario.value_target_relative_root_mean_square,
        attention_target_max_absolute: scenario.attention_target_max_absolute,
        scale,
    })?;

    Ok(ResidualScenarioReport {
        scenario: scenario.clone(),
        plan,
    })
}

/// Returns the deterministic 12-scenario Phase 5 suite.
#[must_use]
pub fn standard_scenarios() -> Vec<ResidualScenario> {
    let dimensions = [16_usize, 32, 64];
    let variants = [
        (0_usize, 0_usize, 0.0_f32, 45_u64),
        (1, 1, 0.05, 50),
        (2, 1, 0.10, 55),
        (2, 3, 0.20, 65),
    ];
    let mut scenarios = Vec::with_capacity(dimensions.len() * variants.len());

    for (dimension_index, dimension) in dimensions.into_iter().enumerate()
    {
        for (variant_index, (key_slots, value_slots, amplitude, budget_percent)) in
            variants.into_iter().enumerate()
        {
            let intrinsic_key_rank = 2 + (dimension_index % 2);
            let intrinsic_value_rank = 2 + ((dimension_index + 1) % 2);
            let maximum_key_rank = intrinsic_key_rank + 2;
            let maximum_value_rank = intrinsic_value_rank + 2;
            let maximum_key_slots_per_token = key_slots.max(3);
            let maximum_value_slots_per_token = value_slots.max(3);
            let seed = 0xE1A5_7500_0000_0000_u64
                ^ ((dimension as u64) << 32)
                ^ ((variant_index as u64) << 24)
                ^ ((intrinsic_key_rank as u64) << 16)
                ^ ((intrinsic_value_rank as u64) << 8)
                ^ budget_percent;

            scenarios.push(ResidualScenario {
                token_count: 20,
                head_dimension: dimension,
                query_count: 4,
                intrinsic_key_rank,
                intrinsic_value_rank,
                injected_key_slots_per_token: key_slots,
                injected_value_slots_per_token: value_slots,
                maximum_key_rank,
                maximum_value_rank,
                maximum_key_slots_per_token,
                maximum_value_slots_per_token,
                budget_numerator: budget_percent,
                budget_denominator: 100,
                residual_amplitude: amplitude,
                signal_amplitude: 1.0,
                key_target_relative_root_mean_square: 1.0e-5,
                value_target_relative_root_mean_square: 1.0e-5,
                attention_target_max_absolute: 2.0e-5,
                seed,
            });
        }
    }

    scenarios
}

/// Runs the deterministic 12-scenario Phase 5 suite.
pub fn run_standard_suite() -> Result<Vec<ResidualScenarioReport>, ResidualError> {
    standard_scenarios()
        .iter()
        .map(run_residual_scenario)
        .collect()
}

/// Serializes Phase 5 reports as stable newline-terminated CSV.
#[must_use]
pub fn suite_to_csv(reports: &[ResidualScenarioReport]) -> String {
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

fn validate_evaluation_input(input: ResidualEvaluationInput<'_>) -> Result<(), ResidualError> {
    require_non_zero("token_count", input.token_count)?;
    require_non_zero("query_count", input.query_count)?;

    if input.key_basis.dimension() != input.value_basis.dimension()
    {
        return Err(ResidualError::InvalidBufferLength {
            name: "value_basis_dimension",
            expected: input.key_basis.dimension(),
            actual: input.value_basis.dimension(),
        });
    }
    let dimension = input.key_basis.dimension();
    require_encodable_dimension(dimension)?;
    let expected_samples = checked_usize_mul(input.token_count, dimension)?;
    let expected_queries = checked_usize_mul(input.query_count, dimension)?;
    require_buffer_length("keys", input.keys, expected_samples)?;
    require_buffer_length("values", input.values, expected_samples)?;
    require_buffer_length("queries", input.queries, expected_queries)?;

    if input.key_slots_per_token > dimension
    {
        return Err(ResidualError::InvalidSlots {
            slots: input.key_slots_per_token,
            maximum: dimension,
        });
    }
    if input.value_slots_per_token > dimension
    {
        return Err(ResidualError::InvalidSlots {
            slots: input.value_slots_per_token,
            maximum: dimension,
        });
    }
    if !input.scale.is_finite() || input.scale <= 0.0
    {
        return Err(ResidualError::InvalidThreshold {
            name: "scale",
            value: f64::from(input.scale),
        });
    }
    Ok(())
}

fn validate_planner_input(input: ResidualPlannerInput<'_>) -> Result<(), ResidualError> {
    require_non_zero("token_count", input.token_count)?;
    require_non_zero("query_count", input.query_count)?;
    if input.maximum_key_basis.dimension() != input.maximum_value_basis.dimension()
    {
        return Err(ResidualError::InvalidBufferLength {
            name: "value_basis_dimension",
            expected: input.maximum_key_basis.dimension(),
            actual: input.maximum_value_basis.dimension(),
        });
    }
    let dimension = input.maximum_key_basis.dimension();
    let expected_samples = checked_usize_mul(input.token_count, dimension)?;
    let expected_queries = checked_usize_mul(input.query_count, dimension)?;
    require_buffer_length("keys", input.keys, expected_samples)?;
    require_buffer_length("values", input.values, expected_samples)?;
    require_buffer_length("queries", input.queries, expected_queries)?;

    if input.maximum_key_slots_per_token > dimension
    {
        return Err(ResidualError::InvalidSlots {
            slots: input.maximum_key_slots_per_token,
            maximum: dimension,
        });
    }
    if input.maximum_value_slots_per_token > dimension
    {
        return Err(ResidualError::InvalidSlots {
            slots: input.maximum_value_slots_per_token,
            maximum: dimension,
        });
    }
    if input.budget_bytes == 0
    {
        return Err(ResidualError::ZeroField {
            field: "budget_bytes",
        });
    }
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
            return Err(ResidualError::InvalidThreshold { name, value });
        }
    }
    if !input.scale.is_finite() || input.scale <= 0.0
    {
        return Err(ResidualError::InvalidThreshold {
            name: "scale",
            value: f64::from(input.scale),
        });
    }
    Ok(())
}

fn reconstruction_metrics_with_residuals(
    samples: &[f32],
    token_count: usize,
    representation: &ResidualRepresentation,
) -> Result<ReconstructionMetrics, ResidualError> {
    let expected = checked_usize_mul(token_count, representation.basis.dimension())?;
    require_buffer_length("samples", samples, expected)?;
    let mut reconstruction = vec![0.0_f32; representation.basis.dimension()];
    let mut elements = 0_u64;
    let mut maximum = 0.0_f64;
    let mut absolute_sum = 0.0_f64;
    let mut residual_squared_sum = 0.0_f64;
    let mut reference_squared_sum = 0.0_f64;

    for (token_index, sample) in samples
        .chunks_exact(representation.basis.dimension())
        .enumerate()
    {
        representation.basis.reconstruct_into(
            representation.coefficients_for(token_index)?,
            &mut reconstruction,
        )?;
        representation
            .residuals
            .add_scaled_to(token_index, 1.0, &mut reconstruction)?;

        for (&reference, &candidate) in sample.iter().zip(reconstruction.iter())
        {
            let reference = f64::from(reference);
            let candidate = f64::from(candidate);
            let error = candidate - reference;
            let absolute = error.abs();
            elements = elements
                .checked_add(1)
                .ok_or(ResidualError::ArithmeticOverflow)?;
            maximum = maximum.max(absolute);
            absolute_sum += absolute;
            residual_squared_sum += error * error;
            reference_squared_sum += reference * reference;
        }
    }

    let count = elements as f64;
    let residual_mean_square = residual_squared_sum / count;
    let reference_mean_square = reference_squared_sum / count;
    let relative_root_mean_square = if reference_mean_square <= ENERGY_FLOOR
    {
        if residual_mean_square <= ENERGY_FLOOR
        {
            0.0
        }
        else
        {
            f64::INFINITY
        }
    }
    else
    {
        (residual_mean_square / reference_mean_square).sqrt()
    };
    let retained_energy = if reference_squared_sum <= ENERGY_FLOOR
    {
        1.0
    }
    else
    {
        (1.0 - residual_squared_sum / reference_squared_sum).clamp(0.0, 1.0)
    };

    Ok(ReconstructionMetrics {
        vectors: token_count,
        elements,
        max_absolute: maximum,
        mean_absolute: absolute_sum / count,
        root_mean_square: residual_mean_square.sqrt(),
        relative_root_mean_square,
        retained_energy,
    })
}

fn attention_metrics_with_residuals(
    input: ResidualEvaluationInput<'_>,
    keys: &ResidualRepresentation,
    values: &ResidualRepresentation,
) -> Result<ProjectedAttentionMetrics, ResidualError> {
    let dimension = input.key_basis.dimension();
    let mut dense_output = vec![0.0_f32; dimension];
    let mut residual_output = vec![0.0_f32; dimension];
    let mut dense_scores = vec![0.0_f32; input.token_count];
    let mut residual_scores = vec![0.0_f32; input.token_count];
    let mut key_query = vec![0.0_f32; keys.basis.rank()];
    let mut value_accumulator = vec![0.0_f32; values.basis.rank()];
    let mut elements = 0_u64;
    let mut maximum = 0.0_f64;
    let mut absolute_sum = 0.0_f64;
    let mut squared_sum = 0.0_f64;
    let mut fingerprint = FNV_OFFSET_BASIS;

    for query in input.queries.chunks_exact(dimension)
    {
        dense_attention(
            input.keys,
            input.values,
            input.token_count,
            dimension,
            query,
            input.scale,
            &mut dense_scores,
            &mut dense_output,
        )?;
        residual_attention(
            keys,
            values,
            query,
            input.scale,
            &mut key_query,
            &mut value_accumulator,
            &mut residual_scores,
            &mut residual_output,
        )?;

        for (&reference, &candidate) in dense_output.iter().zip(residual_output.iter())
        {
            let error = f64::from(candidate) - f64::from(reference);
            let absolute = error.abs();
            elements = elements
                .checked_add(1)
                .ok_or(ResidualError::ArithmeticOverflow)?;
            maximum = maximum.max(absolute);
            absolute_sum += absolute;
            squared_sum += error * error;
            fingerprint = fnv_u32(fingerprint, candidate.to_bits());
        }
    }

    let count = elements as f64;
    Ok(ProjectedAttentionMetrics {
        elements,
        max_absolute: maximum,
        mean_absolute: absolute_sum / count,
        root_mean_square: (squared_sum / count).sqrt(),
        output_fingerprint: fingerprint,
    })
}

#[allow(clippy::too_many_arguments)]
fn dense_attention(
    keys: &[f32],
    values: &[f32],
    token_count: usize,
    dimension: usize,
    query: &[f32],
    scale: f32,
    scores: &mut [f32],
    output: &mut [f32],
) -> Result<(), ResidualError> {
    require_buffer_length("query", query, dimension)?;
    require_buffer_length("scores", scores, token_count)?;
    require_buffer_length("output", output, dimension)?;

    for (score, key) in scores.iter_mut().zip(keys.chunks_exact(dimension))
    {
        *score = dot_f32(query, key) * scale;
    }
    softmax_in_place(scores)?;
    output.fill(0.0);

    for (&weight, value) in scores.iter().zip(values.chunks_exact(dimension))
    {
        for (destination, source) in output.iter_mut().zip(value.iter())
        {
            *destination += weight * *source;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn residual_attention(
    keys: &ResidualRepresentation,
    values: &ResidualRepresentation,
    query: &[f32],
    scale: f32,
    key_query: &mut [f32],
    value_accumulator: &mut [f32],
    scores: &mut [f32],
    output: &mut [f32],
) -> Result<(), ResidualError> {
    keys.basis.project_into(query, key_query)?;
    value_accumulator.fill(0.0);
    output.fill(0.0);

    for (token_index, score) in scores.iter_mut().enumerate()
    {
        *score = (dot_f32(key_query, keys.coefficients_for(token_index)?)
            + keys.residuals.dot(token_index, query)?)
            * scale;
    }
    softmax_in_place(scores)?;

    for (token_index, &weight) in scores.iter().enumerate()
    {
        for (destination, &coefficient) in value_accumulator
            .iter_mut()
            .zip(values.coefficients_for(token_index)?.iter())
        {
            *destination += weight * coefficient;
        }
        values
            .residuals
            .add_scaled_to(token_index, weight, output)?;
    }

    let mut base_output = vec![0.0_f32; output.len()];
    values
        .basis
        .reconstruct_into(value_accumulator, &mut base_output)?;
    for (destination, base) in output.iter_mut().zip(base_output.iter())
    {
        *destination += *base;
    }
    Ok(())
}

fn softmax_in_place(scores: &mut [f32]) -> Result<(), ResidualError> {
    require_non_zero("scores", scores.len())?;
    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for score in scores.iter_mut()
    {
        *score = (*score - maximum).exp();
        sum += *score;
    }
    if !sum.is_finite() || sum <= 0.0
    {
        return Err(ResidualError::InvalidThreshold {
            name: "softmax_sum",
            value: f64::from(sum),
        });
    }
    let reciprocal = sum.recip();
    for score in scores
    {
        *score *= reciprocal;
    }
    Ok(())
}

fn compare_candidates(left: &ResidualCandidate, right: &ResidualCandidate) -> Ordering {
    match (left.quality_guard_met, right.quality_guard_met)
    {
        (true, false) => return Ordering::Less,
        (false, true) => return Ordering::Greater,
        _ =>
        {},
    }

    if left.quality_guard_met
    {
        left.storage
            .total_bytes
            .cmp(&right.storage.total_bytes)
            .then_with(|| total_complexity(left).cmp(&total_complexity(right)))
            .then_with(|| ordered_f64(left.worst_target_ratio, right.worst_target_ratio))
            .then_with(|| candidate_tuple(left).cmp(&candidate_tuple(right)))
    }
    else
    {
        ordered_f64(left.worst_target_ratio, right.worst_target_ratio)
            .then_with(|| left.storage.total_bytes.cmp(&right.storage.total_bytes))
            .then_with(|| total_complexity(left).cmp(&total_complexity(right)))
            .then_with(|| candidate_tuple(left).cmp(&candidate_tuple(right)))
    }
}

fn candidate_tuple(candidate: &ResidualCandidate) -> (usize, usize, usize, usize) {
    (
        candidate.key_rank,
        candidate.value_rank,
        candidate.key_slots_per_token,
        candidate.value_slots_per_token,
    )
}

fn total_complexity(candidate: &ResidualCandidate) -> usize {
    candidate
        .key_rank
        .saturating_add(candidate.value_rank)
        .saturating_add(candidate.key_slots_per_token)
        .saturating_add(candidate.value_slots_per_token)
}

fn residual_pareto_frontier(candidates: &[ResidualCandidate]) -> Vec<ResidualCandidate> {
    let mut frontier: Vec<_> = candidates
        .iter()
        .filter(|candidate| {
            !candidates.iter().any(|other| {
                other.storage.total_bytes <= candidate.storage.total_bytes
                    && other.worst_target_ratio <= candidate.worst_target_ratio
                    && (other.storage.total_bytes < candidate.storage.total_bytes
                        || other.worst_target_ratio < candidate.worst_target_ratio)
            })
        })
        .cloned()
        .collect();
    frontier.sort_by(|left, right| {
        left.storage
            .total_bytes
            .cmp(&right.storage.total_bytes)
            .then_with(|| ordered_f64(left.worst_target_ratio, right.worst_target_ratio))
            .then_with(|| candidate_tuple(left).cmp(&candidate_tuple(right)))
    });
    frontier
}

fn ordered_f64(left: f64, right: f64) -> Ordering {
    left.partial_cmp(&right).unwrap_or(Ordering::Equal)
}

fn target_ratio(value: f64, target: f64) -> f64 {
    if target == 0.0
    {
        if value == 0.0 { 0.0 } else { f64::INFINITY }
    }
    else
    {
        value / target
    }
}

#[allow(clippy::too_many_arguments)]
fn generate_structured_dataset(
    rng: &mut DeterministicRng,
    token_count: usize,
    dimension: usize,
    intrinsic_rank: usize,
    residual_start: usize,
    residual_slots: usize,
    signal_amplitude: f32,
    residual_amplitude: f32,
) -> Result<Vec<f32>, ResidualError> {
    let length = checked_usize_mul(token_count, dimension)?;
    let mut samples = vec![0.0_f32; length];
    let tail = dimension - residual_start;

    if residual_slots > tail
    {
        return Err(ResidualError::InvalidSlots {
            slots: residual_slots,
            maximum: tail,
        });
    }

    for (token_index, sample) in samples.chunks_exact_mut(dimension).enumerate()
    {
        for value in sample.iter_mut().take(intrinsic_rank)
        {
            *value = rng.next_symmetric_f32(signal_amplitude);
        }
        for residual_index in 0..residual_slots
        {
            let coordinate =
                residual_start + (token_index * (residual_slots + 1) + residual_index * 3) % tail;
            let sign = if (token_index + residual_index) & 1 == 0
            {
                1.0
            }
            else
            {
                -1.0
            };
            let scale = 1.0 + residual_index as f32 * 0.25;
            sample[coordinate] += sign * residual_amplitude * scale;
        }
    }
    Ok(samples)
}

fn random_vector(rng: &mut DeterministicRng, length: usize, amplitude: f32) -> Vec<f32> {
    let mut vector = vec![0.0_f32; length];
    for value in &mut vector
    {
        *value = rng.next_symmetric_f32(amplitude);
    }
    vector
}

fn dot_f32(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(&left_value, &right_value)| left_value * right_value)
        .sum()
}

fn fnv_u32(mut hash: u64, value: u32) -> u64 {
    for byte in value.to_le_bytes()
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
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
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }

    fn next_symmetric_f32(&mut self, amplitude: f32) -> f32 {
        let mantissa = (self.next_u64() >> 40) as u32;
        let unit = mantissa as f32 / ((1_u32 << 24) - 1) as f32;
        (unit * 2.0 - 1.0) * amplitude
    }
}

fn require_non_zero(field: &'static str, value: usize) -> Result<(), ResidualError> {
    if value == 0
    {
        return Err(ResidualError::ZeroField { field });
    }
    Ok(())
}

fn require_encodable_dimension(dimension: usize) -> Result<(), ResidualError> {
    let maximum = usize::from(u16::MAX) - 1;
    if dimension > maximum
    {
        return Err(ResidualError::DimensionTooLarge { dimension, maximum });
    }
    Ok(())
}

fn require_buffer_length(
    name: &'static str,
    buffer: &[f32],
    expected: usize,
) -> Result<(), ResidualError> {
    if buffer.len() != expected
    {
        return Err(ResidualError::InvalidBufferLength {
            name,
            expected,
            actual: buffer.len(),
        });
    }
    Ok(())
}

fn checked_usize_mul(left: usize, right: usize) -> Result<usize, ResidualError> {
    left.checked_mul(right)
        .ok_or(ResidualError::ArithmeticOverflow)
}

fn checked_mul(left: u64, right: u64) -> Result<u64, ResidualError> {
    left.checked_mul(right)
        .ok_or(ResidualError::ArithmeticOverflow)
}

fn checked_add(left: u64, right: u64) -> Result<u64, ResidualError> {
    left.checked_add(right)
        .ok_or(ResidualError::ArithmeticOverflow)
}

fn to_u64(value: usize) -> Result<u64, ResidualError> {
    u64::try_from(value).map_err(|_| ResidualError::ArithmeticOverflow)
}

#[cfg(test)]
mod tests {
    use super::{
        CSV_HEADER, ResidualError, ResidualEvaluationInput, ResidualPlannerInput, ResidualScenario,
        SparseResidualMatrix, evaluate_sparse_residual_attention, plan_with_sparse_residuals,
        residual_storage_accounting, run_residual_scenario, standard_scenarios, suite_to_csv,
    };
    use crate::phase2::OrthonormalBasis;

    fn compact_residual_scenario() -> ResidualScenario {
        ResidualScenario {
            token_count: 20,
            head_dimension: 16,
            query_count: 3,
            intrinsic_key_rank: 2,
            intrinsic_value_rank: 3,
            injected_key_slots_per_token: 1,
            injected_value_slots_per_token: 2,
            maximum_key_rank: 4,
            maximum_value_rank: 5,
            maximum_key_slots_per_token: 2,
            maximum_value_slots_per_token: 3,
            budget_numerator: 60,
            budget_denominator: 100,
            residual_amplitude: 0.1,
            signal_amplitude: 1.0,
            key_target_relative_root_mean_square: 1.0e-5,
            value_target_relative_root_mean_square: 1.0e-5,
            attention_target_max_absolute: 2.0e-5,
            seed: 0xE1A5_7500_CAFE_BABE,
        }
    }

    #[test]
    fn storage_accounting_matches_closed_form() {
        let accounting = residual_storage_accounting(16, 8, 2, 3, 1, 2).unwrap();

        assert_eq!(accounting.dense_bytes, 1_024);
        assert_eq!(accounting.coefficient_bytes, 320);
        assert_eq!(accounting.basis_bytes, 160);
        assert_eq!(accounting.residual_index_bytes, 96);
        assert_eq!(accounting.residual_value_bytes, 192);
        assert_eq!(accounting.total_bytes, 768);
        assert_eq!(accounting.savings_bytes, 256);
    }

    #[test]
    fn sparse_residual_ties_use_lowest_coordinate() {
        let basis = OrthonormalBasis::identity(4, 1).unwrap();
        let samples = [1.0_f32, 2.0, -2.0, 0.0];
        let residuals = SparseResidualMatrix::from_samples(&samples, 1, &basis, 1).unwrap();

        assert_eq!(residuals.indices(), &[1]);
        assert_eq!(residuals.values(), &[2.0]);
    }

    #[test]
    fn zero_slots_reserve_no_residual_payload() {
        let basis = OrthonormalBasis::identity(2, 1).unwrap();
        let samples = [1.0_f32, 3.0, 2.0, 4.0];
        let residuals = SparseResidualMatrix::from_samples(&samples, 2, &basis, 0).unwrap();

        assert_eq!(residuals.slots_per_token(), 0);
        assert!(residuals.indices().is_empty());
        assert!(residuals.values().is_empty());
    }

    #[test]
    fn sparse_residual_attention_round_trips_structured_data() {
        let key_basis = OrthonormalBasis::identity(4, 1).unwrap();
        let value_basis = OrthonormalBasis::identity(4, 1).unwrap();
        let keys = [1.0_f32, 0.5, 0.0, 0.0, 2.0, 0.0, -0.25, 0.0];
        let values = [0.5_f32, 0.0, 0.75, 0.0, -1.0, 0.0, 0.0, -0.5];
        let queries = [0.25_f32, 1.0, -0.5, 0.75];
        let evaluation = evaluate_sparse_residual_attention(ResidualEvaluationInput {
            keys: &keys,
            values: &values,
            token_count: 2,
            queries: &queries,
            query_count: 1,
            key_basis: &key_basis,
            value_basis: &value_basis,
            key_slots_per_token: 1,
            value_slots_per_token: 1,
            scale: 0.5,
        })
        .unwrap();

        assert!(evaluation.key_reconstruction.relative_root_mean_square <= 1.0e-7);
        assert!(evaluation.value_reconstruction.relative_root_mean_square <= 1.0e-7);
        assert!(evaluation.attention.max_absolute <= 1.0e-6);
    }

    #[test]
    fn budget_below_rank_one_baseline_is_rejected() {
        let basis = OrthonormalBasis::identity(2, 2).unwrap();
        let keys = [1.0_f32, 0.0, 0.0, 1.0];
        let values = keys;
        let queries = [1.0_f32, 0.0];
        let minimum = residual_storage_accounting(2, 2, 1, 1, 0, 0)
            .unwrap()
            .total_bytes;

        let error = plan_with_sparse_residuals(ResidualPlannerInput {
            keys: &keys,
            values: &values,
            token_count: 2,
            queries: &queries,
            query_count: 1,
            maximum_key_basis: &basis,
            maximum_value_basis: &basis,
            maximum_key_slots_per_token: 1,
            maximum_value_slots_per_token: 1,
            budget_bytes: minimum - 1,
            key_target_relative_root_mean_square: 1.0,
            value_target_relative_root_mean_square: 1.0,
            attention_target_max_absolute: 1.0,
            scale: 1.0,
        })
        .unwrap_err();

        assert_eq!(
            error,
            ResidualError::BudgetBelowMinimum {
                budget_bytes: minimum - 1,
                minimum_bytes: minimum,
            }
        );
    }

    #[test]
    fn residual_plan_improves_structured_baseline() {
        let report = run_residual_scenario(&compact_residual_scenario()).unwrap();

        assert!(!report.plan.baseline.quality_guard_met);
        assert!(report.plan.selected.quality_guard_met);
        assert!(report.residual_improved_quality());
        assert!(report.plan.selected.key_slots_per_token > 0);
        assert!(report.plan.selected.value_slots_per_token > 0);
    }

    #[test]
    fn selected_candidate_never_exceeds_budget() {
        for scenario in standard_scenarios()
        {
            let report = run_residual_scenario(&scenario).unwrap();
            assert!(report.plan.selected.storage.total_bytes <= report.plan.budget_bytes);
        }
    }

    #[test]
    fn scenario_execution_is_bit_deterministic() {
        let scenario = compact_residual_scenario();
        let first = run_residual_scenario(&scenario).unwrap();
        let second = run_residual_scenario(&scenario).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.to_csv_row(), second.to_csv_row());
    }

    #[test]
    fn standard_suite_is_complete_and_residual_effective() {
        let scenarios = standard_scenarios();
        assert_eq!(scenarios.len(), 12);
        let reports: Vec<_> = scenarios
            .iter()
            .map(|scenario| run_residual_scenario(scenario).unwrap())
            .collect();
        let improved = reports
            .iter()
            .filter(|report| report.residual_improved_quality())
            .count();
        let selected_guard = reports
            .iter()
            .filter(|report| report.plan.selected.quality_guard_met)
            .count();

        assert!(improved >= 9);
        assert_eq!(selected_guard, reports.len());
    }

    #[test]
    fn csv_export_has_expected_shape() {
        let reports: Vec<_> = standard_scenarios()
            .iter()
            .map(|scenario| run_residual_scenario(scenario).unwrap())
            .collect();
        let csv = suite_to_csv(&reports);
        let lines: Vec<_> = csv.lines().collect();

        assert_eq!(lines.len(), 13);
        assert_eq!(lines[0], CSV_HEADER);
        assert_eq!(lines[0].split(',').count(), 41);
        assert!(lines[1..].iter().all(|row| row.split(',').count() == 41));
    }
}

//! Phase 2 dense-to-latent projection for the Elastic Latent KV experiment.
//!
//! Phase 2 introduces deterministic basis construction from dense key/value
//! samples, fixed-rank projection, reconstruction metrics and the smallest-rank
//! selection that satisfies a declared relative-RMS target. It deliberately
//! does not introduce adaptive per-token ranks, quantization or production
//! integration.

use crate::{AttentionScratch, CacheError, DenseKvCache, FixedRankLatentCache};
use core::fmt;

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const ENERGY_FLOOR: f64 = 1.0e-30;

/// Errors returned by deterministic Phase 2 projection routines.
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectionError {
    /// A required dimension, count or rank was zero.
    ZeroField {
        /// Name of the zero-valued field.
        field: &'static str,
    },
    /// A flat matrix or sample buffer has an unexpected length.
    InvalidBufferLength {
        /// Human-readable buffer name.
        name: &'static str,
        /// Required number of elements.
        expected: usize,
        /// Supplied number of elements.
        actual: usize,
    },
    /// A rank exceeds the dense dimension or available basis rank.
    InvalidRank {
        /// Requested rank.
        rank: usize,
        /// Maximum accepted rank.
        maximum: usize,
    },
    /// A scalar threshold is non-finite or outside its accepted range.
    InvalidThreshold {
        /// Human-readable threshold name.
        name: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// The dataset contains no direction above the declared norm tolerance.
    DegenerateDataset,
    /// An element-count computation overflowed `usize`.
    ArithmeticOverflow,
    /// A Phase 0 cache operation failed.
    Cache(CacheError),
}

impl fmt::Display for ProjectionError {
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
            Self::DegenerateDataset => write!(
                formatter,
                "dataset contains no direction above the norm tolerance"
            ),
            Self::ArithmeticOverflow => write!(formatter, "projection arithmetic overflow"),
            Self::Cache(error) => write!(formatter, "cache error: {error}"),
        }
    }
}

impl std::error::Error for ProjectionError {}

impl From<CacheError> for ProjectionError {
    fn from(error: CacheError) -> Self {
        Self::Cache(error)
    }
}

/// Row-major orthonormal basis with shape `[dimension, rank]`.
#[derive(Debug, Clone, PartialEq)]
pub struct OrthonormalBasis {
    dimension: usize,
    rank: usize,
    row_major: Vec<f32>,
}

impl OrthonormalBasis {
    /// Builds a basis by deterministic residual-pivoted modified Gram-Schmidt.
    ///
    /// Samples are row-major with shape `[sample_count, dimension]`. At every
    /// rank, the lowest-index sample with maximum residual energy is selected.
    /// The resulting basis prefixes are therefore deterministic on the same
    /// target and build.
    pub fn from_greedy_samples(
        samples: &[f32],
        sample_count: usize,
        dimension: usize,
        max_rank: usize,
        norm_tolerance: f64,
    ) -> Result<Self, ProjectionError> {
        require_non_zero("sample_count", sample_count)?;
        require_non_zero("dimension", dimension)?;
        require_non_zero("max_rank", max_rank)?;

        if max_rank > dimension
        {
            return Err(ProjectionError::InvalidRank {
                rank: max_rank,
                maximum: dimension,
            });
        }

        if !norm_tolerance.is_finite() || norm_tolerance <= 0.0
        {
            return Err(ProjectionError::InvalidThreshold {
                name: "norm_tolerance",
                value: norm_tolerance,
            });
        }

        let expected = checked_len(sample_count, dimension)?;
        require_buffer_length("samples", samples, expected)?;

        let effective_max_rank = max_rank.min(sample_count);
        let tolerance_squared = norm_tolerance * norm_tolerance;
        let mut columns: Vec<Vec<f64>> = Vec::with_capacity(effective_max_rank);
        let mut residual = vec![0.0_f64; dimension];
        let mut best_residual = vec![0.0_f64; dimension];

        for _ in 0..effective_max_rank
        {
            let mut best_index = 0_usize;
            let mut best_norm_squared = -1.0_f64;

            for sample_index in 0..sample_count
            {
                residual_for_sample(samples, sample_index, dimension, &columns, &mut residual);

                let norm_squared = dot_f64(&residual, &residual);
                if norm_squared > best_norm_squared
                {
                    best_norm_squared = norm_squared;
                    best_index = sample_index;
                    best_residual.copy_from_slice(&residual);
                }
            }

            if best_norm_squared <= tolerance_squared
            {
                break;
            }

            residual_for_sample(samples, best_index, dimension, &columns, &mut best_residual);

            // A second orthogonalization pass reduces loss of orthogonality
            // while preserving deterministic operation order.
            orthogonalize_in_place(&mut best_residual, &columns);

            let norm = dot_f64(&best_residual, &best_residual).sqrt();
            if norm <= norm_tolerance
            {
                break;
            }

            for value in &mut best_residual
            {
                *value /= norm;
            }
            canonicalize_sign(&mut best_residual);
            columns.push(best_residual.clone());
        }

        if columns.is_empty()
        {
            return Err(ProjectionError::DegenerateDataset);
        }

        Self::from_columns(dimension, &columns)
    }

    /// Creates an identity basis of the requested rank.
    pub fn identity(dimension: usize, rank: usize) -> Result<Self, ProjectionError> {
        require_non_zero("dimension", dimension)?;
        require_non_zero("rank", rank)?;

        if rank > dimension
        {
            return Err(ProjectionError::InvalidRank {
                rank,
                maximum: dimension,
            });
        }

        let length = checked_len(dimension, rank)?;
        let mut row_major = vec![0.0_f32; length];
        for (diagonal, row) in row_major.chunks_exact_mut(rank).take(rank).enumerate()
        {
            row[diagonal] = 1.0;
        }

        Ok(Self {
            dimension,
            rank,
            row_major,
        })
    }

    /// Returns the dense vector dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the number of basis vectors.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Returns the row-major `[dimension, rank]` basis buffer.
    #[must_use]
    pub fn as_row_major(&self) -> &[f32] {
        &self.row_major
    }

    /// Returns an owned row-major basis buffer suitable for Phase 0 caches.
    #[must_use]
    pub fn to_row_major_vec(&self) -> Vec<f32> {
        self.row_major.clone()
    }

    /// Returns a nested prefix basis with the requested rank.
    pub fn prefix(&self, rank: usize) -> Result<Self, ProjectionError> {
        require_non_zero("rank", rank)?;

        if rank > self.rank
        {
            return Err(ProjectionError::InvalidRank {
                rank,
                maximum: self.rank,
            });
        }

        let length = checked_len(self.dimension, rank)?;
        let mut row_major = vec![0.0_f32; length];

        for (source, destination) in self
            .row_major
            .chunks_exact(self.rank)
            .zip(row_major.chunks_exact_mut(rank))
        {
            destination.copy_from_slice(&source[..rank]);
        }

        Ok(Self {
            dimension: self.dimension,
            rank,
            row_major,
        })
    }

    /// Projects one dense vector into latent coefficients.
    pub fn project_into(
        &self,
        vector: &[f32],
        coefficients: &mut [f32],
    ) -> Result<(), ProjectionError> {
        require_buffer_length("vector", vector, self.dimension)?;
        require_buffer_length("coefficients", coefficients, self.rank)?;

        coefficients.fill(0.0);
        for (basis_row, vector_value) in self
            .row_major
            .chunks_exact(self.rank)
            .zip(vector.iter().copied())
        {
            for (coefficient, basis_value) in coefficients.iter_mut().zip(basis_row.iter())
            {
                *coefficient += basis_value * vector_value;
            }
        }

        Ok(())
    }

    /// Reconstructs one dense vector from latent coefficients.
    pub fn reconstruct_into(
        &self,
        coefficients: &[f32],
        vector: &mut [f32],
    ) -> Result<(), ProjectionError> {
        require_buffer_length("coefficients", coefficients, self.rank)?;
        require_buffer_length("vector", vector, self.dimension)?;

        for (row, output) in vector.iter_mut().enumerate()
        {
            let offset = row * self.rank;
            *output = self.row_major[offset..offset + self.rank]
                .iter()
                .zip(coefficients.iter())
                .map(|(basis, coefficient)| basis * coefficient)
                .sum();
        }

        Ok(())
    }

    fn from_columns(dimension: usize, columns: &[Vec<f64>]) -> Result<Self, ProjectionError> {
        let rank = columns.len();
        let length = checked_len(dimension, rank)?;
        let mut row_major = vec![0.0_f32; length];

        for (row_index, row) in row_major.chunks_exact_mut(rank).enumerate()
        {
            for (column_index, destination) in row.iter_mut().enumerate()
            {
                debug_assert_eq!(columns[column_index].len(), dimension);
                *destination = columns[column_index][row_index] as f32;
            }
        }

        Ok(Self {
            dimension,
            rank,
            row_major,
        })
    }
}

/// Aggregate reconstruction error over a dense sample matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct ReconstructionMetrics {
    /// Number of dense vectors evaluated.
    pub vectors: usize,
    /// Number of scalar elements evaluated.
    pub elements: u64,
    /// Maximum absolute scalar reconstruction error.
    pub max_absolute: f64,
    /// Mean absolute scalar reconstruction error.
    pub mean_absolute: f64,
    /// Root-mean-square scalar reconstruction error.
    pub root_mean_square: f64,
    /// Residual RMS divided by reference RMS.
    pub relative_root_mean_square: f64,
    /// Fraction of dense squared energy retained after reconstruction.
    pub retained_energy: f64,
}

/// Smallest deterministic fixed-rank selection satisfying a target.
#[derive(Debug, Clone, PartialEq)]
pub struct RankSelection {
    /// Selected prefix basis.
    pub basis: OrthonormalBasis,
    /// Requested maximum rank.
    pub maximum_rank: usize,
    /// Relative-RMS target.
    pub target_relative_root_mean_square: f64,
    /// Reconstruction metrics at the selected rank.
    pub metrics: ReconstructionMetrics,
    /// Whether the selected rank satisfies the declared target.
    pub target_met: bool,
}

/// Selects the smallest nested basis prefix meeting a relative-RMS target.
pub fn select_fixed_rank(
    samples: &[f32],
    sample_count: usize,
    dimension: usize,
    maximum_rank: usize,
    target_relative_root_mean_square: f64,
    norm_tolerance: f64,
) -> Result<RankSelection, ProjectionError> {
    if !target_relative_root_mean_square.is_finite() || target_relative_root_mean_square < 0.0
    {
        return Err(ProjectionError::InvalidThreshold {
            name: "target_relative_root_mean_square",
            value: target_relative_root_mean_square,
        });
    }

    let maximum_basis = OrthonormalBasis::from_greedy_samples(
        samples,
        sample_count,
        dimension,
        maximum_rank,
        norm_tolerance,
    )?;

    let available_rank = maximum_basis.rank();
    let mut selected_basis = maximum_basis.prefix(available_rank)?;
    let mut selected_metrics = reconstruction_metrics(samples, sample_count, &selected_basis)?;
    let mut target_met =
        selected_metrics.relative_root_mean_square <= target_relative_root_mean_square;

    for rank in 1..=available_rank
    {
        let basis = maximum_basis.prefix(rank)?;
        let metrics = reconstruction_metrics(samples, sample_count, &basis)?;

        if metrics.relative_root_mean_square <= target_relative_root_mean_square
        {
            selected_basis = basis;
            selected_metrics = metrics;
            target_met = true;
            break;
        }
    }

    Ok(RankSelection {
        basis: selected_basis,
        maximum_rank,
        target_relative_root_mean_square,
        metrics: selected_metrics,
        target_met,
    })
}

/// Computes reconstruction metrics for a dense row-major sample matrix.
pub fn reconstruction_metrics(
    samples: &[f32],
    sample_count: usize,
    basis: &OrthonormalBasis,
) -> Result<ReconstructionMetrics, ProjectionError> {
    require_non_zero("sample_count", sample_count)?;
    let expected = checked_len(sample_count, basis.dimension())?;
    require_buffer_length("samples", samples, expected)?;

    let mut coefficients = vec![0.0_f32; basis.rank()];
    let mut reconstruction = vec![0.0_f32; basis.dimension()];
    let mut elements = 0_u64;
    let mut maximum = 0.0_f64;
    let mut absolute_sum = 0.0_f64;
    let mut residual_squared_sum = 0.0_f64;
    let mut reference_squared_sum = 0.0_f64;

    for sample in samples.chunks_exact(basis.dimension())
    {
        basis.project_into(sample, &mut coefficients)?;
        basis.reconstruct_into(&coefficients, &mut reconstruction)?;

        for (reference, candidate) in sample.iter().zip(reconstruction.iter())
        {
            let reference = f64::from(*reference);
            let candidate = f64::from(*candidate);
            let residual = candidate - reference;
            let absolute = residual.abs();

            elements = elements
                .checked_add(1)
                .ok_or(ProjectionError::ArithmeticOverflow)?;
            maximum = maximum.max(absolute);
            absolute_sum += absolute;
            residual_squared_sum += residual * residual;
            reference_squared_sum += reference * reference;
        }
    }

    let element_count = elements as f64;
    let reference_mean_square = reference_squared_sum / element_count;
    let residual_mean_square = residual_squared_sum / element_count;
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
        if residual_squared_sum <= ENERGY_FLOOR
        {
            1.0
        }
        else
        {
            0.0
        }
    }
    else
    {
        (1.0 - residual_squared_sum / reference_squared_sum).clamp(0.0, 1.0)
    };

    Ok(ReconstructionMetrics {
        vectors: sample_count,
        elements,
        max_absolute: maximum,
        mean_absolute: absolute_sum / element_count,
        root_mean_square: residual_mean_square.sqrt(),
        relative_root_mean_square,
        retained_energy,
    })
}

/// Attention error after projecting dense keys and values into selected bases.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedAttentionMetrics {
    /// Number of scalar output elements evaluated.
    pub elements: u64,
    /// Maximum absolute dense-versus-latent output error.
    pub max_absolute: f64,
    /// Mean absolute dense-versus-latent output error.
    pub mean_absolute: f64,
    /// Root-mean-square dense-versus-latent output error.
    pub root_mean_square: f64,
    /// Stable FNV-1a fingerprint of latent outputs.
    pub output_fingerprint: u64,
}

/// Inputs for projected dense-versus-latent attention evaluation.
#[derive(Debug, Clone, Copy)]
pub struct ProjectedAttentionInput<'a> {
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
    /// Selected key basis.
    pub key_basis: &'a OrthonormalBasis,
    /// Selected value basis.
    pub value_basis: &'a OrthonormalBasis,
    /// Positive finite attention scale.
    pub scale: f32,
}

/// Evaluates dense attention against projected reconstruction-free attention.
pub fn evaluate_projected_attention(
    input: ProjectedAttentionInput<'_>,
) -> Result<ProjectedAttentionMetrics, ProjectionError> {
    require_non_zero("token_count", input.token_count)?;
    require_non_zero("query_count", input.query_count)?;

    if input.key_basis.dimension() != input.value_basis.dimension()
    {
        return Err(ProjectionError::InvalidBufferLength {
            name: "value basis dimension",
            expected: input.key_basis.dimension(),
            actual: input.value_basis.dimension(),
        });
    }

    let dimension = input.key_basis.dimension();
    let expected_samples = checked_len(input.token_count, dimension)?;
    let expected_queries = checked_len(input.query_count, dimension)?;
    require_buffer_length("keys", input.keys, expected_samples)?;
    require_buffer_length("values", input.values, expected_samples)?;
    require_buffer_length("queries", input.queries, expected_queries)?;

    if !input.scale.is_finite() || input.scale <= 0.0
    {
        return Err(ProjectionError::InvalidThreshold {
            name: "scale",
            value: f64::from(input.scale),
        });
    }

    let mut dense = DenseKvCache::new(input.token_count, dimension)?;
    let mut latent = FixedRankLatentCache::new(
        input.token_count,
        dimension,
        input.key_basis.rank(),
        input.value_basis.rank(),
        input.key_basis.to_row_major_vec(),
        input.value_basis.to_row_major_vec(),
    )?;

    let mut key_coefficients = vec![0.0_f32; input.key_basis.rank()];
    let mut value_coefficients = vec![0.0_f32; input.value_basis.rank()];

    for token_index in 0..input.token_count
    {
        let offset = token_index * dimension;
        let key = &input.keys[offset..offset + dimension];
        let value = &input.values[offset..offset + dimension];

        dense.append(key, value)?;
        input.key_basis.project_into(key, &mut key_coefficients)?;
        input
            .value_basis
            .project_into(value, &mut value_coefficients)?;
        latent.append_coefficients(&key_coefficients, &value_coefficients)?;
    }

    let mut dense_output = vec![0.0_f32; dimension];
    let mut latent_output = vec![0.0_f32; dimension];
    let max_rank = input.key_basis.rank().max(input.value_basis.rank());
    let mut dense_scratch = AttentionScratch::new(input.token_count, dimension, max_rank);
    let mut latent_scratch = AttentionScratch::new(input.token_count, dimension, max_rank);

    let mut elements = 0_u64;
    let mut maximum = 0.0_f64;
    let mut absolute_sum = 0.0_f64;
    let mut squared_sum = 0.0_f64;
    let mut fingerprint = FNV_OFFSET_BASIS;

    for query in input.queries.chunks_exact(dimension)
    {
        dense.attention(query, input.scale, &mut dense_output, &mut dense_scratch)?;
        latent.attention_reconstruction_free(
            query,
            input.scale,
            &mut latent_output,
            &mut latent_scratch,
        )?;

        for (reference, candidate) in dense_output.iter().zip(latent_output.iter())
        {
            let residual = f64::from(*candidate) - f64::from(*reference);
            let absolute = residual.abs();

            elements = elements
                .checked_add(1)
                .ok_or(ProjectionError::ArithmeticOverflow)?;
            maximum = maximum.max(absolute);
            absolute_sum += absolute;
            squared_sum += residual * residual;
            fingerprint = hash_f32(fingerprint, *candidate);
        }
    }

    let element_count = elements as f64;
    Ok(ProjectedAttentionMetrics {
        elements,
        max_absolute: maximum,
        mean_absolute: absolute_sum / element_count,
        root_mean_square: (squared_sum / element_count).sqrt(),
        output_fingerprint: fingerprint,
    })
}

/// Deterministic Phase 2 scenario configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionScenario {
    /// Number of dense key/value samples.
    pub token_count: usize,
    /// Dense head dimension.
    pub head_dimension: usize,
    /// Number of attention queries.
    pub query_count: usize,
    /// Intrinsic generated key rank.
    pub intrinsic_key_rank: usize,
    /// Intrinsic generated value rank.
    pub intrinsic_value_rank: usize,
    /// Maximum learned key rank.
    pub maximum_key_rank: usize,
    /// Maximum learned value rank.
    pub maximum_value_rank: usize,
    /// Relative-RMS key target.
    pub key_target_relative_root_mean_square: f64,
    /// Relative-RMS value target.
    pub value_target_relative_root_mean_square: f64,
    /// Additive dense noise amplitude.
    pub noise_amplitude: f32,
    /// Signal coefficient amplitude.
    pub signal_amplitude: f32,
    /// Deterministic seed.
    pub seed: u64,
}

impl ProjectionScenario {
    /// Validates structural and numeric constraints.
    pub fn validate(&self) -> Result<(), ProjectionError> {
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

        for rank in [
            self.intrinsic_key_rank,
            self.intrinsic_value_rank,
            self.maximum_key_rank,
            self.maximum_value_rank,
        ]
        {
            if rank > self.head_dimension
            {
                return Err(ProjectionError::InvalidRank {
                    rank,
                    maximum: self.head_dimension,
                });
            }
        }

        for rank in [self.maximum_key_rank, self.maximum_value_rank]
        {
            if rank > self.token_count
            {
                return Err(ProjectionError::InvalidRank {
                    rank,
                    maximum: self.token_count,
                });
            }
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
        ]
        {
            if !value.is_finite() || value < 0.0
            {
                return Err(ProjectionError::InvalidThreshold { name, value });
            }
        }

        if !self.noise_amplitude.is_finite() || self.noise_amplitude < 0.0
        {
            return Err(ProjectionError::InvalidThreshold {
                name: "noise_amplitude",
                value: f64::from(self.noise_amplitude),
            });
        }

        if !self.signal_amplitude.is_finite() || self.signal_amplitude <= 0.0
        {
            return Err(ProjectionError::InvalidThreshold {
                name: "signal_amplitude",
                value: f64::from(self.signal_amplitude),
            });
        }

        Ok(())
    }
}

/// Complete report for one deterministic Phase 2 scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionScenarioReport {
    /// Original scenario configuration.
    pub scenario: ProjectionScenario,
    /// Selected key basis and reconstruction metrics.
    pub key_selection: RankSelection,
    /// Selected value basis and reconstruction metrics.
    pub value_selection: RankSelection,
    /// Dense-versus-projected attention error.
    pub attention: ProjectedAttentionMetrics,
}

impl ProjectionScenarioReport {
    /// Serializes one stable CSV row.
    #[must_use]
    pub fn to_csv_row(&self) -> String {
        format!(
            concat!(
                "{},{},{},{},{},{},{},{},{:.9e},{:.9e},{:.9e},{:.9e},",
                "{},{},{:.9e},{:.9e},{:.9e},{},{},{:.9e},{:.9e},{:.9e},",
                "{:.9e},{:.9e},{:016x}"
            ),
            self.scenario.seed,
            self.scenario.token_count,
            self.scenario.head_dimension,
            self.scenario.query_count,
            self.scenario.intrinsic_key_rank,
            self.scenario.intrinsic_value_rank,
            self.scenario.maximum_key_rank,
            self.scenario.maximum_value_rank,
            self.scenario.key_target_relative_root_mean_square,
            self.scenario.value_target_relative_root_mean_square,
            self.scenario.noise_amplitude,
            self.scenario.signal_amplitude,
            self.key_selection.basis.rank(),
            u8::from(self.key_selection.target_met),
            self.key_selection.metrics.max_absolute,
            self.key_selection.metrics.relative_root_mean_square,
            self.key_selection.metrics.retained_energy,
            self.value_selection.basis.rank(),
            u8::from(self.value_selection.target_met),
            self.value_selection.metrics.max_absolute,
            self.value_selection.metrics.relative_root_mean_square,
            self.value_selection.metrics.retained_energy,
            self.attention.max_absolute,
            self.attention.root_mean_square,
            self.attention.output_fingerprint,
        )
    }
}

/// Stable CSV header for Phase 2 reports.
pub const CSV_HEADER: &str = concat!(
    "seed,token_count,head_dimension,query_count,intrinsic_key_rank,",
    "intrinsic_value_rank,maximum_key_rank,maximum_value_rank,",
    "key_target_relative_rms,",
    "value_target_relative_rms,noise_amplitude,signal_amplitude,",
    "selected_key_rank,key_target_met,key_reconstruction_max_absolute,",
    "key_reconstruction_relative_rms,key_retained_energy,selected_value_rank,",
    "value_target_met,value_reconstruction_max_absolute,",
    "value_reconstruction_relative_rms,value_retained_energy,",
    "attention_max_absolute,attention_rms,output_fingerprint"
);

/// Runs one deterministic Phase 2 projection and attention scenario.
pub fn run_projection_scenario(
    scenario: &ProjectionScenario,
) -> Result<ProjectionScenarioReport, ProjectionError> {
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
        checked_len(scenario.query_count, scenario.head_dimension)?,
        scenario.signal_amplitude,
    );

    let key_selection = select_fixed_rank(
        &keys,
        scenario.token_count,
        scenario.head_dimension,
        scenario.maximum_key_rank,
        scenario.key_target_relative_root_mean_square,
        1.0e-10,
    )?;
    let value_selection = select_fixed_rank(
        &values,
        scenario.token_count,
        scenario.head_dimension,
        scenario.maximum_value_rank,
        scenario.value_target_relative_root_mean_square,
        1.0e-10,
    )?;

    let scale = (scenario.head_dimension as f32).sqrt().recip();
    let attention = evaluate_projected_attention(ProjectedAttentionInput {
        keys: &keys,
        values: &values,
        token_count: scenario.token_count,
        queries: &queries,
        query_count: scenario.query_count,
        key_basis: &key_selection.basis,
        value_basis: &value_selection.basis,
        scale,
    })?;

    Ok(ProjectionScenarioReport {
        scenario: scenario.clone(),
        key_selection,
        value_selection,
        attention,
    })
}

/// Returns the deterministic 12-scenario Phase 2 suite.
#[must_use]
pub fn standard_scenarios() -> Vec<ProjectionScenario> {
    let dimensions = [8_usize, 16, 32, 64];
    let noises = [0.0_f32, 0.01, 0.05];
    let mut scenarios = Vec::with_capacity(dimensions.len() * noises.len());

    for (dimension_index, dimension) in dimensions.into_iter().enumerate()
    {
        for (noise_index, noise_amplitude) in noises.into_iter().enumerate()
        {
            let intrinsic_key_rank = 1 + ((dimension_index + noise_index) % 4).min(dimension - 1);
            let intrinsic_value_rank =
                1 + ((2 * dimension_index + noise_index + 1) % 5).min(dimension - 1);
            let maximum_key_rank = (intrinsic_key_rank + 4).min(dimension);
            let maximum_value_rank = (intrinsic_value_rank + 4).min(dimension);
            let token_count = (maximum_key_rank.max(maximum_value_rank) + 8).max(16);
            let target = if noise_amplitude == 0.0 { 1.0e-5 } else { 0.08 };
            let seed = 0xE1A5_7200_0000_0000_u64
                ^ ((dimension as u64) << 32)
                ^ ((noise_index as u64) << 24)
                ^ ((intrinsic_key_rank as u64) << 16)
                ^ ((intrinsic_value_rank as u64) << 8);

            scenarios.push(ProjectionScenario {
                token_count,
                head_dimension: dimension,
                query_count: 4,
                intrinsic_key_rank,
                intrinsic_value_rank,
                maximum_key_rank,
                maximum_value_rank,
                key_target_relative_root_mean_square: target,
                value_target_relative_root_mean_square: target,
                noise_amplitude,
                signal_amplitude: 1.0,
                seed,
            });
        }
    }

    scenarios
}

/// Runs the deterministic 12-scenario Phase 2 suite.
pub fn run_standard_suite() -> Result<Vec<ProjectionScenarioReport>, ProjectionError> {
    standard_scenarios()
        .iter()
        .map(run_projection_scenario)
        .collect()
}

/// Serializes reports as stable newline-terminated CSV.
#[must_use]
pub fn suite_to_csv(reports: &[ProjectionScenarioReport]) -> String {
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
) -> Result<OrthonormalBasis, ProjectionError> {
    let raw = random_vector(rng, checked_len(rank, dimension)?, 1.0);
    OrthonormalBasis::from_greedy_samples(&raw, rank, dimension, rank, 1.0e-10)
}

fn generate_dataset(
    rng: &mut DeterministicRng,
    sample_count: usize,
    basis: &OrthonormalBasis,
    signal_amplitude: f32,
    noise_amplitude: f32,
) -> Result<Vec<f32>, ProjectionError> {
    let length = checked_len(sample_count, basis.dimension())?;
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

fn residual_for_sample(
    samples: &[f32],
    sample_index: usize,
    dimension: usize,
    columns: &[Vec<f64>],
    residual: &mut [f64],
) {
    let offset = sample_index * dimension;
    for (destination, source) in residual
        .iter_mut()
        .zip(samples[offset..offset + dimension].iter())
    {
        *destination = f64::from(*source);
    }

    orthogonalize_in_place(residual, columns);
}

fn orthogonalize_in_place(vector: &mut [f64], columns: &[Vec<f64>]) {
    for column in columns
    {
        let coefficient = dot_f64(vector, column);
        for (value, basis_value) in vector.iter_mut().zip(column.iter())
        {
            *value -= coefficient * basis_value;
        }
    }
}

fn canonicalize_sign(vector: &mut [f64]) {
    let mut pivot_index = 0_usize;
    let mut pivot_magnitude = -1.0_f64;

    for (index, value) in vector.iter().copied().enumerate()
    {
        let magnitude = value.abs();
        if magnitude > pivot_magnitude
        {
            pivot_magnitude = magnitude;
            pivot_index = index;
        }
    }

    if vector[pivot_index].is_sign_negative()
    {
        for value in vector
        {
            *value = -*value;
        }
    }
}

fn dot_f64(left: &[f64], right: &[f64]) -> f64 {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right.iter())
        .map(|(left_value, right_value)| left_value * right_value)
        .sum()
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

fn hash_f32(mut hash: u64, value: f32) -> u64 {
    for byte in value.to_bits().to_le_bytes()
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn require_non_zero(field: &'static str, value: usize) -> Result<(), ProjectionError> {
    if value == 0
    {
        return Err(ProjectionError::ZeroField { field });
    }
    Ok(())
}

fn require_buffer_length(
    name: &'static str,
    buffer: &[f32],
    expected: usize,
) -> Result<(), ProjectionError> {
    if buffer.len() != expected
    {
        return Err(ProjectionError::InvalidBufferLength {
            name,
            expected,
            actual: buffer.len(),
        });
    }
    Ok(())
}

fn checked_len(left: usize, right: usize) -> Result<usize, ProjectionError> {
    left.checked_mul(right)
        .ok_or(ProjectionError::ArithmeticOverflow)
}

#[cfg(test)]
mod tests {
    use super::{
        CSV_HEADER, OrthonormalBasis, ProjectedAttentionInput, ProjectionError, ProjectionScenario,
        evaluate_projected_attention, reconstruction_metrics, run_projection_scenario,
        select_fixed_rank, standard_scenarios, suite_to_csv,
    };

    fn compact_exact_scenario() -> ProjectionScenario {
        ProjectionScenario {
            token_count: 12,
            head_dimension: 8,
            query_count: 3,
            intrinsic_key_rank: 2,
            intrinsic_value_rank: 3,
            maximum_key_rank: 5,
            maximum_value_rank: 6,
            key_target_relative_root_mean_square: 1.0e-5,
            value_target_relative_root_mean_square: 1.0e-5,
            noise_amplitude: 0.0,
            signal_amplitude: 1.0,
            seed: 0x1234_5678_9ABC_DEF0,
        }
    }

    #[test]
    fn identity_projection_round_trips_prefix_coordinates() {
        let basis = OrthonormalBasis::identity(4, 3).unwrap();
        let vector = [0.5, -0.25, 1.5, 0.0];
        let mut coefficients = [0.0; 3];
        let mut reconstruction = [0.0; 4];

        basis.project_into(&vector, &mut coefficients).unwrap();
        basis
            .reconstruct_into(&coefficients, &mut reconstruction)
            .unwrap();

        assert_eq!(coefficients, [0.5, -0.25, 1.5]);
        assert_eq!(reconstruction, vector);
    }

    #[test]
    fn greedy_basis_spans_exact_axis_aligned_dataset() {
        let samples = [
            2.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            -1.0, 0.5, 0.0,
        ];

        let basis = OrthonormalBasis::from_greedy_samples(&samples, 3, 3, 3, 1.0e-12).unwrap();
        let metrics = reconstruction_metrics(&samples, 3, &basis).unwrap();

        assert_eq!(basis.rank(), 2);
        assert!(metrics.max_absolute <= 1.0e-6);
        assert!(metrics.relative_root_mean_square <= 1.0e-6);
    }

    #[test]
    fn rank_selection_chooses_smallest_satisfying_prefix() {
        let samples = [
            2.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 0.05, 0.0,
        ];

        let loose = select_fixed_rank(&samples, 3, 3, 3, 0.1, 1.0e-12).unwrap();
        let strict = select_fixed_rank(&samples, 3, 3, 3, 1.0e-6, 1.0e-12).unwrap();

        assert_eq!(loose.basis.rank(), 1);
        assert!(loose.target_met);
        assert_eq!(strict.basis.rank(), 2);
        assert!(strict.target_met);
    }

    #[test]
    fn basis_construction_is_bit_deterministic() {
        let samples = [
            0.2, 0.7, -0.1, 0.4, //
            1.0, -0.3, 0.2, 0.9, //
            -0.4, 0.6, 0.8, -0.2,
        ];

        let first = OrthonormalBasis::from_greedy_samples(&samples, 3, 4, 3, 1.0e-12).unwrap();
        let second = OrthonormalBasis::from_greedy_samples(&samples, 3, 4, 3, 1.0e-12).unwrap();

        let first_bits: Vec<_> = first
            .as_row_major()
            .iter()
            .map(|value| value.to_bits())
            .collect();
        let second_bits: Vec<_> = second
            .as_row_major()
            .iter()
            .map(|value| value.to_bits())
            .collect();

        assert_eq!(first_bits, second_bits);
    }

    #[test]
    fn exact_low_rank_attention_remains_close_to_dense() {
        let report = run_projection_scenario(&compact_exact_scenario()).unwrap();

        assert!(report.key_selection.target_met);
        assert!(report.value_selection.target_met);
        assert_eq!(report.key_selection.basis.rank(), 2);
        assert_eq!(report.value_selection.basis.rank(), 3);
        assert!(report.attention.max_absolute <= 2.0e-5);
    }

    #[test]
    fn explicit_attention_evaluator_accepts_identity_bases() {
        let keys = [1.0, 0.0, 0.0, 1.0];
        let values = [0.5, 1.5, -0.5, 0.25];
        let queries = [0.75, -0.25];
        let basis = OrthonormalBasis::identity(2, 2).unwrap();

        let metrics = evaluate_projected_attention(ProjectedAttentionInput {
            keys: &keys,
            values: &values,
            token_count: 2,
            queries: &queries,
            query_count: 1,
            key_basis: &basis,
            value_basis: &basis,
            scale: 2.0_f32.sqrt().recip(),
        })
        .unwrap();

        assert!(metrics.max_absolute <= 1.0e-6);
    }

    #[test]
    fn standard_suite_is_stable_and_complete() {
        let scenarios = standard_scenarios();

        assert_eq!(scenarios.len(), 12);
        assert_eq!(scenarios.first().unwrap().head_dimension, 8);
        assert_eq!(scenarios.last().unwrap().head_dimension, 64);
        assert!(scenarios.iter().all(|scenario| scenario.validate().is_ok()));
    }

    #[test]
    fn scenario_and_csv_are_bit_deterministic() {
        let scenario = compact_exact_scenario();
        let first = run_projection_scenario(&scenario).unwrap();
        let second = run_projection_scenario(&scenario).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.to_csv_row(), second.to_csv_row());

        let csv = suite_to_csv(&[first]);
        let lines: Vec<_> = csv.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], CSV_HEADER);
    }

    #[test]
    fn invalid_shapes_are_rejected() {
        assert_eq!(
            OrthonormalBasis::from_greedy_samples(&[1.0, 2.0], 2, 2, 2, 1.0e-12),
            Err(ProjectionError::InvalidBufferLength {
                name: "samples",
                expected: 4,
                actual: 2,
            })
        );
    }
}

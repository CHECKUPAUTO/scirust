//! Phase 1 measurement harness for the Elastic Latent KV experiment.
//!
//! Phase 1 does not introduce lossy compression. It measures the fixed-rank
//! runtime established in Phase 0 so later adaptive or quantized designs can
//! be judged against deterministic memory, operation-count and error baselines.

use crate::{AttentionScratch, CacheError, DenseKvCache, FixedRankLatentCache};
use core::fmt;
use core::mem::size_of;

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const RELATIVE_ERROR_FLOOR: f64 = 1.0e-12;

/// Configuration for one deterministic Phase 1 scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentConfig {
    /// Maximum number of tokens reserved by each cache.
    pub capacity_tokens: usize,
    /// Number of tokens populated before attention is measured.
    pub token_count: usize,
    /// Dense key, value and output dimension.
    pub head_dimension: usize,
    /// Latent key rank.
    pub key_rank: usize,
    /// Latent value rank.
    pub value_rank: usize,
    /// Number of deterministic query vectors evaluated.
    pub query_count: usize,
    /// Symmetric random amplitude used for bases, coefficients and queries.
    pub amplitude: f32,
    /// Seed for the deterministic scenario generator.
    pub seed: u64,
}

impl ExperimentConfig {
    /// Validates all structural and numeric constraints.
    pub fn validate(&self) -> Result<(), ExperimentError> {
        for (field, value) in [
            ("capacity_tokens", self.capacity_tokens),
            ("token_count", self.token_count),
            ("head_dimension", self.head_dimension),
            ("key_rank", self.key_rank),
            ("value_rank", self.value_rank),
            ("query_count", self.query_count),
        ]
        {
            if value == 0
            {
                return Err(ExperimentError::ZeroField { field });
            }
        }

        if self.token_count > self.capacity_tokens
        {
            return Err(ExperimentError::TokenCountExceedsCapacity {
                token_count: self.token_count,
                capacity_tokens: self.capacity_tokens,
            });
        }

        if self.key_rank > self.head_dimension
        {
            return Err(ExperimentError::RankExceedsDimension {
                name: "key",
                rank: self.key_rank,
                head_dimension: self.head_dimension,
            });
        }

        if self.value_rank > self.head_dimension
        {
            return Err(ExperimentError::RankExceedsDimension {
                name: "value",
                rank: self.value_rank,
                head_dimension: self.head_dimension,
            });
        }

        if !self.amplitude.is_finite() || self.amplitude <= 0.0
        {
            return Err(ExperimentError::InvalidAmplitude {
                amplitude: self.amplitude,
            });
        }

        Ok(())
    }
}

/// Errors returned by the Phase 1 measurement harness.
#[derive(Debug, Clone, PartialEq)]
pub enum ExperimentError {
    /// A required integer field was zero.
    ZeroField {
        /// Name of the invalid field.
        field: &'static str,
    },
    /// The requested token population exceeds the reserved capacity.
    TokenCountExceedsCapacity {
        /// Number of tokens requested by the scenario.
        token_count: usize,
        /// Number of tokens reserved by the cache.
        capacity_tokens: usize,
    },
    /// A latent rank exceeds the dense head dimension.
    RankExceedsDimension {
        /// Human-readable rank name.
        name: &'static str,
        /// Requested latent rank.
        rank: usize,
        /// Dense head dimension.
        head_dimension: usize,
    },
    /// The random amplitude was non-finite or non-positive.
    InvalidAmplitude {
        /// Invalid amplitude value.
        amplitude: f32,
    },
    /// A byte or operation-count computation overflowed `u64`.
    ArithmeticOverflow,
    /// A Phase 0 cache operation failed.
    Cache(CacheError),
}

impl fmt::Display for ExperimentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::ZeroField { field } => write!(formatter, "{field} must be non-zero"),
            Self::TokenCountExceedsCapacity {
                token_count,
                capacity_tokens,
            } => write!(
                formatter,
                "token_count {token_count} exceeds capacity_tokens {capacity_tokens}"
            ),
            Self::RankExceedsDimension {
                name,
                rank,
                head_dimension,
            } => write!(
                formatter,
                "{name} rank {rank} exceeds head dimension {head_dimension}"
            ),
            Self::InvalidAmplitude { amplitude } => write!(
                formatter,
                "amplitude must be finite and positive, received {amplitude}"
            ),
            Self::ArithmeticOverflow => write!(formatter, "measurement arithmetic overflow"),
            Self::Cache(error) => write!(formatter, "cache error: {error}"),
        }
    }
}

impl std::error::Error for ExperimentError {}

impl From<CacheError> for ExperimentError {
    fn from(error: CacheError) -> Self {
        Self::Cache(error)
    }
}

/// Exact byte accounting for one fixed-capacity scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryFootprint {
    /// Dense key and value payload bytes at declared capacity.
    pub dense_payload_bytes: u64,
    /// Latent key and value coefficient bytes at declared capacity.
    pub latent_payload_bytes: u64,
    /// Key and value basis bytes.
    pub latent_basis_bytes: u64,
    /// Shared Phase 0 scratch bytes.
    pub shared_scratch_bytes: u64,
    /// Dense payload plus shared scratch.
    pub dense_total_bytes: u64,
    /// Latent payload, bases and shared scratch.
    pub latent_total_bytes: u64,
    /// Dense payload bytes divided by latent payload plus basis bytes.
    pub payload_compression_ratio: f64,
    /// Dense total bytes divided by latent total bytes.
    pub total_compression_ratio: f64,
}

impl MemoryFootprint {
    /// Computes exact byte counts from a validated scenario configuration.
    pub fn from_config(config: &ExperimentConfig) -> Result<Self, ExperimentError> {
        config.validate()?;

        let scalar_bytes =
            u64::try_from(size_of::<f32>()).map_err(|_| ExperimentError::ArithmeticOverflow)?;
        let capacity = as_u64(config.capacity_tokens)?;
        let dimension = as_u64(config.head_dimension)?;
        let key_rank = as_u64(config.key_rank)?;
        let value_rank = as_u64(config.value_rank)?;
        let max_rank = key_rank.max(value_rank);
        let rank_sum = checked_add(key_rank, value_rank)?;

        let dense_payload_bytes = checked_mul(
            checked_mul(checked_mul(capacity, dimension)?, 2)?,
            scalar_bytes,
        )?;
        let latent_payload_bytes = checked_mul(checked_mul(capacity, rank_sum)?, scalar_bytes)?;
        let latent_basis_bytes = checked_mul(checked_mul(dimension, rank_sum)?, scalar_bytes)?;
        let shared_scratch_elements =
            checked_add(checked_add(capacity, checked_mul(max_rank, 2)?)?, dimension)?;
        let shared_scratch_bytes = checked_mul(shared_scratch_elements, scalar_bytes)?;
        let dense_total_bytes = checked_add(dense_payload_bytes, shared_scratch_bytes)?;
        let latent_total_bytes = checked_add(
            checked_add(latent_payload_bytes, latent_basis_bytes)?,
            shared_scratch_bytes,
        )?;
        let latent_payload_and_basis = checked_add(latent_payload_bytes, latent_basis_bytes)?;

        Ok(Self {
            dense_payload_bytes,
            latent_payload_bytes,
            latent_basis_bytes,
            shared_scratch_bytes,
            dense_total_bytes,
            latent_total_bytes,
            payload_compression_ratio: ratio(dense_payload_bytes, latent_payload_and_basis),
            total_compression_ratio: ratio(dense_total_bytes, latent_total_bytes),
        })
    }
}

/// Deterministic multiply-accumulate estimates for one attention query.
#[derive(Debug, Clone, PartialEq)]
pub struct OperationEstimate {
    /// Dense query-key multiply-accumulates.
    pub dense_score_macs: u64,
    /// Dense probability-value multiply-accumulates.
    pub dense_value_macs: u64,
    /// Total dense multiply-accumulates.
    pub dense_total_macs: u64,
    /// Query projection multiply-accumulates in the reconstruction-free path.
    pub latent_query_projection_macs: u64,
    /// Latent query-key score multiply-accumulates.
    pub latent_score_macs: u64,
    /// Latent probability-value accumulation multiply-accumulates.
    pub latent_value_accumulation_macs: u64,
    /// Final latent value projection multiply-accumulates.
    pub latent_value_projection_macs: u64,
    /// Total reconstruction-free multiply-accumulates.
    pub latent_total_macs: u64,
    /// Total explicit-reconstruction multiply-accumulates.
    pub explicit_latent_total_macs: u64,
    /// Reconstruction-free total divided by dense total.
    pub optimized_vs_dense_ratio: f64,
    /// Reconstruction-free total divided by explicit latent total.
    pub optimized_vs_explicit_ratio: f64,
}

impl OperationEstimate {
    /// Computes deterministic operation counts from a validated configuration.
    pub fn from_config(config: &ExperimentConfig) -> Result<Self, ExperimentError> {
        config.validate()?;

        let tokens = as_u64(config.token_count)?;
        let dimension = as_u64(config.head_dimension)?;
        let key_rank = as_u64(config.key_rank)?;
        let value_rank = as_u64(config.value_rank)?;

        let dense_score_macs = checked_mul(tokens, dimension)?;
        let dense_value_macs = checked_mul(tokens, dimension)?;
        let dense_total_macs = checked_add(dense_score_macs, dense_value_macs)?;

        let latent_query_projection_macs = checked_mul(dimension, key_rank)?;
        let latent_score_macs = checked_mul(tokens, key_rank)?;
        let latent_value_accumulation_macs = checked_mul(tokens, value_rank)?;
        let latent_value_projection_macs = checked_mul(dimension, value_rank)?;
        let latent_total_macs = checked_add(
            checked_add(latent_query_projection_macs, latent_score_macs)?,
            checked_add(latent_value_accumulation_macs, latent_value_projection_macs)?,
        )?;

        let explicit_key_reconstruction = checked_mul(checked_mul(tokens, dimension)?, key_rank)?;
        let explicit_key_scoring = checked_mul(tokens, dimension)?;
        let explicit_value_reconstruction =
            checked_mul(checked_mul(tokens, dimension)?, value_rank)?;
        let explicit_value_accumulation = checked_mul(tokens, dimension)?;
        let explicit_latent_total_macs = checked_add(
            checked_add(explicit_key_reconstruction, explicit_key_scoring)?,
            checked_add(explicit_value_reconstruction, explicit_value_accumulation)?,
        )?;

        Ok(Self {
            dense_score_macs,
            dense_value_macs,
            dense_total_macs,
            latent_query_projection_macs,
            latent_score_macs,
            latent_value_accumulation_macs,
            latent_value_projection_macs,
            latent_total_macs,
            explicit_latent_total_macs,
            optimized_vs_dense_ratio: ratio(latent_total_macs, dense_total_macs),
            optimized_vs_explicit_ratio: ratio(latent_total_macs, explicit_latent_total_macs),
        })
    }
}

/// Aggregate numeric error metrics over one or more output vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorMetrics {
    /// Number of scalar comparisons included.
    pub samples: u64,
    /// Largest absolute error.
    pub max_absolute: f64,
    /// Mean absolute error.
    pub mean_absolute: f64,
    /// Root-mean-square error.
    pub root_mean_square: f64,
    /// Largest relative error using a small denominator floor.
    pub max_relative: f64,
}

/// Differential comparisons recorded for one scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentialReport {
    /// Dense oracle compared with explicit latent reconstruction.
    pub dense_vs_explicit: ErrorMetrics,
    /// Explicit latent reconstruction compared with reconstruction-free attention.
    pub explicit_vs_reconstruction_free: ErrorMetrics,
    /// Dense oracle compared directly with reconstruction-free attention.
    pub dense_vs_reconstruction_free: ErrorMetrics,
}

/// Complete deterministic Phase 1 report for one scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioReport {
    /// Scenario configuration.
    pub config: ExperimentConfig,
    /// Exact memory accounting.
    pub memory: MemoryFootprint,
    /// Deterministic operation-count estimate for one query.
    pub operations: OperationEstimate,
    /// Aggregate differential error metrics over every query.
    pub differentials: DifferentialReport,
    /// FNV-1a fingerprint over all generated outputs.
    pub output_fingerprint: u64,
}

impl ScenarioReport {
    /// Serializes this report as one stable CSV row.
    #[must_use]
    pub fn to_csv_row(&self) -> String {
        format!(
            concat!(
                "{},{},{},{},{},{},{},{:.9e},",
                "{},{},{},{},{},{},{:.9e},{:.9e},",
                "{},{},{},{:.9e},{:.9e},",
                "{:.9e},{:.9e},{:.9e},{:.9e},{:016x}"
            ),
            self.config.seed,
            self.config.capacity_tokens,
            self.config.token_count,
            self.config.head_dimension,
            self.config.key_rank,
            self.config.value_rank,
            self.config.query_count,
            self.config.amplitude,
            self.memory.dense_payload_bytes,
            self.memory.latent_payload_bytes,
            self.memory.latent_basis_bytes,
            self.memory.shared_scratch_bytes,
            self.memory.dense_total_bytes,
            self.memory.latent_total_bytes,
            self.memory.payload_compression_ratio,
            self.memory.total_compression_ratio,
            self.operations.dense_total_macs,
            self.operations.latent_total_macs,
            self.operations.explicit_latent_total_macs,
            self.operations.optimized_vs_dense_ratio,
            self.operations.optimized_vs_explicit_ratio,
            self.differentials.dense_vs_reconstruction_free.max_absolute,
            self.differentials
                .dense_vs_reconstruction_free
                .root_mean_square,
            self.differentials
                .explicit_vs_reconstruction_free
                .max_absolute,
            self.differentials
                .explicit_vs_reconstruction_free
                .root_mean_square,
            self.output_fingerprint,
        )
    }
}

/// Stable CSV header emitted by [`suite_to_csv`].
pub const CSV_HEADER: &str = concat!(
    "seed,capacity_tokens,token_count,head_dimension,key_rank,value_rank,query_count,amplitude,",
    "dense_payload_bytes,latent_payload_bytes,latent_basis_bytes,shared_scratch_bytes,",
    "dense_total_bytes,latent_total_bytes,payload_compression_ratio,total_compression_ratio,",
    "dense_total_macs,latent_total_macs,explicit_latent_total_macs,",
    "optimized_vs_dense_ratio,optimized_vs_explicit_ratio,",
    "dense_vs_free_max_absolute,dense_vs_free_rms,",
    "explicit_vs_free_max_absolute,explicit_vs_free_rms,output_fingerprint"
);

/// Runs one deterministic scenario against all three Phase 0 attention paths.
pub fn run_scenario(config: &ExperimentConfig) -> Result<ScenarioReport, ExperimentError> {
    config.validate()?;

    let memory = MemoryFootprint::from_config(config)?;
    let operations = OperationEstimate::from_config(config)?;
    let mut rng = DeterministicRng::new(config.seed);

    let key_basis = random_vector(
        &mut rng,
        checked_len(config.head_dimension, config.key_rank)?,
        config.amplitude,
    );
    let value_basis = random_vector(
        &mut rng,
        checked_len(config.head_dimension, config.value_rank)?,
        config.amplitude,
    );

    let mut dense = DenseKvCache::new(config.capacity_tokens, config.head_dimension)?;
    let mut latent = FixedRankLatentCache::new(
        config.capacity_tokens,
        config.head_dimension,
        config.key_rank,
        config.value_rank,
        key_basis.clone(),
        value_basis.clone(),
    )?;

    let mut dense_key = vec![0.0; config.head_dimension];
    let mut dense_value = vec![0.0; config.head_dimension];

    for _ in 0..config.token_count
    {
        let key_coefficients = random_vector(&mut rng, config.key_rank, config.amplitude);
        let value_coefficients = random_vector(&mut rng, config.value_rank, config.amplitude);

        matrix_vector(
            &key_basis,
            config.head_dimension,
            config.key_rank,
            &key_coefficients,
            &mut dense_key,
        );
        matrix_vector(
            &value_basis,
            config.head_dimension,
            config.value_rank,
            &value_coefficients,
            &mut dense_value,
        );

        dense.append(&dense_key, &dense_value)?;
        latent.append_coefficients(&key_coefficients, &value_coefficients)?;
    }

    let max_rank = config.key_rank.max(config.value_rank);
    let mut dense_scratch =
        AttentionScratch::new(config.capacity_tokens, config.head_dimension, max_rank);
    let mut explicit_scratch =
        AttentionScratch::new(config.capacity_tokens, config.head_dimension, max_rank);
    let mut reconstruction_free_scratch =
        AttentionScratch::new(config.capacity_tokens, config.head_dimension, max_rank);

    let mut query = vec![0.0; config.head_dimension];
    let mut dense_output = vec![0.0; config.head_dimension];
    let mut explicit_output = vec![0.0; config.head_dimension];
    let mut reconstruction_free_output = vec![0.0; config.head_dimension];
    let mut dense_vs_explicit = ErrorAccumulator::default();
    let mut explicit_vs_reconstruction_free = ErrorAccumulator::default();
    let mut dense_vs_reconstruction_free = ErrorAccumulator::default();
    let mut fingerprint = FNV_OFFSET_BASIS;
    let scale = (config.head_dimension as f32).sqrt().recip();

    for _ in 0..config.query_count
    {
        fill_random_vector(&mut rng, &mut query, config.amplitude);

        dense.attention(&query, scale, &mut dense_output, &mut dense_scratch)?;
        latent.attention_explicit(&query, scale, &mut explicit_output, &mut explicit_scratch)?;
        latent.attention_reconstruction_free(
            &query,
            scale,
            &mut reconstruction_free_output,
            &mut reconstruction_free_scratch,
        )?;

        dense_vs_explicit.observe_slices(&dense_output, &explicit_output);
        explicit_vs_reconstruction_free
            .observe_slices(&explicit_output, &reconstruction_free_output);
        dense_vs_reconstruction_free.observe_slices(&dense_output, &reconstruction_free_output);

        for value in dense_output
            .iter()
            .chain(explicit_output.iter())
            .chain(reconstruction_free_output.iter())
        {
            fingerprint = hash_f32(fingerprint, *value);
        }
    }

    Ok(ScenarioReport {
        config: config.clone(),
        memory,
        operations,
        differentials: DifferentialReport {
            dense_vs_explicit: dense_vs_explicit.finish(),
            explicit_vs_reconstruction_free: explicit_vs_reconstruction_free.finish(),
            dense_vs_reconstruction_free: dense_vs_reconstruction_free.finish(),
        },
        output_fingerprint: fingerprint,
    })
}

/// Returns the deterministic 24-scenario Phase 1 baseline suite.
#[must_use]
pub fn standard_scenarios() -> Vec<ExperimentConfig> {
    let dimensions = [8_usize, 16, 32, 64];
    let rank_pairs = [(2_usize, 3_usize), (4, 2), (8, 6)];
    let token_counts = [4_usize, 16];
    let amplitudes = [0.05_f32, 0.5, 1.0];
    let mut scenarios =
        Vec::with_capacity(dimensions.len() * rank_pairs.len() * token_counts.len());
    let mut index = 0_u64;

    for (dimension_index, dimension) in dimensions.into_iter().enumerate()
    {
        for (rank_index, (key_rank, value_rank)) in rank_pairs.into_iter().enumerate()
        {
            for (token_index, token_count) in token_counts.into_iter().enumerate()
            {
                let amplitude_index =
                    (dimension_index + rank_index + token_index) % amplitudes.len();
                let seed = 0x51C1_2A7E_0000_0000_u64
                    ^ ((dimension as u64) << 32)
                    ^ ((key_rank as u64) << 24)
                    ^ ((value_rank as u64) << 16)
                    ^ ((token_count as u64) << 8)
                    ^ index;

                scenarios.push(ExperimentConfig {
                    capacity_tokens: token_count,
                    token_count,
                    head_dimension: dimension,
                    key_rank,
                    value_rank,
                    query_count: 3,
                    amplitude: amplitudes[amplitude_index],
                    seed,
                });
                index += 1;
            }
        }
    }

    scenarios
}

/// Runs the deterministic 24-scenario Phase 1 baseline suite.
pub fn run_standard_suite() -> Result<Vec<ScenarioReport>, ExperimentError> {
    standard_scenarios().iter().map(run_scenario).collect()
}

/// Serializes reports as stable newline-terminated CSV.
#[must_use]
pub fn suite_to_csv(reports: &[ScenarioReport]) -> String {
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

#[derive(Debug, Default, Clone)]
struct ErrorAccumulator {
    samples: u64,
    sum_absolute: f64,
    sum_squared: f64,
    max_absolute: f64,
    max_relative: f64,
}

impl ErrorAccumulator {
    fn observe_slices(&mut self, reference: &[f32], candidate: &[f32]) {
        debug_assert_eq!(reference.len(), candidate.len());

        for (reference_value, candidate_value) in reference.iter().zip(candidate)
        {
            let reference_value = f64::from(*reference_value);
            let candidate_value = f64::from(*candidate_value);
            let absolute = (candidate_value - reference_value).abs();
            let denominator = reference_value.abs().max(RELATIVE_ERROR_FLOOR);
            let relative = absolute / denominator;

            self.samples += 1;
            self.sum_absolute += absolute;
            self.sum_squared += absolute * absolute;
            self.max_absolute = self.max_absolute.max(absolute);
            self.max_relative = self.max_relative.max(relative);
        }
    }

    fn finish(self) -> ErrorMetrics {
        debug_assert!(self.samples > 0);
        let sample_count = self.samples as f64;

        ErrorMetrics {
            samples: self.samples,
            max_absolute: self.max_absolute,
            mean_absolute: self.sum_absolute / sample_count,
            root_mean_square: (self.sum_squared / sample_count).sqrt(),
            max_relative: self.max_relative,
        }
    }
}

fn random_vector(rng: &mut DeterministicRng, length: usize, amplitude: f32) -> Vec<f32> {
    let mut vector = vec![0.0; length];
    fill_random_vector(rng, &mut vector, amplitude);
    vector
}

fn fill_random_vector(rng: &mut DeterministicRng, vector: &mut [f32], amplitude: f32) {
    for value in vector
    {
        *value = rng.next_symmetric_f32(amplitude);
    }
}

fn matrix_vector(matrix: &[f32], rows: usize, columns: usize, vector: &[f32], output: &mut [f32]) {
    debug_assert_eq!(matrix.len(), rows * columns);
    debug_assert_eq!(vector.len(), columns);
    debug_assert_eq!(output.len(), rows);

    for (row_index, output_element) in output.iter_mut().enumerate()
    {
        let row_offset = row_index * columns;
        let row = &matrix[row_offset..row_offset + columns];
        *output_element = row
            .iter()
            .zip(vector)
            .map(|(left, right)| left * right)
            .sum();
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

fn checked_len(left: usize, right: usize) -> Result<usize, ExperimentError> {
    left.checked_mul(right)
        .ok_or(ExperimentError::ArithmeticOverflow)
}

fn as_u64(value: usize) -> Result<u64, ExperimentError> {
    u64::try_from(value).map_err(|_| ExperimentError::ArithmeticOverflow)
}

fn checked_add(left: u64, right: u64) -> Result<u64, ExperimentError> {
    left.checked_add(right)
        .ok_or(ExperimentError::ArithmeticOverflow)
}

fn checked_mul(left: u64, right: u64) -> Result<u64, ExperimentError> {
    left.checked_mul(right)
        .ok_or(ExperimentError::ArithmeticOverflow)
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    debug_assert!(denominator > 0);
    numerator as f64 / denominator as f64
}

#[cfg(test)]
mod tests {
    use super::{
        CSV_HEADER, ExperimentConfig, ExperimentError, MemoryFootprint, OperationEstimate,
        run_scenario, standard_scenarios, suite_to_csv,
    };

    fn compact_config() -> ExperimentConfig {
        ExperimentConfig {
            capacity_tokens: 8,
            token_count: 4,
            head_dimension: 8,
            key_rank: 2,
            value_rank: 3,
            query_count: 2,
            amplitude: 0.5,
            seed: 0x1234_5678_9ABC_DEF0,
        }
    }

    #[test]
    fn configuration_validation_rejects_structural_errors() {
        let mut config = compact_config();
        config.token_count = 0;
        assert_eq!(
            config.validate(),
            Err(ExperimentError::ZeroField {
                field: "token_count"
            })
        );

        let mut config = compact_config();
        config.token_count = 9;
        assert_eq!(
            config.validate(),
            Err(ExperimentError::TokenCountExceedsCapacity {
                token_count: 9,
                capacity_tokens: 8,
            })
        );

        let mut config = compact_config();
        config.key_rank = 9;
        assert_eq!(
            config.validate(),
            Err(ExperimentError::RankExceedsDimension {
                name: "key",
                rank: 9,
                head_dimension: 8,
            })
        );
    }

    #[test]
    fn memory_accounting_matches_closed_form() {
        let config = compact_config();
        let footprint = MemoryFootprint::from_config(&config).unwrap();

        assert_eq!(footprint.dense_payload_bytes, 512);
        assert_eq!(footprint.latent_payload_bytes, 160);
        assert_eq!(footprint.latent_basis_bytes, 160);
        assert_eq!(footprint.shared_scratch_bytes, 88);
        assert_eq!(footprint.dense_total_bytes, 600);
        assert_eq!(footprint.latent_total_bytes, 408);
        assert!((footprint.payload_compression_ratio - 1.6).abs() < 1.0e-12);
        assert!((footprint.total_compression_ratio - (600.0 / 408.0)).abs() < 1.0e-12);
    }

    #[test]
    fn operation_accounting_matches_closed_form() {
        let config = compact_config();
        let estimate = OperationEstimate::from_config(&config).unwrap();

        assert_eq!(estimate.dense_score_macs, 32);
        assert_eq!(estimate.dense_value_macs, 32);
        assert_eq!(estimate.dense_total_macs, 64);
        assert_eq!(estimate.latent_query_projection_macs, 16);
        assert_eq!(estimate.latent_score_macs, 8);
        assert_eq!(estimate.latent_value_accumulation_macs, 12);
        assert_eq!(estimate.latent_value_projection_macs, 24);
        assert_eq!(estimate.latent_total_macs, 60);
        assert_eq!(estimate.explicit_latent_total_macs, 224);
    }

    #[test]
    fn scenario_execution_is_bit_deterministic() {
        let config = compact_config();
        let first = run_scenario(&config).unwrap();
        let second = run_scenario(&config).unwrap();

        assert_eq!(first.output_fingerprint, second.output_fingerprint);
        assert_eq!(first.to_csv_row(), second.to_csv_row());
    }

    #[test]
    fn reconstruction_free_path_matches_both_oracles() {
        let report = run_scenario(&compact_config()).unwrap();

        assert!(
            report
                .differentials
                .explicit_vs_reconstruction_free
                .max_absolute
                <= 1.0e-5
        );
        assert!(
            report
                .differentials
                .dense_vs_reconstruction_free
                .max_absolute
                <= 1.0e-5
        );
        assert_eq!(
            report.differentials.dense_vs_reconstruction_free.samples,
            16
        );
    }

    #[test]
    fn standard_suite_is_stable_and_complete() {
        let scenarios = standard_scenarios();

        assert_eq!(scenarios.len(), 24);
        assert_eq!(scenarios.first().unwrap().head_dimension, 8);
        assert_eq!(scenarios.last().unwrap().head_dimension, 64);
        assert!(scenarios.iter().all(|scenario| scenario.validate().is_ok()));
    }

    #[test]
    fn csv_export_has_one_header_and_one_row_per_report() {
        let reports = [run_scenario(&compact_config()).unwrap()];
        let csv = suite_to_csv(&reports);
        let lines: Vec<_> = csv.lines().collect();

        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], CSV_HEADER);
        assert_eq!(lines[1], reports[0].to_csv_row());
    }
}

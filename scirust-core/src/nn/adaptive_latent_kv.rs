//! Deterministic budget-constrained planner for adaptive latent KV storage.
//!
//! Phase 9 selects independent key/value latent ranks, sparse residual slot
//! counts, and FP32/INT8/INT4 storage formats under one strict persistent-memory
//! budget. Selection is exhaustive, allocation-free, and uses stable tie-breaks.

use crate::nn::latent_kv_cache::LatentStorageFormat;
use core::fmt;

const QUALITY_SCALE: u32 = 10_000;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const FORMATS: [LatentStorageFormat; 3] = [
    LatentStorageFormat::Int4,
    LatentStorageFormat::Int8,
    LatentStorageFormat::F32,
];

/// Immutable Phase 9 planning constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveKvPolicyConfig {
    /// Number of tokens reserved by the backend.
    pub capacity_tokens: usize,
    /// Dense head dimension.
    pub dimension: usize,
    /// Smallest allowed key/value rank.
    pub minimum_rank: usize,
    /// Largest allowed key/value rank.
    pub maximum_rank: usize,
    /// Largest allowed residual slot count per token and channel.
    pub maximum_residual_slots: usize,
    /// Strict persistent byte budget across keys and values.
    pub budget_bytes: usize,
}

/// Quality telemetry supplied by calibration or online measurement.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveQualityProfile<'a> {
    /// Cumulative key quality in basis points for ranks `1..=dimension`.
    pub key_rank_quality_bps: &'a [u16],
    /// Cumulative value quality in basis points for ranks `1..=dimension`.
    pub value_rank_quality_bps: &'a [u16],
    /// Additional key quality for slot counts `0..=maximum_residual_slots`.
    pub key_residual_gain_bps: &'a [u16],
    /// Additional value quality for slot counts `0..=maximum_residual_slots`.
    pub value_residual_gain_bps: &'a [u16],
}

/// One independently selected key or value channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveChannelPlan {
    /// Selected latent rank.
    pub rank: usize,
    /// Selected sparse residual slots per token.
    pub residual_slots: usize,
    /// Latent coefficient format.
    pub coefficient_format: LatentStorageFormat,
    /// Sparse residual value format.
    pub residual_format: LatentStorageFormat,
    /// Estimated persistent bytes for this channel.
    pub persistent_bytes: usize,
    /// Deterministic estimated quality in basis points.
    pub quality_bps: u16,
}

/// Complete key/value plan selected under the strict global budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveKvPlan {
    /// Key-channel configuration.
    pub key: AdaptiveChannelPlan,
    /// Value-channel configuration.
    pub value: AdaptiveChannelPlan,
    /// Sum of key/value persistent bytes.
    pub persistent_bytes: usize,
    /// Minimum of key and value quality.
    pub worst_quality_bps: u16,
    /// Stable FNV-1a fingerprint of the selected plan.
    pub fingerprint: u64,
}

/// Result returned by the hysteretic planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveDecision {
    /// Active plan after observing the profile.
    pub plan: AdaptiveKvPlan,
    /// Whether the active plan changed on this observation.
    pub changed: bool,
    /// Number of consecutive confirmations for a pending plan.
    pub pending_confirmations: usize,
}

/// Planner validation or feasibility errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdaptiveKvPolicyError {
    /// A required scalar was zero.
    ZeroField(&'static str),
    /// Rank bounds were invalid.
    InvalidRankBounds,
    /// Residual slots exceeded the dense dimension.
    ResidualSlotsTooLarge,
    /// A quality profile slice had an unexpected length.
    ProfileLength {
        /// Human-readable profile field.
        field: &'static str,
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },
    /// No candidate can satisfy the byte budget.
    BudgetInfeasible,
}

impl fmt::Display for AdaptiveKvPolicyError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::ZeroField(field) => write!(output, "{field} must be non-zero"),
            Self::InvalidRankBounds => write!(output, "adaptive KV rank bounds are invalid"),
            Self::ResidualSlotsTooLarge =>
            {
                write!(
                    output,
                    "adaptive KV residual slots exceed the dense dimension"
                )
            },
            Self::ProfileLength {
                field,
                expected,
                actual,
            } => write!(
                output,
                "{field} profile length mismatch: expected {expected}, got {actual}"
            ),
            Self::BudgetInfeasible => write!(output, "adaptive KV byte budget is infeasible"),
        }
    }
}

impl std::error::Error for AdaptiveKvPolicyError {}

/// Stateful deterministic planner with confirmation hysteresis.
#[derive(Debug, Clone)]
pub struct AdaptiveKvPlanner {
    config: AdaptiveKvPolicyConfig,
    required_confirmations: usize,
    minimum_quality_gain_bps: u16,
    current: Option<AdaptiveKvPlan>,
    pending: Option<AdaptiveKvPlan>,
    pending_confirmations: usize,
}

impl AdaptiveKvPlanner {
    /// Creates a planner. `required_confirmations` must be non-zero.
    pub fn new(
        config: AdaptiveKvPolicyConfig,
        required_confirmations: usize,
        minimum_quality_gain_bps: u16,
    ) -> Result<Self, AdaptiveKvPolicyError> {
        validate_config(config)?;
        if required_confirmations == 0
        {
            return Err(AdaptiveKvPolicyError::ZeroField("required_confirmations"));
        }
        Ok(Self {
            config,
            required_confirmations,
            minimum_quality_gain_bps,
            current: None,
            pending: None,
            pending_confirmations: 0,
        })
    }

    /// Returns the active plan, if calibration has already produced one.
    #[must_use]
    pub const fn current(&self) -> Option<AdaptiveKvPlan> {
        self.current
    }

    /// Selects and observes a candidate plan with deterministic hysteresis.
    pub fn observe(
        &mut self,
        profile: AdaptiveQualityProfile<'_>,
    ) -> Result<AdaptiveDecision, AdaptiveKvPolicyError> {
        let candidate = select_adaptive_plan(self.config, profile)?;
        let Some(current) = self.current
        else
        {
            self.current = Some(candidate);
            self.pending = None;
            self.pending_confirmations = 0;
            return Ok(AdaptiveDecision {
                plan: candidate,
                changed: true,
                pending_confirmations: 0,
            });
        };

        if candidate == current
        {
            self.pending = None;
            self.pending_confirmations = 0;
            return Ok(AdaptiveDecision {
                plan: current,
                changed: false,
                pending_confirmations: 0,
            });
        }

        let quality_gain = candidate
            .worst_quality_bps
            .saturating_sub(current.worst_quality_bps);
        let materially_smaller = candidate.persistent_bytes < current.persistent_bytes;
        if quality_gain < self.minimum_quality_gain_bps && !materially_smaller
        {
            self.pending = None;
            self.pending_confirmations = 0;
            return Ok(AdaptiveDecision {
                plan: current,
                changed: false,
                pending_confirmations: 0,
            });
        }

        if self.pending == Some(candidate)
        {
            self.pending_confirmations = self.pending_confirmations.saturating_add(1);
        }
        else
        {
            self.pending = Some(candidate);
            self.pending_confirmations = 1;
        }

        if self.pending_confirmations >= self.required_confirmations
        {
            self.current = Some(candidate);
            self.pending = None;
            self.pending_confirmations = 0;
            return Ok(AdaptiveDecision {
                plan: candidate,
                changed: true,
                pending_confirmations: 0,
            });
        }

        Ok(AdaptiveDecision {
            plan: current,
            changed: false,
            pending_confirmations: self.pending_confirmations,
        })
    }
}

/// Exhaustively selects the best deterministic plan under `config.budget_bytes`.
pub fn select_adaptive_plan(
    config: AdaptiveKvPolicyConfig,
    profile: AdaptiveQualityProfile<'_>,
) -> Result<AdaptiveKvPlan, AdaptiveKvPolicyError> {
    validate_config(config)?;
    validate_profile(config, profile)?;

    let mut best: Option<AdaptiveKvPlan> = None;
    for key_rank in config.minimum_rank..=config.maximum_rank
    {
        for key_slots in 0..=config.maximum_residual_slots
        {
            for key_coefficient_format in FORMATS
            {
                for key_residual_format in FORMATS
                {
                    let key = channel_candidate(
                        config,
                        key_rank,
                        key_slots,
                        key_coefficient_format,
                        key_residual_format,
                        profile.key_rank_quality_bps,
                        profile.key_residual_gain_bps,
                    );
                    if key.persistent_bytes >= config.budget_bytes
                    {
                        continue;
                    }
                    let remaining = config.budget_bytes - key.persistent_bytes;
                    for value_rank in config.minimum_rank..=config.maximum_rank
                    {
                        for value_slots in 0..=config.maximum_residual_slots
                        {
                            for value_coefficient_format in FORMATS
                            {
                                for value_residual_format in FORMATS
                                {
                                    let value = channel_candidate(
                                        config,
                                        value_rank,
                                        value_slots,
                                        value_coefficient_format,
                                        value_residual_format,
                                        profile.value_rank_quality_bps,
                                        profile.value_residual_gain_bps,
                                    );
                                    if value.persistent_bytes > remaining
                                    {
                                        continue;
                                    }
                                    let candidate = complete_plan(key, value);
                                    if best.is_none_or(|current| better_plan(candidate, current))
                                    {
                                        best = Some(candidate);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    best.ok_or(AdaptiveKvPolicyError::BudgetInfeasible)
}

fn validate_config(config: AdaptiveKvPolicyConfig) -> Result<(), AdaptiveKvPolicyError> {
    if config.capacity_tokens == 0
    {
        return Err(AdaptiveKvPolicyError::ZeroField("capacity_tokens"));
    }
    if config.dimension == 0
    {
        return Err(AdaptiveKvPolicyError::ZeroField("dimension"));
    }
    if config.budget_bytes == 0
    {
        return Err(AdaptiveKvPolicyError::ZeroField("budget_bytes"));
    }
    if config.minimum_rank == 0
        || config.minimum_rank > config.maximum_rank
        || config.maximum_rank > config.dimension
    {
        return Err(AdaptiveKvPolicyError::InvalidRankBounds);
    }
    if config.maximum_residual_slots > config.dimension
    {
        return Err(AdaptiveKvPolicyError::ResidualSlotsTooLarge);
    }
    Ok(())
}

fn validate_profile(
    config: AdaptiveKvPolicyConfig,
    profile: AdaptiveQualityProfile<'_>,
) -> Result<(), AdaptiveKvPolicyError> {
    require_profile_length(
        "key_rank_quality_bps",
        profile.key_rank_quality_bps,
        config.dimension,
    )?;
    require_profile_length(
        "value_rank_quality_bps",
        profile.value_rank_quality_bps,
        config.dimension,
    )?;
    let residual_length = config.maximum_residual_slots + 1;
    require_profile_length(
        "key_residual_gain_bps",
        profile.key_residual_gain_bps,
        residual_length,
    )?;
    require_profile_length(
        "value_residual_gain_bps",
        profile.value_residual_gain_bps,
        residual_length,
    )
}

fn require_profile_length(
    field: &'static str,
    values: &[u16],
    expected: usize,
) -> Result<(), AdaptiveKvPolicyError> {
    if values.len() != expected
    {
        return Err(AdaptiveKvPolicyError::ProfileLength {
            field,
            expected,
            actual: values.len(),
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn channel_candidate(
    config: AdaptiveKvPolicyConfig,
    rank: usize,
    residual_slots: usize,
    coefficient_format: LatentStorageFormat,
    residual_format: LatentStorageFormat,
    rank_quality: &[u16],
    residual_gain: &[u16],
) -> AdaptiveChannelPlan {
    let persistent_bytes = estimate_channel_bytes(
        config.capacity_tokens,
        config.dimension,
        rank,
        residual_slots,
        coefficient_format,
        residual_format,
    );
    let base_quality = u32::from(rank_quality[rank - 1])
        .saturating_add(u32::from(residual_gain[residual_slots]))
        .min(QUALITY_SCALE);
    let residual_retention = if residual_slots == 0
    {
        QUALITY_SCALE
    }
    else
    {
        format_retention_bps(residual_format)
    };
    let retention = format_retention_bps(coefficient_format).min(residual_retention);
    let quality_bps = ((base_quality * retention) / QUALITY_SCALE) as u16;
    AdaptiveChannelPlan {
        rank,
        residual_slots,
        coefficient_format,
        residual_format,
        persistent_bytes,
        quality_bps,
    }
}

fn complete_plan(key: AdaptiveChannelPlan, value: AdaptiveChannelPlan) -> AdaptiveKvPlan {
    let persistent_bytes = key.persistent_bytes.saturating_add(value.persistent_bytes);
    let worst_quality_bps = key.quality_bps.min(value.quality_bps);
    let mut fingerprint = FNV_OFFSET;
    for value in [
        key.rank as u64,
        key.residual_slots as u64,
        format_code(key.coefficient_format),
        format_code(key.residual_format),
        value.rank as u64,
        value.residual_slots as u64,
        format_code(value.coefficient_format),
        format_code(value.residual_format),
        persistent_bytes as u64,
        u64::from(worst_quality_bps),
    ]
    {
        fingerprint ^= value;
        fingerprint = fingerprint.wrapping_mul(FNV_PRIME);
    }
    AdaptiveKvPlan {
        key,
        value,
        persistent_bytes,
        worst_quality_bps,
        fingerprint,
    }
}

fn better_plan(candidate: AdaptiveKvPlan, current: AdaptiveKvPlan) -> bool {
    if candidate.worst_quality_bps != current.worst_quality_bps
    {
        return candidate.worst_quality_bps > current.worst_quality_bps;
    }
    let candidate_sum =
        u32::from(candidate.key.quality_bps) + u32::from(candidate.value.quality_bps);
    let current_sum = u32::from(current.key.quality_bps) + u32::from(current.value.quality_bps);
    if candidate_sum != current_sum
    {
        return candidate_sum > current_sum;
    }
    if candidate.persistent_bytes != current.persistent_bytes
    {
        return candidate.persistent_bytes < current.persistent_bytes;
    }
    let candidate_complexity = candidate.key.rank
        + candidate.value.rank
        + candidate.key.residual_slots
        + candidate.value.residual_slots;
    let current_complexity = current.key.rank
        + current.value.rank
        + current.key.residual_slots
        + current.value.residual_slots;
    if candidate_complexity != current_complexity
    {
        return candidate_complexity < current_complexity;
    }
    plan_lexicographic_key(candidate) < plan_lexicographic_key(current)
}

fn plan_lexicographic_key(plan: AdaptiveKvPlan) -> [usize; 8] {
    [
        plan.key.rank,
        plan.key.residual_slots,
        format_priority(plan.key.coefficient_format),
        format_priority(plan.key.residual_format),
        plan.value.rank,
        plan.value.residual_slots,
        format_priority(plan.value.coefficient_format),
        format_priority(plan.value.residual_format),
    ]
}

/// Estimates persistent bytes for one channel using the Phase 8 layout.
#[must_use]
pub fn estimate_channel_bytes(
    capacity_tokens: usize,
    dimension: usize,
    rank: usize,
    residual_slots: usize,
    coefficient_format: LatentStorageFormat,
    residual_format: LatentStorageFormat,
) -> usize {
    let basis = dimension.saturating_mul(rank).saturating_mul(4);
    let coefficient_row = storage_row_bytes(coefficient_format, rank);
    let coefficient_scale = if coefficient_format == LatentStorageFormat::F32
    {
        0
    }
    else
    {
        4
    };
    let residual_row = residual_slots
        .saturating_mul(2)
        .saturating_add(storage_row_bytes(residual_format, residual_slots));
    let residual_scale = if residual_slots == 0 || residual_format == LatentStorageFormat::F32
    {
        0
    }
    else
    {
        4
    };
    basis.saturating_add(
        capacity_tokens.saturating_mul(
            coefficient_row
                .saturating_add(coefficient_scale)
                .saturating_add(residual_row)
                .saturating_add(residual_scale),
        ),
    )
}

const fn storage_row_bytes(format: LatentStorageFormat, columns: usize) -> usize {
    match format
    {
        LatentStorageFormat::F32 => columns * 4,
        LatentStorageFormat::Int8 => columns,
        LatentStorageFormat::Int4 => columns.div_ceil(2),
    }
}

const fn format_retention_bps(format: LatentStorageFormat) -> u32 {
    match format
    {
        LatentStorageFormat::F32 => QUALITY_SCALE,
        LatentStorageFormat::Int8 => 9_975,
        LatentStorageFormat::Int4 => 9_700,
    }
}

const fn format_priority(format: LatentStorageFormat) -> usize {
    match format
    {
        LatentStorageFormat::Int4 => 0,
        LatentStorageFormat::Int8 => 1,
        LatentStorageFormat::F32 => 2,
    }
}

const fn format_code(format: LatentStorageFormat) -> u64 {
    format_priority(format) as u64
}

#[cfg(test)]
mod tests {
    use super::{
        AdaptiveKvPlanner, AdaptiveKvPolicyConfig, AdaptiveKvPolicyError, AdaptiveQualityProfile,
        estimate_channel_bytes, select_adaptive_plan,
    };
    use crate::nn::latent_kv_cache::LatentStorageFormat;

    fn config(budget_bytes: usize) -> AdaptiveKvPolicyConfig {
        AdaptiveKvPolicyConfig {
            capacity_tokens: 64,
            dimension: 8,
            minimum_rank: 2,
            maximum_rank: 6,
            maximum_residual_slots: 2,
            budget_bytes,
        }
    }

    fn profile() -> AdaptiveQualityProfile<'static> {
        static RANK: [u16; 8] = [3_000, 5_500, 7_000, 8_300, 9_100, 9_550, 9_800, 10_000];
        static VALUE: [u16; 8] = [2_800, 5_200, 6_900, 8_100, 9_000, 9_500, 9_750, 10_000];
        static RESIDUAL: [u16; 3] = [0, 450, 850];
        AdaptiveQualityProfile {
            key_rank_quality_bps: &RANK,
            value_rank_quality_bps: &VALUE,
            key_residual_gain_bps: &RESIDUAL,
            value_residual_gain_bps: &RESIDUAL,
        }
    }

    #[test]
    fn selected_plan_never_exceeds_budget() {
        let plan = select_adaptive_plan(config(1_600), profile()).unwrap();
        assert!(plan.persistent_bytes <= 1_600);
        assert!(plan.key.rank >= 2);
        assert!(plan.value.rank >= 2);
    }

    #[test]
    fn repeated_selection_is_bit_identical() {
        let first = select_adaptive_plan(config(1_600), profile()).unwrap();
        let second = select_adaptive_plan(config(1_600), profile()).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.fingerprint, second.fingerprint);
    }

    #[test]
    fn larger_budget_never_reduces_worst_quality() {
        let small = select_adaptive_plan(config(1_100), profile()).unwrap();
        let large = select_adaptive_plan(config(2_500), profile()).unwrap();
        assert!(large.worst_quality_bps >= small.worst_quality_bps);
    }

    #[test]
    fn quantized_rows_account_for_scales_and_indices() {
        let bytes = estimate_channel_bytes(
            10,
            8,
            4,
            2,
            LatentStorageFormat::Int4,
            LatentStorageFormat::Int8,
        );
        let basis = 8 * 4 * 4;
        let per_token = 2 + 4 + 4 + 2 + 4;
        assert_eq!(bytes, basis + 10 * per_token);
    }

    #[test]
    fn infeasible_budget_is_rejected() {
        assert_eq!(
            select_adaptive_plan(config(1), profile()),
            Err(AdaptiveKvPolicyError::BudgetInfeasible)
        );
    }

    #[test]
    fn planner_requires_confirmed_plan_change() {
        let mut planner = AdaptiveKvPlanner::new(config(2_500), 2, 0).unwrap();
        let first = planner.observe(profile()).unwrap();
        assert!(first.changed);

        static BETTER_RANK: [u16; 8] = [4_000, 6_500, 8_000, 9_000, 9_600, 9_850, 9_950, 10_000];
        static RESIDUAL: [u16; 3] = [0, 300, 600];
        let changed_profile = AdaptiveQualityProfile {
            key_rank_quality_bps: &BETTER_RANK,
            value_rank_quality_bps: &BETTER_RANK,
            key_residual_gain_bps: &RESIDUAL,
            value_residual_gain_bps: &RESIDUAL,
        };
        let pending = planner.observe(changed_profile).unwrap();
        assert!(!pending.changed);
        assert_eq!(pending.pending_confirmations, 1);
        let applied = planner.observe(changed_profile).unwrap();
        assert!(applied.changed);
    }
}

//! Fixed-capacity HOT/WARM/COLD lifecycle control for Elastic Latent KV.

use crate::nn::latent_kv_cache::LatentStorageFormat;
use core::fmt;

/// Temperature tier assigned to a resident token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTemperature {
    /// Most recent tokens; highest-fidelity representation.
    Hot,
    /// Intermediate tokens; balanced representation.
    Warm,
    /// Oldest retained tokens; strongest compression.
    Cold,
}

/// Deterministic compression target for one temperature tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionTier {
    /// Latent coefficient format used by this tier.
    pub coefficient_format: LatentStorageFormat,
    /// Sparse residual format used by this tier.
    pub residual_format: LatentStorageFormat,
    /// Maximum residual slots retained by this tier.
    pub maximum_residual_slots: usize,
    /// Rank divisor relative to the active Phase 9 rank (`1` means unchanged).
    pub rank_divisor: usize,
}

/// Lifecycle configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LifecycleConfig {
    /// Fixed resident token capacity.
    pub capacity_tokens: usize,
    /// Number of newest tokens kept HOT.
    pub hot_tokens: usize,
    /// Number of following tokens kept WARM.
    pub warm_tokens: usize,
    /// HOT representation.
    pub hot: CompressionTier,
    /// WARM representation.
    pub warm: CompressionTier,
    /// COLD representation.
    pub cold: CompressionTier,
}

/// Metadata stored for one resident token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenLifecycle {
    /// Monotonic logical position.
    pub position: u64,
    /// Basis version used to encode this token.
    pub basis_version: u32,
    /// Current temperature.
    pub temperature: CacheTemperature,
    /// Last deterministic access tick.
    pub last_access_tick: u64,
    resident: bool,
}

impl TokenLifecycle {
    const EMPTY: Self = Self {
        position: 0,
        basis_version: 0,
        temperature: CacheTemperature::Cold,
        last_access_tick: 0,
        resident: false,
    };

    /// Returns whether the slot currently contains a resident token.
    #[must_use]
    pub const fn is_resident(self) -> bool {
        self.resident
    }
}

/// Metadata for an evicted token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvictedToken {
    /// Evicted logical position.
    pub position: u64,
    /// Basis version needed to interpret the encoded payload.
    pub basis_version: u32,
    /// Temperature immediately before eviction.
    pub temperature: CacheTemperature,
}

/// Result of admitting one token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Admission {
    /// New logical token position.
    pub position: u64,
    /// Oldest token evicted to preserve fixed capacity.
    pub evicted: Option<EvictedToken>,
}

/// Re-encoding action emitted when a token changes temperature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LifecycleAction {
    /// Logical token position.
    pub position: u64,
    /// Basis version of the resident payload.
    pub basis_version: u32,
    /// Previous temperature.
    pub from: CacheTemperature,
    /// New temperature.
    pub to: CacheTemperature,
    /// Target compression tier.
    pub target: CompressionTier,
}

/// Lifecycle errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleError {
    /// Capacity was zero.
    ZeroCapacity,
    /// HOT+WARM windows exceeded capacity.
    InvalidWindows,
    /// A compression rank divisor was zero.
    ZeroRankDivisor,
    /// Requested token is no longer resident.
    NotResident(u64),
}

impl fmt::Display for LifecycleError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::ZeroCapacity => write!(output, "lifecycle capacity must be non-zero"),
            Self::InvalidWindows => write!(output, "HOT and WARM windows exceed capacity"),
            Self::ZeroRankDivisor => write!(output, "compression rank divisor must be non-zero"),
            Self::NotResident(position) => write!(output, "token position {position} is not resident"),
        }
    }
}

impl std::error::Error for LifecycleError {}

/// Fixed-ring lifecycle controller. No allocation occurs after construction.
#[derive(Debug, Clone)]
pub struct LatentKvLifecycle {
    config: LifecycleConfig,
    slots: Vec<TokenLifecycle>,
    start: usize,
    len: usize,
    next_position: u64,
    tick: u64,
}

impl LatentKvLifecycle {
    /// Constructs an empty fixed-capacity lifecycle ring.
    pub fn new(config: LifecycleConfig) -> Result<Self, LifecycleError> {
        validate_config(config)?;
        Ok(Self {
            config,
            slots: vec![TokenLifecycle::EMPTY; config.capacity_tokens],
            start: 0,
            len: 0,
            next_position: 0,
            tick: 0,
        })
    }

    /// Returns the number of resident tokens.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns whether no token is resident.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Admits a token and deterministically evicts the oldest token when full.
    pub fn admit(&mut self, basis_version: u32) -> Admission {
        self.tick = self.tick.wrapping_add(1);
        let evicted = if self.len == self.config.capacity_tokens
        {
            let previous = self.slots[self.start];
            let evicted = EvictedToken {
                position: previous.position,
                basis_version: previous.basis_version,
                temperature: previous.temperature,
            };
            self.start = (self.start + 1) % self.config.capacity_tokens;
            Some(evicted)
        }
        else
        {
            self.len += 1;
            None
        };

        let write_offset = self.len - 1;
        let write_index = (self.start + write_offset) % self.config.capacity_tokens;
        let position = self.next_position;
        self.next_position = self.next_position.wrapping_add(1);
        self.slots[write_index] = TokenLifecycle {
            position,
            basis_version,
            temperature: CacheTemperature::Hot,
            last_access_tick: self.tick,
            resident: true,
        };
        Admission { position, evicted }
    }

    /// Marks a resident token as accessed. Access does not reorder logical age.
    pub fn touch(&mut self, position: u64) -> Result<(), LifecycleError> {
        self.tick = self.tick.wrapping_add(1);
        let index = self.index_of(position).ok_or(LifecycleError::NotResident(position))?;
        self.slots[index].last_access_tick = self.tick;
        Ok(())
    }

    /// Returns metadata for one resident logical position.
    #[must_use]
    pub fn get(&self, position: u64) -> Option<TokenLifecycle> {
        self.index_of(position).map(|index| self.slots[index])
    }

    /// Returns the target tier for a temperature.
    #[must_use]
    pub const fn tier(&self, temperature: CacheTemperature) -> CompressionTier {
        match temperature
        {
            CacheTemperature::Hot => self.config.hot,
            CacheTemperature::Warm => self.config.warm,
            CacheTemperature::Cold => self.config.cold,
        }
    }

    /// Recomputes temperatures by logical recency and writes transition actions.
    ///
    /// Returns the number of actions written. The caller must provide at least
    /// `len()` action slots to guarantee that every possible transition fits.
    pub fn rebalance_into(&mut self, actions: &mut [LifecycleAction]) -> usize {
        let mut written = 0;
        for offset in 0..self.len
        {
            let index = (self.start + offset) % self.config.capacity_tokens;
            let age_from_newest = self.len - 1 - offset;
            let target_temperature = if age_from_newest < self.config.hot_tokens
            {
                CacheTemperature::Hot
            }
            else if age_from_newest < self.config.hot_tokens + self.config.warm_tokens
            {
                CacheTemperature::Warm
            }
            else
            {
                CacheTemperature::Cold
            };
            let previous = self.slots[index].temperature;
            if previous != target_temperature
            {
                if written < actions.len()
                {
                    actions[written] = LifecycleAction {
                        position: self.slots[index].position,
                        basis_version: self.slots[index].basis_version,
                        from: previous,
                        to: target_temperature,
                        target: self.tier(target_temperature),
                    };
                    written += 1;
                }
                self.slots[index].temperature = target_temperature;
            }
        }
        written
    }

    fn index_of(&self, position: u64) -> Option<usize> {
        if self.len == 0
        {
            return None;
        }
        let oldest = self.slots[self.start].position;
        let offset = position.checked_sub(oldest)? as usize;
        if offset >= self.len
        {
            return None;
        }
        let index = (self.start + offset) % self.config.capacity_tokens;
        (self.slots[index].resident && self.slots[index].position == position).then_some(index)
    }
}

fn validate_config(config: LifecycleConfig) -> Result<(), LifecycleError> {
    if config.capacity_tokens == 0
    {
        return Err(LifecycleError::ZeroCapacity);
    }
    if config.hot_tokens.saturating_add(config.warm_tokens) > config.capacity_tokens
    {
        return Err(LifecycleError::InvalidWindows);
    }
    if config.hot.rank_divisor == 0 || config.warm.rank_divisor == 0 || config.cold.rank_divisor == 0
    {
        return Err(LifecycleError::ZeroRankDivisor);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        CacheTemperature, CompressionTier, LatentKvLifecycle, LifecycleAction, LifecycleConfig,
    };
    use crate::nn::latent_kv_cache::LatentStorageFormat;

    fn tier(format: LatentStorageFormat, slots: usize, divisor: usize) -> CompressionTier {
        CompressionTier {
            coefficient_format: format,
            residual_format: format,
            maximum_residual_slots: slots,
            rank_divisor: divisor,
        }
    }

    fn config() -> LifecycleConfig {
        LifecycleConfig {
            capacity_tokens: 6,
            hot_tokens: 2,
            warm_tokens: 2,
            hot: tier(LatentStorageFormat::F32, 4, 1),
            warm: tier(LatentStorageFormat::Int8, 2, 1),
            cold: tier(LatentStorageFormat::Int4, 1, 2),
        }
    }

    #[test]
    fn ring_evicts_oldest_without_growing() {
        let mut lifecycle = LatentKvLifecycle::new(config()).unwrap();
        for version in 0..6
        {
            assert!(lifecycle.admit(version).evicted.is_none());
        }
        let evicted = lifecycle.admit(9).evicted.unwrap();
        assert_eq!(evicted.position, 0);
        assert_eq!(lifecycle.len(), 6);
        assert!(lifecycle.get(0).is_none());
        assert_eq!(lifecycle.get(6).unwrap().basis_version, 9);
    }

    #[test]
    fn rebalance_emits_deterministic_temperature_transitions() {
        let mut lifecycle = LatentKvLifecycle::new(config()).unwrap();
        for version in 0..6
        {
            lifecycle.admit(version);
        }
        let placeholder = LifecycleAction {
            position: 0,
            basis_version: 0,
            from: CacheTemperature::Hot,
            to: CacheTemperature::Hot,
            target: config().hot,
        };
        let mut actions = [placeholder; 6];
        let count = lifecycle.rebalance_into(&mut actions);
        assert_eq!(count, 4);
        assert_eq!(lifecycle.get(0).unwrap().temperature, CacheTemperature::Cold);
        assert_eq!(lifecycle.get(2).unwrap().temperature, CacheTemperature::Warm);
        assert_eq!(lifecycle.get(5).unwrap().temperature, CacheTemperature::Hot);
    }

    #[test]
    fn basis_versions_survive_temperature_changes() {
        let mut lifecycle = LatentKvLifecycle::new(config()).unwrap();
        lifecycle.admit(42);
        for version in 1..6
        {
            lifecycle.admit(version);
        }
        let mut actions = [LifecycleAction {
            position: 0,
            basis_version: 0,
            from: CacheTemperature::Hot,
            to: CacheTemperature::Hot,
            target: config().hot,
        }; 6];
        lifecycle.rebalance_into(&mut actions);
        assert_eq!(lifecycle.get(0).unwrap().basis_version, 42);
    }
}

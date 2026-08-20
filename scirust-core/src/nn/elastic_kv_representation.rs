//! Representation metadata for elastic KV-cache materializations.
//!
//! This module is intentionally runtime-neutral.  It gives SciRust's KV-cache
//! implementations a deterministic vocabulary for describing *what* is stored
//! without depending on ElasticXxx, EPG, FLAT-ATTENTION, or a particular model.
//! Higher-level runtimes may map these values into their own policy/transition
//! graphs.

use core::fmt;

/// Monotonic version of one materialized KV representation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KvRepresentationEpoch(u64);

impl KvRepresentationEpoch {
    /// Construct an epoch from its raw counter.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the raw epoch counter.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Return the checked successor epoch.
    pub fn next(self) -> Result<Self, KvRepresentationError> {
        self.0
            .checked_add(1)
            .map(Self)
            .ok_or(KvRepresentationError::EpochOverflow)
    }
}

/// Stable, runtime-neutral representation contract identifier.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KvRepresentationId(String);

impl KvRepresentationId {
    /// Construct a non-empty identifier such as `raw.f32`, `slha.int4-r1`, or
    /// `epg.so4.structural`.
    pub fn new(value: impl Into<String>) -> Result<Self, KvRepresentationError> {
        let value = value.into();
        if value.trim().is_empty()
        {
            return Err(KvRepresentationError::EmptyRepresentationId);
        }
        Ok(Self(value))
    }

    /// Borrow the identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Relationship between a stored key and positional/structural transforms.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum KvKeyTransformScope {
    /// Key is stored before positional/structural transformation.
    #[default]
    Raw,
    /// Key is transformed only from token/page-stable metadata, so it remains
    /// reusable by arbitrary future queries.
    TokenStable,
    /// Key materialization depends on a particular query or future context.
    QueryDependent,
}

impl KvKeyTransformScope {
    /// Whether the materialization is reusable for arbitrary future queries.
    pub const fn reusable_for_future_queries(self) -> bool {
        !matches!(self, Self::QueryDependent)
    }
}

/// Metadata attached to a materialized KV tile/page.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvRepresentationMetadata {
    /// Stable representation family identifier.
    pub id: KvRepresentationId,
    /// Schema/contract version for the representation family.
    pub schema_version: u32,
    /// Epoch of the current materialization.
    pub epoch: KvRepresentationEpoch,
    /// Positional/structural transform scope of the stored key.
    pub key_transform_scope: KvKeyTransformScope,
}

impl KvRepresentationMetadata {
    /// Construct metadata.
    pub const fn new(
        id: KvRepresentationId,
        schema_version: u32,
        epoch: KvRepresentationEpoch,
        key_transform_scope: KvKeyTransformScope,
    ) -> Self {
        Self {
            id,
            schema_version,
            epoch,
            key_transform_scope,
        }
    }

    /// Whether this materialization can be reused by future queries.
    pub const fn reusable_for_future_queries(&self) -> bool {
        self.key_transform_scope.reusable_for_future_queries()
    }

    /// Whether another metadata value names the same representation contract,
    /// ignoring the materialization epoch.
    pub fn same_contract(&self, rhs: &Self) -> bool {
        self.id == rhs.id && self.schema_version == rhs.schema_version
    }

    /// Validate the epoch relationship for replacing this materialization with
    /// `target`.
    ///
    /// Changing the representation contract must advance the epoch.  Keeping
    /// the same contract may preserve or advance the epoch, but never regress.
    pub fn validate_successor(&self, target: &Self) -> Result<(), KvRepresentationError> {
        if !self.same_contract(target) && target.epoch <= self.epoch
        {
            return Err(KvRepresentationError::ContractChangeMustAdvanceEpoch {
                from: self.epoch,
                to: target.epoch,
            });
        }
        if self.same_contract(target) && target.epoch < self.epoch
        {
            return Err(KvRepresentationError::EpochRegression {
                from: self.epoch,
                to: target.epoch,
            });
        }
        Ok(())
    }
}

/// Invalid representation metadata or epoch transition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KvRepresentationError {
    /// Representation identifiers may not be blank.
    EmptyRepresentationId,
    /// Epoch counter cannot be incremented further.
    EpochOverflow,
    /// A contract-changing materialization did not advance the epoch.
    ContractChangeMustAdvanceEpoch {
        /// Source epoch.
        from: KvRepresentationEpoch,
        /// Target epoch.
        to: KvRepresentationEpoch,
    },
    /// A materialization regressed its epoch.
    EpochRegression {
        /// Source epoch.
        from: KvRepresentationEpoch,
        /// Target epoch.
        to: KvRepresentationEpoch,
    },
}

impl fmt::Display for KvRepresentationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyRepresentationId =>
            {
                write!(f, "KV representation identifier must not be empty")
            },
            Self::EpochOverflow => write!(f, "KV representation epoch overflow"),
            Self::ContractChangeMustAdvanceEpoch { from, to } => write!(
                f,
                "changing KV representation contract must advance epoch ({} -> {})",
                from.get(),
                to.get()
            ),
            Self::EpochRegression { from, to } => write!(
                f,
                "KV representation epoch regressed ({} -> {})",
                from.get(),
                to.get()
            ),
        }
    }
}

impl std::error::Error for KvRepresentationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(name: &str, epoch: u64, scope: KvKeyTransformScope) -> KvRepresentationMetadata {
        KvRepresentationMetadata::new(
            KvRepresentationId::new(name).unwrap(),
            1,
            KvRepresentationEpoch::new(epoch),
            scope,
        )
    }

    #[test]
    fn query_dependent_key_is_not_reusable() {
        let m = meta("epg.dynamic", 2, KvKeyTransformScope::QueryDependent);
        assert!(!m.reusable_for_future_queries());
        assert!(meta("epg.so4", 2, KvKeyTransformScope::TokenStable).reusable_for_future_queries());
    }

    #[test]
    fn contract_change_requires_new_epoch() {
        let from = meta("epg.so2", 7, KvKeyTransformScope::TokenStable);
        let bad = meta("epg.so4", 7, KvKeyTransformScope::TokenStable);
        let good = meta("epg.so4", 8, KvKeyTransformScope::TokenStable);
        assert!(matches!(
            from.validate_successor(&bad),
            Err(KvRepresentationError::ContractChangeMustAdvanceEpoch { .. })
        ));
        assert!(from.validate_successor(&good).is_ok());
    }
}

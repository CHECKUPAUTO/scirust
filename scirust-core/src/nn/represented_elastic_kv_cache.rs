//! Representation-aware wrapper around [`super::elastic_kv_cache::ElasticKvCache`].
//!
//! The existing cache API remains unchanged.  This wrapper adds explicit
//! materialization metadata and epoch-safe representation changes without
//! coupling SciRust to a policy runtime.

use super::elastic_kv_cache::ElasticKvCache;
use super::elastic_kv_representation::{KvRepresentationError, KvRepresentationMetadata};

/// Existing elastic compressed KV cache plus explicit representation metadata.
pub struct RepresentedElasticKvCache {
    cache: ElasticKvCache,
    representation: KvRepresentationMetadata,
}

impl RepresentedElasticKvCache {
    /// Construct a cache with a single scale per vector and explicit representation metadata.
    pub fn new(d: usize, budget: usize, representation: KvRepresentationMetadata) -> Self {
        Self {
            cache: ElasticKvCache::new(d, budget),
            representation,
        }
    }

    /// Construct a grouped-scale cache with explicit representation metadata.
    pub fn new_grouped(
        d: usize,
        budget: usize,
        group_size: usize,
        representation: KvRepresentationMetadata,
    ) -> Self {
        Self {
            cache: ElasticKvCache::new_grouped(d, budget, group_size),
            representation,
        }
    }

    /// Borrow the current materialization metadata.
    pub const fn representation(&self) -> &KvRepresentationMetadata {
        &self.representation
    }

    /// Replace only the representation metadata after the caller has actually
    /// performed the corresponding re-encode/recompute operation.
    ///
    /// This method validates epoch monotonicity; it deliberately does not
    /// pretend to transform cached bytes itself.
    pub fn commit_representation(
        &mut self,
        target: KvRepresentationMetadata,
    ) -> Result<(), KvRepresentationError> {
        self.representation.validate_successor(&target)?;
        self.representation = target;
        Ok(())
    }

    /// Append one key/value vector through the existing compressed cache codec.
    pub fn append(&mut self, k: &[f32], v: &[f32]) {
        self.cache.append(k, v);
    }

    /// Run attention using the existing deterministic compressed-cache implementation.
    pub fn attention(&self, q: &[f32]) -> Vec<f32> {
        self.cache.attention(q)
    }

    /// Number of resident tiles.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Number of evicted tiles.
    pub fn evicted(&self) -> usize {
        self.cache.evicted()
    }

    /// Current packed compressed footprint.
    pub fn compressed_bytes(&self) -> usize {
        self.cache.compressed_bytes()
    }

    /// Borrow the underlying legacy cache for integrations that do not yet
    /// consume representation metadata.
    pub const fn inner(&self) -> &ElasticKvCache {
        &self.cache
    }

    /// Mutably borrow the underlying legacy cache.
    pub fn inner_mut(&mut self) -> &mut ElasticKvCache {
        &mut self.cache
    }

    /// Consume the wrapper and return the legacy cache plus its final metadata.
    pub fn into_parts(self) -> (ElasticKvCache, KvRepresentationMetadata) {
        (self.cache, self.representation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::elastic_kv_representation::{
        KvKeyTransformScope, KvRepresentationEpoch, KvRepresentationId,
    };

    fn metadata(id: &str, epoch: u64) -> KvRepresentationMetadata {
        KvRepresentationMetadata::new(
            KvRepresentationId::new(id).unwrap(),
            1,
            KvRepresentationEpoch::new(epoch),
            KvKeyTransformScope::TokenStable,
        )
    }

    #[test]
    fn wrapper_preserves_legacy_cache_behaviour() {
        let mut cache = RepresentedElasticKvCache::new(4, 2, metadata("raw.f32", 0));
        cache.append(&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.attention(&[1.0, 0.0, 0.0, 0.0]).len(), 4);
    }

    #[test]
    fn contract_change_cannot_be_committed_without_new_epoch() {
        let mut cache = RepresentedElasticKvCache::new(4, 0, metadata("epg.so2", 4));
        assert!(cache.commit_representation(metadata("epg.so4", 4)).is_err());
        cache.commit_representation(metadata("epg.so4", 5)).unwrap();
        assert_eq!(cache.representation().id.as_str(), "epg.so4");
    }
}

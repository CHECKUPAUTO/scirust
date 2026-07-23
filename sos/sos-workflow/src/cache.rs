//! [`CacheKey`] — the content address of a stage invocation.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Digest, HashAlgo, ObjectId};

use crate::descriptor::StageDescriptor;

/// Domain-separation prefix for cache-key digests.
const CACHE_KEY_DOMAIN: &[u8] = b"sos-workflow:cache-key:v1";

/// The content-addressed key of a single stage invocation (RFC-0002 §08.2) —
/// the mechanism that unifies **reproducibility** and **incremental compute**:
///
/// ```text
/// cache_key = hash( descriptor          // plugin name + version + content hash
///                 ⊕ input_object_ids    // the exact nodes consumed (as a set)
///                 ⊕ config_hash         // the stage configuration
///                 ⊕ seed                // the mandatory seed
///                 ⊕ env_digest )        // toolchain / backend / hardware class
/// ```
///
/// If a `CacheKey` is already known, the stage does not run — its outputs are
/// returned from cache, so re-running an unchanged workflow is provably identical
/// and nearly free. Change **anything** in the key and that stage re-runs.
///
/// Inputs are hashed as a **set** (sorted + deduplicated), so a stage that reads
/// the same nodes in a different order is the same invocation; a stage whose
/// order is semantically significant folds that order into its `config_hash`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CacheKey(Digest);

impl CacheKey {
    /// Compute the cache key of a stage invocation from its ingredients.
    #[must_use]
    pub fn compute(
        descriptor: &StageDescriptor,
        inputs: &[ObjectId],
        config_hash: &Digest,
        seed: u64,
        env_digest: &Digest,
    ) -> Self {
        let mut ins = inputs.to_vec();
        ins.sort_unstable();
        ins.dedup();

        let mut enc = CanonicalEncoder::new();
        enc.value(descriptor);
        enc.seq(&ins);
        enc.bytes(config_hash.as_bytes());
        enc.u64(seed);
        enc.bytes(env_digest.as_bytes());

        Self(HashAlgo::Sha256.hash(CACHE_KEY_DOMAIN, &enc.finish()))
    }

    /// The underlying digest.
    #[must_use]
    pub fn digest(&self) -> &Digest {
        &self.0
    }
}

impl Canonical for CacheKey {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.bytes(self.0.as_bytes());
    }
}

impl core::fmt::Display for CacheKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "cachekey:{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sos_core::SemVer;

    fn digest(tag: &[u8]) -> Digest {
        HashAlgo::default().hash(b"test", tag)
    }

    fn desc() -> StageDescriptor {
        StageDescriptor::new("s", SemVer::new(1, 0, 0), digest(b"plugin"))
    }

    fn oid(tag: &[u8]) -> ObjectId {
        ObjectId::compute(HashAlgo::default(), b"sos-obj:N:v1", tag)
    }

    #[test]
    fn identical_ingredients_hit() {
        let (cfg, env) = (digest(b"cfg"), digest(b"env"));
        let a = CacheKey::compute(&desc(), &[oid(b"x"), oid(b"y")], &cfg, 7, &env);
        let b = CacheKey::compute(&desc(), &[oid(b"x"), oid(b"y")], &cfg, 7, &env);
        assert_eq!(a, b);
    }

    #[test]
    fn input_order_and_duplication_do_not_matter() {
        let (cfg, env) = (digest(b"cfg"), digest(b"env"));
        let a = CacheKey::compute(&desc(), &[oid(b"x"), oid(b"y")], &cfg, 7, &env);
        let b = CacheKey::compute(&desc(), &[oid(b"y"), oid(b"x"), oid(b"x")], &cfg, 7, &env);
        assert_eq!(a, b); // inputs are a set
    }

    #[test]
    fn every_ingredient_changes_the_key() {
        let (cfg, env) = (digest(b"cfg"), digest(b"env"));
        let base = CacheKey::compute(&desc(), &[oid(b"x")], &cfg, 7, &env);
        // different seed
        assert_ne!(
            base,
            CacheKey::compute(&desc(), &[oid(b"x")], &cfg, 8, &env)
        );
        // different config
        assert_ne!(
            base,
            CacheKey::compute(&desc(), &[oid(b"x")], &digest(b"cfg2"), 7, &env)
        );
        // different env
        assert_ne!(
            base,
            CacheKey::compute(&desc(), &[oid(b"x")], &cfg, 7, &digest(b"env2"))
        );
        // different inputs
        assert_ne!(
            base,
            CacheKey::compute(&desc(), &[oid(b"z")], &cfg, 7, &env)
        );
    }
}

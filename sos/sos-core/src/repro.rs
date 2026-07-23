//! Reproducibility metadata — the traceability list the mandate requires.
//!
//! Every object that touches computation carries a [`ReproMeta`]: a **mandatory
//! seed** (the `scirust-bench-schema` rule — a result whose randomness cannot be
//! reproduced is not a reproducible artifact), the RNG algorithm, the content
//! ids of its data inputs, and an [`EnvRecord`] digest capturing toolchain,
//! backends, hardware class, and OS. Together these are the reproducibility key
//! that `sos-repro` pins and re-realizes (RFC-0002 §09).

use serde::{Deserialize, Serialize};

use crate::canonical::{Canonical, CanonicalEncoder};
use crate::hash::{Digest, HashAlgo};
use crate::id::ObjectId;
use crate::version::SemVer;

/// Identifier of the pseudo-random generator whose seed reproduces an object's
/// randomness. A free-form name (e.g. `"SplitMix64"`, `"PCG64"`,
/// `"XorShift64"`) so any deterministic generator can be recorded without a
/// kernel change.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RngId(pub String);

impl RngId {
    /// Construct an RNG identifier.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

impl Canonical for RngId {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.0);
    }
}

/// The exact version of a computational backend used to produce an object.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BackendVersion {
    /// Backend crate/plugin name, e.g. `"scirust-solvers"`.
    pub name: String,
    /// Backend semantic version.
    pub version: SemVer,
    /// Content hash of the backend artifact, pinning the exact code.
    pub content_hash: Digest,
}

impl BackendVersion {
    /// Construct a backend-version record.
    #[must_use]
    pub fn new(name: impl Into<String>, version: SemVer, content_hash: Digest) -> Self {
        Self {
            name: name.into(),
            version,
            content_hash,
        }
    }
}

impl Canonical for BackendVersion {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.name);
        enc.value(&self.version);
        enc.bytes(self.content_hash.as_bytes());
    }
}

/// The environment an object was produced in. Hashed to an `env_digest` and
/// referenced from [`ReproMeta`]; `sos-repro` treats the digest as a lockfile.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EnvRecord {
    /// Rust toolchain, e.g. `"1.89.0-stable"` or a pinned nightly date.
    pub toolchain: String,
    /// The computational backends and their exact versions.
    pub backends: Vec<BackendVersion>,
    /// Hardware class: ISA, CPU/GPU model, BLAS impl, thread count.
    pub hardware: String,
    /// Operating system identifier.
    pub os: String,
}

impl EnvRecord {
    /// Construct an environment record.
    #[must_use]
    pub fn new(
        toolchain: impl Into<String>,
        backends: Vec<BackendVersion>,
        hardware: impl Into<String>,
        os: impl Into<String>,
    ) -> Self {
        Self {
            toolchain: toolchain.into(),
            backends,
            hardware: hardware.into(),
            os: os.into(),
        }
    }

    /// The content digest of this environment record — the reproducibility key.
    #[must_use]
    pub fn digest(&self, algo: HashAlgo) -> Digest {
        algo.hash(b"sos-env:v1", &self.canonical_bytes())
    }
}

impl Canonical for EnvRecord {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.toolchain);
        enc.seq(&self.backends);
        enc.str(&self.hardware);
        enc.str(&self.os);
    }
}

/// The reproducibility metadata attached to every computational object.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReproMeta {
    /// The seed that reproduces this object's randomness. **Mandatory** — a
    /// value of `0` is a deliberate declaration of "no randomness / fixed
    /// seed", never an accident.
    pub seed: u64,
    /// The RNG algorithm the seed drives.
    pub rng: RngId,
    /// Digest of the [`EnvRecord`] this object was produced in.
    pub env_digest: Digest,
    /// Content ids of the data inputs (datasets, parameters, equations) this
    /// object was derived from — the transitive-provenance leaves.
    pub inputs: Vec<ObjectId>,
}

impl ReproMeta {
    /// Construct reproducibility metadata with the mandatory seed and
    /// environment digest. Inputs default to empty; add them with
    /// [`ReproMeta::with_inputs`].
    #[must_use]
    pub fn new(seed: u64, rng: RngId, env_digest: Digest) -> Self {
        Self {
            seed,
            rng,
            env_digest,
            inputs: Vec::new(),
        }
    }

    /// Attach the content ids of this object's data inputs.
    #[must_use]
    pub fn with_inputs(mut self, inputs: Vec<ObjectId>) -> Self {
        self.inputs = inputs;
        self
    }
}

impl Canonical for ReproMeta {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.u64(self.seed);
        enc.value(&self.rng);
        enc.bytes(self.env_digest.as_bytes());
        enc.seq(&self.inputs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env() -> EnvRecord {
        EnvRecord::new(
            "1.89.0-stable",
            vec![BackendVersion::new(
                "scirust-solvers",
                SemVer::new(0, 1, 0),
                HashAlgo::Sha256.hash(b"b", b"solvers"),
            )],
            "x86_64/avx2/openblas",
            "linux",
        )
    }

    #[test]
    fn env_digest_is_deterministic_and_content_sensitive() {
        let e = env();
        assert_eq!(
            e.digest(HashAlgo::Sha256),
            e.clone().digest(HashAlgo::Sha256)
        );
        let mut e2 = env();
        e2.hardware = "aarch64/neon".into();
        assert_ne!(e.digest(HashAlgo::Sha256), e2.digest(HashAlgo::Sha256));
    }

    #[test]
    fn seed_changes_repro_identity() {
        let ed = env().digest(HashAlgo::Sha256);
        let a = ReproMeta::new(1, RngId::new("SplitMix64"), ed);
        let b = ReproMeta::new(2, RngId::new("SplitMix64"), ed);
        assert_ne!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn inputs_change_repro_identity() {
        let ed = env().digest(HashAlgo::Sha256);
        let base = ReproMeta::new(1, RngId::new("SplitMix64"), ed);
        let with = base
            .clone()
            .with_inputs(vec![ObjectId::of(HashAlgo::Sha256, b"d", &1u64)]);
        assert_ne!(base.canonical_bytes(), with.canonical_bytes());
    }
}

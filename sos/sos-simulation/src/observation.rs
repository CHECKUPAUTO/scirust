//! [`Observation`] — a simulation result with its honest determinism level.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{DeterminismLevel, Digest, HashAlgo};

/// Domain-separation prefix for observation digests.
const OBSERVATION_DOMAIN: &[u8] = b"sos-sim:observation:v1";

/// The recorded result of a simulation run: the `output`, the
/// [`DeterminismLevel`] the backend **realized**, and the `seed` that produced
/// it (RFC-0002 §08.5–6).
///
/// The level is not decoration — it is the reproducibility contract of this
/// result: an `L3` observation reproduces to the bit, an `L2` within its
/// certificate, an `L1` in distribution given the seed. Stamping it here means a
/// study built on an `L1` simulation *says so*, and no result is ever presented
/// as more reproducible than its backend allows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Observation<T> {
    /// The simulation output.
    pub output: T,
    /// The determinism level the backend realized for this run.
    pub level: DeterminismLevel,
    /// The seed that drove the run — mandatory, so the run is reproducible.
    pub seed: u64,
}

impl<T> Observation<T> {
    /// Record an observation with its realized determinism level and seed.
    pub fn new(output: T, level: DeterminismLevel, seed: u64) -> Self {
        Self {
            output,
            level,
            seed,
        }
    }

    /// The determinism level this observation realized.
    #[must_use]
    pub fn level(&self) -> DeterminismLevel {
        self.level
    }
}

impl<T: Canonical> Observation<T> {
    /// The content digest of this observation — its content address. Because the
    /// determinism level and seed are hashed in, an `L2` result and an otherwise
    /// identical `L3` result are distinct objects.
    #[must_use]
    pub fn digest(&self) -> Digest {
        HashAlgo::Sha256.hash(OBSERVATION_DOMAIN, &self.canonical_bytes())
    }
}

impl<T: Canonical> Canonical for Observation<T> {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.output);
        enc.value(&self.level);
        enc.u64(self.seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_and_seed_are_part_of_identity() {
        let a = Observation::new(42u64, DeterminismLevel::L3, 7);
        assert_eq!(
            a.digest(),
            Observation::new(42u64, DeterminismLevel::L3, 7).digest()
        );
        // Same output, different level ⇒ different content address.
        assert_ne!(
            a.digest(),
            Observation::new(42u64, DeterminismLevel::L2, 7).digest()
        );
        // Same output/level, different seed ⇒ different content address.
        assert_ne!(
            a.digest(),
            Observation::new(42u64, DeterminismLevel::L3, 8).digest()
        );
    }

    #[test]
    fn serde_roundtrips() {
        let o = Observation::new(vec![1u64, 2, 3], DeterminismLevel::L1, 99);
        let j = serde_json::to_string(&o).unwrap();
        let back: Observation<Vec<u64>> = serde_json::from_str(&j).unwrap();
        assert_eq!(o, back);
    }
}

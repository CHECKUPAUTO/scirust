//! [`Vcr`] — the record/replay memo that makes a simulation run reproducible and
//! nearly free to repeat.

use std::collections::BTreeMap;

use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Digest, HashAlgo};

use crate::error::SimError;
use crate::observation::Observation;
use crate::simulate::{SimDescriptor, Simulate};

/// Domain-separation prefix for a run's record/replay key.
const VCR_KEY_DOMAIN: &[u8] = b"sos-sim:vcr-key:v1";

/// The content-addressed key of a simulation run: `hash(descriptor ⊕ config ⊕
/// seed)`. Identical `(backend, config, seed)` ⇒ identical key ⇒ a replay.
#[must_use]
fn run_key<C: Canonical>(descriptor: &SimDescriptor, config: &C, seed: u64) -> Digest {
    let mut enc = CanonicalEncoder::new();
    enc.value(descriptor);
    enc.value(config);
    enc.u64(seed);
    HashAlgo::Sha256.hash(VCR_KEY_DOMAIN, &enc.finish())
}

/// The result of an [`Vcr::observe`]: the observation, and whether it came from a
/// prior recording (`replayed`) rather than a fresh run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recorded<T> {
    /// The observation, whether freshly run or replayed — identical either way.
    pub observation: Observation<T>,
    /// `true` if this observation was replayed from a recording; `false` if the
    /// backend was run for it.
    pub replayed: bool,
}

/// A **record/replay** memo over one simulation backend — the "VCR" of the effect
/// boundary (RFC-0002 §08.4): in *record* mode a run is performed once and its
/// [`Observation`] stored; a later run with the same `(config, seed)` is
/// *replayed* from that recording rather than recomputed. This is what lets an
/// expensive or one-shot simulation live inside a reproducible workflow — the
/// replay is identical and nearly free.
///
/// The world-touching effect boundary's **capability authorization** (signing,
/// least-privilege) is enforced by the Workflow Engine's executor seam
/// (`sos-workflow` + `sos-registry`), not re-implemented here; this VCR is the
/// deterministic record/replay half.
pub struct Vcr<S: Simulate> {
    seen: BTreeMap<Digest, Observation<S::Output>>,
}

impl<S: Simulate> Vcr<S> {
    /// A fresh, empty VCR.
    #[must_use]
    pub fn new() -> Self {
        Self {
            seen: BTreeMap::new(),
        }
    }

    /// How many distinct runs have been recorded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    /// Whether nothing has been recorded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

impl<S: Simulate> Vcr<S>
where
    S::Output: Clone,
{
    /// Observe `sim` at `(config, seed)`: **replay** the recorded observation if
    /// this run was already performed, otherwise **run** the backend once and
    /// record it.
    ///
    /// # Errors
    /// Propagates a [`SimError`] from the backend on a fresh run. A replay never
    /// fails (it returns the stored recording).
    pub fn observe(
        &mut self,
        sim: &S,
        config: &S::Config,
        seed: u64,
    ) -> Result<Recorded<S::Output>, SimError> {
        let key = run_key(&sim.descriptor(), config, seed);
        if let Some(recorded) = self.seen.get(&key)
        {
            return Ok(Recorded {
                observation: recorded.clone(),
                replayed: true,
            });
        }
        let observation = sim.run(config, seed)?;
        self.seen.insert(key, observation.clone());
        Ok(Recorded {
            observation,
            replayed: false,
        })
    }
}

impl<S: Simulate> Default for Vcr<S> {
    fn default() -> Self {
        Self::new()
    }
}

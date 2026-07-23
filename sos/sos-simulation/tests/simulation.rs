//! End-to-end simulation: backend-independent `Simulate` backends of differing
//! determinism levels, honest level stamping, and record/replay memoization.

use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{DeterminismLevel, SemVer};
use sos_simulation::{Observation, SimDescriptor, SimError, Simulate, Vcr};

/// An exact, bit-reproducible (`L3`) integer simulation.
struct Summation;

struct Range {
    n: u64,
}

impl Canonical for Range {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.u64(self.n);
    }
}

impl Simulate for Summation {
    type Config = Range;
    type Output = u64;

    fn descriptor(&self) -> SimDescriptor {
        SimDescriptor::new("summation", SemVer::new(1, 0, 0))
    }

    fn level(&self) -> DeterminismLevel {
        DeterminismLevel::L3
    }

    fn run(&self, config: &Range, seed: u64) -> Result<Observation<u64>, SimError> {
        if config.n > 1_000_000
        {
            return Err(SimError::InvalidConfig("n too large".into()));
        }
        Ok(Observation::new(
            (0..config.n).sum::<u64>().wrapping_add(seed),
            self.level(),
            seed,
        ))
    }
}

/// A seeded-stochastic (`L1`) simulation: reproducible in distribution given the
/// seed. (A deterministic PRNG stands in so the test itself is stable.)
struct SeededDraw;

impl Simulate for SeededDraw {
    type Config = Range;
    type Output = u64;

    fn descriptor(&self) -> SimDescriptor {
        SimDescriptor::new("seeded-draw", SemVer::new(1, 0, 0))
    }

    fn level(&self) -> DeterminismLevel {
        DeterminismLevel::L1
    }

    fn run(&self, config: &Range, seed: u64) -> Result<Observation<u64>, SimError> {
        // A SplitMix64-style step — deterministic given the seed, declared L1.
        let mut z = seed
            .wrapping_add(config.n)
            .wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Ok(Observation::new(z ^ (z >> 31), self.level(), seed))
    }
}

#[test]
fn a_backend_stamps_its_declared_level_on_every_observation() {
    let l3 = Summation.run(&Range { n: 4 }, 0).unwrap();
    assert_eq!(l3.output, 6);
    assert_eq!(l3.level(), DeterminismLevel::L3);

    let l1 = SeededDraw.run(&Range { n: 4 }, 0).unwrap();
    assert_eq!(l1.level(), DeterminismLevel::L1);
}

#[test]
fn l3_is_bit_reproducible_and_seed_sensitive() {
    let a = Summation.run(&Range { n: 100 }, 7).unwrap();
    let b = Summation.run(&Range { n: 100 }, 7).unwrap();
    assert_eq!(a, b); // bit-identical
    assert_eq!(a.digest(), b.digest());

    let c = Summation.run(&Range { n: 100 }, 8).unwrap();
    assert_ne!(a.digest(), c.digest()); // the seed is part of the result identity
}

#[test]
fn the_vcr_records_then_replays() {
    let sim = Summation;
    let mut vcr = Vcr::new();
    assert!(vcr.is_empty());

    let first = vcr.observe(&sim, &Range { n: 10 }, 3).unwrap();
    assert!(!first.replayed);
    assert_eq!(vcr.len(), 1);

    let replay = vcr.observe(&sim, &Range { n: 10 }, 3).unwrap();
    assert!(replay.replayed);
    assert_eq!(replay.observation, first.observation); // identical, and free
    assert_eq!(vcr.len(), 1); // no new recording
}

#[test]
fn a_different_config_or_seed_is_a_fresh_run() {
    let sim = Summation;
    let mut vcr = Vcr::new();
    let _ = vcr.observe(&sim, &Range { n: 10 }, 3).unwrap();

    // Different seed ⇒ not a replay.
    let other_seed = vcr.observe(&sim, &Range { n: 10 }, 4).unwrap();
    assert!(!other_seed.replayed);
    // Different config ⇒ not a replay.
    let other_cfg = vcr.observe(&sim, &Range { n: 11 }, 3).unwrap();
    assert!(!other_cfg.replayed);
    assert_eq!(vcr.len(), 3);
}

#[test]
fn invalid_config_is_a_clean_error() {
    let err = Summation.run(&Range { n: 2_000_000 }, 0);
    assert!(matches!(err, Err(SimError::InvalidConfig(_))));
}

//! Property-based integration tests for the `sos-core` kernel invariants.
//!
//! These exercise the whole envelope over many seeded-random inputs, using a
//! small deterministic generator (SplitMix64) so the tests themselves are
//! bit-reproducible — the property the kernel is *about*. No external test
//! framework and no randomness that isn't seeded.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};
use sos_core::{Author, Body, DeterminismLevel, Object, ObjectId};

/// Deterministic SplitMix64 — fixed-seed pseudo-randomness for reproducible
/// property tests (mirrors the seeded generators used across the workspace).
struct SplitMix64(u64);
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// A small domain body for testing: a labelled series of integer ticks.
#[derive(Clone, Serialize, Deserialize)]
struct Datum {
    label: String,
    ticks: Vec<u64>,
}

impl Canonical for Datum {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.label);
        enc.seq(&self.ticks);
    }
}
impl Body for Datum {
    const KIND: &'static str = "Datum";
    const SCHEMA_VERSION: u32 = 1;
}

fn seal(d: Datum) -> Object<Datum> {
    Object::builder(d).author(Author::human("tester")).seal()
}

fn random_datum(rng: &mut SplitMix64) -> Datum {
    let n = (rng.next_u64() % 6) as usize;
    let ticks = (0..n).map(|_| rng.next_u64()).collect();
    Datum {
        label: format!("d{}", rng.next_u64()),
        ticks,
    }
}

#[test]
fn sealing_is_reproducible_over_many_inputs() {
    let mut rng = SplitMix64::new(0x1234_5678_9ABC_DEF0); // fixed seed
    for _ in 0..500
    {
        let d = random_datum(&mut rng);
        // Sealing the SAME content twice yields the SAME id (determinism).
        assert_eq!(seal(d.clone()).id, seal(d).id);
    }
}

#[test]
fn distinct_content_yields_distinct_ids_no_collisions() {
    let mut rng = SplitMix64::new(0xC0FF_EE00_1234_5678);
    let mut seen_labels = HashSet::new();
    let mut ids = HashSet::new();
    let mut count = 0usize;
    for _ in 0..1000
    {
        let d = random_datum(&mut rng);
        // Only compare ids across *distinct* contents (dedup on the exact body).
        let key = (d.label.clone(), d.ticks.clone());
        if !seen_labels.insert(key)
        {
            continue;
        }
        let id = seal(d).id;
        assert!(ids.insert(id), "content-hash collision detected");
        count += 1;
    }
    assert!(count > 100, "sanity: exercised a meaningful sample");
}

#[test]
fn every_sealed_object_verifies_and_roundtrips() {
    let mut rng = SplitMix64::new(42);
    for _ in 0..300
    {
        let obj = seal(random_datum(&mut rng));
        assert!(obj.verify_id());
        let json = serde_json::to_string(&obj).unwrap();
        let back: Object<Datum> = serde_json::from_str(&json).unwrap();
        assert_eq!(obj.id, back.id);
        assert!(back.verify_id());
        back.check_id().unwrap();
    }
}

#[test]
fn changing_parents_always_changes_id() {
    let mut rng = SplitMix64::new(7);
    for _ in 0..300
    {
        let d = random_datum(&mut rng);
        let p1 = seal(random_datum(&mut rng)).id;
        let p2 = seal(random_datum(&mut rng)).id;
        let a = Object::builder(d.clone())
            .author(Author::human("t"))
            .parents(vec![p1])
            .seal();
        let b = Object::builder(d)
            .author(Author::human("t"))
            .parents(vec![p2])
            .seal();
        // Distinct parents (p1 != p2 with overwhelming probability here) must
        // produce distinct ids — the Merkle-DAG property.
        if p1 != p2
        {
            assert_ne!(a.id, b.id);
        }
    }
}

#[test]
fn determinism_level_min_over_matches_reference() {
    use DeterminismLevel::{L0, L1, L2, L3};
    let all = [L0, L1, L2, L3];
    let mut rng = SplitMix64::new(99);
    for _ in 0..2000
    {
        let n = (rng.next_u64() % 5) as usize;
        let sample: Vec<DeterminismLevel> =
            (0..n).map(|_| all[(rng.next_u64() % 4) as usize]).collect();
        let got = DeterminismLevel::min_over(sample.iter().copied());
        // Reference: the ordinary minimum, or L3 (identity) for an empty set.
        let want = sample.iter().copied().min().unwrap_or(L3);
        assert_eq!(got, want);
    }
}

#[test]
fn ids_sort_deterministically() {
    // ObjectId is Ord (by digest bytes); SOS relies on this for canonical
    // tie-breaking. Sorting the same set twice yields the same order.
    let mut rng = SplitMix64::new(0xABCD);
    let mut ids: Vec<ObjectId> = (0..200).map(|_| seal(random_datum(&mut rng)).id).collect();
    let mut a = ids.clone();
    a.sort();
    ids.sort();
    assert_eq!(a, ids);
}

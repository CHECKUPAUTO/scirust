//! Phase 2 head-to-head: `S16BoundedMemory` vs the workspace's existing
//! `scirust_retrieval::BoundedSemanticMemory`, on equivalent scripted
//! workloads.
//!
//! The two stores differ by design (logical `u64` ticks + generation-safe ids
//! vs `f64` timestamps + raw `u64` ids; `LinearDecay` here vs `DecaySchedule`
//! there), so the comparison is behavioural, not bit-wise: for the same
//! insertion order, importances, timestamps, and accesses, both must retain the
//! same *concepts* and evict the same *victims*. That demonstrates the Phase 2
//! layer composes the Phase 1 store into the same bounded-memory semantics the
//! workspace already trusts, rather than inventing a divergent policy.

use scirust_hypermemory::{
    ConceptSpec, LinearDecay, NoForgetting, S16BoundedMemory, SimilarityMetric,
};
use scirust_retrieval::{BoundedSemanticMemory, DecaySchedule};
use scirust_simd::hypercomplex::SedenionSimd;

/// A deterministic 16-lane vector for document `i` (distinct basis directions,
/// so cosine search is unambiguous).
fn vec16(i: usize) -> [f32; 16] {
    let mut v = [0.0f32; 16];
    v[i % 16] = 1.0 + (i / 16) as f32;
    v
}

#[test]
fn no_decay_eviction_matches_bounded_semantic_memory() {
    // Importance-only retention (NoForgetting ↔ DecaySchedule::None with
    // recency 1 for everyone — recency then cancels in comparisons, leaving
    // importance ordering identical). Insert 6 docs of distinct importance into
    // capacity-4 stores: both must evict the two least important.
    let importances = [0.9f32, 0.1, 0.7, 0.3, 0.5, 0.8];

    // Ours.
    let mut ours = S16BoundedMemory::new(4, SimilarityMetric::Cosine, NoForgetting);
    let mut our_ids = Vec::new();
    for (i, &imp) in importances.iter().enumerate()
    {
        let spec = ConceptSpec::new(
            vec![i as u8],
            SedenionSimd::from_array(vec16(i)),
            imp,
            i as u64,
        );
        our_ids.push(ours.insert(spec).unwrap().inserted);
    }

    // Theirs.
    let mut theirs = BoundedSemanticMemory::new(16, 4, DecaySchedule::None);
    for (i, &imp) in importances.iter().enumerate()
    {
        theirs.add(i as u64, &vec16(i), imp, i as f64).unwrap();
    }

    // Surviving concept tags, in both stores.
    let our_survivors: Vec<u8> = ours.iter().map(|r| r.payload()[0]).collect();
    let their_survivors: Vec<u64> = theirs.meta().iter().map(|m| m.id).collect();

    // Both must have evicted docs 1 (imp 0.1) and 3 (imp 0.3).
    let mut ours_sorted = our_survivors.clone();
    ours_sorted.sort_unstable();
    let mut theirs_sorted: Vec<u8> = their_survivors.iter().map(|&x| x as u8).collect();
    theirs_sorted.sort_unstable();
    assert_eq!(
        ours_sorted,
        vec![0, 2, 4, 5],
        "ours must keep the 4 most important"
    );
    assert_eq!(
        ours_sorted, theirs_sorted,
        "S16BoundedMemory and BoundedSemanticMemory must retain the same concepts"
    );
}

#[test]
fn recency_decay_protects_accessed_documents_in_both() {
    // Equal importances; one document is accessed late. Under linear decay with
    // the same half-life (ours in ticks, theirs in the same numeric units),
    // inserting one more must evict a *stale* document in both stores — never
    // the freshly accessed one.
    let half_life = 100u64;

    // Ours: 3 docs at tick 0, capacity 3.
    let mut ours = S16BoundedMemory::new(3, SimilarityMetric::Cosine, LinearDecay::new(half_life));
    let mut our_ids = Vec::new();
    for i in 0..3usize
    {
        let spec = ConceptSpec::new(vec![i as u8], SedenionSimd::from_array(vec16(i)), 1.0, 0);
        our_ids.push(ours.insert(spec).unwrap().inserted);
    }
    // Access doc 1 at tick 90 (a cosine query aligned with its direction).
    let q = SedenionSimd::from_array(vec16(1));
    let hits = ours.search(&q, 1, 90).unwrap();
    assert_eq!(ours.get(hits[0].id).unwrap().payload()[0], 1);
    // Insert doc 3 at tick 95 → doc 0 or 2 must be evicted (stale), never doc 1.
    let ins = ours
        .insert(ConceptSpec::new(
            vec![3],
            SedenionSimd::from_array(vec16(3)),
            1.0,
            95,
        ))
        .unwrap();
    let evicted_ours = ours
        .iter()
        .map(|r| r.payload()[0])
        .fold([true; 4], |mut gone, tag| {
            gone[tag as usize] = false;
            gone
        });
    assert!(ins.evicted.is_some());
    assert!(
        !evicted_ours[1],
        "the freshly accessed doc must survive (ours)"
    );
    assert!(!evicted_ours[3], "the new doc must reside (ours)");

    // Theirs: same script with Linear decay and f64 timestamps.
    let mut theirs = BoundedSemanticMemory::new(
        16,
        3,
        DecaySchedule::Linear {
            half_life: half_life as f64,
        },
    );
    for i in 0..3usize
    {
        theirs.add(i as u64, &vec16(i), 1.0, 0.0).unwrap();
    }
    // Access doc 1 at t=90 (search touches it).
    let their_hits = theirs.search(&vec16(1), 1, 90.0);
    assert_eq!(their_hits[0].id, 1);
    theirs.add(3, &vec16(3), 1.0, 95.0).unwrap();
    let their_survivors: Vec<u64> = theirs.meta().iter().map(|m| m.id).collect();
    assert!(
        their_survivors.contains(&1),
        "the freshly accessed doc must survive (theirs)"
    );
    assert!(
        their_survivors.contains(&3),
        "the new doc must reside (theirs)"
    );

    // Cross-check: the same doc (1) survives in both.
    let our_survivors: Vec<u8> = ours.iter().map(|r| r.payload()[0]).collect();
    assert!(our_survivors.contains(&1));
}

#[test]
fn scripted_workload_is_reproducible() {
    // The whole bounded workflow — inserts, searches, evictions, forgetting —
    // repeated twice must produce identical survivor sets and identical hits.
    fn run() -> (Vec<u8>, Vec<u8>) {
        let mut mem = S16BoundedMemory::new(4, SimilarityMetric::Cosine, LinearDecay::new(50));
        for i in 0..8usize
        {
            let spec = ConceptSpec::new(
                vec![i as u8],
                SedenionSimd::from_array(vec16(i)),
                1.0 + (i % 3) as f32,
                i as u64 * 10,
            );
            let _ = mem.insert(spec).unwrap();
        }
        let q = SedenionSimd::from_array(vec16(5));
        let hits = mem.search(&q, 2, 100).unwrap();
        let hit_tags: Vec<u8> = hits
            .iter()
            .map(|h| mem.get(h.id).unwrap().payload()[0])
            .collect();
        let _ = mem.forget(200, 1.5);
        let survivors: Vec<u8> = mem.iter().map(|r| r.payload()[0]).collect();
        (survivors, hit_tags)
    }
    assert_eq!(run(), run(), "bounded workload must be reproducible");
}

//! Phase 3 — F4 falsification harness for bounded residual learning.
//!
//! F4 (research doc §10): *"residual learning destabilizes old concepts — if
//! updating one concept's residual measurably perturbs the effective vectors or
//! rankings of unrelated concepts beyond a stated tolerance, the residual
//! mechanism is rejected."*
//!
//! The Phase 1 design makes F4 provable rather than merely probable: each
//! record's effective vector depends only on its **own** anchor + residual, so
//! a learning step is per-record isolated by construction. These tests pin that
//! down **exactly** (bit-identical, not within a tolerance):
//!
//! * updating one concept leaves every other record's effective vector
//!   bit-identical;
//! * for any query, the relative ranking of all *other* concepts is unchanged
//!   (their scores and the id tie-break are untouched — only the updated
//!   concept may move);
//! * drift from the immutable anchor is capped by the residual bound, no
//!   matter how many steps run;
//! * the whole learning trajectory is deterministic.

use scirust_hypermemory::{
    ConceptId, ConceptSpec, LinearDecay, S16BoundedMemory, S16ExactIndex, S16Store, SearchHit,
    SimilarityMetric,
};
use scirust_simd::hypercomplex::SedenionSimd;

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }

    fn nonzero_sedenion(&mut self) -> SedenionSimd {
        loop
        {
            let mut a = [0.0f32; 16];
            for lane in &mut a
            {
                *lane = self.next_f32();
            }
            let s = SedenionSimd::from_array(a);
            if s.to_array().iter().any(|x| *x != 0.0)
            {
                return s;
            }
        }
    }
}

fn seeded_store(n: u32, seed: u64) -> (S16Store, Vec<ConceptId>) {
    let mut rng = Lcg::new(seed);
    let mut store = S16Store::new();
    let mut ids = Vec::new();
    for i in 0..n
    {
        ids.push(
            store
                .insert(ConceptSpec::new(
                    vec![i as u8],
                    rng.nonzero_sedenion(),
                    1.0,
                    0,
                ))
                .unwrap(),
        );
    }
    (store, ids)
}

// ------------------------------------------------------------------ //
//  Learning does what it claims (before the F4 checks)                //
// ------------------------------------------------------------------ //

#[test]
fn learning_moves_the_effective_vector_toward_the_target() {
    let (mut store, ids) = seeded_store(4, 41);
    let target = SedenionSimd::unit(7);
    let first = store.learn_residual(ids[0], &target, 0.2).unwrap();
    assert!(
        first.similarity_after > first.similarity_before,
        "one step must increase similarity: {first:?}"
    );
    // Repeated steps: the trajectory rises, may pass through a transient peak
    // once the residual-norm clamp engages (the clamped update+rescale
    // iteration converges to a fixed point that can sit slightly below the
    // peak — measured, documented dynamics of this rule, not hidden), and
    // settles. The honest invariants: clear net improvement over the start,
    // and near-stationarity at the end. The ceiling is geometric: with
    // ‖anchor‖ ≈ 2.3 (random dense) and residual bound 1.0, the best cosine
    // to a near-orthogonal target is limited — the bound is exactly what
    // keeps the concept anchored (F4's point).
    let mut last = first;
    for _ in 0..100
    {
        last = store.learn_residual(ids[0], &target, 0.2).unwrap();
    }
    assert!(
        last.similarity_after > first.similarity_before + 0.1,
        "after 100 bounded steps similarity must clearly improve: {} -> {}",
        first.similarity_before,
        last.similarity_after
    );
    assert!(
        (last.similarity_after - last.similarity_before).abs() < 5e-3,
        "the trajectory must be near-stationary at the end: {last:?}"
    );
}

#[test]
fn invalid_rate_or_target_is_rejected_and_state_unchanged() {
    let (mut store, ids) = seeded_store(2, 42);
    let before = store.get(ids[0]).unwrap().clone();
    for bad_rate in [-0.1f32, 1.5, f32::NAN, f32::INFINITY]
    {
        assert!(
            store
                .learn_residual(ids[0], &SedenionSimd::unit(1), bad_rate)
                .is_err()
        );
    }
    assert!(
        store
            .learn_residual(ids[0], &SedenionSimd::ZERO, 0.1)
            .is_err()
    );
    let mut nan = [0.0f32; 16];
    nan[3] = f32::NAN;
    assert!(
        store
            .learn_residual(ids[0], &SedenionSimd::from_array(nan), 0.1)
            .is_err()
    );
    let after = store.get(ids[0]).unwrap();
    assert_eq!(
        &before, after,
        "failed learning must leave the record unchanged"
    );
}

#[test]
fn drift_is_capped_by_the_residual_bound() {
    // Hammer one concept toward an orthogonal target: the residual norm must
    // never exceed the bound, so the effective vector stays anchored.
    let (mut store, ids) = seeded_store(1, 43);
    let target = SedenionSimd::unit(9);
    let mut clamped_seen = false;
    for _ in 0..200
    {
        let out = store.learn_residual(ids[0], &target, 1.0).unwrap();
        clamped_seen |= out.clamped;
        let r = store.get(ids[0]).unwrap().residual();
        let norm = scirust_hypermemory::norm_sqr_ordered(&r).sqrt();
        assert!(
            norm <= scirust_hypermemory::DEFAULT_RESIDUAL_BOUND + 1e-5,
            "residual norm {norm} exceeded the bound"
        );
    }
    assert!(clamped_seen, "aggressive learning must hit the clamp");
}

// ------------------------------------------------------------------ //
//  F4 — per-record isolation, exactly                                  //
// ------------------------------------------------------------------ //

#[test]
fn f4_other_records_are_bit_identical_after_learning() {
    let (mut store, ids) = seeded_store(16, 44);
    let snapshot: Vec<[f32; 16]> = ids
        .iter()
        .map(|&id| store.get(id).unwrap().effective().to_array())
        .collect();

    // Learn hard on one concept.
    for _ in 0..25
    {
        let _ = store
            .learn_residual(ids[5], &SedenionSimd::unit(3), 0.5)
            .unwrap();
    }

    for (i, &id) in ids.iter().enumerate()
    {
        let now = store.get(id).unwrap().effective().to_array();
        if i == 5
        {
            assert_ne!(now, snapshot[i], "the learned concept must have moved");
        }
        else
        {
            assert_eq!(
                now, snapshot[i],
                "F4 violated: unrelated concept {i} drifted"
            );
        }
    }
}

/// The ranking of every concept **except** `exclude`, under `query`.
fn ranking_without(
    index: &S16ExactIndex,
    query: &SedenionSimd,
    k: usize,
    exclude: ConceptId,
) -> Vec<(ConceptId, f32)> {
    let hits: Vec<SearchHit> = index.search(query, k).unwrap();
    hits.into_iter()
        .filter(|h| h.id != exclude)
        .map(|h| (h.id, h.score))
        .collect()
}

#[test]
fn f4_rankings_of_other_concepts_are_unchanged_for_any_query() {
    let (mut store, ids) = seeded_store(24, 45);
    let mut index = S16ExactIndex::new(SimilarityMetric::Cosine);
    index.rebuild_from(store.iter());

    let mut rng = Lcg::new(999);
    let queries: Vec<SedenionSimd> = (0..8).map(|_| rng.nonzero_sedenion()).collect();
    let learned = ids[11];

    let before: Vec<Vec<(ConceptId, f32)>> = queries
        .iter()
        .map(|q| ranking_without(&index, q, 24, learned))
        .collect();

    // Learn, refresh the index entry for the learned concept only.
    for _ in 0..10
    {
        let _ = store
            .learn_residual(learned, &SedenionSimd::unit(14), 0.4)
            .unwrap();
    }
    index.update_concept(store.get(learned).unwrap());

    let after: Vec<Vec<(ConceptId, f32)>> = queries
        .iter()
        .map(|q| ranking_without(&index, q, 24, learned))
        .collect();

    // Exact equality: same order AND bit-identical scores for every other
    // concept, under every query. Only the learned concept may move.
    assert_eq!(
        before, after,
        "F4 violated: other concepts' rankings changed"
    );
}

#[test]
fn f4_learning_trajectory_is_deterministic() {
    fn run() -> (Vec<[f32; 16]>, Vec<f32>) {
        let (mut store, ids) = seeded_store(6, 46);
        let mut sims = Vec::new();
        for step in 0..30
        {
            let target = if step % 2 == 0
            {
                SedenionSimd::unit(2)
            }
            else
            {
                SedenionSimd::unit(13)
            };
            let out = store.learn_residual(ids[step % 6], &target, 0.3).unwrap();
            sims.push(out.similarity_after);
        }
        let vecs = ids
            .iter()
            .map(|&id| store.get(id).unwrap().effective().to_array())
            .collect();
        (vecs, sims)
    }
    assert_eq!(run(), run(), "learning must be bit-reproducible");
}

// ------------------------------------------------------------------ //
//  Bounded-memory integration                                          //
// ------------------------------------------------------------------ //

#[test]
fn bounded_memory_learn_keeps_index_in_lock_step() {
    let mut mem = S16BoundedMemory::new(4, SimilarityMetric::Cosine, LinearDecay::new(100));
    let a = mem
        .insert(ConceptSpec::new(vec![0], SedenionSimd::unit(0), 1.0, 0))
        .unwrap()
        .inserted;
    let b = mem
        .insert(ConceptSpec::new(vec![1], SedenionSimd::unit(1), 1.0, 0))
        .unwrap()
        .inserted;

    // Before learning, a query on unit(7) matches nothing well.
    // Teach concept `a` toward unit(7); afterwards it must win that query,
    // and the learn step must have bumped its recency.
    let out = mem.learn(a, &SedenionSimd::unit(7), 0.8, 50).unwrap();
    assert!(out.similarity_after > out.similarity_before);
    let hits = mem.search(&SedenionSimd::unit(7), 1, 60).unwrap();
    assert_eq!(hits[0].id, a, "the index must reflect the learned vector");
    assert_eq!(mem.get(a).unwrap().metadata().last_access(), 60);
    // Concept `b` untouched.
    assert_eq!(
        mem.get(b).unwrap().effective().to_array(),
        SedenionSimd::unit(1).to_array()
    );
}

//! Phase 1 falsification & determinism harness (end-to-end, public API only).
//!
//! These integration tests exercise the crate exactly as a downstream consumer
//! would, and encode the Phase 1 scientific claims as executable checks:
//!
//! * the sedenion index is bit-identical to the real-vector baseline for
//!   retrieval (falsification criterion F1) — at scale, over seeded corpora;
//! * the exact index agrees with an *independent* audited dense index
//!   (`scirust_retrieval::DenseIndex`);
//! * rankings, expression evaluation, and insert/remove/reinsert sequences are
//!   reproducible;
//! * the canonical zero-divisor identity flows through the whole public API
//!   without a panic and with the structure still recoverable.
//!
//! Randomness is a fixed-seed in-crate LCG (no RNG dependency), following the
//! style used by `scirust-simd`'s own tests.

use scirust_hypermemory::{
    ConceptId, ConceptSpec, ExprLimits, ProductDiagnostics, Real16Index, RelationId, S16ExactIndex,
    S16Expr, S16Relation, S16Store, SearchHit, SimilarityMetric,
};
use scirust_retrieval::DenseIndex;
use scirust_simd::hypercomplex::SedenionSimd;

// ------------------------------------------------------------------ //
//  Deterministic corpus generator (fixed-seed LCG, no RNG dependency)  //
// ------------------------------------------------------------------ //

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
        let bits = (self.next_u64() >> 40) as u32; // top 24 bits
        (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0 // [-1, 1)
    }

    /// A sedenion with at least one non-zero lane (so its effective vector is
    /// always well-defined).
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

/// Map a `ConceptId` to a `u64` that is *monotonic* in the id's `(slot,
/// generation)` order, so `DenseIndex`'s ascending-`u64` tie-break matches this
/// crate's ascending-`ConceptId` tie-break exactly.
fn dense_key(id: ConceptId) -> u64 {
    ((id.slot() as u64) << 32) | (id.generation() as u64)
}

// ------------------------------------------------------------------ //
//  Hand-derived oracle                                                 //
// ------------------------------------------------------------------ //

#[test]
fn hand_derived_top1_and_topk() {
    // Three concepts on orthogonal basis directions; cosine to e2 is exactly
    // 1 for concept 2 and 0 for the others — a hand-checkable ranking.
    let mut store = S16Store::new();
    let mut index = S16ExactIndex::new(SimilarityMetric::Cosine);
    let mut ids = Vec::new();
    for i in 0..3usize
    {
        let id = store
            .insert(ConceptSpec::new(
                vec![i as u8],
                SedenionSimd::unit(i),
                1.0,
                0,
            ))
            .unwrap();
        index.insert_concept(store.get(id).unwrap());
        ids.push(id);
    }
    let hits = index.search(&SedenionSimd::unit(2), 3).unwrap();
    assert_eq!(hits[0].id, ids[2]);
    assert!((hits[0].score - 1.0).abs() < 1e-6);
    // The two orthogonal concepts tie at score 0 → ordered by ascending id.
    assert_eq!(hits[1].id, ids[0]);
    assert_eq!(hits[2].id, ids[1]);
    assert_eq!(hits[1].score, 0.0);
    assert_eq!(hits[2].score, 0.0);
}

// ------------------------------------------------------------------ //
//  F1 — sedenion retrieval == real-vector retrieval (at scale)         //
// ------------------------------------------------------------------ //

#[test]
fn sedenion_index_matches_real16_over_seeded_corpora() {
    for metric in [SimilarityMetric::Cosine, SimilarityMetric::SquaredEuclidean]
    {
        for seed in 0..8u64
        {
            let mut rng = Lcg::new(seed);
            let mut store = S16Store::new();
            let mut sed = S16ExactIndex::new(metric);
            let mut real = Real16Index::new(metric);
            for i in 0..64u32
            {
                let anchor = rng.nonzero_sedenion();
                let id = store
                    .insert(ConceptSpec::new(vec![i as u8], anchor, 1.0, 0))
                    .unwrap();
                sed.insert_concept(store.get(id).unwrap());
                real.insert_concept(store.get(id).unwrap());
            }
            for _ in 0..8
            {
                let q = rng.nonzero_sedenion();
                // Bit-identical ids AND scores: the algebra adds nothing to
                // retrieval over the same 16 components.
                assert_eq!(
                    sed.search(&q, 10).unwrap(),
                    real.search(&q, 10).unwrap(),
                    "F1 violated: sedenion and real-16 rankings diverged \
                     (metric {metric:?}, seed {seed})"
                );
            }
        }
    }
}

// ------------------------------------------------------------------ //
//  Independent cross-check against scirust_retrieval::DenseIndex       //
// ------------------------------------------------------------------ //

#[test]
fn sedenion_index_agrees_with_dense_index() {
    // Controlled corpus of distinct scaled basis directions: cosine scores are
    // exactly 1 (match) or 0 (orthogonal), so the two implementations rank
    // bit-identically with no float-noise tie ambiguity.
    let mut store = S16Store::new();
    let mut sed = S16ExactIndex::new(SimilarityMetric::Cosine);
    let mut dense = DenseIndex::new(16);
    let mut ids = Vec::new();
    for i in 0..16u32
    {
        let anchor = SedenionSimd::unit(i as usize).scale(1.0 + (i % 4) as f32);
        let id = store
            .insert(ConceptSpec::new(vec![i as u8], anchor, 1.0, 0))
            .unwrap();
        sed.insert_concept(store.get(id).unwrap());
        // Feed DenseIndex the same effective (normalized) components.
        dense
            .add(
                dense_key(id),
                &store.get(id).unwrap().effective().to_array(),
            )
            .unwrap();
        ids.push(id);
    }

    for q_lane in [0usize, 5, 11, 15]
    {
        let query = SedenionSimd::unit(q_lane);
        let sed_hits = sed.search(&query, 16).unwrap();
        let dense_hits = dense.search(&query.to_array(), 16);

        // Same length, same id order (mapped), same scores.
        assert_eq!(sed_hits.len(), dense_hits.len());
        for (s, d) in sed_hits.iter().zip(dense_hits.iter())
        {
            assert_eq!(
                dense_key(s.id),
                d.id,
                "ranking order diverged from DenseIndex"
            );
            assert!(
                (s.score - d.score).abs() < 1e-6,
                "score diverged: {} vs {}",
                s.score,
                d.score
            );
        }
    }
}

// ------------------------------------------------------------------ //
//  Determinism                                                         //
// ------------------------------------------------------------------ //

fn scripted_corpus(seed: u64) -> (S16Store, S16ExactIndex, Vec<ConceptId>) {
    let mut rng = Lcg::new(seed);
    let mut store = S16Store::new();
    let mut index = S16ExactIndex::new(SimilarityMetric::Cosine);
    let mut ids = Vec::new();
    for i in 0..40u32
    {
        let anchor = rng.nonzero_sedenion();
        let id = store
            .insert(ConceptSpec::new(
                vec![i as u8],
                anchor,
                1.0 + (i % 3) as f32,
                i as u64,
            ))
            .unwrap();
        index.insert_concept(store.get(id).unwrap());
        ids.push(id);
    }
    (store, index, ids)
}

#[test]
fn repeated_runs_produce_identical_ranking() {
    let (_s1, i1, _ids1) = scripted_corpus(1234);
    let (_s2, i2, _ids2) = scripted_corpus(1234);
    let mut rng = Lcg::new(999);
    for _ in 0..16
    {
        let q = rng.nonzero_sedenion();
        let a: Vec<SearchHit> = i1.search(&q, 12).unwrap();
        let b: Vec<SearchHit> = i2.search(&q, 12).unwrap();
        assert_eq!(a, b, "same inputs must produce identical rankings");
    }
}

#[test]
fn expression_evaluation_is_bitwise_repeatable() {
    let mut rng = Lcg::new(77);
    let mut store = S16Store::new();
    let mut ids = Vec::new();
    for i in 0..6u32
    {
        let id = store
            .insert(ConceptSpec::new(
                vec![i as u8],
                rng.nonzero_sedenion(),
                1.0,
                0,
            ))
            .unwrap();
        ids.push(id);
    }
    let limits = ExprLimits::default();
    // ((a·b)·(c·d))·e — a fixed, non-trivial parenthesization.
    let atom = |i: usize| S16Expr::atom(ids[i]);
    let expr = S16Expr::product(
        S16Expr::product(
            S16Expr::product(atom(0), atom(1)),
            S16Expr::product(atom(2), atom(3)),
        ),
        atom(4),
    );
    let first = expr.evaluate(&store, &limits).unwrap();
    for _ in 0..100
    {
        assert_eq!(
            expr.evaluate(&store, &limits).unwrap().to_array(),
            first.to_array(),
            "expression evaluation must be bit-identical across runs"
        );
    }
    // The digest is likewise stable.
    assert_eq!(expr.digest(), expr.digest());
}

#[test]
fn insert_remove_reinsert_sequence_is_reproducible() {
    // Run the exact same scripted mutation sequence twice; the observable state
    // (ids, iteration order, and a query's ranking) must be identical.
    fn run() -> (Vec<ConceptId>, Vec<SearchHit>) {
        let mut store = S16Store::new();
        let mut index = S16ExactIndex::new(SimilarityMetric::Cosine);
        let mut rng = Lcg::new(2024);
        let mut ids = Vec::new();
        for i in 0..10u32
        {
            let id = store
                .insert(ConceptSpec::new(
                    vec![i as u8],
                    rng.nonzero_sedenion(),
                    1.0,
                    0,
                ))
                .unwrap();
            index.insert_concept(store.get(id).unwrap());
            ids.push(id);
        }
        // Remove a few, then reinsert (reusing freed slots with bumped gens).
        for &victim in &[ids[2], ids[5], ids[7]]
        {
            store.remove(victim).unwrap();
            index.remove(victim);
        }
        let mut rng2 = Lcg::new(4048);
        for i in 100..103u32
        {
            let id = store
                .insert(ConceptSpec::new(
                    vec![i as u8],
                    rng2.nonzero_sedenion(),
                    1.0,
                    0,
                ))
                .unwrap();
            index.insert_concept(store.get(id).unwrap());
        }
        let live: Vec<ConceptId> = store.ids().collect();
        let hits = index.search(&SedenionSimd::unit(3), 8).unwrap();
        (live, hits)
    }

    let (live_a, hits_a) = run();
    let (live_b, hits_b) = run();
    assert_eq!(live_a, live_b, "live-id set/order must be reproducible");
    assert_eq!(
        hits_a, hits_b,
        "ranking after mutation must be reproducible"
    );
}

// ------------------------------------------------------------------ //
//  Zero-divisor, end to end through the public API                     //
// ------------------------------------------------------------------ //

#[test]
fn zero_divisor_flows_through_public_api_without_panic() {
    // Insert the two factors of the canonical identity as ordinary concepts,
    // build the relation, and confirm: exact zero product, both operands
    // non-zero, diagnostics flag it, no panic, structure recoverable.
    let x = SedenionSimd::unit(1) + SedenionSimd::unit(10);
    let y = SedenionSimd::unit(4) - SedenionSimd::unit(15);
    let mut store = S16Store::new();
    let cx = store
        .insert(ConceptSpec::new(b"x".to_vec(), x, 1.0, 0))
        .unwrap();
    let cy = store
        .insert(ConceptSpec::new(b"y".to_vec(), y, 1.0, 0))
        .unwrap();

    let limits = ExprLimits::default();
    let expr = S16Expr::product(S16Expr::atom(cx), S16Expr::atom(cy));
    let relation = S16Relation::build(RelationId::new(1), expr.clone(), &store, &limits).unwrap();

    // Product is exactly the zero sedenion.
    assert_eq!(relation.code().to_array(), [0.0f32; 16]);

    // Diagnostics: both operands norm² = 2, result norm² = 0, flagged.
    let (_code, diag) = expr
        .evaluate_with_diagnostics(
            &store,
            &limits,
            scirust_hypermemory::DEFAULT_NEAR_ZERO_THRESHOLD,
        )
        .unwrap();
    let diag: ProductDiagnostics = diag.unwrap();
    assert_eq!(diag.lhs_norm_sqr(), 2.0);
    assert_eq!(diag.rhs_norm_sqr(), 2.0);
    assert_eq!(diag.result_norm_sqr(), 0.0);
    assert!(diag.near_zero_divisor());
    assert!(diag.finite());
    assert!(!diag.normalization_safe());

    // The original relation structure is still recoverable from the stored
    // expression — we never tried to invert the (zero) product.
    match relation.expr()
    {
        S16Expr::Product { left, right } =>
        {
            assert_eq!(**left, S16Expr::Atom(cx));
            assert_eq!(**right, S16Expr::Atom(cy));
        },
        _ => panic!("expected a product"),
    }
    // And the factors are still individually retrievable and non-zero.
    assert_eq!(store.get(cx).unwrap().anchor().to_array(), x.to_array());
    assert_eq!(store.get(cy).unwrap().anchor().to_array(), y.to_array());
}

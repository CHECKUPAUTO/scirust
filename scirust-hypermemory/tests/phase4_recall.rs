//! Phase 4 — recall of the deterministic IVF index, measured against the
//! Phase 1 exact oracle.
//!
//! The whole point of the oracle-first program: an approximate index is never
//! trusted on faith — its recall is *measured* against [`S16ExactIndex`], and
//! its limiting behaviour is pinned down exactly:
//!
//! * `nprobe ≥ nlist` (scan everything) → results **bit-identical** to the
//!   oracle, for every seeded query;
//! * recall@k is **monotone non-decreasing in `nprobe`** (probed candidate
//!   sets are nested by construction);
//! * builds and searches are bit-reproducible.

use scirust_hypermemory::{
    ConceptSpec, S16ExactIndex, S16IvfIndex, S16Store, SearchHit, SimilarityMetric,
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

fn corpus(n: u32, seed: u64) -> S16Store {
    let mut rng = Lcg::new(seed);
    let mut store = S16Store::new();
    for i in 0..n
    {
        store
            .insert(ConceptSpec::new(
                i.to_le_bytes().to_vec(),
                rng.nonzero_sedenion(),
                1.0,
                0,
            ))
            .unwrap();
    }
    store
}

/// Fraction of the oracle's top-`k` ids that the approximate result recovered.
fn recall(oracle: &[SearchHit], approx: &[SearchHit]) -> f64 {
    if oracle.is_empty()
    {
        return 1.0;
    }
    let found = oracle
        .iter()
        .filter(|o| approx.iter().any(|a| a.id == o.id))
        .count();
    found as f64 / oracle.len() as f64
}

#[test]
fn full_probe_is_bit_identical_to_the_oracle() {
    let store = corpus(500, 7);
    let mut exact = S16ExactIndex::new(SimilarityMetric::Cosine);
    exact.rebuild_from(store.iter());
    let ivf = S16IvfIndex::build(SimilarityMetric::Cosine, 16, 10, store.iter()).unwrap();

    let mut rng = Lcg::new(70);
    for _ in 0..16
    {
        let q = rng.nonzero_sedenion();
        let oracle = exact.search(&q, 20).unwrap();
        let full = ivf.search(&q, 20, ivf.nlist()).unwrap();
        assert_eq!(
            oracle, full,
            "nprobe = nlist must reproduce the oracle bit-for-bit"
        );
    }
}

#[test]
fn recall_is_monotone_in_nprobe_and_reaches_one() {
    let store = corpus(2_000, 8);
    let mut exact = S16ExactIndex::new(SimilarityMetric::Cosine);
    exact.rebuild_from(store.iter());
    let nlist = 32;
    let ivf = S16IvfIndex::build(SimilarityMetric::Cosine, nlist, 10, store.iter()).unwrap();

    let mut rng = Lcg::new(80);
    let queries: Vec<SedenionSimd> = (0..24).map(|_| rng.nonzero_sedenion()).collect();
    let probes = [1usize, 2, 4, 8, 16, 32];

    let mut mean_recalls = Vec::new();
    for &nprobe in &probes
    {
        let mut acc = 0.0f64;
        for q in &queries
        {
            let oracle = exact.search(q, 10).unwrap();
            let approx = ivf.search(q, 10, nprobe).unwrap();
            acc += recall(&oracle, &approx);
        }
        mean_recalls.push(acc / queries.len() as f64);
    }

    // Monotone non-decreasing (nested candidate sets), and exactly 1 at full
    // probe. Print the measured profile so a CI log shows the tradeoff.
    println!("recall@10 by nprobe {probes:?}: {mean_recalls:?}");
    for w in mean_recalls.windows(2)
    {
        assert!(
            w[1] >= w[0] - 1e-12,
            "recall must be monotone in nprobe: {mean_recalls:?}"
        );
    }
    assert_eq!(
        *mean_recalls.last().unwrap(),
        1.0,
        "full probe must reach recall 1.0"
    );
    // A single probed list out of 32 must already beat guessing blindly, but
    // honestly need not be high; only sanity-check it is non-trivial.
    assert!(
        mean_recalls[0] > 0.05,
        "nprobe=1 recall suspiciously low: {mean_recalls:?}"
    );
}

#[test]
fn recall_harness_is_reproducible() {
    fn run() -> Vec<Vec<SearchHit>> {
        let store = corpus(300, 9);
        let ivf = S16IvfIndex::build(SimilarityMetric::Cosine, 8, 5, store.iter()).unwrap();
        let mut rng = Lcg::new(90);
        (0..8)
            .map(|_| ivf.search(&rng.nonzero_sedenion(), 5, 2).unwrap())
            .collect()
    }
    assert_eq!(run(), run(), "IVF build+search must be bit-reproducible");
}

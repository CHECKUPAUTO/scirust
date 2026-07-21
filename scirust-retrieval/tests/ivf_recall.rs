//! IVF measured against the exact oracle — the generalisation of
//! `scirust-hypermemory`'s Phase 4 recall harness to arbitrary dimension.
//!
//! Pinned properties: full probe is bit-identical to `DenseIndex`; recall@k is
//! monotone non-decreasing in `nprobe` and reaches 1.0; the whole experiment
//! is bit-reproducible. The recall profile is printed for the CI log.

use scirust_retrieval::{DenseIndex, IvfIndex};

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

    fn vector(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.next_f32()).collect()
    }
}

const DIM: usize = 32;

fn corpus(n: u64, seed: u64) -> Vec<(u64, Vec<f32>)> {
    let mut rng = Lcg::new(seed);
    (0..n).map(|id| (id, rng.vector(DIM))).collect()
}

fn recall_at_k(
    oracle: &DenseIndex,
    ivf: &IvfIndex,
    queries: &[Vec<f32>],
    k: usize,
    nprobe: usize,
) -> f64 {
    let mut hit = 0usize;
    let mut total = 0usize;
    for q in queries
    {
        let truth = oracle.search(q, k);
        let approx = ivf.search(q, k, nprobe);
        for t in &truth
        {
            total += 1;
            if approx.iter().any(|a| a.id == t.id)
            {
                hit += 1;
            }
        }
    }
    hit as f64 / total as f64
}

#[test]
fn full_probe_is_bit_identical_to_the_oracle() {
    let entries = corpus(1000, 71);
    let mut oracle = DenseIndex::new(DIM);
    for (id, v) in &entries
    {
        oracle.add(*id, v).unwrap();
    }
    let ivf = IvfIndex::build(DIM, 16, 8, &entries).unwrap();

    let mut rng = Lcg::new(710);
    for _ in 0..12
    {
        let q = rng.vector(DIM);
        assert_eq!(
            ivf.search(&q, 10, ivf.nlist()),
            oracle.search(&q, 10),
            "nprobe = nlist must reproduce the exact oracle bit-for-bit"
        );
    }
}

#[test]
fn recall_is_monotone_in_nprobe_and_reaches_one() {
    let entries = corpus(1000, 72);
    let mut oracle = DenseIndex::new(DIM);
    for (id, v) in &entries
    {
        oracle.add(*id, v).unwrap();
    }
    let ivf = IvfIndex::build(DIM, 16, 8, &entries).unwrap();

    let mut rng = Lcg::new(720);
    let queries: Vec<Vec<f32>> = (0..24).map(|_| rng.vector(DIM)).collect();

    let mut prev = 0.0f64;
    let mut profile = String::new();
    for nprobe in [1usize, 2, 4, 8, 16]
    {
        let r = recall_at_k(&oracle, &ivf, &queries, 10, nprobe);
        profile.push_str(&format!("nprobe {nprobe}: recall@10 {r:.3}  "));
        assert!(
            r >= prev,
            "recall must be monotone in nprobe: {prev:.3} -> {r:.3} at nprobe {nprobe}"
        );
        prev = r;
    }
    println!("dim {DIM}, 1000 docs, nlist 16: {profile}");
    assert!(
        (prev - 1.0).abs() < f64::EPSILON,
        "full probe must reach recall 1.0, got {prev:.3}"
    );
}

#[test]
fn build_and_search_are_bit_reproducible() {
    fn run() -> Vec<(u64, f32)> {
        let entries = corpus(300, 73);
        let ivf = IvfIndex::build(DIM, 8, 6, &entries).unwrap();
        let mut rng = Lcg::new(730);
        let mut out = Vec::new();
        for _ in 0..6
        {
            let q = rng.vector(DIM);
            out.extend(ivf.search(&q, 5, 3).into_iter().map(|s| (s.id, s.score)));
        }
        out
    }
    assert_eq!(
        run(),
        run(),
        "the whole experiment must be bit-reproducible"
    );
}

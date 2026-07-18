//! Cleanup/denoising harness — the VSA "cleanup memory" component, measured.
//!
//! A denoising system for this construction is a *cleanup memory*: a noisy
//! code is recognized by nearest-neighbour against the stored prototypes and
//! replaced by the **exact stored** effective vector, gated by an acceptance
//! threshold so garbage is rejected rather than silently snapped to an
//! arbitrary concept. This file measures it (deterministically):
//!
//! * recognition accuracy as a function of noise amplitude;
//! * rejection of uncorrelated inputs at a sane threshold;
//! * idempotence (denoising a denoised code is a fixed point);
//! * **the payoff**: structure retrieval with atom cleanup between stages vs
//!   without — cleanup must recover most of the accuracy that noise destroyed
//!   (in this harness's harsher unit-norm regime, plain accuracy collapses to
//!   ~0.29 at amplitude 0.5 and cleanup recovers ~0.94);
//! * recency bump through `S16BoundedMemory::denoise`;
//! * bit-reproducibility of every profile.

use scirust_hypermemory::{
    ConceptId, ConceptSpec, Encoding, NoForgetting, S16BoundedMemory, S16ExactIndex, S16Store,
    SimilarityMetric, TripleShape, cosine16,
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

fn perturb(base: &SedenionSimd, rng: &mut Lcg, amplitude: f32) -> SedenionSimd {
    let b = base.to_array();
    let mut out = [0.0f32; 16];
    for i in 0..16
    {
        out[i] = b[i] + amplitude * rng.next_f32();
    }
    SedenionSimd::from_array(out)
}

fn seeded_corpus(n: u32, seed: u64) -> (S16Store, S16ExactIndex, Vec<ConceptId>) {
    let mut rng = Lcg::new(seed);
    let mut store = S16Store::new();
    let mut ids = Vec::new();
    for i in 0..n
    {
        ids.push(
            store
                .insert(ConceptSpec::new(
                    i.to_le_bytes().to_vec(),
                    rng.nonzero_sedenion(),
                    1.0,
                    0,
                ))
                .unwrap(),
        );
    }
    let mut index = S16ExactIndex::new(SimilarityMetric::Cosine);
    index.rebuild_from(store.iter());
    (store, index, ids)
}

// ------------------------------------------------------------------ //
//  Recognition accuracy vs noise (deterministic profile)               //
// ------------------------------------------------------------------ //

#[test]
fn denoising_accuracy_degrades_gracefully_with_noise() {
    let (store, index, ids) = seeded_corpus(64, 11);
    let mut rng = Lcg::new(110);
    let amplitudes = [0.1f32, 0.25, 0.5, 1.0];
    let trials_per_amp = 200usize;

    let mut accuracies = Vec::new();
    for &amp in &amplitudes
    {
        let mut correct = 0usize;
        for t in 0..trials_per_amp
        {
            let target = ids[t % ids.len()];
            let clean = store.get(target).unwrap().effective();
            let noisy = perturb(&clean, &mut rng, amp);
            // Threshold 0.0 (cosine): accept anything in the same half-space —
            // this measures pure nearest-neighbour recognition.
            if let Some(d) = index.denoise(&noisy, 0.0).unwrap()
            {
                if d.id == target
                {
                    correct += 1;
                }
            }
        }
        accuracies.push(correct as f64 / trials_per_amp as f64);
    }
    println!("denoise accuracy by amplitude {amplitudes:?}: {accuracies:?}");

    // Graceful degradation: monotone non-increasing, near-perfect at low
    // noise, still useful at moderate noise.
    for w in accuracies.windows(2)
    {
        assert!(
            w[1] <= w[0] + 1e-12,
            "accuracy must not increase with noise: {accuracies:?}"
        );
    }
    assert!(
        accuracies[0] > 0.99,
        "low-noise recognition should be near-perfect: {accuracies:?}"
    );
    assert!(
        accuracies[1] > 0.95,
        "moderate-noise recognition should stay high: {accuracies:?}"
    );
}

#[test]
fn uncorrelated_input_is_rejected_at_a_sane_threshold() {
    let (_store, index, _ids) = seeded_corpus(32, 12);
    let mut rng = Lcg::new(120);
    // Random dense queries are uncorrelated with every stored prototype; with
    // a 0.8-cosine acceptance bar essentially all must be rejected.
    let mut rejected = 0usize;
    let trials = 200usize;
    for _ in 0..trials
    {
        let garbage = rng.nonzero_sedenion();
        if index.denoise(&garbage, 0.8).unwrap().is_none()
        {
            rejected += 1;
        }
    }
    println!("garbage rejected at threshold 0.8: {rejected}/{trials}");
    assert!(
        rejected as f64 / trials as f64 > 0.99,
        "a sane threshold must reject uncorrelated input ({rejected}/{trials})"
    );
}

#[test]
fn denoising_is_idempotent_and_snaps_to_the_exact_prototype() {
    let (store, index, ids) = seeded_corpus(16, 13);
    let mut rng = Lcg::new(130);
    let target = ids[7];
    let clean = store.get(target).unwrap().effective();
    let noisy = perturb(&clean, &mut rng, 0.3);

    let first = index.denoise(&noisy, 0.5).unwrap().expect("should accept");
    assert_eq!(first.id, target);
    // The denoised code is the exact stored prototype, bit for bit.
    assert_eq!(first.code.to_array(), clean.to_array());
    // Denoising the denoised code is a fixed point with a perfect score.
    let second = index
        .denoise(&first.code, 0.5)
        .unwrap()
        .expect("fixed point");
    assert_eq!(second.id, target);
    assert_eq!(second.code.to_array(), first.code.to_array());
    assert!((second.score - 1.0).abs() < 1e-6);
}

#[test]
fn invalid_inputs_are_typed_errors_and_empty_index_is_none() {
    let (_store, index, _ids) = seeded_corpus(4, 14);
    assert!(index.denoise(&SedenionSimd::ZERO, 0.5).is_err());
    assert!(index.denoise(&SedenionSimd::unit(0), f32::NAN).is_err());
    let empty = S16ExactIndex::new(SimilarityMetric::Cosine);
    assert_eq!(empty.denoise(&SedenionSimd::unit(0), 0.5).unwrap(), None);
}

// ------------------------------------------------------------------ //
//  The payoff: structure retrieval with cleanup between stages         //
// ------------------------------------------------------------------ //

/// The 6 orderings of three positions (matches the binding experiment).
const PERMS3: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

/// Structure retrieval as in `binding::structure_retrieval`, but with an
/// optional cleanup stage: each noisy atom is denoised against a store of the
/// three clean atoms before encoding. Returns (accuracy_without, accuracy_with).
fn structure_retrieval_with_and_without_cleanup(
    seed: u64,
    atom_sets: usize,
    trials_per_set: usize,
    noise: f32,
) -> (f64, f64) {
    let enc = Encoding::Sedenion;
    let structures: Vec<([usize; 3], TripleShape)> = PERMS3
        .iter()
        .flat_map(|&p| TripleShape::ALL.iter().map(move |&s| (p, s)))
        .collect();
    let n_struct = structures.len();

    let mut rng = Lcg::new(seed);
    let mut correct_plain = 0usize;
    let mut correct_clean = 0usize;
    let mut trials = 0usize;

    for _ in 0..atom_sets
    {
        // Three atoms, stored in a per-set cleanup memory.
        let atoms = [
            rng.nonzero_sedenion(),
            rng.nonzero_sedenion(),
            rng.nonzero_sedenion(),
        ];
        let mut store = S16Store::new();
        let mut cleanup = S16ExactIndex::new(SimilarityMetric::Cosine);
        let mut atom_ids = Vec::new();
        for (i, a) in atoms.iter().enumerate()
        {
            let id = store
                .insert(ConceptSpec::new(vec![i as u8], *a, 1.0, 0))
                .unwrap();
            cleanup.insert_concept(store.get(id).unwrap());
            atom_ids.push(id);
        }
        // The codebook uses the *effective* (stored, unit-norm) atoms so the
        // cleanup output regenerates codebook entries exactly.
        let canon: Vec<SedenionSimd> = atom_ids
            .iter()
            .map(|&id| store.get(id).unwrap().effective())
            .collect();
        let codebook: Vec<[f32; 16]> = structures
            .iter()
            .map(|&(p, s)| enc.encode(&canon[p[0]], &canon[p[1]], &canon[p[2]], s))
            .collect();

        for _ in 0..trials_per_set
        {
            let target = rng.next_u64() as usize % n_struct;
            let (p, s) = structures[target];
            let noisy = [
                perturb(&canon[0], &mut rng, noise),
                perturb(&canon[1], &mut rng, noise),
                perturb(&canon[2], &mut rng, noise),
            ];

            // Without cleanup: encode straight from the noisy atoms.
            let q_plain = enc.encode(&noisy[p[0]], &noisy[p[1]], &noisy[p[2]], s);
            // With cleanup: denoise each atom first (threshold 0.3; fall back
            // to the noisy atom if rejected — cleanup must never hard-fail the
            // pipeline).
            let cleaned: Vec<SedenionSimd> = noisy
                .iter()
                .map(|n| match cleanup.denoise(n, 0.3).unwrap()
                {
                    Some(d) => d.code,
                    None => *n,
                })
                .collect();
            let q_clean = enc.encode(&cleaned[p[0]], &cleaned[p[1]], &cleaned[p[2]], s);

            for (query, correct) in [(q_plain, &mut correct_plain), (q_clean, &mut correct_clean)]
            {
                let mut best = 0usize;
                let mut best_sim = f32::NEG_INFINITY;
                for (i, code) in codebook.iter().enumerate()
                {
                    let sim = cosine16(&query, code);
                    if sim > best_sim
                    {
                        best_sim = sim;
                        best = i;
                    }
                }
                if best == target
                {
                    *correct += 1;
                }
            }
            trials += 1;
        }
    }
    (
        correct_plain as f64 / trials as f64,
        correct_clean as f64 / trials as f64,
    )
}

#[test]
fn cleanup_recovers_structure_retrieval_accuracy_under_heavy_noise() {
    // At heavy noise the raw pipeline degrades (our prior binding measurement:
    // ~0.87 at noise 0.5). With atom cleanup between stages, correctly
    // recognized atoms regenerate the *exact* codebook entry, so accuracy must
    // recover most of the loss.
    // Note the noise here is relative to *unit-norm* atoms, so amplitude 0.5
    // is a much harsher regime than the earlier binding experiment (raw atoms
    // of norm ≈ 2.3): plain accuracy collapses toward ~0.29. Cleanup cannot
    // reach 1.0 in this regime — when an atom is *mis*-recognized as one of
    // the other stored prototypes, the pipeline regenerates an exactly-wrong
    // codebook entry — but it recovers most of the loss (measured ≈ 0.94).
    let (plain, cleaned) = structure_retrieval_with_and_without_cleanup(21, 60, 60, 0.5);
    println!("structure retrieval at noise 0.5: plain {plain:.4}, with cleanup {cleaned:.4}");
    assert!(
        cleaned >= plain,
        "cleanup must never hurt: plain {plain:.4} vs cleaned {cleaned:.4}"
    );
    assert!(
        cleaned - plain > 0.3,
        "cleanup must recover a large margin at heavy noise: \
         plain {plain:.4} vs cleaned {cleaned:.4}"
    );
    assert!(
        cleaned > 0.9,
        "with cleanup, heavy-noise structure retrieval should exceed 0.9: {cleaned:.4}"
    );

    // At light noise both are near-perfect and cleanup must not hurt.
    let (plain_lo, cleaned_lo) = structure_retrieval_with_and_without_cleanup(22, 30, 30, 0.1);
    println!("structure retrieval at noise 0.1: plain {plain_lo:.4}, with cleanup {cleaned_lo:.4}");
    assert!(cleaned_lo >= plain_lo - 1e-12);
}

// ------------------------------------------------------------------ //
//  Bounded-memory integration and determinism                          //
// ------------------------------------------------------------------ //

#[test]
fn bounded_memory_denoise_bumps_recency_of_the_recognized_concept() {
    let mut mem = S16BoundedMemory::new(4, SimilarityMetric::Cosine, NoForgetting);
    let a = mem
        .insert(ConceptSpec::new(vec![0], SedenionSimd::unit(2), 1.0, 0))
        .unwrap()
        .inserted;
    let clean = mem.get(a).unwrap().effective();
    let mut rng = Lcg::new(140);
    let noisy = perturb(&clean, &mut rng, 0.2);

    let d = mem.denoise(&noisy, 0.5, 77).unwrap().expect("recognized");
    assert_eq!(d.id, a);
    assert_eq!(mem.get(a).unwrap().metadata().last_access(), 77);
    assert_eq!(mem.get(a).unwrap().metadata().access_count(), 1);

    // A rejected input touches nothing.
    let garbage = SedenionSimd::unit(9);
    assert!(mem.denoise(&garbage, 0.99, 99).unwrap().is_none());
    assert_eq!(mem.get(a).unwrap().metadata().last_access(), 77);
}

#[test]
fn denoising_profiles_are_reproducible() {
    fn run() -> (f64, f64) {
        structure_retrieval_with_and_without_cleanup(31, 20, 40, 0.5)
    }
    assert_eq!(
        run(),
        run(),
        "the cleanup experiment must be bit-reproducible"
    );
}

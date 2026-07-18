//! Observation-fusion harness — SciRust's noise toolkit, measured against the
//! naive mean on the cleanup-recognition task (feature `signal-denoise`).
//!
//! Scenario: the same stored code is observed `N` times through a noisy
//! channel; the observations are fused per lane with `scirust-signal`'s
//! denoisers, then the fused estimate is snapped by the cleanup memory. Two
//! regimes make the comparison honest:
//!
//! * **broadband noise** — fusion (any strategy) must massively beat
//!   single-shot recognition, and Kalman-RTS must at least match the mean
//!   (for pure zero-mean broadband noise the mean is already near-optimal —
//!   we do not pretend otherwise);
//! * **impulsive corruption** (occasional gross spikes) — the regime that
//!   justifies the toolkit: Hampel impulse rejection must clearly beat the
//!   naive mean, which a single spike can drag off target.
//!
//! All profiles are deterministic and printed for the CI log.
#![cfg(feature = "signal-denoise")]

use scirust_hypermemory::{
    ConceptId, ConceptSpec, FusionStrategy, S16ExactIndex, S16Store, SimilarityMetric,
    fuse_observations,
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

/// One broadband-noise observation of `clean`.
fn observe_broadband(clean: &SedenionSimd, rng: &mut Lcg, amplitude: f32) -> SedenionSimd {
    let b = clean.to_array();
    let mut out = [0.0f32; 16];
    for i in 0..16
    {
        out[i] = b[i] + amplitude * rng.next_f32();
    }
    SedenionSimd::from_array(out)
}

/// One observation with broadband noise plus occasional gross spikes: each
/// lane is independently replaced by a ±`spike` outlier with probability
/// `p_num/p_den`.
fn observe_impulsive(
    clean: &SedenionSimd,
    rng: &mut Lcg,
    amplitude: f32,
    spike: f32,
    p_num: u64,
    p_den: u64,
) -> SedenionSimd {
    let b = clean.to_array();
    let mut out = [0.0f32; 16];
    for i in 0..16
    {
        let corrupted = rng.next_u64() % p_den < p_num;
        if corrupted
        {
            let sign = if rng.next_u64().is_multiple_of(2)
            {
                1.0
            }
            else
            {
                -1.0
            };
            out[i] = sign * spike;
        }
        else
        {
            out[i] = b[i] + amplitude * rng.next_f32();
        }
    }
    SedenionSimd::from_array(out)
}

/// Recognition accuracy over `trials`: N observations of a random stored
/// concept, fused with `strategy` (or used single-shot when `strategy` is
/// `None`), then cleaned up at threshold 0.0 (pure nearest-neighbour).
#[allow(clippy::too_many_arguments)] // test-harness plumbing, not public API
fn recognition_accuracy(
    store: &S16Store,
    index: &S16ExactIndex,
    ids: &[ConceptId],
    rng: &mut Lcg,
    n_obs: usize,
    trials: usize,
    strategy: Option<FusionStrategy>,
    observe: &mut dyn FnMut(&SedenionSimd, &mut Lcg) -> SedenionSimd,
) -> f64 {
    let mut correct = 0usize;
    for t in 0..trials
    {
        let target = ids[t % ids.len()];
        let clean = store.get(target).unwrap().effective();
        let observations: Vec<SedenionSimd> = (0..n_obs).map(|_| observe(&clean, rng)).collect();
        let recognized = match strategy
        {
            None => index.denoise(&observations[0], 0.0).unwrap(),
            Some(s) => index.denoise_observations(&observations, s, 0.0).unwrap(),
        };
        if let Some(d) = recognized
        {
            if d.id == target
            {
                correct += 1;
            }
        }
    }
    correct as f64 / trials as f64
}

#[test]
fn fusion_rescues_recognition_under_heavy_broadband_noise() {
    let (store, index, ids) = seeded_corpus(64, 51);
    let mut rng = Lcg::new(510);
    let amplitude = 1.0; // the regime where single-shot recognition ≈ 0.23
    let n_obs = 16;
    let trials = 150;

    let kalman = FusionStrategy::Kalman {
        process_var: 1e-6,
        meas_var: 0.33, // ≈ variance of uniform(±1) amplitude-1 noise
    };

    let mut ob = |c: &SedenionSimd, r: &mut Lcg| observe_broadband(c, r, amplitude);
    let single = recognition_accuracy(&store, &index, &ids, &mut rng, 1, trials, None, &mut ob);
    let mean = recognition_accuracy(
        &store,
        &index,
        &ids,
        &mut rng,
        n_obs,
        trials,
        Some(FusionStrategy::Mean),
        &mut ob,
    );
    let kal = recognition_accuracy(
        &store,
        &index,
        &ids,
        &mut rng,
        n_obs,
        trials,
        Some(kalman),
        &mut ob,
    );
    println!(
        "broadband amp {amplitude}, N={n_obs}: single {single:.3}, mean {mean:.3}, kalman {kal:.3}"
    );

    assert!(
        mean > single + 0.3,
        "fusion must massively beat single-shot: single {single:.3}, mean {mean:.3}"
    );
    assert!(
        kal > single + 0.3,
        "Kalman fusion must massively beat single-shot: single {single:.3}, kalman {kal:.3}"
    );
    // Honest: for pure zero-mean broadband noise the mean is near-optimal;
    // Kalman must not be materially worse.
    assert!(
        kal >= mean - 0.05,
        "Kalman must not lose materially to the mean here: mean {mean:.3}, kalman {kal:.3}"
    );
    assert!(kal > 0.9, "fused recognition should be high, got {kal:.3}");
}

#[test]
fn hampel_beats_the_naive_mean_under_impulsive_corruption() {
    let (store, index, ids) = seeded_corpus(64, 52);
    let mut rng = Lcg::new(520);
    let n_obs = 15; // odd, comfortable for the Hampel window
    let trials = 150;

    let mean = FusionStrategy::Mean;
    let hampel = FusionStrategy::HampelKalman {
        half_window: 3,
        n_sigma: 3.0,
        process_var: 1e-6,
        meas_var: 0.05,
    };

    // 15% of lane-observations replaced by ±4.0 spikes over mild broadband
    // noise: a single surviving spike drags a plain mean by ≈ 4/15 ≈ 0.27 per
    // lane — comparable to the signal itself.
    let mut ob = |c: &SedenionSimd, r: &mut Lcg| observe_impulsive(c, r, 0.3, 4.0, 15, 100);
    let acc_mean = recognition_accuracy(
        &store,
        &index,
        &ids,
        &mut rng,
        n_obs,
        trials,
        Some(mean),
        &mut ob,
    );
    let acc_hampel = recognition_accuracy(
        &store,
        &index,
        &ids,
        &mut rng,
        n_obs,
        trials,
        Some(hampel),
        &mut ob,
    );
    println!(
        "impulsive (15% ±4 spikes), N={n_obs}: mean {acc_mean:.3}, hampel+kalman {acc_hampel:.3}"
    );

    assert!(
        acc_hampel > acc_mean + 0.5,
        "the toolkit's impulse rejection must crush the naive mean here: \
         mean {acc_mean:.3}, hampel {acc_hampel:.3}"
    );
    // Honest ceiling: at 15% corruption, occasional spike *clusters* exceed
    // the local Hampel window's breakdown point, so a residual error rate
    // remains (measured ≈ 0.81 vs the mean's ≈ 0.05 — a ~17× improvement).
    assert!(
        acc_hampel > 0.75,
        "impulse-cleaned recognition should be high, got {acc_hampel:.3}"
    );
}

#[test]
fn fusion_is_deterministic() {
    fn run() -> [f32; 16] {
        let mut rng = Lcg::new(53);
        let clean = rng.nonzero_sedenion();
        let obs: Vec<SedenionSimd> = (0..12)
            .map(|_| observe_impulsive(&clean, &mut rng, 0.2, 3.0, 10, 100))
            .collect();
        fuse_observations(
            &obs,
            FusionStrategy::HampelKalman {
                half_window: 2,
                n_sigma: 3.0,
                process_var: 1e-6,
                meas_var: 0.05,
            },
        )
        .unwrap()
        .to_array()
    }
    assert_eq!(run(), run(), "fusion must be bit-reproducible");
}

//! Observation fusion measured against the naive mean on the cleanup
//! recognition task (feature `fusion`) — the stable port of the
//! `scirust-hypermemory` observation-fusion harness, at dimension 32.
//!
//! Two regimes keep the comparison honest: under pure broadband noise the
//! mean is already near-optimal (Kalman must match it, not "beat" it); under
//! impulsive corruption the toolkit's Hampel stage must crush the mean, which
//! a single surviving spike can drag off target. Profiles are deterministic
//! and printed for the CI log.
#![cfg(feature = "fusion")]

use scirust_retrieval::{CleanupMemory, FusionStrategy, fuse_observations};

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

fn codebook(n: u64, seed: u64) -> (CleanupMemory, Vec<Vec<f32>>) {
    let mut rng = Lcg::new(seed);
    let mut mem = CleanupMemory::new(DIM);
    let mut protos = Vec::new();
    for id in 0..n
    {
        let v = rng.vector(DIM);
        mem.insert(id, &v).unwrap();
        protos.push(v);
    }
    (mem, protos)
}

fn observe_broadband(clean: &[f32], rng: &mut Lcg, amplitude: f32) -> Vec<f32> {
    clean
        .iter()
        .map(|&x| x + amplitude * rng.next_f32())
        .collect()
}

/// Broadband noise plus occasional gross spikes: each lane is independently
/// replaced by a ±`spike` outlier with probability `p_num/p_den`.
fn observe_impulsive(
    clean: &[f32],
    rng: &mut Lcg,
    amplitude: f32,
    spike: f32,
    p_num: u64,
    p_den: u64,
) -> Vec<f32> {
    clean
        .iter()
        .map(|&x| {
            if rng.next_u64() % p_den < p_num
            {
                let sign = if rng.next_u64().is_multiple_of(2)
                {
                    1.0
                }
                else
                {
                    -1.0
                };
                sign * spike
            }
            else
            {
                x + amplitude * rng.next_f32()
            }
        })
        .collect()
}

/// Recognition accuracy over `trials`: `n_obs` observations of a stored
/// prototype, fused with `strategy` (single-shot when `None`), then cleaned
/// up at threshold 0.0 (pure nearest-neighbour).
fn recognition_accuracy(
    mem: &CleanupMemory,
    protos: &[Vec<f32>],
    rng: &mut Lcg,
    n_obs: usize,
    trials: usize,
    strategy: Option<FusionStrategy>,
    observe: &mut dyn FnMut(&[f32], &mut Lcg) -> Vec<f32>,
) -> f64 {
    let mut correct = 0usize;
    for t in 0..trials
    {
        let target = t % protos.len();
        let observations: Vec<Vec<f32>> =
            (0..n_obs).map(|_| observe(&protos[target], rng)).collect();
        let recognized = match strategy
        {
            None => mem.clean(&observations[0], 0.0).unwrap(),
            Some(s) => mem.clean_observations(&observations, s, 0.0).unwrap(),
        };
        if let Some(hit) = recognized
        {
            if hit.id == target as u64
            {
                correct += 1;
            }
        }
    }
    correct as f64 / trials as f64
}

#[test]
fn fusion_rescues_recognition_under_heavy_broadband_noise() {
    let (mem, protos) = codebook(64, 91);
    let mut rng = Lcg::new(910);
    // At dim 32 recognition is easier than the research program's dim 16 (the
    // codebook has more room), so the noise must be proportionally heavier to
    // expose the fusion gain: amplitude 3 puts single-shot near chance.
    let amplitude = 3.0;
    let n_obs = 16;
    let trials = 150;

    let kalman = FusionStrategy::Kalman {
        process_var: 1e-6,
        meas_var: 3.0, // ≈ variance of uniform(±3) amplitude-3 noise
    };

    let mut ob = |c: &[f32], r: &mut Lcg| observe_broadband(c, r, amplitude);
    let single = recognition_accuracy(&mem, &protos, &mut rng, 1, trials, None, &mut ob);
    let mean = recognition_accuracy(
        &mem,
        &protos,
        &mut rng,
        n_obs,
        trials,
        Some(FusionStrategy::Mean),
        &mut ob,
    );
    let kal = recognition_accuracy(
        &mem,
        &protos,
        &mut rng,
        n_obs,
        trials,
        Some(kalman),
        &mut ob,
    );
    println!(
        "broadband amp {amplitude}, N={n_obs}, dim {DIM}: single {single:.3}, mean {mean:.3}, kalman {kal:.3}"
    );

    assert!(
        mean > single + 0.2,
        "fusion must clearly beat single-shot: single {single:.3}, mean {mean:.3}"
    );
    assert!(
        kal > single + 0.2,
        "Kalman fusion must clearly beat single-shot: single {single:.3}, kalman {kal:.3}"
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
    let (mem, protos) = codebook(64, 92);
    let mut rng = Lcg::new(920);
    let n_obs = 15; // odd, comfortable for the Hampel window
    let trials = 150;

    let hampel = FusionStrategy::HampelKalman {
        half_window: 3,
        n_sigma: 3.0,
        process_var: 1e-6,
        meas_var: 0.05,
    };

    // 25% of lane-observations replaced by ±6 spikes over mild broadband
    // noise (heavier than the dim-16 research harness — dim 32 dilutes the
    // damage a single spike does): surviving spikes drag a plain mean off
    // target while the Hampel stage rejects them.
    let mut ob = |c: &[f32], r: &mut Lcg| observe_impulsive(c, r, 0.3, 6.0, 25, 100);
    let acc_mean = recognition_accuracy(
        &mem,
        &protos,
        &mut rng,
        n_obs,
        trials,
        Some(FusionStrategy::Mean),
        &mut ob,
    );
    let acc_hampel = recognition_accuracy(
        &mem,
        &protos,
        &mut rng,
        n_obs,
        trials,
        Some(hampel),
        &mut ob,
    );
    println!(
        "impulsive (25% ±6 spikes), N={n_obs}, dim {DIM}: mean {acc_mean:.3}, hampel+kalman {acc_hampel:.3}"
    );

    assert!(
        acc_hampel > acc_mean + 0.4,
        "the toolkit's impulse rejection must crush the naive mean here: \
         mean {acc_mean:.3}, hampel {acc_hampel:.3}"
    );
    // Honest ceiling: spike *clusters* exceeding the local Hampel window's
    // breakdown point leave a residual error rate.
    assert!(
        acc_hampel > 0.75,
        "impulse-cleaned recognition should be high, got {acc_hampel:.3}"
    );
}

#[test]
fn fusion_is_deterministic() {
    fn run() -> Vec<f32> {
        let mut rng = Lcg::new(93);
        let clean = rng.vector(DIM);
        let obs: Vec<Vec<f32>> = (0..12)
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
    }
    assert_eq!(run(), run(), "fusion must be bit-reproducible");
}

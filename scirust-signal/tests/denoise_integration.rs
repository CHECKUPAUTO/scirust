//! Integration-level regression gate for `scirust_signal::denoise`.
//!
//! The `denoise_benchmark` *example* prints the full quality matrix and exits
//! non-zero on a qualitative regression, but `cargo test --workspace` never runs an
//! example's `main`. These tests re-assert the load-bearing qualitative properties
//! through the public API so CI actually enforces them: the right family wins each
//! noise type, the automatic pipelines behave, and the correctness fixes for the
//! periodic-interference router, the mixed-noise cascade and the streaming rank
//! filters stay fixed.
//!
//! Deterministic throughout: noise comes from a fixed-seed LCG (the module's
//! `testutil` helpers are `#[cfg(test)]`-only and invisible here), so every run is
//! byte-identical.

use core::f64::consts::PI;
use scirust_signal::denoise::streaming::StreamingMedian;
use scirust_signal::denoise::{
    NoiseType, SpectralLine, ThresholdMode, Wavelet, denoise_auto, denoise_best, denoise_cascade,
    detect_lines, hampel_filter, harmonic_stack, moving_average, notch_iir,
    wavelet_denoise_leveldep, wiener_white,
};

/// Deterministic 64-bit LCG (mirrors the module's private `testutil::Lcg`).
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let u1 = self.uniform().max(1.0e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

fn snr_db(clean: &[f64], est: &[f64]) -> f64 {
    let sig: f64 = clean.iter().map(|&x| x * x).sum();
    let err: f64 = clean
        .iter()
        .zip(est.iter())
        .map(|(&c, &e)| (c - e) * (c - e))
        .sum();
    10.0 * (sig / err.max(1.0e-30)).log10()
}

const FS: f64 = 1000.0;
const N: usize = 2048;

fn sine(freq: f64) -> Vec<f64> {
    (0..N)
        .map(|i| (2.0 * PI * freq * i as f64 / FS).sin())
        .collect()
}

#[test]
fn rank_family_wins_impulsive_noise() {
    let clean = sine(5.0);
    let mut rng = Lcg::new(11);
    let mut obs: Vec<f64> = clean.iter().map(|&c| c + 0.05 * rng.gauss()).collect();
    for i in (0..N).step_by(37)
    {
        obs[i] += 8.0;
    }
    let hampel = snr_db(&clean, &hampel_filter(&obs, 3, 3.0));
    let linear = snr_db(&clean, &moving_average(&obs, 5));
    assert!(
        hampel > linear + 5.0,
        "hampel {hampel:.1} dB should crush the linear smoother {linear:.1} dB on spikes"
    );
    // denoise_auto must route impulsive noise to the rank family, not a smoother.
    let auto = denoise_auto(&obs, FS);
    assert_eq!(auto.profile.dominant, NoiseType::Impulsive);
    assert!(snr_db(&clean, &auto.output) > snr_db(&clean, &obs));
}

#[test]
fn auto_pipeline_wins_tonal_interference_without_self_notching() {
    // 5 Hz signal + strong 50 Hz interferer + light floor.
    let clean = sine(5.0);
    let mut rng = Lcg::new(23);
    let obs: Vec<f64> = clean
        .iter()
        .enumerate()
        .map(|(i, &c)| c + 1.0 * (2.0 * PI * 50.0 * i as f64 / FS).sin() + 0.1 * rng.gauss())
        .collect();
    let auto = denoise_auto(&obs, FS);
    assert_eq!(auto.profile.dominant, NoiseType::Periodic);
    let gain = snr_db(&clean, &auto.output) - snr_db(&clean, &obs);
    assert!(
        gain > 8.0,
        "auto should recover the tonal record (gain {gain:.1} dB)"
    );
    // The 5 Hz signal must survive: the notch router protects the information tone.
    assert!(
        auto.method.contains("50") || auto.method.contains("notch") || auto.method.contains("hum")
    );
}

#[test]
fn periodic_verdict_never_notches_the_signals_own_tone() {
    // A near-pure tone with a tiny floor: even if some caller reaches the periodic
    // router, it must not notch the tone into oblivion. Drive the router directly
    // through denoise_best's periodic shortlist by making the tone look periodic.
    let clean = sine(8.0);
    let mut rng = Lcg::new(31);
    let obs: Vec<f64> = clean
        .iter()
        .enumerate()
        .map(|(i, &c)| c + 0.9 * (2.0 * PI * 60.0 * i as f64 / FS).sin() + 0.12 * rng.gauss())
        .collect();
    let best = denoise_best(&obs, FS);
    // The winner is a genuine noise remover (a notch of the 60 Hz interferer), not a
    // line enhancer that would keep the tone, and it improves SNR against the 8 Hz
    // signal (which was NOT notched).
    let gain = snr_db(&clean, &best.output) - snr_db(&clean, &obs);
    assert!(
        gain > 5.0,
        "denoise_best should remove the 60 Hz interferer (gain {gain:.1} dB)"
    );
    assert!(!best.method.contains("enhancer"), "method: {}", best.method);
}

#[test]
fn cascade_beats_every_single_family_on_mixed_noise() {
    let clean = sine(5.0);
    let mut rng = Lcg::new(401);
    let mut obs: Vec<f64> = clean
        .iter()
        .enumerate()
        .map(|(i, &c)| c + 1.0 * (2.0 * PI * 50.0 * i as f64 / FS).sin() + 0.2 * rng.gauss())
        .collect();
    for i in (0..N).step_by(37)
    {
        obs[i] += 8.0;
    }
    let casc = denoise_cascade(&obs, FS, 4);
    assert!(casc.stages.len() >= 2, "stages: {:?}", casc.stages);
    let s_casc = snr_db(&clean, &casc.output);
    let bin = FS / 2048.0;
    let singles = [
        hampel_filter(&obs, 3, 3.0),
        notch_iir(&obs, FS, 50.0, (50.0_f64 * 0.05).max(4.0 * bin)),
        wavelet_denoise_leveldep(&obs, 0, ThresholdMode::Soft, Wavelet::Db4),
    ];
    for out in &singles
    {
        assert!(
            s_casc >= snr_db(&clean, out) + 1.0,
            "cascade {s_casc:.1} dB must beat every single family by >= 1 dB"
        );
    }
}

#[test]
fn harmonic_stack_rejects_the_spurious_low_fundamental_family() {
    // Regression: a low signal remnant (7 Hz) and an unrelated interferer (137 Hz)
    // must not be paired at f0 = 3.5 (which let the router notch the signal).
    let mut rng = Lcg::new(11);
    let obs: Vec<f64> = (0..2072)
        .map(|i| {
            let t = i as f64 / 1024.0;
            (2.0 * PI * 7.0 * t).sin()
                + 0.8 * (2.0 * PI * 50.0 * t).sin()
                + 0.6 * (2.0 * PI * 137.0 * t).sin()
                + 0.03 * rng.gauss()
        })
        .collect();
    let lines = detect_lines(&obs, 1024.0, 5);
    assert!(harmonic_stack(&lines).is_none(), "lines: {lines:?}");
    // And the pure-tuple regressions.
    let mk = |f: f64| SpectralLine {
        freq_hz: f,
        fisher_g: 0.3,
        power_ratio: 0.2,
    };
    assert!(harmonic_stack(&[mk(119.1), mk(120.6), mk(122.1)]).is_none());
    assert!(harmonic_stack(&[mk(50.0), mk(100.0), mk(150.0)]).is_some());
}

#[test]
fn streaming_median_survives_a_nan() {
    // A single NaN must not permanently corrupt the sorted window (the old
    // partial_cmp search deleted a live sample and stranded the NaN, degrading the
    // window to even size => half-integer medians forever).
    let mut f = StreamingMedian::new(2);
    let mut sig: Vec<f64> = (0..64).map(|i| i as f64).collect();
    sig[5] = f64::NAN;
    let out: Vec<f64> = sig.iter().map(|&x| f.push(x)).collect();
    // Well after the NaN has left the window, the median of five consecutive
    // integers is an integer again.
    for (i, &v) in out.iter().enumerate().skip(20)
    {
        assert!(v.is_finite(), "non-finite output at {i}");
        assert!(
            (v.fract()).abs() < 1.0e-9,
            "half-integer median {v} at {i} => window size corrupted by the NaN"
        );
    }
}

#[test]
fn wiener_family_handles_stationary_broadband() {
    let clean = sine(4.0);
    let mut rng = Lcg::new(67);
    let obs: Vec<f64> = clean.iter().map(|&c| c + 0.4 * rng.gauss()).collect();
    assert!(snr_db(&clean, &wiener_white(&obs, 0.4)) > snr_db(&clean, &obs));
}

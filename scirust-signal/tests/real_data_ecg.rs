//! Real-data validation of the denoise toolkit: MIT-BIH ECG corrupted by **real
//! recorded noise** (MIT-BIH Noise Stress Test Database) added at controlled SNRs,
//! so the clean record is a ground-truth reference and the SNR improvement is
//! genuine — not measured against synthetic noise the toolkit could be tuned to.
//!
//! Data (Open Data Commons Attribution License v1.0, ODC-BY), committed as
//! `tests/data/ecg_mitbih.csv`:
//! * Clean ECG — MIT-BIH Arrhythmia DB, record 100, lead II (Moody & Mark 2001).
//! * Noise — MIT-BIH Noise Stress Test DB, records `ma` (muscle artifact, broadband)
//!   and `bw` (baseline wander, low-frequency), noise1 (Moody, Muldrow & Mark 1984).
//! * PhysioNet (Goldberger et al., Circulation 2000).
//!
//! The tests are deterministic: the noise is real recorded data from the fixture,
//! scaled to an exact target SNR — no RNG. They assert only the robust, honest
//! findings (see the module comment in `examples/denoise_real_ecg.rs` for the full
//! measured tables and the documented limitations).

use scirust_signal::denoise::{
    NoiseType, ThresholdMode, VstKind, classify, denoise_auto, detect_noise_model, remove_baseline,
    tikhonov_smooth, wavelet_denoise,
};

fn load_fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let raw = include_str!("data/ecg_mitbih.csv");
    let (mut ecg, mut ma, mut bw) = (Vec::new(), Vec::new(), Vec::new());
    for line in raw.lines()
    {
        if line.starts_with('#') || line.is_empty()
        {
            continue;
        }
        let mut it = line.split(',');
        ecg.push(it.next().unwrap().trim().parse::<f64>().unwrap());
        ma.push(it.next().unwrap().trim().parse::<f64>().unwrap());
        bw.push(it.next().unwrap().trim().parse::<f64>().unwrap());
    }
    (ecg, ma, bw)
}

fn sum_sq(x: &[f64]) -> f64 {
    x.iter().map(|&v| v * v).sum()
}

/// Standard SNR-improvement of `est` over the raw observation `noisy`, both against
/// the clean reference: `10·log10(‖noisy − clean‖² / ‖est − clean‖²)`. Positive means
/// the estimate is closer to the truth than the noisy input — the noise-power-reduction
/// ratio, robust to the signal's DC offset.
fn snr_improvement(clean: &[f64], noisy: &[f64], est: &[f64]) -> f64 {
    let e_noisy: f64 = clean
        .iter()
        .zip(noisy)
        .map(|(&c, &n)| (c - n) * (c - n))
        .sum();
    let e_est: f64 = clean
        .iter()
        .zip(est)
        .map(|(&c, &e)| (c - e) * (c - e))
        .sum();
    10.0 * (e_noisy / e_est.max(1.0e-30)).log10()
}

/// `clean + a·noise`, with `a` set so the raw observation sits at exactly `target_db`.
fn corrupt(clean: &[f64], noise: &[f64], target_db: f64) -> Vec<f64> {
    let a = (sum_sq(clean) / (sum_sq(noise) * 10f64.powf(target_db / 10.0))).sqrt();
    clean.iter().zip(noise).map(|(&c, &n)| c + a * n).collect()
}

#[test]
fn vst_detector_never_false_fires_on_real_ecg() {
    // The conservative signal-dependent-noise detector must return Identity on real
    // ECG under real additive noise (muscle artifact and baseline wander), at every
    // tested SNR: ECG noise is additive, not Poisson/multiplicative, so a non-identity
    // verdict would be a false positive that degrades the result. This is the honest
    // real-data check of the selector's "default to Identity on any doubt" design.
    let (ecg, ma, bw) = load_fixture();
    for (name, noise) in [("ma", &ma), ("bw", &bw)]
    {
        for &target in &[12.0f64, 6.0, 0.0]
        {
            let noisy = corrupt(&ecg, noise, target);
            assert_eq!(
                detect_noise_model(&noisy),
                VstKind::Identity,
                "VST detector false-fired on real ECG + {name} noise at {target} dB"
            );
        }
    }
}

#[test]
fn wavelet_shrinkage_removes_real_muscle_artifact() {
    // Broadband muscle artifact is what wavelet shrinkage is built for: the universal
    // soft threshold must improve the SNR against the clean reference at the harder
    // (noise-dominated) SNRs, on genuinely recorded noise.
    let (ecg, ma, _) = load_fixture();
    for &(target, min_gain) in &[(6.0f64, 0.5f64), (0.0, 0.7)]
    {
        let noisy = corrupt(&ecg, &ma, target);
        let den = wavelet_denoise(&noisy, 0, ThresholdMode::Soft);
        let gain = snr_improvement(&ecg, &noisy, &den);
        assert!(
            gain >= min_gain,
            "wavelet gained only {gain:.2} dB on real muscle artifact at {target} dB (want ≥ {min_gain})"
        );
    }
}

#[test]
fn denoise_auto_helps_at_low_snr_on_real_muscle_artifact() {
    // When the noise dominates (0 dB), the classifier's verdict flips from Impulsive
    // (driven by the QRS complexes at high SNR — see the example's documented finding)
    // to Colored, routing to level-dependent wavelet shrinkage with cycle spinning,
    // which recovers a clear margin on real muscle artifact.
    let (ecg, ma, _) = load_fixture();
    let noisy = corrupt(&ecg, &ma, 0.0);
    let auto = denoise_auto(&noisy, 360.0);
    let gain = snr_improvement(&ecg, &noisy, &auto.output);
    assert!(
        gain > 2.0,
        "denoise_auto gained only {gain:.2} dB at 0 dB (method: {})",
        auto.method
    );
    // Length is preserved and the output is finite on real data.
    assert_eq!(auto.output.len(), ecg.len());
    assert!(auto.output.iter().all(|v| v.is_finite()));
}

#[test]
fn qrs_complexes_are_not_mislabeled_as_impulsive_noise() {
    // A real ECG's QRS complexes are sharp, high-crest deflections, so a naive
    // impulsivity gate reads them as spikes and routes to a spike remover. They are a
    // *legitimate periodic feature*, not impulsive noise: with the energy-envelope
    // periodicity veto (detect::periodic_impulse_train) a QRS-dominated record is no
    // longer classified Impulsive. Checked on a real ECG lightly corrupted by real
    // baseline wander (high SNR, so the high-pass residual is QRS-dominated).
    let (ecg, _, bw) = load_fixture();
    let noisy = corrupt(&ecg, &bw, 12.0);
    let p = classify(&noisy, 360.0);
    assert_ne!(
        p.dominant,
        NoiseType::Impulsive,
        "real QRS complexes were mislabeled as impulsive noise (verdict {:?})",
        p.dominant
    );
}

#[test]
fn zero_phase_high_pass_removes_real_baseline_wander() {
    // Real baseline wander (nstdb `bw`) overlaps the ECG's own low-frequency content,
    // so a stiff Tikhonov detrend erodes the signal along with the drift — the honest
    // finding of the example. A physiological zero-phase high-pass (0.5 Hz, the
    // ANSI/AAMI EC11 / AHA cutoff that preserves the ST segment) is the signal-
    // preserving alternative. The target of baseline removal is the *drift-free*
    // morphology, so the ground truth is the same high-pass applied to the clean ECG.
    let (ecg, _, bw) = load_fixture();
    let noisy = corrupt(&ecg, &bw, 0.0); // heavy drift: raw SNR 0 dB
    let target = remove_baseline(&ecg, 360.0, 0.5); // drift-free morphology
    let hp = remove_baseline(&noisy, 360.0, 0.5);

    // How well each estimate recovers the drift-free morphology (edges trimmed).
    let err = |est: &[f64]| -> f64 {
        target[200..ecg.len() - 200]
            .iter()
            .zip(&est[200..ecg.len() - 200])
            .map(|(&t, &e)| (t - e) * (t - e))
            .sum::<f64>()
    };
    let e_raw = err(&noisy);
    let e_hp = err(&hp);
    let gain = 10.0 * (e_raw / e_hp.max(1.0e-30)).log10();
    assert!(
        gain > 10.0,
        "high-pass recovered the drift-free ECG only {gain:.1} dB better than raw"
    );

    // And it beats the stiff Tikhonov detrend at recovering that morphology — the
    // detrend's soft cutoff reaches into the ECG's own low-frequency content.
    let tik: Vec<f64> = noisy
        .iter()
        .zip(&tikhonov_smooth(&noisy, 1.0e4))
        .map(|(&s, &t)| s - t)
        .collect();
    assert!(
        err(&hp) < err(&tik),
        "high-pass ({:.3e}) should recover the morphology better than Tikhonov detrend ({:.3e})",
        err(&hp),
        err(&tik)
    );
}

#[test]
fn fixture_is_well_formed_real_ecg() {
    // Guard the committed fixture: the expected length, finiteness, and that the ECG
    // carries QRS structure (a peak well above the bulk) — i.e. it is a real ECG, not
    // a degenerate or truncated file.
    let (ecg, ma, bw) = load_fixture();
    assert_eq!(ecg.len(), 4096);
    assert_eq!(ma.len(), 4096);
    assert_eq!(bw.len(), 4096);
    assert!(ecg.iter().chain(&ma).chain(&bw).all(|v| v.is_finite()));
    let mean = ecg.iter().sum::<f64>() / ecg.len() as f64;
    let peak = ecg.iter().cloned().fold(f64::MIN, f64::max);
    let std = (ecg.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / ecg.len() as f64).sqrt();
    // An R-peak sits several σ above the mean — the signature of a QRS complex.
    assert!(
        peak - mean > 4.0 * std,
        "no QRS-like peak found (peak {peak:.3}, mean {mean:.3}, std {std:.3})"
    );
}

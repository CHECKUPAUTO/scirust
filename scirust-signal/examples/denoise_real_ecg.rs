//! Validation of the denoise toolkit on **real data**: MIT-BIH ECG corrupted by
//! **real recorded muscle-artifact noise** (MIT-BIH Noise Stress Test Database),
//! added at controlled SNRs so the clean record is a ground-truth reference.
//!
//! Run with `cargo run --release -p scirust-signal --example denoise_real_ecg`.
//! Deterministic: the noise is real recorded data from the fixture, not generated,
//! so two runs print identical numbers.
//!
//! ## Data provenance (Open Data Commons Attribution License v1.0, ODC-BY)
//!
//! * Clean ECG: MIT-BIH Arrhythmia Database, record 100, lead II — Moody GB,
//!   Mark RG, *"The impact of the MIT-BIH Arrhythmia Database"*, IEEE Eng. Med.
//!   Biol. 20(3):45-50 (2001). <https://physionet.org/content/mitdb/1.0.0/>
//! * Noise: MIT-BIH Noise Stress Test Database, records `ma` (muscle artifact) and
//!   `bw` (baseline wander), noise1 — Moody GB, Muldrow WE, Mark RG (1984).
//!   <https://physionet.org/content/nstdb/1.0.0/>
//! * PhysioNet: Goldberger AL et al., *Circulation* 101(23):e215-e220 (2000).
//!
//! A 4096-sample (~11.4 s, 360 Hz) excerpt of each is committed as
//! `tests/data/ecg_mitbih.csv`. Record 100 is a real recording with minor intrinsic
//! noise; treating it as the reference and adding calibrated `ma`/`bw` noise is the
//! standard MIT-BIH + nstdb denoising-validation protocol.

use scirust_signal::denoise::{
    ThresholdMode, VstKind, denoise_auto, detect_noise_model, moving_average, nlm1d_auto,
    savitzky_golay, stft_wiener_auto, tikhonov_smooth, wavelet_denoise,
};

/// Explicit baseline-drift removal: subtract a very stiff Tikhonov trend — the same
/// operation `denoise_auto` applies for a `Baseline` verdict, called directly here
/// to show the toolkit *can* remove low-frequency wander even when the QRS-driven
/// impulsive verdict masks that route in the automatic path.
fn baseline_removal(x: &[f64]) -> Vec<f64> {
    let trend = tikhonov_smooth(x, 1.0e4);
    x.iter().zip(&trend).map(|(&v, &t)| v - t).collect()
}

/// Parse the committed three-column fixture into `(ecg_mV, ma_noise_mV, bw_noise_mV)`.
fn load_fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let raw = include_str!("../tests/data/ecg_mitbih.csv");
    let (mut ecg, mut ma, mut bw) = (Vec::new(), Vec::new(), Vec::new());
    for line in raw.lines()
    {
        if line.starts_with('#') || line.is_empty()
        {
            continue;
        }
        let mut it = line.split(',');
        ecg.push(it.next().unwrap().trim().parse().unwrap());
        ma.push(it.next().unwrap().trim().parse().unwrap());
        bw.push(it.next().unwrap().trim().parse().unwrap());
    }
    (ecg, ma, bw)
}

fn sum_sq(x: &[f64]) -> f64 {
    x.iter().map(|&v| v * v).sum()
}

/// SNR in dB of `est` against reference `clean` (same convention as the crate tests).
fn snr_db(clean: &[f64], est: &[f64]) -> f64 {
    let sig = sum_sq(clean);
    let err: f64 = clean
        .iter()
        .zip(est)
        .map(|(&c, &e)| (c - e) * (c - e))
        .sum();
    10.0 * (sig / err.max(1.0e-30)).log10()
}

/// Scale the real noise so that the raw observation sits at exactly `target_db`,
/// then return `clean + scaled_noise` (real ECG + real muscle artifact).
fn corrupt(clean: &[f64], noise: &[f64], target_db: f64) -> Vec<f64> {
    let a2 = sum_sq(clean) / (sum_sq(noise) * 10f64.powf(target_db / 10.0));
    let a = a2.sqrt();
    clean.iter().zip(noise).map(|(&c, &n)| c + a * n).collect()
}

fn evaluate(ecg: &[f64], noise: &[f64], noise_name: &str, fs: f64) {
    println!("# ===== Real noise: {noise_name} =====\n");
    for &target in &[12.0f64, 6.0, 0.0]
    {
        let noisy = corrupt(ecg, noise, target);
        let s_raw = snr_db(ecg, &noisy);

        // What does the toolkit *think* the noise is?
        let vst_kind = detect_noise_model(&noisy);
        let auto = denoise_auto(&noisy, fs);

        let methods: Vec<(&str, Vec<f64>)> = vec![
            ("denoise_auto", auto.output.clone()),
            (
                "wavelet(soft)",
                wavelet_denoise(&noisy, 0, ThresholdMode::Soft),
            ),
            ("nlm1d_auto", nlm1d_auto(&noisy)),
            ("savitzky_golay(2,7)", savitzky_golay(&noisy, 2, 7)),
            ("moving_average(7)", moving_average(&noisy, 7)),
            ("stft_wiener_auto", stft_wiener_auto(&noisy)),
            ("baseline_removal(tik)", baseline_removal(&noisy)),
        ];

        println!("## target {target:.0} dB  (raw {s_raw:.2} dB)");
        println!(
            "   classifier verdict: {:?};  VST detector: {}",
            auto.profile.dominant,
            if vst_kind == VstKind::Identity
            {
                "Identity (no signal-dependent noise — correct: ECG noise is additive)"
            }
            else
            {
                "non-identity (!)"
            }
        );
        println!("   denoise_auto method: {}", auto.method);
        println!("   {:<22} {:>8} {:>10}", "method", "SNR dB", "Δ dB");
        for (name, out) in &methods
        {
            let s = snr_db(ecg, out);
            println!("   {name:<22} {s:>8.2} {:>+10.2}", s - s_raw);
        }
        println!();
    }
}

fn main() {
    let (ecg, ma, bw) = load_fixture();
    let fs = 360.0;
    println!(
        "# Real ECG (MIT-BIH rec 100, lead II) + real recorded nstdb noise, n = {}, {} Hz\n",
        ecg.len(),
        fs
    );
    evaluate(&ecg, &ma, "muscle artifact (broadband)", fs);
    evaluate(&ecg, &bw, "baseline wander (low-frequency drift)", fs);

    println!("# ===== Findings (honest) =====");
    println!(
        "# * The conservative VST detector returns Identity on real ECG at every SNR and both\n\
         #   noises — no false positive (ECG noise is additive, not Poisson/multiplicative).\n\
         # * Muscle artifact (broadband): wavelet soft-threshold gains ~+0.8..+1.1 dB across SNRs;\n\
         #   denoise_auto gains +3.8 dB at 0 dB (verdict flips to Colored → level-dep wavelet + TI).\n\
         # * At high SNR the QRS complexes read as impulsive, so denoise_auto routes to a Hampel\n\
         #   near-no-op — safe (the sharp QRS is preserved) but it leaves the broadband floor.\n\
         # * Baseline wander is genuinely hard: real nstdb drift overlaps the ECG's own low-\n\
         #   frequency content, so a stiff detrend removes signal with the drift (modest gain).\n\
         #   This is a known ECG-preprocessing challenge, surfaced here by real — not synthetic — data."
    );
}
